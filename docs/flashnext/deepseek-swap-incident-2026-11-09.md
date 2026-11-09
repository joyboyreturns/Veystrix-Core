# DeepSeek Harness freeze, swap incident, 32K prefill, cross-session cache and reasoning — 2026-11-09

Branch `development3.8`, working tree uncommitted (nothing was committed, pushed, reset or cleaned). Machine: MacBook Air M4, 16 GB.
**No real-checkpoint inference was run in this session.** Every number below marked "measured" comes from the user's own run
evidence (system log, session file, cache directory) or from offline tests; nothing here is a new real-model result.

---

## 1. Root-cause analysis

### What the evidence shows (reconstructed after the fact; the server kept no log of its own)

| Time (IST) | Evidence | Source |
|---|---|---|
| 12:05:04 | Previous server (pid 26474, **supervised**: it wrote `veystrix-chat-26474.jsonl`) exits while idle. Swap 1088 MB vs launch baseline 977 MB (+112 MB); no guard stop. | guard trace, unified log |
| 12:06:01 | **A new Python process, pid 46644, starts. It wrote no `veystrix-chat-46644.jsonl`**, so its ChatGuard (the +2.5 GiB swap supervisor) never ran. It also is in no interactive shell history, so it was started by a non-interactive shell. | temp dirs searched everywhere, `~/.zsh_sessions`, unified log |
| 12:40:57 | DeepSeek Harness (DSH) sends the user's "hello" **and, 21 ms later, a second unrelated request (the session-title LLM call)** to the same single-slot server. The main request carries 5.6K chars of system prompt + 32 tool schemas (31K chars of JSON): **8,853 prompt tokens** with the real tokenizer. | DSH session file, real-tokenizer render |
| 12:41:12 | Title answered ("Greeting"). Metal pipelines compile; a 156-token / 120 MB snapshot with **no chunk-plan history** is written to `~/.flashnext/prefix-cache` (a post-turn snapshot that can never be restored). | `a9ace411….json`, unified log |
| 12:41:18 | macOS compressor peaks at **8.05 GiB**; available memory drops to 3.1 GiB at 12:41:40 and **2.72 GiB at 12:42:07**; the kernel starts idle-exiting daemons. | `memorystatus` lines in the unified log |
| 12:41:39 – 12:41:46 | Last Metal compile events from pid 46644 (prefill kernels). **No assistant token was ever produced** (`assistant/attempt` stream = `[]`). | unified log, DSH session |
| 12:42:04 | The user aborts the turn (67 s after sending). | DSH session |
| 12:46:25 | pid 46644 exits (cause not recorded: user or agent action). | unified log (`runningboardd`) |

### Causes, ranked by evidence

1. **The server ran unsupervised (strong, circumstantial).** Everything that makes the AGENTIC server safe is switched on by the
   launcher (`./chat.sh --mode agentic` → `macqwen/cli.py` → `settings.modes.apply_mode` + `settings.launch.apply_chat_environment`),
   not by `macqwen/session.py`: ChatGuard and its **+2.5 GiB swap stop**, 2,048-token chunked prefill, the prefix cache, the
   out-of-process watchdog, the decode pool setting. The "candidate launch command" handed over in the previous session's report
   was a direct `session.py --server` call with a few `VEYSTRIX_*` variables. Without the mode environment
   `FLASHNEXT_PREFILL_CHUNK_TOKENS` is unset, and the code's own banner calls that "one call per prompt (reference numerics)".
   The missing guard trace for pid 46644 is the observable proof that the guard did not run. **So the swap guard did not "fail";
   it was never started.** Not provable from logs: the exact environment of pid 46644 (it cannot be read after exit).
2. **An 8,853-token cold prefill in one unbounded call, competing with a second request (strong).** A DSH "hello" is ~8.9K tokens
   (measured with the real tokenizer on the captured request). At the previously measured 47–54 tok/s that is ~2.7–3.1 minutes of
   *chunked* prefill even when everything works, and the title request replaced the live state first. DSH shows nothing during that
   time (SSE comment heartbeats are invisible), which looks exactly like "stopped responding".
3. **Memory pressure (measured, but swap itself was not).** Compressor 8.05 GiB, available 2.72 GiB at worst. The user reported
   ~5 GB swap; **no swap sample exists for 12:40–12:46** (the only sampler, ChatGuard, was off), so the 5 GB figure and the
   `swap_growth = max(0, high_watermark − launch_baseline)` calculation **cannot be reproduced**. The 2.5 GiB threshold could not
   have been compared against anything because no baseline was captured by pid 46644.
4. **Wasted snapshot write (measured).** A post-turn snapshot was written per turn *even with the prefix cache disabled*
   (`macqwen/server.py`, `kind="turn_complete"`), 120–400 MB serialised each. Its state has no chunk-plan provenance, which the
   exact-prefix lookup treats as "unknown history" and never accepts. Pure cost, no benefit; contributed IO and memory pressure.

### Not proven / ruled out

* Python/GIL deadlock, GPU stall, blocked SSE delivery, lock contention in the cache: **no evidence either way** (no thread dump,
  no flight record existed). The earlier session's protections for those (out-of-process watchdog) were also off.
* Client timeout: DSH had no timeout involvement; the user aborted at 67 s.
* Whether the unsupervised prefill alone reaches 5 GB swap on this machine: **unproven** (would need a real run; not done).

### Why it looked like a freeze
No first token for minutes + no progress output + a client that shows nothing for comment-only heartbeats + a machine under
compressor pressure. None of these is a protocol deadlock; all of them are now surfaced (see §4 "Progress diagnostics").

---

## 2. Implemented changes (exact files)

| Area | File / function | Change |
|---|---|---|
| Launch supervision | `models/flashnext/supervision.py` (new) `check()`; `macqwen/session.py` `main()` | A flashnext `--server` launch **refuses to start** (exit 70, nothing loaded) unless `VEYSTRIX_MODE=agentic`, `FLASHNEXT_CHAT_GUARD=1` and non-zero `FLASHNEXT_PREFILL_CHUNK_TOKENS`. Prints the correct launcher command. |
| Swap stop | `models/flashnext/chat_guard.py` `ChatGuard._stop/_graceful_ok/_grace_then_exit/_hard_exit`, `cancel_grace_seconds()` | A **swap-growth** stop with a job in flight no longer kills the process instantly: admission is already closed, the job cancels at its next streamed-layer boundary and its client gets an actionable 503; the process exits as soon as the worker is idle (+1.5 s flush) or after `FLASHNEXT_GUARD_CANCEL_GRACE_S` (default 20 s, 0 = immediate), whichever is first. Every other stop (battery, thermal, Metal limit, missing probe, monitor failure) is **unchanged and immediate**. |
| Controller | `models/flashnext/memory_controller.py` | `stopped_at`, `swap_increase_mb()`, `baseline_swap_used_mb`, `_peak_swap_growth_mb` (these were read by `/v1/cache/diagnostics` but did not exist, so diagnostics silently reported 0 growth), `register_worker_probe()/worker_busy()`. |
| Server errors | `macqwen/server.py` `_safety_stop_error`, `ModelService.complete` | A safety stop now ends a request with `503 memory_critical_stop` ("…Do NOT retry automatically: restart deliberately with …") on stream and non-stream paths, instead of a silent cancel or an empty error. New requests are refused with the same code. |
| Health | `macqwen/server.py` `health()`, `_worker_state()`; `models/flashnext/progress.py` `Progress.snapshot()` | `/health` and `/ready` report `status: stopped` after a safety stop and `status: stalled` (ready false) when a job is active and nothing (streamed layer or decoded token) advanced for `VEYSTRIX_STALL_SECONDS` (default 180 s). Adds `worker.{stage,progress_ticks,seconds_since_progress,stalled}`. |
| Flight recorder | `models/flashnext/flight_recorder.py` (new); hooks in `server.py` | Private (0600/0700) JSONL `~/.flashnext/logs/server-<pid>.jsonl`, flushed per line, rotating, never stores prompt text: `server_start, request_begin (client, tools, reasoning fields), adopt (prompt/matched/needed tokens, outcome), generate_begin, prefill_progress (≥10 s apart), first_token, generate_end, request_cancelled/error/end, guard_stop, guard_exit`. `FLASHNEXT_FLIGHT_LOG=0` disables. |
| 32K ceilings | `macqwen/server.py` (`MAX_*_DEFAULT`, `_check_prefill`, `_adopt`, `_context_report`); `models/flashnext/settings/modes.py` (`DEFAULT_MAX_COLD_PREFILL`, `DEFAULT_MAX_INCREMENTAL_PREFILL`, `plan_mode`, `apply_mode`, `plan_from_environment`, banner); `macqwen/cli.py` (`--max-cold-prefill`, `--max-incremental-prefill`); `macqwen/session.py` `_apply_tier` | Cold and uncached-increment ceilings are **32,768** and are now enforced **separately** (previously the incremental value was only reported). A startup memory tier still bounds both. AGENTIC default context is now the native **262,144** and is a stated requirement. Defaults never exceed a smaller `--context-limit`; explicit values above it are refused. |
| Post-turn snapshot | `macqwen/server.py` | Now **opt-in** (`FLASHNEXT_TURN_SNAPSHOTS=1` with the prefix cache on). Default writes nothing. |
| Client config | `~/.dsh/profiles/{headless,desktop}/cordis.patch.yml` (backups `cordis.patch.<ts>.pre-effort.bak`); `docs/flashnext/dsh/veystrix-262k.patch.yml` | `reasoningEfforts` for all six levels, `maxRetries: 0`, `session-title-llm` disabled. Remote DeepSeek routes untouched. |
| Registry/tests fixes | `settings/setting_research.py` (`FLASHNEXT_DISK_RESERVE_GB` registered), `test_request_recovery.py` (stale 8K-tier expectation updated to the approved policy) | Fixed two failures that already existed in the working tree. |
| Tests | see §7 | `test_swap_supervisor.py`, `test_cross_session_cache.py`, `test_swap_stop_server.py`, `test_effort_transitions.py`; existing tests updated for the deliberate default changes (262,144 / 32,768). |

---

## 3. Cache architecture (what exists, what this session verified)

* **Tier A – live state** (one conversation: KV + GDN recurrent state + exact token tape). A request that extends the tape prefills
  only the new tokens (`reused` counter).
* **Tier B – RAM snapshots**, 0.75 GB LRU. **Tier C – SSD snapshots**, 24 GiB quota, 50 GiB free-disk reserve, files 0600 in a 0700
  directory, SHA-256 per file, manifest bound to file, atomic write + fsync, crash-safe index rebuilt from manifests at start,
  chained delta KV (a child stores only its new KV; the 113 MB GDN state is stored per snapshot because it is exact and
  non-additive), leases protect parents, eviction drops leaves first and removes orphans.
* **Identity** = engine/checkpoint/tokenizer/format/chunk-policy **fingerprint** + **exact token-ID prefix** (+ chunk-plan
  provenance). A snapshot at *n* tokens is a pure function of those *n* tokens, and a lookup accepts it only when the prompt's
  first *n* tokens are identical, so **sharing is only ever of tokens both sessions really have, and a private suffix can never
  reach another session**. `x-cache-id` / session headers only name the conversation for the reasoning ledger and session
  accounting; they are a hint, not a key and not an authorization.
* **Shared static prefix**: with `FLASHNEXT_PREFIX_CACHE_BOUNDARIES=messages`, a snapshot is taken at every `<|im_end|>\n`
  (the system message including the tool schemas is the first). Any later session whose prompt starts with the same tokens
  restores it from RAM or SSD and prefills only its own tokens.
* **Private continuation**: the live state (same session) or the last message-boundary snapshot of that conversation's own
  prompts. **A state produced by decode steps is never restored** (not bit-identical to a cold prefill, no chunk-plan
  provenance), so after the live state was replaced the engine re-prefills the assistant tokens since the last boundary
  (tens to a few thousand tokens), not the whole conversation.
* **Reasoning state**: `ReasoningLedger` (per conversation key, transcript-matched) restores exact reasoning only when the
  client's assistant messages match recorded turns in order, content and tool calls; never invents or duplicates reasoning.
* **Eviction**: RAM → SSD spill; SSD LRU with chains kept whole; quota and free-disk guard skip writes rather than fail.

**Measured offline (tiny real-class model, `test_cross_session_cache.py`)**

| Scenario | total | matched | prefilled | outcome |
|---|---|---|---|---|
| A creates static prefix | 219 | 0 | 219 | COLD |
| B: new session / new client process, same system+tools | 229 | **152** | **77** | SSD_HIT (1.5 ms) |
| B in the same process | 229 | 152 | 77 | RAM_HIT (0.07 ms) |
| B after RAM eviction | 229 | 152 | 77 | SSD_HIT, bytes read > 0 |
| C: changed tool schema | 229 | 0 | 229 | COLD (no false sharing) |
| D: changed reasoning effort (sentence at start of system) | 219 | 0 | 219 | COLD |
| S1 / S2: identical user text, different private histories | 298 / 302 | 0 / **199** | 298 / 103 | S2 shares exactly the common 199 tokens, none of S1's private tokens |
| A resumes after B replaced the live state | 313 | **214** | **99** | SSD_HIT; final state bit-identical to a cold run |
| Server restart, A resumes | 313 | 214 | 99 | SSD_HIT |
| Corrupted snapshot (flipped byte) | 229 | 0 | 229 | STATE_INVALID → cold, result bit-identical |
| Different engine fingerprint | 229 | 0 | 229 | COLD |
| 8 sessions at a 3 MB quota | – | – | – | disk bytes ≤ budget, no orphaned child |

Every restored run's final token **and every cache array** equals a cold run's, bit for bit; a "hit" always meant skipped computation
(`prefilled = total − matched`). Restore latency here is for a ~150-token state; **it has not been measured at 9–10K tokens**
(that state is ~0.35 GB; expect tens to a few hundred ms from SSD — unmeasured).

**Not implemented:** a client/authorization **namespace partition** of the SSD store. For token-exact sharing it is not needed for
correctness (see Identity); the cost of sharing is only that another local client could observe a timing difference. If you want
hard partitioning, it needs one `PrefixCache` per namespace (not built).

---

## 4. Streaming, cancellation and watchdog

* Heartbeats (`: generating` every `VEYSTRIX_SSE_KEEPALIVE_SECONDS`, default 15 s) come from a separate thread. **Tested with a worker
  blocked in a call that ignores cancellation: heartbeats kept flowing, `/health` answered in < 1 s and reported `stalled`.**
* Cancellation points: between **streamed MoE layers** during prefill (the `layer_completed` callback) and between tokens during
  decode. A client disconnect and a safety stop both use them. **An in-flight Metal command buffer cannot be interrupted from
  Python**; the delay is bounded by one layer's work (seconds) and then by the grace deadline, after which the process exits.
* Idle stream timeout vs hard timeout: Qwen Code uses `streamIdleTimeoutMs: 900000` and `timeout: 3600000` (idle < hard); DSH has
  its own (not changed). Nothing was solved by lengthening timeouts.
* Progress diagnostics: `/health` → `worker`, `/v1/cache/diagnostics`, flight recorder `prefill_progress`.
* Not done: a *maximum stream lifetime* enforced server-side, and a server-side stalled-storage-read test (the SSD restore is
  synchronous inside the worker; a stall there shows as `worker.stalled`).

---

## 5. Context limits (confirmed)

| Limit | Value | Where |
|---|---|---|
| Native / configured context | **262,144** | `modes.AGENTIC_CONTEXT_LIMIT`, `VEYSTRIX_CONTEXT_LIMIT`, `/v1/models` |
| Max cold prefill | **32,768** | `VEYSTRIX_MAX_COLD_PREFILL`, `--max-cold-prefill`, `MAX_COLD_PREFILL_DEFAULT` |
| Max uncached increment per request | **32,768** | `VEYSTRIX_MAX_INCREMENTAL_PREFILL`, `--max-incremental-prefill` |
| Internal chunk | 2,048 (balanced) | `FLASHNEXT_PREFILL_CHUNK_TOKENS` — unchanged |

Verified by tests: 32,768 cold admitted, 32,769 refused with nothing run; a 250,000-token conversation with a 100-token suffix prefills 100;
a 3,000-token increment on 100K of state passes under a 4,096 ceiling and 5,000 is refused; a giant cold prompt is **refused, not split**.
**32K is not validated on the real checkpoint** (evidence exists to 16K: 15,819 tokens, 328.98 s, 176 MB swap growth; 8K: 177 s).
The launcher prints that warning. Do not treat 32K as safe until the swap supervisor has run a 16K and then a 24K/32K real test with your approval.

---

## 6. Compatibility matrices

### Reasoning efforts (wire → applied)

| Effort | DSH 0.2.2-rc.1 wire (real CLI, stub server) | Qwen Code 0.27.0 wire | Server applies |
|---|---|---|---|
| off | `reasoning_effort: "none"` | `reasoning.effort: "none"` | thinking **off** |
| low | `low` | `low` | template `low` |
| medium | `medium` | `medium` | template `medium` |
| high | `high` | `high` | `medium` + repository stopping-rule sentence |
| xhigh | `xhigh` | `xhigh` | template `xhigh` |
| max | `max` | `max` | **clamped to xhigh, reported in `notes`** (identical tokens) |

Limitation: the checkpoint's template has no `max` level and no decoder-level thinking budget; `max` is documented as `xhigh`, not a fifth level.

### Cache cost of switching effort (real tokenizer, DSH-sized 8,853-token prompt)

| Switch | Shared prefix tokens | Consequence |
|---|---|---|
| medium ↔ off | 8,852 / 8,853 | static prefix reused; only the generation prompt differs. In one conversation off→medium→off: **0 rebuilds** |
| medium → high | 7,719 | long prefix shared |
| medium → low, medium → xhigh, low ↔ xhigh | **3** | whole static prefix differs: the template puts the effort sentence **first**. Each effort keeps its own snapshot after first use. A same-conversation switch **rebuilds** (cost ≈ the prompt, ~3 min at 8.9K tokens). Not normalised away. |

**Practical advice:** pick `medium` (or `off`) as the working effort in DSH; switching to `low`/`xhigh` mid-session costs a full prefill once per effort.

### Qwen Code
Settings already in `~/.qwen/settings.json`: `contextWindowSize 262144`, `maxRetries 0`, `streamIdleTimeoutMs 900000`, no `samplingParams`
(which would silence the effort field), `reasoningEffort` per `/effort`. Unchanged this session. Offline-tested: wire shape, 262,144 report,
32,768 limits, reasoning round-trips (existing 14 Qwen client tests + new effort tests). **Not re-run against a live Qwen Code process.**

### DeepSeek Harness
| Item | Result |
|---|---|
| Provider/model selection | headless profile still defaults to remote `deepseek-official/deepseek-flash` (untouched); desktop selects `veystrix/macqwen-flashnext` |
| Effort levels offered and sent | all six, **real client verified on the wire** (stub server) |
| Context window / token cap | 262,144 / `max_completion_tokens: 16128` observed on the wire |
| Other request fields | `store:false`, `stream_options.include_usage`, `tools` (23 in headless, 32 in desktop); all accepted by the server's field check |
| **Session identity** | **none on the wire** (`session_headers: {}`); DSH withholds `sendSessionAffinityHeaders` for openai-completions routes. Conversations are identified by transcript digest. No `x-cache-id`. |
| Reasoning passback | client behaviour not changed; server ledger restores omitted reasoning (tested: tool turn, multi-tool turn, omitted, returned, no duplication) |
| Retry | `maxRetries: 0` set |
| Title request | disabled (it interleaved a second conversation and evicted state) |
| Large reconstructed history | DSH resends the full history each turn (expected); the first request is a ~8.9K-token cold prefill (desktop), later ones extend the live state or restore from the last message boundary |
| Real DSH against the real server | **not done** (needs a real model run) |

---

## 7. Verification (commands and results)

All offline; swap stayed 1.0 GiB / 2.0 GiB total throughout; no checkpoint loaded.

| Command | Result |
|---|---|
| `.venv/bin/python -m unittest discover -s macqwen/tests -t .` (baseline, before changes) | 598 OK |
| `.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t .` (baseline) | 1561 run, 2 failures that already existed (fixed, §2) |
| **after all changes**: MACQWEN suite | **633 run, OK** (was 598; +35 new/updated) |
| **after all changes**: FlashNext unit suite | **1593 run, OK, 3 skipped** (was 1561 with 2 pre-existing failures) |
| `unittest models.flashnext.tests.unit.test_swap_supervisor` | 18 OK |
| `unittest macqwen.tests.test_swap_stop_server` | 15 OK |
| `unittest macqwen.tests.test_effort_transitions` | 18 OK |
| `unittest models.flashnext.tests.unit.test_cross_session_cache` | 14 OK |
| `dsh --profile headless --dump-config` | composes; shows `reasoningEfforts`, `maxRetries: 0`, `session-title-llm disabled` |
| real `dsh headless` × 6 efforts vs stub server (port 18081, isolated `DSH_HOME`) | `reasoning_effort` = none/low/medium/high/xhigh/max, all accepted and mapped |
| direct `session.py --server` | refused, exit 70, "Nothing was loaded" |
| launcher command parse (`cli.command`) | environment shows guard 1, watchdog 1, chunk 2048, 262144 / 32768 / 32768, SSD 24 GB, RAM 0.75 GB |

**Skipped on purpose (need your approval):** any real-checkpoint run (16K/32K prefill, cold vs RAM vs SSD hit latency, TTFT, physical reads,
decode speed with a real DSH or Qwen Code session), the 262K ladder on the real model, a live swap-threshold crossing.
The performance comparison you asked for (cold / RAM hit / SSD hit / continuation / shared prefix / resumed session: TTFT, physical
reads, swap delta, decode speed) is therefore **not produced**; only token-accounting equivalents from the offline model exist.

---

## 8. Readiness and how to run it

**Is it safe to start the server and try a short interactive coding task?** Reasonably, under these conditions: start it *only* with
the launcher below (a direct launch is now refused); expect the **first** DSH request of a session to take ~3 minutes with no output
(8.9K-token cold prefill); keep the title call disabled; stay on `medium` or `off`; watch swap. What is protected: swap growth +2.5 GiB
stops the work cleanly (≤ 20 s) and exits; pressure/available are telemetry only (your decision), so **a sudden OOM, a GPU allocation
failure or heavy compressor growth are not stopped** — the 12:41 incident showed the compressor at 8 GiB with ~2.7 GiB available.
Unknown until a real run: how much swap an 8.9K-token chunked prefill produces on this machine today with Chrome etc. open (the
earlier supervised runs: 0 MB at 8K, 176 MB at 16K). I did **not** start the server myself: loading the checkpoint is a real-model action
that needs your go-ahead, and no other server is running (port 8080 free).

```bash
# 1. start (the only supported way; direct session.py --server is refused)
cd /Users/vincent/Documents/Veystrix
export VEYSTRIX_API_KEY=local
./chat.sh --mode agentic --pool-gb 0 --context-limit 262144 --max-cold-prefill 32768 --max-incremental-prefill 32768

# 2. swap and safety (separate terminal)
sysctl vm.swapusage
curl -s -m 3 http://127.0.0.1:8080/health | python3 -m json.tool      # status ok|stopped|stalled, memory_policy, worker
curl -s -m 3 http://127.0.0.1:8080/v1/models | python3 -m json.tool   # context 262144, max_cold_prefill 32768
curl -s -m 3 http://127.0.0.1:8080/v1/cache/diagnostics | python3 -m json.tool   # ram/ssd bytes, hits, swap growth, last_prefix
ls -lt ~/.flashnext/logs/ | head -3; tail -f ~/.flashnext/logs/server-*.jsonl      # flight recorder (stage of every request)
ls -t "$TMPDIR"/veystrix-chat-*.jsonl | head -1                                      # guard trace (swap samples)

# 3. SSD snapshots (stop the server first: the directory has a single-owner lock; while running use /v1/cache/diagnostics)
.venv/bin/python -m macqwen.cache_tools --inspect

# 4. Qwen Code (settings already point at the server)
VEYSTRIX_API_KEY=local qwen

# 5. DeepSeek Harness desktop: quit and reopen the app so it re-reads the profile, then pick "Veystrix (Flash-Next)"
#    and an effort in the model picker (off/low/medium/high/xhigh/max).
#    headless: dsh --profile headless --patch docs/flashnext/dsh/veystrix-262k.patch.yml  (+ the agent-default-model overlay)
```

**Test cross-session prefix reuse:** (a) in DSH start session A, say "hello" (cold, ~3 min); (b) open a **new** DSH session with the
same effort and say anything; (c) `curl …/v1/cache/diagnostics` → `last_prefix.hit_tokens` should be ≈ the system+tools size
(~8.8K) and `newly_computed_prefill_tokens` only the new tail; `ssd_hits` or `ram_hits` increases; TTFT should drop from minutes to
seconds. This is the **unverified-on-real-checkpoint** expectation; report the numbers.
**Test reasoning levels:** switch effort in the picker between turns; remember low/xhigh cost one cold prefill each the first time.

**`recovery_required` (HTTP 400):** the conversation is mostly uncached and rebuilding it would exceed 32,768 tokens. Start a new
conversation (the static prefix will hit) or shorten the history; do not raise the ceiling blindly.
**`oversized_prefill_rejected`:** one request added more than 32,768 uncached tokens; send less new content per turn.
**`memory_critical_stop` (HTTP 503) / process exit code 75:** swap grew +2.5 GiB over the launch baseline. The in-flight request was
cancelled and the server exited. Do not retry automatically. Check `sysctl vm.swapusage`, read the last lines of
`~/.flashnext/logs/server-<pid>.jsonl` and `veystrix-chat-<pid>.jsonl`, wait for swap to ease (or close heavy apps), then restart with the
command in step 1. Never delete swap files or disable the pager.

---

## 9. Limitations (implemented vs tested vs verified)

| Item | Implemented | Unit-tested | Real-client verified | Real-checkpoint verified |
|---|---|---|---|---|
| Launch refusal without supervisor | yes | yes | direct launch refused (real CLI) | n/a |
| +2.5 GiB swap stop, graceful cancel, hard exit on blocked worker | yes | yes (synthetic readings, blocked worker) | no | **no** |
| Stalled/stopped `/health`, heartbeats under a blocked worker | yes | yes (real HTTP server) | no | no |
| Flight recorder | yes | yes | no | no |
| 262,144 / 32,768 limits | yes | yes | no | 16K prefill only (earlier session) |
| Cross-session static-prefix reuse, resume, restart, eviction, corruption, quota | existing engine, verified | yes (tiny model, bit-exact) | no | earlier session: 4K–16K system-prefix hits only |
| Client/authorization namespace on SSD | **no** | – | – | – |
| Effort ladder, reasoning ledger, tool loops | yes (existing + tests) | yes (real tokenizer) | DSH wire: yes; Qwen wire: previous session | no |
| DSH profile (efforts, retries, title off) | yes | `--dump-config`, 6 real CLI runs | **yes (stub server)** | no |
| Server-side max stream lifetime | no | – | – | – |
| Cancellation of a Metal call already submitted | impossible from Python; bounded by a layer + grace | yes (blocked worker) | – | – |

Known residue: `~/.flashnext/prefix-cache/a9ace411….{json,safetensors}` (156 tokens, no plan) is the dead post-turn snapshot from the
incident; it is harmless, was not deleted (you asked to preserve cache files), and will age out under the LRU/quota.

---

## 10. Real-checkpoint 16K run (approved by the owner, 2026-11-09 14:07–14:20)

Launched with the supported command (§8). Conditions: AC power, battery 80%, Low Power Mode off, other applications open, swap
baseline 1,001 MB captured by the guard at launch (`/health` and `/v1/cache/diagnostics` reported it). Supervisor ON (+2.5 GiB = 2,560 MB
stop), watchdog ON, flight recorder ON. Client: `scratchpad/exp/run16k.py` (streamed, real repo source text, thinking off, 64 output tokens,
independent swap sampler every 5 s that would have aborted at +2 GiB). The 15,685-token prompt is real text, not a synthetic repeat.

| Measure | Result |
|---|---|
| Turn 1, cold, 15,685 prompt tokens | MISS, 15,685 tokens prefilled; **prefill 438.4 s = 35.8 tok/s**; TTFT 439.0 s; wall 459.6 s |
| Earlier supervised 16K run (15,819 tokens) | 328.98 s = 48.1 tok/s — this run was **~33% slower** |
| Decode after the 16K context (pool off) | 3.04 tok/s (64 tokens); turn 2: 2.28 tok/s (10 tokens). No earlier decode figure for `--pool-gb 0` to compare. |
| SSE heartbeats during the 7.3-minute prefill | 30 (≈ every 15 s) |
| **Swap** | baseline 1,001 MB → **peak 2,191 MB (+1,190 MB ≈ 1.16 GiB; controller high-watermark 1,182 MB) = 46% of the 2.5 GiB stop**; settled +471…511 MB afterwards. No stop. Earlier supervised 16K run: +176 MB. |
| When the swap grew | almost all in the first ~55 s (+1,134 MB sampled), then it *fell* during the rest of the prefill (910 MB at 422 s). It did not scale with prompt length. |
| Other memory | pressure level 2 episodes (guard states elevated/recovery, never critical); compressor peak 6.5 GB (+4.8 GiB in 9 s early on); min available 2.5 GB; RSS peak 4.7 GB; MLX peak 5.98 GB (earlier run reported 7.61 GiB) |
| Worker liveness | `seconds_since_progress` max 1.6 s; `/health` answered at every one of 94 five-second samples with `status: ok`, `stalled: false`; `ticks` advanced steadily (one per streamed layer) |
| **Turn 2 (same session, +24 tokens)** | live state extended: matched 15,749, prefilled 24, TTFT 5.57 s |
| **Replay of the identical prompt after the live state was replaced** | **SSD_HIT: 15,678 of 15,685 tokens restored, 7 prefilled, 550 MB read, restore 0.545 s, TTFT 2.84 s** (vs 439 s cold). RAM hits 0, SSD hits 1. |
| SSD cache after the run | 1 entry, 550 MB written once (the message-boundary snapshot taken during turn 1), well inside the 24 GiB quota |

What this does and does not establish:

* It **does** show on the real checkpoint: the launcher-applied supervisor, health and worker diagnostics, heartbeats and the SSD
  snapshot/restore path work at ~16K tokens, and a restored 16K state costs ~3 s instead of ~7 minutes.
* The cold-prefill rate was lower and the early swap growth larger than the earlier supervised run. **The cause was not isolated**;
  the machine was visibly under more pressure (other applications open, compressor surge, pressure level 2). Do not read 36 tok/s as the
  engine's rate; read it as "this machine, in this state, today".
* **Not verified on the real model:** that the restored state is bit-identical to a cold run (sampling temperature was 1.0, so the
  replay text differs and proves nothing either way; exactness is proven only on the small model), 24K/32K prefill, the guard stopping a
  real run, and a real DeepSeek Harness or Qwen Code session.
* Extrapolation, not measurement: at this machine's rate a cold 32K prompt would take roughly 15 minutes. Keep first requests under ~16K
  and rely on the cache for everything after.
