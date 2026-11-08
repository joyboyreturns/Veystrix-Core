# Dynamic memory control and the direct 8K AGENTIC gate (2026-11-08)

Branch `development3.8`, all uncommitted. Owner decision of this session: use memory dynamically (RAM, safe compression,
SSD expert streaming); do not stop on a brief macOS pressure level 2; add a 2.5 GiB swap-growth hard stop; keep independent
protection against sustained critical pressure, GPU/thermal/watchdog failures. One direct real 8K AGENTIC run is authorized.

## 1. What Gate A attempt 2 actually showed

Evidence: `results/flashnext/20261108-224316-gate-a-attempt2/` (`guard-trace.jsonl` at 0.25 s, `samples.jsonl` at 3 s,
`requests.jsonl`). Replay fixture of the guard's own samples: `models/flashnext/tests/bench/fixtures/gate_a_attempt2_samples.json`.

### Per-phase churn (counters are pages; x 16 KiB)

| phase | seconds | compressed | decompressed | paged in | min available | compressor |
|---|---|---|---|---|---|---|
| cold prefill 3,935 tokens | 158 | 81.6 GiB | 72.7 GiB | 126.0 GiB | 3.78 GiB | 0.86 -> 4.35 GiB |
| decode (75 tokens) | 32 | 0.0 | 1.5 | 20.4 | 3.93 | 3.07 -> 1.91 |
| tool turn 1 prefill (55 tokens) | 9.8 | 6.8 | 3.8 | 8.0 | 4.48 | 1.89 -> 4.29 |
| decode | 27 | 0.0 | 0.7 | 15.7 | 3.19 | 3.9 -> 3.36 |
| tool turn 2 prefill (55 tokens) | 9.3 | 15.0 | 12.7 | 7.8 | 3.55 | 3.59 -> 5.59 |
| decode | 11 | 0.1 | 1.0 | 6.3 | 3.06 | 4.5 -> 3.63 |
| tool turn 3 prefill (55 tokens), guard stop | 2.9 | 5.6 | 2.8 | 2.4 | 3.38 | 3.59 -> 6.34 |

All churn is in PREFILL. Decode never compresses (it only decompresses what prefill pushed out). A 55-token tool-result prefill
compresses 5.6-15 GiB inside 3-10 s, a higher rate (0.7-1.6 GiB/s) than the 4K cold prefill (0.52 GiB/s).

### Where it did and did not come from

| candidate | verdict | evidence |
|---|---|---|
| Metal / MLX active memory | no | active 3.15-3.22 GiB, flat across all three tool turns (3.19, 3.21, 3.18) |
| MLX allocator cache | no | 0.2-1.44 GiB during prefill, 0.00-0.30 GiB in decode; cannot explain 3-6 GiB of compressor |
| KV / recurrent state growth | no | +55 tokens is about 1.5 MB of attention KV; GatedDeltaNet state is fixed size |
| prefix snapshots | no | one 643-token snapshot (tens of MB) |
| Python-owned buffers | no growth trend | RSS 0.83-3.97 GiB, no ratchet across turns |
| other processes | small | compressor before 0.84 GiB, after the server exited 1.45 GiB: others account for about +0.6 GiB; the other ~3.2 GiB of compressed pages were the server's own (RSS does not count compressed pages) |
| **short-prefill read path** | **most likely** | `FLASHNEXT_PREFILL_RING_MIN_TOKENS` is 512, so a 55-token prefill never uses the F_NOCACHE ring: it reads routed experts with buffered `pread` into freshly allocated NumPy arrays (page cache fills; file-backed 4-7 GiB; free memory 0.07 GiB all run) while the server's own anonymous pages (resident weights, scratch) are squeezed into the compressor and decompressed on the next layer. The cold prefill uses the ring but still churns 0.4 GiB per layer pass |
| macOS file cache | contributes | file-backed pages 3.2 -> 7 GiB; "available" counts them, so the 2 GiB floor never saw a problem |

What is NOT proven: the split between the ring's anonymous slots, the resident weights and the per-layer NumPy arrays inside
the compressed 3.2 GiB. The offline trace cannot separate them. What was added to settle it on the next real run is listed in
section 5 (per-phase churn table in the analysis, before/after memory per request, controller ledger).

The guard stop itself: the watchdog read macOS level 2 for one sample (available 3.46 GiB, swap flat at 717.5 MB, swap-out 0).
The new controller replays all 214 recorded samples without stopping and ends ELEVATED, not CRITICAL
(`test_the_run_that_the_old_guard_stopped_is_survived_and_ends_elevated_not_critical`).

### The tool behaviour

Raw outputs are logged (`requests.jsonl` `raw_output`). The model's format was valid: step 1 two parallel `glob` calls with
patterns `services/**/config/*` and `services/**/*.yaml`; step 2 two more globs; step 3 one more. Every `glob` returned the
same two files (the real file and `gateway.example.yaml`), and the model kept widening the pattern instead of calling `grep`.
The parser, ids, argument schema and SSE stream are correct for such output (`macqwen/tests/test_parallel_tool_calls.py` pins
the attempt-2 strings: two calls, distinct ids, indexed deltas, one `tool_calls` finish). `dsh` executed every call
(`completed`). So this is model behaviour on an unguided task with an ambiguous listing, not a protocol fault.

For the 8K gate the task states the tool order ("First use the glob tool to find the gateway configuration file, then use the
grep tool on that file ...") and the project holds only the real file, so `glob` has one answer and `grep` is the obvious next
step. The model still has to emit both calls with valid arguments, and the gate passes only on real model-generated calls that the
harness executed. This changes the task, not the checks; it is recorded here so the result is not read as the unguided behaviour.

## 2. The staged controller (`models/flashnext/memory_controller.py`)

States: NORMAL, ELEVATED, RECOVERY, CRITICAL. ChatGuard feeds it one sample per second together with the existing probes.

| stage | trigger | what happens |
|---|---|---|
| ELEVATED | pressure level >= 2; any swap-out or swap-in inside the window; swap used grew > 64 MiB inside the window; compressor +1.5 GiB in the window while < 4 GiB available | recorded; chunk cap steps down one candidate at the next chunk edge and is held; bounded wait (<= 20 s per boundary, <= 120 s per request) at chunk edges; the next chunk's expert rows are read with F_NOCACHE (identical bytes); optional allocation blocked (`forbid_optional`); at request end RAM snapshots move to SSD |
| RECOVERY | causes cleared < 5 s ago | still counts as elevated for budgets; no waiting |
| NORMAL | calm for 5 s | `recovery_events` +1 |
| CRITICAL | see below | the guard stops the process (exit 75) |

CRITICAL reasons, each independent: pressure level 4 held 2 s; level >= 2 continuously for 45 s (dips shorter than 5 s do not
reset it); swap used grew 2.5 GiB over the session baseline; active swap-out or swap-in of 512 MiB inside 10 s; any missing or
unreadable probe; any exception in the controller or the monitor. These remain unchanged in ChatGuard and are not overridden by
the controller: available memory < 2 GiB, compressor growth over 2 GiB with < 3 GiB available, MLX active+cache within 0.5 GB of
the Metal limit, battery < 25%, Low Power Mode, thermal throttling, startup available < 9 GiB, and the out-of-process watchdog.

All thresholds are environment-configurable and strictly parsed (`FLASHNEXT_SWAP_GROWTH_STOP_GIB`, `FLASHNEXT_PRESSURE_PERSIST_S`,
`FLASHNEXT_CRITICAL_CONFIRM_S`, `FLASHNEXT_PRESSURE_RECOVER_S`, `FLASHNEXT_SWAP_WINDOW_S`, `FLASHNEXT_SWAP_ACTIVITY_STOP_MIB`,
`FLASHNEXT_THROTTLE_MAX_WAIT_S`, `FLASHNEXT_THROTTLE_REQUEST_CAP_S`); a bad value fails the launch.

### Swap: an increase is not active swapping

* **Swap increase** = `vm.swapusage` used, minus the session's recorded baseline. A total on disk. It can be old, cold memory of other
  applications written once; it says nothing about what the engine is doing now. Hard stop at +2.5 GiB (owner's limit); fresh growth
  inside the window only marks ELEVATED.
* **Active swap-out / swap-in** = the `vm_stat` counters moving. Pages being written to or re-read from the swap file NOW; this is what
  slows inference and wears the SSD. Stop on 512 MiB inside 10 s; below that it only marks ELEVATED.
* Zero swap-outs does not mean enough memory: attempt 2 had zero swap-outs and 5.5 GiB of compressor growth. That is why compressor
  growth, pressure and availability are separate signals.
* Owner-authorized change of earlier behaviour: the benchmark watchdog's 64 MB swap-growth stop, its "any swap-out" stop and its
  64 MiB cumulative swap-in stop are not applied to chat/server runs any more (they were the strictest possible rule). They are replaced
  by the rules above. Benchmark runs keep them (the harness is untouched). Nothing else was loosened.

The ledger (`/health` `cache.memory_controller`, the `shutdown` trace event, `analysis.json`) records swap at launch, peak swap
increase, peak swap-out and swap-in per window, peak compressor growth, recovery events, elevated seconds, throttled seconds and the
stop reason.

## 3. Between tool turns (`FlashNextBackend._release_between_turns`)

After every request (success, cancel or error), with the activity record still held so no purge can overlap: `gpu_sync.synchronize()`
(the deadlock-safe barrier; never `mx.synchronize`), `prefill_ring.fence()`, `gc.collect()`, trim the MLX allocator cache when it holds
more than 64 MiB, restore buffered reads, and move RAM snapshots to SSD only while ELEVATED. The live KV, recurrent state, decode pool
and exact prefix snapshots are untouched. Before and after (MLX active/cache/peak, RSS, compressor, wired, and the five-way split
active computation / reclaimable own / reclaimable file cache / other applications / unreleasable) are logged per request as
`turn_memory`, so any growth between tool turns is visible.

## 4. Admission for 8K

New tier `8k` (context 12288, cold prefill 10240) under the budgeted plan, computed with the same estimate formula as every other
tier (6.0 GiB). Attempt-2's measured process peak (5.43 GiB at 3.7K) is inside it. The headroom rule (physical - wired - compressor -
peak - 1.0 harness - 0.75 x other anonymous >= 4 GiB) and the 9 GiB start gate are unchanged. An explicit `--context-limit 12288`
is a requirement, so only tiers with context >= 12288 are eligible (`long`, `8k`).

## 5. What is validated, and what is not

(See section 7 for the final test numbers.) Not validated on the real model: whether F_NOCACHE reads while ELEVATED, chunk step-down and
bounded waiting actually reduce the compression churn. They are the response the owner asked for and are exactly identical in bytes
read; the effect is what the 8K run measures (`analysis.json` `phase_churn`).

Affine KV: integration is in the inference path and verified offline (cache append, restore refusal for quantized snapshots, strict
config, fingerprint, tiny-model logits). It is NOT enabled for the 8K run. Analytic KV size, 12 attention layers: bf16 27.8 KB/token
(0.21 GiB at 8K, 0.32 GiB at 12K), affine4 about 3.5x smaller for K and V, so at most 0.15 GiB at 8K. An attempt at a quick allocation
probe was discarded (it did not exercise the real update path and gave negative numbers); no peak-memory saving was measured on the
real model, and none is claimed. The per-chunk attention transient (48 B per query-key pair) is the larger term.

## 6. The owner's waiver (`models/flashnext/memory_waiver.py`)

Preflight refused the 8K launch (headroom 3.71-3.87 GiB against 4.0; start gate 8.68 GiB against 9.0; other anonymous memory
3.1-3.4 GiB, nothing closeable that is not macOS itself). The owner then said: launch anyway, stop only at 2.5 GB swap. Implemented as
the explicit opt-in `FLASHNEXT_MEMORY_WAIVER=swap-only` (`gate_a --waive-memory`), printed in the server banner, written to the
admission log and the summary. WAIVED: the admission refusal, the 9 GiB start gate, available-memory floor, compressor-with-low-availability,
pressure level 4 and persistent level 2, the swap-rate stops (all still measured and recorded). KEPT: the +2.5 GiB swap-growth stop,
the Metal-limit check, missing-probe fail-closed, battery floor, Low Power Mode, thermal, the stall watchdog, GPU sync rules.
Tests: `test_owner_waiver_*` (guard), `WaiverAdmissionTests` (without the waiver the same verdict still exits 69).

## 7. The real 8K run (attempt 1, AC power 62 -> 71%, waived) — FAILED, not a safety stop

Evidence: `results/flashnext/20261108-232323-gate-8k-attempt1/`. Outcome: the harness hit its 900 s wall cap with the cold prefill
still running; no request row was ever written (a row is written when generation ends), no tool call, no answer, follow-up exit 1
(session closed). Server ready in 9 s, shutdown clean (exit 0), no guard stop, no stall-watchdog kill.

| measured | value |
|---|---|
| first request | not recorded (request never completed); the scripted-backend probe gives 8,205 tokens |
| MLX peak / active max | 5.29 GiB / 5.06 GiB (4K attempt: 5.06 / 4.7) |
| min available | 2.60 GiB (below the old 3 GiB line: only survived because of the waiver) |
| compressor | 0.95 -> up to 5.7 GiB; max growth +4.88 GiB |
| pressure | level 2 for two episodes (peak 18 s), never level 4 |
| swap | flat at 717.5 MB: increase 0.0 MiB, swap-out 0, swap-in 1.4 MiB peak |
| controller | 3 recovery events, 87.7 s elevated, 17.8 s throttled at one chunk edge, no stop (the old guard would have stopped at the first level-2 sample) |
| churn | 286 GiB compressed, 270 GiB decompressed, 423 GiB paged in over the prefill |
| battery | 62 -> 71% (AC) |

What the trace shows: productive streaming (about 0.7 GiB/s of page-ins, the same rate as the 4K run) in three stretches, and two
long stretches with ZERO disk reads and almost no compression while MLX active memory kept spiking 3.2 -> 4.5 GiB at the same cadence:
t = 210-300 s (resumed by itself) and t = 750 s to the end (more than 290 s, never resumed). The watchdog did not call it a stall (the
process was active). The phase never left "prefill". So the prefill is far slower than the 4K rate predicts (about 330 s at 25 tok/s) and
the last stretch looks like GPU/CPU work without expert reads, or a wait.

Root cause: NOT identified. The offline trace cannot say which. It is not memory (pressure 1, 5-6 GiB available, swap flat during the
stretch), and it is not the controller (3 short throttles, 17.8 s in total). Candidates, none verified: (a) a long-context sparse-attention
or snapshot step that runs at positions beyond 4K, (b) a wait on a Metal command buffer, (c) a throttled GPU. What was added so the next
run shows it: `FLASHNEXT_CHUNK_LOG` (one JSON line per finished chunk, live: start/end, seconds, state, headroom, MLX active/peak) and
`StackProbe` (a native `sample` of the server after 360 s and every 240 s into `stack-N.txt`). No further full-checkpoint run was made.

## 8. Status against the 12 criteria

1 full 8K input processed: NOT shown. 2 decode began: no. 3-9 glob / grep / final answer: not reached. 10 follow-up: no. 11 memory within
safe limits: yes by the waiver's own limits (swap +0, pressure 2 for at most 18 s); NOT within the pre-waiver limits (min available 2.60 GiB).
12 clean shutdown: yes. Recommendation for 12K: none until the 8K prefill completes; see the next-run requirements in section 7.
