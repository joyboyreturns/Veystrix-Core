# Qwen Code integration, 262K context and dynamic reasoning (2026-11-09)

Status: offline work only. No real-checkpoint run was made for this task, and none is proposed without approval (section 7).
Qwen Code compatibility tests below prove the wire protocol. They are **not** evidence that a long real prefill completes.

## 1. Provider config (`~/.qwen/settings.json`)

Installed client: Qwen Code 0.22.0 (Desktop 0.2.2). Its real schema is `modelProviders.openai[]` with
`id, name, baseUrl, envKey, generationConfig`. `wireApi`, `openai-effort` and `capabilities.reasoning` do not exist in this
version, so they are not written. The file was created (it did not exist); any existing providers would have been kept.

```json
{
  "modelProviders": {"openai": [{
    "id": "macqwen-flashnext", "name": "Veystrix Qwen3.8 Flash-Next Q4 (local)",
    "baseUrl": "http://127.0.0.1:8080/v1", "envKey": "VEYSTRIX_API_KEY",
    "generationConfig": {"timeout": 3600000, "streamIdleTimeoutMs": 900000, "maxRetries": 0, "contextWindowSize": 45056}}]},
  "tools": {"exclude": ["agent", "skill", "tool_search", "list_agents", "get_goal", "update_goal", "todo_write", "cron_create",
                        "cron_list", "cron_delete", "loop_wakeup", "enter_worktree", "exit_worktree", "record_artifact",
                        "report_findings", "web_fetch", "zoom_image", "read_mcp_resource", "task_stop", "send_message",
                        "monitor", "notebook_edit"]},
  "security": {"auth": {"selectedType": "openai"}},
  "model": {"name": "macqwen-flashnext"},
  "memory": {"enableManagedAutoMemory": false, "enableManagedAutoDream": false, "enableAutoSkill": false},
  "ui": {"enableFollowupSuggestions": false}
}
```

* No `samplingParams` and no `extra_body`: either would suppress or override `/effort` (measured, section 3).
* `maxRetries: 0`: a retry after a timeout would restart a prefill the server may still be running.
* `timeout` 1 h bounds the whole request, `streamIdleTimeoutMs` 15 min bounds silence between SSE events. Before this change
  the server sent nothing between the first role chunk and the first decoded token, so a long prefill was indistinguishable
  from a dead stream. `_complete_with_keepalive` now writes an SSE comment (`: generating`) every 15 s
  (`VEYSTRIX_SSE_KEEPALIVE_SECONDS`) while a request runs; it ends with the request (tests in `PrefillKeepAliveTests`).
  A stalled prefill is still caught by the out-of-process watchdog, not by the client timeout.
* The four `memory`/`ui` switches stop Qwen Code's background requests (auto-memory, dream, skill extraction, follow-up
  suggestions). Each would otherwise be a second model request, which evicts the live state of the main conversation.
* `tools.exclude` is not cosmetic. Measured with the real tokenizer against the real server (first request for the prompt
  "hello"): untrimmed Qwen Code sends 23 tools (28 in yolo mode) and **19.6K-21.9K tokens**, more than every admission tier
  (largest 16,384). Excluding the 22 non-essential tools leaves 6 tools (edit, write_file, glob, grep_search, read_file,
  run_shell_command) at **10.1K tokens**, or 3 read-only tools at **8.6K** in plan mode. About 5K of that is Qwen Code's own
  system prompt, which settings cannot shrink. Even trimmed, a request nearly fills the 8k tier (12,288), so Qwen Code is
  realistic only on the `long` tier (16,384), which has not been validated on the real checkpoint (largest real cold prefill:
  3,935 tokens).
* `contextWindowSize: 45056` is deliberately NOT the server's window (16,384). Measured with the installed client (0.27.0, the one on
  this machine) against the real server code: Qwen Code estimates its own prompt at about the real token count plus a fixed ~21.8K
  overhead, and sets `max_completion_tokens = window - estimate`; with 16384 or 20480 that left 1 token and the server refused the
  request with HTTP 400 (`max_completion_tokens=1 cannot hold a tool call`). Its auto-compaction fires at
  `min(0.85 x window, window - 20000 - 13000)` of server-reported prompt tokens, so 45056 puts compaction at about 12,056 real tokens,
  before the server's hard 16,384 limit, and keeps `max_completion_tokens` at 9.7K-14.9K for real prompts of 8.3K-13.9K (the
  server clamps it to what is left of its context). The server's `suggested_client_context_window` (20480) is the DeepSeek
  Harness value and is wrong for Qwen Code. The server still enforces 16,384 and refuses anything larger with a visible error.
  admitted to a smaller tier, `/v1/models` says so and a prompt over the limit gets a visible `context_length_exceeded` error
  (Qwen Code prints it as `[API Error: ...]` with exit code 0). Lower the number to match.
* `VEYSTRIX_API_KEY` must be set in the shell that starts `qwen`. A local server started without `--server-api-key`
  accepts any value, so `export VEYSTRIX_API_KEY=local` is enough.

## 2. Context: four numbers, never one

`/v1/models` reports them separately in `context`:

| Field | Meaning | Value here |
|---|---|---|
| `native_positional_capability` | checkpoint `max_position_embeddings` (rope_theta 1e7, mrope interleaved, partial rotary 0.25) | 262144 |
| `configured_server_limit` | `--context-limit` / `FLASHNEXT_CONTEXT_LIMIT` | set per launch |
| `memory_admitted_max_cold_prefill` | what admission allows to be prefilled from cold on this machine now | set per launch |
| `validated_on_this_machine` | largest real cold prefill that finished | 3,935 tokens (8K not passed) |

A prompt over the configured limit gets HTTP 400 `context_length_exceeded`; nothing is truncated.

### What 262,144 would cost on this Mac (analytic, not measured)

* Attention scratch is 48 B per (query, key) pair, so a 256-token chunk at 262K position is about 3.2 GiB. Activation memory
  is bounded per chunk, but not independent of position.
* KV cache for the 12 attention layers: about 27.8 KB/token bf16, so about 7.3 GiB at 262K (about 2 GiB if affine4 were
  validated at that length, which it is not).
* Every chunk re-streams all routed experts from SSD, so cost is per chunk, not per token. At the measured 25 tok/s a cold 262K
  prefill is hours.
* On a 16 GB machine with a 113 GB checkpoint this does not pass admission. An explicit `--context-limit` above every tier is a
  requirement and is refused (`VEYSTRIX_CONTEXT_REQUIRED`). Only the owner-set `FLASHNEXT_MEMORY_WAIVER=swap-only` relaxes memory
  thresholds, and it keeps the +2.5 GiB swap stop.

Verdict: 262,144 is the checkpoint's native capability, not a supported operating point. It is exposed only as an experimental
snippet (below), never as the default, and advertising it does not change admission.

Experimental snippet (not applied; replace `contextWindowSize` only after a launch that was really admitted at that size):

```json
"generationConfig": {"contextWindowSize": 262144}
```

## 3. Reasoning effort (`/effort`)

Measured against the real client with a capture server: `/effort low|medium|high|xhigh` persists `model.reasoningEffort` and the
generic OpenAI provider sends a nested `reasoning: {"effort": "<level>"}` on each request (`max` is clamped to `xhigh` by the
client). A non-empty `samplingParams` suppresses that injection, which is why none is configured.

Server side (`macqwen/reasoning.py`, `macqwen/server.py`):

| Request | Result |
|---|---|
| `reasoning.effort` or `reasoning_effort` = low / medium / high / xhigh | thinking on at that level |
| `max` | `xhigh`, with a note that max has no distinct level |
| `none`, `off`, `false`, `reasoning:false`, `enable_thinking:false` | thinking off; an explicit disable beats an effort in the same request, noted |
| `chat_template_kwargs.enable_thinking` | honoured; other template kwargs refused |
| unknown level, `reasoning.max_tokens`, other `reasoning` keys | HTTP 400, never ignored |
| nothing | the server defaults, reported as `server-default` |

Honest limit: the checkpoint template has three effort sentences (xhigh default, medium, low) plus on/off. `high` is the template's
medium plus the repo's own extra sentence, so four levels are distinct prompts but there is **no decoder-level thinking budget**,
and `max` is not separate from `xhigh`.

Isolation: the controls are resolved per request into a thread-local, never written to the shared session preferences, so a title
or summary request with other controls cannot change the next main request. Each response carries `X-Veystrix-Reasoning`
(selected level, source, wire field, notes), and the request log has a `reasoning` record.

Cache identity: the effort sentence is at the start of the system turn, so changing level changes every prefix token. The exact
token comparison then misses and the conversation is rebuilt. That is correct but costs a full prefill, so switch effort between
tasks, not mid-way through a long one. With thinking on, a resent history also rebuilds, because the client echoes replies without
their reasoning and the template re-renders those turns as empty think blocks (test
`test_with_thinking_on_a_resent_history_rebuilds_...`; the first difference is after the system prompt, not in the effort text).
With thinking off the live state is reused across turns.

Tests: `macqwen/tests/test_reasoning.py` (14, real tokenizer and template): per-level rendered prompts differ, low→high→medium→
xhigh→low follows each request and stores nothing on the session, auxiliary request does not leak.

## 4. Long-prefill stall (preserved 8K attempt)

Evidence: `results/flashnext/20261108-232323-gate-8k-attempt1/`. Established offline:

1. The adaptive scheduler stepped chunks down on moderate pressure or a thin comfort margin. This engine streams every expert
   once per chunk, so smaller chunks cost about the same wall time each. Replaying the recorded trace: 13 chunk passes instead of
   9, 423 GiB paged in against about 290 GiB for the plan. Fixed: it now steps down only on macOS pressure level 4 or under
   2 GiB available, and steps up after one safe chunk (`prefill_scheduler.py`, replay regression test).
2. A tiny real-dimension attention timing is flat at 0.49 s per chunk at 8K, so attention is not the stall.
3. Zero page-in stretches are consistent with F_NOCACHE reads, which bypass the file cache and do not appear as vm_stat page-ins.
   The uncached-after-ELEVATED switch is now opt-in (`FLASHNEXT_UNCACHE_WHEN_ELEVATED=1`).

**Not established:** why the second half of the real run still did not finish in 900 s. New instrumentation is ready for the next
approved run: `FLASHNEXT_CHUNK_LOG` (per-chunk JSONL), `StackProbe` stack samples, controller state per chunk. Raising the timeout
would not be a fix; the watchdog, limits and cancellation are unchanged.

## 5. What was verified offline (real Qwen Code 0.22.0, real server, real tokenizer, scripted model)

`macqwen/tests/test_qwen_code_client.py`, 14 tests, all pass (31 s):

* `/v1/models`: id `macqwen-flashnext`, four context numbers, reasoning capability block.
* Bearer auth: wrong key gives `401 invalid API key` surfaced as `[API Error: ...]`, nothing generated; right key works.
* Streaming and usage: SSE with `include_usage` reaches the client; exactly one request per prompt once background features are off.
* Tools: sequential `glob` then `read_file` continuation (live state reused, no XML leaks), two parallel `read_file` calls in one reply with both results returned, an argument the tool does not accept is refused by the server (`tool 'read_file' has no parameter ...`) and nothing runs, `write_file` is blocked in plan mode and works in yolo mode in a temp repo.
* Effort: low, high, medium, xhigh, low each reach the server as `reasoning.effort`, produce four different prompts, store nothing on the session; two client homes with different efforts do not leak; `max` arrives as `xhigh`.
* Accounting: the prompt-token increment for 4K, 8K and 12K user content matches the real tokenizer within 64 tokens. Request sizes with the trimmed tool set: 8,610 tokens baseline ("hello", plan mode, 3 tools), 12,176 / 16,300 / 20,376 with 4K / 8K / 12K of user content. Untrimmed: 21.9K tokens, 28 tools.
* Overflow: a prompt over the server limit gets a visible context error and no generation.
* Cancellation: killing the client process group stops the server-side generation (a `SIGKILL` of only one pid leaves Qwen Code's child alive, so the connection stays open: use Ctrl-C or kill the group).

Regression: MACQWEN 571 tests OK; FlashNext 1,532 tests, 3 skipped, 1 failure (`FLASHNEXT_UNCACHE_WHEN_ELEVATED` not in the settings registry), fixed by registering it; the registry and memory-controller suites pass afterwards. The full FlashNext suite has not been re-run since that one-line registry fix.

## 6. Launch

```
cd /Users/vincent/Documents/Veystrix
./chat.sh --mode agentic --context-limit 16384     # serves http://127.0.0.1:8080/v1 once memory admission passes
# a refusal here is the memory gate working; it is never bypassed by asking for 262144
```

In another terminal:

```
export VEYSTRIX_API_KEY=local          # any value for a local server started without --server-api-key
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool    # check context.configured_server_limit
cd ~/some/project && qwen              # /effort low|medium|high|xhigh applies to the next request
```

Plugged in, Low Power Mode off. Do not start a real session until the next full-checkpoint experiment (section 7) is approved.

## 7. Proposal for the next full-checkpoint experiment (needs approval, not run)

* Config: one 8K real run, `FLASHNEXT_CHUNK_LOG` and stack sampling on, scheduler fix in, plugged in, Low Power Mode off,
  waiver `swap-only` only if the owner re-grants it for that run.
* Memory readiness: startup 9 GiB gate unless waived; +2.5 GiB swap stop retained.
* Acceptance: prefill completes with chunk log showing 9 passes at 1,024 tokens (±1), no step-down below 512, no stop,
  then a tool turn and a follow-up reuse the live state (`reused` increments, appended tokens < 200).
* Not proposed: 12K, 262K.

## 8. Conclusions and limits

* Protocol compatibility with Qwen Code 0.22.0 is established offline. It is not evidence that a long real prefill works.
* The practical blocker is size: even trimmed, Qwen Code's first request is 8.6K-10.1K tokens, and every added turn grows it.
  That needs the `long` tier (16,384), whose cold prefill (about 10K tokens) is exactly the thing the real 8K attempt failed to
  finish. Until the stall is explained and a real run passes, a real Qwen Code session should be considered unproven.
* Effort levels: four distinct prompts, no thinking budget, `max` equals `xhigh`.
* No 262K run, no 12K run, no real-checkpoint run was made.
