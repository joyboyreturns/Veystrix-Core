# First AGENTIC run that loaded, prefilled, decoded and shut down cleanly (2026-11-08)

Evidence: `results/flashnext/20261108-agentic-smoke-retry-4-max-tokens-1/`. One launch, battery power (46%, accepted by the
owner, so the run is **preliminary** for performance), `./chat.sh --mode agentic --pool-gb 0`, automatic memory preparation,
independent watchdog on, real 113 GB Q4 checkpoint, the same read-only glob/grep workspace, localhost with a random key.

## Result by stage

| # | Stage | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Autonomous memory preparation and admission | **PASS** | Gate 8.93 GiB: the preparation purged file cache by itself (+2,318 MB in 0.2 s, to 11.09 GiB), then admitted at the **compact** tier (headroom 4.15 GiB against 4.00; context 2048, cold prefill up to 1536). Nobody freed memory by hand. |
| 2 | Cold prefill completes | **PASS** | 902 tokens, 20.0 s. 643 tokens were restored from the SSD prefix cache (0.10 s, 133 MB) and 259 prefilled. |
| 3 | Decode begins without hanging | **PASS** | First token at 20.05 s; no deadlock, no watchdog action. |
| 4 | Valid tool call | **FAIL (not reached)** | The model emitted exactly one token, `<tool_call>`, and stopped with `finish: length`. Cause below: the harness asked for one output token. This is neither a model tool-use failure nor a protocol success. |
| 5 | Tool result returns to the model | not reached | |
| 6 | Correct final answer | not reached | |
| 7 | No guard trip, no swap-out | **PASS** | Pressure 1 throughout, 0 swap-outs, only the usual `compressor_warning` event (as in the earlier completed prefill); no guard stop. |
| 8 | Clean shutdown | **PASS** | Ctrl-C to exit in 1.0 s, return code 0. The watchdog exited by itself afterwards. |

## Root cause of stage 4 (client configuration, found offline)

The harness sent `max_completion_tokens: 1`. It computes that as the declared `contextWindow` minus a fixed internal
prompt estimate of about 4,828 tokens, floored at 1, and the patch declared 4096. The server did what it was asked: it
generated one token and reported `length`. Measured offline with the real harness against a capturing mock (table in the
evidence directory): window 6144 gives 1316, and a model entry `maxTokens` caps the request exactly.

Fixes (none needs a model run to verify):

* `docs/flashnext/dsh/veystrix-agentic.patch.yml`: `contextWindow: 6144`, `maxTokens: 1024`. Verified offline: the harness
  now sends `max_completion_tokens: 1024`. The server still enforces its own smaller window and clamps to what fits, so
  these numbers only give the harness room to ask. Pinned by a test.
* The server no longer hands a client the raw `<tool_call>` marker as ordinary text when the output limit cuts a tool
  call off. It answers `422 model_tool_call_invalid` ("the model began a tool call but the output limit of N tokens ended
  it first"), also as the terminating event of a stream, which now carries the standard `type`/`param`/`code` object.
  Ordinary text that merely hits the limit is unchanged (`finish_reason: length`).

## Measurements (preliminary: battery)

| | This run | Earlier completed prefill | Earlier stopped prefill |
| --- | --- | --- | --- |
| Prompt tokens | 902 | 898 | 898 |
| MLX peak | 4.11 GiB | 4.11 GiB | 4.62 GiB |
| Process RSS, max | 4.09 GiB | 4.10 GiB | 3.72 GiB |
| Available memory, min | 4.30 GiB | 4.68 GiB | 3.26 GiB |
| Pressure, max | 1 | 1 | 2 (stopped) |
| Other anonymous memory at start | about 3.5 GiB | about 2.8 GiB | about 8.9 GiB |

The MLX peak of 4.11 GiB is identical to the earlier completed run. The compact tier budgeted 5.5 GiB for the prefill and
the run needed about 4.1 GiB of MLX plus process memory, so the tier model was conservative here, consistently with its
rounding up. One more real data point; it does not extend the evidence beyond roughly 900 tokens.

**Prefill speed.** The log's 45.0 tokens per second divides all 902 tokens by the time. The honest rate is 259 tokens
prefilled in 19.9 s, about 13 tokens per second, almost the same wall time as the earlier cold 898-token prefill (18.5 s).
One data point, but it says the cost of a prefill here is dominated by streaming experts from the SSD, not by token
count, so the prefix-cache restore saved little time in this case. This is an observation, not a proven mechanism, and it
means the cost of a conversation switch (see `universal-agentic-api.md`) is about 19 s whatever the size.

**Cache behavior.** An exact SSD hit of 643 tokens (the system-prompt boundary snapshot, `plan_edges [643]`) from a previous
process, restored in 0.10 s with no plan rejections. Equality against an uncached run was not checked in this run.

**Request log note.** `tool_calls: 1` in `requests.jsonl` counts the substring `<tool_call>` in the raw text; no valid tool
call existed.

## What this does and does not establish

Established (preliminary, one run): the repaired engine completes a real cold prefill and begins decoding without the
GPU/GIL deadlock; autonomous memory preparation and tiered admission work end to end on the real machine; the watchdog
stays quiet during a healthy run and exits with its worker; shutdown is clean.

Not established: a valid tool call from the real model, the tool-result continuation, a correct final answer, any
decode speed (one token), and anything above about 900 tokens.

Per the standing condition, **no retry was made.** One further run would be needed to complete the round trip, with the
corrected patch; it needs its own approval.
