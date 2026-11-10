# Phase II architecture results — in progress

Updated 2026-11-10. **The optimized long-lived server is not safety-validated.** The machine was on battery during this work, so no AC-only real-checkpoint experiment was started. The previously observed 8K-after-decode swap breach remains open. Do not use the candidate as a production server or interpret the offline tests below as proving the 3 GiB ceiling.

**Routing correction:** The historical server default was a 0.85 retained-mass threshold, which drops routed experts. Its recorded current-checkpoint decode trace contains 4–9 routed expert IDs per layer/token, not the required 10. Consequently, the historical throughput and same-setting A/B exactness numbers below do not verify full top-10 model behavior or performance. The layer-major architecture and snapshots remain in place, but all model exactness, physical I/O, safety and performance gates must be repeated at full routing.

## Architecture and evidence

The existing layer-major 8,192-token windows, two-half expert prefetch ring, inner snapshots and exact persistent cache remain in place and opt-in. The default remains the reference chunk-major server, now with the full-routing threshold of 1.0. Changing a numerical routing profile invalidates the live state for replay and the prefix namespace; a saved session with a different routing profile is refused rather than silently restored. Legacy approximate options warn when selected. The new layer-major admission gate checks fresh swap and pressure readings, a live guard, and estimated headroom after optional recovery; it reserves for output tokens and planned prefix snapshots. It defaults the candidate's absolute guard to 3,072 MiB with a 768 MiB margin. A separate watchdog samples swap through `sysctlbyname` every 0.25 seconds by default and can immediately terminate the identity-checked worker at the safety margin; admission requires its fresh heartbeat. These are conservative estimates, not bounds on allocator peaks or concurrent external allocations. Sampling and termination latency cannot guarantee a ceiling against an arbitrarily fast spike. A supervised live safety gate is still required, particularly because its allocation estimates were calibrated on pruned routing.

An interrupted or failed expert-pool fill now drains submitted writes and invalidates the affected slots before reuse. Descriptor initialization is serialized. This protects subsequent requests from partially populated cache entries; it is not a measured speed optimization. An experimental `--enable ngram-dedup` removes duplicate row reads within each bounded lookup and restores original order; it is disabled by default and only has synthetic exactness evidence. Decode profile summaries now preserve actual source layer IDs and report p95 layer-arrival gaps. Those host gaps include several stages and are not GPU execution times.

The external research and compatibility caveats are recorded in [phase2-reference-research.md](phase2-reference-research.md). No new native engine or Metal kernel has passed an end-to-end gate. Existing MTP experiments and predicted expert prefetch remain disabled; no new real-checkpoint speculation measurement was made.

On the settled offline tree, `.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py'` ran 1,779 tests successfully (3 skipped); `.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'` ran 686 successfully. Both exited 0. These are checkpoint-free regressions, not a model exactness or production-safety gate. An earlier unqualified discovery did not load the test package's temporary warm-cache isolation, and an earlier FlashNext run overlapped source edits; neither is used as final evidence.

After the full-routing correction and watchdog review, the same commands ran 1,791 FlashNext tests (3 skipped) and 689 MacQwen tests, both exit 0. A subsequent test-only isolation change was followed by 78 focused prefix, routing, smoke-gate and one-sync tests, all passing. The full-routing unit test checks ten distinct expert IDs for synthetic one-row and multi-row inputs. It does not replace a real-checkpoint exactness gate.

## Claude Code session, 2026-11-10: first real full-routing evidence

Commit `0071fb2` (offline: 1,803 FlashNext tests with 3 skipped, 690 MacQwen, both exit 0). The owner allowed battery operation; every arm records source and charge.

**Full top-10 routing is confirmed on the real checkpoint through the supervised launcher.** Command: `.venv/bin/python -m models.flashnext.tests.bench.bench_decode_profile --source-tokens 100 --tokens 24 --arms io --power-rules battery-ok --settle 30`, results in `results/flashnext/20261110-115952-decode-profile-manual/`. Chunk-major reference, pool 2 GB, threshold 1.0, 2,239-token prompt, battery 72% → 71% discharging, Low Power Mode off, swap flat at 853.5 MiB for all 21 safety samples. All 1,152 recorded layer calls (24 tokens × 48 layers) routed exactly 10 distinct experts; the old pruned trace had 4–9. This confirms the setting reaches the model. It does not by itself verify logits against an independent full-routing reference.

Diagnostic (profiled arm; not a speed result, one arm, no resolution band): decode 2.08 tok/s, median 439 ms/token; host-measured `io_wait` 352 ms/token, `router_sync` 49.5 ms/token (overlapping host timers, not GPU time); about 1.09 GB/token of process physical reads over 23 later tokens (includes housekeeping; the pruned-routing figure was about 0.73 GB/token, a different run); 2,239-token prefill took 63.5 s (about 35 tok/s, profiled, chunk-major). The optional pool I/O counters were not instrumented on this path (`expert_io_available: false`), so cache hits are not reported.

Uncommitted-work review outcomes: the oversized-pool fallback streams every routed expert in original order and leaves the pool untouched (the G32 cast is a no-op on bf16 executor output); `bench_phase2_after_turns.py` now has an explicit `--power-rules battery-ok` mode with a charge floor (default 40%, never below 25%), a pre-8K power recheck and recorded power rules. Its own headroom estimate refused the 8K request at 7.9 GB available versus about 11.5 GB required before any purge, which is a valid safe refusal. Synthetic tests only; the 8K-after-turns gate has not run.

## Performance

These are prior recorded results on this checkpoint, not new Phase II measurements. Conditions and raw arms are in [overnight-architecture-results.md](overnight-architecture-results.md). The new code has no measured throughput claim.

| Measurement | Reference | Prior opt-in candidate |
|---|---:|---:|
| 8K cold prefill | 48.3 tok/s | 89.5 tok/s |
| 16K cold prefill | 44.7 tok/s | 69.2 tok/s |
| Short decode, 2.7K context | 2.77 tok/s | 3.60 tok/s buffered pool fill |
| Three-turn sustained decode | 2.13–2.71 tok/s | 2.56–3.50 tok/s buffered pool fill |
| 128-token sustained decode | Not measured in this Phase II configuration | Not measured |
| 8K physical prefill reads | 277–279 GB | 89 GB |
| Physical decode reads | ~0.73 GB/token in prior profile | buffered fill reduced physical reads by ~36% in the 2.7K comparison; different run |
| Prior diagnostic decode split | ~290 ms/token I/O wait; ~72 ms/token router synchronization | actual GPU execution and overlap not isolated |
| Peak swap | Prior 16K arms up to 2.78 GiB | 8K-after-decode reached 3.33 GiB and was stopped |
| Cache and restoration | Prior single follow-up matched 2,424/2,452 tokens; new-process SSD restored 2,107 tokens | No new Phase II validation |
| Real client TTFT / throughput | Not measured with these candidate paths | Not measured |

The prior 0.73 GB/token and ~2.5 GB/s imply an illustrative ~292 ms I/O floor if the reads do not overlap, not an exact per-token critical path. Buffered reads, filesystem cache, thermal state, and overlapping work change that calculation. A full GPU timeline and traces from multiple current coding/tool prompts remain outstanding.

An exploratory offline replay used the existing `20261109-231546-decode-profile-manual/p2-io-profile` route trace, splitting its 84 profiled tokens into 42 training and 42 holdout tokens. At 651 simulated slots, global LRU had 11,610 misses, fixed per-layer LRU 10,942 and the layer-cycle Least-Stale approximation 10,894 out of 15,823 accesses. The holdout began with an empty simulated pool. The same replay's idealized width-2 verification required 12,632 simulated misses versus 11,610 at width 1, despite assuming perfect acceptance and zero drafting cost. This is one short trace, simulated expert-record demand rather than physical I/O, and the policies do not necessarily model the production pool exactly. It warrants a broader trace study, not a runtime promotion or conclusion about actual MTP throughput.

## Safety and commands

Reference execution (chunk-major and full routing; not yet benchmarked at full routing):

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2
```

Layer-major candidate, **experimental and not validated for long-lived use**. This command now uses full routing and does not reproduce the earlier pruned-routing timings:

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable layer-major --enable layer-prefetch
```

Buffered decode candidate, separate from the preceding prefill combination; its full-routing throughput has not been measured:

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable pool-buffered-fill
```

The candidate admission gate can be disabled with `FLASHNEXT_REQUEST_ADMISSION=0`; that rollback removes a safety layer and is not validated for the previously failing scenario. `FLASHNEXT_WATCHDOG_MEMORY=0` disables the independent memory backstop, causing admission to refuse unless admission is also explicitly disabled. The absolute guard can be overridden through its existing environment settings; disabling or weakening it does not make a workload safe. An admission refusal is HTTP 503 for non-streaming requests; streaming headers are already open, so clients receive a protocol-specific SSE error. A refused session may need to rebuild or restore its live state. A watchdog safety stop terminates the worker, with best-effort client notification and a local report; restart and cleanup need live validation.

After the long-session gate passes, the documented clients are `VEYSTRIX_API_KEY=local qwen` with the provider setup in [qwen-code-integration-2026-11-09.md](qwen-code-integration-2026-11-09.md), and `DSH_HOME=<profile> VEYSTRIX_API_KEY=local dsh headless --patch docs/flashnext/dsh/veystrix-agentic.patch.yml "task"`. Neither has been validated with the Phase II candidate.

## Next executable work

1. On AC, verify the new gate and independent watchdog together under representative conditions; first measure worst-case request allocation and snapshot staging. Refuse the live test if preflight or monitoring cannot provide adequate margin. Then run the bounded earlier-turns → 8K scenario, stopping at the first anomaly.
2. Capture current-checkpoint coding and tool-use routes and a buffered-fill decode profile, with unprofiled controls. Measure physical SSD reads, wait distributions and GPU execution separately; do not add overlapping timers. Prototype demand-read/resident-expert overlap or native submission only if those measurements justify it.
3. Measure n-gram dedup with exact tokens, logits and state, physical reads and peak memory; reject it if end-to-end performance or safety regresses. Revisit 16K window scheduling and exact MTP only under the same gates.
4. After the safety gate, validate restart/restoration, streaming/cancellation, all three API protocols and bounded real Qwen Code and DeepSeek Harness sessions. No 262K execution has been validated.

The smoke verdict now requires the expected tool call and arguments, a completed error-free stream, and valid Anthropic and Responses envelopes. The independent watchdog also fails closed on a persistently unreadable progress record or its own unexpected loop failure and skips slow stall diagnostics while memory monitoring is active. These are offline tests; failure and cleanup on the actual server remain open.

Task status and prior test commands are in [phase2-task-ledger.md](phase2-task-ledger.md). Local commit `a3a07d1` contains the offline safety, cache-fill, profiling, n-gram and documentation work. The original checkpoint and existing commits are unchanged; no push was made.
