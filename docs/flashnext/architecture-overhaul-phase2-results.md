# Phase II architecture results — in progress

Updated 2026-11-10. **The optimized long-lived server is not safety-validated.** The machine was on battery during this work, so no AC-only real-checkpoint experiment was started. The previously observed 8K-after-decode swap breach remains open. Do not use the candidate as a production server or interpret the offline tests below as proving the 3 GiB ceiling.

**Routing correction:** The historical server default was a 0.85 retained-mass threshold, which drops routed experts. Its recorded current-checkpoint decode trace contains 4–9 routed expert IDs per layer/token, not the required 10. Consequently, the historical throughput and same-setting A/B exactness numbers below do not verify full top-10 model behavior or performance. The layer-major architecture and snapshots remain in place, but all model exactness, physical I/O, safety and performance gates must be repeated at full routing.

## Architecture and evidence

The existing layer-major 8,192-token windows, two-half expert prefetch ring, inner snapshots and exact persistent cache remain in place and opt-in. The default remains the reference chunk-major server, now with the full-routing threshold of 1.0. Changing a numerical routing profile invalidates the live state for replay and the prefix namespace; a saved session with a different routing profile is refused rather than silently restored. Legacy approximate options warn when selected. The new layer-major admission gate checks fresh swap and pressure readings, a live guard, and estimated headroom after optional recovery; it reserves for output tokens and planned prefix snapshots. It defaults the candidate's absolute guard to 3,072 MiB with a 768 MiB margin. A separate watchdog samples swap through `sysctlbyname` every 0.25 seconds by default and can immediately terminate the identity-checked worker at the safety margin; admission requires its fresh heartbeat. These are conservative estimates, not bounds on allocator peaks or concurrent external allocations. Sampling and termination latency cannot guarantee a ceiling against an arbitrarily fast spike. A supervised live safety gate is still required, particularly because its allocation estimates were calibrated on pruned routing.

An interrupted or failed expert-pool fill now drains submitted writes and invalidates the affected slots before reuse. Descriptor initialization is serialized. This protects subsequent requests from partially populated cache entries; it is not a measured speed optimization. An experimental `--enable ngram-dedup` removes duplicate row reads within each bounded lookup and restores original order; it is disabled by default and only has synthetic exactness evidence. Decode profile summaries now preserve actual source layer IDs and report p95 layer-arrival gaps. Those host gaps include several stages and are not GPU execution times.

The external research and compatibility caveats are recorded in [phase2-reference-research.md](phase2-reference-research.md). No new native engine or Metal kernel has passed an end-to-end gate. Existing MTP experiments and predicted expert prefetch remain disabled; no new real-checkpoint speculation measurement was made.

On the settled offline tree, `.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py'` ran 1,779 tests successfully (3 skipped); `.venv/bin/python -m unittest discover -s veystrix/tests -t . -p 'test_*.py'` ran 686 successfully. Both exited 0. These are checkpoint-free regressions, not a model exactness or production-safety gate. An earlier unqualified discovery did not load the test package's temporary warm-cache isolation, and an earlier FlashNext run overlapped source edits; neither is used as final evidence.

After the full-routing correction and watchdog review, the same commands ran 1,791 FlashNext tests (3 skipped) and 689 Veystrix tests, both exit 0. A subsequent test-only isolation change was followed by 78 focused prefix, routing, smoke-gate and one-sync tests, all passing. The full-routing unit test checks ten distinct expert IDs for synthetic one-row and multi-row inputs. It does not replace a real-checkpoint exactness gate.

## Claude Code session, 2026-11-10: first real full-routing evidence

Commit `0071fb2` (offline: 1,803 FlashNext tests with 3 skipped, 690 Veystrix, both exit 0). The owner allowed battery operation; every arm records source and charge.

**Full top-10 routing is confirmed on the real checkpoint through the supervised launcher.** Command: `.venv/bin/python -m models.flashnext.tests.bench.bench_decode_profile --source-tokens 100 --tokens 24 --arms io --power-rules battery-ok --settle 30`, results in `results/flashnext/20261110-115952-decode-profile-manual/`. Chunk-major reference, pool 2 GB, threshold 1.0, 2,239-token prompt, battery 72% → 71% discharging, Low Power Mode off, swap flat at 853.5 MiB for all 21 safety samples. All 1,152 recorded layer calls (24 tokens × 48 layers) routed exactly 10 distinct experts; the old pruned trace had 4–9. This confirms the setting reaches the model. It does not by itself verify logits against an independent full-routing reference.

Diagnostic (profiled arm; not a speed result, one arm, no resolution band): decode 2.08 tok/s, median 439 ms/token; host-measured `io_wait` 352 ms/token, `router_sync` 49.5 ms/token (overlapping host timers, not GPU time); about 1.09 GB/token of process physical reads over 23 later tokens (includes housekeeping; the pruned-routing figure was about 0.73 GB/token, a different run); 2,239-token prefill took 63.5 s (about 35 tok/s, profiled, chunk-major). The optional pool I/O counters were not instrumented on this path (`expert_io_available: false`), so cache hits are not reported.

Uncommitted-work review outcomes: the oversized-pool fallback streams every routed expert in original order and leaves the pool untouched (the G32 cast is a no-op on bf16 executor output); `bench_phase2_after_turns.py` now has an explicit `--power-rules battery-ok` mode with a charge floor (default 40%, never below 25%), a pre-8K power recheck and recorded power rules. Its own headroom estimate refused the 8K request at 7.9 GB available versus about 11.5 GB required before any purge, which is a valid safe refusal. Synthetic tests only; the 8K-after-turns gate has not run.

**8K-after-turns gate, 2026-11-10: safely postponed, not run.** Command: `.venv/bin/python -m models.flashnext.tests.bench.bench_phase2_after_turns --power-rules battery-ok --min-battery 40 --pool-gb 2` (battery 71%, discharging, swap 853.5 MiB). Preflight, after its own purge attempt, found 11,635 MiB available against the 11,908 MiB the layer-major candidate needs (3,072 MiB floor + footprint + 512 MiB margin), so it stopped before starting any model process (`results/flashnext/20261110-120207-phase2-after-turns-manual/`). This is a valid safe refusal. The floor was not lowered. Re-run after freeing about 300 MiB or more of memory (for example by closing other applications); no model-side change is needed.

## Full-routing decode baseline and buffered pool fill, 2026-11-10 (measured)

Real checkpoint, threshold 1.0, chunk-major reference, 2 GB pool, 2,739-token prompt (`--source-tokens 600`), 64 greedy decode tokens, battery-ok (71% → 58% across all runs, discharging, Low Power Mode off, swap 845–854 MiB throughout), each arm a fresh supervised process with the independent watchdog. Raw arms are in `results/flashnext/20261110-120244-`, `-121327-`, `-122054-` and `-122626-decode-profile-manual/`. Commands: `bench_decode_profile --source-tokens 600 --tokens 64 --arms plain,io,host,plain2` (default fill) and the same with `--buffered-fill`.

| Condition (unprofiled arms) | Decode tok/s | Process physical read per later output token |
|---|---:|---:|
| Default pool fill, order A: 2.220, 2.202 (first batch); order B: 2.187, 2.164 | 2.16–2.22 | 1.04–1.06 GB |
| Buffered pool fill (`--enable pool-buffered-fill`), order A: 2.795, 2.867; order B: 2.853, 2.841 | 2.80–2.87 | 0.70–0.74 GB |

Prefill of the 2,739-token prompt took 72.2–73.8 s in every arm (about 37 tok/s, chunk-major reference at full routing; buffered fill does not change prefill). Correction: an earlier draft of this section said 2,239 tokens; that was the separate 100-source-token run.

Order A ran default first, then buffered; order B ran buffered first, then default, so the gain does not come from run order. Within-condition spread is at most about 3% and the gap is about 28–30%, with physical reads about 30% lower. Token IDs were identical across all 11 arms (digest `1e055db49628`), including profiled arms. This is exactness at the greedy token-ID level for one prompt, not a logits comparison, and does not cover other prompts, longer decodes, multi-turn sessions or sustained page-cache pressure. Reads are process I/O and include housekeeping. Fresh-process arms ran back to back on a fanless Mac with a 45–60 s settle.

Profiled arms (`io`, `host`) decoded within about 2% of the unprofiled controls (2.18 versus 2.20–2.22), so the diagnostic split is representative here. Default fill: median 453 ms/token (p95 about 532 ms), host-measured `io_wait` 354 ms/token and `router_sync` 64 ms/token, 48 read calls per token. Buffered fill: median 345 ms/token (p95 478 ms), `io_wait` 259 ms/token, `router_sync` 65 ms/token. These host timers overlap and are not GPU time; the I/O wait still dominates, and buffered fill removed about 95 ms/token of it. GPU execution time is not yet separated. Every one of 3,072 profiled layer calls per arm routed exactly 10 distinct experts. The pool hit/miss counters are not fed on this path, so logical cache hits are not reported.

Decision: buffered fill stays opt-in. The evidence is strong for this setting but narrow, and AGENT_INVARIANTS require promotion evidence beyond a single prompt. It is the best justified decode improvement so far and a candidate default once multi-turn and longer-context arms with memory traces agree. Use `VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable pool-buffered-fill`. The 8K layer-major prefill claims (48.3 → 89.5 tok/s) remain pruned-routing history, not full-routing results.

**Protocol smoke on the reference server, 2026-11-10 (full routing, battery 58% → 55%, swap flat at 845.5 MiB).** `bench_integration_smoke --server reference --power-rules battery-ok` (`results/flashnext/20261110-123259-integration-smoke-manual/`). Passed: tool turn returned the correct `read_file` call and arguments; cancelling a roughly 4.6K prefill after 15 s returned the server to ready in 2 s and the next request succeeded; Anthropic Messages and OpenAI Responses returned valid envelopes. Flagged by the unchanged verdict: the streamed follow-up completed cleanly (3 events, no error, `finish_reason: tool_calls`) but carried no text, because the model chose another tool call instead of answering. That is model behavior, not a protocol error, but the smoke did not pass and the verdict was not loosened. The same smoke on the layer-major candidate (`20261110-123142-`) made no protocol checks: its request admission refused every request at 4.8 GiB available versus 6.5–8.9 GiB required, which is correct safe behavior and a usability limit of that candidate on this machine.

**8K-after-turns gate, second attempt, 2026-11-10 (owner had closed other apps): refused by the server's request admission.** Preflight passed (about 11.9 GiB available) and the layer-major candidate started under the watchdog, but its first short request was refused with `memory_admission_refused` (4.94 GiB available, 6.26 GiB required) once the model was resident (`results/flashnext/20261110-123719-phase2-after-turns-manual/`). Swap stayed at 845.5 MiB and no 8K request was sent. Loading the engine takes roughly 7 GiB of the 11.9 GiB, so the candidate's conservative requirement (idle headroom of about 5.5 GB plus a layer-major extra of about 1.6 GB) cannot be met on this 16 GB machine even for a tiny request. The reference chunk-major server admitted the same requests. The requirement was calibrated on pruned routing and has not been re-measured at full routing; it was not lowered.

## Full-routing layer-major 8K prefill, paired (measured, 2026-11-10)

Owner decision, 2026-11-10: the only inference-engine memory limit is swap in use at 3,072 MiB. The layer-major request-admission check (available-memory requirement, pressure refusal, per-request headroom estimate) was removed; it now refuses only at the swap ceiling, and the early-stop margin is 0 (stop exactly at 3 GiB). Earlier failed runs (`results/flashnext/20261110-123719-`, `-124338-`, and the first `-1243` retry) were refusals by the removed check at 4.9–5.3 GiB available against 6.3–9.3 GiB required; `FLASHNEXT_REQUEST_ADMISSION=0` set in the launching shell does not reach the server, so it did not help. The pre-request purge remains (it never refuses). Battery and thermal checks remain. The strict gate's 3 GiB available-memory floor stopped a single-arm 8K reference run at 2,727 MiB available with swap flat; `swap-only` is the rule used from here on.

Command: `.venv/bin/python -m models.flashnext.tests.bench.bench_layer_gate --candidate layer-major --prefetch --reads prefill --pool-gb 2 --source-tokens 6000 --rounds 3 --settle 30 --memory-rules swap-only --power-rules battery-ok`; raw arms in `results/flashnext/20261110-125313-layer-gate-manual/`. Full routing (threshold 1.0), about 8K-token prompt, reversed pair order, fresh supervised process per arm, independent watchdog. Started on battery at about 49%; the Mac was charging partway through, so the run is preliminary under `docs/measurement-standard.md`.

| Pair | Chunk-major reference | Layer-major + prefetch | Speed-up |
|---|---:|---:|---:|
| 1 | 166.6 s | 95.2 s | 1.75× |
| 2 | 184.0 s | 100.1 s | 1.84× |
| 3 | 184.5 s | 108.6 s | 1.70× |

Mean-time ratio 1.76×; the harness's median tok/s ratio was 1.839×. About 300 GB of physical SSD reads per reference arm against about 90 GB per layer-major arm. MLX peak 9.74 GB in layer-major arms. All six arms produced the identical output digest (`67264f3720`) and all three pair validations passed (token-ID level, one prompt, 8 output tokens; not a logits comparison). Peak swap 845–1,565 MiB in reference arms and 1,278–1,621 MiB in layer-major arms, always below 3,072 MiB; minimum available memory in layer-major arms was about 1.4–1.7 GiB. Swap climbed across the session and was not released between arms. This is the first full-routing evidence for layer-major prefill; it replaces the pruned-routing 89.5 tok/s history. It does not cover decode (unchanged), 16K, multi-turn state, or real clients.

**8K-after-turns session on layer-major, 2026-11-10: completed.** `bench_phase2_after_turns --power-rules battery-ok --min-battery 40 --pool-gb 2` under the swap-only rule (`results/flashnext/20261110-131831-phase2-after-turns-manual/`; battery 57% and charging, so preliminary). Two short turns through the normal supervised server (the second reused 28 cached tokens, 21 new), then a cold 8,139-token request with 0 cached tokens that finished normally (8 output tokens). Peak swap 1,353 MiB, minimum available memory 1,644 MiB, never near the 3,072 MiB limit; the harness's pre-request swap reading was 1.32 GiB and it admitted the request under the swap-only rule. This is the first completed run of this gate. Limits: the earlier breach (8K after a long decode, pruned routing, five-second monitor) is not reproduced by this scenario, whose earlier turns were one token each, and it is a single run with no cleanup, cancellation or restart evidence. Prefill time of the 8K request was not separated here.

## Cross-session cache and client warm-up, 2026-11-10 (partial; hand-off notes)

**New-process prefix restoration works at full routing (reference server).** `bench_session_restore --power-rules battery-ok` (`results/flashnext/20261110-132124-session-restore-manual/`): a cold 2,732-token prompt prefilled in 73.3 s; a fresh process on the same cache directory with the same long prefix and a different last line restored 2,107 tokens from SSD (290 MB) and prefilled the remaining 645 tokens in 25.6 s, about 2.9x faster. Swap flat at 1,337 MiB. One pair, no resolution band.

**Warm pass on the supervised layer-major server** (`./serve.sh --warm`, 2 clients x 4 efforts; first server run): DeepSeek Harness low cold 69.7 s (5,427 tokens); medium restored 5,385 of 5,397 tokens in 6.1 s; high and xhigh restored 5,385 tokens (SSD hit, RAM hit) so the system prompt is shared across thinking levels as designed (`cache-across-efforts-2026-11-09.md`), but each took about 96 s for only 55 new tokens (see problem 2). Qwen Code low ran 126 s and then the server was stopped by a guard (problem 1), so Qwen Code and its other efforts are unwarmed and no real `dsh` or `qwen` task has been run at full routing. Overnight autonomous sessions are therefore not validated.

**Problem 1, Metal-limit guard stop (fixed offline, unverified live).** The supervised server exited 75: "MLX holds 12.22 GB (active + cache), within 0.5 GB of the 12.71 GB Metal limit" during Qwen Code's roughly 10.7K-token cold layer-major prefill. The guard adds reclaimable MLX buffer cache to active memory. `chat_guard._recheck_after_cache_release` now clears the MLX cache once and re-reads before stopping; active memory near the limit still stops. Offline tests only (3 new). Whether active memory alone stays under the limit for a 10.7K or larger layer-major prefill is not measured; if it does not, the stop is a real GPU-limit stop and layer-major needs smaller windows for such prompts.

**Problem 2, layer-major makes small prefills slow (fixed offline, unverified live).** Layer-major engaged for any multi-chunk prefill and sweeps all 512 experts of every layer, so a 55-token increment took about 96 s. `FLASHNEXT_LAYER_MAJOR_MIN_TOKENS` (default 4096, `layer_major.min_tokens()`) keeps smaller prefills chunk-major. The 4096 value is a conservative guess; the break-even between 2.7K (chunk-major 73 s) and 8K (layer-major 95-109 s) has not been measured, and prefills of 4K-8K under layer-major are unmeasured. The existing layer-major integration tests set the threshold to 0.

**State at stop.** The server was stopped (nothing running). Changes for problems 1 and 2 have focused tests only (chat guard and diagnostics 171, layer-major 55, prefill 299 pass); the full FlashNext and Veystrix suites were not re-run after them. The untracked `docs/flashnext/claude-code-phase2-handoff.md` was left as the owner's file.

**Next, in order:** restart the layer-major server, finish `./serve.sh --warm` (Qwen Code, then repeat DeepSeek to confirm small effort-turn prefills are now seconds), run real `dsh headless` and `qwen` tasks on a scratch repo at several efforts and confirm cache hits through `/health`, check MLX active memory on a 10.7K layer-major prefill, measure the layer-major break-even at 2K-6K, then multi-hour overnight runs under `./serve.sh --overnight` with cache hit rates and swap logged. Also open: decode GPU split, speculation, 16K, full-suite re-run.

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
