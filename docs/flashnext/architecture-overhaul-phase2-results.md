# Phase II architecture results — in progress

Updated 2026-11-10. **The optimized long-lived server is not safety-validated.** The machine was on battery during this work, so no AC-only real-checkpoint experiment was started. The previously observed 8K-after-decode swap breach remains open. Do not use the candidate as a production server or interpret the offline tests below as proving the 3 GiB ceiling.

## Architecture and evidence

The existing layer-major 8,192-token windows, two-half expert prefetch ring, inner snapshots and exact persistent cache remain in place and opt-in. The default remains the reference chunk-major server. The new layer-major admission gate checks fresh swap and pressure readings, a live guard, and estimated headroom after optional recovery; it reserves for output tokens and planned prefix snapshots. It defaults the candidate's absolute guard to 3,072 MiB with a 768 MiB margin. A separate watchdog samples swap through `sysctlbyname` every 0.25 seconds by default and can immediately terminate the identity-checked worker at the safety margin; admission requires its fresh heartbeat. These are conservative estimates, not bounds on allocator peaks or concurrent external allocations. Sampling and termination latency cannot guarantee a ceiling against an arbitrarily fast spike. A supervised live safety gate is still required.

An interrupted or failed expert-pool fill now drains submitted writes and invalidates the affected slots before reuse. Descriptor initialization is serialized. This protects subsequent requests from partially populated cache entries; it is not a measured speed optimization. An experimental `--enable ngram-dedup` removes duplicate row reads within each bounded lookup and restores original order; it is disabled by default and only has synthetic exactness evidence. Decode profile summaries now preserve actual source layer IDs and report p95 layer-arrival gaps. Those host gaps include several stages and are not GPU execution times.

The external research and compatibility caveats are recorded in [phase2-reference-research.md](phase2-reference-research.md). No new native engine or Metal kernel has passed an end-to-end gate. Existing MTP experiments and predicted expert prefetch remain disabled; no new real-checkpoint speculation measurement was made.

On the settled offline tree, `.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py'` ran 1,779 tests successfully (3 skipped); `.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'` ran 686 successfully. Both exited 0. These are checkpoint-free regressions, not a model exactness or production-safety gate. An earlier unqualified discovery did not load the test package's temporary warm-cache isolation, and an earlier FlashNext run overlapped source edits; neither is used as final evidence.

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

Reference rollback (existing default):

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2
```

Prior measured prefill configuration, **experimental and not validated for long-lived use**:

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable layer-major --enable layer-prefetch
```

Prior measured decode configuration, separate from the preceding prefill combination:

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

Task status and prior test commands are in [phase2-task-ledger.md](phase2-task-ledger.md). The original checkpoint and existing commits are unchanged; no push was made.
