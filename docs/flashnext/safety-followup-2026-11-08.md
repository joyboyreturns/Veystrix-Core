# Safety follow-up — 2026-11-08

Status: **not ready for F2Bm**. The full checkpoint was not loaded. Existing uncommitted work was preserved.

## Implemented

* Enabling automatic reclamation now also enables the memory-manager lifecycle hooks and request counting. When lifecycle policies are enabled, backend requests are serialized through cleanup; nested managed generation raises. Both features remain disabled by default.
* Reclaimer synchronization and ring fences propagate failure before reclamation. A failed cleanup is recorded and poisons future backend generations in the process. Scratch ring release retains an owned pool if either GPU fence, close, or disposal fails.
* SSD restoration copies bytes from an open source file to an anonymous, disk-backed temporary file while hashing, then materializes tensors from that verified private file. Verification is mandatory on this path even if a caller supplies `verify_hash=False`. This costs a temporary disk copy and write for each disk restore. This addresses pathname replacement between verification and load, not cross-process index/lease correctness or all manifest integrity issues.
* The launch preflight and benchmark launch gate now interpret the configured 9 as 9 GiB (9,663,676,416 bytes), while the existing ordinary runtime floor remains 2 x 1000 MiB. Missing launch-memory readings fail closed. The script's `available_gb` is decimal GB; its threshold option is GiB. Other manager and runtime reporting still uses legacy units and needs unification.
* An arm child failure or a reported validation gate failure now stops the orchestrator with a nonzero status. It records failures and, for exactness, mismatch details. Fixture-specific identity, per-request references, tool-argument checks and immediate per-request aborts are not yet implemented.
* Final-run power check rejects an AC reading accompanied by a discharging battery as ambiguous. A local `pmset -g batt` sample reported both `AC Power` and `discharging`; the previous failure cannot be conclusively attributed to a parser defect from a later sample. Low Power Mode was unreadable in a sandboxed sampler, so it must be checked from the execution environment.

## Results

The final FlashNext unit suite: 1,247 run, 5 skipped, 1,242 passed. The 136 benchmark/power tests also passed after the power edit. MACQWEN: 427 passed. Focused prefix suite: 44 passed, including same-size corruption and original-path replacement during load. An earlier full suite reported failures in tests that still asserted the old continue-on-failure behavior; those tests were updated and the next full run passed.

100-request small-model workload: PASS; sampled MLX active after warmup 24,923.3 KiB first, last and maximum; sampled cache maximum 2,275.7 KiB. 300 requests: PASS; active first 24,923.3 KiB, last 24,907.5 KiB, maximum 24,939.4 KiB; sampled cache maximum 2,277.9 KiB. Sampling was every ten requests after request 20, not a true peak. Pressure and pools are simulated. The workload checks its own in-process lease bookkeeping, cached same-plan output/state equality, and disk hashes; it does not prove cross-process safety or indefinite operation.

## P0 issues still open

1. Process-level reader/writer coordination, active leases, dependent chains, stale indexes and crash recovery are not fully implemented or adversarially tested. The private verified copy protects loaded bytes from path replacement, but does not make eviction and publication coordination safe across processes.
2. Some manifest fields used to reconstruct snapshots are not authenticated inside the verified safetensors payload; stale or altered structural metadata needs a safe versioned format and adversarial tests.
3. Full request lifecycle protection is not proven for all nested, concurrent, cancellation and fence-failure paths. In particular, a scratch-ring failure retained outside the auto-reclaim cleanup path requires a terminal process-level policy review.
4. F2Bm fixture identity, checkpoint/configuration provenance, plan/reference applicability, and immediate token/state/tool-argument mismatch stops are not validated. The generic post-arm failure gate does not replace these.
5. Readiness is not unified end to end: memory and power probes and runtime reporting still need fail-closed and unit-consistency review, including fresh swap-out, thermal and power observations. No readiness or approval request should be interpreted as launch authorization.

Do not run F2Bm until these issues are resolved and the required fresh preflight passes. No model weights, routing or Q4 math were changed; no commit, push, purge or full-model inference was performed.
