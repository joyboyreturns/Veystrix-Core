# Independent audit — 2026-11-08

This is a partial audit, not approval for a real-model run. Existing uncommitted changes predate this audit except for the edits described below. No model inference on the 113 GB checkpoint was launched.

## Changes and verification

* Removed the per-process `PrefixCache._verified_keys` shortcut. A disk load with hash verification enabled now hashes the file on each load. A regression test verifies a same-size modification after an earlier successful restore is a miss. This does not eliminate a concurrent replacement race between hashing and `mx.load`, nor establish cross-process lease safety.
* The memory preflight no longer invokes privileged purge on low memory unless `--force-purge` is explicitly passed. Its `available_gb` now means decimal GB; its legacy threshold is preserved as units of 1000 MiB (9 means 9.437184 decimal GB), including both initial and final checks. This corrects a unit label without lowering the old effective threshold. The runtime harness still reports MB in MiB and applies its existing thresholds; conversion and rounding differences remain to be unified.
* FlashNext unit suite: 1,241 run, 5 skipped, 1,236 passed. VEYSTRIX unit suite: 427 passed. Prefix cache suite: 43 passed. Focused preflight suite: 8 passed. An earlier FlashNext run had three failures because it began before the preflight tests were updated; the rerun passed.

## Memory

The 9 unit launch gate and the running child's 2 unit ordinary floor are distinct; the child's conditional compressor guard remains unchanged. `vm_stat` free, inactive and speculative are used as an estimate, not proof that all inactive pages are immediately available. Purgeable, file-backed, anonymous and active counts overlap other categories and cannot simply be added. Apple XNU's Mach `free_count` includes speculative pages; verify the `vm_stat` presentation separately when changing the calculation. A file-backed page count is system-wide and is not a measurement of Veystrix-owned RAM. It can represent normal reclaimable filesystem cache.

One preflight sample showed 6.768 decimal GB estimated available, 5,480 MiB file-backed, 251 MiB purgeable, 753 MiB compressor, 2,274 MiB wired and 7,186 MiB anonymous. Those are overlapping categories. It was blocked against the preserved 9.437184 decimal GB effective launch gate; no purge was attempted. The readiness check independently reported checkpoint metadata READY but overall BLOCKED, reporting the final-run AC condition failed. Pressure, swap-out rate, thermal and competing inference need fresh verification before any launch; a cumulative swapout count is not an activity rate.

## Lifecycle and performance evidence

The existing small-model benchmark reported PASS at 100 requests (1,964.7 KiB first-to-last MLX active growth) and 300 requests (1,932.7 KiB). At 300 it reported 3 RAM hits, 264 SSD hits, 31 simulated pool releases, 30 simulated rebuilds and 30 simulated cancellations. It does not measure post-warmup slope, true 1/2 GB pool transitions, process restart, complete fault injection, or long-session stability; its pool and pressure are simulated.

The existing 50-trial microbenchmark, executed once per suffix length, gave the following isolated baseline/optimised timing ratios: 1: 1.36x; 8: 1.39x; 9: 1.11x; 16: 1.75x; 21: 1.28x; 32: 1.26x; 64: 1.23x; 128: 1.19x. At 21 tokens it counted 2,457,600,000 vs 614,400,000 file-read bytes (75% reduction). The test checks the final output for bit equality, uses fixed baseline-first ordering and a tiny generated checkpoint, and does not measure physical NVMe traffic or end-to-end TTFT. Treat timing as exploratory, not a controlled full-model speed claim.

## Known unresolved risks

* `FLASHNEXT_AUTO_MEMORY_RECLAIM=1` alone does not activate backend cleanup: `_after_request` returns unless `FLASHNEXT_MEMORY_MANAGER=1` is also set. The reclaimer catches synchronization failures and continues to cleanup; it also takes actions before proving another request cannot race with cleanup. These paths need correction and concurrency/cancellation tests before the acceptance criteria can be claimed.
* The SSD checksum fix has a time-of-check/time-of-use window; basic `PrefixCache` has in-process leases, whereas profile locking needs a complete cross-process reader/eviction audit. Pin limits, exact fingerprint coverage and safe corruption misses need further adversarial testing.
* Preflight and runtime memory calculations, swap-out detection, fail-closed pressure/power probes, fixture identity, and immediate abort on output/state mismatch need further work. The readiness result above is not launch authorization.
* Full-model reference digests and KV hashes were not independently matched to the exact fixture. Do not claim cache exactness or throughput for the real model from these small tests.

## Candidate F2Bm command — not yet launch-ready

After the above gaps are resolved, a fresh safe preflight passes, fixture identity is verified, and the user approves, the candidate is:

```sh
FLASHNEXT_AUTO_MEMORY_RECLAIM=1 .venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory \
  --checkpoint "$HOME/models/Qwen3.8-Flash-Next-MLX-4bit-MTP" \
  --sizes 2k --conditions F2Bm --rounds 1 --decode 64 \
  --requests agent:A,agent:B,long:A,agent:B \
  --start-available-gb 9 --min-available-gb 2 --expert-pool-gb 2 --final
```

F2Bm selects balanced chunking at 2048, prefix cache and memory manager; the ring and eager linear state are selected by the condition overlay. Validate all overlays and inherited environment again before use. The command is a candidate, not proof that the harness currently stops at the first reference mismatch.
