# Cache and lifecycle follow-up — 2026-11-08

Status: F2Bm is **not ready**. No full-checkpoint inference, purge, commit or push was performed. This document records the changes and the remaining limitations.

## Changes

* Cache instances claim a nonblocking `flock` on `.veystrix-owner.lock` before scanning or mutating a directory and retain it for their lifetime. Standalone snapshot file helpers use the same ownership registry. The lock file is not unlinked; the OS releases the lock on process termination. Acquisition order is registry → directory ownership → instance lock → profile index lock. Competing processes receive `CacheOwnershipError` rather than scanning or evicting. Actual subprocess tests cover contention, release and process termination. Multiple instances in the **same** process share the lock and retain separate indexes; their concurrent mutation has not been proved safe.
* Snapshot format 3 binds fingerprint, snapshot and parent identity, token counts and digest, chunk plan, layer/position metadata, and tensor names, shapes and dtypes into metadata inside the checksummed safetensors payload. The private disk-backed copy remains in use. Changed manifest fields, payload corruption and replacement during load are tested. This detects accidental corruption and inconsistent metadata; it is not cryptographic authentication against a party able to rewrite both payload and checksums. Format-2 entries are not loaded; incompatible entries are preserved rather than removed. Profile eviction no longer deletes foreign-fingerprint entries. Scanning only deletes recognized cache temporary names, not arbitrary files.
* Scratch-ring, pool-release and direct ring-fence synchronization errors now poison the process lifecycle, retain the pool and propagate the error. The `make_room` path no longer converts a synchronization failure into a successful continuation; post-request cleanup avoids releasing further resources after a lifecycle fault. Tests exercise the injected failure and retention paths, but do not prove every GPU failure or cancellation stage.
* The already established 9 GiB launch threshold and separate 2 x 1000 MiB runtime floor were not lowered. Full unit unification and fail-closed monitoring of all power, thermal, memory and swap readings remain open. A previous machine sample reported both AC and a discharging battery; final mode treats it as ambiguous.

## Evidence and limits

* Real separate-process cache contention and termination tests passed as part of the prefix tests. A 90-test focused prefix/profile run passed after the ownership and manifest changes; a 47-test profile/endurance run passed after preserving foreign entries.
* The final full FlashNext suite ran 1,251 tests, with 5 skipped and 1,246 passing. VEYSTRIX: 427 passed.
* The 100-request small-model workload passed: sampled active memory after warmup 24,891.3 KiB first and 24,939.3 KiB last/max; sampled cache max 2,275.8 KiB. The 300-request workload passed: sampled active first 24,891.3 KiB, last/max 24,955.4 KiB; sampled cache max 2,277.9 KiB. Samples are every ten requests after warmup, not continuous peaks; pressure and pools are simulated. These results are not full-model stability evidence.
* One warm-cache 32 MiB file microbenchmark took 13.3 ms for a hash-only read and 16.7 ms for hash, private-copy write and reread. The copy adds one logical write and one logical reread per snapshot byte on top of hashing the source; this is not a physical SSD/NVMe traffic measurement and is not a full 172 MiB restore benchmark.

## Remaining blockers

1. F2Bm fixture identity, checkpoint and tokenizer provenance, execution flags and chunk plan have not been tied to the published reference digests. The harness does not yet enforce immediate per-request token/state/tool-argument comparisons, so the current command is not authorized for real-model execution.
2. Same-process cache instances share directory ownership without a shared mutable index lock around every operation. Cross-process locks are advisory and assume cooperating processes; external modifications that ignore locks are not prevented. Index crash recovery and all writer/reader/eviction races require further fault-injection validation.
3. Complete GPU-failure and cancellation-stage coverage, real 1/2 GB pool transitions, post-failure backend state, and full end-to-end safety have not been proved.
4. Launch/runtime memory units, fail-closed missing probes, no-active-swap verification and safe power/thermal status still need end-to-end validation. No fresh full-model launch preflight was performed because these P0 conditions are unresolved.
