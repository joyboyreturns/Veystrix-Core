# Early compression burst in the 1K F2 prefill: offline investigation

Branch `development3.8`. Offline only: no model was loaded and no inference ran for this note. Evidence comes
from the two completed 1K F2 runs (`results/flashnext/20261107-133433-*` and `-134606-*`, both battery,
PRELIMINARY) and from reading the code. **Evidence** and **hypothesis** are labelled. No production file changed.

## 1. What the traces show (evidence)

| Item | Run 1 | Run 2 |
|---|---|---|
| Resident weights after load (MLX active) | 2.79 GB | 2.79 GB |
| Process RSS at the end of load | 3.17 GB | 1.92 GB |
| Compressor growth at the first prefill sample (t=2.05 s) | +69 MiB | +1,219 MiB |
| Peak compressor growth | +2,579 MiB (27.2 s) | +3,186 MiB (3.9 s) |
| RSS from 4 s on, minimum | 1.53 GB | 1.52 GB |
| Pages decompressed during prefill | 184k (3.0 GB) | 177k (2.9 GB) |
| Pages compressed during prefill (gross) | 410k (6.7 GB) | 338k (5.5 GB) |

* Weights are materialised at load, not in prefill: `loader.py:240` calls `mx.eval(model.parameters())` and
  `after_load.active` is already 2.79 GB.
* Prefill adds three things to MLX active memory: the 1.62 GB scratch pool (an anonymous `mmap`, wrapped
  zero-copy), about 1.0 GB of retained KV and state (active ends prefill at 3.79 GB with the pool freed), and a
  per-layer transient of up to about 0.9 GB. The MLX allocator cache holds a steady 1.31 GB through prefill
  (peak 1.35 GB).
* The last 0.45 GB step (5.50 to 5.96 GB at 26.9 s) matches full-sequence logits: 1,025 x 248,320 x 2 B =
  0.51 GB. `prefill.py` keeps them on purpose, because last-row projection changed logits by up to 0.125.
* **The compressed pages are largely this process's own.** RSS excludes compressed pages. After 4 s the process
  holds only 1.5 GB resident while MLX active is 4.4 to 5.5 GB, so about 3 GB of its own pages are not resident.
  That matches the 2.1 to 3.2 GiB compressor lump. About 3 GB was then decompressed during prefill, roughly the
  weight size.
* Run 2 started with 1 GiB less available memory and had already lost about 0.9 GB of weights (RSS 1.92 GB
  against 2.79 GB) when load finished. Compression is therefore driven by total demand, not by one prefill step.
* The pre-load footprint of other processes is large (anonymous 4.5 to 5.4 GB, file cache 3.5 to 4.3 GB at
  arm start), so the margin depends on what else is running.

## 2. Hypotheses, ranked by how well the evidence supports them

1. **Cold dense weights are compressed, then decompressed layer by layer.** Supported by RSS, the decompress
   total and the timing. Not proven. The traces do not say which pages (weights, ring or scratch pages, MLX
   cache) were compressed.
2. **MLX allocator cache (1.3 GB resident, not reclaimable except by compression) adds to the pressure.**
   Supported by the steady 1.31 GB. The size of its effect on the lump is unknown.
3. **`os.pread` allocation churn: not supported for F2.** In the F2 prefill every layer goes through the ring
   (0 fallbacks, `preadv` into a preallocated pool), so no `bytes` objects are created. The lump still
   appears. F0u also showed it. Churn can add transient memory on the F0 path and on fallbacks, and does not
   explain the lump.
4. **Scratch-pool size: rejected.** F2s (256 experts) was worse and fell back to buffered reads.

Open discrepancy: the loader's own rules put 3.18 GB in the resident set (2.388 dense, 0.397 embedding,
0.397 `lm_head`) but MLX reports 2.79 GB after load. One 0.397 GB table is unaccounted for. A load-only run
would show which.

## 3. Ranked proposals

| # | Change | Expected saving | Cost | Correctness risk | Needs |
|---|---|---|---|---|---|
| 1 | **Load-only diagnostic** (arm L0: load, idle 5 s, trace at 0.25 s; optional L1: touch every weight once) | none directly; shows whether compression starts at load or in prefill and which tables are resident | one model load, no inference | none | benchmark-only code, then one approved real load |
| 2 | **512 MB allocator cache cap** (`FLASHNEXT_PREFILL_CACHE_LIMIT_MB`, conditions F3a/F4, already implemented, offline-tested only) | up to about 0.8 GB of resident cache (1.31 GB to 0.5 GB) | more fresh allocations, unmeasured | none to values: the allocator only changes buffer reuse | one approved F3a 1K arm, then digest and KV-hash comparison with `2f84f7d220e782b8` / `fd83510bd53f9c82` |
| 3 | **`FLASHNEXT_STREAM_EMBED=1`** (existing opt-in, valid because `native_mtp` defaults to off) | up to 0.4 GB of mostly cold anonymous memory | up to 1,025 row reads in prefill, one per decode token | low: same `mx.dequantize` path, not yet checked on the real model | one approved arm. It stays off in production. |
| 4 | **Record experts per layer in `PrefillRing.stats`** (benchmark-visible counter only) | none directly. If layers need at most about 400 experts at 1K, a 400-slot ring saves about 0.3 GB | none | a smaller ring risks `fallback_capacity` (F2s showed this), so decide only after seeing the data | offline counter, then data from the next approved run |
| 5 | `FLASHNEXT_READ=preadv` on the non-ring path | removes transient `bytes` objects and one copy on F0 and fallbacks | none expected | none (same bytes), but it changes a production read path, so opt-in only | not for F2 |

Combined ceiling of 2 + 3 + 4: about 1.5 GB less resident memory. That is a ceiling, not an expectation, and
it will not remove the weight compression if hypothesis 1 holds.

## 4. Not recommended

* **MLX wired memory:** not approved. It could stop macOS from reclaiming pages.
* **Chunked prefill or last-row logits:** different matmul and reduction shapes are not bit-exact (the 0.125
  logit difference is recorded in `prefill.py`).
* **A decode pool above 2 GB:** the margin above the 3 GiB stop floor is about 0.4 GB in the second run.
* **Any change to Q4, routing or the default production environment.**

## 5. Proposed order, pending approval

1. Implement the L0 arm and the per-layer counter (benchmark-only), with unit tests.
2. One real L0 load (no inference), then decide between 2 and 3 from its result.
3. One F3a 1K arm, then one STREAM_EMBED arm, each compared on digest and KV hash, then memory.

## Sequential requests: what stays resident, and a per-request cleanup proposal (2026-11-08, offline)

Source: trace of `results/flashnext/20261108-003402-prefill-memory-manual/` (request A, then request B in one process, 2K
balanced planner, expert pool 2 GB).

Measured, not assumed:

| moment | MLX active | MLX cache | note |
|---|---|---|---|
| request A, end of prefill | 2.98 GB | 0.06 GB | scratch ring pool already disposed |
| request A, decode token 1 | 5.13 GB | 0.07 GB | the 2 GB decode pool is created lazily here |
| request A, end of decode | 5.13 GB | 0.08 GB | what stays resident between requests |
| request A, mid prefill chunk | 4.9 to 5.6 GB | 1.5 to 2.3 GB | per-chunk working set, released at the end of the prefill |
| request B, mid prefill | 5.2 to 7.0 GB | 0.7 to 2.1 GB | ring now inside the resident decode pool |

1. The premise that about 2.1 GB of allocator cache is retained across requests is not what the trace shows: the cache is
   0.08 GB after request A. The 1.6 to 2.3 GB value is the in-prefill chunk working set (request A had it too).
2. What is retained is the 2 GB decode expert pool (`store._flashnext_expert_pool`, never freed; `for_store` creates it on
   first use). Its presence changes the next prefill: `prefill_ring.acquire` takes the "later turn" branch (no scratch pool;
   ring inside the decode pool), the validated first-turn path is not used, and the peak rose from 5.82 GB (A) to 6.97 GB (B).
3. `FlashNextBackend.reset()` only swaps the KV cache and clears positions. It frees neither pool nor allocator cache, and
   it should not: it runs for every request, including short ones where a warm pool is worth keeping.
4. The decode pool is reached only through `expert_pool.for_store` at decode time, so dropping it is safe once the GPU is idle
   and no prefill ring is leased; the next decode recreates it. Arrays that still view it keep the memory valid.
5. Prefix snapshots are in `PrefixCache` (RAM and SSD) and are independent of both pools.

Options considered:

| Option | Effect | Verdict |
|---|---|---|
| `mx.clear_cache()` after every request | frees about 0.08 GB | cheap, but does not address the problem |
| Cap the allocator cache (`prefill_memory` cache limit) | trims in-prefill working set (already a research option) | separate, helps peak not residency |
| Free the decode pool at `reset()` | frees 2 GB | too broad: hurts short and follow-up requests |
| **Free the decode pool before a fresh ring-eligible prefill** (opt-in) | request B takes request A's path and peak; the pool is recreated at B's first decode token | **smallest correct change** |
| Never create the decode pool until needed | already true (lazy) | no change needed |
| Release on cancellation | nothing to add: a cancelled prefill goes through `prefill_ring.release`; a cancelled decode leaves the pool, and the next fresh prefill drops it | covered |

Proposal (`models/flashnext/pool_lifecycle.py`, NOT wired in): `FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=1`; in
`generate()`, on the fresh-conversation path, call `make_room(language, tokens)` before the prefill. It releases only if
the flag is on, a decode pool exists, the ring is configured and scratch-capable, and the prompt reaches the ring minimum.
Order: refuse while a ring is leased, `mx.synchronize()`, detach from the store, drop references, `gc.collect()`,
`mx.clear_cache()`. Cost: the new decode pool starts cold (request A's decode was also cold: 4,963 hits against 13,222
misses). Prefix snapshots are preserved; nothing in `PrefixCache` is touched.

Validated offline (13 unit tests, no model): policy truth table, off by default, fence before detach and cache clear
after, a leased ring blocks the release, a failing fence leaves the pool attached and usable, idempotent, a surviving
array view stays valid, `for_store` recreates a cold pool, nothing in inference imports the module, the prefix-cache
module is not imported. Not yet validated on the real model: that request B's peak returns to the request A profile.
That needs one approved real-model arm (A then B with the flag on).

## Decode-pool release: implemented (opt-in) and validated on the real model (2026-11-08, AC 76%, PRELIMINARY)

`FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=1` (off by default; registered research setting). `FlashNextBackend.generate` is now a thin
wrapper that counts the call while the flag is on (`pool_lifecycle.in_flight`) and runs `_generate`, whose signature is unchanged.
After `check_cancel()` and before any prefill path, a fresh conversation (empty tape or replay, empty cache, no draft decoder,
decided before the tape changes) calls `pool_lifecycle.make_room`. It releases only if the flag is on, a decode pool exists, the prompt
is ring-eligible with the scratch option, no other request is in flight and no ring is leased; then it fences the GPU, detaches the pool,
drops every reference, collects and clears the allocator cache. Any error leaves the pool attached and is reported in
`backend.last_pool_release`. Prefix-cache snapshots are not touched. Offline: 25 pool tests (all branches, concurrency, cancellation-order,
repeat, wiring order, registry), unit suite 1102 OK, macqwen suite 427 OK.

Real run: one fresh process, `F2Bnr` (balanced 2,048 plan, restore off, eager linear state, 512-expert scratch ring, 2 GB decode pool, release
flag on), request A then request B (full cold prefill, plan [2046, 21]), `results/flashnext/20261108-005555-prefill-memory-manual/`. Started at 9.8 GB
free or inactive, pressure normal, no swapping. No guard fired.

| | request A | request B (flag on) | request B alone, fresh process (F2Bn) | B after A, flag off (guard stop) |
|---|---|---|---|---|
| decode pool before prefill | none | existed, **released 1,999,880,192 B in 0.026 s** (MLX active 4.95 to 2.95 GB) | none | existed, kept |
| ring | scratch, acquired | scratch, acquired (96 stagings) | scratch | in-pool |
| MLX peak | 5.82 GB | 5.98 GB | 5.82 GB | 6.97 GB (stopped) |
| compressor growth | 1.69 GB (from arm start) | 0.63 GB | 3.45 GB | 6.02 GiB |
| min available | 3.72 GB | 4.22 GB | 3.18 GB | 2.65 GiB |
| swap | none | none | none | none |
| prefill | 33.6 s, 61.5 tok/s | 32.2 s, 64.2 tok/s | 33.8 s | |
| decode | 1.97 tok/s | 2.08 tok/s | | |
| decode pool after | created at first token | recreated at first token (cold: 6,693 hits, 16,946 misses; A: 4,963 and 13,222) | | |
| digest / KV hash | `e326048fd867d4be` / `555f71e4ff34a9b5` | `3349ced58d7837d1` / `a387f8af541ce78c` | `3349ced58d7837d1` / `a387f8af541ce78c` | |

Bit-exactness: request B equals the fresh-process `F2Bn` reference in all 64 generated token ids, digest, KV hash and 124/124 state tensors
(worst difference 0); request A equals the earlier cold A in all 124 tensors. The release changes memory, not results.
