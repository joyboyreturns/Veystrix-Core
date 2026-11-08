# Adaptive memory manager (opt-in) and the short-prefill memory audit

Status: implemented, offline-tested, **not run on the real model**. Off by default (`FLASHNEXT_MEMORY_MANAGER=1` enables it);
`FLASHNEXT_MEMORY_SHORT_UNCACHED=1` is a second, separate opt-in. Q4 weights, routing, KV/recurrent/convolution state and prefix-cache
contents are never changed; existing guards (benchmark child, MLX limits) are untouched.

## 1. Memory lifecycle audit (from the 2026-11-07/08 real traces, 16 GB M4)

| Component | Size | When it exists | Released by |
|---|---|---|---|
| MLX model weights (dense + embeddings) | 2.8 to 3.0 GB active | load to exit | never |
| Scratch ring pool | about 1.6 GB | during a ring-eligible prefill (prompt of 512 tokens or more, first turn) | disposed at the end of the prefill (0.02 s; MLX active 4.62 to 2.98 GB) |
| Decode expert pool | +2.15 GB (2 GB pool plus overhead) | created lazily at the first decode token, then for the life of the process | nothing, before `pool_lifecycle` |
| KV, recurrent, convolution state | about 0.17 GB for a 2K prompt | the conversation | reset or replaced |
| MLX allocator cache | 1.5 to 2.3 GB inside a 2K-token prefill chunk, 0.06 to 0.08 GB between requests | prefill only | cleared per chunk and at the end (not retained across requests) |
| RAM prefix snapshot | 172 MB per 2,046-token entry (budget 0.75 GB), also on SSD (write-through) | until evicted | LRU; `shrink_ram` drops the RAM copy only |
| macOS file cache | 3.4 to 6.7 GB file-backed in the traces | buffered expert reads fill it 1:1 | the OS |
| macOS compressor | 0.6 to 5 GB | under pressure | the OS |

Phases (what changes): a cold long prefill adds the scratch ring and chunk working set (MLX peak 5.82 GB at a 2,046 chunk) and, with a
decode pool already resident, runs the ring inside it (peak 6.97 GB: the guard stop); a RAM hit adds nothing but the suffix (21 tokens,
0.36 ms restore); an SSD restore reads 172 MB in 0.07 s; decode adds the pool; cancellation leaves the pool and the descriptors as they are
(now cleaned in a `finally`); consecutive conversations kept the pool and, before the release, accumulated peak.

### Why the first short prompt caused a compression burst
Trace of the guard-stopped arm (`20261108-010847`, 512 plan, first case = a 40-token prompt):

* t = 0 to 4 s, model load: anonymous memory 2.3 to 5.4 GB (the 2.8 GB of weights), available 9.65 to 6.1 GB.
* t = 4 to 14 s, the first short prefill (no ring, because below 512 tokens): **MLX active flat at 2.9 GB**, compressor 0.9 to 4.1 GB
  (growth 3.3 GB, warning at 7.2 s), anonymous memory 5.4 to 3.9 GB, file-backed 6.7 to 5.4 GB: macOS compressing anonymous pages.
* t = 14.75 s: the decode pool is created, after the burst peaked. It then drained to 2.0 GB growth, and the steady state held at 3.1 to 3.2 GB
  available, on the guard's two thresholds.

So the burst is **not** caused by the decode pool (it did not exist yet) and not by MLX allocations (flat). It is the OS making room while the first
forward pass of a fresh process pulls in cold weights and reads experts through the buffered file cache. The same short, no-ring path in a warm
process (request B of the RAM-hit run) swelled the compressor by 4.4 GB and drained it in 4 s, again with MLX flat. Not proven: which pages
were compressed. The earlier F0/F0u investigation showed the lump with uncached reads too, so the file cache alone is not a sufficient explanation;
a load-only arm (L0) was proposed then and still separates "weights touched" from "reads".

## 2. Short-prefill read path (Phase 3), measured in isolation
Production expert reads use buffered descriptors (`store.py`; only the decode pool, the ring and the sidecar use `F_NOCACHE`). Below 512 tokens the
ring is off, so these reads go through the file cache. Microbenchmark, 768 MB of 1.3 MB random reads, no model:

| read mode | file-backed delta | anonymous delta retained | throughput |
|---|---|---|---|
| buffered `preadv`, buffer per layer | +767 MB | +186 MB | 1,633 MB/s |
| `F_NOCACHE`, buffer per layer | +154 MB | +14 MB | 1,411 MB/s |
| `F_NOCACHE`, one reused buffer | +153 MB | +1 MB | 1,379 MB/s |

Buffered reads grow the file cache by the bytes read; uncached reads do not (and are about 14% slower in this isolated test). Ranked candidates:
(1) per-request `F_NOCACHE` on the store descriptors during a short prefill (`SafeTensorStore.set_uncached_reads`, restored after the request;
results identical because the bytes and their order are unchanged), opt-in as `FLASHNEXT_MEMORY_SHORT_UNCACHED`; (2) a reusable host buffer (saves
at most about 0.2 GB retained, a change to the read path: not taken); (3) reusing the decode pool for 9 to 511 token prefills (a new execution path
for the Metal kernel: not taken without measurement); (4) temporary MLX buffers and synchronization: not measurable offline. Only (1) is implemented, as an
experiment whose effect on the real burst is unknown: the proposed run compares `F2Bm` with `F2Bmu`.

## 3. Architecture
`models/flashnext/memory_manager.py`, a pure policy (`plan`) plus a thin applier (`MemoryManager`), called from `FlashNextBackend` before every
prefill and, in a `finally`, after every request:

* **Estimator** (conservative, calibrated from real traces, recorded predicted-versus-observed in the report): ring prefill 2.06 GB + 0.46 MB per chunk
  token (matches 2.3 GB at 512 and 3.0 GB at 2,046), +1.15 GB if a pool is resident, short prefill 0.30 GB + 8 MB per token.
* **Policy**: release the resident pool before a ring-eligible fresh prefill only if keeping it leaves less than comfort (3.5 GiB) plus 0.75 GiB;
  size the next pool as 2 GB if it keeps comfort, else 1 GB if it clears the 3 GiB guard line, else refuse (`MemoryBudgetRefused`, before any
  allocation; the conversation is marked replayable); move RAM prefix snapshots to SSD when available memory is under 5 GiB or predicted decode
  headroom under comfort; never detach a pool with a request in flight or a ring leased; short prefills and follow-up turns keep the warm pool.
* **Hysteresis**: a pool grows back only after a 120 s cooldown and 1 GiB of extra headroom; a safety shrink is never delayed.
* **Applier**: `pool_lifecycle.release_decode_pool` (fence, detach, drop references, clear cache), `expert_pool.set_budget_gb` (size of the next pool),
  `PrefixCache.shrink_ram` (RAM copy only; writes out first if not on SSD; no-op without an SSD tier), `store.set_uncached_reads`.
* **After the request** (success, cancellation or error): restore buffered descriptors, clear the allocator cache above 128 MB; never raises.

Expected savings (estimates; the real run measures them): 2.0 GB reclaimed and the 2K peak 6.97 to 5.98 GB (measured) when a pool is released before the
second long prompt; 1 GB less decode footprint when the 1 GB pool is chosen under pressure, at an unmeasured decode-speed cost (hit rate with a half-size
pool is not known, so the manager prefers 2 GB and records every downsize); 0.17 GB per evicted RAM snapshot (restore from SSD 0.07 s instead of 0.4 ms);
up to the bytes read of file cache for an uncached short prefill (9 GB read for a 21-token suffix).

## 4. Tests (offline, 28 manager tests plus pool, prefix-cache and condition tests)
Policy tables (keep, release, 1 GB, refuse, hysteresis, busy, short, eviction, uncached flag), estimator against the measured points, `vm_stat` parsing against
the benchmark's definition of available, a 2,000-step randomized stress over synthetic budgets (never plans below the guard, never detaches a busy pool, pool
budgets stay in the allowed set, size changes far below the request count), a steady machine rebuilds nothing, and the sequence cold long, short cached tool call,
another long, short cached continuation, cancellation, recovery on a simulated 16 GiB machine (stays above the guard, one live pool at the end, no leak, refusal then
recovery when memory returns, eviction under pressure, uncached reads set and undone). Backend wiring: suffix-sized planning after a prefix hit, refusal handling,
`finally` cleanup, off means no change.

## 5. Remaining risks
Estimator constants are from one machine state and three prompt sizes; the short-prefill figure is the least certain. The manager reads `vm_stat` (a subprocess) once
per request. A pool released for a long prompt restarts cold (6,693 hits against 16,946 misses in the measured run). `F_NOCACHE` slows reads by about 14% in isolation;
its effect on the real burst is unknown. The 1 GB pool's decode speed is unmeasured. Concurrency is unit-tested only (the server is single-threaded today).

## 6. Proposed real-model validation (needs approval; not launched)
One fresh process, condition `F2Bm` (balanced 2,048, cache on, manager on), requests `agent:A,agent:B,long:A,agent:B`, 64 decode tokens, launch gate `--start-available-gb 9` (NOT `--min-available-gb 9`, which is the running child's stop floor and stops every arm while the weights load; default floor 2 GB), all other guards unchanged:
A cold (digest `e326048fd867d4be`), B RAM hit (digest `3349ced58d7837d1`, KV `a387f8af541ce78c`), a different long prompt (decision: release or keep, recorded with predicted
and observed headroom), then B again as a hit. Pass: B bit-identical both times, no guard, minimum available at least 3.5 GB, a manager report per request, and decode
tok/s within 10% of the earlier 2.0 tok/s unless the 1 GB pool was chosen (then report the change). Afterwards, only if approved: the same run with `F2Bmu`, and the
7-case smoke suite with the automatic pool.


## 7. First F2Bm launch (2026-11-08 10:41): stopped by an operator error, no request ran

Gate was met (11.3 GB available at start, READY, AC 80%, pressure 1) but the command carried `--min-available-gb 9`, which the
harness also hands to the child as its continuous stop floor. Loading the weights lowered available memory from 11.3 to
8.19 GB (MLX active 2.79 GB, file cache +2.7 GB) within 2 s and the child stopped in phase "prompts" before request A. No
decode, no manager decision, no digest: nothing about the manager was exercised. `--start-available-gb` is the new launch
gate (check-ready and orchestrator only); `--min-available-gb` stays the child floor (default 2 GB).
Results: `results/flashnext/20261108-104121-prefill-memory-manual/` (kept).
