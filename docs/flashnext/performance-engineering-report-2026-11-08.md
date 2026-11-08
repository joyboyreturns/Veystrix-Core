# Performance engineering report, 2026-11-08 (offline pass)

Author: performance engineer role. Scope: prefix caching, short and long prefill, decode, memory. **No real-model run was
made** (owner approval required; machine on battery at 59%, so anything timed here is preliminary). No tracked or
Codex-modified file was edited, nothing was committed. Everything below is either an existing recorded result (path
given), a synthetic measurement (labelled), or an offline replay of a recorded trace (labelled).

## Summary

| Area | Finding | Evidence class | Action |
|---|---|---|---|
| Prefix restore | RAM hit 0.5 ms; 2K SSD hit 0.07-0.11 s of a 4.6 s cached TTFT (about 2%). Not the bottleneck. | real (Stage C, `20261107-233437`, `20261108-004230`) + synthetic | none; do not weaken the verified-copy read for this |
| Lookup hashing | `chain_key` is 0.13 ms at 2K tokens, 0.61 ms at 10K (one probe per stored length). Pure-Python `int.to_bytes` loop; `array` gives identical bytes in 0.016/0.084 ms. | synthetic microbench | not worth a patch (under 0.1% of TTFT); noted |
| Chained restore, 10K | SSD restore of a 5-snapshot chain reads 856 MB in 1.6 s; about 450 MB of it is the ancestors' GatedDeltaNet state, which `restore()` never uses (only the leaf's recurrent state is used; ancestors contribute KV deltas). | synthetic, shapes of the real model | design note below; needs a format change, so Codex's call |
| Short-prefill pool reuse | The synthetic benchmark's 75% hit rate is its own setting (75% warm residency), not a measurement. Replaying a recorded session: a new request's prefill finds only **1.6-4.8%** of its experts in a 2 GB pool left by the previous decode. | offline replay of recorded trace | expect almost no SSD saving from `FLASHNEXT_SHORT_PREFILL_POOL_REUSE=1` on the real model; keep opt-in and off |
| Decode replacement policy | At 2 GB, production LRU is 46.6% vs ARC 48.9%, BIP 40.4%; offline optimum 67.9%. No cheap online policy beats LRU by enough to matter. | offline replay | no change to `ExpertPool` |
| 1 GB pool | LRU hits **0%** at 1 GB (per-token working set larger than the pool, cyclic scan); BIP reaches 28.8%, ARC 28.7%. A 1 GB pool under LRU costs 1 GB and saves nothing. | offline replay | tell the memory manager (below) |
| "2 GB is optimal" | The evidence doc (12.9) compares `arm-1-defaults` and `arm-2-pool6`. `arm-1-defaults` has `FLASHNEXT_EXPERT_POOL_GB=0`, so that comparison is **no pool vs 6 GB** (3.42-3.47 vs 3.79-3.93 tok/s, +10-13%). No recorded decode run used a 2 GB pool. | recorded results, env fields | correct the claim; measure a 2 GB decode arm if approved |

## Details

### Pool geometry
One expert record is 3,211,264 bytes, so a 2 GB pool holds 622 records, which is 2.5% of the 24,576 (48 layers x 512
experts) in the model, about 13 per layer. A 21-token suffix touches roughly 60 or more distinct experts per layer, so
even a perfectly aligned pool could cover at most about a fifth of a suffix's experts. The replay agrees:

| Pool | decode hit (LRU) | prefill-resident after previous decode (LRU) |
|---|---|---|
| 1 GB (311 slots) | 0.0% | 0.0-2.4% |
| 2 GB (622 slots) | 44-48% | 0.0%, 1.6%, 4.8% (turns 0, 1, 2) |
| 4 GB (1,245 slots) | 62-65% | 0.0%, 4.2%, 7.3% |

Replay limits, stated plainly: one session of three 256-token turns (`results/flashnext/20260922-221213-route-trace-manual`),
recorded on an older run (about 8 routed experts per token after pinned experts, 46/46/35-token prompts, unrelated topics).
An agent loop that continues one conversation may overlap more than unrelated topics do, and the current checkpoint
routes 10 experts per token. The 75% in the synthetic benchmark is therefore unsupported, and the true value is
unmeasured, not zero. Reproduce with the new
`python -m models.flashnext.tests.bench.bench_pool_policy_trace TRACE.json.gz --gb 1 2 4`.

### What bounds the real numbers
* Cached suffix, 21 tokens: 9.3 GB of expert reads in 4.4 s is 2.1 GB/s. The recorded isolated ceiling on the real read
  pattern is about 3.0-3.1 GB/s, so the floor is about 3.0 s, at most roughly 30% below today.
* Cold 2K balanced prefill: 58.8 GB in 33.9 s is 1.7 GB/s against the same ceiling, so overlap of reads and GPU work is the
  only large lever left (a floor near 19 s if reads were perfectly hidden behind compute, which they cannot be, because
  layer L's routing needs layer L-1's output). Neither number can be improved or even attributed offline.
* Routing, Q4 and architecture are out of scope, so fewer experts per token is not an option; fewer **bytes read** only
  helps if residency rises, and the pool is 2.5% of the experts.

### Memory-manager note (for Codex)
`memory_manager.py` may shrink the pool from 2 GB to 1 GB under pressure. With LRU and this routing, 1 GB hits about
nothing, so the shrink saves 1 GB of RAM at no loss, and the choice that matters is 2 GB or 0, not 2 GB or 1 GB. If a
1 GB pool is wanted, BIP (insert new records at the cold end, promote on hit, every 8th insert at MRU) is a 4-line change
in `ExpertPool.assign` and measured 28.8% here, but it is worse than LRU at 2 GB and above (40.4% vs 46.6%), so it would
have to be selected by pool size. Not implemented: unverified on the real checkpoint and `expert_pool.py` is modified by Codex.

### Chained SSD restore (4K-10K)
`restore()` takes recurrent and convolution state from the leaf only and KV deltas from every link, yet `read_entry`
loads every ancestor's full file (about 113 MB of recurrent state each) and `_load_entry` admits it to the RAM tier. For
a 10K prompt with five snapshots that is roughly 450 of 856 MB. The hash covers the whole file, so the bytes cannot be
skipped without a layout change (state in its own checksummed file). It costs about 0.8 s per cold-RAM 10K restore against
minutes of prefill it replaces, so it is a memory-efficiency note, not a speed priority.

## Proposed real-model measurements (each needs your approval, 9 GB gate, plugged in)
1. Decode, 128 tokens, pool 2 GB vs 0 vs 6 GB, same prompt, blocked order: closes the "2 GB is optimal" gap and gives the
   real LRU hit rate (`expert_io_stats`) with the 10-expert checkpoint.
2. Cached 21-token suffix with `FLASHNEXT_EXPERT_IO_DIAGNOSTICS=1` and pool reuse on and off, for real hit rate, copy
   cost and TTFT; the replay predicts the on/off difference is within noise.
3. Raw read-pattern throughput of the 21-token suffix (per-layer unique experts, `F_NOCACHE`) to find out whether 2.1 GB/s
   is below what this pattern can do. Read-only I/O, no model load.

## Verification
* The new tool's decode numbers reproduce the scratch simulation exactly (LRU 0.0/46.6/63.6%, ARC 28.7/48.9/63.6%, BIP 28.8/40.4/52.1%,
  Belady 53.9/67.9/78.3% at 1/2/4 GB). Its single prefill-resident figure pools all turns, including turn 0 against an empty pool
  (LRU 1.1/2.1/3.8%); the per-turn split in the table above is from the scratch run.
* Unit tests: 7 pass. A first run failed one test (my scan case had 10 experts in one event, which the batch-protection rule
  turns into 78% hits); the test was wrong, not the tool, and was changed to a one-expert-per-event cycle.
* A Haiku agent also ran the 165 existing tests that use `output_path` (OK). The full flashnext and macqwen suites were not re-run,
  because no production file changed.

## Deliverables in this pass
* `models/flashnext/tests/bench/bench_pool_policy_trace.py` and `models/flashnext/tests/unit/test_bench_pool_policy_trace.py`
  (new files; no `--json` option).
* This report. No production code changed, so there is no before/after timing to report.
