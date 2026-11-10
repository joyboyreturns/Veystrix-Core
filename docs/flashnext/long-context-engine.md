# Long-context engine: chunked prefill and exact prefix caching (opt-in, uncommitted)

Branch `development3.8`. Everything below is off by default. Tags: **real** (checkpoint, this machine, battery,
PRELIMINARY), **synthetic** (random weights or random tensors of the real shapes), **estimate**. A 10K prompt has
**not** been run on the real model.

## 1. Validated on the real model: the linear-state fix

One 1K F2 arm with `FLASHNEXT_EVAL_LINEAR_STATE=1` (condition `F2L`, battery 80%, all guards on): **real**

| | Reference F2 | F2L |
|---|---|---|
| Output digest / KV hash / first token | `2f84f7d220e782b8` / `fd83510bd53f9c82` / 760 | identical |
| Tokens (prefilled / generated), finish | 1,025 / 77, stop | identical |
| Retained above weights at end of prefill | 0.992 GB | **0.236 GB** (-0.756 GB = 36 layers x 21 MB) |
| MLX peak | 6.34 GB | **5.58 GB** |
| Prefill | 40.5 tok/s | 40.2 tok/s (single runs, noise) |
| Decode | 2.09 tok/s | 2.11 tok/s |
| Ring fallbacks, swap-outs, swap-ins | 0, 0, 12 pages | 0, 0, 0 |
| Min available / max compressor growth | 3,450 MiB / +2,579 MiB | 3,470 MiB / +3,178 MiB |

The mechanism (lazy GatedDeltaNet conv state keeps each layer's prefill input alive) is confirmed. The early
compression burst is not reduced by it: growth is in the range seen between earlier runs. Digest and KV hash are
bit-identical, as expected from a change that only moves when an evaluation runs.

## 2. What was built

| Piece | Where | Flag |
|---|---|---|
| Eager conv-state evaluation | `linear_state.py` | `FLASHNEXT_EVAL_LINEAR_STATE=1` |
| Chunked prefill with absolute chunk edges, boundary callbacks, per-chunk state evaluation | `prefill.py` | `FLASHNEXT_PREFILL_CHUNK_TOKENS=N` (0 = off, minimum 64) |
| Exact two-tier prefix cache (RAM LRU, SSD, chained KV deltas) | `prefix_cache.py` | constructed by the session below |
| Prefill glue: longest-prefix restore, boundary snapshots, report | `prefix_session.py`, hook in `veystrix/backends/flashnext.py` | `FLASHNEXT_PREFIX_CACHE=1`, `_DIR`, `_RAM_GB` (0.75), `_DISK_GB` (20), `_MIN_TOKENS` (512), `_BOUNDARIES` (`system` or `chunks`) |
| Memory projection | `tests/bench/long_context_budget.py` | |
| Synthetic restore benchmark | `tests/bench/bench_prefix_cache_synthetic.py` | |
| Harness condition `F2L` | `prefill_memory_harness.py` | |

Tests added: 11 (linear state), 32 (prefix cache), 17 (chunked prefill), 16 (prefix session), 11 (budget), 4 (F2L
condition). Full flashnext suite 968 tests OK (5 skipped); veystrix 427 OK. Mutation checks were run on each module and
every surviving mutant got a test.

### Chunked prefill: what is and is not exact
* The stored cache holds the whole prompt afterwards; each chunk evaluates all cache states and clears the allocator
  cache, so temporaries and retained inputs do not accumulate across chunks.
* **Not bit-identical to one call.** On the small random model, chunked and single-call last-position logits differ
  by about 5e-7 (fp32), every split tried, including `[299, 1]`. Kernel shapes depend on the row count. So the
  single call stays the reference, and chunking needs a digest check on the real model before it is trusted. Not
  assumed exact.
* The last chunk decides full logits versus the fused head from the **whole** prompt length, exactly as the single
  call would. Above 2,048 tokens no vocabulary-wide logits are built at all.
* Chunk edges are absolute multiples of the chunk size, so a run restored at position N splits the suffix exactly like a
  cold run that passed N. That is what makes cached and cold runs bit-identical (tested).
* `want_logits` and `want_hidden` callers (MTP, speculative) are never chunked.
* Known gap: the prefill progress callback counts layers once, so UI progress stalls at the first chunk's share.

## 3. Prefix cache

**What a reusable prefix is for this model.** Not KV alone. After N tokens the state is: 12 QSA layers (`keys`, `values`,
`index_keys`, `index_position_ids`, sliceable by token), 36 GatedDeltaNet layers (fp32 recurrent state plus conv
state, plus two extra slots on the PLE layer; **not sliceable, exists only at N**), the token tape, and the
language model's `_position_ids` / `_rope_deltas`. The last is easy to miss: **a fresh engine has neither, and its
first call on a restored cache numbers the suffix from 0** (tested: positions 0..71 instead of 128..199). They are stored
with every entry.

**Design.**
* Snapshots exist only at boundaries the prefill is stopped at: the end of the first system message
  (`<|im_end|>\n`), optionally every chunk edge (`_BOUNDARIES=chunks`). A prefix cannot be reused anywhere else.
* Entries are chained: KV is stored as a delta over the parent, so a 10K snapshot extending an 8K one stores 2K of KV; the
  recurrent state is a full copy per entry (113 MB for 36 layers).
* Key = SHA-256 chain over a fingerprint and the exact token ids; the tokens are compared again on load (collision
  tested). The fingerprint is the existing session compatibility record (checkpoint, tokenizer and template, engine
  files, routing profile) plus the chunk size, ring and QSA chunk settings, because those change numerics.
* A prefix is used only if strictly shorter than the prompt (one token must remain to prefill). Any mismatch is a miss,
  never an approximation.
* RAM tier: byte budget, LRU. Under pressure a parent may leave RAM once written to SSD; without an SSD tier it is pinned
  while a child is resident.
* SSD tier: safetensors per entry plus a manifest committed last, written to a temporary name, fsynced, renamed; mode
  0600 in a 0700 directory; SHA-256 of the file checked on load; unchanged entries are never rewritten; budget evicts
  leaves first by access time; leftover temporaries, incompatible fingerprints and truncated or orphaned entries are
  removed at start. A corrupt entry removes its descendants and the lookup falls back to the next shorter prefix.
* Values are stored as they are in the live cache (bf16 and fp32): no lossy codec. Restored arrays are new objects,
  so later generation cannot change a stored snapshot (tested by changing the live cache in place).
* Only for fresh conversations on the standard target decoder. Native MTP and the fused decoder keep their own caches
  and skip it. The server's existing strict-prefix reuse (`_adopt`) still handles a conversation that simply grows.

**Synthetic restore cost** (real shapes, random values, page cache warm after the write; this is restoration, not prefill): **synthetic**

| Prefix | Edges | SSD size | Capture | SSD write | RAM restore | SSD restore with SHA-256 | SSD restore, no hash |
|---|---|---|---|---|---|---|---|
| 2,048 | 1 | 172 MB | 0.000 s | 0.09 s | 0.000 s | 0.07 s | 0.01 s |
| 4,096 | 2 | 345 MB | 0.001 s | 0.19 s | 0.006 s | 0.20 s | 0.01 s |
| 10,000 | 5 | 856 MB | 0.004 s | 0.48 s | 0.010 s | 0.94 s | 0.03 s |

The hash dominates (about 0.9 GB/s). A cold read from the SSD adds roughly 0.3 s per 856 MB at 3 GB/s (estimate, not
measured: the page cache cannot be dropped without sudo). A system-boundary snapshot alone at 10K is about 0.4 GB.
Either way, restoring costs about a second, against minutes of prefill.

## 4. Cold 10K prefill projection (estimate)

Peak MLX memory with the fix (`long_context_budget.py`, calibrated at 1K only, conservative there):

| Tokens | Single call | Chunk 2,048 | Chunk 1,024 |
|---|---|---|---|
| 4,096 | 7.85 GB | 7.12 GB | 6.56 GB |
| 8,192 | 9.81 GB | 7.64 GB | 6.87 GB |
| 10,000 | 10.68 GB | 7.87 GB | 7.01 GB |

(Guards stop at 12.21 GB. Without the fix: 16.6 GB at 8K and 18.9 GB at 10K single-call.)

Time: at 1K a prefill is read-bound (25.3 s: 16.6 s of read waits, about 8.7 s other). At 10K every layer routes all
512 experts, about 75 GB per pass. Single call: about 25 to 30 s of reads plus about 90 s of compute, so about 2 to 2.5
minutes (roughly 65 to 85 tok/s). Chunks of 2,048 re-read about 60 GB per chunk (about 300 GB, about 95 s at 3.2 GB/s),
so about 3 to 3.5 minutes (roughly 50 to 55 tok/s). Smaller chunks cost more reads. These are estimates; the chunk size
must be chosen from measured 2K and 4K runs, not from this.

## 5. oMLX (Apache-2.0, `jundot/omlx`), what was borrowed and what was not

Read for design only; no code was copied. Ideas adopted: block-hash chains for prefix identity, sliceable KV
stored separately from non-sliceable recurrent state, boundary snapshots for GatedDeltaNet layers (oMLX writes
"GDN sidecars" per block boundary), safetensors files with an async, atomic write path, a disk budget that
survives restarts. Not adopted: paged GPU blocks and continuous batching (single sequence here), lossy GDN
state codecs (`int8`, `rht_int8`; ours are exact by requirement), per-block fixed-size slicing (our boundaries are the
system prompt and chunk edges, which are where exact recurrent state exists). If code is borrowed later, keep its
Apache-2.0 header and a NOTICE entry.

## 6. Server and API facts (DeepSeek-style clients)
* `veystrix/server.py` serves OpenAI `/v1/chat/completions` and `/v1/responses`, and Anthropic `/v1/messages`. Each call
  renders the whole conversation (system plus tools plus history) through the chat template, tokenizes it, and
  `_adopt` keeps the live cache only if the tape is a strict prefix; otherwise it resets. That reset is where the prefix
  cache now applies. Tools sit in the system block, so the system-boundary snapshot covers them.
* Usage already reports `prompt_tokens`; a cached-token count (`prompt_tokens_details.cached_tokens`) is **not yet
  exposed**. `backend.last_prefix` holds the report (hit tokens, tier, restore seconds, bytes, snapshots).
* The advertised `context_window` falls back to the model's `max_position_embeddings` (262,144) or a default. Nothing
  validates that. It should be capped to a validated size before a client relies on it. Not changed here.
* Request cancellation during a restore or a chunked prefill is not specially handled yet; the existing cancel check runs
  per layer through `set_prefill_progress`.

## 7. Decode (not implemented; ranked from recorded data)
Measured: 2.1 tok/s, 67% pool misses (9,739 hits, 19,457 misses over 77 tokens), 63.4 GB read in decode, about 0.82 GB
per token, about 1.7 GB/s (the SSD ceiling here is about 3.1 GB/s, so decode is not purely read-bound).
1. Expert residency policy (frequency-aware admission, per-layer sizing): the only lever on bytes per token. A hit
   rate rising from 33% to 45% would cut reads by about 18%. The slab policy already in use (`skew`, frozen
   profile) is a starting point. Needs routing traces, which exist (`bench_route_trace`).
2. Native MTP: **recorded results reject it on this machine** (`research.md`): depth 1 reached 89% acceptance and
   still lost (1.25 against 1.31 tok/s with resident experts; 1.6 against 1.7 in the complete chat). Each verified
   position routes its own experts, so reads grow with draft depth. Re-measure only with the oMLX Lightning kernels,
   and auto-disable if slower.
3. Avoidable synchronizations and kernel fusion: recorded gains were below the 20% acceptance threshold.
Nothing here should be built before Stage 1 and 2 are validated on the real model.

## 8. Not verified on the real model
* Chunked prefill output versus the single call (digest at 1K, 2K, 4K).
* The prefix cache end to end (restore, SSD tier, restart) and its memory effect while a prefill runs.
* Any prompt above 1,025 tokens; the 10K memory and time figures above.
* Cold SSD read time of a restored snapshot.

## 9. Proposed real-model ladder (each step needs approval; none has run)
1. 1K F2L with `FLASHNEXT_PREFILL_CHUNK_TOKENS=512`, system-boundary snapshot, then the same prompt with a changed
   question (restore from RAM), then a fresh process (restore from SSD). Compare tokens to the reference digest; report
   any difference and its size.
2. 2K and 4K the same way, choosing chunk size from the timings.
3. 8K, then 10K, only after the previous step ends without a guard stop and leaves at least 2 GB of headroom.
4. Final numbers plugged in with Low Power Mode off; battery runs are PRELIMINARY.

## Real-model validation of chunked prefill and the prefix cache (2026-11-07, PRELIMINARY)

All runs: Q4 MTP checkpoint, `FLASHNEXT_EVAL_LINEAR_STATE=1`, `FLASHNEXT_PREFIX_CACHE=1`, 512-token chunks, 512-expert scratch ring, 2 GB decode pool, all guards on, battery 69-75% with Low Power Mode off. One measurement per arm. Result directories are under `results/flashnext/` and are not committed.

**Chunked versus unchunked (1K, 1,025 tokens).** Not bit-identical. Digest `ee761769891b452a` against `2f84f7d220e782b8`, KV hash differs, generated sequences agree for 48 tokens and then differ (both continuations faithful to the source). Final-state relative L2 error grows from about 2e-3 in layer 0 to 0.1-0.25 in layer 47 across all layers and state types; position and offset tensors are bit-identical. The first 14 positions, which see an identical causal prefix and involve no state hand-off, already differ by 9e-3 (layer 3) to 1e-1 (layer 47), so shape-dependent bf16 noise of that size exists without any chunk logic. A fp32 control on the small model agrees to 2e-7. Consistent with kernel-shape drift; routing flips as the cause of the heavy per-token tail are untested. Chunked prefill therefore needs a digest check, and a cached run must be compared with an uncached run under the same chunk plan.

**Cached against uncached under the same chunk plan: bit-identical in every case.**

| Prefix | Restore | Suffix | Tokens, digest, KV hash | State tensors |
|---|---|---|---|---|
| 14-token system message (1K) | fresh process, SSD, 0.075 s | 1,023 | identical | 124 of 124 bit-identical |
| 2,046-token coding-agent system message | same process, RAM, 0.5 ms | 21 | identical (64 tokens) | 124 of 124 bit-identical |
| same, restored after a restart | fresh process, SSD, 0.113 s, 172 MB | 21 | identical | 124 of 124 bit-identical |

Snapshots taken in separate processes are tensor-identical with the same key, fingerprint and token hash. The stored entry's checksum is unchanged after restore, prefill and decode, and the files on disk are byte-identical before and after.

**Cold prefill at 2K (2,068 tokens, chunks 512/512/512/510/22).** 65.4 s, 31.6 tok/s, 150.2 GB SSD expert reads, MLX peak 5.34 GB, snapshot 172 MB written in 0.11 s. The cold run is reproducible (same digest and KV hash on repeat).

**What reuse saves (2,067-token prompt, 2,046 reused, 21 prefilled).** Prefill 65.5 s to 4.4 s (RAM) or 4.8 s (SSD restart): time to first token 65.6 s to 4.6 s or 5.0 s, total latency 96.8 s to 34.3 s or 34.8 s, SSD expert reads 149.6 GB to 9.3 GB (93.8% avoided). Restoration speed (a 172 MB state copy) is not prefill throughput: the remaining 21-token suffix still costs about 4 s because every chunk, however small, stages its experts.

**Chunk cost.** Measured cold chunks fit `seconds = 3.9 + 22 ms x tokens` (16 points, worst residual 1.3 s): a fixed expert-staging cost of about 3.9 s per chunk and about 45 tok/s asymptotically. A 512-token chunk costs about 15 s (about 34% above the asymptote). The fit comes from chunks of 21 to 1,025 tokens; its extrapolation to 4K-10K is an estimate and ignores attention cost at long context.

**Chunking policy for 4K-10K (estimates).**
- Use 2,048-token chunks. Modelled cold prefill of a system prompt plus a 22-token question: 4K about 102 s (512-chunks: 126 s), 8K about 200 s (247 s), 10K about 244 s (302 s). The extra transient memory (about 0.4 GB over 512-chunks by the budget model) must be confirmed on the real model before 8K.
- Plan chunks against the snapshot boundary rather than only absolute multiples. A boundary a few tokens past an absolute edge creates a tiny chunk that costs a full staging pass (a 2,088-token system message would add about 5 s). Split each segment (system message, then the rest) into equal pieces of at most the chunk size, so a restored run and a cold run still agree by construction. Not implemented; the planner is unchanged.
- A snapshot at the system boundary pays the one-time cold cost once; later requests pay only for their suffix.

**Safety margin.** Every arm stayed inside the guards (pressure level 1, no swap-out, no guard stop). The compressor warning fired in several arms, and in the two-request and restart arms minimum available memory was about 3.2 GB against the 3 GiB stop line (the stop needs growth above 2 GiB and available memory below 3 GiB), a margin of roughly 0.14 GB. Back-to-back arms leave the machine less settled; allow it to recover between arms.

## Boundary-aware chunk planner (opt-in, `FLASHNEXT_PREFILL_CHUNK_POLICY=balanced`)

Implemented and unit-tested; not yet run on the real model.

**Why.** The original "grid" policy cuts at every absolute multiple of the chunk size plus each snapshot boundary. A boundary a few tokens past a grid line therefore creates a tiny chunk, and every chunk costs a full 48-layer expert-staging pass (measured: about 3.9 s fixed per chunk). A 2,088-token system message with 512-token chunks would have run a separate 40-token chunk (about 5 s).

**Policy.**
- `FLASHNEXT_PREFILL_CHUNK_TOKENS` is now bounded to 64..2,048 (an upper bound keeps transient memory bounded). 0 still means off.
- `FLASHNEXT_PREFILL_CHUNK_POLICY` is `grid` (default, byte-for-byte the old planner, pinned by a test that embeds the old algorithm) or `balanced`.
- Balanced: the prompt is split at its anchors (the system-message end, or every message end with `FLASHNEXT_PREFIX_CACHE_BOUNDARIES=messages`), and each segment is cut into the fewest equal pieces of at most the chunk size. Chunk sizes inside a segment differ by at most one token. Anchors are always edges, so a snapshot is always taken exactly at its boundary, even when the boundary falls inside what a grid chunk would have been; nothing already processed is rewound.
- The plan is a function of the whole prompt (its end, chunk size and anchors), never of how many tokens are cached. A restored run takes the tail of the cold plan, so cached and uncached continuations use identical chunks from the reused boundary onward by construction, for any system-message length. A hit that is not an edge (for example a "chunks"-mode snapshot under a different prompt end) shortens only the first span; the state is still an exact continuation, but bit-identity with a cold run then holds only when the end and anchors agree (tested both ways).
- Examples (system message, then a 22-token question; 2,048-token chunks): 2,046 gives `[2046, 22]`; 2,088 gives `[1044, 1044, 22]`; 4,096 gives `[2048, 2048, 22]`; 10,000 gives `[2000 x5, 22]`.
- The cache key includes the policy, the boundary mode and the minimum token count when the policy is `balanced`; the default `grid` key is unchanged, and a grid prefix never serves a balanced run.
- Fallback: leave the policy unset, or set the chunk size to 512, to get exactly the strategy that was validated on the real model.

**New boundary mode.** `FLASHNEXT_PREFIX_CACHE_BOUNDARIES=messages` snapshots at every message end (system, user, assistant, and tool results, which arrive inside a user turn), so an agent loop can reuse the longest stored conversation prefix. Opt-in; each snapshot costs about 170 MB at 2K, so mind the disk budget.

**Tests added** (all offline): planner coverage, size bound and minimal piece count over many ends, chunk sizes (including 100, 777, 1000, 1500) and anchor sets; restored plan equal to the cold plan's tail at every edge and anchor; non-edge hits; tool-message anchors; grid regression against the embedded old algorithm; validation of policy and chunk size; on the small model, restored (RAM and SSD) equals cold bit-for-bit for system messages of 14, 63, 64, 100, 129, 200 and 257 tokens, executed spans equal the plan, tool-message conversations reuse the latest message end, partial hits, fingerprint isolation, and cancellation (after the snapshot, in the first chunk, and between chunks: the stored entry and its files are untouched, no partial files are left, and the next request equals a cold run).

## Memory budget for the 2K system prompt at 2,048-token chunks (prediction, 2026-11-07)

Budget model (`long_context_budget.py`, calibrated on 1K): at 2,068 tokens the predicted peak is 6.17 GB with 512-token chunks and 6.87 GB with 2,048-token chunks (+0.70 GB: 0.54 GB per-token transients plus 0.15 GB attention scores, both bounded by the chunk). The model overpredicted the measured 512-token run by about 0.4 GB (measured 5.11 GB active plus 0.67 GB allocator cache). The calibrated expectation for 2,048-token chunks is about 5.8 GB active (6.5 GB with the allocator cache), 5.7 GB under the Metal guard line (12.21 GB). The persistent state at 2,068 tokens is 0.17 GB, the same for any chunk size. No logits are built for non-final chunks, and a layer cannot overflow the 512-slot ring because the model has only 512 experts.

The machine-level limits matter more than Metal. Earlier 512-chunk runs consumed 3.9 to 4.4 GB of available memory during prefill (and up to 5.6 GB over a whole run). At 2,048-token chunks, add about 0.7 GB.

## First real-model run of the balanced 2,048-token planner (2026-11-08, battery 58%, PRELIMINARY)

One cold 2K arm (`F2B`, `results/flashnext/20261108-002253-prefill-memory-manual/`, agent prompt A, 2,068 tokens), run only
after the offline suite passed (1075 tests OK, 5 skipped) and the gate held (9.6 GB free or inactive, pressure normal,
`--check-ready` READY, Low Power Mode off). No guard fired. Nothing larger was run.

| | C1 (512-token grid chunks) | F2B (balanced, 2,048) |
|---|---|---|
| chunk plan | 512, 512, 512, 510, 22 | 2,046, 22 (the system boundary is the only anchor) |
| prefill | 65.4 s (31.6 tok/s) | 33.9 s (61.1 tok/s) |
| SSD reads in prefill | 150 GB | 58.8 GB |
| MLX peak (prefill) | 5.11 GB | 5.82 GB (predicted +0.70) |
| compressor growth (prefill) | 1.27 GB | 2.65 GB (warning threshold 2 GiB crossed, stop rule not met) |
| min available (prefill) | 4.59 GB | 3.27 GB (stop line 3 GiB: thin) |
| snapshot | 2,046 tokens, 172 MB | 2,046 tokens, 172 MB, 0.11 s, plan edge [2046] recorded |
| tokens / digest / KV hash | 52 / `9049f1254e5e2481` / `b2bc0ae3d01b362e` | 49 / `e326048fd867d4be` / `555f71e4ff34a9b5` |

Equivalence tiers (see quality-validation-policy.md):

* Tier A (bit-identical): not applicable and not met. F2B versus C1 changes the chunk plan, so tokens and state are not
  expected to match. The tier A check for this plan (F2B cached versus F2Bn uncached) has NOT been run.
* Tier B (numerical): the first token matches; generated tokens diverge at index 2 (C1 "I'll start by exploring the
  repository structure ..." / F2B "I'll explore the repository ..."). Both replies are coherent and make the same tool call
  (list_directory on `.`, depth 3 versus 2). Margins were not recorded on the real run, so whether index 2 was a near tie is
  unknown. Offline dump comparison: `rope_deltas` identical, `position_ids` equal on the overlap; KV relative L2 per position
  is flat in position (layer 3 about 0.012, layer 11 0.025, layer 43 0.13, layer 47 0.14, growing with depth; the same
  size as the 1K chunked-versus-unchunked drift). Boundary spike ratio at the F2B edge 2046: 1.54 (worst layer 47);
  at C1's own edges up to 2.03; at two control positions that are not edges 1.32. All far below the 5 (investigate) and
  20 (suspected defect) thresholds, so there is no sign of a hand-off defect at the edge. Not run on the real model: routing
  flips, logit margins.
* Tier C (task level): the real-model quality suite has not run.

Reading: the balanced planner halves cold prefill time and cuts SSD reads by 61% at 2K, at +0.7 GB peak and a thinner memory
margin. It is not shown to preserve behavior: only an offline drift profile consistent with bf16 shape noise and one
coherent but different greedy output. It stays opt-in.

## Stage 1 attempt: uncached same-plan reference (2026-11-08, AC power 58%, PRELIMINARY)

Arm 1 = `F2Bn` (session plan, restore and snapshots off), requests `agent:A,agent:B` in one process,
`results/flashnext/20261108-003402-prefill-memory-manual/`, started at 11.1 GB free or inactive, pressure normal.

* Request A finished. Chunk plan [2046, 22], digest `e326048fd867d4be`, KV hash `555f71e4ff34a9b5`, 49 tokens, and all 124
  dump tensors are bit-identical to the earlier cold `F2B` run (a separate process that also wrote a snapshot). So the
  result is repeatable across processes, snapshot-taking does not change the result at the balanced plan, and `F2Bn`
  really has the same plan as `F2B` (an earlier version of `F2Bn` without a session had no anchors and would have cut
  1,034 + 1,034; fixed, with `FLASHNEXT_PREFIX_CACHE_RESTORE=0`).
* Request B (a second cold 2,068-token prefill in the same process) was stopped by the memory guard: compressor growth
  6.02 GiB with 2.65 GiB available (rule: warn above 2 GiB growth, stop if also under 3 GiB). MLX active reached 6.97 GB
  and 2.1 GB of MLX cache was still held from request A. This is a resource stop, not a correctness failure, and it was
  not retried. Consequence: at 2,048-token chunks a second cold long prompt in a warm process does not fit on this
  machine; it matters for 4K (release the MLX cache and decode pool between requests, or run cold long prompts in a fresh
  process).
* Stage 1 is therefore incomplete: no reference for request B, and the cached arms were not run.

## Stage 1 result: cached versus uncached, request B, balanced 2,048 planner (2026-11-08, AC power, PRELIMINARY)

Two fresh-process arms, request B alone (2,067 tokens, 64 decode tokens), each started at 9.3 and 10.0 GB free or
inactive, pressure normal, no swap activity. No guard fired. The A-then-B RAM-hit arm was not run.

* Reference: `F2Bn` (session plan, restore and snapshots off), `results/flashnext/20261108-003951-prefill-memory-manual/`.
  Chunks [2046, 21], no lookup, no snapshot.
* Cached: `F2B` restoring the 2,046-token snapshot from SSD (written by the earlier cold A run, copied to a new directory;
  files byte-identical before and after the hit), `results/flashnext/20261108-004230-prefill-memory-manual/`.
  Hit 2,046 tokens (tier ssd), 21 prefilled, chunk [21], plan rejections 0, snapshot manifest `n_tokens` 2046,
  `plan_edges` [] (no earlier edge), format 2.

| | uncached (F2Bn) | SSD hit (F2B) |
|---|---|---|
| generated token ids (64), digest | `3349ced58d7837d1` | `3349ced58d7837d1` (identical) |
| KV hash | `a387f8af541ce78c` | `a387f8af541ce78c` (identical) |
| state tensors (KV, GDN recurrent and conv, indexer, position) | 124 | 124/124 bit-identical, worst diff 0 |
| output text and tool-call text | identical (the 64-token limit cut the call after `run_command`, `pwd && ls`, `working_directory /`, `timeout_seconds 30`) | identical |
| prefix reused / restore | 0 | 2,046 tokens, 0.069 s, 172 MB read |
| prefill / TTFT | 33.8 s / 34.0 s | 4.4 s / 4.7 s |
| SSD expert reads, prefill | 58.7 GB | 9.2 GB |
| SSD expert reads, decode | 55.1 GB | 47.3 GB |
| MLX peak (whole arm) | 5.82 GB | 5.34 GB |
| compressor growth, prefill / decode | 3.45 GB / 0.07 GB | 0.72 GB / 0.83 GB |
| minimum available | 3.18 GB (prefill; stop line 3 GiB) | 3.58 GB (decode) |
| swap | none (used 861.5 MB throughout, no swap-out or swap-in) | none |

Tier A holds for this case: cached continuation from SSD is bit-identical to the uncached run with the identical balanced
plan, in a fresh process. The model's tool choice for this prompt (`run_command` with a `working_directory` argument that is not
in the production schema) is a behaviour observation about the model and is the same in both arms. The uncached 2K
prefill alone dips to 3.18 GB available, 0.1 GB above the stop line.

## Stage 1 (RAM hit) passed; Stage 2 arm 1 stopped by the memory guard (2026-11-08, AC 80%, PRELIMINARY)

**Stage 1**, `results/flashnext/20261108-010535-prefill-memory-manual/` (`F2Br`: balanced 2,048 plan, cache on, release flag on, fresh process, A then B,
started at 10.1 GB free or inactive). A cold (chunks [2046, 22], snapshot 2,046 tokens in 0.108 s, RAM and SSD). B: **RAM hit, 2,046 tokens reused**
(restore 0.36 ms, 0 bytes read from SSD, 21 tokens prefilled, plan rejections 0). B digest `3349ced58d7837d1`, KV hash `a387f8af541ce78c`, 64 token ids and
124/124 state tensors identical to the fresh-process uncached `F2Bn` reference; A identical to the earlier cold A. Snapshot unchanged (RAM checksum `bae8de01cbdd84fc`
before and after B; SSD data file hash equal to the earlier run's). The pool release was correctly skipped for B (`prefill is not eligible for the scratch ring`,
sized by the 21-token suffix, not the 2,067-token prompt) and no second decode pool was created (events: only the load-time and A's decode creations).

| | A (cold) | B (RAM hit) |
|---|---|---|
| TTFT / total | 33.8 s / 57.7 s | 4.6 s / 33.3 s |
| SSD reads, prefill / decode | 58.8 / 43.1 GB | 9.2 / 46.2 GB |
| MLX peak | 5.82 GB | 5.62 GB |
| min available | 4.17 GB | 3.78 GB |
| compressor growth (max) | 0.62 GB | 4.45 GB, drained to 0.6 GB in 4 s; MLX active flat at 5.2 to 5.3 GB |
| swap | none | none |

The compressor swell in B is system-level (MLX active did not move): a short prefill that does not use the ring reads experts through the normal path and the
resident anonymous memory briefly rises by about 2.5 GB. It stayed within the stop rule (warn above 2 GiB growth, stop only when also under 3 GiB available).

**Stage 2 arm 1** (`F2Pnr`, validated 512-token plan, `--suite smoke`, `results/flashnext/20261108-010847-prefill-memory-manual/`, started at 9.65 GB): the child
guard stopped it during the first case (code_palindrome, a short prompt): compressor growth 2.01 GiB while 2.99 GiB was available (floor 3 GiB). No case
completed, so Stage 2 produced no quality data. The first short (about 40 token) prefill without the ring raised compressor growth from 0 to 3.3 GB and anonymous memory
to 5.4 GB; the compressor then drained only to 2.0 GB growth and the steady state held at 3.1 to 3.2 GB available (MLX active 5.1 GB with the decode pool, file-backed
3.5 GB), so the arm sat on both thresholds. Not retried; guards not relaxed. The arms after it (`F2Bnr`, `F2Br`) were not run.
