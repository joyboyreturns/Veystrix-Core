# 10K-token system prompt on a 16 GB M4: audit, projections, plan

Branch `development3.8`. Offline work only: no real-model inference ran for this note. Tags: **structural** (config
or code), **measured** (real run on this machine, battery, PRELIMINARY), **synthetic** (this machine, random
weights or random tensors, no checkpoint), **estimate** (arithmetic). Nothing here is a measurement of a 2K+ prompt.

## 1. Architecture facts that decide the design

* 48 layers: 36 GatedDeltaNet (linear attention, fp32 recurrent state) and 12 full-attention layers with QSA
  (Qwen Sparse Attention) indexers. **structural**
* The residual stream is 4 hyper-connection streams, so every hidden tensor is `[tokens, 10,240]` bf16, 20 KB per
  token. **structural**
* Persistent state per sequence: 36 x 3.15 MB recurrent state, plus 12 x 2 KB KV and 12 x 0.27 KB indexer keys
  per token. About 0.14 GB at 1K and 0.39 GB at 10K. **structural**
* The recurrent state cannot be rewound or trimmed; only the KV part can. `veystrix/server.py` `_adopt` says so and
  keeps the cache only when the tape is a strict prefix of the new prompt. **structural**
* The whole pending prompt goes through **one** model call (`flashnext.py` `prefill_language`, no chunking).
  Every MoE layer handles all `tokens x 10` routed pairs at once (`expert_cache._issue`). **structural**
* Full logits are built only up to 2,048 tokens (`prefill.py`: `min(QSA_CHUNK_THRESHOLD, 8192)`). Above that the
  call skips them and the token comes from the fused head on the last row. A 10K prompt does not build the 5 GB
  logits tensor. **structural**
* The QSA indexer's dense mask would be `tokens x 512 x tokens` bytes (51 GB at 10K). `qsa_chunk.py` already
  routes large calls to a query-chunked sparse path (chunk 2,048). **structural**
* Unfused attention scores: SDPA with head_dim 256 and a bool mask used 46, 193 and 440 MB at 1K, 2K and 3K
  tokens, which is 48 B per (query, key) pair (24 heads x bf16). **synthetic**. With query chunks of 2,048 that is
  about 1 GB at 10K. A full 10K x 10K call would need about 4.8 GB.

## 2. The main finding: prefill holds about 0.85 MB per prompt token

End-of-prefill MLX active memory minus the weights, from completed F2 arms: **measured**

| Prompt tokens | 129 | 257 | 1,025 |
|---|---|---|---|
| Retained above weights | 0.233 GB | 0.345 GB | 0.992 GB |

That is a straight line: about 0.12 GB plus 0.85 MB per token. The intercept is the GatedDeltaNet state. The
persistent KV and state are only 0.03 MB per token, so about 0.8 MB per token is not state. The same arm's memory
after decode (0.33 GB above weights and pool) shows most of it is released at the first decode step.

**Mechanism, shown in isolation (synthetic):** a GatedDeltaNet layer with the checkpoint's shapes keeps its
convolution state `cache[0]` as a lazy slice of `conv_input` (20 KB per token). Nothing the layer outputs depends on
it, and the streaming MoE evaluates only layer outputs. Measured per layer: 42 to 47 KB per token held until
`cache[0]` is evaluated, then 3.3 MB (the real state) once it is. Over 36 layers that is the right order and sign.
**It has not been confirmed on the real model**, and I would not rely on the exact share until a real run with the
fix shows end-of-prefill retained memory falling. For prompts above 2,048 tokens `_prefill_target` evaluates the
states at the end, which frees the memory only after the peak has been reached.

Two earlier guesses were wrong and are recorded here so they are not repeated: an isolated layer test without
`gc.collect()` appeared to retain 130 MB (it was garbage), and a synthetic slice test showed pinning only in some
graph shapes. The evaluated-versus-unevaluated test above is the one that holds.

## 3. Projected peak, `python -m models.flashnext.tests.bench.long_context_budget` (estimate)

Counts weights, the 512-expert scratch ring, retained memory, logits, per-token transients (hyper-connection
streams and MoE), attention-score chunk and the allocator cache. Calibrated at 1K only: it projects 7.68 GB
against 7.31 GB measured (MLX active 5.96 plus cache 1.35), so it is conservative there. Metal limit 12.71 GB; the
guards stop at 12.21 GB.

| Tokens | As is | Fits | With the state fix | Fits |
|---|---|---|---|---|
| 1,024 | 7.68 GB | yes | 6.83 GB | yes |
| 2,048 | 9.57 GB | yes | 7.88 GB | yes |
| 4,096 | 11.22 GB | yes (0.99 GB spare) | 7.85 GB | yes |
| 8,192 | 16.55 GB | **no** | 9.81 GB | yes |
| 10,000 | 18.91 GB | **no** | 10.68 GB | yes, thin |

* As is, a single-call 8K or 10K prefill would hit the guard. The fixed column leans on two estimates: eight live
  copies of the hyper-connection tensors, and 190 KB per token of MoE transients (synthetic, MoE only).
* Prefill time (estimate): 1K takes 25 s, read bound (53 GB). At 10K every layer routes all 512 experts, about
  75 GB read (about 25 s at 3 GB/s) while GPU time grows with tokens, so expect minutes, not seconds. This is why
  reuse (section 5) matters more than speed.
* macOS also needs room: at 1K the second run was within 0.4 GB of the 3 GiB compressor stop. A 10 GB MLX
  footprint leaves almost no margin on 16 GB, which is why chunking (Priority B) is still needed even with the fix.

## 4. Priorities

**A. Logits.** Already handled above 2,048 tokens. Between 512 and 2,048 tokens the full-logit path costs 0.25 to
1.0 GB. `FLASHNEXT_PREFILL_FULL_LOGITS_MAX_TOKENS` exists, but a last-row projection was recorded as changing
logits by up to 0.125, so it is not bit-exact; lowering it needs a digest test and your acceptance. Not
recommended for the 10K goal.

**B. Bounded prefill.** Not built. Facts: GatedDeltaNet carries state, so chunking is mathematically sound there,
and the QSA path already handles `past_len > 0`; the multi-turn path already prefills increments on a warm cache.
But `qmm` and attention kernels depend on the row count (the 0.125 logit difference is an example), so chunked
output is **not assumed bit-exact**. The test is the 1K reference (digest `2f84f7d220e782b8`, KV hash
`fd83510bd53f9c82`) prefilled as two chunks. Cost: each chunk re-reads its experts (about 400 per layer), so SSD
reads grow with the chunk count. Needs your approval to build.

**C. System-prompt caching.** The server holds one slot and drops it whenever a request is not an extension. A 10K
prefix reuse design that fits the architecture: snapshot the cache at the exact token boundary after the system
prompt (recurrent state cannot be recovered later), about 0.4 GB at 10K; key it by the token ids, model
fingerprint and engine fingerprint (`frankenstein._session_fingerprint` already computes these for sessions); on a
request whose first N tokens equal a snapshot, copy it in and prefill only the rest. Invalidate on any token
mismatch, tool-definition change, or setting change that alters numerics. Disk persistence already exists
(`save_session`, `load_session`). Not built.

**D. Memory options.** The state fix (new, below) is the largest saving found. The 512 MB cache cap and
`FLASHNEXT_STREAM_EMBED=1` are still unmeasured; measure each alone, after the state fix.

**E. SSD.** The ring reaches 3.2 GB/s with 16.6 s of waiting per 1K. Every layer at 10K needs all 512 experts, so
the ring's 512-expert capacity is exactly full with no spare. No change proposed before the 2K and 4K numbers.

## 5. Implemented in this change (opt-in, uncommitted)

* `models/flashnext/linear_state.py`, flag `FLASHNEXT_EVAL_LINEAR_STATE=1` (default off, registered in
  `setting_research.py`, applied from `loader.py` like the other patches). After each GatedDeltaNet call of 64 or
  more tokens it evaluates `cache[0]`. The operations are unchanged and only their timing moves, so values and state
  are bit-identical by construction and by test. Decode (one token) is never touched.
* `tests/unit/test_linear_state.py`: 11 tests, including bit-identical outputs and state on a small real layer and
  a check that the held bytes drop below 10% of `conv_input`. Mutation checks (never evaluates, wrong state,
  evaluates during decode) were each caught.
* `tests/bench/long_context_budget.py` and `tests/unit/test_long_context_budget.py`: the projection above, with 9 tests.
* Full unit suites: flashnext 890 (OK, 5 skipped) and veystrix 427 (OK). The 9 budget tests were run on their own.

## 6. Proposed real-model stages (each needs your approval; none has run)

1. **1K F2 with the fix**, one arm, AC preferred: pass if digest, KV hash and token count equal the reference and
   end-of-prefill retained memory falls from 0.99 GB to about 0.3 GB. This also checks the mechanism.
2. Then 2K, 4K, 8K, 10K one at a time, each only after the previous one finishes without a guard stop and with at
   least 2 GB of headroom, and each preceded by `--check-ready`.
3. Add a per-layer active-memory trace to the 1K arm only if stage 1 does not show the expected drop.
