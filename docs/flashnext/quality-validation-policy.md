# Flash-Next quality and correctness validation policy

Status: policy plus tooling, offline-tested. The real-model suite and the 2K balanced run have **not** run yet
(waiting on memory, see "Real-model runs" below). Nothing here changes a production default.

Objective: make prefill, decode, memory use and prefix caching faster and cheaper **without** lowering the quality of
the Qwen3.8-Flash-Next Q4 model. Longer context alone is not a result; a feature counts only when it keeps memory
stable and keeps task quality.

## 1. Non-negotiables (what an optimization may never do)

* Keep the Q4 affine G32 weights, the quantization, the routing (exact top-10 of 512, exact softmax and renorm) and
  the model mathematics as they are.
* No approximate attention, no expert skipping or substitution, no lossy state compression (KV or GatedDeltaNet
  recurrent/conv state), no reduced-precision state that the reference keeps at higher precision.
* Every optimization is opt-in (a flag or setting that is off by default). Turning it on must not change the code
  path of the verified default.
* An explicit fallback to the last verified path always exists (section 7).

## 2. Three kinds of evidence, never mixed

| Tier | Claim | How it is shown | Used for |
|---|---|---|---|
| **A. Bit-identical** | Same generated token ids, same complete continuation state, same positions | Token ids, KV hash, state tensors, `_position_ids` / `_rope_deltas` equal exactly | Cache versus uncached run **with the identical chunk plan** |
| **B. Numerically close** | Different execution order (chunking, another chunk size, another planner), states and routing within explained bounds | `numeric_diagnostics` (logits, top-1/top-2 margins, routing, KV, recurrent state, boundary spike ratio) | Chunked versus unchunked, plan A versus plan B |
| **C. Task level** | Same pass/fail on coding, reasoning, instruction, tool-call, multi-turn, retrieval cases | `quality_suite` + `quality_compare` | Every candidate, in addition to A or B |

Tier B is never reported as tier A. A task-level pass never excuses a failed tier A or an unexplained tier B finding.

## 3. Reference hierarchy

1. **F0**: production baseline, no pool during prefill, unchunked. The original path.
2. **F2L**: unchunked with ring, scratch pool and eager linear-state evaluation. Verified bit-identical to F0 on the
   real 1K prompt, so it is a safe reference for chunked runs.
3. **F2Pn / F2Bn**: the F2P / F2B session with restore and snapshots switched off (`FLASHNEXT_PREFIX_CACHE_RESTORE=0`):
   always cold, with exactly the chunk plan (boundaries and anchors) a cached run would continue. These are the
   same-plan references. A run with no prefix session at all is NOT one: it has no anchors, so a balanced plan would
   cut a 2,068-token prompt as 1,034 + 1,034 instead of 2,046 + 22.
4. **F2P / F2B**: the same chunk plans **with** the prefix cache.

Comparisons: F2P versus F2Pn and F2B versus F2Bn are tier A. F2Pn / F2Bn versus F2L are tier B and tier C.
F2B versus F2P (planner change) is tier B and tier C.

Determinism: greedy decoding (temperature 0), thinking off for the suite, fixed tokenization from the production chat
template, `mx.random.seed(0)`, same checkpoint, same expert pool settings, fresh process per arm. Prompt token ids
and their digest are recorded, so a difference in tokenization is visible before a difference in output.

## 4. Prefix cache rules (tier A)

A cached continuation must give bit-identical generated tokens, complete continuation state and position handling
compared with an uncached run **using the identical chunk plan**.

Chunk-plan provenance (implemented): a snapshot records the chunk edges it was produced by (`Entry.plan`, manifest
`plan_edges`, safetensors metadata `plan`, format version 2). On lookup the session computes the current prompt's
cold plan and refuses any snapshot whose edge history is not the prefix of that plan (`PrefixSession.compatible`,
`stats.rejected_plan`). Consequences:

* A balanced-plan snapshot is never used under a different cut history, and a grid snapshot is never reused where the
  cold run would have cut differently, even if the token prefix matches.
* Reuse is also bounded by the strictly-shorter rule, the SHA-256 chain key with a token re-check, and the fingerprint
  (model/session compatibility, chunk size, ring and qsa settings, and for the balanced policy the policy,
  boundaries and minimum tokens).
* A refused entry is a miss, never an error; the run falls back to a cold prefill of the whole prompt.
* Old format-1 snapshots are discarded on scan. Provenance conflicts on `put` keep the existing entry and count in
  `stats.plan_conflicts`.

Required checks for any cache change: offline unit tests (done), then a real-model cached-versus-same-plan-uncached
comparison with identical token ids and identical state dumps (tier A), run before the change can be enabled.

Known gap, to be closed by the suite: a cached suffix shorter than the ring minimum (512 tokens) runs without the ring
while the cold run used it for that span. The 1K real run matched bit-for-bit, but the evidence is one prompt; the
quality suite re-tests this with many short suffixes (every case after the first per system prompt).

## 5. Chunked versus unchunked (tier B)

BF16 drift between chunk shapes is **investigated, not accepted by default**. `numeric_diagnostics.compare`
reports:

* logits at the final position and per decode step (max/relative difference, and whether the top-1/top-2 margin is
  larger than twice the largest candidate logit difference, "decision safe");
* expert routing per layer: flip fraction, and whether every flip is a near tie (the k-th and (k+1)-th gate
  probabilities within the observed noise), because routing is exact and a non-tie flip is a defect;
* KV state (per layer) and GatedDeltaNet recurrent (fp32) and conv (bf16) state, position ids on the overlap;
* per-position KV error profile and the **boundary spike ratio**: error at chunk edges divided by the layer's median
  per-position error. Natural drift is smooth (ratios up to about 5 as outliers). An injected conv hand-off defect
  gave about 140. Aggregate error, logits and tokens did not reveal that defect; the position profile did.
* generation: greedy token ids, first divergence, margin at the divergence.

Interpretation (initial thresholds; tighten after real-model data):

| Finding | Action |
|---|---|
| fp32 chunk versus whole state error about 1e-7 relative | expected |
| bf16 smooth drift, spike ratio < 5, flips only at near ties, decisions safe | tier B accepted, still needs tier C |
| spike ratio 5 to 20 | investigate before enabling; not enough to block a test run |
| spike ratio >= 20, or a routing flip that is not a near tie, or a shape/position mismatch | suspected defect; **fails** |
| tokens diverge where the margin was large | fails; where the margin was a near tie, report and check tier C |

## 6. Task-level suite (tier C)

`models/flashnext/tests/bench/quality_suite.py`. Programmatic checks only (no model judge), greedy, thinking off,
each case a fresh conversation, saved after every case.

| Category | Cases (full) | Check |
|---|---|---|
| coding | 4 functions | reply code run against tests in a restricted interpreter (static AST checks, builtin whitelist, CPU limit, timeout) |
| reasoning | 4 | exact number or yes/no |
| instruction | 4 | counts, word limits, forbidden/required words, JSON keys and types, case |
| tool_call | 5 | production tool set rendered by the chat template; exactly one call, production parser reads it, required params, value patterns |
| multi_turn | 3 | facts carried over turns, derived values, a tool-result turn |
| retrieval early / middle / late | 3 + 1 two-needle | exact codes planted at about 5%, 45% and 100% of a ~2.2K-token system prompt, with decoys |

Tiers: `smoke` (7 cases, every category but retrieval_middle) and `full` (about 24). Case ids can also be listed.

Run (each arm a fresh process, guards and traces on, one real-model arm needs approval):

```
python -m models.flashnext.tests.bench.bench_prefill_memory --checkpoint vontra-mtp --sizes 1k \
    --conditions F2L --rounds 1 --expert-pool-gb 2 --suite smoke        # reference
... --conditions F2Bn --suite smoke                                       # balanced plan, no cache
... --conditions F2B  --suite smoke --prefix-dir <fresh dir> --prefix-min-tokens 8   # balanced plan, cached
python -m models.flashnext.tests.bench.quality_compare REF.json CAND.json --relation same-plan|reordered
```

Gate (`quality_compare`):

* `same-plan` (cached versus uncached, identical plan): **FAIL** on any token difference in any turn of any case.
* `reordered` (chunked versus unchunked, other chunk size or planner): token identity is reported, not required;
  **FAIL** if a case that passed on the reference fails on the candidate or any category loses passes.
* Cases missing from either run fail the gate.

Small, deterministic suites do not prove general quality. They detect large regressions and cache or state defects
that change behavior; combined with tiers A and B they are the evidence required below, not a benchmark score.

## 7. Enablement and fallback

An optimization may be enabled by default only if **all** hold: tier A passes where it applies; tier B shows no
suspected defect and no unexplained routing change; tier C shows no regression on the full suite; memory guards stay
clean on the target workload; evidence covers the sizes it is enabled for (2K evidence does not cover 10K).

If any check fails, the optimization stays off and the failure is recorded. Fallback ladder (each step is a setting
change, no code change):

1. Unset `FLASHNEXT_PREFILL_CHUNK_POLICY` (back to the grid planner, whose output is byte-identical to the old planner).
2. Chunk size 512 (the validated chunked path) or `FLASHNEXT_PREFILL_CHUNK_TOKENS` unset (unchunked).
3. `FLASHNEXT_PREFIX_CACHE` unset (cold prefill every request; the cache directory can be deleted at any time).
4. All Flash-Next experiment flags unset: the production path.

The 2,048-token balanced planner is opt-in (`FLASHNEXT_PREFILL_CHUNK_POLICY=balanced` with a chunk size up to 2,048;
anything larger is rejected).

## 8. Real-model runs (not run)

* No real-model arm runs without explicit approval per workload, no automatic retries, no automatic move to a larger size.
* The 2K F2B run waits for about 9 GB available memory, normal pressure and a passing `--check-ready`; user
  applications are never closed automatically. Battery runs are marked PRELIMINARY.
* When it runs: one cold 2K F2B arm with all guards and diagnostics, compared with the validated 512-token chunk
  reference (C1: tokens digest `9049f1254e5e2481`, KV hash `b2bc0ae3d01b362e`, peak 5.34 GB, compressor growth
  1.27 GB, min available 4.59 GB). Then stop; 4K is a separate decision.

## 9. Remaining risks

* Real-model evidence so far is 1K (and the 2K C-stages at 512-token chunks); the balanced planner, 2,048-token
  chunks and the suite have only offline tests.
* The routing-flip hypothesis (flips only at near ties) is proven on small models, not yet measured on the real one.
* Repeatability of chunked runs across processes is not shown on the real model.
* Cached short suffixes run without the ring (see section 4).
* Memory margin at 2,048-token chunks is predicted (+0.70 GB over 512 chunks), not measured; thin on a 16 GB machine
  with other applications open.
* Programmatic task cases are small; thresholds in section 5 are initial and should be revised with real-model data.
