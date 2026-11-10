# Long-context AGENTIC: audit of the memory/KV work, and what is now connected (2026-11-08)

Branch `development3.8`, nothing committed, pushed or merged; `main` and PR #1 untouched; no model checkpoint was loaded in this session (tokenizer and chat template only). Everything below is **offline**. Measurements and estimates are labelled.

## 1. Verdict

Antigravity's latest layer was mostly **not connected to the runtime**, and the pieces that were connected were thin or unsafe:

| Claim | What the code did | Verdict |
|---|---|---|
| Adaptive chunks 2048/1024/512/256 | `prefill_scheduler.py` had **no importer** outside tests. `prefill._run_chunks` used a static plan. | **Not connected.** Now connected (section 3). |
| Scheduler picks chunks from live headroom | It subtracted the weights and the 1.6 GB ring from host "available", memory that is already resident and already excluded. A loaded process with 6 GiB available fell to the smallest chunk. It also imported constants from `tests/bench`. | **Wrong model.** Replaced by an increment model in `models/flashnext/prefill_budget.py`. |
| Affine Q4 attention KV consumed by attention | `quantized_kv.apply()` is called by the loader and wraps `LanguageModel.make_cache`. Upstream `base.scaled_dot_product_attention` dispatches on `hasattr(cache, "bits")` to `quantized_scaled_dot_product_attention`. `AffineQuantizedQSAKVCache` is a four-line subclass of upstream `QSAQuantizedKVCache`. | **Connected and consumed** (proved on the tiny real model, section 4). No hidden BF16 K/V copy (packed `uint32` arrays only). |
| Snapshot isolation BF16 vs affine4 | Fingerprint carried only the dtype string, **not group size or bits**, and invalid `FLASHNEXT_KV_GROUP_SIZE`/`BITS` silently fell back to defaults. | **Gap.** Fixed: dtype, bits and group size are in the key; bad values raise. |
| 3.55x KV reduction | Applies to K/V data only. The indexer keys and position ids stay BF16 (268 of 2,316 B per layer-token). | Whole-cache reduction is about 2.7x; and KV is small anyway (section 5). |
| Cosine > 0.98 validates the cache | See section 4: mean cosine 0.995 coexists with 10% typical and 40-60% worst-row attention error and a changed attended key in 4-45% of rows (outlier channels). | **Not sufficient.** affine4 stays opt-in, labelled not recommended. |
| Mode configuration matches flags | `modes.py` did not own `FLASHNEXT_KV_CACHE_DTYPE`, the adaptive flag, or the KV group/bits, so an inherited value in `~/.frankenstein/env` could silently change numerics inside a mode. | **Gap.** Now owned; defaults `bf16` and off. |
| Sessions with quantized KV | `sessions.py` (exact session save) assumes dense arrays and would die with an `AttributeError` on a tuple. | Now a clear `SessionError`. |
| `qsa_chunk` pooled-key reuse | exact `type()` check excluded the affine subclass. | `isinstance` (the flag is off by default). |
| CHAT unchanged | CHAT still uses one-call prefill. New code is inert unless `--enable long-context` (AGENTIC only) or a KV mode is chosen. `quantized_kv.apply()` is a pass-through for `bf16`. | Offline-verified only; the real CHAT regression run has not been repeated (section 9). |

Baseline before my changes: VEYSTRIX 524 OK, FlashNext unit 1439 OK (3 skipped). After: VEYSTRIX 537 OK, FlashNext unit 1461 OK (3 skipped) (section 10).

## 2. The central finding about memory

KV is not the long-context problem. At 12,288 tokens bf16 KV plus indexer keys is **0.34 GB** for all 12 full-attention layers (27.8 KB per token); affine4 would save about 0.2 GB, affine8 about 0.15 GB, of a 12.7 GB Metal limit. What grows with a long prompt and can hurt is the **per-chunk attention score matrix**, which is `queries x keys x 24 heads x 2 bytes`: for a 2,048-token chunk at 12K positions that is 1.2 GB, regardless of KV dtype. So the lever is chunk length as a function of position, and the quantized KV is a distraction for memory (it matters only if its quality were free, and it is not).

Expert streaming makes small chunks expensive: recorded prefill cost looks SSD-bound (259 tokens after a restore took 19.9 s, about the same as a cold 898 tokens). So the plan must use the **largest** chunk that fits, not the smallest that is safe.

## 3. What was built and connected

* `models/flashnext/prefill_budget.py` (production module, was `tests/bench/long_context_budget.py`, which is now a shim): the constants, plus `chunk_increment_bytes(chunk, position)` = transient per token + unfused attention scores + KV appended. Per-pair score cost (48 B) is measured; the per-token transient (354 KB) is an estimate.
* `prefill.py`, policy `budgeted` (`FLASHNEXT_PREFILL_CHUNK_POLICY=budgeted`, `FLASHNEXT_PREFILL_CHUNK_BUDGET_MB`, default 1000): each chunk is the largest of 2048/1024/512/256 whose estimated increment at its position fits the budget. Chunks shrink as context grows. It is a function of the prompt end, the anchors (system boundary, message ends) and the budget only, never of live memory, so a cold run and a restored run take the same chunks and stay bit-identical. Prefix-stable (verified in tests). Budget and candidate sizes are part of the cache fingerprint.
* Live guard (`FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS=1`, budgeted only): between chunks `prefill_scheduler` reads available memory and pressure and steps the cap down at once (pressure above 1 or under 4 GiB available, which is above the unchanged 3 GiB guard floor), up after two safe chunks. A step re-plans only the remainder. Every deviation is recorded in a `PlanTrace`; snapshots taken afterwards carry the edges that really ran, so they can only miss, never be mistaken for the cold plan's state. A run that cannot read the machine adapts nothing.
* `PlanTrace` per-chunk telemetry (seconds, resident memory, MLX peak increment when the chunk set a new process peak, host headroom, pressure). It does not reset the MLX peak counter (the harness and guards read the process-wide peak).
* `prefill_readiness`: a `long` tier (context 16,384, cold prefill 14,336) tried first only under the budgeted plan; its budget is the measured 5.40 GiB at 2,046 tokens plus KV growth plus 0.3 GiB, rounded up (ESTIMATE; 6.25 GiB at 14,336). Unchanged: the 9 GiB gate, pressure rule, 4 GiB headroom, all runtime guards.
* `modes.py`: `--enable long-context` (AGENTIC only; default context limit 16,384, still lowered by admission), `--enable kv-affine8`, `--enable kv-affine4` (labelled). All flags owned by the mode.
* `veystrix/context_budget.py` and server: separate advertised client window / server window / input / reserved output / continuation headroom. `/v1/models` publishes `suggested_client_context_window` (server window + the client's 4,096 safety margin). A tool request with `max_completion_tokens` under 64 is a 400 `max_tokens_too_small` before any prefill, with the fix in the message (the one-token failure cannot recur). Overflow errors give client-side compaction guidance and never truncate. `last_budget` and `last_divergence` (where a rebuild first differs from the live tape) appear in `/health` cache state.
* Fixtures: `models/flashnext/tests/bench/fixtures/agentic_fixture.py` builds 4K/8K/12K requests with the real tokenizer and chat template (4,108 / 8,232 / 12,308 tokens), facts only in tool results, a decoy config with other numbers. `dsh_wire_probe.py` runs the **real installed DeepSeek Harness 0.2.2-rc.1** against the real server with a scripted model.

## 4. KV quantization: measurements (synthetic attention, real shapes; NOT real-model KV)

The last column is for the outlier-channel rows (with flat attention the top key is ill-defined). Peaked attention, 24 query heads, 2 KV heads, head_dim 256, S = 4,096 and 12,288; outlier channels emulate large-magnitude K/V channels. Probe: scratchpad `probe_sdpa_quality.py`, `probe_sdpa_mem.py`.

| config | mean cosine | typical relative error | worst-row relative error (outlier channels, sharp) | attended key changed |
|---|---|---|---|---|
| 8-bit, group 64 | 0.9997 | 1-2% | 0.09-0.24 | 1-4% |
| 4-bit, group 32 | 0.991-0.997 | 8-13% | 0.32-0.58 | 4-36% |
| 4-bit, group 64 | 0.982-0.996 | 9-17% | 0.42-0.62 | 6-45% |

Scratch memory of the attention call is the same for bf16 and quantized (about 1.1x one score buffer: 157 MB at L=256, S=12,288 and 315 MB at L=512), so the quantized path saves persistent KV only. Speed: quantized SDPA ranged from 10% faster to 37% slower than fused bf16 in single noisy micro-benchmark runs (no conclusion). On the tiny random-weight model last-token logits cosine vs bf16 is 1.00000 (affine8) and 0.99725 (affine4), which says the plumbing is right and nothing about a trained model.

Decision: bf16 is the default and the reference. affine8 is the only candidate worth a real-model quality run; affine4 is not recommended. Neither is claimed exact. Real-model quality (filenames, tool names, JSON arguments, retrieval, multi-tool reasoning) needs a checkpoint run and is not done.

## 5. Memory budget by stage

Tags: **M** measured on this machine, **E** estimate, **N** not measured.

| Stage | Term | Value | Tag |
|---|---|---|---|
| LOAD | MLX active after load (dense core, embeddings, head) | 2.79 GB | M |
| PREFILL | scratch expert ring (512 experts) | 1.62 GB | M |
| PREFILL | attention scores | 48 B per (query, key) pair; 1.2 GB for 2,048 x 12,288 | M per pair |
| PREFILL | hyper-connection and MoE transient | 354 KB per live token | E |
| PREFILL | allocator cache (cleared per chunk) | up to 1.35 GB | M at 1K |
| PREFILL | measured MLX peak, 2,046 tokens, one chunk | 5.40 GiB | M |
| CHUNK TRANSITION | state evaluation, `clear_cache` | no extra | E |
| DECODE / TOOL CALL / TOOL RESULT | KV 27.8 KB per token (bf16), GatedDeltaNet state 0.11 GB fixed | 0.34 GB at 12K | structural |
| CACHED RESTORE | RAM tier 0.75 GB cap; restore peak vs cold | not measured above 2K | N |
| host | other apps' anonymous memory x 0.75 compression, wired, compressor | per readiness check | M (n=3 runs) |

Estimated MLX peak for the budgeted plan (resident weights + ring + state/KV + worst chunk): 4.9 GiB at 4K, 5.2 at 8K, 5.5 at 12K (E; the budget 1,000 MB is about what a 2,048-token chunk adds at position 0, which the 5.40 GiB measurement already contains). The 4.3-6.8 GiB profile figures were not verified and are superseded by these estimates until a run's per-chunk trace calibrates them.

Chunk plans (generated): 3,585 tokens, system anchor 900: `[900, 1024, 1024, 637]`; 7,671: 8 chunks; 11,786: 12 chunks of at most 1,024 after the first. Estimated worst chunk 0.54 / 0.74 / 0.94 GB. Time: unmeasured; if each chunk costs the SSD-bound floor seen at 1K, a 12K cold prefill is minutes, which is what the prefix cache exists to avoid.

## 6. DeepSeek Harness context (real client, scripted model; no model memory involved)

Measured over the wire with the real `dsh` against the real server (probe `dsh_wire_probe`):

| task size | first request (real tokens) | `max_completion_tokens` sent | client window used | tool turns |
|---|---|---|---|---|
| 4K | 3,585 | 1,024 | 12,288 (server 8,192 + 4,096) | glob, grep, final; turn 2 prefilled 33, turn 3 prefilled 60 |
| 8K | 7,671 | 1,024 | 16,384 (server 12,288 + 4,096) | same, 33 and 60 |
| 12K | 11,786 | 1,024 | 20,480 (server 16,384 + 4,096) | same, 33 and 60 |

The harness's default profile (23 tools) is **5,577 tokens for a trivial task**, so a normal first request of about 12K is plausible once project context is added; the owner's exact 12K could not be reproduced here (it depends on their project files). The harness subtracts a fixed 4,096 and its own input estimate (about 0.8x the real count) before sizing `max_completion_tokens`; the earlier "window = server limit" configuration produced 1. The required client window is the server window plus 4,096, published by `/v1/models`; do not assume the older table (12,288 / 16,384 / 24,576) for every request.

Real risk not testable offline: a tool turn extends the live state only if the client's re-rendered history re-encodes to exactly the tokens the model generated. With canonical scripted text it does (reused, 33 tokens). A real model's thinking text (if not echoed back) or a non-canonical BPE split would diverge and trigger a full rebuild of up to 12K tokens per tool turn. The server now records where (`last_divergence`), and the prefix cache would still supply the system and tools prefix. The first real run will show which it is.

## 7. Prefix cache

Verified offline on the tiny real model: cold vs restored runs bit-identical under the budgeted plan (token and all state tensors); snapshots at system, user and tool-result boundaries with a chunk edge on each; a live step-down records its deviation and the first (pre-deviation) snapshot still follows the cold plan; budget changes the key; affine KV round trip (capture, SSD write, read, restore, continuation logits bit-equal) and a different group size cannot be read back. Existing tests cover restart, integrity, eviction and single-owner locking (unchanged). Not measured: cold vs RAM-hit vs SSD-hit time on the real model above 2K, restore peak memory.

## 8. GPU and memory lifecycle

Unchanged and not re-verified on hardware: the GIL/deadlock repair (`gpu_sync`), out-of-process watchdog, guards, auto-purge. This work adds no new GPU synchronization: the chunk boundary evaluates cache states and clears the cache exactly as the 2K F2Br path did; the live guard reads `vm_stat`/`sysctl` with the same helper the purge thread uses, between chunks on the main thread, only when adaptive chunks are on. 0 GB expert pool stays the conservative default. Whether the last layers' lazily-unevaluated MoE work leaves buffers in flight at a boundary is the existing design; only a real run can confirm it for more chunks.

## 9. Status

| Workload | Scripted | Small MLX | Real Q4 | Multiple real tools | Memory safe |
|---|---|---|---|---|---|
| CHAT regression | n/a | unit/flow tests pass | **earlier run only** (2.7-3.0 tok/s); not repeated | n/a | earlier run |
| 4K | pass (real tokenizer, 2 tools + follow-up; real dsh) | plan, restore, boundaries pass | **not run** | not run | not run |
| 8K | pass | same | not run | not run | not run |
| 12K | pass | same | not run | not run | not run |
| affine4 / affine8 KV | n/a | plumbing and snapshot round trip pass | not run | not run | not run |

No real-checkpoint claim is made for 4K, 8K or 12K.

## 10. Test results

Final regression (Haiku subagent, unittest, repo venv; spot-checked):

| Suite | Before my changes | After |
|---|---|---|
| `veystrix/tests` | 524 OK | **537 OK** |
| `models/flashnext/tests/unit` | 1,439 OK, 3 skipped | **1,461 OK, 3 skipped** (the skips are the absent Vontra checkpoint) |
| new: `test_budgeted_prefill`, `test_agentic_long_fixtures`, `test_context_budget` | n/a | 30 OK |

The line `relation: same-plan verdict: FAIL` printed during the FlashNext run comes from `quality_compare.py`'s report formatter exercised by a test of its FAIL branch; the test passes. Eight existing runtime tests that used `max_tokens` of 1, 3 or 16 together with tools were changed to 64, because a tool request under 64 output tokens is now a 400; the cut-off-tool-call path they covered is still tested, and a new test covers the 400.

## 11. Gate A (real 4K) readiness: proposal, NOT started, needs approval

Measured readiness now (read-only, 21:5x local): on **battery** (40%), available 8.91 GiB (start gate needs 9.00: **fails**), pressure 1, swap 0.70 GiB, other processes 3.33 GiB anonymous. Headroom check: standard tier (6.0 GiB) 3.87 GiB (needs 4.00, fails), compact (5.5) 4.37 passes, long (6.25) 3.12 fails. So today the run would be refused by the product's own gate (exit 69, nothing loaded). It needs plugged-in power, Low Power Mode off, and about 0.2 GiB (standard) to 1.2 GiB (long) less other anonymous memory (quit the heaviest app; auto-purge only helps the file-cache part).

Proposed single run (do not run without approval):

```
./chat.sh --mode agentic --pool-gb 0 --auto-purge --enable long-context --context-limit 8192
```
then one real `dsh headless` task using `docs/flashnext/dsh/veystrix-agentic.patch.yml` with the 4K fixture task (about 3.6K first request: glob, grep, final, one follow-up). Continue only if the admitted tier is `standard` or `long` (a `compact` or `minimal` tier cannot hold it: abort). Expected (estimates): MLX peak about 5.0 GiB, plan `[900, 1024, 1024, ~640]` (system anchor about 900), max chunk increment about 0.55 GB, cold prefill minutes not seconds, tool turns prefilling tens of tokens if the tape re-encodes (else a rebuild: read `last_divergence`). Stop on any guard, GPU, watchdog or integrity event; no retry. Record: per-chunk trace (`Report.chunk_trace`), minimum free memory, compressor growth, swap-out, cache reuse, tool validity, answer correctness. 8K and 12K each need their own approval after reading Gate A's per-chunk increments against the estimate.

## 12. PR #1 review

PR #1 (draft, docs only, 3 commits) is consistent with the evidence except: it says the AGENTIC preset has an initial 4,096-token context limit (`modes.py` has 8,192, and 16,384 with `long-context`); it does not mention the budgeted plan, KV modes or the client-window rule. Its warning that the remote branch lacks `modes.py` still holds. Not edited; local docs here are the source for a later update.

## 12. Gate A real attempts (battery, 2026-11-08)

Attempts used: **2 of 3**. Attempt 3 is NOT started: attempt 2 ended in a safety-guard stop, which the approval makes a STOP condition.

| Attempt | Evidence dir | Result |
|---|---|---|
| 1 (22:25) | `results/flashnext/20261108-222532-gate-a-attempt1/` | Not a model failure. The orchestrator had loaded a tokenizer (0.52 GiB), so admission fell to the minimal tier and the 3,935-token request was refused. No GPU work. Fixed: precomputed task file, no heavy imports, hard dry-admission gate, needs-based admission, request-boundary recovery. |
| 2 (22:43) | `results/flashnext/20261108-224316-gate-a-attempt2/` | Admitted at tier standard (headroom 4.11 GiB), ready 3-6 s. Real cold prefill of 3,935 tokens 158.2 s (24.9 tok/s, 4 chunks 643/1024/1024/1024, no step-down, process peak 4.97-5.43 GiB), decode 2.33 tok/s, two tool-turn prefills of 55 tokens (~10 s each) reused live state. **Stopped at t=258 s by ChatGuard: macOS memory pressure level 2** (exit 75) during the third tool-turn prefill. Follow-up never ran. |

Measured in attempt 2 (guard-trace.jsonl, 0.25 s samples, watchdog 1 s): pressure was 1 in every sample except the last. Free memory was 0.07 GiB from t=10 s to the end (the page cache filled it). "Available" never fell below 3.06 GiB (the 2 GiB floor was not reached). Compressor went 0.84 -> 4.4 GiB within 10 s of the load and oscillated 1.9-5.7 GiB; at the stop it was 6.34 GiB (growth +5.49 GiB). Swap-outs 0, swap used flat at 717.5 MB. Wired memory moved between ~2.3 and ~6.0 GiB with the prefill/decode phases. MLX active 3.2 GiB, peak 5.06 GiB, cache up to 1.44 GiB. Server RSS 1-4 GiB. After the process exited the compressor fell 4.66 -> 1.45 GiB and free memory rose to 6.6 GiB, so about 3 GiB of compressed memory belonged to the server (RSS does not show it).

Reading: the engine's own working set (wired ~4.5 GiB above idle plus ~3 GiB compressed plus 1-3 GiB RSS) on a 16 GiB machine that already held ~3.3 GiB of other anonymous memory kept macOS compressing for the whole run; a prefill burst then tipped it to level 2. Admission counted page cache as available and did not see this. The context length is not the cause (the first, 4K cold prefill ran 158 s with pressure 1; the trip was on a 55-token turn). Guards were not touched.

Model behaviour in attempt 2: the model emitted two parallel `glob` calls per step and repeated glob variants; it never called `grep` in the 3 steps that ran. This is unresolved: it may be model quality at this prompt or the trimmed tool profile, and needs a run that is not cut short.

Orchestrator defect fixed offline: `analyze()` crashed on a string `/health` reading after the server died (`_cache_state`, regression `test_a_server_that_died_leaves_a_string_health_reading...`).
