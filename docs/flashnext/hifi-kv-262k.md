# Flash-Next 262K KV qualification, 2026-11-11

**Experimental. BF16 remains the default.** Group 64 reached 262,144 synthetic cache rows on Metal and completed one real 4,097-token cold-prefill comparison on the installed checkpoint. The BF16 and tiered4 arms emitted identical 32-token outputs for that prompt. Long-context retrieval and broader coding quality remain unqualified.

## Baseline and changes

The installed Q4 mixed-precision checkpoint declares 262,144 positions, 48 layers, 12 QSA layers and 36 GatedDeltaNet layers. The cache is separate from the weights. Each QSA layer has 24 query heads and BF16 K and V of shape `[1, 2, N, 256]`. The indexer has four query heads, one raw BF16 index key of width 128 per token, and integer text/MRoPE positions. It pools complete four-token blocks and selects 512; an incomplete causal tail is kept. GDN keeps FP32 recurrent matrices and BF16 convolution state; the combined structural estimate is about 110 MiB plus the PLE extras. The existing memory model's historical after-load MLX reading was 2.60 GiB and its mapped scratch ring was 1.51 GiB. The configured decode pool is up to 2 GB and normally replaces the scratch ring between phases. These are prior measurements/configuration, separate from this run. Metal reported a 12,713,115,648-byte recommended working set on this M4.

Default AGENTIC still uses BF16, a 2,048-token balanced prefill plan, streamed top-k experts, and its existing exact prefix cache. The context advertised by `/v1/models`, request ceilings for cold and incremental prefill, a verified workload, and the minimum actual live attention offset are four different quantities. Increasing or advertising a ceiling gives no qualification evidence.

`qsa-gather` retains the upstream block scorer, causal validity, `argpartition` tie policy and RoPE handling. It gathers the selected tokens in order and runs MLX SDPA on those rows. Eight queries are processed at a time by default. At 262K, the largest indexer score tensor for that tile is 8 MiB in FP32; a single selected BF16 K/V pair is about 16 MiB. The tiered path can materialize both tiers' selected rows and packed intermediates; these sizes are tensor shapes, not a peak estimate. Exact ranking still scans all complete **pooled blocks**, with a bounded query tile; the raw reference indexer is linear in history. No full-history token mask or full-history dequantization is used by the gather path. The previous path remains selectable. Mixed layouts fail closed on unsupported masks and native MTP; this is a scalar-offset causal text path. A different SDPA reduction order can change rounding even with BF16. Real checkpoint logit margins and selection agreement remain a gate.

`kv-tiered4` writes group-affine INT4 once from the original activations and retains an INT8 copy of a recent window (8,192 tokens by default, plus up to 255 tokens of page alignment). Aging drops expired INT8 rows; their INT4 rows already exist. There is no second full-context cache and no INT8-to-INT4 rounding cascade. The tier boundary is taken at the end of each prefill call; chunk plans are part of the numerical contract. No checkpoint weights, router, indexer precision, or GDN state were changed. Group 32 and 64 are supported for the mixed formats. The affine quantizer fits each row's groups online; it has no checkpoint-derived calibration, outlier mask, or rotation.

| Mode | Launch flag | Populated attention K/V at 262K, G64 | Total projected live state incl. indexer + GDN |
|---|---|---:|---:|
| BF16 K/V | default | 6.00 GiB | 6.89 GiB |
| INT8 K/V | `--enable kv-affine8` | 3.19 GiB | 4.08 GiB |
| BF16 K / INT4 V | `--enable kv-k16v4` | 3.84 GiB | 4.74 GiB |
| INT8 K / INT4 V | `--enable kv-k8v4` | 2.44 GiB | 3.33 GiB |
| INT4 primary / INT8 recent | `--enable kv-tiered4` | 1.79 GiB | 2.68 GiB |
| INT4 K/V | `--enable kv-affine4` | 1.69 GiB | 2.58 GiB |

The table is storage arithmetic for the checkpoint geometry, including affine metadata. It excludes capacity slack, the optional pooled indexer cache, weights, expert buffers, snapshots, and temporaries. `kv-tiered4` is the first long-context candidate because it fits the projected live-state budget while preserving more precision for recent attention; compare `kv-k8v4` and `kv-k16v4` if its quality regresses.

The [SGLang tiered implementation](https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang/blob/main/patches/kv_tiers.py) uses direct INT4 and an INT8 recent ring, with Triton kernels and different scale/group rules. Its [checkpoint and measurements](https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang) do not establish this implementation's numerical quality. The [vLLM Flash-Next patch](https://github.com/andreasgru/vllm-qwen38-flash-next-kv-quant) implements NVIDIA-specific FP8/NVFP4. The [Qwegn pack](https://huggingface.co/qwegnai/Qwen3.8-Flash-Next-hq8g32-hq4g64) is a different weight format and runtime. No external implementation was copied. [MLX Swift TurboQuant](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/kv-cache-quantization.md) initially retains raw prompt KV. [KIVI](https://proceedings.mlr.press/v235/liu24bz.html), [KVQuant](https://arxiv.org/abs/2401.18079), [SKVQ](https://arxiv.org/abs/2405.06219), [GEAR](https://proceedings.mlr.press/v262/kang24a.html), and [SAW-INT4](https://arxiv.org/abs/2604.19157) motivate sensitivity and rotation experiments after paired checkpoint data. No calibrated/rotated mode or active SSD paging was promoted or implemented; paging remains a fallback if low-bit candidates fail qualification.

## Real checkpoint comparison

One sequential BF16-then-tiered4 pair used the same 4,097-token prompt and 32-token greedy decode on the installed checkpoint. The only memory stop was 3,072 MiB of total swap. Power and thermal checks stayed active. The earlier staged startup checks refused at 6.48 vs 9.52 GiB and 7.82 vs 9.02 GiB of required headroom.

| KV cache | Prefill tokens/s | Decode tokens/s | Peak MLX active (MiB) | Peak swap used (MiB) |
|---|---:|---:|---:|---:|
| BF16 | 45.23 | 2.17 | 5,710.7 | 902 |
| Tiered INT4 / INT8 | 39.07 | 2.04 | 5,563.5 | 902 |

The 32 generated token IDs matched. At this length, the recent INT8 tier covers all attended rows, so the pair does not measure retrieval from aged INT4 rows. This is a single timing pair.

## macOS app and harnesses

In the app, choose **Model → Flash-Next KV cache → Tiered INT4 / INT8 (experimental)**. Press **Start** or **Switch to Selected Model**. The Model and Agents views show the precision reported by the running worker. The existing model folder and installation ID serve both precisions, so Hermes, Qwen Code, DeepSeek Harness, and other API clients keep the same model ID and base URL. The app starts Flash-Next with the 3 GiB absolute swap policy.

The bundled router loaded the real checkpoint in tiered4 and confirmed the 3,072 MiB stop with zero margin in `/health`. It passed requests shaped like Hermes, Qwen Code and DeepSeek Harness, plus generic Chat Completions, OpenAI Responses streaming and Anthropic Messages streaming. These were API-level checks. The app and its routing tests passed 65 Swift tests and 75 focused Python tests.

For a matching terminal launch, set `FLASHNEXT_MEMORY_POLICY=absolute-swap-only`. The prefill benchmark and context ladder accept `--swap-ceiling-only`. The context ladder checks that the server reports a 3,072 MiB ceiling with a zero margin before sending prompts.

## Snapshot and diagnostics behavior

The AGENTIC cache stores exact token IDs, checkpoint/engine and precision identity, RoPE state, plan provenance and raw indexer/GDN state. Mixed snapshots store packed primary and indexer deltas. The leaf supplies the complete recent INT8 window and recurrent state. A restore joins packed rows and does not expand history to BF16. The existing checksum, atomic manifest, lease, quota and plan checks apply. Tests exercise a tiny model across an on-disk reopen, including exact continuation logits, packed state and recurrence. The named local chat session format still supports BF16 only; use AGENTIC prefix snapshots for compressed state. A change in precision, layer overrides, or kernel invalidates reuse. The engine source identity also changes across builds. A compacted or reformatted transcript can only use a matching token/plan prefix; the server applies its normal cold-admission ceiling on a miss. Replay a bounded prefix or resume from the original transcript if rebuilding exceeds that limit.

`GET /v1/cache/diagnostics` exposes Flash-Next live attention offsets, logical K/V, scale/bias, indexer, GDN and weight bytes, the mapped expert-pool extent, MLX active/cached/peak counters, and the last gather shape. The existing endpoint reports RAM/SSD snapshots and swap growth; prefill flight-recorder events report scratch mapping and ring-view sizes. Logical tensor categories and mapped extents are attribution, not extra memory to add to allocator totals or RSS. Metal-wrapped expert mappings already contribute to MLX active allocations; the ring view may alias the same mapping. MLX peak is active allocation, not a macOS physical-footprint peak.

## Measured synthetic Metal probe

The probe uses checkpoint geometry for **one** attention layer, incrementally writes random BF16 activations in 2K steps, and times a tile of eight selected queries. No embeddings, model layers, expert I/O, or model decoding are timed. These timings are diagnostics, not end-to-end throughput or a controlled speed ratio. The peak column is the highest observed MLX active reading in that stage, including append and the attention probe. Guard sampling reported zero swap growth at all completed synthetic stages.

| Tiered G64 rows | One-layer cache (MiB) | Highest MLX active reading (MiB) | Eight-query tile (ms) | Projected model live state (GiB) |
|---:|---:|---:|---:|---:|
| 4,096 | 7.55 | 63.8 | 5.98 | 0.196 |
| 8,192 | 15.09 | 71.9 | 3.20 | 0.284 |
| 16,384 | 21.69 | 121.3 | 5.84 | 0.362 |
| 32,768 | 34.88 | 137.0 | 12.52 | 0.516 |
| 65,536 | 61.25 | 168.5 | 9.07 | 0.825 |
| 131,072 | 114.00 | 256.1 | 7.17 | 1.443 |
| 196,608 | 166.75 | 361.5 | 6.49 | 2.062 |
| 262,144 | 219.50 | 467.0 | 7.64 | 2.680 |

| 32K mode, G64 | One-layer cache (MiB) | Highest MLX active reading (MiB) | Append rows/s | Eight-query tile (ms) |
|---|---:|---:|---:|---:|
| bf16 | 72.38 | 157.7 | 476,593 | 3.85 |
| affine8 | 42.38 | 103.0 | 619,715 | 3.38 |
| k16v4 | 49.38 | 111.3 | 716,315 | 3.43 |
| k8v4 | 34.38 | 91.0 | 760,264 | 2.70 |
| tiered4 | 34.88 | 137.0 | 458,109 | 12.52 |
| affine4 | 26.38 | 79.0 | 795,868 | 2.77 |

Group 32 also completed the synthetic ladder: at 262,144 rows, one-layer logical storage was 236.00 MiB, the highest MLX active reading was 501.0 MiB, and the 12-layer-plus-GDN projection was 2.873 GiB.

Run the probe below to reproduce the measurements.

## Reproduce and roll back

From the repository, run the read-only gate before launching a real model. Connect AC power, keep Low Power Mode off, and use one model process. Stop on the first guard or quality failure and repeat only that stage after diagnosing it.

```sh
.venv/bin/python -m models.flashnext.prefill_readiness
.venv/bin/python -m models.flashnext.tests.bench.bench_hifi_kv --mode tiered4 --group 64
```

Once readiness passes, try the opt-in tiered profile with separate cache storage. Start at a 4K qualification target before passing a higher `--max-stage` to the driver. The server context setting is only the admission ceiling.

```sh
VEYSTRIX_API_KEY=local ./chat.sh --mode agentic --enable kv-tiered4 --port 8765 \
  --context-limit 262144 --max-cold-prefill 4096 --max-incremental-prefill 2048 \
  --pool-gb 0 --threshold 1.0 --no-auto-purge --prefix-dir "$HOME/.flashnext/prefix-hifi-kv"
```

`FLASHNEXT_KV_GROUP_SIZE=32`, `FLASHNEXT_KV_RECENT_TOKENS=8192` (a multiple of 256), and `FLASHNEXT_QSA_GATHER_TILE=8` (1–32) tune the experiment at launch. `FLASHNEXT_KV_LAYER_MODES='{"3":"bf16","47":"k8v4"}'` overrides absolute zero-based QSA layer indices; recurrence layers are rejected. Use `--enable qsa-gather` alone to isolate the QSA change with BF16, or combine it with `kv-affine8` / `kv-affine4` for a uniform low-bit comparator. For local chat, use `./chat.sh --mode chat --enable kv-tiered4`; mode selection is fixed for the process. Gather and mixed profiles use strict staged memory guards. Server clients use their existing OpenAI/Anthropic protocol and model ID; no harness files are changed.

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_context_ladder \
  --url http://127.0.0.1:8765 --checkpoint "$HOME/Models/Qwen3.8-Flash-Next-MLX-4bit-MTP" \
  --max-stage 4096 > context-ladder-4k.jsonl
```

The driver stops on a refusal, unsafe host sample, missing cache reuse, wrong repository retrieval result, or a minimum live QSA offset below the stage target. It records requests and timings as they complete. Use successive explicit targets `8192`, `16384`, `32768`, `65536`, `131072`, `196608`, `262144` only after the previous run passes; each invocation begins with a fresh transcript and runs the preceding steps in order. The retrieval check is a small deterministic cross-file calculation, not the full coding suite.

For paired coding and tool tests, stop the server and run the existing guarded `bench_prefill_memory` suite with the same checkpoint, planner, sampling and expert settings in each fresh process. Use `F2Bnr` (same-plan cold reference) and `--suite full --conditions F2Bnr --sizes 4k --rounds 1 --expert-pool-gb 0 --final` for the first pair, with the explicit checkpoint path. Run the BF16 reference first (`FLASHNEXT_KV_CACHE_DTYPE=bf16 FLASHNEXT_QSA_KERNEL=reference`), then BF16 gather, followed by each compression arm (`FLASHNEXT_KV_CACHE_DTYPE=tiered4 FLASHNEXT_QSA_KERNEL=gather`, for example). Compare the resulting arm JSON files using `python -m models.flashnext.tests.bench.quality_compare REF.json CANDIDATE.json --relation kv-precision`. Use the same-plan state digest for restored vs unrestored and the numerical diagnostics for logits, margins, routing and boundaries. The full existing suite exercises executable Python and protocol checks. Multi-file JS/TS and Rust/C++, sustained tool-result chains, and coding dependent on 192K/262K evidence still need paired real-model runs before promotion. One round is for quality/diagnostics; use a reversed multi-round schedule for performance claims.

To roll back, stop the candidate process and launch the existing BF16 reference profile without KV flags. Keep the same strict guard and bounded request caps for a fair comparison:

```sh
FLASHNEXT_MEMORY_POLICY=strict-staged VEYSTRIX_API_KEY=local ./chat.sh --mode agentic \
  --port 8765 --context-limit 262144 --max-cold-prefill 4096 --max-incremental-prefill 2048 \
  --pool-gb 0 --threshold 1.0 --no-auto-purge
```

## Delivery status

- Implemented and tested on Metal: gather attention, mixed and tiered storage, per-layer settings, versioned RAM/SSD prefix state, diagnostics, synthetic benchmark, and the guarded ladder driver.
- 454 focused Flash-Next and protocol tests passed during initial development. The real checkpoint completed one 4K comparison. The incremental ladder and broader coding-quality qualification remain pending.
- Existing uncommitted work was preserved in `unlimited-generation` (starting HEAD `4a6e745`). No branch switch, commit, push, checkpoint change, harness rewrite, or Nemotron backend change was made.

Files changed for this work:

- `models/flashnext/{qsa_chunk,mixed_kv,quantized_kv,loader}.py`: gather attention, compressed storage and load-time validation.
- `models/flashnext/{prefix_cache,prefix_session,sessions}.py`: compatible packed prefix capture/restore and explicit named-session format checks.
- `models/flashnext/{kv_telemetry,prefill_ring,settings/modes}.py`, `veystrix/backends/flashnext.py`, `veystrix/server.py`: settings, guarded launch and diagnostics.
- `models/flashnext/tests/bench/{bench_context_ladder,bench_hifi_kv,bench_prefill_memory,quality_suite,quality_compare}.py`: incremental validation, Metal probe and paired comparisons.
- `models/flashnext/tests/unit/{test_hifi_kv,test_modes,test_quality_suite}.py`: selected-set, continuation, compatibility and qualification checks.
- This report and `docs/flashnext/quality-validation-policy.md`: methods, results and qualification criteria.
