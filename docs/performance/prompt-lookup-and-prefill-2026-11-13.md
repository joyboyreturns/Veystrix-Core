# Prompt lookup and prefill experiments, 2026-11-13

The component timings below use tiny or synthetic fixtures on the M4.
Each timing is the median of six alternating pairs. The live checkpoint
comparison appears below.

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Tiny Flash-Next model, 24 greedy tokens, history seeded with the expected continuation | 31.76 ms, 24 forwards | 18.94 ms, 11 forwards |
| Capture and handoff of a 721,514-byte Flash-Next snapshot | 1.290 ms | 0.089 ms |
| Same snapshot, including durable disk commit | 1.290 ms | 1.296 ms |

The prompt lookup fixture deliberately favors copying. The timed arms
produced identical tokens. Separate synthetic checks cover draft acceptance,
rollback, and the next continuation. Set `FLASHNEXT_PROMPT_LOOKUP=1` for
greedy, full-routing BF16 KV turns without separate reasoning budgets.
The decoder samples plain throughput and pauses verification when it loses.

`FLASHNEXT_PREFILL_SEED_DECODE=1` copies one frequently routed expert per
layer from completed prefill ring reads into a free decode slot. The
Q4/G32 fixture copied all nine parts of one 3,072,000-byte record and
served the next request for that expert as a pool hit. The option does
not allocate a pool or evict an existing expert to seed it.

`FLASHNEXT_PREFIX_ASYNC_WRITE=1` queues the last eligible snapshot of a
standard prefix-cache prefill. Earlier recovery checkpoints and profiled
snapshots stay synchronous. The background writer uses the existing
transaction, integrity checks, and disk quota. A snapshot is durable only
after that write finishes. Cache operations that need the same transaction
can still wait for the writer.

Nemotron exposes the same three features:

- `VEYSTRIX_NEMOTRON_PROMPT_LOOKUP=1` verifies at most four draft tokens.
  It runs singleton operations one layer at a time, so the token block
  can reuse that layer's expert weights while Mamba and KV state advance
  in their usual order. A rejection restores the previous state and
  replays the accepted prefix through ordinary decode.
- `VEYSTRIX_NEMOTRON_PREFILL_SEED_DECODE=1` promotes one hot expert from
  each completed layer-major prefill layer into spare decode-cache space.
  It reuses existing array handles. Chunk-major prefill already admits
  demand reads to the decode cache.
- `VEYSTRIX_NEMOTRON_PREFIX_ASYNC_WRITE=1` reserves up to 64 MiB in the
  admission plan and moves eligible terminal snapshot writes off the
  foreground path. Larger snapshots and intermediate recovery
  checkpoints use the ordinary writer.

`FLASHNEXT_METAL_FMA_DEQUANT=1` and
`VEYSTRIX_NEMOTRON_METAL_FMA_DEQUANT=1` contract the affine Q4 dot-product
epilogue with explicit FMA, with the rest of the Metal compilation in safe
math mode. These options can change rounding, so cache identities encode the
choice. `VEYSTRIX_MLX_AUTO_BUFFER_CAPS=1` sets the MLX child process's
limits to 200 operations and 1,024 MiB per command buffer, unless already set.
All of the new options default to off.

## Real-checkpoint server comparison

Both arms used the same guarded HTTP path, greedy sampling, a fixed
repetitive code prompt, 24 generated tokens, and fresh server processes and
prefix directories. The candidate enabled all five options for the model.
The OS file cache was retained. These are diagnostic results for one prompt
and one Mac. A final process footprint is not a peak working-set measurement.

Flash-Next matched all 24 token IDs in all six arms. Median decode fell
from **2.12 to 1.74 tok/s**. The median paired effect was **−0.36 tok/s**,
outside the **0.065 tok/s** baseline resolution band. Median time to first
text increased from **23.88 to 26.05 seconds**. The foreground snapshot
call fell from a median of **80 ms to 1 ms**. The guard declined the decode
pool on these runs, so the result does not measure persistent-pool seeding.
See [the Flash-Next arms](../../results/flashnext/20261113-042302-performance-v2-manual/performance-arms.jsonl).

Nemotron matched all 24 token IDs in all six arms with INT8 KV. Median
decode rose from **3.95 to 4.38 tok/s** and the median paired gain was
**0.42 tok/s** against a **0.08 tok/s** resolution band. Median time to
first text rose from **7.28 to 8.68 seconds**, while total request time
fell from **12.67 to 12.33 seconds**. Median physical reads fell from
**26.55 to 22.31 GB**. The 576-token prompt stayed on chunk-major prefill,
so it did not exercise the new layer-major seeding or an aligned prefix
snapshot. See [the Nemotron arms](../../results/nemotron35/20261113-043013-performance-v2-manual/performance-arms.jsonl).

The `performance-v2` test is available for both models in the Veystrix
test terminal. For a longer code-review workload, run the same benchmark
module with `--performance-v2 --large-workload`. This workload uses a
fixed 2 GB decode pool and an independent absolute 5 GiB swap stop.
Flash-Next runs native MTP in both arms, so prompt lookup is off for that
comparison. The local Nemotron checkpoint advertises MTP in its model
config but has no MTP tensors in its weight index. Its candidate uses the
implemented prompt-lookup verifier instead.

The large Nemotron pair used 5,564 real project-code tokens and produced
the same 96 token IDs in both arms. Decode went from **4.64 to 3.36
tok/s**, first text was **24.19 vs. 24.21 seconds**, and total request
time went from **44.43 to 52.39 seconds**. The candidate seeded
**129,088,512 bytes** into the 2 GiB decode pool. The final prompt
position was not a reusable chunk edge, so this pair did not time the
background snapshot writer. One pair is diagnostic and does not
establish a resolution band.
See [the large Nemotron arms](../../results/nemotron35/20261113-043959-performance-v2-code-review-manual/performance-arms.jsonl).

Flash-Next's first verified native MTP attempt reached the 5 GiB swap
stop during whole-prompt MTP warmup. The bounded warmup processes at
most 1,024 tokens at a time, retains only the target's final vocabulary
row, and skips the draft vocabulary projection for history tokens.
The failed arm is preserved in [the swap-stop record](../../results/flashnext/20261113-043839-performance-v2-code-review-manual/performance-arms.jsonl).

With bounded warmup, both large Flash-Next arms completed the
5,594-token prompt and produced the same 96 token IDs. Runtime
telemetry verified native MTP at depth 3 and the fixed 2 GB decode
mapping in both arms. Decode was **0.95 vs. 0.90 tok/s**, first text
was **191.41 vs. 198.81 seconds**, and total request time was **287.96
vs. 300.00 seconds**. The highest sampled swap use was **2,581 MiB**
in the baseline and **2,535 MiB** in the candidate. This single pair
establishes completion and token parity under the requested limit.
It does not establish a speed gain.
See [the large Flash-Next arms](../../results/flashnext/20261113-044219-performance-v2-code-review-manual/performance-arms.jsonl).
