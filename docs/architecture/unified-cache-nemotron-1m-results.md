# Shared cache and Nemotron context results

## Checkout and migration

The work used the local `unlimited-generation` branch on the 16 GiB Apple
Silicon Mac. The original Qwen3.8 Flash-Next MLX 4-bit MTP and Nemotron 3.5
Lightning 30B-A3B MLX 4-bit checkpoints were available under `~/models`.
The installed workspace runtime had MLX 0.32.2 and native Metal access.
The sandboxed runtime could not access a Metal device, so native tests and
benchmarks ran with the existing host safeguards outside the sandbox.

| Legacy path | Verified and deleted pairs | Accounted bytes removed |
| --- | ---: | ---: |
| `~/.flashnext/prefix-cache` | 239 | 35,092,811,541 |
| `~/.cache/veystrix/nemotron35/prefix` | 52 | 10,213,458,072 |

The [inventory](legacy-cache-inventory.json) preceded deletion. The
[deletion result](legacy-cache-deletion.json) records the Qwen ownership
lock that was preserved. The app had no custom cache location configured.
The running Python process was Blender's MCP server. No cache file handles
or inference-server listeners were detected before deletion. The new
`~/.cache/veystrix/prefix-cache-v1` initially reported zero snapshots and
an 80,000,000,000-byte quota with a 50,000,000,000-byte reserve.

## Cache architecture

`veystrix/universal_cache.py` owns accounting, configuration and eviction.
A process-shared lock protects scans, reads, writes and reclaim. Committed
files provide the snapshot registry. The small atomic JSON registry stores
quota, hits, last reuse, pins and the latest cleanup outcome. A new root
takes its initial quota from the launch environment or app settings.
Subsequent quota changes are persisted by the same manager for both models.
The configured quota is distinct from SSD free space, RAM snapshot budget,
expert residency and checkpoint storage.

Eviction ranks eligible leaves using token count as a recomputation proxy,
hit count, recency and bytes. A surviving Qwen delta protects its ancestors.
Active Qwen chains and Nemotron recovery checkpoints use shared file
leases. Profile pins are limited to half the global quota when set. If
protected entries prevent a quota reduction, the current quota stays in
effect. Cache statistics are counts of known files and their logical sizes.
APFS snapshots and clones can affect physical space returned by deletion.

Qwen retains its exact-token and chunk-plan checks, native or configured
KV representation, and GatedDeltaNet state. Managed Qwen restores hash and
load through the same open descriptor under the cache lock, saving the
temporary verification copy. Nemotron keeps its model-specific codec,
including the exact token tape and attention offset, checked tensor data,
convolution state and SSM state. Checkpoint, tokenizer, numerical mode and
chunk configuration participate in the compatibility fingerprint.

The app invokes the model-free manager in a background task. The engine
can be stopped while inspecting storage or reclaiming snapshots. The
settings view reports per-model counts, usage, hits, misses and restored
tokens. Reclaim previews are recalculated at deletion time. Changing
Nemotron's configured context takes effect on restart.

## Nemotron attention and admission

The installed Nemotron-H attention implementation has no rotary or
learned positional embeddings. NVIDIA validated the architecture at
1,048,576 tokens on its [vLLM serving configuration](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).
Its [FP8 recommendation](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
describes its own runtime. MLX's [fused attention API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
takes floating-point arrays. The Veystrix path uses affine INT8 packing
with groups of 64 and per-group scale and bias.

The cache appends to 4,096-token blocks. At short contexts a single block
can use fused MLX attention. At longer contexts the engine visits all KV
blocks in order, computes causal scores in 128-query tiles and combines
block maxima and denominators in FP32. No sliding-window truncation is
applied. Existing Mamba state arithmetic is preserved. The theoretical
1M attention payload is 3.1875 GiB with BF16 scale and bias, including
3 GiB of packed values. This estimate excludes the dense core, expert
pool, recurrent state, activations, allocator reserve and attention scratch.

The final default logical window is 1,048,576 tokens. Each cold prefill or
increment may process up to 65,536 uncached tokens. The server and app
expose the admitted context and mark million-token inference validation
as pending. Memory pressure, swap, cancellation, power and watchdog checks
remain active. The final limits were checked without running a 64K or 1M
prefill.

## Real-checkpoint measurements

These are one before/after pair per workload, with models loaded
sequentially and separate cache roots for the baseline and candidate.
The machine was on AC power. The Qwen workload used a repository excerpt,
exact routing, uncached expert reads, pool off and 32 generated tokens.
The warm Qwen arm used a new engine process and a different user message
with the same system prefix.

| Qwen metric | Baseline cold | Shared cache cold | Baseline warm | Shared cache warm |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens | 2,569 | 2,569 | 2,569 | 2,569 |
| Restored tokens | 0 | 0 | 2,107 | 2,107 |
| Prefill elapsed, seconds | 72.84 | 72.44 | 35.17 | 35.11 |
| Uncached prefill, tokens/s | 35.27 | 35.46 | 13.14 | 13.16 |
| SSD restore, seconds | — | — | 0.319 | 0.307 |
| Decode, tokens/s | 1.46 | 1.37 | 1.04 | 1.43 |
| Expert SSD reads through decode, GB | 197.90 | 197.90 | 98.59 | 98.59 |
| Prefill Metal peak, GB | 5.77 | 5.77 | 5.14 | 5.14 |
| Peak total system swap, MiB | 959 | 959 | 1,019 | 959 |

The before and after Qwen traces match for all 32 generated token IDs,
final logits and snapshot state digests in each arm. The comparison shows
no reliable cold throughput or decode improvement. Streaming TTFT and
peak RSS were unavailable in these traces. This pair predates the final
same-descriptor restore optimization. In a separate six-pair test on the
same 289.7 MB of real snapshot files, median checksum-and-load time fell
from 168 ms to 139 ms. That test excludes chain reconstruction and model
execution. The optimization also eliminates the temporary payload write.

Nemotron used the same 4-bit weights, a 1 GiB expert pool and two coding
requests sharing a 969-token system prompt. The baseline did not save that
boundary, so its second request recomputed the whole prompt. The INT8
candidate saved and restored the compatible boundary. The coding requests
were capped at 32 output tokens, which demonstrated relevant completions
but was too short for a full coding-quality comparison.

| Nemotron metric | BF16 cold | INT8 cold | BF16 second request | INT8 warm |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens | 1,009 | 1,009 | 1,010 | 1,010 |
| Restored tokens | 0 | 0 | 0 | 969 |
| Prefill elapsed, seconds | 8.46 | 11.90 | 7.97 | 3.20 |
| Uncached prefill, tokens/s | 119.25 | 84.82 | 126.80 | 12.81 |
| Time to first token, seconds | 9.30 | 12.36 | 8.46 | 3.73 |
| SSD restore, seconds | — | — | — | 0.046 |
| Decode, tokens/s | 4.62 | 4.57 | 4.44 | 5.35 |
| Expert reader bytes, per request, GB | 36.11 | 45.10 | 35.70 | 28.87 |
| Expert pool hit rate, cumulative | 37.7% | 37.9% | 36.5% | 37.4% |
| Metal peak, process, GB | — | 4.32 | — | 4.32 |
| RSS peak, process, GB | — | 3.04 | — | 3.04 |
| Peak total system swap, MiB | 1,020 | 1,020 | 1,020 | 1,020 |

Nemotron's new system checkpoint splits an otherwise single prefill on the
cold run, increasing expert reads and TTFT in this workload. On reuse it
eliminates most of the prefill and lowers TTFT by about 56%. The decode
measurements do not establish a general improvement. The blocked attention
unit test compared INT8 against BF16 causal attention across block and
query boundaries, with a maximum allowed absolute error of 0.025. It also
verified exact restore of the quantized representation and FP32 recurrent
state. Full coding and tool-use equivalence remains unverified on real
weights.

The [measurement directory](../../results/unified-cache-20261111/) holds
the before/after data, supervisor logs, and the long-run request events.

## Context validation on this Mac

| Item | Result |
| --- | --- |
| Checkpoint positional metadata | 262,144 |
| NVIDIA validated architectural capability | 1,048,576 in its vLLM setup |
| Configured default | 1,048,576 |
| Uncached prefill limit per request | 65,536 |
| Real startup admission with final defaults | 1,048,576 |
| Largest completed INT8 prompt in this run | 1,010 tokens |
| Long retrieval attempt | 14,641 input tokens, stopped after advancing 14,606 |
| 1M retrieval and continuation | Unverified |

After setting the final defaults, a supervised real-checkpoint startup
confirmed 1,048,576 tokens through both `/health` and `/v1/models`.
Both endpoints reported 65,536 uncached tokens per request. The watchdog
was running and the test submitted zero prompts before stopping the engine.
See the [startup record](../../results/unified-cache-20261111/nemotron-final-no-prompt.json)
and [metadata-only admission calculation](../../results/unified-cache-20261111/nemotron-final-admission-only.json).

The long retrieval prompt's measured token count was 14,641 after
tokenization. The existing guard stopped it after swap grew by 1.42 GiB,
exceeding the run's 1 GiB growth limit. The process cancelled cleanly with
exit code 75. Total swap was about 2.42 GiB immediately after the stop.
The run produced no retrieval answer. No second long run was attempted
under that pressure. A real 32K-to-1M retrieval sweep, complete coding and
tool-use checks on real weights, and repeated throughput pairs remain open.

## macOS build

The repository's pinned engine build and macOS release scripts produced an
ad-hoc signed local build. The app opened with the engine stopped and
loaded live cache status from its bundled runtime. The Settings preview
reported zero eligible snapshots on the empty cache, and the clear
confirmation appeared and was cancelled. Both real checkpoint workloads
used the guarded CLI rather than starting a heavyweight model from the
packaged UI after the swap stop.

| Artifact | Path under the checkout | Size |
| --- | --- | ---: |
| App bundle | `dist/unified-cache/Veystrix.app` | 688 MiB on disk |
| Executable | `dist/unified-cache/Veystrix.app/Contents/MacOS/Veystrix` | 2,933,792 bytes |
| Disk image | `dist/unified-cache/Veystrix-3.8.1-arm64.dmg` | 402,880,493 bytes |
| ZIP | `dist/unified-cache/Veystrix-3.8.1-arm64.zip` | 241,962,133 bytes |

The app signature, bundled cache module, bundled INT8 module and offline
cache status command were verified. SHA-256, ZIP integrity and disk image
checks passed. The app build is local and has no notarization ticket.

## Regression tests and code map

| Suite | Result |
| --- | --- |
| Veystrix Python core | 823 collected, 815 passed, 8 skipped |
| Flash-Next Python unit suite | 1,856 collected, 1,853 passed, 3 skipped |
| Nemotron Python unit suite | 109 collected, 101 passed, 8 skipped |
| macOS Swift tests | 65 passed |

The Python suites cover shared quota, file accounting, exact prefix
matching, fingerprint isolation, external process leases, pins,
ancestry, restart, corruption, interrupted writes and cancellation.
They also cover OpenAI and Anthropic request handling with model-free
or tiny-checkpoint tests. Installed third-party harnesses were not all
exercised against both real checkpoints in this run.

- `veystrix/universal_cache.py` and `veystrix/cache_migration.py` provide
  the shared registry, administration and legacy inventory.
- `models/flashnext/prefix_cache.py`, `prefix_profiles.py` and
  `prefix_session.py` integrate Qwen's original codec and chunk-plan checks.
- `models/nemotron35/attention.py`, `state.py`, `backend_core.py`,
  `prefix_cache.py`, `admission.py`, `memory.py` and `readiness.py` handle
  blocked attention, persistence and memory estimates. The Nemotron backend
  records the system boundary and exposes admission and runtime metrics.
- `veystrix/cli.py`, `veystrix/cache_tools.py`, `veystrix/server.py` and
  the supervisor forward shared configuration and report the admitted context.
- The Swift package includes `CacheManagement.swift`, `CacheSettings.swift`
  and updates to the existing Settings, launch plan and health models.

The final limit change was rechecked with 29 focused core tests, the
109-test Nemotron unit suite and all 65 Swift tests. The startup check
above used metadata requests only.

## Follow-up throughput work

The later [Nemotron short benchmarks](../../results/nemotron-short-20261111/)
measured short system-anchored prefill with expert reads shared across
the system and user chunks. Greedy cold prefill went from 11.98 to
9.12 seconds and decode after the cold prompt went from 4.49 to 5.02
tokens/s. Warm decode was approximately flat.

The [Qwen short benchmark](../../results/qwen-short-20261111/) measured
72.91 to 33.22 seconds of cold prefill and 2.22 to 2.78 tokens/s decode
with the experimental short-throughput profile. The profile uses more
memory and remains opt-in. The follow-up prompts ranged from 1,009 to
2,569 input tokens.
