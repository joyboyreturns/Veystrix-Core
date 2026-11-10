# Implementation plan and promotion gates

## Product priorities

Nemotron is the intended **daily** Hermes model after real-machine verification; Flash-Next is the **advanced** model and remains the working default until then. Both must ultimately use SSD-backed MLX inference. Manual model selection, separate cache namespaces, no model co-residency on a 16 GB machine by default, and no automatic tool replay are requirements.

## Ordered work

1. **Reference and safety (in progress).** Record exact NVIDIA BF16/NVFP4 metadata, layer order, tensor naming, routing, Mamba and attention equations, model and software licenses. The synthetic fixture validates parts of installed mlx-lm; it is not a real-weight reference. Harden configuration, snapshot and record bounds before accepting untrusted data.
2. **Dense core and memory feasibility.** Measure actual non-expert allocations after a compatible checkpoint is provided. Enforce an explicit budget including Mamba recurrent/convolution state, KV, temporary activation, GPU working buffers, SSD pool and OS reserve. Design bounded offload if core cannot fit. Never use the full in-memory synthetic reference for 30B inference.
3. **Exact streaming.** Design model-aware expert index and 4-bit affine format, validate offsets, shapes, scales and biases, and preserve all six routed experts and one shared expert. Adapt generic low-level Flash-Next facilities without changing their production paths. The current synthetic reader is synchronous, caches host bytes and makes copies; it is not the production implementation. Add bounded GPU pool, cancellation, prefetch and asynchronous I/O only after correctness and measured benefit.
4. **Backend and sessions.** Implement a full shared backend adapter with model-specific tokenizer, deterministic chunked prefill, decode and cancellation. Validate Mamba and attention state ownership, fingerprinted persistence, corruption rejection, session isolation and safe resource release on manual switch. No MTP until verifier correctness and performance are measured.
5. **Hermes and API.** Register `veystrix-nemotron35-lightning` only once the backend works. Verify `/v1/models`, streaming and non-streaming chat, exact JSON tools, tool results, multiple turns, cancellation, context admission and usage with synthetic execution. Hermes owns tool execution. Maintain the loopback endpoint and local API-key controls.
6. **Two-model cleanup.** Audit all imports before removing Bonsai-2, K2-Horizon and Qwen27B; preserve any shared components Flash-Next needs and all required notices. Remove legacy tests and documentation only after replacing or extracting dependencies and passing the Flash-Next regression suite. This remains incomplete.
7. **Promotion.** On actual checkpoint availability, run numerical comparisons and paired benchmarks with explicit consent and `docs/measurement-standard.md`. Targets are 120–150+ tokens/s cold 8K prefill, 8–10+ sustained decode, 12+ stretch and at least 1.5× quality-correct Flash-Next. These are targets only; retain Qwen as default if correctness, memory, thermals, tools, sustained sessions or speed fail.

## Current gates

- No real-checkpoint loader, no usable Nemotron launch, no Hermes real-model flow, no real performance or memory measurements.
- No production-ready persistent snapshots or model-switch lifecycle.
- Existing legacy models are not yet removed.
- Synthetic MLX and streamed-expert tests pass, but are not evidence of real-checkpoint numerical equivalence.
