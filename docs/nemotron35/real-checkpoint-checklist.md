# Real checkpoint validation checklist

The user authorized download of the mlx-community 4-bit checkpoint; do not load full model weights until the bounded runtime is implemented and explicitly authorized. When a compatible local checkpoint is later provided, inspect metadata without loading weights:

```sh
cd /Users/vincent/Documents/Veystrix-nemotron35
./chat.sh --model nemotron35-lightning --checkpoint /path/to/model --check
```

A successful check **does not mean inference is supported**. The official BF16 or NVFP4 assets must not be assumed to be MLX affine-4-bit. Obtain a legally permitted, verifiably converted MLX format separately only after a real loader and conversion provenance checks are implemented. The CLI refuses inference today; there is no working Nemotron launch or model-switch command yet. The existing Flash-Next commands remain unchanged, e.g. `./serve.sh`.

Before promotion: review NVIDIA Open Model / OpenMDW 1.1 license, model card, MLX/third-party notices and each converted checkpoint's terms; validate exact tensor names/shapes/quantization, full routing and shared experts against authoritative references; prove snapshot identity and restart continuation; measure dense-core, KV, Mamba, pool and peak temporary allocation; validate guard/cancellation behavior; test Hermes streamed multi-turn tool calls; then run the paired benchmark plan with explicit user authorization. Do not set Nemotron as default until correctness, safe multi-hour use and measured performance have passed.
