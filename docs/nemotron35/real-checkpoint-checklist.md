# Real checkpoint validation checklist

The checkpoint is downloaded and structurally verified. Real use was approved by the owner on 2026-11-11; `VEYSTRIX_NEMOTRON_AUTHORIZED=0` disables it everywhere. When a compatible local checkpoint is later provided, inspect metadata without loading weights:

```sh
cd /Users/vincent/Documents/Veystrix-nemotron35
./chat.sh --model nemotron35-lightning --checkpoint /path/to/model --check
```

A successful check means the layout and (with `--hashes`) the download are intact; it **does not prove numerical correctness**. The official BF16 or NVFP4 assets must not be assumed to be MLX affine-4-bit. Obtain a legally permitted, verifiably converted MLX format separately only after a real loader and conversion provenance checks are implemented. Launch: `./chat.sh --model nemotron35-lightning --checkpoint DIR` or the supervisor `python -m models.flashnext.overnight --model nemotron35-lightning --checkpoint DIR`. The existing Flash-Next commands remain unchanged, e.g. `./serve.sh`.

Before promotion: review NVIDIA Open Model / OpenMDW 1.1 license, model card, MLX/third-party notices and each converted checkpoint's terms; validate exact tensor names/shapes/quantization, full routing and shared experts against authoritative references; prove snapshot identity and restart continuation; measure dense-core, KV, Mamba, pool and peak temporary allocation; validate guard/cancellation behavior; test Hermes streamed multi-turn tool calls; then run the paired benchmark plan with explicit user authorization. Do not set Nemotron as default until correctness, safe multi-hour use and measured performance have passed.
