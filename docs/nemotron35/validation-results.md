# Validation results

No actual Nemotron weights or Flash-Next weights were downloaded or loaded. The installed MLX runtime executes a deterministic tiny 3-layer hybrid model and synthetic affine-4-bit SSD expert records, including a saved-and-reloaded local synthetic checkpoint and token-level streaming adapter. Tests cover hybrid full versus token and chunk prefill, reference SSM recurrence, grouped-query KV attention, top-k routing and shared expert combination, a real hybrid layer using streamed synthetic experts, affine 4-bit reconstruction, bounded host LRU/hits/integrity, safetensors shard-header and index consistency checks, indexed per-expert affine 4-bit loads and bounded logical cache, synthetic state snapshot exact continuation and corruption/prefix rejection, cancellation reset, memory estimates and missing checkpoint.

On this Mac the synthetic suite passed 26 tests; Flash-Next guard and server-protocol regression tests passed 47. An initial synthetic fixture used state width 4, which violates the installed single-token Metal SSM kernel's 32-wide requirement and produced a mismatch. A valid width 32 fixture passes at `rtol=1e-4, atol=1e-4`; this does not establish full-model precision or routing equivalence. An initial regression invocation included `test_power_gate`, absent on local main, and failed with one import error; rerunning the two present suites passed.

There are no real-checkpoint numerical results, SSD bandwidth measurements, benchmark results, Hermes real-model tool calls, long-running reliability results, or production-readiness claims. The generic record reader and synthetic state format have not received the robustness review required for untrusted checkpoints.

The latest existing Qwen branch CI run (`42879656444`) reported two launcher-test failures caused by a fake `/repo` without the newly required `.git` marker, plus one keepalive test relying on timing rather than synchronization. The local Qwen worktree and this local branch now use temporary checkout markers and an event-driven keepalive test. Both targeted launcher/server suites pass locally; the hosted workflow has not been rerun, since no push is permitted.

The Qwen worktree’s full `veystrix/tests` discovery subsequently passed 779 tests (55 skipped) with a temporary home and offline mode. Flash-Next unit discovery passed 1,836 tests (8 skipped). The current workflow’s remaining Qwen27B, K2-Horizon and Bonsai-2 suites passed 22, 58 and 175 tests respectively (5 Bonsai-2 skips). Hosted CI has not been rerun.

Per-token versus batched float32 quantized matrix multiplication showed a maximum relative discrepancy about 1.6e-7 in the tiny fixture; the test uses rtol=1e-6, atol=1e-6 for this different reduction order. This is not a real-weight comparison.

After integrating the Qwen branch locally, a 26-test synthetic suite plus launcher and server tests passed. The 4-bit checkpoint metadata is affine/group 64; the shard download and all real-weight inference remain incomplete.


## 2026-11-11 update: real-layout path

Evidence gathered **without loading the real model**:

- The downloaded mlx-community checkpoint (4 shards, 17,775,339,392 bytes) passes `python -m models.nemotron35.check <dir> --hashes`: 729 tensors match the exact expected inventory (names, shapes, dtypes), shard extents are exact, the index `total_size` equals the tensor bytes, and all four SHA-256 digests match the digests recorded by huggingface_hub. This proves layout and download integrity only.
- At full size, the mlx-lm `nemotron_h.Model` built with the streamed `SwitchMLP` replacement has exactly the dense parameter names and shapes of the checkpoint (lazy arrays, nothing allocated).
- On tiny synthetic checkpoints in the identical stacked layout, the streamed model matches the in-memory quantized reference: prefill within bf16 rounding, cached decode exact, chunked equals single pass, pool within budget even at size 0.
- Chat framing equals `chat_template.jinja` rendering with the real tokenizer (open, non-thinking prompt, tool results, turn join, hostile content cannot forge structure); the shared server parses the model's `<tool_call>` format.

Not evidenced: real numerical equivalence with NVIDIA's weights, expert-read throughput, memory peaks, swap, thermals, decode or prefill speed, long-context behavior, Hermes sessions with real model output.
