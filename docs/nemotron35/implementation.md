# Implementation status and audit

Status: **experimental.** The streamed backend exists, is integrated into the shared CLI, session, server, supervisor and guards,
and passes model-free tests on tiny synthetic checkpoints in the exact real tensor layout. **No inference has been run on the
real checkpoint**, so correctness, memory and speed on real weights are unknown. Qwen (Flash-Next) remains the default; there is
no automatic routing. Real-weight loading refuses to start unless `VEYSTRIX_NEMOTRON_AUTHORIZED=1` is set.

## What exists

| Area | Module | Notes |
| --- | --- | --- |
| Checkpoint verification | `models/nemotron35/inventory.py`, `config.py`, `check.py` | Exact 729-tensor affine-4-bit inventory derived from `config.json`; shard header, extent (no truncation, no trailing bytes), index `total_size`, shard-name/symlink safety, partial-download refusal, optional SHA-256 against the digest recorded by huggingface_hub. NVFP4 and non-affine refused. |
| Streamed experts | `streamed_experts.py` | Validated stacked `switch_mlp.fc1/fc2` slice reader (`pread` of the six per-expert slices), byte-bounded LRU pool shared by all layers, exact top-6 routed experts grouped by unique expert, per-expert `mx.eval` so eviction releases memory, cancellation between reads. Shared expert and router are resident dense weights. |
| Model loader | `streamed_model.py` | Uses installed mlx-lm `nemotron_h.Model` with `SwitchMLP` swapped for the streamed module, so parameter names are identical to the checkpoint. Dense core read resident (about 1.17 GiB); routed experts never loaded. |
| Admission | `admission.py` | Plans context, chunk and pool against `min(Metal limit, RAM minus OS reserve, available)`; refuses when the dense core cannot fit; per-request gate (pressure, 2.5 GiB swap line, context). Activation bytes are a conservative **estimate**. |
| Engine core and backend | `backend_core.py`, `veystrix/backends/nemotron35.py` | Chunked prefill (last-position logits only), streaming decode, sampler, stop tokens 2 and 11, think budget, cancellation (mid-prefill forces clean replay; between tokens state stays valid), stats incl. pool hits and expert bytes. Chat framing matches `chat_template.jinja` (tested with the real tokenizer). |
| Persistence | `state.py` + backend | Named sessions namespaced by a model identity hash (config, tokenizer, template, index size, quantization, runtime version); fingerprint over identity and tokens, tensor SHA-256, atomic write; corrupt or foreign snapshots refused. bf16 digest bug fixed. |
| Integration | `veystrix/session.py`, `cli.py`, `checkpoints.py`, `server.py`, `models/flashnext/overnight.py`, `supervision.py` | `--model nemotron35-lightning`; power gate, chat guard (swap/thermal/battery) and supervised-server rule apply; verified-checkpoint discovery; `/v1/models` slug `veystrix-nemotron35-lightning`; supervisor `--model nemotron35-lightning` (auto-warm off). |

## Not implemented / unverified

- Real-checkpoint inference of any kind (numerical match, tokens/s, memory peak, swap, thermals).
- Asynchronous I/O, prefetch and GPU-resident expert pool tuning; reads are synchronous. About 5.6 MB per expert per layer.
- Automatic prefix/persistent cache across requests (server `_prefix_session` is Flash-Next only). Named sessions exist; per-turn automatic snapshots do not. A prompt whose template-rendered history differs from the tape (the template rewrites historical thinking) cold-replays.
- Desktop app (Swift) model picker, validator and launch plan: not changed.
- MTP: not implemented.
- Hosted CI for this branch has not run.
