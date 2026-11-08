# Veystrix

**Local Apple Silicon inference for SSD-streamed Mixture-of-Experts models.** Veystrix is an experimental continuation of [MACQWEN](https://github.com/1architect/macqwen-releases). Its current focus is running the Qwen3.8-Flash-Next MLX Q4 checkpoint on a 16 GB MacBook Air M4, with exact expert routing, bounded memory, reusable context, and eventually a coding-agent server.

> **Release-status warning (8 November 2026):** The `CHAT`/`AGENTIC` mode commands below refer to an **uncommitted local development working tree** and are **not yet available in the published GitHub source**. This documentation update is staged separately so the instructions can ship with the implementation. Do not expect a fresh clone of this repository to recognize `--mode` until the corresponding code is published.

## Current status

| Capability | Evidence | Status |
|---|---|---|
| Q4 checkpoint loading and token generation | Real 113 GB checkpoint on M4 MacBook Air | Demonstrated |
| Personal multi-turn CHAT with `--pool-gb 0` | Five completed responses; ~2.7–3.0 decode tokens/sec | **Works in local development tree** |
| CHAT with fixed 2 GB decode expert pool | Compressor/available-memory guard stopped first answer | Not recommended as an initial setting |
| CHAT auto-purge | Startup recovery executed; no OS purge during inference | Experimental |
| Balanced 2,048-token prefill | Historical 2K full-model run ~61–64 prefill tokens/sec | Preliminary, specific workload |
| Same-plan 2K prefix caching | 2,046-token RAM/SSD restores; identical 64-token continuation and 124 state tensors | Validated in isolated full-model experiments |
| AGENTIC API server | Server protocols exist; two-mode launch implemented only locally | **End-to-end harness integration unverified** |
| Agentic repeated tool calls and quality suite | Not yet run end-to-end on full checkpoint | Experimental |
| 4K / 8K / 10K long-context operation | Offline work only at larger sizes | Not verified |

Data are from a single machine and selected runs; they do not establish general performance or guarantee no quality regression.

## Model

The focus checkpoint is `Qwen3.8-Flash-Next-MLX-4bit-MTP`, approximately 113 GB on disk, using affine Q4 group-32 expert weights. Selected MoE experts are streamed from SSD; the whole checkpoint is **not** loaded into 16 GB RAM. The model weights, routing and quantization should remain unchanged by cache and scheduling optimizations.

Place an appropriately licensed, compatible checkpoint at:

`~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`

See the legacy [Flash-Next technical brief](docs/flashnext/brief.md) and [measurement record](docs/measurement-standard.md) for the existing engine architecture and test conventions.

## CHAT mode — personal interactive use

**Requires local unpublished two-mode changes.** This is the configuration that successfully completed the user's real multi-turn session:

```bash
cd ~/Documents/Veystrix
./chat.sh --mode chat --pool-gb 0 --auto-purge
```

- One process loads the checkpoint and keeps the conversation state across turns.
- Replies stream in Terminal. The default plain profile does not supply tools.
- `--pool-gb 0` avoids allocating an optional decode expert pool, which caused a memory guard stop at 2 GB in an earlier run.
- `--auto-purge` enables conditional macOS file-cache recovery at startup/idle only, when authorized; it is **not** a purge-every-five-seconds command and does not run during inference.
- Launch still depends on memory pressure, available memory, swap, power and GPU safety checks.

At the `you>` prompt, ask a question. Useful interactive commands include:

| Command | Purpose |
|---|---|
| `/help` | Display commands supported by the active CLI |
| `/new` or `/reset` | Start a fresh conversation without reloading the checkpoint |
| `/session` | View save/load options |
| `/status` | Inspect runtime settings |
| `/quit` | Exit |

The CLI's output-token settings have changed during development; run `/help` or `/config` and use commands the running version confirms. `/max 1000` is not a supported command in the tested session.

### What was actually measured

One full-checkpoint pool-off CHAT session completed five turns, including questions about rainbows, quantum mechanics, and C data structures. Decode was ~2.7–3.0 tokens/sec. A preceding 2 GB-pool attempt stopped after outputting the beginning of a response; the guard saw 2.04 GiB compressor growth with only 2.94 GiB available. Keep that guard enabled.

## AGENTIC mode — local API for coding agents

**Not yet validated with the intended DeepSeek Harness.** The local development CLI contains a second mode using the **same inference core**:

```bash
cd ~/Documents/Veystrix
./chat.sh --mode agentic
```

The intended server listens on `http://127.0.0.1:8080`. The existing server documents:

| Protocol | Endpoint |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` |
| OpenAI Responses | `POST /v1/responses` |
| Anthropic Messages | `POST /v1/messages` |

The endpoint list is **not** proof that a particular agent harness will work; check its actual request protocol, model name, tool-call encoding, streaming and cancellation behavior. Test on localhost first. **Do not expose an unauthenticated server publicly.**

The local AGENTIC preset was reported to enable balanced 2,048-token prefill and prefix caching, with an initial 4,096-token context limit. Some adaptive-memory, prompt-profile and short-prefill features remain opt-in and have only offline tests. The supported 2K real-model prefix-cache experiments do **not** prove a long continuous coding session. See the [AGENTIC validation checklist](docs/flashnext/agentic-validation.md).

### AGENTIC acceptance gates

Before describing the server as ready for coding-agent use, verify: actual protocol compatibility; exact tool schema and arguments; same-plan cached/uncached state equality; multi-turn tool results; no progressive memory growth; cancellation; error handling; and guarded operation on the full checkpoint. Test a short prompt and one harmless tool interaction before attempting a long system prompt.

## Memory and cache safety

- The current experimental startup check requires **9 GiB (9,663,676,416 bytes)** available, with separate lower runtime floors and a conditional compressor-growth stop. These are different gates.
- File-backed macOS cache can be reclaimable; “Cached Files” in Activity Monitor is not the amount of RAM permanently unavailable.
- Conditional `sudo -n /usr/sbin/purge` requires a separate, narrow administrator permission and should not run during generation; purging can also hurt SSD-streaming performance.
- A decode expert pool is optional. A fixed 2 GB pool triggered a guard in one CHAT run; pool-off completed several turns. Do not enable 6 GB simply because an older MACQWEN README recommended it.
- Prefix-cache snapshots must match exact tokens, checkpoint/configuration identity, and prefill chunk-plan history. A cache miss is preferable to incorrect reused state.
- Keep protected active and ancestor snapshots, checksum/manifest validation, GPU ownership, and resource guards intact.

## Build and test

The locally installed runtime uses a managed Python environment. For code in this repository, the existing test entry points are:

```bash
.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py' -q
.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py' -q
```

The latest local work includes additional mode and cache tests not yet pushed to this repository. Run tests against the **same code revision** that you intend to launch.

## Documentation

- [CHAT and AGENTIC validation / setup guide](docs/flashnext/agentic-validation.md)
- [Documentation index](docs/README.md)
- [Flash-Next brief](docs/flashnext/brief.md)
- [Flash-Next handoff](docs/flashnext/handoff.md)
- [Security guidance](SECURITY.md)
- [Measurement standard](docs/measurement-standard.md)

Earlier MACQWEN runtime guides (K2-Horizon, Bonsai-2, Qwen3.8-27B) remain linked from the documentation index. Legacy measurements are useful history, not current Veystrix default recommendations.

## License and acknowledgments

Source code retains the repository's [MIT license](LICENSE). Models and dependencies have independent licenses. This project builds on MACQWEN, Apple MLX/Metal, the Qwen architecture, and the relevant public checkpoint authors; see [NOTICE](NOTICE).