# Veystrix Core

> **Private repository (`joyboyreturns/Veystrix-Core`).** Veystrix is proprietary freeware. This
> repository holds the engine, the macOS app, internal documentation and the full development
> history. The public home (README, releases, update feed, license) is
> [`joyboyreturns/Veystrix`](https://github.com/joyboyreturns/Veystrix). See
> [legal/LICENSE_PROVENANCE.md](legal/LICENSE_PROVENANCE.md) for licensing status.

Veystrix streams sparse Mixture-of-Experts language models from SSD on
low-memory Apple Silicon Macs. The primary runtime keeps the dense model
core in unified memory while streaming selected expert rows and n-gram
rows from the checkpoint on demand.

Two runtimes are supported: Flash-Next (the default) and the experimental
NVIDIA Nemotron 3.5 Lightning streamed runtime (see `docs/nemotron35/`; real
inference needs explicit authorization and has not been validated). The earlier
K2-Horizon, Bonsai-2 and Qwen3.8-27B research runtimes were removed.

## Current status (evidence recorded 2026-11-08, one machine)

Merged from the `docs/chat-agentic-guides-20261108` branch. The `--mode chat|agentic` launcher and the supervised
`./serve.sh` are present in this tree; the 2026-11-08 "unpublished working tree" warning no longer applies to this
repository (it applied to the old public MIT repository).

| Capability | Evidence | Status |
|---|---|---|
| Q4 checkpoint loading and token generation | Real 113 GB checkpoint on M4 MacBook Air | Demonstrated |
| Personal multi-turn CHAT with `--pool-gb 0` | Five completed responses; ~2.7-3.0 decode tokens/sec | Works on the development machine |
| CHAT with a fixed 2 GB decode expert pool | Compressor/available-memory guard stopped the first answer | Not recommended as an initial setting |
| CHAT auto-purge | Startup recovery executed; no OS purge during inference | Experimental |
| Balanced 2,048-token prefill | Historical 2K full-model run ~61-64 prefill tokens/sec | Preliminary, specific workload |
| Same-plan 2K prefix caching | 2,046-token RAM/SSD restores; identical 64-token continuation and 124 state tensors | Validated in isolated full-model experiments |
| AGENTIC API server + harnesses | Protocol and mock-server tests pass offline (Hermes CLI, Qwen Code); real-model harness runs are not complete | End-to-end on the real checkpoint **unverified** |
| 4K / 8K / 10K+ long-context operation | See [docs/flashnext/](docs/flashnext/) status files; layer-major runs are preliminary and opt-in | Not verified as a long-lived server |

Data are from a single machine and selected runs; they do not establish general performance or guarantee no quality
regression. Public performance claims live in the public repository's `docs/performance.md` and only include full
top-10 routing results.

## Primary runtime: Flash-Next (SSD-streamed sparse MoE)

| Item | Value |
|---|---|
| Checkpoint | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (installed; alias `vontra-mtp`) |
| Weights | 22 safetensors shards, 3,747 indexed tensors (incl. 76 MTP), 113.2 GB / 105.4 GiB |
| Quantisation | Uniform 4-bit affine, group 32; multimodal modules and MoE router gates stay BF16 |
| Architecture | `qwen4_exp` sparse MoE: 48 layers, 512 routed experts (top-10) + 1 shared, vocab 248,320, 262,144 context |
| Language stack | 125B total / 6B active, plus 51B n-gram table (20M entries) and a 4B MTP draft block (MTP disabled in production) |
| Launch | `./chat.sh --model flashnext --checkpoint vontra-mtp` |

Streaming means each token reads only the experts and n-gram rows it
needs. `exact-quality` is the normal routing profile; the other profiles
(`standard`, `cache-aware`, `fast-quality`, `fused-quality`) are research
controls — see the [Flash-Next brief](docs/flashnext/brief.md).

Current chat defaults live in `models/flashnext/settings/launch.py` (`launch.py` still sets `FLASHNEXT_EXPERT_POOL_GB=6`, promoted 2026-09-23 from pool measurements; the 2026-11-08 real CHAT runs found a 2 GB pool tripping the memory guard on this 16 GB Mac, so pass `--pool-gb 0` explicitly and see the memory guidance under Quick start):
Metal runtime, a 6 GB application-owned expert pool read in place by the
Metal kernels (it replaces the static slab pack and the expert pins),
GPU keep-warm on, one host sync per decode layer, compiled GatedDeltaNet
glue, and the exact opt-in bundle (streamed embedding, compiled
glue/norm, cached norm gain, interactive QoS, parallel n-gram reads, both
QSA flags, stream-pack chunk 2). Every default keeps the exact token
digest. The pool needs about 6 GB of free memory; on a busier Mac, roll
it back with an explicit value, for example:

```bash
FLASHNEXT_EXPERT_POOL_GB=0 ./chat.sh --model flashnext --checkpoint vontra-mtp
```

## Reference performance (Flash-Next, installed checkpoint)

| Operation | Result |
|---|---|
| Current defaults (6 GB expert pool), 128 tokens | 3.79–3.93 tok/s; 4.09 tok/s over tokens 65–128; +11.3% over the pool-off defaults, 3/3 pairs, two-SE 1.9%, digest `e19af44d5268e9d1` |
| Pool-off defaults (slab and 8 pins), 128 tokens, same session | 3.42–3.47 tok/s, 294–310 MB/token |
| Pool-off defaults before the host/GPU-glue stack, 128 tokens | 3.05–3.09 tok/s (earlier session); stack +6.6% inside 8.0% |
| 128-token paired arms, 8 pins, keep-warm off / on | 2.33–2.41 / 2.86–2.93 tok/s |
| `bench_production`, 96 tokens, keep-warm off / on | 2.19 / 2.55 tok/s |
| Chat, 8 pins (policy), 72 tokens | about 2.14 tok/s, about 350 MB/token |
| Historical 60-slot protocol, 32 pins, 32 tokens, 4 arms | 2.28 tok/s median (2.02–2.77), 2.36 tok/s tail, 402 MB/token |
| Historical oQ4 (not installed), 60-slot pack, 32 tokens | 3.08 tok/s, 3.00 tok/s tail, 279.7 MB/token |
| Historical REAP-288 (not installed), terminal sanity, 32 tokens | 3.74 tok/s median, 3.45 tok/s tail, 193.3 MB/token |
| Long-prompt prefill near 5,000 tokens (historical oQ4) | About 40–50 tok/s; 62.19 tok/s in a synthetic diagnostic (not production) |

Measured on an M4 Mac with 16 GB unified memory and a 256 GB SSD, on a
quiet machine (no other heavy applications); free memory moves the rate
more than most code changes, because the expert stream reads less when
more RAM holds experts. The oQ4/REAP rows cannot be reproduced here.
Short 32-token arms sit inside a ~40-token warm-up window and read higher
than longer answers. Evidence: [results/flashnext/](results/flashnext/)
(latest: `20260923-153742-expert-pool/`, `20260923-133612-read-replay-native/`),
[Flash-Next handoff](docs/flashnext/handoff.md). The earlier goal of
3 tok/s decode without quantizing or pruning is met; the current target is
4 tok/s (250 ms/token).

## Quick start

Requirements: Apple Silicon Mac, Python 3.12, a fast SSD, ~120 GB free for the checkpoint. The startup memory
check requires **9 GiB (9,663,676,416 bytes)** available; the runtime guards have separate, lower floors and a
conditional compressor-growth stop. On a 16 GB Mac the optional decode expert pool is the main memory lever: pool-off
(`--pool-gb 0`) completed real multi-turn chat, while a fixed 2 GB pool tripped the guard in one run. Do not enable a
6 GB pool just because older text recommends it. Prefer the supervised launcher (`./serve.sh`, see
[docs/MANUAL_LAUNCH.md](docs/MANUAL_LAUNCH.md)); an unsupervised direct launch once froze the Mac.

```bash
git clone https://github.com/joyboyreturns/Veystrix-Core.git   # private; needs access
cd Veystrix-Core
./chat.sh setup
hf download Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
./chat.sh --model flashnext --checkpoint vontra-mtp
```

With exactly one compatible checkpoint installed, Veystrix selects it
automatically. Set `VEYSTRIX_MODEL_ROOT` when checkpoints live outside
`~/models`. A full path also works with `--checkpoint`, and the
launcher validates the checkpoint before loading.

Run the project live-test terminal:

```bash
./tests/run.sh
```

It asks which installed runtime and checkpoint to test, discovers that
runtime's cases, and writes each run to its own folder under
`results/<runtime>/`. See [docs/testing.md](docs/testing.md).

## Daily use

Inside the chat, use:

```text
/help [all]
/new
/session save|load|list|delete [name]
/config [section] ...
/status
/quit
```

Use `/status` for model, profile, routing, context, and memory
information. Flash-Next extras: `/config model gpu-keepwarm off`
disables keep-warm and saves the choice. Use
`/config display animate off` to disable output animation.

## Shared inference cache

Qwen3.8 Flash-Next and Nemotron 3.5 Lightning share one **80 GB** persistent
prefix quota at `~/.cache/veystrix/prefix-cache-v1`. The cache retains a
separate **50 GB** free-space reserve. The manager keeps Qwen delta chains
and Nemotron hybrid states in separate compatibility namespaces.

In the macOS app, open **Settings → Universal Cache** to inspect live usage,
change the quota, preview a reclaim operation, or clear eligible states.
The same operations are available without starting a model:

```sh
.venv/bin/python -m veystrix.universal_cache status
.venv/bin/python -m veystrix.universal_cache preview 10
.venv/bin/python -m veystrix.universal_cache reclaim 10
.venv/bin/python -m veystrix.universal_cache quota 80
```

`--prefix-dir` and `VEYSTRIX_CACHE_ROOT` select a parent for the dedicated
cache directory. `VEYSTRIX_CACHE_GB` initializes the quota for a new root.
Settings and the `quota` command update the stored quota live. Cache
administration leaves model files and named sessions outside its scope.

Nemotron defaults to a **1,048,576-token logical context window** with
block-backed affine INT8 attention KV and FP32 Mamba SSM state. Each
request can prefill up to **65,536 uncached tokens**. Longer conversations
accumulate context across requests. The engine admits the configured
window only when memory and swap checks pass. Set a smaller window in
macOS Settings, with `--context-limit` on a direct Nemotron CLI launch,
or with `VEYSTRIX_NEMOTRON_CONTEXT` on a supervised launch. See the
[validation results](docs/architecture/unified-cache-nemotron-1m-results.md).

Qwen's experimental throughput profile combines 4K hybrid layer-major
prefill and buffered decode-pool reads. Enable it in the app's Engine
settings or launch with `./serve.sh --enable short-throughput --pool-gb 2`.
The profile uses more memory than the standard path. See the
[short Qwen benchmark](results/qwen-short-20261111/) and
[short Nemotron benchmarks](results/nemotron-short-20261111/).

## CHAT and AGENTIC modes, memory and cache safety

```bash
./chat.sh --mode chat --pool-gb 0 --auto-purge     # personal interactive use, one process keeps conversation state
./chat.sh --mode agentic                           # local API for coding agents (127.0.0.1:8080); supervised: ./serve.sh
```

* `--auto-purge` enables conditional macOS file-cache recovery at startup/idle only, when authorized by a narrow
  `sudo -n /usr/sbin/purge` permission; it never runs during inference (purging can hurt SSD-streaming speed).
  "Cached Files" in Activity Monitor is reclaimable, not permanently unavailable RAM.
* Interactive commands: `/help`, `/new` or `/reset`, `/session`, `/status`, `/quit`. Use only commands the running
  version confirms with `/help`.
* The server speaks OpenAI Chat Completions (`POST /v1/chat/completions`), OpenAI Responses (`/v1/responses`) and
  Anthropic Messages (`/v1/messages`). An endpoint list is not proof a harness works: check its request protocol,
  model name, tool-call encoding, streaming and cancellation. Never expose the server without authentication.
* Acceptance gates before calling AGENTIC "ready": protocol compatibility, exact tool schema and arguments, same-plan
  cached/uncached state equality, multi-turn tool results, no progressive memory growth, cancellation, error
  handling, and guarded operation on the full checkpoint. See
  [docs/flashnext/agentic-validation.md](docs/flashnext/agentic-validation.md).
* Prefix-cache snapshots must match exact tokens, checkpoint/configuration identity and prefill chunk-plan history; a
  cache miss is preferable to reused incorrect state. Keep protected active/ancestor snapshots, checksum/manifest
  validation, GPU ownership and the resource guards intact. Never prune experts; routing, quantization and weights
  stay unchanged by cache and scheduling work.

## Native agent workflows

`veystrix workflow` runs single-agent, swarm, and detached overnight goals with
isolated agent contexts, a persistent task graph, and one inference slot. The
[native workflows guide](docs/engineering/native-workflows.md) includes an
offline mock example, CLI commands, and recovery behavior.

## The Veystrix app and releases

* `apps/macos/Veystrix`: native SwiftUI app (dashboard, explicit Start/Stop/Restart, model-folder validation without
  reading weights, harness settings for Hermes / Qwen Code / DeepSeek Harness / OpenCode / pi / Kimi Code /
  OpenAI-compatible clients, logs and sanitized diagnostics). It never loads a model on launch.
* Build: `packaging/engine/build_engine_bundle.sh` then `packaging/macos/make_release.sh --engine dist/engine`.
  Produces `Veystrix-<ver>-arm64.dmg` and `.zip`, `SHA256SUMS.txt` and `BUILD-INFO.txt`.
* **Free channel (default, no Apple membership):** ad-hoc signed, not notarized; macOS Gatekeeper warns on first open.
  **Developer ID channel (ready, not activated):** signing + notarization + the same Sparkle EdDSA update feed when
  credentials exist. See [docs/release/FREE_CHANNEL.md](docs/release/FREE_CHANNEL.md) and
  [docs/release/RELEASE_PROCESS.md](docs/release/RELEASE_PROCESS.md).
* Public repository (README, releases, update feed): https://github.com/joyboyreturns/Veystrix. This repo is private.

## Local API server

Start the local server with:

```bash
./chat.sh --server
```

The compatibility command `/server` is also accepted inside chat. The
default address is `http://127.0.0.1:8080`, and the server processes one
generation at a time.

| Protocol | Endpoint |
|---|---|
| OpenAI Responses | `/v1/responses` |
| OpenAI Chat Completions | `/v1/chat/completions` |
| Anthropic Messages | `/v1/messages` |

Localhost does not require authentication by default. A non-local bind
requires a shared key:

```bash
export VEYSTRIX_SERVER_API_KEY="PRIVATE_VALUE"
./chat.sh --server --host 0.0.0.0
```

Allow browser requests only for a trusted origin:

```bash
./chat.sh --server --allow-origin http://localhost:3000
```

Read [SECURITY.md](SECURITY.md) before changing the host, allowing browser
origins, or enabling repository tools.

## Keys and local data

Manage optional Tavily and Context7 keys inside the chat:

```text
/config keys
/config keys set tavily
/config keys set context7
/config keys delete tavily
```

Key input does not echo. Session files can contain private prompts and model
state; keep them private.

| Data | Default location |
|---|---|
| Preferences | `~/.veystrix/preferences.json` |
| API keys | `~/Library/Application Support/Veystrix/api_keys.json` |
| Flash-Next sessions | `~/.cache/flashnext/sessions/` |
| Nemotron sessions | `~/.cache/veystrix/nemotron35/sessions/` |

The launcher uses `.venv` by default. Developers can provide a validated
override with `VEYSTRIX_PYTHON` or a model-specific `VEYSTRIX_*_PYTHON` variable.

## Troubleshooting

- No checkpoint appears: pass its full path with `--checkpoint`.
- A download is incomplete: resume the `hf download` command into the same directory.
- The Vontra download is ~105 GiB: confirm free space before retrying.
- Several checkpoints are installed: provide both `--model` and `--checkpoint`.
- Generation slows down: close memory-heavy applications and retry.
- A secondary runtime is not selected: pass its model and checkpoint aliases explicitly.
- A command has changed: run `/help` for the active command list.

## Tests and development

Run the shared unit tests:

```bash
.venv/bin/python -m unittest discover -s veystrix/tests -t . -p 'test_*.py'
```

Run a runtime's unit tests with the same pattern, replacing `RUNTIME`:

```bash
.venv/bin/python -m unittest discover -s models/RUNTIME/tests/unit -t . -p 'test_*.py' -q
```

Use `./tests/run.sh` for live-model evidence. Every run writes to
`results/<runtime>/`. Read [docs/testing.md](docs/testing.md),
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[measurement standard](docs/measurement-standard.md) before changing code or
running experiments. Do not run benchmarks without approval of the plan.

## Documentation and layout

The [documentation index](docs/README.md) lists the current component
documents. Release, packaging, security and
launch guides: [release process](docs/release/RELEASE_PROCESS.md), [free channel](docs/release/FREE_CHANNEL.md),
[validation status](docs/release/VALIDATION.md), [packaging](docs/packaging/PACKAGING.md),
[security audit](docs/security/SECURITY_AUDIT_2026-11-10.md), [manual launch](docs/MANUAL_LAUNCH.md),
[Hermes integration](integrations/hermes/README.md), [pi integration](integrations/pi/README.md), [AGENTIC validation](docs/flashnext/agentic-validation.md). Each active runtime follows the same structure:

- `brief.md`: purpose, scope, status, and main results;
- `research.md`: dated measurements, decisions, and rejected approaches;
- `handoff.md`: current operation, validation, constraints, and next work.

Historical records are in [docs/archive](docs/archive/README.md). They preserve
provenance and may contain obsolete commands.

```text
apps/macos/Veystrix/        Native SwiftUI app (VeystrixKit logic + UI + Sparkle)
packaging/                  engine bundle, macOS app/DMG/ZIP/appcast scripts, legal generators
integrations/hermes/        Hermes Agent profile configurator and offline mock server
integrations/pi/            pi agent extension pointer (canonical source: veystrix/pi_extension.py)
legal/, LICENSES/           freeware license draft, provenance, SBOM, third-party notices
scripts/security/           secret scan and public-exposure audit
veystrix/                    Shared chat, commands, settings, tools, and test engine
tests/                      Project live-test terminal
models/flashnext/           Flash-Next streaming runtime and benchmarks
models/nemotron35/         Experimental Nemotron 3.5 Lightning streamed runtime and model-free tests
docs/                       Current guides, evidence, and historical records
results/                    Per-run measurement records (one folder per run)
```

## License

Veystrix is proprietary freeware; it is not open source as a whole. Code published earlier as VEYSTRIX under the MIT License keeps that license and notice (see LICENSE, LICENSES/, legal/). Models and dependencies use their
own licenses. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The Vontra
checkpoint card carries the Qwen Community License 1.0; review it before
use or redistribution.

## Acknowledgements

Veystrix uses and builds on these third-party projects and platform tools:

- Apple macOS, Apple Silicon, Metal, Objective-C++ Metal APIs, Xcode, and
  Instruments provide the runtime platform, custom kernels, and trace tools.
- MLX, MLX-LM, and MLX-VLM provide tensor execution, model loading, and model
  architecture support.
- Transformers, NumPy, Requests, and Hugging Face Hub provide tokenization,
  array operations, HTTP access, and checkpoint access.
- Hugging Face hosts the public checkpoint files and model discussions used in
  this project and its evidence record.
- The Qwen team provides the Qwen model architecture, tokenizer, sampling
  guidance, and checkpoint family used by Veystrix.
- Vontra provides the public Flash-Next MLX checkpoints used in this work.
- NVIDIA provides Nemotron 3.5 Lightning, and mlx-community its public MLX 4-bit conversion.
- Tavily and Context7 provide optional search and documentation tools for the
  repository-tool chat profile.
- Python provides the runtime and standard-library components.
- GitHub provides repository hosting, issue tracking, and GitHub Actions CI.
- The Veystrix research record credits external reports and source notes where
  they affect a measurement or a decision.
