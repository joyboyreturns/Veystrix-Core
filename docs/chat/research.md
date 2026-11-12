# Shared chat design record

This is the active design record for the shared chat. Current operation
belongs in [`handoff.md`](handoff.md); runtime-specific evidence belongs in
the corresponding runtime record.

## Problem

The project first used one chat per model, so features diverged. One shared
interface now starts each runtime through its compatible launcher. The managed
`.venv` is the default; validated runtime overrides remain supported.

## Main design decisions

| Decision | Reason |
|---|---|
| One shared session loop | A chat feature reaches every model |
| One command table | Aliases and settings cannot diverge |
| One preferences schema | Each setting has one default and validator |
| Separate model environments | Each runtime can select a compatible MLX stack |
| Lazy backend imports | The shared package stays pure Python |
| Profiles independent from models | Tools define chat behavior, not model capability |
| Backend-owned persistence | Each model has a different exact cache format |
| Backend progress callback | The UI uses measured work instead of prediction |
| Separate reasoning capacity | Long reasoning has extra generation space |
| Streaming protocol filter | Tool activity appears before tool execution |
| Execution-only tool clock | Display delays do not corrupt tool duration |
| Backend-owned live routing | Model-specific routing changes without reload |

## Package boundary

`veystrix/` contains the shared interface. `models/` contains model-exclusive runtime code. Backends are the only shared modules that import
model code.

The current backend interface follows the operations used by `session.py`: conversation setup, append operations, generation, reset, status,
and session persistence.

## Feature union

The shared chat keeps these features from the former chats:

- Thinking activation and display.
- Reply token limits.
- Reasoning effort and answer streaming.
- Routing profiles and exact sessions.
- Workspace awareness and editable prompts.
- Repository tools with approval controls.
- Pure JSON benchmark output.

## Generation limits

`max_tokens` is the answer allowance. A saved value of `-1` selects the profile default. The defaults are 4,096 tokens for `plain` and 2,048
tokens for the repository-tool profile.

`think_budget` adds 512 tokens by default when thinking is enabled. Generation uses the answer allowance plus this reasoning capacity.
Thinking-disabled turns add nothing.

The old schema accepted `0` as no reasoning budget. The new schema uses `-1`. A saved `0` fails validation and resolves to the new 512-token
default.

The model backend still receives one generation ceiling. The runtime does not count reasoning and answer phases independently. The extra
capacity protects the usual answer space when reasoning stays within its expected budget.

## Prefill progress

The old display predicted progress from prompt size. That prediction could remain at zero or show an inaccurate rate.

The backend interface now accepts `on_prefill_progress(done, total)`. The Qwen27B adapter forwards its native chunk progress.

Flash-Next reports each completed streamed MoE layer. The adapter maps the completed-layer ratio to prompt-token-equivalent progress. It
reserves the last step until prefill finishes. This prevents an early 100 percent display.

The UI smooths rates from consecutive progress callbacks. Its moving colour animation continues between callbacks. The final statistics
calculate true prefill throughput from prompt tokens divided by complete prefill time.

The progress bar uses filled `█` and unfilled `░` cells. It omits border characters. Its width adjusts to the terminal width.

## Tool lifecycle

`ToolCallStreamFilter` remains model-agnostic and detects the established
`<tool_call>` contract. The K2-Horizon package translates its native JSON
protocol to that contract before output reaches the shared chat. K2-specific
measurements and decisions were removed with that runtime. The filter
hides protocol markup and emits the tool name after `<function=NAME>` arrives.

The tool UI uses these states:

| State | Display | Timer |
|---|---|---|
| Protocol starts | `Preparing tool` | Not started |
| Function name arrives | Action label | Not started |
| Repository call starts | Action label | Starts |
| Repository call ends | Completed action and duration | Stops |

The UI keeps very fast tools visible for at least 200 ms. It captures elapsed time before this display delay. A successful line has no icon,
emoji, or green status word. Failed tools keep an explicit failure label.

## Output layout and statistics

The ready line reports model, chat profile, and a `/help` hint. `/status`
reports thinking status, routing profile, context, and memory. `/help` shows
six primary commands. `/help all` shows compatibility commands and aliases.
Dispatch still accepts every compatibility command.

Visible reasoning starts directly below the input prompt. The stream removes leading tag newlines. It leaves one blank line before the
answer.

Reasoning words use a darker four-step fade. Answer words use the normal fade. Both animations run outside model timing.

Plain mode reports one generation segment. The repository-tool profile can generate several segments around tool calls. Statistics sum prompt
tokens, output tokens, prefill time, and decode time.

The final line reports prompt rate, decode rate, context size, and wall time. The stop reason follows on the next line. `truncated` means
generation reached its ceiling before a final answer or tool call completed.

Flash-Next owns live routing configuration. The shared `/config model` command
passes `cache-aware` and `swap-epsilon` to that backend. The compatibility
`/settings` command remains accepted. This
keeps routing state outside the shared chat package.

## Profile prompts

Each profile stores its custom system prompt in a separate file beside the preferences file. Profile changes reset conversation state and
rebuild the toolbox. The loaded model remains in memory.

A one-time migration moves the legacy `system_prompt` preference into the active profile file. The preference field remains empty after
migration.

## Message language

Flash-Next saved-session validation errors now use English. This change covers names, headers, checksums, compatibility, tensors, cache
state, and corruption. It does not change the saved-session schema or payload format.

## Terminal input

Bracketed paste keeps all pasted newlines in one input message. Pressing Enter after the paste submits it.

Input text keeps control-marker strings as text. It does not inject chat control tokens into the transcript.

## Private API keys

The repository contained no hard-coded API keys during the 2026-08-31 audit. The audit also checked high-confidence secret patterns in Git
history.

The `/config keys` interface stores Tavily and Context7 keys outside the repository in both profiles. Compatibility `/keys` and `/api-keys` commands remain accepted. Key input does not echo. Inline key arguments are rejected. Approval and workspace settings remain agent-only.

Secret-like environment variables are removed from tool and checker child processes.

## Validation baseline

The restructure completed with these checks:

- Shared unit tests pass in the system Python environment.
- Backend tests pass in each model environment.
- Flash-Next restores an existing exact session after relocation.
- Flash-Next benchmark mode emits one valid JSON line on standard output.

The completed restructure plan remains in [`docs/archive/chat/restructure-plan.md`](../archive/chat/restructure-plan.md).

## 2026-11-12 serving, cache lookup, and arm64 release

The source starts at `unlimited-generation` revision
`f97011c384a36b65f32deab3f7f69e14b0789f79`, ten commits ahead of local
`main` at `68fc14bb53f1376096c24950e81aa89b9f085245`. The comparison
baseline includes the tracked and untracked runtime changes present at the
start of the work. Per-file manifests, status, and tracked patches are under
`dist/performance-iteration`. The checkpoints file includes installation IDs,
local Hugging Face revisions, config and index hashes, tokenizer and template
hashes, and shard sizes. Both arms used Python 3.12.16 with MLX 0.32.2,
MLX-LM 0.31.3 and MLX-VLM 0.6.17 on the 16 GiB M4.

### Short production HTTP path

We ran three fresh-process pairs per model in alternating order through
`tests/run.sh`. Both arms used BF16 KV, greedy sampling, the same prompt and
template, eight output tokens, and a fresh application cache for each arm.
The OS file cache stayed intact. Flash-Next used `short-throughput` in both
arms. Its guard declined the optional decode pool in this run. Nemotron used
2,048-token chunks and an 8,192-token layer-major window. The short prompt
fit inside one chunk.

| Metric, median | Flash-Next baseline → candidate | Nemotron baseline → candidate |
| --- | ---: | ---: |
| Prompt tokens processed | 37 → 37 | 36 → 36 |
| Time to first visible delta, s | 10.60 → 11.01 | 4.02 → 3.94 |
| Uncached prefill, tokens/s | 3.98 → 3.88 | 9.93 → 10.18 |
| Runtime decode, tokens/s | 1.69 → 1.63 | 3.80 → 3.83 |
| Wire deltas after the first, deltas/s | 2.02 → 2.00 | 4.04 → 4.02 |
| Physical process reads, GB | 24.48 → 24.55 | 12.33 → 12.31 |
| Process footprint after request, GB | 5.23 → 5.15 | 4.88 → 4.87 |
| System swap at request boundaries, MiB | 789.12 → 789.12 | 789.12 → 789.12 |

Raw arms, request event times, resource counters, health and wire output
digests are in
[Flash-Next](../../results/flashnext/20261112-224130-server-short-paired/server-arms.jsonl)
and [Nemotron](../../results/nemotron35/20261112-224330-server-short-paired/server-arms.jsonl).
All six text digests matched within each model. We did not collect token-ID
digests or a same-condition resolution band. Flash-Next paired decode changes
range from −7.14% to −3.55%. Nemotron paired decode changes range from −1.30%
to +3.95%. The results do not qualify a single-request speed promotion.
The physical counter covers the worker process and can include the health
probe after the request. It is distinct from logical expert bytes, MLX
allocation, and process footprint.

### Serving and cache

The router can hold simultaneous same-model leases for compatible callers.
A model change still waits for leases to drain. The connection handler
owns socket delivery, and streaming aborts on response-buffer overflow.
The engine still executes one generation at a time. The macOS chat keeps
tasks, stop controls, and errors per conversation.

The router can reuse a bounded, model-independent JSON encoding when an
authenticated session resends identical input. The destination worker still
runs its own template and tokenizer. Model state stays in its checkpoint
and runtime namespace. Router input hits and restored tokens are separate
counters.

Nemotron hashes the queried token tape once across candidate lengths and
opens the exact token tape only for a matching key. It leases the selected
snapshot while validating and materializing state outside the global cache
lock. The synthetic continuation test checks exact next-token logits and
that reclaim respects the lease. Flash-Next packs the query's token bytes
once for its existing snapshot keys. The disk formats stay compatible.

The [synthetic 102,400-token lookup arms](../../results/nemotron35/20261112-224520-cache-lookup-paired/lookup-arms.jsonl)
contain 32 Nemotron sidecars and 32 unrelated Flash-Next manifests. Median
first lookup took 11.20 → 2.88 ms. Median repeated lookup took
3.44 → 1.86 ms. The fixture loads metadata and token tapes. Real 100K+
tensor restore, exact real-weight continuation, context sweeps, numerical
mode qualification, and interleaved generation remain open.
