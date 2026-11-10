# Veystrix — CHAT and AGENTIC validation guide

> **Development note (8 November 2026; updated 2026-11-10):** this guide was written while `--mode chat|agentic` was an uncommitted working tree. The implementation is now committed in this repository (`development3.81`), but the **AGENTIC acceptance checks below have not all been run on the real checkpoint**: treat it as an acceptance guide, not proof of readiness.

## Why two modes?

Both modes share one Q4 MLX/Metal Flash-Next inference core: model weights, tokenizer, exact MoE routing, SSD expert loading and runtime safety rules. They differ in UX and scheduling, not model mathematics.

| | CHAT | AGENTIC |
|---|---|---|
| Entry | `./chat.sh --mode chat` | `./chat.sh --mode agentic` |
| Intended interface | Interactive terminal | Local HTTP API server |
| Workload | Personal multi-turn conversation | Coding agents, tool calls, repeated large system prefixes |
| Caching | Live conversation state; disk prefix cache off in the tested configuration | Exact prefix cache and bounded RAM/SSD tiers (experimental) |
| Prefill | One call per prompt (reference numerics in tested chat) | Balanced 2,048-token chunks (needs workload checks) |
| Guard | Mandatory | Mandatory |
| Full-model evidence | Five completed 0-GB-pool chat responses | Isolated 2K cached prefix comparisons, **no end-to-end harness proof** |

## CHAT: first real-model baseline

The successful tested local command was:

```bash
cd ~/Documents/Veystrix
./chat.sh --mode chat --pool-gb 0 --auto-purge
```

It completed multiple turns with ~2.7–3.0 generated tokens/sec on a fanless M4 MacBook Air 16 GB. The model loads from a ~113 GB SSD checkpoint; only selected experts are streamed. The 0-GB pool configuration avoids reserving another ~2 GB of RAM. A previous 2-GB fixed-pool run was stopped by the memory guard.

Commands supported in the tested Terminal experience include `/new`, `/reset`, `/status` and `/quit`. Use `/help` to inspect token-limit settings; some earlier proposed shortcut commands were not accepted.

**Never weaken the safety stop to make the model finish.** For the 2-GB pool failure, compressor growth hit 2.04 GiB while available memory dropped to 2.94 GiB, below the conditional 3 GiB floor.

## AGENTIC: integration plan

The local server is intended to listen at `127.0.0.1:8080` and accept OpenAI Chat Completions, Responses and Anthropic Messages endpoints. **The user's DeepSeek Harness protocol is not yet confirmed.** Verify it by inspecting the actual harness client configuration and requests (locally, without transmitting prompts elsewhere).

### Stage A — verify the exact request contract

Inspect:
1. Endpoint and JSON schema actually sent by the harness.
2. Streaming event types and finish reason semantics.
3. Message roles (system, developer, user, assistant, tool).
4. Tool definitions, function-call IDs, tool argument encoding, and tool-result messages.
5. Model identifier and context/token limit settings.
6. Authentication and cancellation behavior.

Only implement the smallest missing compatibility adapter, not a new inference engine. Keep the server bound to localhost. The endpoint list on its own does not establish compatibility.

### Stage B — short, guarded end-to-end test

In a single process, use a *small real MLX model* to exercise the server and harness adapter. Verify:
- Correct first user response.
- One harmless tool call with valid schema and arguments.
- A tool-result follow-up without lost state.
- Streaming, cancellation and error handling.
- Runtime guard presence and shutdown cleanup.

Then, **with owner approval** and when startup memory and power probes pass, run one bounded full-checkpoint AGENTIC smoke test using a short (well below 2K) system prompt. Do not use the 10K system prompt as the first test.

### Stage C — exact prefix-cache reuse

Use an unmodified shared system prefix followed by two distinct user turns:
1. Cold request produces a snapshot at a valid boundary.
2. RAM-tier repeated request restores the exact prefix.
3. Restarted process restores the same snapshot from SSD.
4. Compare cached versus uncached using **identical chunk plans**.
5. Require exact generated tokens, KV, recurrent, convolution, indexer and position state.
6. Reject incompatible or corrupt snapshots, preserve active leases and eviction dependencies.

Use isolated cache directories and a single writable owner until cross-process correctness is fully verified.

### Stage D — long-session stability

Test multi-turn tool cycles and prompt growth. Track:
- Cache-hit rate and exact reused-token count
- Cold versus cached TTFT
- Prefill and decode tokens/sec
- Bytes issued to expert reads (not necessarily physical NVMe bytes)
- Peak MLX allocations, available RAM and compression
- Swap-out, memory guard decisions, pool-size decisions
- Snapshot integrity and clean shutdown

Do not silently truncate essential instructions at the context limit. A 4,096-token local preset is **not** evidence that 4,096 tokens, let alone 10K, work reliably at full-model scale.

### Stage E — grow context only after validated

Progression: a successful short AGENTIC flow → 2K same-plan prefix check → 2K multi-request stability and quality smoke → 4K → 8K → 10K. Require new approval and readiness checks for each heavy stage. Any guard stop ends that heavy test; diagnose offline rather than automatically retrying.

## Caching and memory policy

- The active CHAT baseline uses live conversation state without persistent prefix caching.
- AGENTIC should persist an *exact*, compatible system-prefix snapshot once, with RAM hot tier and SSD cold tier.
- Snapshot identity must include model/tokenizer/config fingerprint and compatible chunk-plan provenance.
- Bound SSD usage and preserve a free-space reserve. Do not delete model shards, active snapshots or dependent ancestors.
- Reclaim unnecessary Veystrix-owned allocations between requests; retain warm decode experts only when headroom permits.
- File cache is managed by macOS. Optional startup/idle purge may free it but can increase model SSD traffic. Do not purge during active inference.
- A fixed 1-GB or 2-GB pool is not known to be globally optimal. The 0-GB CHAT baseline was safe for several turns; a fixed 2-GB CHAT pool caused a guard stop in one run.

## Local testing and reporting

```bash
.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py' -q
.venv/bin/python -m unittest discover -s veystrix/tests -t . -p 'test_*.py' -q
```

Every real checkpoint test should record fixture IDs, actual settings, power/thermal state, prefill and decode timings, cache behavior, guard outcomes, and a trace path. Distinguish observed real-model results, small-model results, and estimates.

## Troubleshooting

- **Mode option unrecognized:** the published remote branch has not yet received `--mode`. Use the code revision containing `models/flashnext/settings/modes.py`.
- **Launch blocked:** the guarded startup threshold is 9 GiB, not 9 decimal GB. Check actual memory and swap probes. Do not bypass it.
- **2 GB CHAT pool guard stop:** use the tested `--pool-gb 0` baseline until safe dynamic selection is validated on the full model.
- **Cached response mismatch:** stop and capture snapshots/trace, verify execution fingerprint and chunk history; do not accept a corrupt cache hit.
- **Unknown harness protocol:** inspect the real client request before claiming compatibility.
- **Slow decode:** SSD expert misses dominate; pool-hit reports and throughput must be measured on the real model, not inferred from a synthetic 75% hit ratio.
