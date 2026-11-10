# Qwen-to-Nemotron feature parity

The Qwen branch was merged locally after the initial branch was pushed. The application, settings and shared server sources are present in this worktree, but their behavior is **not automatically compatible** with Nemotron. Qwen remains the working default. The two-model-only cleanup and a production Nemotron loader remain open.

| Qwen feature | Nemotron requirement | Status |
| --- | --- | --- |
| Desktop model selection and checkpoint validation | Explicit Flash-Next/Nemotron selection, persistent independent paths, no auto-switch | Not implemented; UI currently accepts Flash-Next only |
| Power, battery floor, AC-only and Low Power Mode | Same user controls with identical semantics | Shared UI exists; not wired or tested for Nemotron |
| Guard, swap/thermal pressure, watchdog and restart | Architecture-neutral guard, safe cancellation; never replay tool actions | Not implemented for Nemotron |
| Engine supervisor and diagnostic logs | Manual selection, bounded restarts, actionable errors | Not wired for Nemotron |
| `/health` activity, prefill progress, decode rate, persistent cache use | Accurate live counters and dashboard UI | Shared UI exists; Nemotron telemetry not implemented |
| OpenAI/Anthropic APIs, streaming and usage | Model-specific template, honest limits and accounting | Shared server exists; no production adapter or Nemotron validation |
| Hermes reasoning alias and multi-turn tools | Correct thinking semantics, tool IDs and arguments, no replay | Shared fixes present; not verified with Nemotron |
| Prompt and persistent cache, session isolation | Exact tokenizer/model/precision/state fingerprint; Mamba + KV restore | Synthetic snapshot exists; no production integration |
| SSD experts, bounded pool, prefetch | Exact 6-of-128 + shared; bounded memory and safe async operations | Synthetic and stacked-index slice prototypes only |
| Admission, 16 GB planning, single-model residency | Conservative measured core/working-set limit and fail-closed selection | Arithmetic estimator only; no production enforcement |
| Context and incremental prefill | Enforced measured context, bounded chunks | Synthetic only; no real-model evidence |
| MTP | Verified speculative tokens only, default off | Not implemented |
| Packaging and signing | Include Nemotron runtime, metadata, assets and dependencies | Not implemented or validated |

Do not advertise parity, use model auto-routing, or make Nemotron the default until every applicable row has passed synthetic, integration, safety and real-checkpoint validation. Real downloads do not imply permission to load the model or claim speed.
