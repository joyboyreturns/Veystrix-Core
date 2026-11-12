# Experimental model-router work in progress

This router is enabled in the rebuilt app as an experimental feature; the evidence and limitations below apply. `veystrix.model_router` exposes a model catalog and serializes requests through `ModelCoordinator`, which checks exact installation identity, rejects ambiguous family aliases and protects an owned conversation against a different session's model handoff. `ProcessDriver` launches one guarded engine process group and refuses to load a replacement if the previous group has not exited. Requests are not automatically retried. `harness_models` merges two registry entries into Qwen Code and DeepSeek profile shapes, and `harness_configure` can apply these updates with backups after an explicit action.

A local fake-engine test switched a → b → a through a single endpoint with distinct process groups, and the installed DeepSeek CLI composed a temporary catalog containing both real local model IDs. These are not real-model generation or UI tests. The rebuilt app starts the router; model selection across unrelated sessions is still restricted, and full harness workflows are not yet verified.

Remaining work before enabling: test real supervised Qwen/Nemotron loads and termination under the existing memory/swap/watchdog policies; restore unattended restart supervision; ensure client cancellation and long model-load heartbeats across all API transports; validate session affinity when each real harness changes models, including sessions lacking explicit IDs and auxiliary requests; ensure recoverable state and safe handling of failed loads; implement and verify the Hermes provider refresh; run all three real harnesses; and validate the installed bundle through complete harness workflows. Current tests do not establish these guarantees.

## 2026-11-11 local smoke evidence

On the M4 Mac, a temporary router on port 18081 loaded the installed Qwen checkpoint, served a 15-token prompt with `max_tokens=4`, switched to the installed Nemotron checkpoint for the same conversation, and switched back to Qwen. Responses identified the requested exact IDs and contained `OK`, `OK.`, and `OK`. The sampled system swap remained 716.5 MiB after each request. These were short non-streaming smoke tests, not correctness, performance, endurance, tool-call, or harness integration validation. The router and its test workers exited after the test.

An initial real startup exposed a CLI probe parsing `--mode` as an ambiguous abbreviation; the CLI now disables abbreviations in that preliminary probe. The router is being wired into the app candidate, but installed-harness request routing and app packaging remain under validation.

A second smoke test exercised the packaged router's streaming path across Qwen → Nemotron → Qwen. All streams identified the requested installation and ended with `[DONE]`; sampled swap stayed at 716.5 MiB. Nemotron did not follow the instruction to reply `OK` in that streaming test, so this is routing evidence, not evidence of numerical equivalence or reliable model quality. The app integration has been compiled, but real client model switching, long-session recovery, cancellation, and supervised crash recovery remain unverified. Cross-session handoffs are deliberately refused with `model_handoff_required` until explicitly authorized, which can prevent model changes across separate harness sessions.

The app now uses this router in the development candidate. The installed DeepSeek desktop picker was observed showing both registered model names, and selection of each was verified without sending a task. A packaged startup test also preloaded Nemotron and successfully served a first Qwen request. The app's refreshed model catalogs update Qwen Code and DeepSeek configuration with backups and create/update a separate Hermes profile; a full real coding task in each harness has not been verified. `swift build` passes; `swift test` cannot run with the installed Command Line Tools because `TestingMacros` is unavailable.
# Context defaults

The Flash-Next launcher requests a 262,144-token context window. The Nemotron
launcher passes no `--context-limit`: the backend admits its configured
1,048,576-token window when its own memory plan fits the machine (the macOS app
also sets `VEYSTRIX_NEMOTRON_CONTEXT=1048576`). Because `--context-limit`
would override that default on a Nemotron launch, it is deliberately absent
from the launch line.

Harness catalog entries advertise a family-configured window from
`veystrix.model_registry.catalog_context`: Flash-Next uses the installed
checkpoint's reported native context capped at 262,144 (32,768 when metadata
does not report a window), Nemotron 3.5 Lightning is configured at
1,048,576 regardless of the checkpoint's `max_position_embeddings` (the
Lightning mixer has no positional cap on this axis), and unknown families fall
back to 32,768. The router catalog adds `context_window`,
`suggested_client_context_window` and `max_output_tokens` to every entry; the
loaded server's memory-admitted window, reported on `/v1/models` and
`/health`, always wins over the estimate. Nemotron admission still refuses a
launch when the requested window does not fit its memory budget. Qwen's
maximum cold and incremental prefills remain 32,768 tokens each; the larger
context requires reusable state and is not a verified 262K cold prefill.
Neither model has been validated at 262K (Qwen) or 1M (Nemotron) prompts on
the target Mac.

