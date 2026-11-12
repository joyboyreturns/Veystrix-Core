# Core implementation ledger — 2026-11-11

This is a partial work ledger, not a release or performance claim. Work is based on main `68fc14b`, in an isolated worktree. The Qwen branch `fee112b` predates Nemotron integration; Nemotron branch `61c84fe` is merged into main; the remote docs branch is `a694831`. Existing worktrees were clean at inspection. No branches were merged or rewritten.

| Requirement | Source / plan | Validation / rollback | Status |
|---|---|---|---|
| Full-quality Qwen baseline and safety | `models/flashnext/settings/setting_routing.py`, `docs/flashnext/phase2-task-ledger.md` | Default threshold 1.0; historical 0.85 can discard top-10 routes. Layer-major remains opt-in. Repeat supervised multi-request tests before promotion. | Existing guard inspected; new real benchmarks pending |
| Nemotron real equivalence | `models/nemotron35/verify_real.py`, `docs/nemotron35/validation-results.md` | Compare real tensors and state under memory guards before performance work. | Pending; synthetic coverage is not real-weight proof |
| Model identity | `veystrix/server.py` | Reject mismatched explicit IDs before scheduling; synthetic Nemotron HTTP regression. Roll back the guard as one change if needed, but this restores unsafe silent routing. | Implemented for single loaded backend |
| Installed registry and switching | `veystrix/model_registry.py`, `veystrix/checkpoints.py`, `veystrix/server.py`, `veystrix/session.py` | Metadata-only inventory assigns unique IDs per canonical installation path; IDs change when moved and do not fingerprint replaced contents. Prove stronger checkpoint identity and serialized lifecycle before advertising unloaded models as runnable. | Partial: inventory detects both local checkpoints; server lists only loaded model, no switching |
| Harnesses | `integrations/hermes/`, `apps/macos/Veystrix/Sources/VeystrixKit/HarnessProfiles.swift` | Qwen Code snippet uses selected ID; JSON parse and unit coverage. Test installed clients and model transitions when lifecycle exists. | Snippet corrected; integrations pending |
| Chat and app | `apps/macos/Veystrix/` | Build, UI streaming/cancellation/history tests, packaging and launch checks. | Pending |
| Dynamic Duo | New persistent orchestration and harness adapters | Fault injection, exact identity, supervised end-to-end review, measured switch costs. | Not implemented |
| Release and branches | `README.md`, `docs/`, `packaging/` | Update branch-specific descriptions only after inspecting each branch; build and validate artifacts. | Pending; no public changes |

The historical overnight report records a 3.33 GiB swap breach during an 8K prefill after tool activity with auto-purge disabled. A later differently configured run passed, but is not proof that this incident is resolved. The later full-routing 8K-after-turns run did not establish repeated long-decode-then-8K safety. Polling cannot guarantee a hard swap ceiling.

The overnight warmer now sends the supervised model ID, and the manual warmer uses the ID reported by `/health`. This fixes the warm-up regression introduced by strict server model validation; tests use synthetic backends only.
