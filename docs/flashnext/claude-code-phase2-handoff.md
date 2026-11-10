# Claude Code handoff: Veystrix Phase II

This is a continuation prompt and working plan for the owner to give Claude Code. The previous Codex goal is **paused**. The mission is not complete. Read the entire document and the referenced files, inspect the actual repository and machine, and continue the work rather than reporting a plan alone.

## Prompt to give Claude Code

You are the principal inference-systems architect for Veystrix. Your objective is to make the existing Qwen3.8 Flash-Next engine genuinely useful for local agentic coding on the owner's fanless MacBook Air M4 with 16 GB unified memory. Research, implement, benchmark, debug and integrate changes autonomously. Make local commits when verified. Preserve the original checkpoint, user files, local Git history, and all current work. Do not push, force-push, rewrite history, alter `main`, delete user files, stop unrelated processes, manipulate swapfiles, disable OS protection, or introduce silent quality reductions.

The owner explicitly authorizes substantial architecture work: C/C++/Objective-C++, native I/O, MLX/Metal kernels, cache and scheduling changes, exact speculative decoding, controlled checkpoint tests, protocol and client integration, full regression suites, and local commits. Do not stop for routine approval. Ask only when truly necessary to proceed. Use a durable ledger so work survives context compaction or interruption.

**Updated power permission:** the owner now allows work and controlled model experiments while the Mac is on battery. This supersedes the earlier AC-only condition, including for the previously failing 8K-after-turns experiment. It does **not** waive any other requirement: check battery charge, power source, thermal health, Low Power Mode, disk space, memory headroom, pressure and swap before and during a run; use an independent watchdog; stop on the first serious anomaly. The existing tooling uses a 25% battery floor; use at least that floor, and choose a more conservative one if measurement or conditions warrant. Do not pretend battery and AC results are interchangeable: record source, charge, charging/discharging state and thermal conditions for every arm. Keep experiments bounded and avoid concurrent full-model workloads. The current untracked `bench_phase2_after_turns.py` hardcodes AC in multiple places; deliberately update and test it to support a recorded battery-allowed mode before using it. Never silently bypass its other safety checks.

The owner's non-negotiable constraints are full **top-10 expert routing**, original Qwen3.8-Flash-Next-MLX-4bit-MTP weights and numerical semantics, all 512 experts per layer, reasoning modes and tool calling, 262,144-token architectural context support, safe 32K cold/incremental admission boundaries, SSD/RAM prefix caches and cross-session restoration, streaming and cancellation, OpenAI Chat Completions and Responses, Anthropic Messages, Qwen Code, and DeepSeek Harness. Do not drop experts, change quantization, silently approximate logits, or claim an untested context length. Explicit legacy approximate switches exist, but are **not** acceptable as the candidate or the baseline for this mission.

The absolute system swap ceiling is **3,072 MiB**. It is a ceiling, not an operating target. Proactively reserve headroom, refuse/defer when unsafe, and stop at the first serious pressure anomaly. Do not use purge as the only safeguard, assume another application's swap is free, or retry a refused heavy request indefinitely. No polling system can guarantee catching every fast spike; demonstrate a defensible margin, independent monitoring, cleanup, and clear failure behavior before the previously failing live scenario. If preflight or monitoring is unsafe, stop the live experiment and continue useful offline work.

## Repository and current state

- Repository: `/Users/vincent/Documents/Veystrix`.
- Required branch: `development3.8`. At handoff it is ahead of `origin/development3.8` by 17 commits; do not push.
- Original checkpoint: `/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`. Do not modify it.
- Latest committed HEAD at handoff: `338b2c9` (`fix(flashnext): restore full expert routing and guard cache state`). Preceded by `b4f6a2c` and `a3a07d1` for offline Phase II work. Preserve all earlier local commits.
- The working tree is **dirty by design** because the user interrupted a running turn. Inspect `git status --short --branch`, `git diff`, `git diff --check`, and untracked files before touching anything. Do not reset or discard these changes. They may contain a partially written or untested benchmark from an interrupted agent.
- Last observed power state was battery, 74% and discharging; swap was about 878 MiB and disk had about 203 GiB available. These are **stale observations**, not a current preflight. Check again.
- The previous Codex goal is paused. Claude should execute the mission independently; the owner may provide this file using Claude's goal mechanism.

Read these first, then inspect source and raw benchmark artifacts:

1. `docs/flashnext/architecture-overhaul-phase2-results.md`
2. `docs/flashnext/phase2-task-ledger.md`
3. `docs/flashnext/overnight-architecture-results.md`
4. `docs/flashnext/architecture-overhaul.md`
5. `docs/flashnext/AGENT_INVARIANTS.md`
6. `docs/flashnext/handoff.md`
7. `docs/flashnext/research.md`
8. `docs/flashnext/phase2-reference-research.md`
9. `docs/measurement-standard.md`

The owner instructions in this document override older local documents' requirements to ask again for routine experiment permission or to require AC. Do not override the user's quality, safety, checkpoint, Git, or measurement requirements.

## Critical correction: historical performance used pruned routing

Do not treat the old 89.5 tok/s 8K prefill result or prior decode figures as verified full-model results. The former server default was `threshold=0.85`, a retained-probability-mass policy that can omit routed experts. The recorded current-checkpoint decode trace contains **4–9** expert IDs per layer call, not the required ten. The existing A/B exactness gates compared paths under the same pruned configuration, so those gates do not establish equivalence to full top-10 inference.

The earlier results remain useful engineering clues under that configuration: historical 8K cold prefill was 48.3 → 89.5 tok/s, 16K 44.7 → 69.2 tok/s, 8K physical prefill reads about 277–279 → 89 GB, and sustained decode roughly 2.1–2.7 → 2.6–3.5 tok/s. A prior diagnostic saw about 290 ms/token I/O wait and about 72 ms/token at router synchronization; these are coarse host-side, potentially overlapping measurements, **not** isolated GPU execution time. Previous cost estimates and admission budgets were calibrated on a different routing workload. Re-measure the full-routing checkpoint before using these numbers for speed, safety, or capacity claims.

Commit `338b2c9` changes the normal routing threshold default to **1.0**, adds warnings for explicitly selected legacy approximations, makes the supervised paired harness explicitly pass `--threshold 1.0`, and adds synthetic tests for ten distinct routes. Prefix snapshots include the routing profile in their fingerprint. A live routing-profile change invalidates the old cache and triggers transcript replay; a saved session with a different profile is refused instead of silently restoring numerically incompatible state. These measures have not been verified with the real checkpoint after the change. Verify full top-10 behavior through the actual launch path and model, not merely the setting name or synthetic test.

## What has been accomplished

The earlier architecture work is committed and should be preserved:

- Layer-major prefill using up to 8,192-token windows, two-half expert-ring overlap, trailing-small-chunk handling, and inner snapshots that preserve system-prefix boundaries.
- Exact persistent SSD prefix snapshot restoration, RAM/SSD prefix caches, buffered expert-pool fills, reversed-order paired benchmarks, decode profiling infrastructure, and supervised integration smoke tooling.
- Candidate request admission using fresh swap/pressure readings, estimated request, output and snapshot headroom, a default 3,072 MiB absolute ceiling with a 768 MiB margin, and a requirement for a fresh independent-watchdog heartbeat.
- An independent watchdog sampling swap directly through macOS `sysctlbyname`, default 0.25 seconds for the layer-major candidate. It uses identity-checked worker termination, fails closed on missing probes or a persistently unreadable progress record, and avoids lengthy stall diagnostics while memory monitoring is active. This is a backstop, **not** proof of a hard ceiling.
- Failure-path protection drains expert-pool fill writes and invalidates failed slots before reuse. N-gram per-lookup deduplication is an **unmeasured opt-in prototype** (`--enable ngram-dedup`), not a claimed performance win.
- Integration-smoke verdicts were strengthened to reject missing/malformed expected tool calls, SSE errors followed by `[DONE]`, and invalid Anthropic/Responses envelopes.

On the last committed full-routing tree, the full offline suites exited successfully: **1,791 FlashNext tests run, 3 skipped; 689 Veystrix tests run**. After a test-only isolation edit, 78 focused tests passed. These do not establish live model exactness, server memory safety, or real client compatibility.

## Uncommitted work at interruption — review before relying on it

At handoff the following modifications exist in addition to this handoff document:

- `veystrix/backends/flashnext.py` and `veystrix/tests/test_flashnext_backend.py`: routing-change replay preserves the previous `turn_closed` flag to avoid adding a duplicate end-of-message marker. The agent reported 13 focused tests passed.
- `models/flashnext/expert_cache.py`, `expert_pool.py`, and `tests/unit/test_expert_pool.py`: when the routed unique expert count exceeds pool capacity, use the streamed path rather than overwrite slots still needed by the same dispatch; direct oversized assignments fail before mutation. Review fallback arithmetic, G32 BF16 cast, route order, error handling and lifetimes. The agent reported 119 focused pool/ring tests passed; no checkpoint test.
- `models/flashnext/tests/bench/bench_decode_profile.py` and `tests/unit/test_decode_profile.py`: a `--buffered-fill` option, AC as the current default (which can be explicitly changed for your battery experiments), and a process physical-read delta divided by later output tokens rather than the smaller timed-tail count. This is process I/O and may include housekeeping, not expert-only bytes. It marks optional expert-I/O statistics unavailable when no layers were instrumented. A combined focused command completed with **37 tests, exit 0** before the interruption; inspect the precise changes and rerun suitable focused tests.
- `models/flashnext/tests/bench/bench_layer_gate.py` and `tests/unit/test_layer_gate_probe.py`: the interrupted safety agent was adding a strict proactive swap stop and faster strict-mode monitoring. Review the diff and tests. Strict-mode behavior is different from older historical `swap-only` runs; do not mislabel comparisons.
- Untracked `models/flashnext/tests/bench/bench_phase2_after_turns.py` and `models/flashnext/tests/unit/test_phase2_after_turns.py`: an **unfinished, unreviewed, never-run-on-checkpoint** bounded two-short-turns-then-cold-8K safety harness. The agent was interrupted while working on it. It currently hardcodes AC, uses estimated snapshot and allocation reserves that are not measured bounds, and delegates monitoring to `bench_layer_gate.run_arm`. Audit every assumption, test error/cleanup paths, support the newly authorized battery mode explicitly, and validate safe refusal before any live use. It should not be described as approved merely because its synthetic tests pass.

Inspect current state because files may have changed after this handoff. Run relevant focused tests, fix problems, rerun appropriate full suites after edits settle, document outcomes, and make verified local commits. Do not claim a performance improvement without real-checkpoint, full-routing evidence.

## Execution plan and acceptance gates

1. **Recover and verify.** Check branch, status, Git log, active processes, machine power/thermal/memory/disk state, the model path, local tool availability, and the uncommitted files. Read the listed documents and relevant code/results. Record changed source fingerprints and settings in any experiment. Resolve or safely preserve all interrupted work; never run two full-model workloads concurrently.
2. **Safety first.** Review admission estimates against full top-10 allocation, existing KV/recurrent state, MLX active/retained memory, expert and prefetch buffers, prefix snapshot/serialization staging, and other system pressure. Test high initial swap, missing/stale telemetry, cancellation, worker termination, cleanup, retry, and low headroom synthetically. Validate that independent watchdog liveness and its absolute stop are actually active, not merely configured. Observe power, pressure, available memory, compression, process RSS, MLX active/peak and recommended Metal working-set headroom as practical. Do not infer a hard ceiling from a polling interval.
3. **Validate full routing.** Confirm all ten experts are selected through the real server and checkpoint for representative prefill/decode routes; verify old pruned prefix/session state cannot be silently reused. Use deterministic token IDs, logits and state gates. If model or memory conditions make this unsafe, stop live work and continue offline.
4. **Re-establish the baseline.** Run controlled full-routing, same-machine comparisons with fresh processes, independent supervision, physical bytes, peak memory/swap, power/thermal conditions, exactness gates, interleaved/reversed order and enough arms to establish a resolution band. Start small, progress to 8K only when the safeguards and preflight support it, and stop immediately on anomaly. The previous 8K-after-decode breach peaked around 3.33 GiB with a five-second monitor; do not repeat that unsafe setup. Test 16K only after 8K safety is established and headroom is demonstrably adequate.
5. **Decode first.** Capture fresh coding and tool-use routes on the original checkpoint under full top-10 routing. Build an honest per-token/per-layer timeline that separates router/host synchronization, cache lookup, native read submission and wait, host-to-MLX exposure and copies, GPU dispatch and actual GPU execution, expert accumulation, dense/GDN work and token commit. Account for overlap instead of adding overlapping timers. Report physical SSD bytes and latency distributions, not only logical hits. Profile buffered and unbuffered pool fill separately, with unprofiled controls.
6. **Implement the best justified decode improvement.** Investigate native C++/Objective-C++ I/O and coordination, direct-slot execution, adaptive resident placement, scan-pollution resistance, demand read/compute overlap, and exact Metal kernel layouts/fusion. Prototype first and pursue an integrated native path if evidence is strong. A prior code audit found that simply running resident and missing experts separately and adding final outputs can change G32 rounding: the current fused reduction accumulates in float32 across route ranks and casts once. A promising untested direction is bounded per-rank scratch during reads followed by one ordered accumulation, with explicit buffer lifetime and GPU/CPU fencing. Do not increase the pool past a safe budget or lock memory without accounting for platform limits. Keep rollback paths and reject a candidate if exactness, latency, I/O, memory, state restoration, or streaming gets worse.
7. **Revisit exact speculative decoding only with a cost model.** Inspect `mtp.py` and `speculative.py`; test small adaptive draft depths, target-verification batches, expert unions, accepted target-verified tokens/second and rollback. Evaluate n-gram drafting where appropriate. Leave speculation disabled if it loses or correctness is not proven.
8. **Improve prefill and n-grams.** Preserve the layer-major and inner-snapshot design while revalidating its full-routing behavior. Investigate 4K/8K/hybrid or larger safe windows, repeated expert passes, late-window GPU/thermal cost, bounded expert-major superblocks and exact route-rank accumulation, n-gram duplicate/coalesced reads, and adaptive snapshot placement. Do not allocate an all-prompt intermediate that makes 16K/32K unsafe. Compare end-to-end memory and physical I/O, not only microbenchmarks.
9. **Run the long-session safety gate.** After synthetic checks and current-condition preflight, run the bounded earlier-turns → cold approximately 8K case through the normal supervised server, with independent watchdog, conservative request admission, and first-anomaly stops. Record the initial state, each turn, pre-8K admission decision, maximum swap, pressure/compression, cleanup, cancellation and persistent-state behavior. A safe refusal is a valid result; never force the request to run.
10. **Validate real workflows only after the relevant gate.** Test OpenAI Chat Completions and Responses, Anthropic Messages, streaming, tool calls/results, reasoning, cancellation, restart and cross-session restoration. Then run bounded Qwen Code and DeepSeek Harness sessions, including the real approximately 8.6K-token client prompt, sequentially. Record TTFT, output throughput, physical bytes, prefix hits, full task wall time and memory across turns. Do not claim 262K execution without a safe genuine run.
11. **Finish and document.** Run all applicable exactness gates and both full offline suites, plus relevant real-checkpoint/server tests. Re-run only where changes or unresolved failures justify it. Update `docs/flashnext/architecture-overhaul-phase2-results.md` with actual architecture, performance table, command lines, exactness and safety evidence, failed experiments, rollbacks, client configurations, local commits, and ranked executable outstanding tasks. Update the ledger. Commit verified work locally; do not push. Do not call the project production-ready solely because unit tests pass.

Relevant regression commands, from the repository root:

```bash
.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py'
.venv/bin/python -m unittest discover -s veystrix/tests -t . -p 'test_*.py'
```

Keep `-t .`: the Veystrix test package sets an isolated warm-cache directory. Older unqualified discovery failed the isolation assertion and must not be used as final evidence. Expect some test output to contain intentional failures from tests of gate reporting; use the actual unittest summary and process exit code.

## Research and measurement standards

Review the references in `phase2-reference-research.md` and the original owner mission, especially moe-offload, Vates, streamlx, Palm-Infra, Apple's SpecMD, Layered Prefill, MLX custom Metal kernel documentation, vLLM Metal offload, and MacPaw's Apple Silicon study. Check model architecture, quantization, license, MLX version and hardware differences before transferring ideas. In particular, moe-offload's public Qwen3.8 native path uses a different Q2 repack and different hardware and contains read-error-handling assumptions unsuitable for direct copying.

Use the project measurement standard: schema 1 records via `veystrix.results`/`MeasurementRun`, preserve raw arms, interruptions and validation failures, record checkpoint/source/environment/prompt/sampler/cache/provenance, exact token digests and state, physical bytes and memory, and reversed-order paired arms with a resolution band. Profiled runs may perturb synchronization; report diagnostic splits separately from unprofiled performance. Historical cache simulations are logical-demand models, not physical SSD results. Respect fanless thermal drift, avoid benchmarking until unsafe preflight conditions are resolved, and keep the checkpoint and user data unchanged.

## Commands and rollback context

Reference chunk-major server with the current full-routing default:

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2
```

Experimental layer-major candidate (not yet validated for a long-lived full-routing server):

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable layer-major --enable layer-prefetch
```

Experimental buffered decode fill, separate from the preceding unvalidated combination:

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable pool-buffered-fill
```

Do not call any of these the fastest validated full-routing mode until measured. `FLASHNEXT_REQUEST_ADMISSION=0` and `FLASHNEXT_WATCHDOG_MEMORY=0` are explicit **unsafe/unvalidated** rollback switches for the candidate; disabling a guard is not an acceptable way to pass a safety gate. The independent watchdog and candidate gate have not been validated together under the full-routing long-session workload.

For Qwen Code, follow the provider configuration in `docs/flashnext/qwen-code-integration-2026-11-09.md`; invoke with `VEYSTRIX_API_KEY=local qwen` after the server is ready and the safety gate passes. For DeepSeek Harness, the documented starting form is `DSH_HOME=<profile> VEYSTRIX_API_KEY=local dsh headless --patch docs/flashnext/dsh/veystrix-agentic.patch.yml "task"`. Neither has been validated with the new full-routing candidate.

## Communication and final report

Work autonomously and report concrete progress and blockers. Clearly distinguish measured facts, simulations, estimates and unvalidated code. If a target cannot be reached within the hardware and safe-memory budget, show the measured limiting cost and retain the fastest verified architecture rather than an approximate substitute. Keep the full goal open until the requested end state has actually been established.
