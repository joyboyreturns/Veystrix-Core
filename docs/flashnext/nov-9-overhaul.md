# November 9 Veystrix overhaul: Claude Code handoff

This is a handoff, not a claim that the overhaul is finished. It was prepared on November 9, 2026 from the local worktree at
`/Users/vincent/Documents/Veystrix`. The original owner directive is reproduced in full at the end of this document. Inspect
the current tree and results again before acting: this document is a snapshot and later work may have changed them.

## Prompt to give Claude Code

Paste the following into Claude Code from the repository root:

```text
Read docs/flashnext/nov-9-overhaul.md in full, including the complete original owner directive in its appendix. Then inspect
the actual current worktree and the evidence it references. Continue the Veystrix architecture overhaul described there.

Preserve all existing uncommitted changes and user data. Do not start over, reset the worktree, delete caches, change the
checkpoint, modify main, force-push, or silently promote an experimental path. The owner previously authorized autonomous
implementation, checkpoint-free tests, and bounded, supervised real-checkpoint experiments under the original directive;
follow the full safety and measurement requirements in that directive and the applicable project documents.

The latest real-checkpoint gate was stopped by the external memory-safety monitor. Do not automatically retry that run or
weaken its guard. Review the recorded stop, explain the new evidence and safety premise for any future live experiment, and
continue useful offline implementation and analysis in the meantime. A stopped or incomplete arm is not a correctness or
performance result. Never bypass exactness, pretend a simulated result is a real measurement, or describe an unverified
experiment as an improvement.

Work on both prefill and sustained decode, plus normal-server and coding-client integration. Keep chunk-major execution as
the reference and all unproven alternatives opt-in. Preserve the native 262,144-token architectural limit, separate 32,768-token
cold/incremental admission ceilings, exact prefix and persistent caches, reasoning continuity, tool and streaming protocols,
watchdogs, cancellation and memory controls. Do not treat a configured 262K limit as a real-checkpoint 262K validation.

Use the project test terminal and append-only measurement standard. For any appropriately authorized and safe live work,
capture the checkpoint, commit, worktree, source fingerprint, configuration, power, thermal and memory state; use the
supervised AGENTIC launcher, its out-of-process watchdog and +2.5 GiB swap-growth stop. Use fresh processes and at least
three paired, reversed-order arms, preserve the 32-token comparison horizon, require exact token/state evidence, report raw
arms and resolution bands, and distinguish physical SSD bytes from staged bytes and warm file-cache effects. Never bypass a
failed or incomplete correctness gate to run a performance comparison.

Investigate, implement, test and integrate rather than stopping at a proposal. Revalidate the current worktree, make changes
only within the owner's authorized scope, and report measured results and remaining gaps candidly. The objective and
priorities, research sources, earlier findings, exact stop, tests, artifacts and outstanding deliverables are in the handoff.
```

Pasting this prompt is intended to request continuation; the owner's original scope and its safety constraints still apply.
The previous Codex goal was cleared before this handoff was requested. Creating this document did not restart the overhaul.

## Objective and non-negotiable limits

The owner wants Qwen3.8-Flash-Next to be genuinely useful for agentic coding on a **MacBook Air M4, 16 GB unified memory**.
The requested outcome is a substantial, measured improvement in both cold prefill and sustained decode, while retaining the
original 4-bit model, correctness, reasoning, 262K architectural context, caches, server protocols, and coding clients.

Targets are 80–100 tokens/s for 8K–16K cold prefill and 5 tokens/s sustained decode, with harder goals of 150+ prefill and
8–10+ decode. These are aspirations, not measured outcomes. A longer-term 300 tokens/s prefill aspiration is subject to a
credible physical cost model. Cached follow-ups should improve as well.

The original owner directive explicitly authorizes ordinary project changes, offline work, controlled checkpoint experiments,
and local commits; it supersedes historical per-benchmark approval language in project documents, but **does not waive machine
safety, correctness gates, the measurement protocol, or the instruction not to retry automatically after a safety stop**.
It prohibits lowering model precision, changing dimensions or routed experts, weakening reasoning, disabling swap protection,
deleting user data, killing unrelated processes, modifying the original checkpoint, force-pushing, rewriting history, or
modifying `main`. It asks for controlled, non-continuous load on this fanless computer. It allows local commits only; no commit
or push was made during the work summarized here.

## Current worktree and where to start

- Branch at handoff: `development3.8`, tracking `origin/development3.8`.
- Inspected base commit: `e48b64bd0fca4ab9605d35dc5b7dfcc5af2112f2`.
- Checkpoint used in the attempted gate: `/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`.
- The worktree is dirty, with pre-existing changes from earlier agents as well as the later changes below. Preserve all of them.
  Run `git status --short --branch`, `git diff`, and inspect untracked files; `git diff --stat` alone does not include untracked
  code, documents, or result folders. The current state, not this list, is authoritative.
- At handoff the modified tracked files included `macqwen/backends/flashnext.py`, `macqwen/cli.py`, `macqwen/server.py`,
  corresponding tests, `models/flashnext/chat_guard.py`, `expert_cache.py`, `overnight.py`, `prefill.py`, `prefix_session.py`,
  settings and tests, and `serve.sh`. Untracked work included the architecture document, layer-major implementation, gate and
  replay harnesses, tests, terminal cases, and result directories. None should be assumed disposable.

Read these first, and treat claims in older reports as historical until checked against code and evidence:

1. `docs/flashnext/architecture-overhaul.md` — current design, caveats, results, and the stopped measurement.
2. `docs/flashnext/AGENT_INVARIANTS.md` — runtime, numerical, interpretation, and measurement invariants. Some historical
   per-run approval language is superseded by the owner's explicit original directive reproduced below; safety is not.
3. `docs/flashnext/handoff.md` and `docs/flashnext/research.md` — prior experiments, negative findings, and controls.
4. `docs/flashnext/performance-engineering-report-2026-11-08.md`.
5. `docs/flashnext/flashnext-recovery-qwen-code-262k-report-2026-11-09.md`.
6. `docs/measurement-standard.md`, `docs/testing.md`, `docs/flashnext/modes.md`, and
   `docs/flashnext/universal-agentic-api.md`.

There are historical inconsistencies: older reports mention 16,384-token prefill defaults while current code and the owner's
directive specify separate 32,768-token cold and incremental admission ceilings. Older documents also disagree on the extent
of Qwen Code and DeepSeek Harness validation. Check the code, current configuration, and actual test artifacts; do not turn a
synthetic protocol test or an advertised limit into a full live client or 262K-context claim.

## Architecture already present

The existing Flash-Next path processes chunks through all layers. The target model is described in the local architecture report
as 48 layers: 36 GatedDeltaNet and 12 sparse-attention layers, hidden size 2,560 with four hyper-connection streams, 512 routed
experts per layer and top-10 routing. Its reported expert record size is 3,072,000 bytes. Confirm these facts from the actual
checkpoint and loader when they affect an experiment.

The current proposal and implementation separate compute, streamed expert data, persistent model state, and control/safety.
Important existing components include `models/flashnext/store.py`, `expert_pool.py`, `prefill_ring.py`, `pool_ring.py`,
`prefix_cache.py`, `prefix_session.py`, `memory_controller.py`, the server watchdog, and `macqwen/backends/flashnext.py`.
The normal server path and its cache and reasoning behavior must remain usable.

### Layer-major prefill

`models/flashnext/layer_major.py` contains a window planner, `ExpertWindow`, a layer-major window driver, and a rough cost model.
`models/flashnext/expert_cache.py` has an opt-in `_LAYER_WINDOW` hook that can stage all experts in a layer once and reuse them
across prompt chunks. `prefill.py` has the layer-order branch and the normal head path; `prefix_session.py` records execution
provenance and includes it in snapshot compatibility. A chunk's causal cache order is intended to remain unchanged within each
layer. Small-model tests cover attention state, recurrent state, hidden states, chunk shapes, restored prefixes and real-sized
streamed expert operations. **These are checkpoint-free results, not real-model proof.**

The normal mode remains chunk-major. Experimental launch options are:

- `--enable layer-major`: layer-major with an 8,192-token window.
- `--enable layer-major-4k`: layer-major with a 4,096-token window.
- `--enable layer-major-hybrid-4k`: an experimental 4K window with the original chunk-major path for windows of at most
  2,048 tokens. The cutoff is a heuristic; neither traffic reduction nor speed has been demonstrated on the real model.

Use `./serve.sh` for supervised experiments. The mode owns the relevant environment flags, so merely exporting a window value
may not override a mode preset; inspect `models/flashnext/settings/modes.py`. Do not enable more than one of these options.
The hybrid execution plan has a distinct `layer-major-v1/w4096/hybrid-small-v1` cache identity. Chunk-major retains its old
identity; incompatible snapshots must not be mixed or deleted simply to make testing easier.

Pure layer-major snapshots occur at window boundaries, so an interruption can lose more work than chunk-major. The hybrid
reference-path windows retain completed chunk boundaries. Adaptive chunk re-planning inside a layer-major window is not built.
Potential extra activation memory, mixed causal attention, GatedDeltaNet state, position state, I/O and cancellation must be
measured or verified rather than assumed safe.

Recent safety correction: on an exception in a layer-major window, `ExpertWindow.abort_layer()` now fences the staged expert
ring before retiring its lease. If the fence fails, operands are retained for the enclosing ring's failure handling rather
than marking a potentially active buffer reusable. Focused tests exercise failure, cancellation, fencing order, and an actual
staged test ring. The latest focused layer-major run passed **42 tests** after this change. The full suites below were run before
this final ring-cleanup change, so run the appropriate suites again after further changes.

Another correction makes each `WindowReport` subtract prior cumulative `ExpertWindow` counters, avoiding double counting
earlier windows. Staged bytes are not physical SSD bytes.

### Gate and instrumentation

- `models/flashnext/tests/bench/bench_layer_gate.py` and `layer_gate_probe.py` are opt-in diagnostic infrastructure.
- Terminal cases: `models/flashnext/tests/cases/case_layer_gate.py` (`layer-major-gate`) and
  `case_layer_hybrid_gate.py` (`layer-hybrid-gate`). The project terminal is `./tests/run.sh`.
- The intended gate launches fresh supervised AGENTIC servers with isolated temporary prefix caches and preferences, greedy
  decoding, a 0 GB pool in both arms, the same checkpoint, and reversed-order pairs. It checks that the independent watchdog
  is live, checks the server's +2.5 GiB swap-growth threshold, and monitors AC/battery, Low Power Mode, thermal, memory pressure,
  available memory, compressor, and swap. It stops at the first safety or exactness failure.
- It records prompt identity, the first selected token, 32 generated token IDs, final-row logit hash, all supported attention
  and recurrent cache-array hashes, position metadata, chunk edges and execution order. The comparison fails closed on missing
  or malformed evidence and verifies that a multi-chunk layer-major segment really ran. Physical read and MLX peak counters are
  recorded, but probe overhead and uncontrolled OS file-cache state make this a **correctness diagnostic**, not a cold-throughput
  promotion result. The state probe hashes exported buffers without an extra unconditional full-size `tobytes()` copy.
- The 32-token gate includes a tool schema as prompt text. It does **not** exercise a live tool call: the server rejects actual
  tool-enabled requests with fewer than 64 output tokens. Tool-use validation remains a separate required test with its own
  baseline and horizon.
- `--no-auto-purge` was fixed to override an inherited `FLASHNEXT_AUTO_PURGE=1`. The supervisor gained optional isolated
  preferences/prefix paths, a private supervisor log directory via `VEYSTRIX_SUPERVISOR_LOG_DIR`, and a no-auto-purge forwarding
  option. The gate uses these to avoid changing the owner's preferences and persistent cache. Read their implementations and
  tests before modifying them.
- The gate's current extra stop rules record each safety sample and refuse below 3 GiB available, or when compressor growth
  exceeds 2 GiB and available memory is below 4 GiB. These are additional protections, not replacements for the server's own
  guard. Do not remove them to force a result.

## Live measurement attempts and safety stop

The following are recorded attempts, not a successful comparison:

1. `results/flashnext/20261109-185204-layer-major-gate/`: server and watchdog started correctly; the API rejected the request
   before inference because an active tool schema with a 32-token output cap violated the 64-token tool minimum.
2. `results/flashnext/20261109-185343-layer-major-gate/`: server started, but the diagnostic rejected the request before
   inference because FlashNext sampling was not greedy. The isolated preferences were corrected afterward.
3. `results/flashnext/20261109-185459-layer-major-gate/`: a supervised, greedy reference arm completed 2,739-token prefill
   and generated 32 tokens, but the external safety monitor stopped the run around the end of decoding after available memory
   fell below its then-3,000 MiB floor. No candidate arm ran. The supervisor, model process and independent watchdog exited.

The third attempt recorded 90.37 seconds prefill, 137.25 GB of process physical reads during prefill, 4.91 GiB MLX peak, and
32 generated tokens in 13.51 seconds. The copied `reference-safety-trace.jsonl` records a 2,693.5 MiB minimum available-memory
sample during decode, a 6,363.6 MiB compressor peak during prefill versus 244.2 MiB at startup, and 842 swapout pages during
the arm. Swap usage did not exceed the +2.5 GiB threshold. These are system-wide observations; they do not identify the cause
of the compression. The raw state probe and `layer-gate.jsonl` are in the result folder.

The attempt was recorded as a **resource stop and incomplete comparison**, not a pass, fail of numerical correctness, or
promotion-quality rate. Its reference probe predates the later final-row-logit capture and cannot supply a complete pair for
the strengthened gate. The OS file cache was not controlled as a cold benchmark. Do not call the result a layer-major gain or
use it to skip the gate. In accordance with the owner directive, no automatic retry followed the safety stop.

## Historical baselines and physical model

The architecture document cites prior supervised runs of approximately 54.2, 47.0 and 48.1 tokens/s at 4K, 8K and 16K, and
earlier pool-off decode around 3.04 tokens/s. These are historical numbers with their own workloads and controls, not an A/B
against the new implementation. Check their source records and provenance before quoting them. The older 2026-11-09 recovery
report also reports 3,954 / 8,309 / 15,819-token stages with 54.16 / 47.03 / 48.08 tokens/s; reconcile exact workload and
measurement definitions before comparing them to anything new.

The local cost model estimates one complete 48-layer expert pass at ~75.5 GB and logical routed decode traffic at ~1.47 GB per
token before hits. It predicts significant savings from expert reuse over multiple chunks, but assumptions about SSD bandwidth,
expert unions, attention growth, activation staging, snapshots, cache state, and overlap must be measured. The implementation
does not yet have double-buffered overlap. A synthetic earlier test reported that strided expert-record views caused
`gather_qmm` materialization while compact part-major views avoided that penalty; it does not prove every real execution is
copy-free. Inspect `store.py`, `prefill_ring.py`, `pool_ring.py`, NumPy/MLX/DLPack conversion and actual installed MLX behavior.

## Decode and speculative work still required

No new production decode speedup was implemented or demonstrated in this handoff. Do not present simulation as physical I/O.
The owner requires independent investigation of actual bytes per generated token, safe pool sizes and physical hit/miss rates,
admission/eviction and per-layer residency, Python routing and synchronization cost, native I/O/execution where justified,
kernel fusion, routing-aware prefetch, and exact speculative verification. The 1 GB global-LRU regime previously thrashed;
do not reuse it as a supposed win without a distinct measured policy and workload.

`models/flashnext/tests/bench/bench_cache_holdout.py` replays an **older, different checkpoint's** existing route trace. It
trains static frequency pins on turns 0–1 and evaluates turn 2, starting simulated caches empty. It reports simulated hits and
estimated record traffic, not measured physical reads or latency:

| Simulated slots | Global LRU | ARC | Equal per-layer LRU | Static frequency pins |
| ---: | ---: | ---: | ---: | ---: |
| 311 | 0.0% | 28.6% | 34.5% | 11.0% |
| 622 | 48.0% | 50.4% | 51.7% | 13.5% |
| 1,245 | 64.7% | 64.4% | 65.5% | 17.0% |

See `results/flashnext/20261109-190028-cache-holdout-manual/cache-holdout.json` and
`results/flashnext/20261109-190616-cache-holdout-manual/cache-holdout.json`. Static pins' cold fill is counted separately.
At 311 slots some per-layer partitions cannot contain a whole event's expert set; the simulation assumes separate transient
storage but does not account for its memory or I/O. The 1,245-slot/4 GB point is an offline simulation, not evidence that such
a pool is safe on this Mac. No new eviction policy was promoted. A current-checkpoint route trace and actual physical miss
measurement are needed before deciding whether the per-layer premise is useful.

The same older trace was used for idealized verification blocks of 1–7 target positions (corresponding to draft depths 0–6),
assuming perfect acceptance and free drafts. For a 622-slot simulated global LRU it estimated 643, 694, 844, 769, 700, 666,
and 636 MB of demand records per output token, respectively. Union ordering, cache state, actual acceptance, draft overhead,
GPU verification and rollback all matter; these are neither target-checkpoint measurements nor actual physical-byte bounds.
Review `models/flashnext/mtp.py`, `speculative.py`, the native checkpoint, exact verifier and rollback tests. An older recorded
depth-3 MTP comparison on its own workload produced identical short digests but ran ~33.6%–42.4% slower and had ~55.6%
acceptance; it does not settle all depths, prompts or checkpoints. Keep native MTP off by default until current evidence supports
a change. Preserve both greedy and sampled target semantics; approximate drafting is not equivalent to exact target execution.

## Research sources requested by the owner

Study algorithms and implementations where available, and assess compatibility rather than transferring reported speed numbers:

- MLSys 2026 Layered Prefill: https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html
  — layer-group scheduling and prefill/decode interleaving. The current windowed layer-major prototype is not that complete
  concurrent scheduler. Compare chunk-major, pure layer-major, a hybrid or wavefront, and adaptive scheduling.
- Qwen MoE offload: https://github.com/Jakevin/moe-offload — native SSD workers, per-layer caches, router-directed prefetch,
  C++ execution and quantized kernels. Do not substitute its Q2 weights or treat its different hardware as a benchmark target.
- Apple SpecMD: https://machinelearning.apple.com/research/specmd-expert-prefetching — Least-Stale and layer/cycle-aware
  caching. Test against recorded physical misses before implementing a production policy.
- FlashMoE: https://arxiv.org/abs/2601.17063 — recency/frequency, residency and SSD avoidance; learned policies require
  justified training and runtime costs.
- SP-MoE: https://arxiv.org/abs/2510.10302 — draft-aware expert prefetch and verification pipelining.
- Vates: https://github.com/AMOS144/Vates — native out-of-core and speculative execution, on a different model/configuration.
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html — especially implicit
  contiguous copies, strides, compilation caching and dependency safety. Check the installed MLX version.

## Validation state and next work

Observed checkpoint-free test results:

- Complete `macqwen` discovery: **686 tests, OK** (`/private/tmp/veystrix-macqwen-final.log`).
- Complete FlashNext unit discovery: **1,687 tests, OK, 3 skipped** (`/private/tmp/veystrix-flashnext-final.log`). The full
  run preceded the final gate-record validation tightening and ring-abort cleanup. Focused tests passed after those changes;
  the latest focused layer-major run was **42 tests, OK**. Temporary logs may disappear; rerun after changes.
- A prior full-suite invocation without `-t .` missed test package initialization and failed its warm-cache safety assertion;
  the correctly initialized rerun passed. The negative `quality_compare.py` output and battery/guard warnings in some tests
  are deliberate fixtures; inspect exit status and summary, not isolated output lines.

Useful checkpoint-free commands from the repository root:

```bash
.venv/bin/python -m unittest discover -s macqwen/tests -t .
.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t .
.venv/bin/python -m unittest models.flashnext.tests.unit.test_layer_major
.venv/bin/python -m unittest models.flashnext.tests.unit.test_layer_gate_probe models.flashnext.tests.unit.test_cache_holdout
git diff --check
```

For future live work, inspect the current machine and the exact test case before starting. The earlier attempt was stopped for
resources; do not simply rerun or reduce safeguards. Establish a genuinely new, safe premise and follow the original directive
and project stop rules. If the exact state/logit/token gate cannot be safely completed, keep the candidate disabled and continue
other useful offline work; do not treat lack of a result as proof of equivalence or impossibility.

The requested final integration audit remains open: independently validate cold and warm prefill, RAM prefix hits, persistent
SSD hits and restart restoration, decode throughput, model state, reasoning continuity, tool-use turns, cancel/resume,
memory-limit handling, OpenAI Chat/Responses and Anthropic compatibility, Qwen Code and DeepSeek Harness. Include real coding
instructions, tool schemas and repository content. The two gate cases only contain a schema as text and do not replace actual
tool-call or client tests. A 262K advertised architecture is not evidence of a real 262K conversation. Verify startup and client
connection instructions against the current installations; `./serve.sh` is the supported supervised server launcher and the
normal local API is documented at `http://127.0.0.1:8080/v1`, but do not claim clients work until tested.

The owner ultimately requests a measured baseline-versus-optimized table for prefill, decode, TTFT, physical bytes, cache hit
rate and memory; a physical I/O/compute breakdown; unsuccessful attempts and limitations; safe defaults/rollback flags; startup
and client instructions; passing complete suites; updated architecture documentation; and a concise final engineering report.
No claim of measured improvement for the new layer-major or hybrid paths is supported by the available evidence.

## Appendix: original owner directive

The following is the original pasted directive, reproduced verbatim for continuity. It is historical user-provided task text;
read it alongside the actual current worktree and the safety-stop evidence above.

---

/goal

# VEYSTRIX — COMPLETE THE AUTONOMOUS PERFORMANCE ARCHITECTURE OVERHAUL

## PRIMARY DIRECTIVE

You are now responsible for completing the entire Veystrix architecture overhaul, not merely designing or partially implementing it.

**The owner explicitly authorizes you to proceed autonomously with implementation, controlled real-checkpoint benchmarking, optimization, regression testing, and final integration.**

Do not repeatedly ask for approval to continue, benchmark, test, or implement ordinary project changes.

Be highly ambitious, optimistic, technically creative, and persistent. Assume there are still meaningful performance improvements to discover.

Do not conclude that a performance target is impossible simply because the first approach fails. Explore alternative execution strategies, kernels, memory layouts, and I/O architectures.

However, distinguish engineering ambition from measured facts. Never fabricate performance improvements, compromise the machine's safety, or claim success without evidence.

Your central goal is:

**Make Qwen3.8-Flash-Next genuinely useful for agentic coding on the user's MacBook Air M4 with 16 GB unified memory by substantially accelerating both cold prefill and sustained decoding.**

Finish the work rather than returning another proposal.

---

## 1. RESUME THE EXISTING WORK

The previous agent reported:

- `macqwen`: 683 tests passed.
- `flashnext`: 1,673 tests passed, 3 skipped.
- The apparently failing output from `quality_compare.py` is an intentional negative-test fixture.
- The battery-warning output is also a deliberate safety test.
- The architecture-overhaul document already exists locally.
- Real-checkpoint A/B validation has not yet been performed.
- No work has been committed or pushed.

Treat these reports as a starting point, not proof that the latest working tree is still passing.

**Do not start over.**

Inspect and preserve every existing local modification.

Read:

- `docs/flashnext/architecture-overhaul.md`
- `docs/flashnext/AGENT_INVARIANTS.md`
- `docs/flashnext/handoff.md`
- `docs/flashnext/research.md`
- `docs/flashnext/performance-engineering-report-2026-11-08.md`
- `docs/flashnext/flashnext-recovery-qwen-code-262k-report-2026-11-09.md`
- `docs/measurement-standard.md`

Determine which architectural features are already implemented, which are only prototypes, and which still need integration.

Preserve working cache, reasoning, 262K-context, and harness functionality.

---

## 2. RESEARCH-DRIVEN ARCHITECTURE WORK

Use these specific external research sources as starting points. Examine their algorithms and, where possible, implementations.

### A. Layered prefill — MLSys 2026

https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html

Investigate using layer-oriented processing to reduce repeatedly loading MoE experts when prefilling multiple prompt chunks.

Continue the existing Veystrix layer-major implementation.

Compare:

1. Existing chunk-major prefill.
2. Pure layer-major prefill.
3. Hybrid layer-group or wavefront prefill.
4. Adaptive scheduling based on memory and prompt length.

Account for hidden-state staging, mixed causal attention, recurrent-state dependencies, and all extra intermediate I/O.

Do not assume layer-major will be faster. Demonstrate that its reduction in expert reads exceeds the cost of storing and restoring intermediate activations.

Preserve the model's numerical and causal correctness.

### B. Qwen3.8 MoE offloading

https://github.com/Jakevin/moe-offload

This is particularly relevant because it implements Qwen3.8 expert offloading on Apple Silicon.

Study:

- Native C/C++ SSD read workers.
- Per-layer expert caches.
- Reduced Python and MLX synchronization.
- Router-directed prefetching.
- MLX C++ execution.
- Custom quantized expert execution.

Its checkpoints and hardware differ from ours. Preserve Veystrix's actual 4-bit weights and model behavior.

Do not copy the alternative Q2 quantization or use its reported 64 GB performance as an expected result on our 16 GB machine.

Explore whether moving genuinely hot control-path operations from Python into native code reduces measured latency.

### C. Apple's SpecMD

https://machinelearning.apple.com/research/specmd-expert-prefetching

Study its Least-Stale expert-cache eviction approach.

Compare its ideas against Veystrix's existing LRU and previously tested offline policies.

Only implement a new cache policy if it demonstrably reduces physical expert misses or execution time on our recorded workloads.

### D. FlashMoE

https://arxiv.org/abs/2601.17063

Investigate recency-frequency-based expert replacement, adaptive per-layer residency, and SSD I/O avoidance.

Evaluate whether lightweight policies can approach the advantages of more complex learned policies.

Do not introduce a trained cache predictor unless the execution and training costs are justified.

### E. SP-MoE

https://arxiv.org/abs/2510.10302

Investigate draft-aware expert prefetch, bounded prefetch distance, and pipelined verification.

### F. Vates

https://github.com/AMOS144/Vates

Study its native out-of-core data path and speculative execution.

It targets a different MoE checkpoint. Transfer implementation ideas only after validating architectural compatibility.

### G. MLX custom Metal kernels

https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

Pay special attention to implicit contiguous copies, compile caching, tensor stride handling, and Metal execution dependencies.

Verify compatibility with Veystrix's installed MLX version.

---

## 3. IMMEDIATE NEXT ACTION: COMPLETE LAYER-MAJOR VERIFICATION

The previous agent already proposed a digest gate at 2K–4K tokens, followed by an 8K A/B comparison.

**I now authorize that measurement plan.**

Execute it autonomously, subject to all existing safety constraints.

Before running:

- Confirm AC power and acceptable battery state.
- Confirm a healthy machine, available storage, and thermal conditions.
- Confirm the supervised AGENTIC launcher is used.
- Confirm the out-of-process watchdog is active.
- Confirm the +2.5 GiB swap-growth stop remains enabled.
- Verify the baseline and candidate use the identical checkpoint.
- Capture the baseline commit, worktree state, configuration, and source fingerprint.

Run the smallest digest and state-equivalence gate first.

Check:

- Token IDs and logits where relevant.
- Attention cache correctness.
- GatedDeltaNet recurrent state.
- Chunk-boundary semantics.
- Numerical deviations and their downstream effects.
- Physical SSD bytes.
- Memory peak.
- Correct cancellation and cleanup.

If the gate passes, execute the 8K A/B using the project's paired, reversed-order measurement rules.

If it fails, diagnose the divergence instead of abandoning the architecture immediately.

Try to isolate chunk-shape numerical drift from genuine causal or execution bugs.

Do not silently relax exactness or quality gates.

If a variant cannot preserve the required semantics, retain the old path as the production reference and seek a different speedup.

---

## 4. PREFILL SPEED: PURSUE MULTIPLE ARCHITECTURAL LEVERS

Do not assume the layer-major approach alone will reach the performance target.

Research and implement promising complementary improvements.

### 4.1 Reuse experts across prompt blocks

Reduce redundant reads of identical experts across chunks or layer groups.

Prefer bounded, measurable reuse over speculative caching that consumes too much unified memory.

### 4.2 Improve expert execution

Investigate whether prefill can use more efficient custom Metal kernels for quantized MoE operations.

Optimize routing layout, gathers, dequantization, matrix operations, and reductions where necessary.

Avoid changing BF16 numerical boundaries without explicit correctness validation.

### 4.3 Reduce data copies

Inspect all conversions between SSD, NumPy, mmap, MLX, and Metal.

Prove whether allegedly zero-copy buffers are actually copied internally.

Investigate reusable, properly aligned buffers and stable kernel-compatible layouts.

### 4.4 Improve I/O scheduling

Compare the current Python worker pool against a minimal native I/O implementation.

Evaluate request batching, scatter/gather reads, file-layout locality, and memory ownership.

Distinguish buffered reads from `F_NOCACHE` behavior. Bypassing the page cache is not always faster.

Use overlap only where real execution dependencies allow it.

### 4.5 Improve long-prefill scheduling

Implement adaptive execution policies that select the fastest safe strategy for a given prompt length, memory state, and cache state.

Do not impose a single execution strategy if short and long prompts benefit from different designs.

---

## 5. DECODE SPEED: MANDATORY SECOND WORKSTREAM

**The overhaul is not complete until decode performance has also been addressed.**

Currently, sustained decoding is only a few tokens per second.

Explore the following independently and then in validated combinations.

### 5.1 Physical I/O reduction

Profile bytes read per generated token.

Measure actual expert-cache hit rates at different safe pool sizes.

Evaluate improved cache admission, per-layer residency, and eviction.

Avoid the previously observed ineffective 1 GB LRU regime unless a different tested policy makes it useful.

### 5.2 Native execution hot path

Profile Python overhead, route extraction, synchronization, Metal dispatch, and host-to-device staging.

Investigate moving bottleneck-critical work into native C++/Objective-C++/Metal or MLX C++.

Do not rewrite everything in C++ without evidence. Replace the measured expensive parts first.

### 5.3 Kernel fusion

Investigate fused routing-related operations, quantized projections, shared-expert computation, and compatible normalization or activation operations.

Measure the effects of each change on the critical path.

### 5.4 Routing-aware prefetch

Use recorded traces and a prediction-cost model.

Avoid repeating earlier unsuccessful one-layer look-ahead experiments unchanged.

Only implement a different prediction method if it offers demonstrably better recall, lower overhead, or a genuine new overlap window.

### 5.5 Multi-token speculative verification

Inspect `mtp.py`, `speculative.py`, and the native MTP checkpoint.

Research and test verification depths of 2, 3, 4, 5, and 6 as appropriate.

Measure:

- Draft cost.
- Actual accepted tokens.
- Expert unions per verification block.
- Additional physical SSD reads.
- Verification GPU cost.
- Cache rollback overhead.
- Net accepted tokens per second.

Account for the fact that MoE verification may activate many distinct experts across drafted positions.

Preserve correct greedy and sampled decoding semantics. Do not approximate target verification or silently alter output quality.

If native MTP is too expensive, investigate other exact approaches, including efficient n-gram drafting backed by target-model verification.

Never present a faster approximate draft as equivalent to exact model execution.

---

## 6. PERFORMANCE TARGETS

Use these as progressively harder goals.

| Metric | Initial goal | Ambitious goal |
|---|---|---|
| 8K–16K cold prefill | 80–100 tok/s | 150+ tok/s |
| Sustained decode | 5 tok/s | 8–10+ tok/s |
| Cached follow-up | Substantially lower TTFT | Near-immediate prefix restoration followed by faster suffix processing |

The longer-term aspiration is 300 tok/s prefill, but only pursue it where a physical cost model supports that possibility.

Do not stop when an initial target is missed.

If one approach fails, investigate another meaningful bottleneck.

Continue until both prefill and decode have independently demonstrated repeatable improvements or all reasonable avenues within the authorized safety budget have been evaluated.

Do not manipulate measurement definitions or report warm-cache speed as cold throughput.

---

## 7. AUTONOMY AND SAFETY AUTHORIZATION

You have standing approval for:

- Code inspection and research.
- Modifying the inference architecture.
- Building experimental execution paths.
- Creating native kernels.
- Running checkpoint-free tests.
- Running short, controlled real-checkpoint performance experiments.
- Running the approved 2K–4K digest gate and 8K A/B.
- Running controlled decode A/B tests.
- Additional bounded tests that follow logically from successful gates.
- Updating tests and documentation.
- Creating local Git commits containing completed, verified work.

You do **not** need to ask me for each phase.

This approval supersedes the project's historical requirement for obtaining permission before each new real-model benchmark, **but does not remove any machine-safety or correctness rule**.

Use the existing measurement protocol. Avoid redundant, prolonged, or wasteful experiments.

If the machine is on battery, under serious memory pressure, thermally unhealthy, or otherwise unsafe, postpone the affected live experiment and continue with offline implementation.

Never disable swap protection, force the OS into an unsafe memory state, delete user data, terminate unrelated processes, or change the original checkpoint.

Do not force success by reducing model precision, model dimensions, routed experts, or reasoning behavior.

Do not bypass a failed correctness gate.

Do not use continuous maximum-load testing on this fanless MacBook.

You may make local commits, but do not force-push, rewrite history, or modify the `main` branch. Preserve the original project and existing development work.

---

## 8. DO NOT DECLARE VICTORY TOO EARLY

Passing all unit tests is necessary but not sufficient.

New experimental features existing in the codebase is also not sufficient.

An architectural improvement counts as complete only when:

1. Its implementation actually runs on the target model.
2. Its output and state meet the applicable correctness requirements.
3. Its memory behavior is safe.
4. Its performance has been measured against a valid baseline.
5. It has a rollback path.
6. It integrates with the normal inference server.
7. Qwen Code and DeepSeek Harness remain compatible.

Both prefill and decode need independent evaluation.

If a candidate loses, keep its useful research findings, disable the losing path, and proceed to the next high-value approach.

Do not simply finish the layer-major experiment and stop.

---

## 9. AGENTIC CODING AND LONG-CONTEXT INVARIANTS

Preserve:

- Native 262,144-token architectural context.
- Separate 32,768-token cold and incremental admission ceilings, subject to safe memory policy.
- Chunk-bounded incremental execution.
- Existing exact prefix reuse.
- Persistent SSD cache.
- Cross-session cache restoration.
- Reasoning-aware session continuity.
- All supported reasoning efforts.
- OpenAI Responses and Chat Completions compatibility.
- Anthropic Messages compatibility.
- Qwen Code integration.
- DeepSeek Harness integration.
- Tool-call compatibility and streaming.
- The server watchdog, cancellation, and safety controls.

Do not confuse an advertised 262K context limit with real-checkpoint validation of a full 262K conversation.

Any changed execution plan must participate in snapshot compatibility and cache fingerprinting.

Do not invalidate or delete the existing SSD cache merely to simplify implementation.

A successful overhaul must remain a reliable agentic-coding engine, not become a fragile benchmark-only executor.

---

## 10. FINAL INTEGRATION AND VALIDATION

After implementation, independently test:

- Cold prefill.
- Warm prefill.
- RAM prefix cache hits.
- Persistent SSD prefix hits.
- Decode throughput.
- Model-state equivalence.
- Thinking and reasoning continuity.
- Tool-use turns.
- Cancel/resume.
- Server restart.
- Memory-limit handling.
- Qwen Code and DeepSeek Harness protocol compatibility.

Run both complete unit suites and report actual totals.

For real-model tests, use the existing measurement-standard protocol. Keep comparisons paired, reproducible, and limited by safety controls.

Include workloads with real coding instructions, tool schemas, and repository content rather than only repetitive synthetic prompts.

Produce a final comparison table showing baseline and optimized values for prefill, decoding, TTFT, physical bytes, cache hit rate, and memory.

Document unsuccessful attempts and why they failed.

---

## 11. FINAL DELIVERABLES

Finish with:

1. Working, integrated architectural improvements.
2. Passing complete test suites.
3. Real-model performance measurements.
4. A physical I/O and compute bottleneck breakdown.
5. Updated `docs/flashnext/architecture-overhaul.md`.
6. A concise engineering report describing every significant change.
7. Safe configuration defaults and rollback flags.
8. Working Veystrix startup instructions.
9. Working Qwen Code and DeepSeek Harness connection instructions.
10. A clear final statement of measured prefill and decode gains, or the remaining physical limitations.

Do not report unmeasured improvements as accomplished.

If the new architecture is ready, activate only the improvements that have passed the established validation gates.

Leave unproven experiments available behind explicit opt-in flags.

## FINAL INSTRUCTION

**Take full engineering ownership of this project.**

Be positive, ambitious, analytical, and persistent.

The owner wants a substantial increase in prefill and decode speed, not another round of speculative planning.

Implement, evaluate, iterate, optimize, test, and integrate.

You have permission to complete the necessary controlled measurements without asking again.

If the current approach is insufficient, find a better one.

Keep the hardware safe and the original model intact.

**Start by inspecting the existing uncommitted architecture-overhaul implementation, execute the approved layer-major validation, and continue directly into decode optimization and full integration.**