# Memory admission audit and the measured admission policy (2026-11-09)

Offline only: no checkpoint was loaded and no guard threshold was lowered. Code is uncommitted. The evidence is the seven preserved
real-checkpoint runs of 2026-11-08, reduced by `models/flashnext/tests/bench/admission_audit.py` into
`tests/bench/fixtures/admission_traces.json` and replayed by `tests/unit/test_admission_model.py`.

## 1. The refusal that started this

6.43 GiB available against the 9 GiB gate, 1.69 GiB estimated headroom against 4 GiB, 6.82 GiB other-process anonymous memory,
pressure normal, nothing loaded. Both rules fired, and the second also blocked the automatic purge, because it said file cache
could not help.

## 2. What the real runs show

Gate figure = free + inactive + speculative (what ChatGuard gates on). "Fall" = start figure minus its lowest point in the run.

| run | prefill tokens | start | other anon | load fall | total fall | lowest | MLX peak | outcome |
|---|---|---|---|---|---|---|---|---|
| chat-success | 60 | 9.70 | 5.58 | 3.63 | 6.46 | 3.24 | 3.36 | success |
| agentic-first-stop | 898 | 9.59 | 6.31 | 3.40 | 6.33 | 3.26 | 4.62 | stopped by the compressor rule of that day (retired) |
| agentic-gpu-hang | 902 | 11.22 | 2.69 | 3.32 | 6.54 | 4.68 | 4.11 | GPU deadlock (repaired), not memory |
| agentic-1k | 902 | 10.66 | 3.54 | 3.29 | 6.36 | 4.30 | 4.11 | success |
| agentic-4k | 3,935 | 9.71 | 3.33 | 3.21 | 6.65 | 3.06 | 5.06 | success |
| agentic-8k | ~8,200 | 8.68 | 3.35 | 2.53 | 6.08 | 2.60 | 5.29 | prefill did not finish; waived start |

The fall is **6.1-6.65 GiB in every run**, from a 60-token chat turn to an 8K prefill, while the MLX peak moves 1.2 GiB. Subtracting
the 0.19 GiB DeepSeek Harness client leaves 6.46 GiB for the engine alone (the CHAT run). Nine real purges returned 1.4-3.1 GiB
(median 2.0).

## 3. Answers to the seven questions

1. **Is anonymous memory overcounted?** Yes. The old budget charged 0.75 x other anonymous memory plus the compressor already in
   use. The compressor does not follow other processes: with 2.69 GiB of other anonymous memory it still grew 3.15 GiB; with 6.31 it
   grew 5.5. The engine's own churn dominates, so the term predicts nothing the gate figure does not already contain.
2. **Are file-backed and compressed memory accounted correctly?** The budget ignored file cache on purpose, which is why it never
   saw the one recovery lever that works: a purge returns the active file pages (1.4-3.1 GiB on record). It also added the
   compressor on top of the anonymous squeeze, counting the same pages twice.
3. **Are MLX working-set peaks estimated accurately?** Conservatively, by 0.6-0.9 GiB: the tier budget was 6.0 GiB for any prefill
   of 2,046+ tokens; the real MLX peaks are 4.11 (902 tokens), 5.06 (3,935) and 5.29 (8,200). The new model uses these points and
   extrapolates 0.12 GiB per 1,000 tokens beyond 8,200 (unmeasured, stated as such).
4. **Does the 1 GiB harness reservation match the client?** No. Measured: DeepSeek Harness headless 0.19 GiB, Qwen Code headless
   CLI 0.21 GiB peak RSS. The allowance is now 0.25 GiB inside the model. A desktop client started after the server is not
   reserved for; the runtime guards cover it.
5. **Do the 9 GiB gate and the 4 GiB headroom duplicate margins?** The headroom rule did. Measured fall 6.65 + the 2.0 GiB floor
   ChatGuard enforces + 0.3 GiB margin is 9.0 GiB: the gate was already the right rule for a 4K run. The headroom rule added a
   3 GiB floor (stale: the floor has been 2 GiB since the staged controller) plus 0.66 GiB on top of it. Across the six comparable
   runs it did not rank them by the margin they really had: it called the 8K run safer than the chat run (0.6 GiB of real margin
   against 1.24) and would have refused the successful chat and 4K starts.
6. **Can the adaptive prefill scheduler use a smaller working set?** Safely, but it buys little. Per-chunk MLX increment at
   position 4,096: 1.29 GiB at 2,048 tokens, 0.60 at 1,024, 0.29 at 512, 0.14 at 256. Yet the gate figure fell about the same
   at 4.11 and 5.29 GiB MLX peak, and every chunk re-streams all experts, so halving the chunk doubles the I/O. The critical-only
   step-down already in place is the right use; it is not an admission lever.
7. **Can recovery improve admission without closing applications?** Yes, and it was disabled by the budget. See section 4.

## 4. What changed (uncommitted)

* `models/flashnext/admission_model.py` (new): `drop = 6.46 + client + max(0, MLX peak(tokens) - MLX peak(3,935))`;
  required start figure = `drop + 2.0 floor + 0.3 margin`, never below 9 GiB; +0.5 GiB when other anonymous memory is 6 GiB or
  more (the only memory-driven failure on record had 6.31); stage minima reported for load and for the first cold prefill;
  expected purge gain = the smallest on record (1.4 GiB), capped by the file cache that exists.
* `prefill_readiness.py`: the gate uses that requirement per tier (CHAT keeps its plain 9 GiB path); the anonymous budget stays as
  an advisory line and never refuses; a purge is blocked only by pressure/swap-out, or when the cache cannot close the gap.
* `resource_prep.py`: the purge target follows the tier's requirement; the failure report states the shortfall.
* Unchanged: ChatGuard (2 GiB floor, battery, Low Power Mode, thermal, startup 9 GiB for CHAT), the staged controller
  (+2.5 GiB swap stop, 45 s sustained pressure, level 4), GPU/GIL watchdog, between-turn release.
* Consequence worth knowing: below about 4K tokens every tier needs the same start figure, so degrading to a smaller tier no longer
  rescues a tight machine (it never saved more than about 0.3 GiB).

Validation: the model never predicts a lowest figure above what a run reached (largest pessimism 0.9 GiB, at the 8K run), and
replaying every complete trace through today's controller gives no stop and no floor breach. Tests rewritten because they encoded
the retired veto: `test_prefill_readiness`, `test_resource_prep`, `test_request_recovery`, `test_gate_a`, `test_agentic_4k_offline`.

## 5. What each workload needs at launch (gate figure, GiB)

| workload | cold prefill | needs | with 6 GiB+ other anon |
|---|---|---|---|
| CHAT | ~60 tokens | 9.00 | 9.26 |
| DeepSeek Harness, 4K input | 3,935 | 9.01 | 9.51 |
| Qwen Code trimmed (8.6K first request) | 8,610 | 9.29 | 9.79 |
| Qwen Code with 10K input | ~10,150 | 9.47 | 9.97 |
| 12K | 12,288 | 9.72 | 10.22 |
| 12K+ long tier | 14,336 | 9.96 | 10.46 |

These are memory figures only. Validated on this Mac: CHAT, and a 3,935-token cold prefill. An 8K prefill did not finish in 900 s
for a reason still not explained, so Qwen Code and 12K+ remain unproven whatever the figure says.

## 6. Your refused state, and the honest expectation

At 6.43 GiB available the 4K requirement is 9.02, plus 0.5 because other anonymous memory is 6.82 GiB: 9.52. That is short by 3.1 GiB.
The new policy no longer blocks the purge, but a purge on record returned 3.1 GiB once in nine tries (2.0 median). So expect: refusal at
first, a purge attempt that usually closes about two thirds of the gap, and then admission only if the machine really has the memory.
Veystrix needs the gate figure to fall by about 6.5 GiB; on a 16 GB machine that means roughly 9 GiB available before it starts. The
audit removed false refusals (the chat-success and 4K states) and made recovery effective; it cannot make 6.4 GiB enough.
Closing applications is still the only way to gain more than the file cache, and `~/.flashnext/close-apps.json` remains opt-in.

## 7. Not done, and the next step

No real run was made. To validate the model beyond the 4K region, one approved real run at the 8K tier with `FLASHNEXT_CHUNK_LOG` and
stack sampling would also test the extrapolation above 8,200 tokens. Acceptance: lowest gate figure within 0.3 GiB of the model's
prediction (7.18 GiB fall at 10,240 tokens), no stop, prefill completes.
