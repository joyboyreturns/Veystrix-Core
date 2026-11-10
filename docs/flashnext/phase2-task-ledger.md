# Phase II task ledger

Updated 2026-11-10. Branch `development3.8`; starting HEAD `6663a8a`, clean and 14 commits ahead of remote. Do not push or alter the checkpoint.

Critical correction: the previous launch defaulted to a 0.85 retained-mass routing threshold. The current-checkpoint trace contains 4–9 experts per layer/token, not the required top 10. Previous speed and same-setting exactness evidence cannot establish full-routing behavior. Re-run the model gates at threshold 1.0, including safety estimates, before promotion.

The default threshold is now 1.0 and the supervised paired harness passes it explicitly. Synthetic tests check ten distinct expert routes and separation from 0.85 prefix snapshots; a routing-profile change triggers transcript replay, and incompatible saved sessions are refused. Full-routing live performance, memory, and state validation remain open.

- [x] Recover prior measurements and safety incident. The 8K layer-major win remains opt-in; 8K after decode breached 3 GiB swap with a five-second monitor. The 4.6K passing smoke does not validate that workload.
- [ ] Strengthen and test memory admission, cancellation, and independent monitoring. Candidate admission and an independent swap backstop now have focused synthetic coverage; their combined live behavior and cleanup remain unvalidated. Do not claim a hard limit from polling alone.
- [ ] Isolate decode costs, investigate native I/O, kernel execution and residency; retain changes only with correctness and performance evidence.
- [ ] Investigate 16K prefill, n-gram path, and exact speculative decoding.
- [ ] Run supervised AC-powered real-checkpoint tests only after safety preflight. At initial inspection power was battery (80%); no live model run is authorized under the specified AC condition.
- [ ] Validate multi-turn 8K safety, restoration, APIs, Qwen Code and DeepSeek Harness in that order; do not launch concurrent model workloads.
- [ ] Run regression suites, write phase II results and reproducible commands, and commit verified changes locally.

Existing physical measurements are in `overnight-architecture-results.md` and `results/flashnext/`. They are historical evidence, not phase II measurements. No new throughput claim should be made without the prescribed exactness-gated paired measurements.

Regression commands: `.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'` and the same command with `-s models/flashnext/tests/unit`. Use `-t .`: it loads the MacQwen test package's temporary warm-cache isolation. The unqualified initial run did not do so and failed the corresponding isolation assertion; an initial FlashNext run overlapped source edits and was invalid. On the settled tree MacQwen ran 686 successfully and FlashNext ran 1,779 successfully with 3 skipped (both exit 0).

## 2026-11-10 Claude Code session

- [x] Recover and review interrupted work; committed as `0071fb2`. Offline suites: 1,803 FlashNext (3 skipped), 690 MacQwen, exit 0.
- [x] Battery-allowed mode for the after-turns harness (explicit flag, 40% default floor, recorded).
- [x] Real-checkpoint full-routing confirmation: 1,152/1,152 layer calls routed 10 distinct experts (`results/flashnext/20261110-115952-decode-profile-manual/`). Prefix-profile invalidation against the real model is still synthetic-only.
- [ ] Open: full-routing paired baseline with exactness gates, 8K-after-turns safety gate, decode timeline and optimization, speculation, 16K, API/client validation, final documentation.
- [~] 8K-after-turns gate attempted on battery: postponed at preflight (11,635 MiB available, 11,908 MiB needed); nothing started. Re-run when about 300 MiB more is available.
