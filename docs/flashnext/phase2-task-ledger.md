# Phase II task ledger

Updated 2026-11-10. Branch `development3.8`; starting HEAD `6663a8a`, clean and 14 commits ahead of remote. Do not push or alter the checkpoint.

- [x] Recover prior measurements and safety incident. The 8K layer-major win remains opt-in; 8K after decode breached 3 GiB swap with a five-second monitor. The 4.6K passing smoke does not validate that workload.
- [ ] Strengthen and test memory admission, cancellation, and independent monitoring. Candidate admission and an independent swap backstop now have focused synthetic coverage; their combined live behavior and cleanup remain unvalidated. Do not claim a hard limit from polling alone.
- [ ] Isolate decode costs, investigate native I/O, kernel execution and residency; retain changes only with correctness and performance evidence.
- [ ] Investigate 16K prefill, n-gram path, and exact speculative decoding.
- [ ] Run supervised AC-powered real-checkpoint tests only after safety preflight. At initial inspection power was battery (80%); no live model run is authorized under the specified AC condition.
- [ ] Validate multi-turn 8K safety, restoration, APIs, Qwen Code and DeepSeek Harness in that order; do not launch concurrent model workloads.
- [ ] Run regression suites, write phase II results and reproducible commands, and commit verified changes locally.

Existing physical measurements are in `overnight-architecture-results.md` and `results/flashnext/`. They are historical evidence, not phase II measurements. No new throughput claim should be made without the prescribed exactness-gated paired measurements.

Regression commands: `.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'` and the same command with `-s models/flashnext/tests/unit`. Use `-t .`: it loads the MacQwen test package's temporary warm-cache isolation. The unqualified initial run did not do so and failed the corresponding isolation assertion; an initial FlashNext run overlapped source edits and was invalid. On the settled tree MacQwen ran 686 successfully and FlashNext ran 1,779 successfully with 3 skipped (both exit 0).
