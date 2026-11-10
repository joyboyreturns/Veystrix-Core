# Handoff: Veystrix runtime recovery

## Goal and state

Complete the user's runtime recovery mission: real Qwen3.8-Flash-Next 4-bit on M4/16 GB, DeepSeek Harness and Qwen Code tool loops, durable cache continuation across restart, 262144 logical context and configurable 32768 cold/incremental work, correct guard/telemetry. Do not claim completion until the full real-client gates pass. Work on `development3.8`, starting HEAD and last known origin SHA `93e5312478834dd1ba964fcdb5b3aff8cdeb8b5b`; inspect `git status --short --branch` and `git log` for current state and candidate commits. See progress document for candidate files and test outcomes.

## Confirmed evidence

Previous DeepSeek request had 9427 prompt tokens and 32 tools; last reported progress 9230, then no first token before absolute guard stopped at 2963 MiB swap (2816 MiB margin, 3072 MiB ceiling). The progress was a layer-derived estimate, not verified token progress; do not infer a 197-token final chunk. Root cause remains unknown. Startup legacy tiers capped configured 32768 at 4096/14336. Failed SSD persistence could silently evict an unpersisted RAM leaf.

## Candidate work and limitations

- `models/flashnext/prefill.py`: chunk plan, chunk, final forward/projection/evaluation flight events. `prefix_session.py`: snapshot events. `macqwen/backends/flashnext.py`: backend return/decode events. MLX lazy execution means events mark host boundaries, not necessarily completion of GPU work. More detailed expert, layer-major, memory and transport instrumentation is needed.
- `models/flashnext/prefix_cache.py::_evict_ram`: retain entries without durable backing and search other candidates. Regression in `macqwen/tests/test_phase3_remediations.py`. This can exceed RAM budget after persistent write failure: implement safe, explicit invalidation under pressure, with accurate reporting, before release.
- `models/flashnext/prefill_readiness.py::active_tiers`: assess configured agentic workload with bounded <=2048 chunks and KV-growth estimate, rather than legacy tier. Tests in `test_prefill_readiness.py`. Audit actual per-request admission and diagnostics; a configured cap does not guarantee safe execution. Existing `prefix_session.py` uses `min(max_inc, max_cold)` for restored requests; investigate intended separate semantics and cache-miss checks.
- `.github/workflows/ci.yml`: includes `development3.8` push. No CI run verified.

Targeted tests: 87 and 42 passed under Metal outside sandbox; full MacQwen suite passed 725 tests in 69.195 seconds and FlashNext passed 1809 tests in 169.589 seconds (3 skipped). Sandboxed MLX import fails with `No Metal device available`; `.venv` has no pytest. Current machine was on battery (77%); no real model run this session. Swap was 1140.81 MiB when sampled, not the value at the incident. `qwen` version 0.27.0; `dsh` not found on PATH. Five-hour usage telemetry unavailable.

## Next actions, in order

1. Read `prefill.py::_run_chunks`, `_prefill_target`, `expert_cache.py` progress callback, and `prefix_session.py::snapshot`; add missing stage/memory/expert and transport instrumentation, verify actual chunk progress and bounded cancellation. Finish safe cache failure behavior and tests.
2. On AC with adequate measured headroom and unchanged guard, run `./serve.sh`, inspect server logs and diagnostics. Progress through short, 2K, 4K, 8K and 9K cold requests; stop at safety margin or stall. Trace the real failing workload and establish a root cause before claiming a fix. Check previously saved snapshots without deleting them.
3. Locate installed DeepSeek CLI and config (see `docs/flashnext/dsh/veystrix-262k.patch.yml` and incident report), run real file read/search/edit/test/fix tool workflow; then run real Qwen Code multi-file workflow with two continuation turns. Test cancellation and streaming. Do not alter unrelated providers.
4. Prove actual RAM/SSD and restart restoration with compatible fingerprints, accounting, corrupt/missing parents, failure/cleanup/quota behavior. Validate above-limit and cache-hit admission and startup diagnostics. Run both full unittest suites, HTTP fake-backend tests, `git diff --check`, and inspect GitHub CI.
5. Commit focused validated work and push to `development3.8` when safe. Do not merge main or force push. Keep unvalidated candidates clearly labelled.

## Commands and locations

`source .venv/bin/activate`; `.venv/bin/python -m unittest discover -s macqwen/tests -t . -p 'test_*.py' -q`; `.venv/bin/python -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py' -q`. `./serve.sh`, `./serve.sh --status`, `curl -s http://127.0.0.1:8080/health`, `curl -s http://127.0.0.1:8080/v1/cache/diagnostics` (supply configured auth if required). `qwen` for installed CLI; DeepSeek example: `dsh --profile headless --patch docs/flashnext/dsh/veystrix-262k.patch.yml` after locating executable and confirming provider selection. Logs `~/.flashnext/logs/`; incident details `docs/flashnext/deepseek-swap-incident-2026-11-09.md`; model `/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`.

Preserve existing sessions, caches and user changes. Keep all experts and checkpoint precision, logical context, watchdog and 3072 MiB absolute system-swap ceiling with margin. Do not automatically stress-test 32K. No full-model performance or release-readiness claim is supported by this session.
