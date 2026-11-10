# Validation checklist and last recorded status (2026-11-10, development3.81)

Run before every release. "Ran" means executed on this repo state; "NOT RUN" means not executed and **not** claimed.

| Check | Command | Status |
|---|---|---|
| Shared runtime unit tests | `python -m unittest discover -s macqwen/tests -t . -p 'test_*.py'` | Ran: 791 tests OK (8 skipped, incl. the gated Hermes e2e) |
| Flash-Next / Qwen27B / K2 / Bonsai unit tests | the four `models/*/tests/unit` discoveries | Ran: 1818 (3 skipped) / 22 / 58 / 175 (5 skipped), all OK |
| Swift app logic tests | `swift test --package-path apps/macos/Veystrix` | 40 pass (swift-testing) |
| App builds (dev, ad-hoc) and launches with engine stopped | `packaging/macos/build_app.sh` then run | Ran: opens, no child process |
| Hermes real-client e2e vs mock | `VEYSTRIX_HERMES_E2E=1 python -m unittest macqwen.tests.test_hermes_integration` | 25/25 pass, zero egress |
| Release packaging tests | `python -m unittest macqwen.tests.test_release_packaging` | pass |
| Secret scan (tree + history) | `python scripts/security/secret_scan.py [--history]` | 0 hits |
| Public exposure audit | `python scripts/security/public_exposure_audit.py --public DIR --core DIR` | run when the public repo exists |
| Engine bundle build + import smoke | `packaging/engine/build_engine_bundle.sh` | Ran (scratch): imports + `--help` ok |
| Developer ID signing, notarization, stapling | `packaging/macos/make_release.sh --notarize` | **NOT RUN — no credentials** |
| Sparkle key generation, signed appcast, update install | `make_appcast.sh` | **NOT RUN — no credentials** |
| Real-checkpoint engine launch / chat / benchmark | — | **NOT RUN — OWNER TRAVELLING** |
| Layer-major long-lived server safety; 16K+ throughput; logit-level equivalence | — | **NOT RUN — OWNER TRAVELLING** |
| Hardware validation of the signed app (entitlements for MLX/Metal) | — | **NOT RUN** |
| Interactive chat flow test; `test_safe_memory_preflight` | — | **NOT RUN** |
