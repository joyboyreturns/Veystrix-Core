# Packaging the engine inside the app

**Layout** inside `Veystrix.app/Contents/Resources/engine`: `python/` (python-build-standalone CPython 3.12 pinned in
`packaging/engine/runtime.pin`, SHA-256 verified), `lib/` (engine packages), `licenses/`, `BUNDLE-MANIFEST.txt`.
Built by `packaging/engine/build_engine_bundle.sh`; smoke-tested by `packaging/engine/smoke_import.py` (imports and
`--help` only: no model, no Metal session, no server).

**Compiled, not protected.** Engine modules ship as `.pyc` files. This keeps source out of casual view and avoids
accidental edits; it is **not** obfuscation or copy protection. Anyone can decompile `.pyc`, and the engine was
built partly by reverse-engineering model behaviour, so no secrecy is claimed for the algorithms. Exception:
`models/flashnext/settings/setting_*.py` stay as source because the registry discovers them by globbing.

**Excluded on purpose**: opencv (its wheels embed an LGPL FFmpeg; removed after install and asserted absent), the
K2-Horizon and Bonsai-2 backends (`trust_remote_code`), tests, benchmarks, docs, and every model checkpoint.
The ~113 GB checkpoint is never bundled or downloaded; the user chooses a folder.

**Measured 2026-11-10 (scratch build, not signed):** bundle 682 MB; imports and `--help` pass with the bundled
interpreter. **Not verified:** nested Mach-O signing + notarization of this bundle, Hardened Runtime entitlements
for MLX/Metal on a signed build, launching the real engine from the bundle (needs a model: NOT RUN, owner
travelling), size on disk of the final DMG.
