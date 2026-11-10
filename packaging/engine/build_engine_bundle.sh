#!/bin/bash
# Build the self-contained engine runtime that Veystrix.app embeds in Contents/Resources/engine.
#   packaging/engine/build_engine_bundle.sh [OUT_DIR]        (default: dist/engine)
# Layout:  OUT/python/   pinned CPython (python-build-standalone) with the engine's pinned dependencies
#          OUT/lib/      veystrix/, models/{flashnext,qwen27b}/ as compiled .pyc, plus data files
#          OUT/licenses/ license texts for everything inside
#          OUT/BUNDLE-MANIFEST.txt
# Honest limits: .pyc files are an anti-clutter and anti-accident measure, NOT protection. They decompile
# easily; do not describe the bundle as obfuscated or protected. Dependencies are downloaded from PyPI with
# pinned versions (network needed, no model weights). Nothing here loads or imports a model.
# Excluded on purpose: opencv (LGPL FFmpeg build; see legal/COMMERCIALIZATION_READINESS.md), the K2-Horizon and
# Bonsai-2 backends (they use trust_remote_code), tests, docs, benchmarks, and every checkpoint.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$ROOT/dist/engine}"; WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$(dirname "$OUT")"; OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"   # absolute: later steps cd
source "$HERE/runtime.pin"
[[ "$(uname -m)" == "arm64" ]] || { echo "ERROR: Apple silicon build host required" >&2; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/licenses"

echo "==> Python runtime $PBS_FILE"
CACHE="${VEYSTRIX_DOWNLOAD_CACHE:-$ROOT/dist/downloads}"; mkdir -p "$CACHE"
if [[ ! -f "$CACHE/$PBS_FILE" ]]; then
  curl -fsSL -o "$CACHE/$PBS_FILE" "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/${PBS_FILE//+/%2B}"
fi
echo "$PBS_SHA256  $CACHE/$PBS_FILE" | shasum -a 256 -c - || { echo "ERROR: runtime checksum mismatch" >&2; rm -f "$CACHE/$PBS_FILE"; exit 1; }
tar -xzf "$CACHE/$PBS_FILE" -C "$WORK" && mv "$WORK/python" "$OUT/python"
PY="$OUT/python/bin/python3"

echo "==> Pinned dependencies (opencv excluded)"
"$PY" - "$ROOT/pyproject.toml" > "$WORK/requirements.txt" <<'PYEOF'
import sys, tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
bad = [d for d in deps if "==" not in d]
if bad: sys.exit(f"unpinned dependency in pyproject.toml: {bad}")
print("\n".join(d for d in deps if not d.lower().startswith(("opencv", "cv2"))))
PYEOF
"$PY" -m pip install --no-cache-dir --disable-pip-version-check --only-binary=:all: -r "$WORK/requirements.txt"
# opencv arrives transitively (mlx-vlm). Its wheels carry an LGPL FFmpeg build, so it is removed from the bundle and
# the smoke test asserts it is gone. Features that need it (video input) are unavailable in the packaged app.
"$PY" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
rm -rf "$OUT"/python/lib/python3.12/site-packages/cv2
# The engine never opens a window: drop Tcl/Tk, IDLE and the stdlib test suite (smaller bundle, fewer files to sign).
rm -rf "$OUT"/python/lib/python3.12/{test,idlelib,tkinter,turtledemo,turtle.py} "$OUT"/python/lib/python3.12/lib-dynload/_tkinter*.so \
       "$OUT"/python/lib/{tcl*,tk*,itcl*,thread*,Tix*,sqlite3*} "$OUT"/python/bin/{idle*,2to3*,pydoc*} 2>/dev/null || true

echo "==> Engine sources -> compiled lib"
for pkg in veystrix models/__init__.py models/flashnext models/qwen27b; do
  mkdir -p "$OUT/lib/$(dirname "$pkg")" 2>/dev/null || true
  ditto "$ROOT/$pkg" "$OUT/lib/$pkg"
done
find "$OUT/lib" \( -name tests -o -name bench -o -name __pycache__ -o -name '*.md' \) -prune -exec rm -rf {} +
# Compile next to the source (-b = legacy .pyc location, importable without the .py), then drop the .py files.
# The settings registry discovers setting_*.py by globbing, so those stay as source.
"$PY" -m compileall -q -b -j 0 "$OUT/lib"
find "$OUT/lib" -name '*.py' ! -path '*/settings/setting_*.py' -delete
[[ -f "$OUT/lib/models/flashnext/settings/registry.pyc" ]] || { echo "ERROR: registry missing"; exit 1; }

echo "==> License texts"
"$PY" "$ROOT/packaging/legal/generate_legal.py" --out "$WORK/legal" --emit-texts "$OUT/licenses" >/dev/null \
  || { cp "$ROOT/legal/THIRD_PARTY_NOTICES.md" "$OUT/licenses/"; cp -R "$ROOT/LICENSES/." "$OUT/licenses/"; echo "   (generate_legal --emit-texts unavailable: copied notices only)"; }

echo "==> Smoke test (imports only, no model)"
"$PY" "$HERE/smoke_import.py" "$OUT"

( cd "$OUT" && { echo "Veystrix engine bundle, built $(date -u +%FT%TZ)"; echo "runtime: $PBS_FILE"; "$PY" -m pip freeze | sort; } > BUNDLE-MANIFEST.txt )
echo "==> Bundle ready: $OUT ($(du -sh "$OUT" | cut -f1)). Sign nested Mach-O files with packaging/macos/sign_app.sh."
