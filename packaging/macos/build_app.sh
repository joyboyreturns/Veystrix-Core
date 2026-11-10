#!/bin/bash
# Build Veystrix.app (arm64) from the Swift package. Usage:
#   packaging/macos/build_app.sh [--engine DIR]     # DIR = output of packaging/engine/build_engine_bundle.sh
# Environment: VEYSTRIX_SIGN_IDENTITY (Developer ID Application ...), VEYSTRIX_SPARKLE_PUBLIC_KEY (base64 EdDSA public key),
#              VEYSTRIX_VERSION, VEYSTRIX_BUILD_NUMBER.
# Without a signing identity the app is ad-hoc signed: the FREE channel (not notarized; Gatekeeper warns on first open).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENGINE_DIR=""
while [[ $# -gt 0 ]]; do case "$1" in --engine) ENGINE_DIR="$(cd "$2" 2>/dev/null && pwd)" || die "engine directory $2 not found"; shift 2;; *) die "unknown option $1";; esac; done

APP="$DIST/$APP_NAME.app"
log "Building $APP_NAME $SHORT_VERSION (build $BUILD_NUMBER) for $ARCH [channel: $(channel)]"
cd "$APP_DIR"
swift build -c release --arch "$ARCH" --product Veystrix 2>&1 | grep -vE "ld: warning: search path" | tail -5
BIN_DIR="$(swift build -c release --arch "$ARCH" --show-bin-path)"
[[ -x "$BIN_DIR/Veystrix" ]] || die "binary not found in $BIN_DIR"

rm -rf "$APP"; mkdir -p "$APP/Contents/"{MacOS,Resources,Frameworks}
cp "$BIN_DIR/Veystrix" "$APP/Contents/MacOS/Veystrix"

# Sparkle.framework from the SwiftPM binary artifact
SPARKLE_FW="$(find "$APP_DIR/.build" -type d -name Sparkle.framework -path '*macos-arm64_x86_64*' | head -1)"
[[ -n "$SPARKLE_FW" ]] || die "Sparkle.framework not found; run 'swift build' once to fetch it"
ditto "$SPARKLE_FW" "$APP/Contents/Frameworks/Sparkle.framework"
install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP/Contents/MacOS/Veystrix" 2>/dev/null || true

# Minimum macOS: the UI alone runs on 14, but a bundled engine inherits the minimum OS of its MLX binaries
# (MLX 0.32.2 is built for macOS 26.2), so the bundle declares what the engine really needs.
MIN_MACOS="14.0"
if [[ -n "${ENGINE_DIR:-}" ]]; then
  MLX_SO="$(ls "$ENGINE_DIR"/python/lib/python3.*/site-packages/mlx/core.*.so 2>/dev/null | head -1 || true)"
  [[ -n "$MLX_SO" ]] && MIN_MACOS="$(otool -l "$MLX_SO" | awk '/minos/{print $2; exit}')"
fi
log "LSMinimumSystemVersion = $MIN_MACOS"
# Info.plist (Sparkle key injected; empty key => updates disabled in the app)
sed -e "s/@MIN_MACOS@/$MIN_MACOS/" -e "s/@SHORT_VERSION@/$SHORT_VERSION/" -e "s/@BUILD_NUMBER@/$BUILD_NUMBER/" \
    -e "s|@SPARKLE_PUBLIC_ED_KEY@|$SPARKLE_PUBLIC_KEY|" \
    "$APP_DIR/Support/Info.plist.template" > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist" >/dev/null
[[ -n "$SPARKLE_PUBLIC_KEY" ]] || log "NOTE: no Sparkle public key: this build has updates disabled"

# Icon, licenses, notices
python3 "$APP_DIR/Support/Icon/make_icon.py" "$APP/Contents/Resources/AppIcon.icns" 2>/dev/null \
  || "$ROOT/.venv/bin/python" "$APP_DIR/Support/Icon/make_icon.py" "$APP/Contents/Resources/AppIcon.icns" \
  || log "WARNING: icon not generated (Pillow missing)"
cp "$ROOT/legal/FREEWARE-LICENSE.txt" "$APP/Contents/Resources/LICENSE.txt"
cp "$ROOT/legal/THIRD_PARTY_NOTICES.md" "$APP/Contents/Resources/THIRD_PARTY_NOTICES.md"
cp -R "$ROOT/LICENSES" "$APP/Contents/Resources/Licenses"
echo "$SHORT_VERSION" > "$APP/Contents/Resources/VERSION"

# Engine runtime (optional)
if [[ -n "$ENGINE_DIR" ]]; then
  [[ -x "$ENGINE_DIR/python/bin/python3" ]] || die "$ENGINE_DIR is not an engine bundle"
  log "Validating engine runtime from $ENGINE_DIR"
  "$ENGINE_DIR/python/bin/python3" "$ROOT/packaging/engine/smoke_import.py" "$ENGINE_DIR" || die "engine bundle validation failed"
  log "Embedding engine runtime from $ENGINE_DIR"
  ditto "$ENGINE_DIR" "$APP/Contents/Resources/engine"
else
  log "NOTE: no --engine: this app finds the engine only through Settings > Developer (source checkout)"
fi

if [[ -n "$ENGINE_DIR" ]]; then
  "$APP/Contents/Resources/engine/python/bin/python3" "$ROOT/packaging/engine/smoke_import.py" "$APP/Contents/Resources/engine" || die "embedded engine validation failed"
fi
"$SCRIPT_DIR/sign_app.sh" "$APP"
log "Built $APP"
