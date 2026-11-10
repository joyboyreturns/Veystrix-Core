#!/bin/bash
# Sign Veystrix.app inside-out (Hardened Runtime for Developer ID builds). Usage: packaging/macos/sign_app.sh <Veystrix.app>
# With VEYSTRIX_SIGN_IDENTITY set: Developer ID signing + secure timestamp (required for notarization).
# Without: ad-hoc signing for local development only. This script never reports a development signature as a release.
source "$(dirname "$0")/common.sh"
APP="${1:?usage: sign_app.sh <Veystrix.app>}"
ENT_APP="$APP_DIR/Support/Veystrix.entitlements"
ENT_ENGINE="$APP_DIR/Support/Engine.entitlements"
if is_release_signing; then
  ID="$SIGN_IDENTITY"; EXTRA=(--options runtime --timestamp)
  security find-identity -v -p codesigning | grep -qF "$SIGN_IDENTITY" || die "signing identity not found in the keychain: $SIGN_IDENTITY"
else
  # Ad-hoc code has no Team ID, so under Hardened Runtime library validation it cannot load the embedded Sparkle
  # framework. Development builds therefore omit the runtime flag. Release builds (one Developer ID team) keep it.
  ID="-"; EXTRA=()
  log "No VEYSTRIX_SIGN_IDENTITY: ad-hoc signing WITHOUT Hardened Runtime. DEVELOPMENT ARTIFACT, NOT FOR DISTRIBUTION."
fi
sign() { codesign --force --sign "$ID" ${EXTRA[@]+"${EXTRA[@]}"} "$@"; }

# 1. Sparkle's nested helpers, then the framework (order matters: innermost first)
FW="$APP/Contents/Frameworks/Sparkle.framework"
if [[ -d "$FW" ]]; then
  while IFS= read -r p; do sign "$p"; done < <(find "$FW" \( -name "*.xpc" -o -name "Updater.app" -o -name "Autoupdate" \) -not -path '*/Resources/*/*/*' 2>/dev/null | sort -r)
  sign "$FW"
fi
# 2. Engine runtime: every Mach-O executable and library, with the engine entitlements for executables
ENGINE="$APP/Contents/Resources/engine"
if [[ -d "$ENGINE" ]]; then
  log "Signing engine runtime Mach-O files (this can take a while)"
  while IFS= read -r f; do
    if file "$f" | grep -q "Mach-O.*executable"; then sign --entitlements "$ENT_ENGINE" "$f"; else sign "$f"; fi
  done < <(find "$ENGINE" -type f \( -perm +111 -o -name "*.so" -o -name "*.dylib" -o -name "*.metallib" \) -exec sh -c 'file "$1" | grep -q "Mach-O" && echo "$1"' _ {} \;)
fi
# 3. The app
sign --entitlements "$ENT_APP" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | tail -3
if is_release_signing; then
  codesign -dv --verbose=2 "$APP" 2>&1 | grep -E "Authority|TeamIdentifier|flags|Timestamp" | head -6
else
  echo "ad-hoc" > "$APP/Contents/Resources/DEVELOPMENT-BUILD"
fi
