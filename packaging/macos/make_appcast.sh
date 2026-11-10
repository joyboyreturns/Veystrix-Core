#!/bin/bash
# Generate / update the Sparkle appcast for a signed release and validate it. Usage:
#   packaging/macos/make_appcast.sh <release-dir-with-zip> <tag>
# The EdDSA PRIVATE key is read from the macOS keychain (generate_keys, run once on the owner's Mac) or from the
# VEYSTRIX_SPARKLE_PRIVATE_KEY env var in CI (stdin to generate_appcast; never written to disk or Git).
# Sparkle's tools come from the pinned release tarball, whose SHA-256 is checked before anything is run.
source "$(dirname "$0")/common.sh"
RELEASE_DIR="${1:?release directory containing Veystrix-<ver>-arm64.zip}"; TAG="${2:?tag, e.g. v3.8.1}"
is_release_signing || die "refusing to build an update feed for a development build (no Developer ID identity). Public distribution is BLOCKED."
TOOLS="$DIST/sparkle-tools"; mkdir -p "$TOOLS"
if [[ ! -x "$TOOLS/bin/generate_appcast" ]]; then
  TB="$TOOLS/Sparkle-$SPARKLE_VERSION.tar.xz"
  curl -fsSL -o "$TB" "https://github.com/sparkle-project/Sparkle/releases/download/$SPARKLE_VERSION/Sparkle-$SPARKLE_VERSION.tar.xz"
  echo "$SPARKLE_TARBALL_SHA256  $TB" | shasum -a 256 -c - || die "Sparkle tarball checksum mismatch; not extracting"
  tar -xf "$TB" -C "$TOOLS" ./bin
fi
PREFIX="https://github.com/joyboyreturns/Veystrix/releases/download/$TAG/"
FEED="$ROOT/updates/appcast.xml"
mkdir -p "$ROOT/updates"; [[ -f "$FEED" ]] && cp "$FEED" "$RELEASE_DIR/appcast.xml"
if [[ -n "${VEYSTRIX_SPARKLE_PRIVATE_KEY:-}" ]]; then
  printf '%s' "$VEYSTRIX_SPARKLE_PRIVATE_KEY" | "$TOOLS/bin/generate_appcast" --ed-key-file - --download-url-prefix "$PREFIX" "$RELEASE_DIR"
else
  "$TOOLS/bin/generate_appcast" --download-url-prefix "$PREFIX" "$RELEASE_DIR"        # key from the login keychain
fi
python3 "$SCRIPT_DIR/validate_appcast.py" "$RELEASE_DIR/appcast.xml" ${VEYSTRIX_SPARKLE_PUBLIC_KEY:+--public-key "$VEYSTRIX_SPARKLE_PUBLIC_KEY" --archive-dir "$RELEASE_DIR"}
cp "$RELEASE_DIR/appcast.xml" "$FEED"
log "Appcast written to updates/appcast.xml. Commit it to the PUBLIC repo only after the release assets are live."
