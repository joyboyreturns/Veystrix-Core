#!/bin/bash
# Build, sign, package, (notarize) and checksum a release candidate.
#   packaging/macos/make_release.sh [--engine DIR] [--notarize]
# Channels (see common.sh):
#   free          default when VEYSTRIX_SIGN_IDENTITY is unset. Ad-hoc signed, NOT notarized. No Apple membership needed.
#   developer-id  VEYSTRIX_SIGN_IDENTITY set; add --notarize (needs notarytool credentials) for Gatekeeper-clean builds.
# Both write the same file names plus BUILD-INFO.txt that states the channel honestly.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENGINE=(); NOTARIZE=0
while [[ $# -gt 0 ]]; do case "$1" in --engine) ENGINE=(--engine "$2"); shift 2;; --notarize) NOTARIZE=1; shift;; *) die "unknown option $1";; esac; done
if [[ $NOTARIZE -eq 1 ]] && ! is_release_signing; then
  die "--notarize needs a Developer ID identity (VEYSTRIX_SIGN_IDENTITY). Without one, build the free channel: run again without --notarize."
fi
mkdir -p "$DIST"
"$SCRIPT_DIR/build_app.sh" "${ENGINE[@]}"
CH="$(channel)"; NOTARIZED="no"; [[ $NOTARIZE -eq 1 ]] && NOTARIZED="yes (after notarize.sh succeeds)"
cat > "$DIST/BUILD-INFO.txt" <<INFO
Veystrix $SHORT_VERSION (build $BUILD_NUMBER), $ARCH
Channel:          $CH
Code signature:   $([[ "$CH" == developer-id ]] && echo "Developer ID ($SIGN_IDENTITY)" || echo "ad-hoc (no Developer ID)")
Notarized:        $NOTARIZED
Update key:       $([[ -n "$SPARKLE_PUBLIC_KEY" ]] && echo "Sparkle EdDSA public key $SPARKLE_PUBLIC_KEY" || echo "none (updates disabled)")
Built:            $(date -u +%FT%TZ)
Integrity:        verify the SHA-256 in SHA256SUMS.txt against the file you downloaded.
$([[ "$CH" == free ]] && echo "First open:       macOS will say it cannot verify the app. See docs/installation.md (Privacy & Security > Open Anyway).")
INFO
"$SCRIPT_DIR/make_dmg.sh" "$DIST/$APP_NAME.app"
SUF="$(artifact_suffix)"
if [[ $NOTARIZE -eq 1 ]]; then "$SCRIPT_DIR/notarize.sh" "$DIST/Veystrix-$SHORT_VERSION-$ARCH$SUF.dmg"; sed -i '' 's/^Notarized:.*/Notarized:        yes/' "$DIST/BUILD-INFO.txt"; fi
( cd "$DIST" && shasum -a 256 "Veystrix-$SHORT_VERSION-$ARCH$SUF".{dmg,zip} > SHA256SUMS.txt && cat SHA256SUMS.txt )
echo
if [[ "$CH" == free ]]; then
  echo "  Free channel build: ad-hoc signed, NOT notarized. Fine to publish as a free download with the Gatekeeper instructions"
  echo "  (docs/release/FREE_CHANNEL.md). It is NOT a Developer ID release; do not describe it as one."
else
  [[ $NOTARIZE -eq 1 ]] && log "Developer ID signed and notarized release candidate." || log "Developer ID signed but NOT notarized; add --notarize before publishing."
fi
