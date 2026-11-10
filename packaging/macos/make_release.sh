#!/bin/bash
# Build, sign, package, (notarize) and checksum a release candidate. Usage: packaging/macos/make_release.sh [--engine DIR] [--notarize]
# Public distribution requires: VEYSTRIX_SIGN_IDENTITY, notarization (--notarize) and VEYSTRIX_SPARKLE_PUBLIC_KEY.
# Anything else produces a clearly labelled DEV artifact and the script says so.
source "$SCRIPT_DIR/common.sh"
ENGINE=(); NOTARIZE=0
while [[ $# -gt 0 ]]; do case "$1" in --engine) ENGINE=(--engine "$2"); shift 2;; --notarize) NOTARIZE=1; shift;; *) die "unknown option $1";; esac; done
mkdir -p "$DIST"
"$SCRIPT_DIR/build_app.sh" "${ENGINE[@]}"
"$SCRIPT_DIR/make_dmg.sh" "$DIST/$APP_NAME.app"
SUF="$(artifact_suffix)"
if [[ $NOTARIZE -eq 1 ]]; then "$SCRIPT_DIR/notarize.sh" "$DIST/Veystrix-$SHORT_VERSION-$ARCH$SUF.dmg"; fi
( cd "$DIST" && shasum -a 256 "Veystrix-$SHORT_VERSION-$ARCH$SUF".{dmg,zip} > SHA256SUMS.txt && cat SHA256SUMS.txt )
if is_release_signing && [[ $NOTARIZE -eq 1 && -n "${VEYSTRIX_SPARKLE_PUBLIC_KEY:-}" ]]; then
  log "Release candidate is SIGNED and NOTARIZED. Verify with tools/verify-download.sh, then publish (docs/release/RELEASE_PROCESS.md)."
else
  echo
  echo "  *** DEVELOPMENT ARTIFACT: NOT signed with Developer ID and/or NOT notarized and/or no update key. ***"
  echo "  *** Public distribution is BLOCKED. Do not upload these files to a public release.            ***"
fi
