#!/bin/bash
# Notarize and staple. Usage: packaging/macos/notarize.sh <file.dmg|file.zip> ...
# Requires: a Developer ID signed build and notarytool credentials stored once with:
#   xcrun notarytool store-credentials VEYSTRIX_NOTARY --apple-id <id> --team-id <TEAM> --password <app-specific-password>
# (or an App Store Connect API key: --key <p8> --key-id <id> --issuer <uuid>). Credentials never go in Git.
source "$(dirname "$0")/common.sh"
is_release_signing || die "notarization needs a Developer ID signed build. VEYSTRIX_SIGN_IDENTITY is not set; this is a development build and public distribution is BLOCKED."
PROFILE="${VEYSTRIX_NOTARY_PROFILE:-VEYSTRIX_NOTARY}"
[[ $# -gt 0 ]] || die "usage: notarize.sh <file> ..."
for f in "$@"; do
  log "Submitting $(basename "$f") to Apple notary service"
  xcrun notarytool submit "$f" --keychain-profile "$PROFILE" --wait --output-format json | tee "$f.notary.json"
  grep -q '"status" *: *"Accepted"' "$f.notary.json" || die "notarization was not Accepted for $f (see $f.notary.json; fetch the log with: xcrun notarytool log <id> --keychain-profile $PROFILE)"
  [[ "$f" == *.dmg ]] && xcrun stapler staple "$f" && xcrun stapler validate "$f"
done
