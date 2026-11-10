#!/bin/bash
# Make the zip and dmg artifacts. Usage: packaging/macos/make_dmg.sh <Veystrix.app>   (outputs into $DIST)
source "$(dirname "$0")/common.sh"
APP="${1:-$DIST/$APP_NAME.app}"; [[ -d "$APP" ]] || die "no app at $APP"
SUF="$(artifact_suffix)"
ZIP="$DIST/Veystrix-$SHORT_VERSION-$ARCH$SUF.zip"
DMG="$DIST/Veystrix-$SHORT_VERSION-$ARCH$SUF.dmg"
rm -f "$ZIP" "$DMG"
log "Creating $ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
[[ -f "$DIST/BUILD-INFO.txt" ]] && cp "$DIST/BUILD-INFO.txt" "$STAGE/BUILD-INFO.txt"
cp "$ROOT/legal/FREEWARE-LICENSE.txt" "$STAGE/LICENSE.txt"; cp "$ROOT/legal/THIRD_PARTY_NOTICES.md" "$STAGE/THIRD_PARTY_NOTICES.md"
log "Creating $DMG"
hdiutil create -quiet -volname "Veystrix $SHORT_VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
if is_release_signing; then codesign --force --sign "$SIGN_IDENTITY" --timestamp "$DMG"; fi
log "Done: $(basename "$ZIP"), $(basename "$DMG")"
