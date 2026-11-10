# Shared settings for the Veystrix macOS packaging scripts. Source this file; do not run it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$ROOT/apps/macos/Veystrix"
DIST="${VEYSTRIX_DIST:-$ROOT/dist}"
BUNDLE_ID="io.github.joyboyreturns.veystrix"
APP_NAME="Veystrix"
# Marketing version and build number come from one place.
SHORT_VERSION="${VEYSTRIX_VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1 | sed 's/\.dev[0-9]*$//')}"
BUILD_NUMBER="${VEYSTRIX_BUILD_NUMBER:-$(printf '%d%02d%03d' "${SHORT_VERSION%%.*}" "$(echo "$SHORT_VERSION" | cut -d. -f2)" "$(echo "$SHORT_VERSION" | cut -d. -f3)")1}"   # MAJOR.MINOR.PATCH -> MMmmmpppB, e.g. 3.8.1 -> 3080011 (build 1); strictly increasing
ARCH="arm64"
# Signing: a Developer ID Application identity, e.g. "Developer ID Application: Name (TEAMID)". Unset => ad-hoc DEVELOPMENT build.
SIGN_IDENTITY="${VEYSTRIX_SIGN_IDENTITY:-}"
SPARKLE_VERSION="2.10.1"
SPARKLE_TARBALL_SHA256="1a856d2a9edb8e628de88121d7be04cf0e2167f8641650b78fca00b14244cf27"   # matches the digest GitHub publishes for the 2.10.1 release asset

# Sparkle update PUBLIC key (not a secret). The private key lives only in the owner's keychain / CI secret store.
SPARKLE_PUBLIC_KEY="${VEYSTRIX_SPARKLE_PUBLIC_KEY:-$(tr -d '[:space:]' < "$SCRIPT_DIR/sparkle_public_key.txt" 2>/dev/null || true)}"

# Two release channels, both legitimate:
#   developer-id  signed with a Developer ID + Hardened Runtime and notarized (needs an Apple Developer Program membership)
#   free          ad-hoc signed, NOT notarized: no Apple membership needed; macOS Gatekeeper warns on first open.
#                 Integrity rests on the published SHA-256 and (for updates) the Sparkle EdDSA signature.
# Same file names in both channels; BUILD-CHANNEL inside the app and BUILD-INFO.txt say which one a build is.
is_release_signing() { [[ -n "$SIGN_IDENTITY" && "$SIGN_IDENTITY" != "-" ]]; }
channel() { if is_release_signing; then echo "developer-id"; else echo "free"; fi; }
artifact_suffix() { echo ""; }
log() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
