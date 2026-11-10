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

is_release_signing() { [[ -n "$SIGN_IDENTITY" && "$SIGN_IDENTITY" != "-" ]]; }
artifact_suffix() { if is_release_signing; then echo ""; else echo "-DEV-UNSIGNED"; fi; }
log() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
