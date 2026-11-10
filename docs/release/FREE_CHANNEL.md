# Free channel: shipping Veystrix without an Apple Developer membership

Decision (owner, 2026-11-10): Veystrix is a free macOS app and does **not** require an Apple Developer Program
membership at this stage. Signing, notarization and the Sparkle feed stay ready for later activation.

## What the free channel is

| | Free channel (default) | Developer ID channel (ready, not activated) |
|---|---|---|
| Needs | nothing | Apple Developer Program, Developer ID cert, notary credentials |
| Code signature | ad-hoc (`codesign -s -`), no Hardened Runtime | Developer ID + Hardened Runtime |
| Notarized | **No** | Yes (`make_release.sh --notarize`) |
| First open on a downloaded copy | macOS Gatekeeper warns; user approves once | opens normally |
| Integrity anchor | SHA-256 in `SHA256SUMS.txt` | same + Apple signature/notarization |
| In-app updates | Sparkle, authenticated by the **EdDSA** signature (key already generated) | same |
| File names | `Veystrix-<ver>-arm64.dmg/.zip`, `SHA256SUMS.txt`, `BUILD-INFO.txt` | same |

`BUILD-INFO.txt` (also inside the DMG and as `Contents/Resources/BUILD-CHANNEL`) states the channel. A free build must
never be described as Developer-ID-signed or notarized.

## Build it

```sh
packaging/engine/build_engine_bundle.sh dist/engine          # pinned Python + engine, no model
packaging/macos/make_release.sh --engine dist/engine         # free channel (no identity set)
```

Nothing here loads a model. Output in `dist/` (git-ignored).

## What users see on first open (macOS 15 and later)

Downloaded apps carry a quarantine flag. An ad-hoc-signed app is blocked with "cannot verify it is free of malware".
Instructions for users (also in the public `docs/installation.md`):

1. Drag Veystrix to Applications and open it once (it will be blocked).
2. System Settings > Privacy & Security > scroll to the message about Veystrix > **Open Anyway** > confirm.
3. Or, for users comfortable with Terminal, after verifying the SHA-256:
   `xattr -dr com.apple.quarantine /Applications/Veystrix.app`

These are Apple's documented steps for unidentified developers; the exact wording changes between macOS versions.
Not yet tested on a downloaded copy of a published release (no release exists).

## Security trade-offs, honestly

* Without notarization Apple has not scanned the build and macOS cannot vouch for the publisher. The user's only
  protection against a tampered download is verifying the SHA-256 from the official Releases page (HTTPS, GitHub account
  `joyboyreturns`) and not running re-hosted copies. The EULA says so.
* Sparkle updates are still authenticated: the app only installs an archive whose EdDSA signature verifies against the
  public key compiled into it. **Guard the private key** (login keychain + an offline backup). Whether Sparkle installs
  an update over an ad-hoc-signed app end to end has **not** been verified yet: test with two successive local builds
  before relying on it.
* Without Hardened Runtime the app has weaker OS-level exploit mitigations than a notarized build.

## Activating the Developer ID channel later

Set the secrets listed in `docs/release/RELEASE_PROCESS.md` (certificate `.p12`, notary credentials, identity). The
release workflow switches channel automatically; no script change is needed. Existing free-channel users keep updating
via Sparkle, but the Apple signing identity changes from ad-hoc to Developer ID: test that transition on a spare Mac
first (Sparkle may refuse an update whose code-signing identity differs when the old build was ad-hoc signed).
