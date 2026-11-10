# Release process (private repo -> public Releases)

Status 2026-11-10: **no public release has been made.** Everything below that needs credentials is blocked until
the owner provides them. Nothing here is simulated.

## One-time owner setup

1. **Apple Developer Program** membership. Create a *Developer ID Application* certificate (Xcode > Settings >
   Accounts > Manage Certificates, or developer.apple.com). Export it as a `.p12` with a strong password.
2. **Notarization credentials**: an app-specific password (appleid.apple.com) or an App Store Connect API key.
   Locally: `xcrun notarytool store-credentials VEYSTRIX_NOTARY --apple-id <id> --team-id <TEAMID> --password <app-specific>`.
3. **Sparkle EdDSA keys**: run Sparkle's `generate_keys` once on the owner's Mac (private key goes to the login
   keychain). `generate_keys -p` prints the public key: set it as `VEYSTRIX_SPARKLE_PUBLIC_KEY` (it is compiled into
   `Info.plist`). Export the private key with `generate_keys -x file` into a password manager; never into Git.
   **Losing it means existing installs can never be updated.**
4. **Public repo token**: a fine-grained personal access token limited to repository `joyboyreturns/Veystrix` with
   only *Contents: Read and write*. Secret name `PUBLIC_RELEASES_TOKEN`.
5. In the private repo create the GitHub environment `release` (required reviewer = owner) and store:
   `VEYSTRIX_DEVID_P12_BASE64`, `VEYSTRIX_DEVID_P12_PASSWORD`, `VEYSTRIX_SIGN_IDENTITY`, `VEYSTRIX_NOTARY_APPLE_ID`,
   `VEYSTRIX_NOTARY_TEAM_ID`, `VEYSTRIX_NOTARY_PASSWORD`, `VEYSTRIX_SPARKLE_PRIVATE_KEY`,
   `VEYSTRIX_SPARKLE_PUBLIC_KEY`, `PUBLIC_RELEASES_TOKEN`.
6. Complete the bracketed fields in `legal/FREEWARE-LICENSE.txt` and get it reviewed.

## Each release

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`; regenerate notices from the *bundle's* environment:
   `dist/engine/python/bin/python3 packaging/legal/generate_legal.py --version X.Y.Z --public-out <public checkout>`.
2. Run the full validation list (`docs/release/VALIDATION.md`), on AC power for any measured claim.
3. Tag the private repo: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`. The Release workflow tests, builds
   the engine bundle, signs, notarizes, builds the appcast and creates a **draft** release in the public repo. If any
   credential is missing it builds a clearly labelled DEV artifact and publishes nothing.
4. Review the draft: download from the public page, `shasum -a 256 -c`, `spctl -a -vv Veystrix.app`,
   `xcrun stapler validate Veystrix-*.dmg`; run once on hardware. Publish the draft.
5. Commit `updates/appcast.xml` (from the workflow artifact) to the public repo **after** the assets are live;
   update README/CHANGELOG/performance pages there.
6. Run `scripts/security/public_exposure_audit.py` and a fresh unauthenticated clone check.

## Local (manual) release candidate

`packaging/engine/build_engine_bundle.sh dist/engine`, then
`VEYSTRIX_SIGN_IDENTITY="Developer ID Application: …" VEYSTRIX_SPARKLE_PUBLIC_KEY=… packaging/macos/make_release.sh --engine dist/engine --notarize`.
Without the identity the output is named `*-DEV-UNSIGNED*` and the script says public distribution is blocked.
