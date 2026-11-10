# Commercialization readiness report (free distribution first)

Audited 2026-11-10. Status key: **READY**, **PARTIAL**, **BLOCKED**, **OPEN** (needs the owner or a lawyer).

| Area | Status | Notes |
|---|---|---|
| Private core / public showcase split | READY | Core private (`Veystrix-Core`); public repo is a fresh history with no source. |
| Freeware license text | PARTIAL | Draft v0.1 written (`legal/FREEWARE-LICENSE.txt`). Needs the owner's legal name, jurisdiction and a lawyer's review. |
| MIT / third-party notices | PARTIAL | Generated and bundled (`THIRD_PARTY_NOTICES.md`, `Licenses/`). Upstream-copied passages need individual notices (provenance OPEN 4). |
| Ownership of relicensed code | OPEN | **The first 204 commits appear to be by another person (`1architect` / `gioma`).** The owner can distribute the combined work as proprietary freeware only with the MIT notice kept, and must not describe that code as owner-authored (provenance OPEN 1, 2). Lawyer review recommended before announcing. |
| AI-assisted authorship | OPEN | Provenance OPEN 3. |
| Dependency licenses | PARTIAL | No GPL/AGPL found. MPL-2.0 (certifi, tqdm) needs notices; `opencv-python` bundles LGPL FFmpeg: **excluded from the engine bundle** (removed after install, asserted absent by `packaging/engine/smoke_import.py`, 2026-11-10); video-input features that need it are unavailable in the packaged app. |
| SBOM | READY | SPDX 2.3 JSON generated (`legal/sbom.spdx.json`); regenerate at each release. |
| Developer ID signing, Hardened Runtime | BLOCKED | No Developer ID Application identity in the keychain. Scripts and entitlements are written and tested with an ad-hoc signature only. |
| Notarization | BLOCKED | Needs an Apple Developer Program account, an App Store Connect API key or app-specific password. |
| Sparkle EdDSA signing key | BLOCKED | Must be generated on the owner's machine (`generate_keys`) and stored in the keychain; the public key is then put in Info.plist. No key was generated in this unattended session on purpose. |
| Engine bundle (Python runtime) | PARTIAL | Build script written; source is compiled to `.pyc`, which is only light obfuscation (see `docs/packaging/PACKAGING.md`). Not validated with a real checkpoint. |
| Hardware validation | NOT RUN | No checkpoint was loaded (owner travelling). Public docs say so. |
| Privacy | READY | No telemetry; update checks optional; no mandatory account. |
| Support / security contact | PARTIAL | GitHub issues + private vulnerability reporting on the public repo; confirm enabled. |
| Trademark | OPEN | Not cleared. |
| Export control | OPEN | Cryptography use is limited to HTTPS/EdDSA via system and Sparkle libraries; confirm with counsel. |
| Reverse-engineering limits | READY | Documented honestly: a bundled Python engine can be extracted; the license does not rely on obscurity. |
