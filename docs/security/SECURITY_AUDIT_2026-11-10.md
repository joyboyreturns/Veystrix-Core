# Security and provenance audit — 2026-11-10 (development3.81)

Scope: the private engine repository (`Veystrix-Core`), the new macOS app, the packaging scripts and the public
repository content. Method: source inspection and automated pattern scans only. **No model was loaded, no engine
was started, no real-checkpoint test was run.** A clean scan is evidence, not proof; this is not a third-party audit.

## 1. Cracked or unauthorized software ("REA-related" and similar)

* Searched tracked files and history for crack/keygen/activation-bypass terms, license-check circumvention,
  patcher scripts and file names such as `rea*`. Result: **nothing found.** One text hit ("bypassed" in a Metal
  kernel note) is unrelated.
* Tracked binaries: **none** (no `.dmg/.pkg/.app/.dylib/.exe/.zip`, no Mach-O). Largest tracked files are PNG traces
  and a JSONL result log (≤ 3.7 MB).
* No DRM, anti-debugging or license-enforcement code exists in the repository, and none was added.
* Limit: this cannot prove that no third-party file was ever copied in from an unauthorized source. Provenance of
  authorship is a separate, still-open item: see `legal/LICENSE_PROVENANCE.md`.

## 2. Secrets

* `scripts/security/secret_scan.py` (token patterns: GitHub, OpenAI/Anthropic-style, AWS, Google, Slack, Hugging
  Face, private-key PEM headers, JWT-shaped) over the working tree: **0 hits**; over every blob reachable from any
  ref (3,989 blobs): **0 hits**. Deliberately fake tokens in the app's redaction test are allowlisted by path.
* Signing material patterns (`*.p12`, `*.p8`, `*.pem`, `*.key`, notary profiles, Sparkle private keys) are in
  `.gitignore`. Release secrets are only read from the GitHub `release` environment or the owner's keychain.
* Not verified: secrets that were committed and deleted from a branch not present in this clone.

## 3. Network exposure, authentication, origins

| Component | Finding |
|---|---|
| Engine HTTP server (`macqwen/server.py`, `models/flashnext/overnight.py`) | Defaults to `127.0.0.1`. The app always passes `--host 127.0.0.1` and has no setting to change it. Bearer/`x-api-key` compared with `hmac.compare_digest`. Foreign browser `Origin` headers are refused unless explicitly allowed with `--allow-origin`; requests without an Origin header (non-browser clients) are unaffected. |
| `web_terminal.py` (repo root) | **Binds `0.0.0.0` and gives a shell** to anyone holding a token (token stored in `~/.frankenstein/web_token`). The token requirement mitigates but a LAN-reachable shell is high risk. It is **not** part of the app or the engine bundle (the bundle contains only `macqwen/`, `veystrix/`, `models/{flashnext,qwen27b}`) and is **not** published in the public repository. Recommendation: keep out of any release; consider removing or binding loopback by default. |
| Hermes integration | Profile points only at `http://127.0.0.1:<port>/v1`. |
| App | Talks to the engine on loopback and to the update feed on `github.com` (optional, user-switchable). No other network client code. |
| Search/doc tools (`macqwen/tools/*`: Tavily, free search, Context7, api_guard) | Outbound HTTP **only when an agent tool is invoked by the user's workflow**; they are not started by the engine server path used by the app. Not part of the first public app's documented feature set. |

## 4. Telemetry

No analytics, crash reporting or phone-home code in the app, engine or packaging (searched for common SDK names and
for outbound calls; the only automatic outbound call is the Sparkle update check). Sparkle's opt-in system
profiling is not enabled (`SUEnableSystemProfiling` is absent from `Info.plist`; `SUEnableAutomaticChecks` and
automatic downloads are `false`, so no check happens until the user asks or turns it on). The Hermes
end-to-end run observed **zero connection attempts** to anything but the loopback mock (tripwire proxy captured none).

## 5. Update signature validation

* Sparkle 2.10.1, pinned; its tarball SHA-256 is verified before the tools run.
* `SUPublicEDKey` ships in `Info.plist`; with it empty the app **disables updates** (development builds).
* `packaging/macos/validate_appcast.py` enforces https + host allowlist + 64-byte EdDSA signatures + strictly
  decreasing build numbers, and can verify each signature against the public key with a self-contained RFC 8032
  verifier (tested on the RFC vectors and with tampering).
* Not verified: a real signed update installing end to end (no Developer ID, no Sparkle key yet).

## 6. Code-execution surfaces to keep out of the first public app

* `models/k2_horizon/backend.py` passes `trust_remote_code=True` to a tokenizer loader (also the Bonsai-2 backend
  family). These backends are excluded from the engine bundle.
* `models/flashnext/tests/bench/*` mention the flag in benchmarks; not shipped.
* The compiled `.pyc` bundle is **not** protection against inspection and must not be described as such.

## 7. Sandbox and entitlements

The app is not sandboxed (it must spawn an interpreter and read a user-chosen multi-GB folder). Hardened Runtime
entitlements are in `apps/macos/Veystrix/Support/*.entitlements`; whether MLX's Metal JIT and `mmap` of the model
need `allow-jit` / `allow-unsigned-executable-memory` or `disable-library-validation` is **unverified until a
signed bundle is run on hardware**. Do not claim notarized compatibility before that test.

## Open items

1. Decide the fate of `web_terminal.py`.
2. Real signed + notarized build and a real-hardware run (NOT RUN: owner travelling).
3. Third-party review of the EULA and provenance (`legal/`).
