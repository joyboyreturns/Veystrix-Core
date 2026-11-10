# Hermes Agent + Veystrix

Hermes Agent (NousResearch, tested v0.23.1) can drive the local Veystrix engine as a custom OpenAI-compatible
provider. Installed from the official installer (`curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`
— review the script first; the pinned copy we reviewed is not stored in Git).

## Configure a dedicated profile (never touches your default profile)

```sh
python3 integrations/hermes/configure_profile.py --dry-run     # show the merged config, write nothing
python3 integrations/hermes/configure_profile.py               # writes ~/.hermes/profiles/veystrix/config.yaml (+ .bak-<UTC>)
```

It sets provider `veystrix` → `http://127.0.0.1:8080/v1`, key from `VEYSTRIX_API_KEY`, model `macqwen-flashnext`,
no fallbacks, no cloud routes, auxiliary tasks local or off, update-check and telemetry off. `--port N` and
`--context N` are available; the context must be ≥ 65,536 (Hermes refuses smaller windows).

## Use it (engine started by you, on AC power; see docs/MANUAL_LAUNCH.md)

```sh
HERMES_HOME=~/.hermes/profiles/veystrix hermes          # or:  hermes -p veystrix
hermes -p veystrix -z "one-shot prompt"                  # detached gateway run
```

* The engine must report a window ≥ 64K: `./serve.sh` passes `--context-limit 262144`. Its admission guard still
  refuses an *uncached* prompt above `--max-cold-prefill` (32,768 by default), so Hermes' compression happens
  below the advertised window.
* **Cancellation:** in `-z` mode Ctrl-C **detaches** ("accepted work continues at the gateway") and does not cancel
  the model request: the engine keeps prefilling. To stop it: `hermes -p veystrix gateway stop --yes`.
* Use the id from `GET /v1/models` (`macqwen-flashnext`).

## What is tested (offline, no model)

`macqwen/tests/test_hermes_integration.py` runs the **real Hermes CLI** against `mock_veystrix_server.py`, a
protocol-only imitation of the engine: streaming with `reasoning_content`, tool call + tool-result continuation,
multi-turn persistence across invocations, context-length error not retried, admission refusal retried at most once,
cancellation via gateway stop and reconnection, model/context discovery, no unexpected routes, and a tripwire proxy
asserting **zero** outbound connections. Enable with `VEYSTRIX_HERMES_E2E=1`.

**NOT verified:** behaviour against the real 113 GB checkpoint (speed, quality, long prefill, memory guard
interaction). NOT RUN — owner travelling.
