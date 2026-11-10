# MACQWEN → Veystrix: staged migration

Veystrix is the current name. The rename is staged so that nothing the owner already depends on
breaks: caches, saved sessions, scripts, harness clients and measurement records.

## Stage 1 — done on `development3.81`

| Layer | Change | Compatibility |
|---|---|---|
| Import name | `import veystrix` / `python -m veystrix`. `veystrix.<module>` is the same object as `macqwen.<module>`. | `import macqwen...` unchanged. |
| Console script | `veystrix` added. | `macqwen` script kept. |
| Package metadata | distribution `veystrix`, version `3.8.1.dev0`, license, URLs. | Editable installs of the old name keep working until re-installed. |
| Environment variables | `VEYSTRIX_<NAME>` is accepted for every `MACQWEN_<NAME>` the engine reads (`macqwen/env_compat.py`); the Veystrix name wins if both are set. | Old names unchanged. A drift test fails if a new `MACQWEN_` read is not aliased. |
| Files | `~/.veystrix/preferences.json`, `~/Library/Application Support/Veystrix/api_keys.json`. | The old file is copied once on first use, never moved or deleted; preferences also keep reading the old path as a fallback. |
| Strings | CLI/setup messages, `/v1/models` display names (`Veystrix Flash-Next`, ...), User-Agent, web terminal title. | Model **slugs** are unchanged, so clients that select by slug are unaffected. |
| Docs | README, SECURITY, CONTRIBUTING, CHANGELOG, NOTICE rebranded. | `docs/**` research records, `results/**` and commit history keep the old name on purpose (historical evidence). |

## Deliberately NOT renamed

* **`macqwen/` and `models/flashnext/` directories.** Saved sessions pickle module paths; the loader
  already rewrites `models.flashnext.` for older files (`models/flashnext/sessions.py`). Renaming modules
  would orphan existing sessions and prefix caches.
* **`~/.flashnext/`, `~/.cache/flashnext/` and other cache directories.** Moving them would discard
  hundreds of GB of cache and break admission records.
* **Historical docs and `results/`.** They are evidence; their commands refer to the names that existed.
* **`FLASHNEXT_*` engine flags** (these name the Flash-Next runtime, not the product).

## Later stages (not started)

1. Move the implementation into `veystrix/` and make `macqwen/` the thin alias (needs a session-format
   version bump and a one-time, reversible cache-index migration).
2. Deprecation warnings for `MACQWEN_*` and the `macqwen` script.
3. Remove the aliases only after a release in which no supported client uses them.

## Rollback

Everything in Stage 1 is additive. Reverting the commit restores the previous behavior; files that were
copied to the new locations are harmless.
