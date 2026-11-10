#!/usr/bin/env python3
"""Configure a dedicated Hermes Agent profile to use the local Veystrix engine.

    python3 integrations/hermes/configure_profile.py --dry-run
    python3 integrations/hermes/configure_profile.py                # writes ~/.hermes/profiles/veystrix/config.yaml

What it does (and nothing else):
  * backs up the existing config.yaml / .env next to themselves (config.yaml.bak-<UTC timestamp>) before any write;
  * MERGES into the existing config: only the keys below are set; other providers and settings are preserved;
  * points the model at http://127.0.0.1:<port>/v1 as a custom OpenAI-compatible endpoint;
  * turns off every route that could reach a cloud provider by accident: no fallback providers, auxiliary tasks
    pinned to the same local endpoint (or disabled), no borrowed Codex/Claude logins, no passive update check, no
    shared-metrics telemetry, and no web/browser/image/TTS toolsets;
  * never invents a secret: the key is read from VEYSTRIX_API_KEY. The profile .env gets the documented loopback
    placeholder `local` only if the variable is not already set there (the local server accepts any value unless
    it was started with a server key).

It never starts Hermes, the Veystrix engine, or any model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_MODEL_ID = "macqwen-flashnext"       # what the engine serves for the Flash-Next runtime (GET /v1/models .id)
# Hermes Agent refuses a model whose window is below 64,000 tokens (agent_init._enforce_minimum_context, v0.23.1), and its
# compression floor for windows under 512K is 75% of the window. 65,536 is the smallest value it accepts. The supervised launcher
# (serve.sh) reports a 262,144-token window with --max-cold-prefill 32768, so Hermes will compress long before the window is
# full, but an uncached prompt above 32,768 tokens would still be refused by the engine's admission guard.
DEFAULT_CONTEXT = 65536
KEY_ENV = "VEYSTRIX_API_KEY"
# Extra model requests evict the engine's live cache, so they are disabled rather than redirected where possible.
AUX_TASKS_REDIRECT = ("compression", "session_search", "vision", "web_extract")
AUX_TASKS_OFF = ("title_generation", "background_review")
LOCAL_TOOLSETS = ["terminal", "file", "todo", "skills"]


def desired_config(port: int, model_id: str, context_length: int) -> dict:
    base = f"http://127.0.0.1:{port}/v1"
    aux: dict = {}
    for task in AUX_TASKS_REDIRECT:
        aux[task] = {"provider": "custom", "model": model_id, "base_url": base, "api_key_env": KEY_ENV}
    for task in AUX_TASKS_OFF:
        aux[task] = {"enabled": False, "provider": "custom", "model": model_id, "base_url": base, "api_key_env": KEY_ENV}
    return {
        # `providers.veystrix` is a named custom provider (identity in config, credential in key_env); the model section selects it.
        "model": {"default": model_id, "provider": "veystrix", "base_url": base, "context_length": context_length},
        "providers": {"veystrix": {"name": "Veystrix", "base_url": base, "key_env": KEY_ENV,
                                    # a cold 8K-token prefill takes minutes; the engine sends ': generating' every 15 s meanwhile
                                    "request_timeout_seconds": 3600, "stale_timeout_seconds": 3600}},
        "fallback_providers": [],
        "auxiliary": aux,
        # Hermes floors api_max_retries at 1 (agent_init.py), so one retry after a transient 5xx remains; auto-recovery is off.
        # local_stream_stale_timeout is how long a local stream may be silent before Hermes drops it (default 900 s).
        "agent": {"api_max_retries": 1, "auto_recovery_cycles": 0, "local_stream_stale_timeout": 3600},
        "auth": {"adopt_external_logins": False},
        "updates": {"check": False},
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        "platform_toolsets": {"cli": list(LOCAL_TOOLSETS)},
    }


def deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path) -> dict:
    import yaml
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a YAML mapping; refusing to touch it")
    return data


def backup(path: Path, stamp: str) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, target)
    os.chmod(target, 0o600)
    return target


def ensure_env_placeholder(env_path: Path) -> bool:
    """Add KEY=local only if the profile .env has no VEYSTRIX_API_KEY. Returns True if it wrote."""
    text = env_path.read_text() if env_path.exists() else ""
    if any(line.split("=", 1)[0].strip() == KEY_ENV for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#")):
        return False
    with open(env_path, "a") as handle:
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write("# Veystrix loopback key. 'local' is the documented placeholder; the engine accepts any value unless it was\n"
                     "# started with a server key, in which case put that key here instead.\n"
                     f"{KEY_ENV}=local\n")
    os.chmod(env_path, 0o600)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hermes-home", type=Path, default=Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "profiles" / "veystrix",
                    help="the profile directory to configure (default ~/.hermes/profiles/veystrix)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VEYSTRIX_PORT", "8080")))
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    ap.add_argument("--dry-run", action="store_true", help="print the merged config and change nothing")
    args = ap.parse_args(argv)

    if not (1024 <= args.port <= 65535):
        raise SystemExit("port must be between 1024 and 65535")
    config_path = args.hermes_home / "config.yaml"
    env_path = args.hermes_home / ".env"
    import yaml
    merged = deep_merge(load_yaml(config_path), desired_config(args.port, args.model_id, args.context_length))
    rendered = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)
    if args.dry_run:
        print(rendered)
        return 0
    if not args.hermes_home.is_dir():
        raise SystemExit(f"{args.hermes_home} does not exist. Create the profile first:  hermes profile create veystrix --no-alias")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups = [b for b in (backup(config_path, stamp), backup(env_path, stamp)) if b]
    tmp = config_path.with_suffix(".yaml.tmp")
    tmp.write_text(rendered)
    os.chmod(tmp, 0o600)
    os.replace(tmp, config_path)
    wrote_key = ensure_env_placeholder(env_path)
    print(json.dumps({"configured": str(config_path), "base_url": f"http://127.0.0.1:{args.port}/v1", "model": args.model_id,
                      "backups": [str(b) for b in backups], "env_placeholder_added": wrote_key}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
