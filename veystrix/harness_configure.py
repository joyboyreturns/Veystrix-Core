"""Configure installed harness catalogs with backups. No weights are loaded."""
from __future__ import annotations
import argparse
import datetime
import json
import os
import re
from pathlib import Path
import shutil
import tempfile

from veystrix.harness_models import qwen_config, deepseek_config, model_context
from veystrix.model_registry import discover


def load_yaml12(text):
    # DSH uses YAML 1.2: unquoted "off" is a string, not YAML 1.1 false.
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    Loader.yaml_implicit_resolvers = {key: [(tag, pattern) for tag, pattern in values
                                            if tag != "tag:yaml.org,2002:bool"]
                                      for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
    return yaml.load(text, Loader=Loader)


def _write(path: Path, content: str):
    if path.is_symlink():
        raise ValueError("refusing to overwrite a symlink")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = path.read_text() if path.exists() else None
    if current == content:
        return "unchanged"
    if path.exists():
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(path.name + ".veystrix-backup-" + stamp)
        shutil.copy2(path, backup)
        os.chmod(backup, 0o600)
    fd, temporary = tempfile.mkstemp(prefix=".veystrix-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return "updated (backup created)" if current is not None else "created"


def run(*, home=None, port=8080, apply=False):
    import yaml
    home = Path(home or Path.home())
    models = discover()
    if not models:
        raise ValueError("no complete supported installations found")
    reports = {}
    qwen = home / ".qwen" / "settings.json"
    if qwen.exists():
        try:
            data = json.loads(qwen.read_text())
            output = json.dumps(qwen_config(data, models, port), indent=2, ensure_ascii=False) + "\n"
            reports["Qwen Code"] = _write(qwen, output) if apply else "ready to update"
        except (OSError, ValueError) as exc:
            reports["Qwen Code"] = "failed: " + str(exc)
    else:
        reports["Qwen Code"] = "not installed or no settings file"
    for profile in ("desktop", "headless"):
        path = home / ".dsh" / "profiles" / profile / "cordis.patch.yml"
        label = "DeepSeek " + profile
        if not path.exists():
            reports[label] = "not installed or no profile"
            continue
        try:
            data = load_yaml12(path.read_text())
            output = yaml.safe_dump(deepseek_config(data, models, port), sort_keys=False, allow_unicode=True)
            reports[label] = _write(path, output) if apply else "ready to update"
        except (OSError, ValueError, yaml.YAMLError) as exc:
            reports[label] = "failed: " + str(exc)
    # Configure only the dedicated profile, preserving the user's main Hermes config.
    hermes_dir = home / ".hermes"
    if not hermes_dir.is_dir():
        reports["Hermes"] = "not installed"
    else:
        path = hermes_dir / "profiles" / "veystrix" / "config.yaml"
        try:
            data = load_yaml12(path.read_text()) if path.exists() else {}
            if data is None: data = {}
            if not isinstance(data, dict): raise ValueError("Hermes profile is not a mapping")
            providers = data.setdefault("custom_providers", [])
            if not isinstance(providers, list) or not all(isinstance(p, dict) for p in providers):
                raise ValueError("invalid Hermes custom providers")
            base = f"http://127.0.0.1:{port}/v1"
            matches = [p for p in providers if p.get("name", "").lower() == "veystrix"]
            if len(matches) > 1: raise ValueError("multiple Veystrix providers exist")
            if matches and matches[0].get("base_url") not in (None, base):
                raise ValueError("existing Veystrix provider uses a different endpoint")
            provider = matches[0] if matches else {"name": "Veystrix"}
            if not matches: providers.append(provider)
            provider.update({"base_url": base, "key_env": "VEYSTRIX_API_KEY",
                             "models": {m.id: {"context_length": model_context(m)} for m in models}})
            model = data.setdefault("model", {})
            if not isinstance(model, dict): raise ValueError("invalid Hermes model settings")
            model.setdefault("default", models[0].id)
            model.setdefault("provider", "custom:veystrix")
            model.setdefault("base_url", base)
            data.setdefault("fallback_providers", [])
            output = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
            reports["Hermes"] = _write(path, output) if apply else "ready to update dedicated profile"
            reports["Hermes"] += "; launch with HERMES_HOME=~/.hermes/profiles/veystrix"
        except (OSError, ValueError, yaml.YAMLError) as exc:
            reports["Hermes"] = "failed: " + str(exc)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--apply", action="store_true", help="apply changes after backing up existing files")
    args = parser.parse_args()
    try:
        result = run(port=args.port, apply=args.apply)
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0

if __name__ == "__main__": raise SystemExit(main())
