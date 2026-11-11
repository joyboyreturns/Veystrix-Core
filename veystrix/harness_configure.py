"""Configure installed harness catalogs with backups. No weights are loaded."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

from veystrix.harness_models import qwen_config, deepseek_config
from veystrix.model_registry import discover


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
            data = yaml.safe_load(path.read_text())
            output = yaml.safe_dump(deepseek_config(data, models, port), sort_keys=False, allow_unicode=True)
            reports[label] = _write(path, output) if apply else "ready to update"
        except (OSError, ValueError, yaml.YAMLError) as exc:
            reports[label] = "failed: " + str(exc)
    # Hermes supports dynamic GET /v1/models for its custom-provider picker. Its
    # dedicated profile is configured by integrations/hermes/configure_profile.py.
    reports["Hermes"] = "use Veystrix custom provider and refresh its model picker"
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
