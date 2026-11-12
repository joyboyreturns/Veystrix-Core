"""Launch Kimi with a private profile for a single Qwen–Nemotron swarm."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib
from urllib.parse import urlsplit
from uuid import uuid4

import tomlkit


def main() -> None:
    kimi = shutil.which("kimi")
    if kimi is None:
        raise SystemExit("kimi is not on PATH")

    source = Path(os.environ.get("KIMI_CODE_HOME", Path.home() / ".kimi-code")) / "config.toml"
    with source.open("rb") as config_file:
        config = tomllib.load(config_file)

    models = config.get("models", {})
    qwen_alias = config.get("default_model")
    nemotron_alias = config.get("secondary_model", {}).get("default_model")
    qwen = models.get(qwen_alias, {})
    nemotron = models.get(nemotron_alias, {})
    if not qwen.get("model", "").startswith("veystrix-flashnext-") or not nemotron.get("model", "").startswith(
        "veystrix-nemotron35-lightning-"
    ):
        raise SystemExit("Select Veystrix Qwen as the Kimi main model and Veystrix Nemotron as its secondary model.")

    provider_name = qwen.get("provider")
    provider = config.get("providers", {}).get(provider_name, {})
    address = urlsplit(provider.get("base_url", ""))
    if (
        provider_name != nemotron.get("provider")
        or provider.get("type") != "openai"
        or address.scheme != "http"
        or address.hostname != "127.0.0.1"
    ):
        raise SystemExit("Both model aliases must use the same Veystrix provider at http://127.0.0.1.")

    headers = {
        name: value
        for name, value in provider.get("custom_headers", {}).items()
        if name.lower() != "x-veystrix-workflow"
    }
    provider["custom_headers"] = {**headers, "X-Veystrix-Workflow": uuid4().hex}
    profile = {
        "default_model": qwen_alias,
        "providers": {provider_name: provider},
        "models": {qwen_alias: qwen, nemotron_alias: nemotron},
        "secondary_model": {"default_model": nemotron_alias},
    }
    profile_dir = Path(tempfile.mkdtemp(prefix="kimi-veystrix-"))
    with open(
        profile_dir / "config.toml", "x", encoding="utf-8", opener=lambda path, flags: os.open(path, flags, 0o600)
    ) as config_file:
        config_file.write(tomlkit.dumps(profile))

    print(f"Kimi swarm profile: {profile_dir}", flush=True)
    environment = dict(os.environ, KIMI_CODE_HOME=str(profile_dir), KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY="1")
    os.execve(kimi, [kimi, *sys.argv[1:]], environment)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot launch Kimi swarm: {exc}") from None
