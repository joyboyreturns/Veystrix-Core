"""Private API keys stored outside the repository."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile


DEFAULT_PATH = "~/Library/Application Support/Veystrix/api_keys.json"
# Before Veystrix 3.8.1 the file lived under the old project name. It is copied (never moved or
# deleted) to DEFAULT_PATH the first time the default store is opened.
LEGACY_PATH = "~/Library/Application Support/MACQWEN/api_keys.json"


def migrate_legacy(new: str | Path, legacy: str | Path) -> bool:
    """Copy a legacy key file to ``new`` (mode 0600, parent 0700) when ``new`` is absent.

    Returns True if a copy was made. The legacy file is left in place so a downgrade still works.
    """
    new_path, old_path = Path(new).expanduser(), Path(legacy).expanduser()
    if new_path.exists() or not old_path.is_file():
        return False
    try:
        data = old_path.read_text()
        new_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(new_path.parent, 0o700)
        descriptor = os.open(new_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(data)
        os.chmod(new_path, 0o600)
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class Service:
    name: str
    environment: str
    required: bool


SERVICES = {
    "tavily": Service("Tavily", "TAVILY_API_KEY", True),
    "context7": Service("Context7", "CONTEXT7_API_KEY", False),
}

_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY", re.I)


def sanitized_environment() -> dict[str, str]:
    """Return the process environment without secret-like variables."""
    return {
        name: value
        for name, value in os.environ.items()
        if not _SECRET_NAME.search(name)
    }


class KeyStore:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path).expanduser()
        if self.path == Path(DEFAULT_PATH).expanduser():
            migrate_legacy(self.path, LEGACY_PATH)
        self._external = {
            service.environment: os.environ.get(service.environment)
            for service in SERVICES.values()
        }
        self.values = self._load()
        self._secure_existing()
        self.apply()

    def _secure_existing(self) -> None:
        if not self.path.is_file():
            return
        os.chmod(self.path, 0o600)
        os.chmod(self.path.parent, 0o700)

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, TypeError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            name: value.strip()
            for name, value in raw.items()
            if name in SERVICES and isinstance(value, str) and value.strip()
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=".api_keys.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(self.values, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def apply(self) -> None:
        for name, value in self.values.items():
            os.environ[SERVICES[name].environment] = value

    def resolve(self, name: str) -> str:
        key = name.strip().lower()
        if key not in SERVICES:
            raise ValueError(f"unknown service {name!r}; use: {', '.join(SERVICES)}")
        return key

    def set(self, name: str, value: str) -> None:
        key = self.resolve(name)
        secret = value.strip()
        if not secret or "\n" in secret or "\r" in secret:
            raise ValueError("API key must be one non-empty line")
        self.values[key] = secret
        self._save()
        os.environ[SERVICES[key].environment] = secret

    def delete(self, name: str) -> bool:
        key = self.resolve(name)
        if key not in self.values:
            return False
        del self.values[key]
        self._save()
        environment = SERVICES[key].environment
        external = self._external.get(environment)
        if external is None:
            os.environ.pop(environment, None)
        else:
            os.environ[environment] = external
        return True

    def source(self, name: str) -> str:
        key = self.resolve(name)
        if key in self.values:
            return "Application Support"
        if os.environ.get(SERVICES[key].environment, "").strip():
            return "environment"
        return "missing"

    def status(self) -> str:
        rows = []
        for name, service in SERVICES.items():
            source = self.source(name)
            need = "required" if service.required else "optional"
            state = f"configured in {source}" if source != "missing" else "not configured"
            rows.append(f"  {name:<10} {state} ({need})")
        return "\n".join(rows)
