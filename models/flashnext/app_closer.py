"""OPTIONAL graceful closing of applications the owner explicitly lists. Disabled unless authorized.

Nothing here runs by default. It needs a config file the owner writes (``~/.flashnext/close-apps.json``)::

    {"enabled": true, "apps": ["Slack", "Mail"], "wait_seconds": 20}

and then only the listed applications are asked to quit, the way a person would (AppleScript ``quit``), so an application
with unsaved work shows its own save prompt and stays open: this module waits, reports it as still running, and gives up. It
never sends a signal, never kills, never forces. A fixed protected list (the window server, Finder, the Dock, Siri, terminals,
Claude Code, the login window) can never be closed even if it is listed. The application name is restricted to plain
characters so it cannot be used to inject AppleScript.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

CONFIG_PATH = "~/.flashnext/close-apps.json"
PROTECTED = frozenset(name.lower() for name in (
    "WindowServer", "Finder", "Dock", "SystemUIServer", "loginwindow", "Siri", "Siri AI", "ControlCenter",
    "NotificationCenter", "Spotlight", "Terminal", "iTerm", "iTerm2", "Claude", "claude", "claude.exe", "Claude Code",
    "Gemini", "gemini", "Antigravity", "antigravity", "agy",
    "kernel_task", "launchd", "python", "Python", "chat.sh"))
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,60}$")


def _run(command, timeout=30):
    return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


class AppCloser:
    def __init__(self, path=CONFIG_PATH, runner=_run, sleep=time.sleep, clock=time.monotonic):
        self.path = os.path.expanduser(path)
        self._run, self._sleep, self._clock = runner, sleep, clock

    def config(self) -> dict:
        try:
            with open(self.path) as handle:
                value = json.load(handle)
        except (OSError, ValueError):
            return {"enabled": False, "apps": [], "wait_seconds": 20.0}
        if not isinstance(value, dict):
            return {"enabled": False, "apps": [], "wait_seconds": 20.0}
        apps = value.get("apps")
        wait = value.get("wait_seconds", 20.0)
        return {"enabled": value.get("enabled") is True,        # only a literal true; "yes", 1 and "true" do not count
                "apps": [a for a in apps if isinstance(a, str)] if isinstance(apps, list) else [],
                "wait_seconds": float(wait) if isinstance(wait, (int, float)) and not isinstance(wait, bool) and 0 < wait <= 120 else 20.0}

    @property
    def enabled(self) -> bool:
        cfg = self.config()
        return bool(cfg["enabled"] and cfg["apps"])

    def _running(self, name: str) -> bool:
        try:
            return self._run(["pgrep", "-x", name], 10).returncode == 0
        except Exception:
            return False

    def close_allowed(self) -> list:
        """Ask each allowed, running, unprotected application to quit; wait; report. Returns one row per listed app."""
        cfg = self.config()
        if not (cfg["enabled"] and cfg["apps"]):
            return []
        results = []
        for name in cfg["apps"]:
            row = {"app": name, "requested": False, "closed": False, "reason": ""}
            results.append(row)
            if not _SAFE_NAME.match(name):
                row["reason"] = "name not allowed"
                continue
            if name.lower() in PROTECTED:
                row["reason"] = "protected application, never closed"
                continue
            if not self._running(name):
                row["reason"] = "not running"
                row["closed"] = True
                continue
            try:
                self._run(["osascript", "-e", f'tell application "{name}" to quit'], 20)
                row["requested"] = True
            except Exception as error:
                row["reason"] = f"quit request failed: {type(error).__name__}"
                continue
            end = self._clock() + cfg["wait_seconds"]
            while self._clock() < end and self._running(name):
                self._sleep(0.5)
            row["closed"] = not self._running(name)
            if not row["closed"]:
                row["reason"] = "still running (it may be asking to save); not forced"
        return results
