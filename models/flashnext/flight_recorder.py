"""Persistent request-lifecycle log for the server (the "flight recorder").

Why it exists: on 2026-11-09 a DeepSeek Harness request stalled a server that kept no log of its own, so the last completed
stage of the request could not be recovered afterwards. This module writes one JSON object per line, flushed at once, so
the stage a request reached survives the process (including an ``os._exit`` by the safety guard).

* The file is private (0600) in a private directory (0700): ``~/.flashnext/logs/server-<pid>.jsonl`` by default,
  ``FLASHNEXT_FLIGHT_LOG`` overrides the path, ``FLASHNEXT_FLIGHT_LOG=0`` turns it off.
* It never stores prompt or completion text. Request records carry sizes, hashes of nothing, stage names, counters,
  timings and the cache outcome.
* One writer lock, bounded size (rotates to ``.1`` at ``FLASHNEXT_FLIGHT_LOG_MAX_MB``, default 64), and every write is
  best effort: a failing disk must never fail a request.
"""
from __future__ import annotations

import json
import os
import threading
import time

FLAG = "FLASHNEXT_FLIGHT_LOG"
FLAG_MAX_MB = "FLASHNEXT_FLIGHT_LOG_MAX_MB"
DEFAULT_DIR = "~/.flashnext/logs"


class FlightRecorder:
    def __init__(self, path: str | None, max_bytes: int = 64 << 20, clock=time.time):
        self.path = path
        self.max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._handle = None
        self._size = 0
        self.failures = 0
        self.events = 0

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _open(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, mode=0o700, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(fd, 0o600)
        self._handle = os.fdopen(fd, "a", buffering=1, encoding="utf-8")
        self._size = os.fstat(fd).st_size

    def _rotate(self):
        self._handle.close()
        self._handle = None
        os.replace(self.path, self.path + ".1")
        self._open()

    def record(self, event: str, **fields) -> None:
        if self.path is None:
            return
        row = {"t": round(self._clock(), 3), "event": event, "pid": os.getpid(), **fields}
        try:
            line = json.dumps(row, default=str, separators=(",", ":")) + "\n"
        except (TypeError, ValueError):
            line = json.dumps({"t": row["t"], "event": event, "error": "unserializable fields"}) + "\n"
        with self._lock:
            try:
                if self._handle is None:
                    self._open()
                if self._size + len(line) > self.max_bytes:
                    self._rotate()
                self._handle.write(line)
                self._handle.flush()
                self._size += len(line)
                self.events += 1
            except OSError:
                self.failures += 1

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                finally:
                    self._handle = None


_NULL = FlightRecorder(None)
_ACTIVE: list = [_NULL]


def resolve_path(environ=None, pid: int | None = None) -> str | None:
    environ = os.environ if environ is None else environ
    configured = environ.get(FLAG)
    if configured is not None and configured.strip().lower() in ("0", "off", "false", "no"):
        return None
    if configured and configured.strip() not in ("1", "on", "true", "yes"):
        return os.path.expanduser(configured)
    return os.path.join(os.path.expanduser(DEFAULT_DIR), f"server-{pid or os.getpid()}.jsonl")


def install(environ=None) -> FlightRecorder:
    """Create the process recorder (called once by the server). Returns it."""
    environ = os.environ if environ is None else environ
    try:
        max_mb = float(environ.get(FLAG_MAX_MB, "64"))
    except ValueError:
        max_mb = 64.0
    recorder = FlightRecorder(resolve_path(environ), int(max(1.0, max_mb) * (1 << 20)))
    _ACTIVE[0] = recorder
    return recorder


def uninstall() -> None:
    old, _ACTIVE[0] = _ACTIVE[0], _NULL
    old.close()


def current() -> FlightRecorder:
    return _ACTIVE[0]


def record(event: str, **fields) -> None:
    _ACTIVE[0].record(event, **fields)
