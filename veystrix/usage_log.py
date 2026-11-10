"""Local usage statistics: one JSON line per finished request, written by the server and read by the desktop app.

Records hold counts and timings only: model, token counts, seconds. No prompt, no output text, no client identity.
The file lives in ``~/Library/Application Support/Veystrix/usage/requests.jsonl`` (``VEYSTRIX_USAGE_LOG`` overrides the
path, ``VEYSTRIX_USAGE_LOG=0`` turns recording off). Writing is best effort: a failure here never affects a request.
Statistics are lifetime: nothing is ever deleted or overwritten. At 64 MB the file is renamed to
``requests.<unix-time>.jsonl`` (an archive the app still reads) and a new one starts.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ENV = "VEYSTRIX_USAGE_LOG"
MAX_BYTES = 64 * 1024 * 1024
_lock = threading.Lock()
_local = threading.local()


def default_path() -> Path:
    return Path.home() / "Library" / "Application Support" / "Veystrix" / "usage" / "requests.jsonl"


def path() -> Path | None:
    raw = os.environ.get(ENV)
    if raw == "0":
        return None
    return Path(raw).expanduser() if raw else default_path()


def set_warm(flag: bool) -> None:
    """Mark the current request thread's next record as an automatic warm-up (kept apart from real use)."""
    _local.warm = bool(flag)


def _warm() -> bool:
    return bool(getattr(_local, "warm", False))


def build_record(model: str, counts: dict, stats, wall_seconds: float, *, now: float | None = None,
                 warm: bool | None = None) -> dict:
    prompt, cached, completion = int(counts["prompt"]), int(counts["cached"]), int(counts["completion"])
    prefill = float(getattr(stats, "prefill_seconds", 0.0) or 0.0)
    decode = float(getattr(stats, "seconds", 0.0) or 0.0)
    return {"ts": round(time.time() if now is None else now, 3), "model": str(model or "unknown"),
            "prompt": prompt, "cached": cached, "uncached": max(0, prompt - cached), "completion": completion,
            "total": prompt + completion, "wall_s": round(float(wall_seconds), 3), "prefill_s": round(prefill, 3),
            "decode_s": round(decode, 3), "finish": str(getattr(stats, "finish", "") or ""),
            "warm": _warm() if warm is None else bool(warm)}


def append(record: dict, target: Path | None = None) -> bool:
    """Append one record. Returns False (and never raises) when recording is off or the write failed."""
    try:
        target = target or path()
        if target is None:
            return False
        line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                if target.stat().st_size > MAX_BYTES:
                    archive = target.with_name(f"{target.stem}.{int(time.time())}{target.suffix}")
                    n = 0
                    while archive.exists():          # never overwrite an earlier archive
                        n += 1
                        archive = target.with_name(f"{target.stem}.{int(time.time())}-{n}{target.suffix}")
                    os.rename(target, archive)
            except FileNotFoundError:
                pass
            fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        return True
    except Exception:
        return False


def record_request(service, stats, wall_seconds: float, counts_fn) -> None:
    """Called once per finished request by the server. ``counts_fn(service, stats)`` is ``usage_counts``."""
    try:
        if path() is None:
            return
        append(build_record(getattr(service, "model", ""), counts_fn(service, stats), stats, wall_seconds))
    except Exception:
        pass
