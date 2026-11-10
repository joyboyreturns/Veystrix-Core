"""Opt-in decode profile: per-token wall time, per-layer routes and the existing timers, written once per request.

Off unless ``FLASHNEXT_DECODE_PROFILE_DIR`` names a directory. When on, ``begin()`` runs where the backend leaves
prefill for decode and ``finish()`` runs when the request ends. Between them a route-trace callback records, for every
streamed MoE call, the layer id, the routed expert ids and a timestamp, so one supervised server request yields:

* the current checkpoint's real decode routes (the cache-policy replays used an older checkpoint),
* per-token wall time and per-layer wall time from consecutive timestamps (host clock; GPU time is not separated here),
* the ``expert_cache`` timers (``FLASHNEXT_PROFILE_IO=1``), the host-window totals (``FLASHNEXT_HOST_WINDOW=1``) and the
  expert I/O counters, reset at ``begin()`` so they cover decode only.

It reads and records. It changes no result, no read, no schedule. ``begin`` and ``finish`` never raise.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

FLAG_DECODE_PROFILE_DIR = "FLASHNEXT_DECODE_PROFILE_DIR"

_LOCK = threading.Lock()
_STATE: dict = {"active": False, "events": [], "t0": 0.0, "previous": None, "last_layer": -1}


def directory() -> Path | None:
    value = os.environ.get(FLAG_DECODE_PROFILE_DIR)
    return Path(value) if value else None


def enabled() -> bool:
    return directory() is not None


def active() -> bool:
    return bool(_STATE["active"])


def _record(layer_id: int, routed) -> None:
    now = time.perf_counter()
    with _LOCK:
        if _STATE["active"]:
            _STATE["events"].append((now, int(layer_id), [int(e) for e in routed]))


def begin() -> bool:
    """Start recording decode. Returns True when a recording started."""
    if not enabled() or _STATE["active"]:
        return False
    try:
        from . import expert_cache, expert_io_stats, hostwindow

        expert_cache.reset_profile()
        hostwindow.reset()
        expert_io_stats.reset_stats()
        with _LOCK:
            _STATE["events"] = []
            _STATE["t0"] = time.perf_counter()
            _STATE["active"] = True
        expert_cache.set_route_trace(_record)
        return True
    except Exception:
        _STATE["active"] = False
        return False


def tokens_from_events(events) -> list[dict]:
    """Group (time, layer, routed) events into tokens: a token ends when the layer id stops increasing.

    Returns one dict per token with its start/end offsets, the per-layer arrival gaps and the routes by layer.
    """
    tokens: list[dict] = []
    current = None
    last_layer = -1
    for stamp, layer, routed in events:
        if current is None or layer <= last_layer:
            current = {"start": stamp, "stamps": [], "layers": [], "routes": {}}
            tokens.append(current)
        current["stamps"].append(stamp)
        current["layers"].append(layer)
        current["routes"][str(layer)] = routed
        last_layer = layer
    return tokens


def summarize(events, t0: float, end: float) -> dict:
    tokens = tokens_from_events(events)
    token_rows = []
    for index, token in enumerate(tokens):
        nxt = tokens[index + 1]["start"] if index + 1 < len(tokens) else end
        gaps = [b - a for a, b in zip(token["stamps"], token["stamps"][1:])]
        token_rows.append({
            "token": index,
            "ms": (nxt - token["start"]) * 1000.0,
            "layers": len(token["layers"]),
            "layer_ids": token["layers"],
            "layer_gap_ms": [round(g * 1000.0, 3) for g in gaps],
            "routes": token["routes"],
        })
    return {"tokens": token_rows, "elapsed_s": end - t0}


def finish(label: str = "decode") -> Path | None:
    """Stop recording and write ``<dir>/<label>-<unix ms>.json``. Never raises; returns the path or None."""
    if not _STATE["active"]:
        return None
    try:
        from . import expert_cache, expert_io_stats, hostwindow

        end = time.perf_counter()
        with _LOCK:
            events = list(_STATE["events"])
            t0 = _STATE["t0"]
            _STATE["active"] = False
            _STATE["events"] = []
        expert_cache.set_route_trace(None)
        body = summarize(events, t0, end)
        body["profile_io_enabled"] = bool(expert_cache.profile_enabled())
        body["timers"] = expert_cache.profile_totals()
        body["host_windows"] = hostwindow.totals()
        body["expert_io"] = expert_io_stats.get_stats().as_dict()
        target = directory()
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{label}-{int(time.time() * 1000)}.json"
        path.write_text(json.dumps(body))
        return path
    except Exception:
        _STATE["active"] = False
        try:
            from . import expert_cache

            expert_cache.set_route_trace(None)
        except Exception:
            pass
        return None
