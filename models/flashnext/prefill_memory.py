"""Opt-in cap on MLX's allocator cache during one prefill (off by default).

MLX keeps freed buffers in a cache and reuses them. Its default cache limit is
the whole memory limit (16.32 GB on a 16 GB M4), so a long prefill can hold
gigabytes of freed buffers. They are not "active" memory, so
``mx.get_active_memory`` and ``mx.get_peak_memory`` do not show them, but they
are resident and count toward Metal's recommended working set.

``FLASHNEXT_PREFILL_CACHE_LIMIT_MB`` (megabytes, 0 or unset = off) caps the
cache for the duration of one prefill and restores the previous limit on every
exit path. MLX enforces the limit lazily, at the next allocation: lowering the
limit left 128 MB cached until the next allocation trimmed it to 4 MB. Entering
the cap therefore trims an oversized cache at once, so the footprint is bounded
from the first layer.

Nothing here changes any value computed by the model. The cost of a small cap
is more fresh allocations; it has not been measured on the real model.
"""
from __future__ import annotations

import contextlib
import os

FLAG_CACHE_MB = "FLASHNEXT_PREFILL_CACHE_LIMIT_MB"
MIB = 1024 * 1024


def cache_limit_mb() -> int:
    """Requested cache cap in megabytes (0 = off). Bad values raise."""
    raw = os.environ.get(FLAG_CACHE_MB, "0") or "0"
    try:
        value = int(float(raw))
    except ValueError as error:
        raise ValueError(f"{FLAG_CACHE_MB} must be a number of megabytes") from error
    if value < 0:
        raise ValueError(f"{FLAG_CACHE_MB} must not be negative")
    return value


def enabled() -> bool:
    return cache_limit_mb() > 0


def _record_cap(event, mx, mb):
    try:
        from models.flashnext import flight_recorder
        flight_recorder.record(event, limit_mib=mb, cache_bytes=int(mx.get_cache_memory()),
                               active_bytes=int(mx.get_active_memory()))
    except Exception:
        pass  # telemetry must not alter inference or prevent restoring the limit


@contextlib.contextmanager
def cache_cap(megabytes=None):
    """Cap MLX's allocator cache inside the block. Yields the previous limit,
    or None when the cap is off. The previous limit is restored on exit."""
    mb = cache_limit_mb() if megabytes is None else int(megabytes)
    if mb <= 0:
        yield None
        return
    import mlx.core as mx

    limit = mb * MIB
    previous = mx.set_cache_limit(limit)
    try:
        # Do not relax a tighter process-wide allocator cap set by an owner.
        effective = min(previous, limit)
        if effective != limit:
            mx.set_cache_limit(effective)
        if mx.get_cache_memory() > effective:
            mx.clear_cache()            # the limit alone trims only at the next allocation
        _record_cap("prefill_allocator_cap_begin", mx, effective / MIB)
        yield previous
    finally:
        _record_cap("prefill_allocator_cap_end", mx, min(previous, limit) / MIB)
        mx.set_cache_limit(previous)
