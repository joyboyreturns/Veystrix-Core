"""Conservative request-boundary gate for the experimental layer-major runtime.

This is a guard, not a guarantee against other processes allocating concurrently.
It counts inactive anonymous pages as unavailable, takes a fresh strict sample
after optional recovery, and never assumes that a purge will succeed.
"""
from __future__ import annotations

import os
import re
import subprocess
import time

from . import memory_controller, purge_recovery

GIB = 2 ** 30
# ~27 KiB/token is the documented bf16 KV/index state for 12 QSA layers.
# Two copies plus metadata/rounding is budgeted; this is an estimate, not a
# measured upper bound for all allocator or serialization implementations.
TOKEN_RESERVE_BYTES = 64 * 1024
# A snapshot retains ~113 MB of recurrent state; round up for metadata.
SNAPSHOT_RESERVE_BYTES = 128 * 2 ** 20


class RequestMemoryRefused(RuntimeError):
    """Retriable admission failure; no inference allocations have started."""


def strict_read():
    def run(*args):
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout

    vm = run("vm_stat")
    stats = run("sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level")
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Anonymous pages"):
        if not re.search(rf"(?m)^\s*{re.escape(label)}\s*:\s*\d+\.", vm):
            raise ValueError(f"missing {label}")
    if not re.search(r"page size of \d+ bytes", vm):
        raise ValueError("missing page size")
    swap = re.search(r"used = ([\d.]+)M", stats)
    pressure = re.search(r"kern\.memorystatus_vm_pressure_level:\s*(\d+)", stats)
    if not swap or not pressure:
        raise ValueError("missing swap or pressure")
    return purge_recovery.parse_vm_stat(vm, int(pressure.group(1)), int(float(swap.group(1)) * 2**20))


def enabled(environ=None):
    env = os.environ if environ is None else environ
    if env.get("VEYSTRIX_MODE") == "chat":
        return False
    return (env.get("FLASHNEXT_PREFILL_ORDER", "").strip().lower() == "layer"
            and env.get("FLASHNEXT_REQUEST_ADMISSION", "1") != "0")


def check(tokens: int, *, max_output_tokens: int = 0, snapshot_count: int = 0, read=None, environ=None):
    env = os.environ if environ is None else environ
    if not enabled(env):
        return  # explicit rollback, not a validated safe configuration
    from . import progress
    if not progress.memory_watch_ready(env):
        raise RequestMemoryRefused("independent memory watchdog is unavailable or stale; retry when monitoring resumes")
    try:
        reading = (read or strict_read)()
        policy = memory_controller.Policy.from_environment(env)
    except Exception as exc:
        raise RequestMemoryRefused(f"memory admission unavailable: {type(exc).__name__}: {exc}; retry when probes work") from exc
    if policy.swap_ceiling_mib <= 0:
        raise RequestMemoryRefused("memory admission requires an absolute swap guard; retry with the guard enabled")
    controller = memory_controller.current()
    if controller is None or not controller.can_admit():
        raise RequestMemoryRefused("memory admission requires a running safety guard; restart the server with the guard enabled")
    if controller.policy.swap_ceiling_mib <= 0:
        raise RequestMemoryRefused("the running safety guard has no absolute swap limit; restart with the guard enabled")
    policy = controller.policy
    last_sample = controller.last.get("mono_t")
    if last_sample is None or time.monotonic() - last_sample > 3.0:
        raise RequestMemoryRefused("memory guard sample is stale; retry when monitoring resumes")
    if reading.swap_used >= (policy.swap_ceiling_mib - policy.swap_ceiling_margin_mib) * 2**20:
        raise RequestMemoryRefused("swap is already at the safety margin; retry after system memory pressure falls")
    # Owner decision 2026-11-10: the only limit is swap in use at the absolute ceiling. Available memory, pressure and
    # per-request headroom estimates are no longer admission conditions (they refused runs the swap rule would allow);
    # they remain telemetry. ``tokens``, ``max_output_tokens`` and ``snapshot_count`` are accepted for compatibility.
