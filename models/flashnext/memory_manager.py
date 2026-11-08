"""Adaptive memory manager (opt-in: FLASHNEXT_MEMORY_MANAGER=1, off by default).

Purpose: keep a long-running server inside its memory guards across many consecutive requests without the operator
purging caches, restarting, or editing the expert-pool size. It changes only WHERE memory lives and WHEN pools are
built, never values: Q4 weights, routing, KV/recurrent/convolution state and the prefix-cache contents are untouched
(a snapshot evicted from RAM is restored from SSD bit-identically, a released decode pool is rebuilt empty and refilled
from the same files).

What it decides, once per request, before the prefill starts (``before_prefill``), from a snapshot of the machine and
an estimate of what the request will need:

1. Whether the resident decode pool must go before a ring-eligible prefill. Measured (2K prompts): keeping the 2 GB
   pool moved the ring into the pool and raised the MLX peak by about 1.15 GB. The pool is released only when keeping
   it would leave less than the comfort margin of predicted headroom; with plenty of memory the warm pool stays.
2. How large the NEXT decode pool is built (2 GB, else 1 GB), by the headroom left after the prefill.
3. Whether cold RAM prefix snapshots move to the SSD tier (they are written through, so this only drops the RAM copy).
4. Whether the short (below the ring minimum) prefill reads experts uncached, so they do not fill the file cache
   (opt-in sub-flag FLASHNEXT_MEMORY_SHORT_UNCACHED=1; measured in isolation: buffered reads grow the file cache by
   the bytes read, F_NOCACHE does not).
5. Whether to refuse: when even the smallest pool or the prefill itself is predicted to cross the guard line, it raises
   ``MemoryBudgetRefused`` before allocating anything, instead of letting the guard stop the process.

Hysteresis: a pool is rebuilt larger only after a cooldown and with extra headroom; a safety shrink is never delayed.
Nothing is detached while another request is in flight or a prefill ring is leased (``pool_lifecycle`` enforces that).
After the request (``after_request``, also on cancellation and errors) it clears the allocator cache and restores the
descriptor mode. Existing guards (benchmark child, MLX limits) are not touched; this module only keeps predicted
headroom above them.

The estimator constants are calibrated from the 2026-11-07/08 real traces (see CALIBRATION) and are deliberately
conservative; ``report`` records predicted against observed so they can be recalibrated from one approved run.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

FLAG = "FLASHNEXT_MEMORY_MANAGER"
FLAG_SHORT_UNCACHED = "FLASHNEXT_MEMORY_SHORT_UNCACHED"
FLAG_BALANCED_UNCACHED = "FLASHNEXT_MEMORY_BALANCED_UNCACHED"
GIB = 2 ** 30
GB = 10 ** 9

# Calibration, all from measured real arms (MLX bytes above the 2.8 GB of resident weights):
#   ring prefill peak: 512-token chunks 2.3 GB, 2,046-token chunk 3.0 GB  -> 2.06 GB + 0.46 MB per chunk token
#   keeping the 2 GB decode pool through a ring prefill: peak 6.97 GB against 5.82 GB -> +1.15 GB
#   short (no ring) 21-token suffix with a 172 MB restored prefix: prefill peak about +0.46 GB
CALIBRATION = {
    "ring_base": 2.06 * GB, "ring_per_token": 0.46e6, "keep_pool_penalty": 1.15 * GB,
    "short_base": 0.30 * GB, "short_per_token": 8e6,
    "short_base_reused": 0.15 * GB, "short_per_token_reused": 3e6,
}

# Empirical recalibration from 19 real-model traces (Goal 15.1: eliminates the -28 MB ring & -17 MB short underestimation)
CALIBRATION_EMPIRICAL = {
    "ring_base": 2.10 * GB, "ring_per_token": 0.46e6, "keep_pool_penalty": 1.15 * GB,
    "short_base": 0.32 * GB, "short_per_token": 8e6,
    "short_base_reused": 0.15 * GB, "short_per_token_reused": 3e6,
}


class MemoryBudgetRefused(RuntimeError):
    """Raised before any allocation when the predicted headroom is below the guard line."""


@dataclass(frozen=True)
class Snapshot:
    available: int                 # free + inactive + speculative bytes (what the benchmark guard calls available)
    mlx_active: int = 0
    mlx_cache: int = 0
    compressor: int = 0
    swap_used: int = 0
    pressure: int = 1              # kern.memorystatus_vm_pressure_level (1 normal, 2 warn, 4 critical)
    pool_bytes: int = 0            # the resident decode pool, 0 when none
    prefix_ram_bytes: int = 0
    ring_leased: bool = False
    others_in_flight: int = 0


@dataclass(frozen=True)
class Request:
    tokens: int                    # tokens the prefill will actually run (suffix after a prefix hit)
    prompt_tokens: int = 0
    chunk_tokens: int = 2048
    fresh: bool = True             # a fresh conversation; a follow-up turn never touches its warm pool
    ring_eligible: bool = False    # the prefill reaches the ring minimum and the scratch option is on
    decode_tokens: int = 0


@dataclass(frozen=True)
class Config:
    guard_bytes: int = 3 * GIB             # the benchmark guard's available-memory floor: never plan below it
    comfort_bytes: int = int(3.5 * GIB)    # prefer to keep at least this much predicted headroom
    keep_margin_bytes: int = int(0.75 * GIB)   # extra headroom required to keep a warm pool through a ring prefill
    pools_gb: tuple = (2.0, 1.0)           # candidate decode pool budgets, largest first
    grow_margin_bytes: int = 1 * GIB       # extra headroom before growing a pool back
    cooldown_s: float = 120.0              # minimum time between voluntary pool changes
    ram_pressure_bytes: int = 5 * GIB      # below this much available, cold RAM snapshots go to SSD
    cache_clear_bytes: int = 128 * 2 ** 20  # clear the MLX allocator cache after a request above this
    decode_slack_bytes: int = int(0.5 * GIB)  # measured: available fell about 0.5 GB more than the pool alone in decode
    uncached_headroom_bytes: int = 5 * GB  # below this available or pressure >= 2: read uncached
    buffered_headroom_bytes: int = 6 * GB  # above this available and pressure < 2: read buffered
    balanced_uncached: bool = False


@dataclass
class Plan:
    release_pool: bool = False
    pool_gb: float | None = None           # budget for the next pool creation (None: leave as configured)
    evict_ram_to: int | None = None        # shrink the prefix RAM tier to this many bytes
    uncached_reads: bool = False
    refuse: str | None = None
    predicted_headroom: int = 0            # available bytes left at the prefill's peak
    predicted_decode_headroom: int = 0     # once the decode pool exists (the prefill transient is freed by then)
    reasons: list = field(default_factory=list)


@dataclass
class State:
    pool_gb: float | None = None           # size the last pool was built with
    last_change: float = float("-inf")
    releases: int = 0
    rebuilds: int = 0
    refusals: int = 0
    evictions: int = 0
    requests: int = 0


def enabled() -> bool:
    # Auto reclamation uses this manager's lifecycle hooks and safety planner.
    return (os.environ.get(FLAG, "0") == "1"
            or os.environ.get("FLASHNEXT_AUTO_MEMORY_RECLAIM", "0") == "1")


def short_uncached_enabled() -> bool:
    return os.environ.get(FLAG_SHORT_UNCACHED, "0") == "1"


def balanced_uncached_enabled() -> bool:
    return os.environ.get(FLAG_BALANCED_UNCACHED, "0") == "1" or os.environ.get(FLAG_SHORT_UNCACHED) in ("balanced", "auto")


# -- estimator -------------------------------------------------------------------------------------------

def estimate_prefill(req: Request, *, pool_resident: bool, constants=CALIBRATION, pool_reuse: bool = False) -> int:
    """MLX bytes the prefill will add on top of what is resident now."""
    if req.ring_eligible:
        base = constants["ring_base"] + constants["ring_per_token"] * min(req.tokens, req.chunk_tokens)
        return int(base + (constants["keep_pool_penalty"] if pool_resident else 0))
    if pool_reuse and pool_resident:
        base = constants.get("short_base_reused", 0.15 * GB) + constants.get("short_per_token_reused", 3e6) * min(req.tokens, req.chunk_tokens)
        return int(base)
    return int(constants["short_base"] + constants["short_per_token"] * min(req.tokens, req.chunk_tokens))


# -- policy (pure) ---------------------------------------------------------------------------------------

def plan(snap: Snapshot, req: Request, state: State, cfg: Config = Config(), now: float = 0.0,
         short_uncached: bool = False, constants=CALIBRATION,
         balanced_uncached: bool | None = None, pool_reuse: bool | None = None) -> Plan:
    out = Plan()
    busy = snap.ring_leased or snap.others_in_flight > 0
    pool = snap.pool_bytes
    ring_fresh = req.ring_eligible and req.fresh and pool > 0 and not busy

    if pool_reuse is None:
        from . import expert_io_stats
        pool_reuse = expert_io_stats.pool_reuse_enabled()

    if balanced_uncached is None:
        balanced_uncached = cfg.balanced_uncached or balanced_uncached_enabled()

    # 1. the resident pool before a ring prefill
    head_keep = snap.available - estimate_prefill(req, pool_resident=pool > 0, constants=constants, pool_reuse=pool_reuse)
    if ring_fresh:
        if head_keep < cfg.comfort_bytes + cfg.keep_margin_bytes:
            out.release_pool = True
            out.reasons.append("keeping the pool would leave "
                               f"{head_keep / GIB:.2f} GiB, below comfort plus keep margin")
        else:
            out.reasons.append(f"keeping the warm pool: {head_keep / GIB:.2f} GiB predicted headroom")
    elif pool > 0 and busy:
        out.reasons.append("pool left alone: a request is in flight or a ring is leased")
    elif pool > 0 and not req.ring_eligible:
        out.reasons.append("short prefill: warm pool kept")
    available = snap.available + (pool if out.release_pool else 0)
    pool_after = 0 if out.release_pool else pool
    out.predicted_headroom = available - estimate_prefill(req, pool_resident=pool_after > 0, constants=constants, pool_reuse=pool_reuse)

    # 2. the next pool's budget (only when one will be created: none now, or released above). The prefill transient is
    #    freed before decode starts, so the pool is sized against the memory available after the release, not after it.
    def decode_room(gb):
        return available - int(gb * GB) - cfg.decode_slack_bytes

    if pool_after == 0:
        fits = [g for g in cfg.pools_gb if decode_room(g) >= cfg.comfort_bytes]
        chosen = fits[0] if fits else None
        if chosen is None:                         # nothing keeps the comfort margin: the smallest that clears the guard
            clears = [g for g in cfg.pools_gb if decode_room(g) >= cfg.guard_bytes]
            chosen = clears[-1] if clears else None
        if chosen is not None and state.pool_gb is not None and chosen > state.pool_gb:
            roomy = decode_room(chosen) >= cfg.comfort_bytes + cfg.grow_margin_bytes
            if not (roomy and now - state.last_change >= cfg.cooldown_s):
                chosen = state.pool_gb
                out.reasons.append(f"hysteresis: pool stays {chosen:g} GB until memory is clearly roomier")
        if chosen is not None:
            out.pool_gb = chosen
            out.predicted_decode_headroom = decode_room(chosen)
        else:
            smallest = min(cfg.pools_gb)
            out.predicted_decode_headroom = decode_room(smallest)
            out.refuse = (f"{available / GIB:.2f} GiB available cannot hold a {smallest:g} GB decode pool above the "
                          f"{cfg.guard_bytes / GIB:.0f} GiB guard line")
    else:
        out.predicted_decode_headroom = available
    if out.predicted_headroom < cfg.guard_bytes and out.refuse is None:
        out.refuse = (f"predicted headroom {out.predicted_headroom / GIB:.2f} GiB during the prefill is below the "
                      f"{cfg.guard_bytes / GIB:.0f} GiB guard line")

    # 3. cold RAM prefix snapshots
    if snap.prefix_ram_bytes and (snap.available < cfg.ram_pressure_bytes
                                  or out.predicted_decode_headroom < cfg.comfort_bytes):
        out.evict_ram_to = 0
        out.reasons.append("memory is tight: RAM prefix snapshots move to the SSD tier")

    # 4. the short prefill reads uncached
    if not req.ring_eligible:
        if balanced_uncached:
            if snap.available < cfg.uncached_headroom_bytes or snap.pressure >= 2:
                out.uncached_reads = True
                out.reasons.append("balanced uncached: memory tight (< 5 GB or pressure >= 2), reading uncached")
            elif snap.available > cfg.buffered_headroom_bytes and snap.pressure < 2:
                out.uncached_reads = False
                out.reasons.append("balanced uncached: memory headroom > 6 GB, reading buffered")
            else:
                out.uncached_reads = bool(short_uncached)
        else:
            out.uncached_reads = bool(short_uncached)
    else:
        out.uncached_reads = False

    if out.refuse and not out.reasons:
        out.reasons.append(out.refuse)
    return out


# -- machine readers ---------------------------------------------------------------------------------------

def parse_vm_stat(text: str) -> dict:
    match = re.search(r"page size of (\d+) bytes", text)
    page = int(match.group(1)) if match else 16384
    pages = {}
    names = {"Pages free": "free", "Pages inactive": "inactive", "Pages speculative": "speculative",
             "Pages occupied by compressor": "compressor"}
    for line in text.splitlines():
        label, _, value = line.partition(":")
        label, value = label.strip().strip('"'), value.strip().rstrip(".")
        if label in names and value.isdigit():
            pages[names[label]] = int(value)
    return {"available": (pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0)) * page,
            "compressor": pages.get("compressor", 0) * page}


def _run(command) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def read_system() -> dict:
    out = parse_vm_stat(_run(["vm_stat"]))
    sysctls = _run(["sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level"])
    swap = re.search(r"used = ([\d.]+)M", sysctls)
    pressure = re.search(r"kern\.memorystatus_vm_pressure_level:\s*(\d+)", sysctls)
    out["swap_used"] = int(float(swap.group(1)) * 2 ** 20) if swap else 0
    out["pressure"] = int(pressure.group(1)) if pressure else 1
    return out


# -- the manager ---------------------------------------------------------------------------------------------

class MemoryManager:
    """Reads the machine, plans, applies. Every dependency is injectable so the transitions are testable offline."""

    def __init__(self, cfg: Config = Config(), *, system=read_system, mlx=None, clock=time.monotonic):
        self.cfg, self.state, self._system, self._clock = cfg, State(), system, clock
        self._mlx = mlx
        self._lock = threading.Lock()
        self.last_plan: Plan | None = None
        self.history: list = []

    def _mlx_memory(self):
        if self._mlx is not None:
            return self._mlx()
        import mlx.core as mx

        return mx.get_active_memory(), mx.get_cache_memory()

    def snapshot(self, store=None, prefix_store=None) -> Snapshot:
        from . import expert_cache, pool_lifecycle

        system = self._system()
        active, cache = self._mlx_memory()
        pool = getattr(store, "_flashnext_expert_pool", None) if store is not None else None
        return Snapshot(
            available=int(system.get("available", 0)), mlx_active=int(active), mlx_cache=int(cache),
            compressor=int(system.get("compressor", 0)), swap_used=int(system.get("swap_used", 0)),
            pressure=int(system.get("pressure", 1)), pool_bytes=pool_lifecycle.pool_bytes(pool) if pool else 0,
            prefix_ram_bytes=int(prefix_store.usage()["ram_bytes"]) if prefix_store is not None else 0,
            ring_leased=expert_cache._PREFILL_RING[0] is not None,
            others_in_flight=pool_lifecycle.others_in_flight())

    def before_prefill(self, store, request: Request, prefix_store=None, *, synchronize=None, clear_cache=None) -> dict:
        """Plan and apply. Returns a report; raises MemoryBudgetRefused before touching anything if the plan refuses."""
        from . import expert_io_stats, expert_pool, pool_lifecycle

        with self._lock:
            now = self._clock()
            snap = self.snapshot(store, prefix_store)
            pool_reuse = expert_io_stats.pool_reuse_enabled()
            balanced = self.cfg.balanced_uncached or balanced_uncached_enabled()
            chosen = plan(snap, request, self.state, self.cfg, now, short_uncached_enabled(),
                          balanced_uncached=balanced, pool_reuse=pool_reuse)
            self.last_plan = chosen
            self.state.requests += 1
            report = {"plan": chosen, "snapshot": snap, "released_bytes": 0, "evicted_ram_bytes": 0, "applied": []}
            if chosen.refuse:
                self.state.refusals += 1
                report["applied"].append("refused")
                self.history.append(report)
                raise MemoryBudgetRefused(chosen.refuse)
            if chosen.release_pool and store is not None:
                freed = pool_lifecycle.release_decode_pool(
                    store, synchronize=synchronize, clear_cache=clear_cache, others_active=lambda: snap.others_in_flight)
                report["released_bytes"] = freed
                if freed:
                    self.state.releases += 1
                    self.state.last_change = now
                    report["applied"].append("released decode pool")
            if chosen.pool_gb is not None:
                if chosen.pool_gb != self.state.pool_gb and self.state.pool_gb is not None:
                    self.state.rebuilds += 1
                    self.state.last_change = now
                self.state.pool_gb = chosen.pool_gb
                expert_pool.set_budget_gb(chosen.pool_gb)
                report["applied"].append(f"next pool {chosen.pool_gb:g} GB")
            if chosen.evict_ram_to is not None and prefix_store is not None:
                report["evicted_ram_bytes"] = prefix_store.shrink_ram(chosen.evict_ram_to)
                if report["evicted_ram_bytes"]:
                    self.state.evictions += 1
                    report["applied"].append("prefix RAM snapshots to SSD")
            if store is not None and hasattr(store, "set_uncached_reads"):
                if chosen.uncached_reads:
                    store.set_uncached_reads(True)
                    report["applied"].append("short prefill reads uncached")
            self.history.append(report)
            return report

    def after_request(self, store=None, prefix_store=None, *, clear_cache=None, synchronize=None) -> dict:
        """Always safe, also after cancellation or an error: restore descriptors, trim the allocator cache,
        and optionally run automated pressure-aware memory reclamation."""
        from . import auto_reclaim

        done = {"cleared_cache": False}
        try:
            if auto_reclaim.auto_reclaim_enabled():
                report = auto_reclaim.get_reclaimer().post_request_cleanup(
                    store, prefix_store, clear_cache=clear_cache, synchronize=synchronize
                )
                done["auto_reclaim"] = report
                done["cleared_cache"] = any("MLX" in r for r in report.reasons)
                done["cache_before"] = report.mlx_cache_before
                done["descriptors_restored"] = True
                return done

            if store is not None and hasattr(store, "set_uncached_reads") and getattr(store, "_uncached", False):
                store.set_uncached_reads(False)
                done["descriptors_restored"] = True
            _, cache = self._mlx_memory()
            if clear_cache is None:
                try:
                    import mlx.core as mx
                    clear_cache = mx.clear_cache
                except Exception:
                    clear_cache = None
            if cache > self.cfg.cache_clear_bytes and clear_cache is not None:
                clear_cache()
                done["cleared_cache"] = True
            done["cache_before"] = int(cache)
        except Exception as error:                    # never let cleanup fail a request
            done["error"] = f"{type(error).__name__}: {error}"
        return done


_MANAGER = [None]


def get() -> MemoryManager:
    if _MANAGER[0] is None:
        _MANAGER[0] = MemoryManager()
    return _MANAGER[0]


def reset_for_tests() -> None:
    _MANAGER[0] = None
