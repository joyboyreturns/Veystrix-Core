"""Staged memory controller: what the chat/server safety guard does about macOS memory pressure.

Why it exists (Gate A attempt 2, 2026-11-08, ``results/flashnext/20261108-224316-gate-a-attempt2/``): the old rule stopped the
process on the first one-second sample that read macOS pressure level 2, with 3.4 GiB still available, zero swap-outs and a
healthy run behind it. Compression churn during prefill is normal for this engine on a 16 GB machine (tens of GiB
compressed and decompressed per cold prefill); one level-2 sample is not a failure. This controller keeps four stages:

``NORMAL``    nothing to do.
``ELEVATED``  pressure level >= 2, any swap activity, or a compressor surge while little memory is available. The engine
              records it, stops optional allocation, and at its own safe boundaries (between prefill chunks, between
              requests) lowers the next chunk budget and waits a bounded time for the machine to calm down.
``RECOVERY``  the cause cleared less than ``recover_s`` ago; still treated as elevated for budgets.
``CRITICAL``  the guard stops the process. Reasons, each independent of the others:

              * pressure level 4 (macOS critical) held for ``critical_confirm_s``;
              * pressure level >= 2 continuously (short dips below do not reset it) for ``persist_s``: unrecoverable;
              * total swap used grew by ``swap_growth_stop_gib`` (default 2.5 GiB, the owner's decision) over the
                session's baseline;
              * active swapping: ``swap_activity_stop_mib`` swapped out (or in) inside ``swap_window_s`` seconds;
              * a missing or unreadable probe, or the controller itself failing (the guard stops on any exception).

A swap INCREASE (the total on disk grew) is different from ACTIVE swap-out (pages being written now). A swap increase of
a few hundred MiB can be old, cold memory of other applications pushed out once; active swap-out and swap-in at a high
rate mean the working set no longer fits. Neither alone decides safety: the swap-growth stop is a hard limit, the activity
stop is a rate, and neither may override the independent stops that stay in ``ChatGuard`` (available-memory floor,
compressor growth with low availability, Metal limit, battery, Low Power Mode, thermal, the out-of-process watchdog).

The controller never frees anything itself and never touches a GPU buffer: it only reports a state. Consumers read it at
points where the engine is idle or between evaluated chunks (``checkpoint``, ``scale``, ``forbid_optional``).
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

NORMAL, ELEVATED, RECOVERY, CRITICAL = "normal", "elevated", "recovery", "critical"
GIB = 2 ** 30
MIB = 2 ** 20

from . import memory_policy

FLAG_SWAP_GROWTH = "FLASHNEXT_SWAP_GROWTH_STOP_GIB"
FLAG_PERSIST = "FLASHNEXT_PRESSURE_PERSIST_S"
FLAG_CRITICAL_CONFIRM = "FLASHNEXT_CRITICAL_CONFIRM_S"
FLAG_RECOVER = "FLASHNEXT_PRESSURE_RECOVER_S"
FLAG_SWAP_WINDOW = "FLASHNEXT_SWAP_WINDOW_S"
FLAG_SWAP_ACTIVITY = "FLASHNEXT_SWAP_ACTIVITY_STOP_MIB"
FLAG_THROTTLE_WAIT = "FLASHNEXT_THROTTLE_MAX_WAIT_S"
FLAG_THROTTLE_REQUEST = "FLASHNEXT_THROTTLE_REQUEST_CAP_S"
FLAGS = (FLAG_SWAP_GROWTH, FLAG_PERSIST, FLAG_CRITICAL_CONFIRM, FLAG_RECOVER, FLAG_SWAP_WINDOW, FLAG_SWAP_ACTIVITY,
         FLAG_THROTTLE_WAIT, FLAG_THROTTLE_REQUEST)


def _number(name: str, default: float, low: float, high: float, environ=None) -> float:
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number") from None
    if not (low <= value <= high) or value != value:
        raise ValueError(f"{name}={raw!r} is outside {low}..{high}")
    return value


@dataclass(frozen=True)
class Policy:
    swap_growth_stop_gib: float = 2.5     # owner's decision: total swap grown past this over the session baseline stops
    persist_s: float = 45.0               # pressure level >= 2 for this long without a calm spell is unrecoverable
    critical_confirm_s: float = 2.0       # macOS level 4 held this long stops
    recover_s: float = 5.0                # calm for this long ends an episode
    swap_window_s: float = 10.0
    swap_activity_stop_mib: float = 512.0  # swapped out (or in) inside the window
    swap_elevated_growth_mib: float = 64.0  # swap increase above this (the old hard stop, now a signal)
    swap_activity_elevated_mib: float = 8.0  # swap-out/in inside the window must reach this to mark ELEVATED (the real 8K run
                                             # raised a false episode on 1.4 MiB of swap-in)
    surge_mib: float = 1536.0             # compressor growth inside the window ...
    surge_available_mib: float = 4096.0   # ... only counts while less than this is available
    throttle_max_wait_s: float = 20.0     # longest single wait at a safe boundary
    throttle_request_cap_s: float = 120.0  # total waiting per request
    elevated_scale: float = 0.5           # factor on future chunk / scratch budgets while elevated
    swap_only: bool = False               # approved agentic policy: only the swap-growth hard stop (and missing probe) can stop

    def telemetry_only_checks(self) -> list[str]:
        return list(memory_policy.TELEMETRY_ONLY_CHECKS) if self.swap_only else []

    @classmethod
    def from_environment(cls, environ=None, swap_only: bool | None = None) -> "Policy":
        d = cls()
        effective_swap_only = (
            memory_policy.is_swap_growth_policy(environ)
            if swap_only is None
            else bool(swap_only)
        )
        return cls(
            swap_only=effective_swap_only,
            swap_growth_stop_gib=_number(FLAG_SWAP_GROWTH, d.swap_growth_stop_gib, 0.25, 64, environ),
            persist_s=_number(FLAG_PERSIST, d.persist_s, 5, 3600, environ),
            critical_confirm_s=_number(FLAG_CRITICAL_CONFIRM, d.critical_confirm_s, 1, 60, environ),
            recover_s=_number(FLAG_RECOVER, d.recover_s, 1, 300, environ),
            swap_window_s=_number(FLAG_SWAP_WINDOW, d.swap_window_s, 2, 120, environ),
            swap_activity_stop_mib=_number(FLAG_SWAP_ACTIVITY, d.swap_activity_stop_mib, 16, 65536, environ),
            throttle_max_wait_s=_number(FLAG_THROTTLE_WAIT, d.throttle_max_wait_s, 0, 600, environ),
            throttle_request_cap_s=_number(FLAG_THROTTLE_REQUEST, d.throttle_request_cap_s, 0, 3600, environ),
        )


@dataclass
class Controller:
    policy: Policy = field(default_factory=Policy)
    page_bytes: int = 16384
    emit: object = None                   # callable(event: str, **fields) for the trace; optional

    def __post_init__(self):
        self._lock = threading.RLock()
        self.state = NORMAL
        self.admit_work = True
        self.baseline = None              # {"t", "swap_used_mb", "compressor_mb"}
        self.window = deque()             # (t, swapout_pages, swapin_pages, compressor_mb)
        self.pressure_since = None        # start of the current level >= 2 episode
        self.level4_since = None
        self.calm_since = None
        self.episode_start = None
        self.reasons: list[str] = []
        self.episodes: list[dict] = []
        self.recoveries = 0
        self.elevated_seconds = 0.0
        self._last_t = None
        self.peak = {"swap_increase_mib": 0.0, "swapout_window_mib": 0.0, "swapin_window_mib": 0.0,
                     "compressor_growth_mib": 0.0, "pressure_level": 1, "pressure_seconds": 0.0}
        self.last = {}
        self._probes: dict[str, bool] = {}
        self.throttled_s = 0.0
        self.request_throttled_s = 0.0
        self.stop_reason = None
        self.stopped_at = None            # time.time() of the CRITICAL transition (for the report and the grace period)
        self._calm_started = None         # when the elevated causes cleared (RECOVERY clock)

    # -- evaluation ----------------------------------------------------------------------------------------------

    def observe(self, now: float, *, pressure, swap_used_mb, swapout_pages, swapin_pages, available_mb,
                compressor_mb) -> str | None:
        """Feed one machine sample. Returns a stop reason (CRITICAL) or None. Raises nothing for missing numbers:
        a missing number is a stop reason, because a controller that cannot see is not allowed to keep going."""
        self._probes = {
            "pressure": pressure is not None,
            "swap_used_mb": swap_used_mb is not None,
            "swapout": swapout_pages is not None,
            "swapin": swapin_pages is not None,
            "available_mb": available_mb is not None,
            "compressor_mb": compressor_mb is not None,
        }
        missing = [k for k, ok in self._probes.items() if not ok]
        if missing:
            return self._stop(f"memory controller probe unavailable: {', '.join(missing)}")
        if not isinstance(swap_used_mb, (int, float)) or swap_used_mb < 0:
            return self._stop(f"invalid swap measurement: {swap_used_mb!r}")
        with self._lock:
            if self.baseline is None:
                self.baseline = {"t": now, "swap_used_mb": float(swap_used_mb), "compressor_mb": float(compressor_mb)}
            dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
            self._last_t = now
            self.window.append((now, int(swapout_pages), int(swapin_pages), float(compressor_mb), float(swap_used_mb)))
            while len(self.window) > 2 and now - self.window[0][0] > self.policy.swap_window_s:
                self.window.popleft()
            first = self.window[0]
            span = now - first[0]
            out_mib = max(0, swapout_pages - first[1]) * self.page_bytes / MIB
            in_mib = max(0, swapin_pages - first[2]) * self.page_bytes / MIB
            surge_mib = max(0.0, float(compressor_mb) - first[3])
            swap_recent = float(swap_used_mb) - first[4]       # swap growth inside the window (the signal)
            swap_increase = float(swap_used_mb) - self.baseline["swap_used_mb"]
            growth = float(compressor_mb) - self.baseline["compressor_mb"]
            level = int(pressure)
            p = self.policy

            # pressure episodes: level >= 2 continues through dips shorter than recover_s
            if level >= 2:
                if self.pressure_since is None:
                    self.pressure_since = now
                self.calm_since = None
            elif self.pressure_since is not None:
                self.calm_since = now if self.calm_since is None else self.calm_since
                if now - self.calm_since >= p.recover_s:
                    self.pressure_since = None
            pressure_seconds = 0.0 if self.pressure_since is None else now - self.pressure_since
            self.level4_since = (self.level4_since or now) if level >= 4 else None

            self.peak["swap_increase_mib"] = max(self.peak["swap_increase_mib"], swap_increase)
            self.peak["swapout_window_mib"] = max(self.peak["swapout_window_mib"], out_mib)
            self.peak["swapin_window_mib"] = max(self.peak["swapin_window_mib"], in_mib)
            self.peak["compressor_growth_mib"] = max(self.peak["compressor_growth_mib"], growth)
            self.peak["pressure_level"] = max(self.peak["pressure_level"], level)
            self.peak["pressure_seconds"] = max(self.peak["pressure_seconds"], pressure_seconds)
            self.last = {"mono_t": round(now, 2), "pressure": level, "available_mb": round(float(available_mb), 1),
                         "swap_increase_mib": round(swap_increase, 1), "swapout_window_mib": round(out_mib, 2),
                         "swapin_window_mib": round(in_mib, 2), "compressor_window_growth_mib": round(surge_mib, 1),
                         "compressor_growth_mib": round(growth, 1), "pressure_seconds": round(pressure_seconds, 1),
                         "window_s": round(span, 1)}

            # CRITICAL: each rule is independent
            # Under the owner's waiver (memory_waiver.py) only the swap-growth stop applies; the rest is measured, not enforced.
            if not p.swap_only:
                if self.level4_since is not None and now - self.level4_since >= p.critical_confirm_s:
                    return self._stop(f"memory pressure level {level} for {now - self.level4_since:.1f}s")
                if pressure_seconds >= p.persist_s:
                    return self._stop(f"memory pressure level >= 2 for {pressure_seconds:.0f}s without a calm spell "
                                      f"(limit {p.persist_s:.0f}s)")
            effective_swap = max(swap_increase, self.peak["swap_increase_mib"])
            if effective_swap >= p.swap_growth_stop_gib * 1024:
                return self._stop(f"swap grew {effective_swap / 1024:.2f} GiB over the session baseline "
                                  f"(hard limit {p.swap_growth_stop_gib:g} GiB)")
            if not p.swap_only:
                if out_mib >= p.swap_activity_stop_mib:
                    return self._stop(f"active swap-out: {out_mib:.0f} MiB in {span:.0f}s "
                                      f"(limit {p.swap_activity_stop_mib:.0f} MiB per {p.swap_window_s:.0f}s)")
                if in_mib >= p.swap_activity_stop_mib:
                    return self._stop(f"active swap-in: {in_mib:.0f} MiB in {span:.0f}s "
                                      f"(limit {p.swap_activity_stop_mib:.0f} MiB per {p.swap_window_s:.0f}s)")

            # ELEVATED / RECOVERY / NORMAL
            reasons = []
            if level >= 2:
                reasons.append(f"pressure level {level}")
            if out_mib >= p.swap_activity_elevated_mib or in_mib >= p.swap_activity_elevated_mib:
                reasons.append(f"swap activity (out {out_mib:.0f} MiB, in {in_mib:.0f} MiB in {span:.0f}s)")
            if swap_recent > p.swap_elevated_growth_mib:
                reasons.append(f"swap grew {swap_recent:.0f} MiB in {span:.0f}s ({swap_increase:.0f} MiB over the session baseline)")
            if surge_mib >= p.surge_mib and float(available_mb) < p.surge_available_mib:
                reasons.append(f"compressor +{surge_mib / 1024:.1f} GiB in {span:.0f}s with {float(available_mb) / 1024:.1f} GiB available")
            previous = self.state
            if reasons:
                self.state = ELEVATED
                if previous == NORMAL:
                    self.episode_start = now
                self.reasons = reasons
            elif previous in (ELEVATED, RECOVERY):
                calm = now if self._calm_started is None else self._calm_started
                self._calm_started = calm
                if now - calm >= p.recover_s:
                    self.state = NORMAL
                    self.recoveries += 1
                    self.episodes.append({"start_s": round(self.episode_start - self.baseline["t"], 1),
                                          "seconds": round(now - (self.episode_start or now), 1),
                                          "peak_pressure": self.peak["pressure_level"]})
                    self.reasons = []
                    self._calm_started = None
                else:
                    self.state = RECOVERY
            if reasons:
                self._calm_started = None
            if self.state != NORMAL:
                self.elevated_seconds += dt
            if self.state != previous and self.emit is not None:
                self.emit("memory_state", state=self.state, previous=previous, reasons=list(self.reasons), **self.last)
            return None

    def _stop(self, reason: str) -> str:
        with self._lock:
            self.state = CRITICAL
            self.admit_work = False
            if self.stop_reason is None:
                self.stop_reason = reason
                self.stopped_at = time.time()
        if self.emit is not None:
            try:
                self.emit("memory_state", state=CRITICAL, reasons=[reason], **self.last)
            except Exception:
                pass
        return reason

    def can_admit(self) -> bool:
        with self._lock:
            return self.admit_work and self.state != CRITICAL

    def is_stopped(self) -> bool:
        return not self.can_admit()

    # -- what the rest of the engine may ask ---------------------------------------------------------------------

    def elevated(self) -> bool:
        return self.state in (ELEVATED, RECOVERY, CRITICAL)

    def scale(self) -> float:
        return self.policy.elevated_scale if self.elevated() else 1.0

    def begin_request(self) -> None:
        with self._lock:
            self.request_throttled_s = 0.0

    def checkpoint(self, label: str = "", *, max_wait: float | None = None, sleep=time.sleep,
                   clock=time.monotonic) -> dict:
        """Wait, bounded, at a safe boundary while ELEVATED (not RECOVERY: that has already calmed). The caller holds no
        GPU work that this wait could starve: it is called between evaluated chunks and between requests."""
        if memory_policy.is_swap_growth_policy():
            # Swap-growth-only policy: memory pressure and compressor churn are telemetry-only.
            # Do not artificially stall prefill progress or sleep between evaluated chunks.
            return {"waited_s": 0.0, "state": self.state}
        started = clock()
        limit = self.policy.throttle_max_wait_s if max_wait is None else max_wait
        waited = 0.0
        while self.state == ELEVATED and waited < limit:
            if self.request_throttled_s + waited >= self.policy.throttle_request_cap_s:
                break
            sleep(0.5)
            waited = clock() - started
        with self._lock:
            self.throttled_s += waited
            self.request_throttled_s += waited
        if waited > 0 and self.emit is not None:
            self.emit("memory_throttle", label=label, waited_s=round(waited, 2), state=self.state)
        return {"waited_s": round(waited, 2), "state": self.state}

    def swap_increase_mb(self) -> float:
        """Current swap growth over the launch baseline in MiB (0 before the first sample)."""
        with self._lock:
            return float(self.last.get("swap_increase_mib", 0.0))

    @property
    def baseline_swap_used_mb(self) -> float:
        with self._lock:
            return float((self.baseline or {}).get("swap_used_mb") or 0.0)

    @property
    def _peak_swap_growth_mb(self) -> float:
        with self._lock:
            return float(self.peak.get("swap_increase_mib", 0.0))

    def ledger(self) -> dict:
        """Everything the owner asked to see about swap and compression, for the request log and the final report."""
        with self._lock:
            base = self.baseline or {}
            cur_swap = self.last.get("swap_increase_mib", 0.0)
            peak_swap = self.peak.get("swap_increase_mib", 0.0)
            return {
                "policy_identifier": (
                    memory_policy.POLICY_SWAP_GROWTH_ONLY if self.policy.swap_only else "threshold-governed"
                ),
                "swap_baseline_mb": base.get("swap_used_mb"),
                "current_swap_growth_mb": round(cur_swap, 1),
                "peak_swap_growth_mb": round(peak_swap, 1),
                "threshold_gib": self.policy.swap_growth_stop_gib,
                "threshold_mb": self.policy.swap_growth_stop_gib * 1024,
                "stop_reason": self.stop_reason,
                "stopped_at": self.stopped_at,
                "can_admit": self.can_admit(),
                "probe_availability": dict(self._probes),
                "telemetry_only_checks": self.policy.telemetry_only_checks(),
                "disclaimer": memory_policy.DISCLAIMER if self.policy.swap_only else "",
                "state": self.state,
                "reasons": list(self.reasons),
                "recovery_events": self.recoveries,
                "episodes": list(self.episodes[-20:]),
                "elevated_seconds": round(self.elevated_seconds, 1),
                "throttled_seconds": round(self.throttled_s, 1),
                "swap_at_launch_mb": base.get("swap_used_mb"),
                "peak": dict(self.peak),
                "last": dict(self.last),
                "swap_growth_stop_gib": self.policy.swap_growth_stop_gib,
            }


# -- process-wide handle -------------------------------------------------------------------------------------------

_ACTIVE: list = [None]


def install(controller: Controller | None) -> None:
    _ACTIVE[0] = controller


def current() -> Controller | None:
    return _ACTIVE[0]


def state() -> str:
    c = _ACTIVE[0]
    return c.state if c is not None else NORMAL


def elevated() -> bool:
    c = _ACTIVE[0]
    return bool(c is not None and c.elevated())


def scale() -> float:
    c = _ACTIVE[0]
    return c.scale() if c is not None else 1.0


def forbid_optional() -> bool:
    """True while an optional allocation (pool growth, extra RAM snapshots, bigger scratch) must wait."""
    return elevated()


def checkpoint(label: str = "", **kwargs) -> dict:
    c = _ACTIVE[0]
    return c.checkpoint(label, **kwargs) if c is not None else {"waited_s": 0.0, "state": NORMAL}


def begin_request() -> None:
    c = _ACTIVE[0]
    if c is not None:
        c.begin_request()


# The server registers how to ask "is an inference job running right now?". The guard uses it to decide between an
# immediate exit (idle: nothing in flight) and a short cooperative-cancel grace period (a job can stop at its next
# streamed-layer boundary and tell its client why).
_WORKER_PROBE: list = [None]


def register_worker_probe(probe) -> None:
    _WORKER_PROBE[0] = probe


def worker_busy() -> bool:
    """True when a job is running. No registered probe (chat, benchmarks, unit tests) means no server job exists, so a stop
    exits at once exactly as before; a probe that raises counts as busy (the grace period is bounded either way)."""
    probe = _WORKER_PROBE[0]
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        return True


def can_admit_work() -> bool:
    c = _ACTIVE[0]
    return c.can_admit() if c is not None else True


def is_stopped() -> bool:
    c = _ACTIVE[0]
    return c.is_stopped() if c is not None else False


def ledger() -> dict | None:
    c = _ACTIVE[0]
    return c.ledger() if c is not None else None


# -- memory categories ---------------------------------------------------------------------------------------------

def categorize(reading, *, mlx_active: int = 0, mlx_cache: int = 0, rss: int = 0) -> dict:
    """Split the machine's memory into the five groups the owner asked to keep apart (bytes).

    ``reading`` is a ``purge_recovery.Reading`` (pages). Approximations are labelled: other-application anonymous memory
    is system anonymous minus this process's RSS; compressed memory is reported on its own because RSS does not include it."""
    page = reading.page
    anonymous = reading.anonymous * page
    return {
        "active_computation": int(mlx_active),                       # weights, KV, recurrent state, scratch ring (MLX-accounted)
        "reclaimable_own": int(mlx_cache),                           # MLX allocator cache; trim is safe when idle
        "reclaimable_file_cache": min(reading.file_backed, reading.inactive + reading.speculative) * page,
        "other_applications": max(0, anonymous - int(rss)),          # approximate
        "unreleasable": reading.wired * page,                        # wired: kernel, GPU-resident buffers
        "compressed": reading.compressor * page,
        "swap_used": int(reading.swap_used),
        "available": int(reading.available),
        "pressure": int(reading.pressure),
    }
