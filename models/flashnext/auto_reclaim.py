"""Automatic Request-to-Request Memory Reclamation (opt-in: FLASHNEXT_AUTO_MEMORY_RECLAIM=1).

Implements pressure-aware request-to-request memory rebalancing for long-running DeepSeek Harness
agent workflows across repeated tool calls, long system prompts, and multi-turn conversations.

Workflow: REQUEST -> INFERENCE -> SAFE CLEANUP -> MEMORY REBALANCING -> NEXT REQUEST

Three operating states with hysteresis:
- NORMAL: Preserve warm 2.0 GB decode pool, keep hot RAM snapshots, trim MLX allocator only if large (>256 MB).
- CONSTRAINED (< 5.5 GB headroom or pressure >= 2): Evict cold RAM prefix snapshots to SSD, trim MLX cache,
  consider sizing down decode pool to 1.0 GB on next build.
- CRITICAL (< 3.5 GB headroom or pressure >= 4): Release unused buffers, release pool if unleased,
  prevent dangerous allocations.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

FLAG_AUTO_RECLAIM = "FLASHNEXT_AUTO_MEMORY_RECLAIM"
FLAG_AUTO_PURGE = "FLASHNEXT_AUTO_PURGE"

GIB = 2 ** 30
GB = 10 ** 9


class PressureState(str, Enum):
    NORMAL = "NORMAL"
    CONSTRAINED = "CONSTRAINED"
    CRITICAL = "CRITICAL"


def auto_reclaim_enabled() -> bool:
    return os.environ.get(FLAG_AUTO_RECLAIM, "0") == "1"


def auto_purge_enabled() -> bool:
    return os.environ.get(FLAG_AUTO_PURGE, "0") == "1"


@dataclass(frozen=True)
class ReclaimConfig:
    constrained_headroom_bytes: int = int(5.5 * GB)
    critical_headroom_bytes: int = int(3.5 * GB)
    recovery_margin_bytes: int = int(0.5 * GB)
    cooldown_s: float = 60.0
    cache_clear_normal_bytes: int = 256 * 1024 * 1024
    purge_cooldown_s: float = 1800.0  # at least 30 minutes between purges
    max_daily_purges: int = 2
    min_file_backed_for_purge_mb: float = 3000.0


@dataclass
class ReclaimReport:
    state_before: str = PressureState.NORMAL.value
    state_after: str = PressureState.NORMAL.value
    available_before: int = 0
    available_after: int = 0
    mlx_cache_before: int = 0
    mlx_cache_after: int = 0
    ram_prefix_evicted_bytes: int = 0
    pool_released_bytes: int = 0
    pool_retained: bool = True
    pool_capacity_gb: float | None = None
    purge_executed: bool = False
    purge_recovered_bytes: int = 0
    reasons: list[str] = field(default_factory=list)


class AutoReclaimer:
    def __init__(self, cfg: ReclaimConfig = ReclaimConfig(), system=None, clock=time.time):
        self.cfg = cfg
        self._system = system
        self._clock = clock
        self.current_state = PressureState.NORMAL
        self.last_state_change = 0.0
        self.purge_timestamps: list[float] = []
        self.history: list[ReclaimReport] = []
        self._lock = threading.Lock()

    def _read_system(self) -> dict:
        if self._system is not None:
            return self._system()
        from . import memory_manager
        return memory_manager.read_system()

    def evaluate_state(self, available_bytes: int, pressure_level: int, now: float) -> PressureState:
        """Evaluate pressure state with hysteresis."""
        # Immediate upward transitions on memory pressure (safety first)
        if available_bytes < self.cfg.critical_headroom_bytes or pressure_level >= 4:
            if self.current_state != PressureState.CRITICAL:
                self.current_state = PressureState.CRITICAL
                self.last_state_change = now
            return self.current_state

        if available_bytes < self.cfg.constrained_headroom_bytes or pressure_level >= 2:
            if self.current_state == PressureState.NORMAL:
                self.current_state = PressureState.CONSTRAINED
                self.last_state_change = now
            elif self.current_state == PressureState.CRITICAL:
                # Downward from CRITICAL needs cooldown
                if now - self.last_state_change >= self.cfg.cooldown_s:
                    self.current_state = PressureState.CONSTRAINED
                    self.last_state_change = now
            return self.current_state

        # Downward transition to NORMAL requires cooldown and margin above threshold
        normal_recovery_threshold = self.cfg.constrained_headroom_bytes + self.cfg.recovery_margin_bytes
        if available_bytes >= normal_recovery_threshold and pressure_level < 2:
            if self.current_state != PressureState.NORMAL:
                if now - self.last_state_change >= self.cfg.cooldown_s:
                    self.current_state = PressureState.NORMAL
                    self.last_state_change = now

        return self.current_state

    def post_request_cleanup(
        self,
        store=None,
        prefix_store=None,
        *,
        clear_cache=None,
        synchronize=None,
        now: float | None = None
    ) -> ReclaimReport:
        """Safe post-request memory reclamation across Phase 2, 3, 5 requirements.
        
        Runs in try/finally after each request (including errors/cancellations).
        """
        with self._lock:
            now = self._clock() if now is None else now
            from . import pool_lifecycle, prefill_ring

            report = ReclaimReport()
            sys_before = self._read_system()
            avail_before = int(sys_before.get("available", 0))
            pressure_before = int(sys_before.get("pressure", 1))
            report.available_before = avail_before
            report.state_before = self.current_state.value

            # 1. GPU synchronization: ensure lazy MLX operations are evaluated before freeing
            if synchronize is not None:
                synchronize()

            # 2. Release expired prefill scratch ring fence
            prefill_ring.fence()

            # 3. Restore uncached descriptors if they were flipped
            if store is not None and hasattr(store, "set_uncached_reads") and getattr(store, "_uncached", False):
                try:
                    store.set_uncached_reads(False)
                    report.reasons.append("restored buffered file descriptors")
                except Exception:
                    pass

            # 4. Read MLX allocator memory
            cache_before = 0
            try:
                import mlx.core as mx
                cache_before = mx.get_cache_memory()
                if clear_cache is None:
                    clear_cache = mx.clear_cache
            except Exception:
                clear_cache = None
            report.mlx_cache_before = cache_before

            # 5. Evaluate pressure state
            state = self.evaluate_state(avail_before, pressure_before, now)
            report.state_after = state.value

            # 6. Pressure-aware action dispatch
            pool = getattr(store, "_flashnext_expert_pool", None) if store is not None else None
            pool_bytes = pool_lifecycle.pool_bytes(pool) if pool else 0
            in_flight = pool_lifecycle.others_in_flight()

            if state == PressureState.CRITICAL:
                # Critical: aggressively clear cache, evict RAM prefix, release pool if idle
                report.reasons.append("CRITICAL pressure: aggressive reclamation")
                if clear_cache is not None:
                    clear_cache()
                    report.reasons.append("cleared MLX allocator cache")
                if prefix_store is not None:
                    evicted = prefix_store.shrink_ram(0)
                    report.ram_prefix_evicted_bytes = evicted
                    if evicted:
                        report.reasons.append(f"evicted {evicted / 1e6:.1f} MB RAM prefix snapshots to SSD")
                if pool is not None and in_flight == 0:
                    freed = pool_lifecycle.release_decode_pool(store, synchronize=synchronize, clear_cache=clear_cache)
                    report.pool_released_bytes = freed
                    report.pool_retained = False
                    if freed:
                        report.reasons.append(f"released {freed / 1e9:.2f} GB decode pool to protect runtime floor")

            elif state == PressureState.CONSTRAINED:
                # Constrained: clear MLX cache, move cold RAM snapshots to SSD, size down pool on next build
                report.reasons.append("CONSTRAINED memory: trimming caches")
                if clear_cache is not None:
                    clear_cache()
                    report.reasons.append("trimmed MLX allocator cache")
                if prefix_store is not None:
                    evicted = prefix_store.shrink_ram(0)
                    report.ram_prefix_evicted_bytes = evicted
                    if evicted:
                        report.reasons.append(f"moved {evicted / 1e6:.1f} MB cold RAM prefix snapshots to SSD")
                # Maintain warm pool if already resident, but signal next pool build should use 1.0 GB
                if pool is not None:
                    report.pool_retained = True
                    report.pool_capacity_gb = getattr(pool, "slots", 0) * getattr(pool, "stride", 0) / 1e9
                    from . import expert_pool
                    expert_pool.set_budget_gb(1.0)
                    report.reasons.append("retained warm decode pool; next build budgeted at 1.0 GB")

            else:  # NORMAL
                # Normal: retain warm decode pool, keep RAM snapshots, clear allocator only if bloated
                if cache_before > self.cfg.cache_clear_normal_bytes and clear_cache is not None:
                    clear_cache()
                    report.reasons.append(f"trimmed MLX cache ({cache_before / 1e6:.1f} MB > 256 MB threshold)")
                if pool is not None:
                    report.pool_retained = True
                    report.pool_capacity_gb = getattr(pool, "slots", 0) * getattr(pool, "stride", 0) / 1e9
                    report.reasons.append("NORMAL state: warm decode pool retained")

            # 7. Collect fresh post-cleanup measurements
            sys_after = self._read_system()
            report.available_after = int(sys_after.get("available", 0))
            try:
                import mlx.core as mx
                report.mlx_cache_after = mx.get_cache_memory()
            except Exception:
                report.mlx_cache_after = 0

            self.history.append(report)
            return report

    def exceptional_startup_purge(self, now: float | None = None) -> tuple[bool, int, str]:
        """SUPERSEDED, not called by the engine: the live mechanism is ``purge_recovery`` (no daily count limit,
        activity gate, learned backoff, idle monitor). Kept, with its old 2-per-day and 30-minute limits, for its
        tests. Returns (success, recovered_bytes, reason).
        """
        now = self._clock() if now is None else now
        if not auto_purge_enabled():
            return False, 0, "FLASHNEXT_AUTO_PURGE is disabled (default)"

        # Rate limiting: max 2 purges in 24 hours
        one_day_ago = now - 86400.0
        self.purge_timestamps = [t for t in self.purge_timestamps if t >= one_day_ago]
        if len(self.purge_timestamps) >= self.cfg.max_daily_purges:
            return False, 0, f"maximum daily purges ({self.cfg.max_daily_purges}/24h) reached"

        # Cooldown: at least 30 minutes since last purge
        if self.purge_timestamps and (now - self.purge_timestamps[-1]) < self.cfg.purge_cooldown_s:
            rem = int(self.cfg.purge_cooldown_s - (now - self.purge_timestamps[-1]))
            return False, 0, f"purge cooldown active ({rem}s remaining)"

        # Verify filesystem cache presence before attempting purge
        sys_state = self._read_system()
        avail_before = int(sys_state.get("available", 0))
        file_backed_mb = float(sys_state.get("file_backed_mb", 0.0))
        if file_backed_mb < self.cfg.min_file_backed_for_purge_mb:
            return False, 0, f"file-backed cache ({file_backed_mb:.1f} MB) too small for purge recovery"

        # Execute non-interactive purge only: never prompt for password
        cmd = ["sudo", "-n", "/usr/sbin/purge"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode != 0:
                return False, 0, f"sudo -n purge failed with exit code {res.returncode} (password required or disabled)"
        except Exception as e:
            return False, 0, f"purge execution error: {e}"

        # Settle and measure recovered memory
        time.sleep(1.0)
        sys_after = self._read_system()
        avail_after = int(sys_after.get("available", 0))
        recovered = max(0, avail_after - avail_before)

        self.purge_timestamps.append(now)
        return True, recovered, f"recovered {recovered / 1e6:.1f} MB via startup purge"


_RECLAIMER = [None]


def get_reclaimer() -> AutoReclaimer:
    if _RECLAIMER[0] is None:
        _RECLAIMER[0] = AutoReclaimer()
    return _RECLAIMER[0]


def reset_for_tests() -> None:
    _RECLAIMER[0] = None
    from . import expert_pool
    expert_pool.set_budget_gb(None)
