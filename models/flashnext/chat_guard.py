"""Opt-in chat supervisor using the benchmark's existing probes and safety rules.

On a guard trip the benchmark exits the process: a GPU operation may be in
flight and freeing its buffers from the monitoring thread is not safe.
"""
from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from models.flashnext import memory_controller as MC
from models.flashnext import memory_waiver
from models.flashnext.tests.bench import prefill_memory_harness as H
from models.flashnext.tests.bench.bench_prefill_memory import MemoryTracer, RealProbe


class StrictProbe(RealProbe):
    def machine(self):
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=True)
        text = result.stdout
        required = ("page size of", "Pages free", "Pages inactive", "Pages speculative",
                    "Pages occupied by compressor", "Swapins", "Swapouts")
        if (not re.search(r"page size of \d+ bytes", text) or
                any(part not in text for part in required) or
                any(not re.search(rf"(?m)^\s*{re.escape(label)}\s*:\s*\d+\.", text)
                    for label in required[1:])):
            raise ValueError("incomplete vm_stat output")
        return H.parse_vm_stat(text)


class GuardTracer(MemoryTracer):
    def _run(self):
        try:
            super()._run()
        except BaseException as error:
            reason = f"monitor failed: {type(error).__name__}: {error}"
            try:
                self._write({"guard": reason, "phase": None})
            finally:
                self.on_guard(reason)


class ChatGuard:
    def __init__(self, path, *, probe=None, sampler=None, exit_fn=os._exit, controller=None, clock=time.monotonic):
        self.probe = probe or StrictProbe()
        self.sampler = sampler or H.Sampler(runner=self._read_command)
        self.exit_fn = exit_fn
        self.path = Path(path)
        self.start_sample = None
        self.start_vm = None
        self.watchdog = None
        self.last_power_check = 0.0
        self.tracer = None
        self.closed = False
        self.controller = controller
        self.clock = clock

    @staticmethod
    def _read_command(command):
        return subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout

    def _reason(self, row):
        for key in ("mlx_active", "mlx_cache", "available_mb", "compressor_mb", "swapout", "swapin"):
            if row.get(key) is None:
                return f"critical probe unavailable: {key}"
        if not row.get("metal_limit_bytes"):
            return "critical probe unavailable: Metal working-set limit"
        waived = memory_waiver.active()
        reason = H.guard_reason(row, min_available_gb=0.0 if waived else H.MIN_AVAILABLE_GB,
                                compressor_stop_available_mib=0.0 if waived else H.COMPRESSOR_STOP_AVAILABLE_GIB * 1024)
        if reason:
            return reason
        now = time.monotonic()
        if self.start_sample is None or now - self.last_power_check >= 1:
            self.last_power_check = now
            sample = self.sampler.sample()
            for key in ("battery_percent", "power_source", "low_power", "pressure_level", "swap_used_mb"):
                if sample.get(key) is None:
                    return f"critical probe unavailable: {key}"
            # The benchmark parser reports no recorded warning as False. Require
            # recognizable output as well, since an empty response also parses False.
            thermal = self.sampler.runner(["pmset", "-g", "therm"])
            if not thermal or "error" in thermal.lower():
                return "critical probe unavailable: thermal status"
            if "thermal warning level" not in thermal.lower():
                return "critical probe unavailable: thermal warning status"
            if sample.get("thermal_speed_limit") is None and "No CPU power status has been recorded" not in thermal:
                return "critical probe unavailable: thermal speed limit"
            if sample.get("thermal_warning"):
                return "thermal warning"
            vm = {"swapout": row["swapout"], "swapin": row["swapin"]}
            if self.start_sample is None:
                if row["available_mb"] < 9 * 1024 and not waived:
                    return "startup available memory is below 9 GiB"
                self.start_sample, self.start_vm = sample, vm
                # The watchdog keeps battery, Low Power Mode, available-memory floor, compressor-with-low-availability and
                # thermal. Pressure and swap are decided by the staged controller below (owner decision 2026-11-08), so
                # the watchdog's immediate pressure-2 stop, its 64 MB swap-growth stop and its "any swap-out" stop are
                # switched off here and replaced by: level 4 held, level >= 2 persisting, +2.5 GiB swap, active swap rate.
                policy = H.SafetyPolicy(pressure_stop_level=99, swap_growth_mb=float("inf"), swapin_limit_mib=float("inf"))
                if waived:
                    policy.min_available_gb = 0.0
                    policy.compressor_stop_available_mib = 0.0
                self.watchdog = H.Watchdog(policy, sample)
                if self.controller is None:
                    self.controller = MC.Controller(MC.Policy.from_environment(swap_only=waived), page_bytes=policy.page_bytes,
                                                    emit=self.tracer.event)
                MC.install(self.controller)
            sample["available_mb"] = row["available_mb"]
            sample["compressor_mb"] = row["compressor_mb"]
            stop = self.watchdog.check(sample, {"swapout": 0, "swapin": 0},
                                       {"compressor_mb": self.tracer.compressor_start_mb})
            controller_stop = self.controller.observe(
                self.clock(), pressure=sample["pressure_level"], swap_used_mb=sample["swap_used_mb"],
                swapout_pages=row["swapout"], swapin_pages=row["swapin"], available_mb=row["available_mb"],
                compressor_mb=row["compressor_mb"])
            self.tracer.event("watchdog_sample", memory_state=self.controller.state, **sample)
            return stop or controller_stop
        return None

    def _stop(self, reason):
        print(f"FlashNext safety guard stopped the process: {reason}\nTrace: {self.path}",
              file=sys.stderr, flush=True)
        self.exit_fn(H.GUARD_EXIT_CODE)

    def start(self):
        metal = self.probe.metal_limit()
        self.tracer = GuardTracer(self.probe, self.path, interval=0.25,
                                  guard=self._reason, on_guard=self._stop, metal_limit=metal,
                                  strict_readings=True,
                                  compressor_warn_mib=H.MAX_COMPRESSOR_GROWTH_GIB * 1024)
        # Synchronous first sample means missing probes block model loading.
        try:
            self.tracer.step()
        except Exception as error:
            reason = f"monitor failed: {type(error).__name__}: {error}"
            try:
                self.tracer._write({"guard": reason, "phase": None})
            finally:
                self._stop(reason)
            raise RuntimeError(reason) from error
        if self.tracer.guard_reason:
            raise RuntimeError(self.tracer.guard_reason)
        from models.flashnext import trace_phase

        trace_phase.register(self.tracer.phase)
        self.tracer.start()
        atexit.register(self.close)
        return self

    def close(self):
        if self.tracer is not None and not self.closed:
            self.closed = True
            self.tracer.event("shutdown", memory_ledger=self.controller.ledger() if self.controller else None)
            if MC.current() is self.controller:
                MC.install(None)
            self.tracer.stop()

