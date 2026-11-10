"""Offline tests of the memory diagnostics added to the prefill benchmark after the first
real-model run reached critical memory pressure: the stop sample is recorded, a 1 s memory
trace, the child's own memory guard, the reduced decode pool and the short-prompt sizes.

Fakes only: no checkpoint, no model, no GPU. ``os._exit`` is mocked wherever a guard could
fire, so a test can never end the test process.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from veystrix import results
from models.flashnext import expert_pool, prefill, prefill_ring
from models.flashnext.tests.bench import bench_prefill_memory as B
from models.flashnext.tests.bench import prefill_memory_harness as H
from models.flashnext.tests.unit import test_bench_prefill_memory as T

GB = 1_000_000_000
PAGE = 16384

# vm_stat output as macOS prints it (page size 16384), trimmed to what the parser reads.
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100000.
Pages active:                                 200000.
Pages inactive:                                50000.
Pages speculative:                             10000.
Pages throttled:                                   0.
Pages wired down:                             120000.
Pages purgeable:                                1000.
"Translation faults":                      999999.
Pages copy-on-write:                       375940047.
Pages zero filled:                        3727884038.
Pages reactivated:                         135806105.
Pages purged:                               30553944.
File-backed pages:                            385769.
Anonymous pages:                              151460.
Pages stored in compressor:                   101407.
Pages occupied by compressor:                  40000.
Decompressions:                            139903581.
Compressions:                              178943544.
Pageins:                                   111120816.
Pageouts:                                         51856.
Swapins:                                        2952265.
Swapouts:                                       3798763.
"""


class VmStatTests(unittest.TestCase):
    def test_available_memory_is_free_plus_inactive_plus_speculative(self):
        parsed = H.parse_vm_stat(VM_STAT)
        self.assertEqual(parsed["page_bytes"], PAGE)
        self.assertAlmostEqual(parsed["available_mb"], 160000 * PAGE / 1048576, places=0)
        self.assertAlmostEqual(parsed["compressor_mb"], 40000 * PAGE / 1048576, places=0)

    def test_the_page_cache_and_anonymous_memory_are_read(self):
        parsed = H.parse_vm_stat(VM_STAT)
        self.assertAlmostEqual(parsed["file_backed_mb"], 385769 * PAGE / 1048576, places=0)
        self.assertAlmostEqual(parsed["anonymous_mb"], 151460 * PAGE / 1048576, places=0)

    def test_the_labels_match_what_this_machines_vm_stat_prints(self):
        # The first version of the parser guessed "Pages compressed"; real vm_stat says
        # "Compressions". If macOS ever renames a counter this fails against the live tool.
        import subprocess

        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        if "page size of" not in out:
            self.skipTest("vm_stat is not available here")
        counters = H.parse_vm_stat(out)["counters"]
        for key in ("swapin", "swapout", "compress", "decompress", "pagein", "pageout"):
            self.assertIn(key, counters, key)

    def test_counters_are_read_in_pages(self):
        counters = H.parse_vm_stat(VM_STAT)["counters"]
        self.assertEqual((counters["swapin"], counters["swapout"]), (2952265, 3798763))
        self.assertEqual((counters["compress"], counters["decompress"]), (178943544, 139903581))
        self.assertEqual((counters["pagein"], counters["pageout"]), (111120816, 51856))

    def test_a_different_page_size_scales_the_megabytes(self):
        text = VM_STAT.replace("16384", "4096")
        self.assertAlmostEqual(H.parse_vm_stat(text)["available_mb"], 160000 * 4096 / 1048576, places=0)

    def test_garbage_gives_zero_not_a_crash(self):
        parsed = H.parse_vm_stat("")
        self.assertEqual((parsed["available_mb"], parsed["counters"]), (0.0, {}))


class GuardReasonTests(unittest.TestCase):
    def reading(self, **changes):
        base = {"available_mb": 6000.0, "mlx_active": 5 * GB, "mlx_cache": GB,
                "metal_limit_bytes": H.METAL_LIMIT_BYTES}
        base.update(changes)
        return base

    def test_a_healthy_reading_does_not_stop(self):
        self.assertIsNone(H.guard_reason(self.reading()))

    def test_low_available_memory_stops(self):
        reason = H.guard_reason(self.reading(available_mb=1999.0))
        self.assertIn("available memory", reason)
        self.assertIsNone(H.guard_reason(self.reading(available_mb=2000.0)))

    def test_the_available_threshold_is_configurable(self):
        self.assertIsNotNone(H.guard_reason(self.reading(available_mb=3000.0), min_available_gb=3.5))
        self.assertIsNone(H.guard_reason(self.reading(available_mb=3000.0), min_available_gb=1.0))

    def test_mlx_close_to_the_metal_limit_stops(self):
        limit = H.METAL_LIMIT_BYTES
        near = self.reading(mlx_active=limit - 600_000_000, mlx_cache=200_000_000)   # 0.4 GB headroom
        self.assertIn("Metal limit", H.guard_reason(near))
        enough = self.reading(mlx_active=limit - 700_000_000, mlx_cache=100_000_000)  # 0.6 GB headroom
        self.assertIsNone(H.guard_reason(enough))

    def test_the_cache_counts_toward_the_allocator_headroom(self):
        limit = H.METAL_LIMIT_BYTES
        self.assertIsNotNone(H.guard_reason(self.reading(mlx_active=limit - GB, mlx_cache=600_000_000)))

    def test_an_unknown_limit_or_reading_does_not_stop(self):
        self.assertIsNone(H.guard_reason(self.reading(metal_limit_bytes=None, mlx_active=50 * GB)))
        self.assertIsNone(H.guard_reason(self.reading(available_mb=None)))


class StopSampleTests(unittest.TestCase):
    """The sample that triggers a stop must be in the watchdog's record."""

    def dog(self, **policy):
        return H.Watchdog(H.SafetyPolicy(page_bytes=PAGE, **policy), T.good_sample())

    def test_every_stop_rule_leaves_its_sample_in_last(self):
        cases = [
            (T.good_sample(pressure_level=4, available_mb=900.0, swap_used_mb=2004.0), None, "pressure"),
            (T.good_sample(low_power=True, pressure_level=1), None, "Low Power"),
            (T.good_sample(power_source="battery", battery_percent=20, pressure_level=1), None, "battery"),
            (T.good_sample(available_mb=1500.0, pressure_level=1), None, "available"),
            (T.good_sample(pressure_level=1), {"swapout": 79000}, "swapout"),
            (T.good_sample(pressure_level=1), {"swapin": 5000}, "swap-in"),
        ]
        for sample, vm, word in cases:
            dog = self.dog()
            reason = dog.check(sample, vm)
            self.assertIn(word, reason, word)
            self.assertEqual(dog.last["pressure_level"], sample["pressure_level"], word)
            self.assertEqual(dog.last["swap_used_mb"], sample["swap_used_mb"], word)

    def test_the_level_4_sample_shows_its_pressure_and_swap(self):
        dog = self.dog()
        dog.check(T.good_sample(pressure_level=4, available_mb=900.0, swap_used_mb=2004.44),
                  {"swapin": 100, "swapout": 78976})
        self.assertEqual((dog.last["pressure_level"], dog.last["swap_used_mb"]), (4, 2004.44))
        self.assertEqual((dog.last["available_mb"], dog.last["swapout_pages"]), (900.0, 78976))

    def test_low_available_memory_stops_before_any_pressure_level(self):
        dog = self.dog()
        self.assertIsNone(dog.check(T.good_sample(available_mb=2500.0, pressure_level=1)))
        reason = dog.check(T.good_sample(available_mb=1800.0, pressure_level=1))
        self.assertIn("1.80 GB", reason)
        self.assertIn("2.0 GB", reason)

    def test_an_unreadable_available_figure_does_not_stop(self):
        self.assertIsNone(self.dog().check(T.good_sample(available_mb=None)))
        self.assertIsNone(self.dog().check(T.good_sample()))          # the sampler may not supply it

    def test_the_available_floor_is_the_policys(self):
        dog = self.dog(min_available_gb=0.5)
        self.assertIsNone(dog.check(T.good_sample(available_mb=800.0)))

    def test_run_child_traces_the_stop_sample_and_marks_it(self):
        clock = T.Clock()
        samples = iter([T.good_sample(pressure_level=1, available_mb=5000.0),
                        T.good_sample(pressure_level=4, available_mb=800.0, swap_used_mb=2004.0)])
        sampler = SimpleNamespace(sample=lambda: next(samples))
        dog = H.Watchdog(H.SafetyPolicy(page_bytes=PAGE), T.good_sample())
        result = H.run_child(["x"], {}, timeout=600, sampler=sampler, watchdog=dog,
                             popen=lambda c, env: T.FakeProc(0, polls=100), sleep=clock.sleep,
                             clock=clock, grace=1.0)
        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(result.trace), 2)
        self.assertEqual(result.trace[-1]["pressure_level"], 4)
        self.assertIn("pressure", result.trace[-1]["stop"])
        self.assertNotIn("stop", result.trace[0])
        summary = H.trace_summary(result.trace)
        self.assertEqual((summary["max_pressure_level"], summary["min_available_mb"]), (4, 800.0))

    def test_the_sampler_adds_available_memory_when_given_a_reader(self):
        answers = {("pmset", "-g", "batt"): T.BATT_AC, ("pmset", "-g", "therm"): T.THERM_OK,
                   ("pmset", "-g"): T.PMSET_NORMAL,
                   ("sysctl", "vm.swapusage"): "vm.swapusage: total = 2048.00M  used = 770.44M  free = 1277.56M",
                   ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1"}
        sampler = H.Sampler(runner=lambda cmd: answers[tuple(cmd)], available=lambda: 7123.4)
        self.assertEqual(sampler.sample()["available_mb"], 7123.4)


class ExitCodeTests(unittest.TestCase):
    def test_the_guard_exit_code_is_classified(self):
        status, message = H.classify_exit(H.GUARD_EXIT_CODE)
        self.assertEqual(status, "memory_guard")
        self.assertIn("trace", message)
        self.assertEqual(H.classify_exit(137)[0], "exit_137")


class FakeMachineProbe(T.FakeProbe):
    """The test probe plus the readings the tracer takes."""

    def __init__(self, available=8000.0):
        super().__init__()
        self.available = available
        self.pressure = 1
        self.swap = 770.44
        self.fail_machine = False
        self.compressor = 300.0
        self.compressor_step = 0.0          # added on every reading, to model a growing compressor

    def machine(self):
        if self.fail_machine:
            raise OSError("vm_stat unavailable")
        self.compressor += self.compressor_step
        return {"available_mb": self.available, "compressor_mb": self.compressor,
                "file_backed_mb": 6000.0, "anonymous_mb": 2400.0,
                "counters": {"swapin": 7, "swapout": 3, "compress": 11, "decompress": 5, "pagein": 9}}

    def swap_used_mb(self):
        return self.swap

    def pressure_level(self):
        return self.pressure


class TracerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "arm.trace.jsonl"
        self.exits = []

    def tracer(self, probe=None, **kwargs):
        kwargs.setdefault("guard", lambda row: H.guard_reason(row))
        kwargs.setdefault("on_guard", self.exits.append)
        kwargs.setdefault("metal_limit", H.METAL_LIMIT_BYTES)
        tracer = B.MemoryTracer(probe or FakeMachineProbe(), self.path, **kwargs)
        self.addCleanup(tracer.stop)          # closes the trace file
        return tracer

    def lines(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_a_row_has_every_requested_field(self):
        phase = B.Phase("prefill")
        tracer = self.tracer(phase=phase, store=T.FakeStore())
        row = tracer.step()
        for key in ("t", "phase", "mlx_active", "mlx_cache", "mlx_peak", "rss", "available_mb",
                    "compressor_mb", "file_backed_mb", "anonymous_mb", "swap_used_mb", "pressure_level",
                    "swapin", "swapout", "compress", "decompress", "pool"):
            self.assertIn(key, row, key)
        self.assertEqual((row["phase"], row["mlx_active"], row["rss"]), ("prefill", 4 * GB, 5 * GB))
        self.assertEqual((row["available_mb"], row["compressor_mb"]), (8000.0, 300.0))
        self.assertEqual((row["swap_used_mb"], row["pressure_level"], row["swapout"]), (770.44, 1, 3))
        self.assertEqual(row["pool"], {"exists": False})

    def test_the_phase_comes_from_the_main_thread(self):
        phase = B.Phase("load")
        tracer = self.tracer(phase=phase)
        tracer.step()
        phase.set("prefill")
        tracer.step()
        self.assertEqual([r["phase"] for r in tracer.rows], ["load", "prefill"])

    def test_the_pool_state_is_traced_once_a_pool_exists(self):
        pool = SimpleNamespace(slots=651, _slot_of={1: 0, 2: 1}, hits=4, misses=6)
        store = SimpleNamespace(_flashnext_expert_pool=pool)
        row = self.tracer(store=store).step()
        self.assertEqual(row["pool"], {"exists": True, "slots": 651, "used": 2, "hits": 4, "misses": 6})

    def test_every_row_is_flushed_to_disk_as_it_is_written(self):
        tracer = self.tracer()
        tracer.step()
        self.assertEqual(len(self.lines()), 1)          # readable before stop(): survives a kill
        tracer.event("expert_pool_creating", size_bytes=2_000_000_000)
        tracer.step()
        kinds = ["event" if "event" in row else "sample" for row in self.lines()]
        self.assertEqual(kinds, ["sample", "event", "sample"])

    def test_an_event_records_when_and_during_which_phase(self):
        phase = B.Phase("decode")
        tracer = self.tracer(phase=phase)
        tracer.event("expert_pool_created", slots=651, mapped_bytes=2_000_000_000)
        row = self.lines()[0]
        self.assertEqual((row["event"], row["during"], row["slots"]), ("expert_pool_created", "decode", 651))

    def test_low_available_memory_calls_the_guard_once_after_writing_the_reason(self):
        probe = FakeMachineProbe(available=900.0)
        tracer = self.tracer(probe)
        tracer.step()
        tracer.step()
        self.assertEqual(len(self.exits), 1)
        self.assertIn("available memory", self.exits[0])
        guard_rows = [r for r in self.lines() if "guard" in r]
        self.assertEqual(len(guard_rows), 1)
        self.assertEqual(guard_rows[0]["guard"], self.exits[0])

    def test_mlx_near_the_metal_limit_calls_the_guard(self):
        probe = FakeMachineProbe()
        probe.active = H.METAL_LIMIT_BYTES - 300_000_000
        tracer = self.tracer(probe)
        tracer.step()
        self.assertEqual(len(self.exits), 1)
        self.assertIn("Metal limit", self.exits[0])

    def test_a_healthy_machine_never_calls_the_guard(self):
        tracer = self.tracer()
        for _ in range(5):
            tracer.step()
        self.assertEqual(self.exits, [])

    def test_the_default_guard_exit_is_the_guard_exit_code(self):
        tracer = B.MemoryTracer(FakeMachineProbe(available=500.0), self.path,
                                guard=lambda row: H.guard_reason(row), metal_limit=None)
        self.addCleanup(tracer.stop)
        with mock.patch.object(os, "_exit") as leave:
            tracer.step()
        leave.assert_called_once_with(H.GUARD_EXIT_CODE)

    def test_a_failing_reading_does_not_kill_the_tracer(self):
        probe = FakeMachineProbe()
        probe.fail_machine = True
        row = self.tracer(probe).step()
        self.assertIsNone(row["available_mb"])
        self.assertEqual(row["mlx_active"], 4 * GB)
        self.assertEqual(self.exits, [])                # no reading is not low memory

    def test_the_thread_samples_repeatedly_and_stops_cleanly(self):
        tracer = self.tracer(interval=0.01)
        tracer.start()
        time.sleep(0.15)
        tracer.stop()
        count = len(tracer.rows)
        self.assertGreaterEqual(count, 3)
        time.sleep(0.05)
        self.assertEqual(len(tracer.rows), count)       # nothing after stop
        self.assertFalse([t for t in threading.enumerate() if t.name == "memory-tracer"])

    def test_the_summary_gives_per_phase_extremes_events_and_the_guard(self):
        phase = B.Phase("prefill")
        probe = FakeMachineProbe()
        tracer = self.tracer(probe, phase=phase)
        tracer.step()
        probe.available, probe.active, probe.swap = 3000.0, 6 * GB, 900.0
        tracer.step()
        phase.set("decode")
        tracer.event("expert_pool_created", slots=651, mapped_bytes=GB)
        probe.available = 800.0
        tracer.step()
        summary = H.summarize_trace(tracer.rows)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["phases"]["prefill"]["available_mb"], 3000.0)
        self.assertEqual(summary["phases"]["prefill"]["mlx_active"], 6 * GB)
        self.assertEqual(summary["phases"]["prefill"]["swap_used_mb"], 900.0)
        self.assertEqual(summary["phases"]["decode"]["available_mb"], 800.0)
        self.assertEqual([e["event"] for e in summary["events"]], ["expert_pool_created"])
        self.assertIn("available memory", summary["guard"])


class CompressorGrowthTests(unittest.TestCase):
    """Growth over 2 GiB is a warning; it stops an arm only with less than 3 GiB available."""
    LIMIT = 2048.0          # MiB: 2 GiB
    FLOOR = 3072.0          # MiB: 3 GiB

    def dog(self, **policy):
        return H.Watchdog(H.SafetyPolicy(page_bytes=PAGE, **policy), T.good_sample())

    def sample(self, compressor, available=6000.0, **changes):
        return T.good_sample(compressor_mb=compressor, available_mb=available, **changes)

    def test_the_defaults_are_two_gib_of_growth_and_three_gib_available(self):
        policy = H.SafetyPolicy()
        self.assertEqual((policy.compressor_limit_mib, policy.compressor_stop_available_mib), (2048.0, 3072.0))
        self.assertEqual((H.MAX_COMPRESSOR_GROWTH_GIB, H.COMPRESSOR_STOP_AVAILABLE_GIB), (2.0, 3.0))

    def test_growth_over_the_limit_with_plenty_available_only_warns(self):
        dog = self.dog()
        start = self.sample(300.0)
        self.assertIsNone(dog.check(self.sample(300.0 + self.LIMIT + 1), None, start))
        self.assertTrue(dog.last["compressor_warning"])
        self.assertEqual(len(dog.warnings), 1)
        self.assertEqual(dog.warnings[0]["available_mb"], 6000.0)

    def test_growth_over_the_limit_with_less_than_three_gib_available_stops(self):
        start = self.sample(300.0)
        reason = self.dog().check(self.sample(300.0 + self.LIMIT + 1, available=2500.0), None, start)
        self.assertIn("compressor grew", reason)
        self.assertIn("available", reason)
        self.assertIn("floor 3 GiB", reason)
        self.assertIn("limit 2 GiB", reason)

    def test_both_boundaries_are_strict(self):
        start = self.sample(300.0)
        self.assertIsNone(self.dog().check(self.sample(300.0 + self.LIMIT, available=2500.0), None, start))
        self.assertIsNone(self.dog().check(self.sample(300.0 + self.LIMIT + 1, available=self.FLOOR), None, start))
        self.assertIsNotNone(self.dog().check(self.sample(300.0 + self.LIMIT + 1, available=self.FLOOR - 1), None, start))

    def test_low_available_alone_between_the_floors_does_not_stop(self):
        # 2.0 to 3.0 GiB available is fine unless the compressor has also grown over its limit
        start = self.sample(300.0)
        self.assertIsNone(self.dog().check(self.sample(900.0, available=2500.0), None, start))
        self.assertIsNone(self.dog().check(self.sample(900.0, available=2001.0), None, start))

    def test_the_two_gib_available_floor_stops_with_or_without_growth(self):
        start = self.sample(300.0)
        self.assertIn("available memory", self.dog().check(self.sample(300.0, available=1900.0), None, start))
        self.assertIn("available memory", self.dog().check(self.sample(5000.0, available=1900.0), None, start))

    def test_the_128_token_f0_growth_with_ample_memory_would_now_continue(self):
        # measured: 0.31 to 3.06 GB with 5.7 GB available and pressure 1
        dog = self.dog()
        self.assertIsNone(dog.check(self.sample(3060.0, available=5700.0), None, self.sample(310.0)))
        self.assertTrue(dog.last["compressor_warning"])

    def test_growth_is_measured_from_each_arms_own_start_not_the_runs(self):
        dog = H.Watchdog(H.SafetyPolicy(page_bytes=PAGE), self.sample(300.0))
        second = self.sample(2900.0)
        self.assertIsNone(dog.check(self.sample(3000.0, available=2500.0), None, second))
        self.assertIsNotNone(dog.check(self.sample(2900.0 + 2049.0, available=2500.0), None, second))

    def test_a_shrinking_compressor_is_not_growth(self):
        dog = self.dog()
        self.assertIsNone(dog.check(self.sample(100.0, available=2500.0), None, self.sample(3000.0)))
        self.assertFalse(dog.last["compressor_warning"])

    def test_without_a_baseline_or_a_reading_nothing_stops_or_warns(self):
        for sample, start in ((self.sample(9000.0, available=2500.0), None),
                              (self.sample(9000.0, available=2500.0), T.good_sample()),
                              (T.good_sample(available_mb=2500.0), self.sample(300.0))):
            dog = self.dog()
            self.assertIsNone(dog.check(sample, None, start))
            self.assertFalse(dog.last["compressor_warning"])

    def test_without_an_available_reading_growth_cannot_stop_it(self):
        dog = self.dog()
        self.assertIsNone(dog.check(T.good_sample(compressor_mb=5000.0), None, self.sample(300.0)))
        self.assertTrue(dog.last["compressor_warning"])

    def test_the_limits_are_configurable_through_the_policy(self):
        dog = self.dog(compressor_limit_mib=512.0, compressor_stop_available_mib=4000.0)
        self.assertIn("compressor grew", dog.check(self.sample(900.0, available=3900.0), None, self.sample(300.0)))
        self.assertIsNone(self.dog(compressor_limit_mib=512.0).check(
            self.sample(900.0, available=3900.0), None, self.sample(300.0)))

    def test_the_triggering_and_the_warning_samples_record_the_figures(self):
        dog = self.dog()
        dog.check(self.sample(3060.0, available=2500.0, pressure_level=1), None, self.sample(310.0))
        self.assertEqual((dog.last["compressor_mb"], dog.last["compressor_growth_mb"],
                          dog.last["available_mb"]), (3060.0, 2750.0, 2500.0))
        self.assertTrue(dog.last["compressor_warning"])

    # -- every independent safeguard still stops, with a compressor warning active ---------

    def warning_sample(self, **changes):
        return self.sample(3500.0, available=6000.0, **changes)

    def test_pressure_swap_battery_power_and_available_floor_still_stop_during_a_warning(self):
        start = self.sample(300.0)
        self.assertIn("pressure", self.dog().check(self.warning_sample(pressure_level=2), None, start))
        self.assertIn("pressure", self.dog().check(self.warning_sample(pressure_level=4), None, start))
        self.assertIn("swapout", self.dog().check(self.warning_sample(), {"swapout": 1}, start))
        self.assertIn("swap-in", self.dog().check(self.warning_sample(), {"swapin": 5000}, start))
        self.assertIn("Low Power", self.dog().check(self.warning_sample(low_power=True), None, start))
        self.assertIn("battery", self.dog().check(
            self.warning_sample(power_source="battery", battery_percent=20), None, start))
        self.assertIn("AC power", self.dog(final=True).check(
            self.warning_sample(power_source="battery", battery_percent=60), None, start))
        self.assertIn("available memory", self.dog().check(
            self.sample(3500.0, available=1900.0), None, start))

    def test_thermal_throttling_still_stops_during_a_warning(self):
        dog, start = self.dog(), self.sample(300.0)
        self.assertIsNone(dog.check(self.warning_sample(thermal_speed_limit=80), None, start))
        self.assertIn("thermal", dog.check(self.warning_sample(thermal_speed_limit=80), None, start))

    def test_the_metal_headroom_guard_is_independent_of_the_compressor_rule(self):
        row = {"available_mb": 8000.0, "mlx_active": H.METAL_LIMIT_BYTES - 300_000_000, "mlx_cache": 0,
               "metal_limit_bytes": H.METAL_LIMIT_BYTES, "compressor_growth_mb": 0.0}
        self.assertIn("Metal limit", H.guard_reason(row))
        self.assertIn("Metal limit", H.guard_reason({**row, "compressor_growth_mb": 3000.0}))

    def test_the_child_guard_follows_the_same_conditional_rule(self):
        row = {"available_mb": 6000.0, "mlx_active": GB, "mlx_cache": 0,
               "metal_limit_bytes": H.METAL_LIMIT_BYTES, "compressor_growth_mb": 2049.0}
        self.assertIsNone(H.guard_reason(row))                                            # warning only
        self.assertIn("compressor grew", H.guard_reason({**row, "available_mb": 3000.0}))
        self.assertIsNone(H.guard_reason({**row, "available_mb": 3000.0, "compressor_growth_mb": 2048.0}))
        self.assertIsNone(H.guard_reason({**row, "available_mb": 3000.0, "compressor_growth_mb": None}))
        self.assertIsNone(H.guard_reason({**row, "available_mb": 3000.0}, compressor_limit_mib=4096.0))
        self.assertIn("available", H.guard_reason({**row, "available_mb": 1900.0, "compressor_growth_mb": 0.0}))
        self.assertIsNotNone(H.guard_reason({**row, "available_mb": 3500.0},
                                            compressor_stop_available_mib=4096.0))

    # -- run_child and the trace ---------------------------------------------------------

    def drive(self, samples, arm_start, polls=100):
        clock = T.Clock()
        queue = list(samples)
        sampler = SimpleNamespace(sample=lambda: queue.pop(0) if len(queue) > 1 else queue[0])
        dog = H.Watchdog(H.SafetyPolicy(page_bytes=PAGE), arm_start or T.good_sample())
        return H.run_child(["x"], {}, timeout=600, sampler=sampler, watchdog=dog, arm_start=arm_start,
                           popen=lambda c, env: T.FakeProc(0, polls=polls), sleep=clock.sleep,
                           clock=clock, grace=1.0)

    def test_a_warning_lets_the_arm_continue_and_is_counted(self):
        samples = [self.sample(400.0), self.sample(3060.0), self.sample(3100.0)]
        result = self.drive(samples, self.sample(310.0), polls=4)
        self.assertEqual(result.status, "ok")
        summary = H.trace_summary(result.trace)
        over = [x for x in result.trace if x["compressor_growth_mb"] > 2048]
        self.assertGreaterEqual(len(over), 2)
        self.assertEqual(summary["compressor_warnings"], len(over))
        self.assertEqual(summary["max_compressor_growth_mb"], 2790.0)

    def test_growth_then_low_available_stops_and_marks_the_stop_sample(self):
        samples = [self.sample(400.0), self.sample(3060.0, available=6000.0),
                   self.sample(3300.0, available=2500.0)]
        result = self.drive(samples, self.sample(310.0))
        self.assertEqual(result.status, "stopped")
        self.assertEqual(len(result.trace), 3)
        self.assertIn("compressor", result.trace[-1]["stop"])
        self.assertTrue(result.trace[1]["compressor_warning"])
        self.assertNotIn("stop", result.trace[1])                  # the warning sample itself did not stop it
        self.assertEqual(result.trace[-1]["available_mb"], 2500.0)

    def test_run_child_without_an_arm_start_never_stops_on_compression(self):
        result = self.drive([self.sample(9000.0, available=2500.0)], None, polls=3)
        self.assertEqual(result.status, "ok")

    def test_the_sampler_reports_compressor_and_page_cache_from_one_read(self):
        answers = {("pmset", "-g", "batt"): T.BATT_AC, ("pmset", "-g", "therm"): T.THERM_OK,
                   ("pmset", "-g"): T.PMSET_NORMAL,
                   ("sysctl", "vm.swapusage"): "vm.swapusage: total = 2048.00M  used = 770.44M  free = 1277.56M",
                   ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1"}
        reading = {"available_mb": 7000.0, "compressor_mb": 310.0, "file_backed_mb": 6000.0}
        sample = H.Sampler(runner=lambda cmd: answers[tuple(cmd)], memory=lambda: reading).sample()
        self.assertEqual((sample["available_mb"], sample["compressor_mb"], sample["file_backed_mb"]),
                         (7000.0, 310.0, 6000.0))


class TracerGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.exits = []

    def tracer(self, probe, **kwargs):
        kwargs.setdefault("guard", lambda row: H.guard_reason(row))
        kwargs.setdefault("on_guard", self.exits.append)
        kwargs.setdefault("metal_limit", H.METAL_LIMIT_BYTES)
        tracer = B.MemoryTracer(probe, Path(self.directory.name) / "t.jsonl", **kwargs)
        self.addCleanup(tracer.stop)
        return tracer

    def growing(self, available):
        probe = FakeMachineProbe(available=available)
        probe.compressor, probe.compressor_step = 310.0, 1000.0
        return probe

    def test_the_tracer_baselines_the_compressor_at_its_first_reading(self):
        tracer = self.tracer(self.growing(8000.0), phase=B.Phase("prefill"))
        self.assertEqual(tracer.step()["compressor_growth_mb"], 0.0)
        self.assertEqual(tracer.step()["compressor_growth_mb"], 1000.0)

    def test_growth_with_ample_memory_warns_once_and_never_calls_the_guard(self):
        tracer = self.tracer(self.growing(8000.0), phase=B.Phase("prefill"), compressor_warn_mib=2048.0)
        for _ in range(6):
            tracer.step()                                   # growth 0, 1000, 2000, 3000, 4000, 5000
        self.assertEqual(self.exits, [])
        warnings = [r for r in tracer.rows if r.get("event") == "compressor_warning"]
        self.assertEqual(len(warnings), 1)                  # once, at the first sample over the limit
        self.assertEqual(warnings[0]["during"], "prefill")  # in the phase it happened in
        self.assertEqual(warnings[0]["growth_mb"], 3000.0)
        self.assertEqual(warnings[0]["available_mb"], 8000.0)

    def test_growth_with_less_than_three_gib_available_calls_the_guard_with_the_phase(self):
        probe = self.growing(2600.0)                        # above the 2 GB floor, below the 3 GiB one
        tracer = self.tracer(probe, phase=B.Phase("prefill"), compressor_warn_mib=2048.0)
        for _ in range(3):
            tracer.step()
        self.assertEqual(self.exits, [])                    # growth 2000: not over the limit yet
        tracer.step()                                       # growth 3000 with 2600 MiB available
        self.assertEqual(len(self.exits), 1)
        self.assertIn("compressor grew", self.exits[0])
        guard_row = [r for r in tracer.rows if "guard" in r][0]
        self.assertEqual(guard_row["during"], "prefill")
        sample_rows = [r for r in tracer.rows if r.get("phase")]
        self.assertEqual(sample_rows[-1]["compressor_growth_mb"], 3000.0)    # the triggering sample is recorded
        self.assertEqual(sample_rows[-1]["available_mb"], 2600.0)

    def test_low_available_without_growth_does_not_stop_between_the_floors(self):
        probe = FakeMachineProbe(available=2500.0)
        tracer = self.tracer(probe)
        for _ in range(4):
            tracer.step()
        self.assertEqual(self.exits, [])

    def test_a_fast_sample_reads_only_the_cheap_things(self):
        tracer = self.tracer(FakeMachineProbe(), phase=B.Phase("prefill"))
        full, fast = tracer.step(full=True), tracer.step(full=False)
        self.assertNotIn("fast", full)
        self.assertEqual(fast["fast"], True)
        for key in ("rss", "swap_used_mb", "pressure_level"):
            self.assertIsNotNone(full[key], key)
            self.assertIsNone(fast[key], key)
        for key in ("mlx_active", "mlx_cache", "mlx_peak", "available_mb", "compressor_mb",
                    "compressor_growth_mb", "swapout", "pool"):
            self.assertIn(key, fast)
            self.assertIsNotNone(fast[key], key)

    def test_the_guard_runs_on_fast_samples_too(self):
        tracer = self.tracer(FakeMachineProbe(available=900.0))
        tracer.step(full=False)
        self.assertEqual(len(self.exits), 1)

    def test_a_fast_sample_leaves_the_summary_correct(self):
        tracer = self.tracer(FakeMachineProbe(), phase=B.Phase("prefill"))
        tracer.step(full=True)
        tracer.step(full=False)
        summary = H.summarize_trace(tracer.rows)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["phases"]["prefill"]["available_mb"], 8000.0)
        self.assertEqual(summary["phases"]["prefill"]["rss"], 5 * GB)       # the fast row's None is skipped

    def test_during_prefill_the_thread_samples_fast_with_a_full_row_each_interval(self):
        tracer = self.tracer(FakeMachineProbe(), phase=B.Phase("prefill"), interval=0.2, fast_interval=0.01)
        tracer.start()
        time.sleep(0.5)
        tracer.stop()
        rows = [r for r in tracer.rows if r.get("phase")]
        fast, full = [r for r in rows if r.get("fast")], [r for r in rows if not r.get("fast")]
        self.assertGreater(len(fast), 4)                       # about 50 are expected; a loaded CI runner delivers far fewer
        self.assertGreaterEqual(len(full), 2)                # the first, then one about every 0.2 s
        self.assertLess(len(full), len(fast))

    def test_outside_prefill_the_thread_keeps_the_slow_interval(self):
        tracer = self.tracer(FakeMachineProbe(), phase=B.Phase("decode"), interval=0.1, fast_interval=0.01)
        tracer.start()
        time.sleep(0.45)
        tracer.stop()
        rows = [r for r in tracer.rows if r.get("phase")]
        self.assertFalse([r for r in rows if r.get("fast")])
        self.assertLess(len(rows), 8)

    def test_without_a_fast_interval_every_sample_is_full(self):
        tracer = self.tracer(FakeMachineProbe(), phase=B.Phase("prefill"), interval=0.02)
        tracer.start()
        time.sleep(0.15)
        tracer.stop()
        self.assertFalse([r for r in tracer.rows if r.get("fast")])


class InstrumentationTests(unittest.TestCase):
    def setUp(self):
        self.probe = T.FakeProbe()
        self.store = T.FakeStore()
        self.environ = {}
        self.env = T.FakeEnvironment(self.probe, self.store, self.environ)
        self.backend = T.FakeBackend(self.env, T.fake_language(self.store))
        self.prompts = {"system": "s", "first": "a", "later": "b"}

    def run_condition(self, name, phase, extra=None, events=None):
        cond = H.CONDITIONS[name]
        with B.Instrument(self.probe, self.env.prefill, self.env.ring, self.backend.language,
                          phase=phase, events=events) as inst:
            return B.run_scenario(self.backend, self.probe, inst, self.store, cond, self.prompts,
                                  128, self.environ, hasher=lambda cache: "kv", extra=extra)

    def test_the_phases_are_visited_in_order(self):
        names = []

        class Recording(B.Phase):
            def set(self, name):
                names.append(name)
                super().set(name)

        self.run_condition("F2", Recording())
        wanted = ["feed", "prefill", "ring-acquire", "ring-release", "post-prefill", "decode", "turn-end"]
        positions = [names.index(n) for n in wanted]
        self.assertEqual(positions, sorted(positions), names)
        self.assertEqual(names[-1], "turn-end")

    def test_the_phase_is_restored_after_acquire_and_release(self):
        seen = []
        phase = B.Phase()
        original = self.env._acquire

        def acquire(language, tokens):
            seen.append(("inside-acquire", phase.name))
            return original(language, tokens)

        self.env.ring.acquire = acquire
        self.run_condition("F2", phase)
        self.assertIn(("inside-acquire", "ring-acquire"), seen)
        self.assertEqual(phase.name, "turn-end")

    def test_an_instrument_without_a_phase_still_works(self):
        with B.Instrument(self.probe, self.env.prefill, self.env.ring, self.backend.language) as inst:
            turns = B.run_scenario(self.backend, self.probe, inst, self.store, H.CONDITIONS["F0"],
                                   self.prompts, 128, self.environ, hasher=lambda cache: "kv")
        self.assertEqual(turns[0]["prefilled_tokens"], 1000)

    def test_pool_creation_is_reported_with_its_size_and_the_class_is_restored(self):
        events = []

        def fake_init(pool, store, size_bytes, group_size=32, lock=False):
            pool.slots, pool._mm = 651, bytearray(10)

        with mock.patch.object(expert_pool.ExpertPool, "__init__", fake_init):
            phase = B.Phase("decode")
            with B.Instrument(self.probe, self.env.prefill, self.env.ring, self.backend.language,
                              phase=phase,
                              events=lambda name, **fields: events.append((name, fields))):
                self.assertIsNot(expert_pool.ExpertPool.__init__, fake_init)
                expert_pool.ExpertPool(None, 2_000_000_000)
            self.assertIs(expert_pool.ExpertPool.__init__, fake_init)
        self.assertEqual([name for name, _ in events], ["expert_pool_creating", "expert_pool_created"])
        self.assertEqual(events[0][1], {"size_bytes": 2_000_000_000})
        self.assertEqual(events[1][1], {"slots": 651, "mapped_bytes": 10})

    def test_the_pool_class_is_not_touched_without_an_event_sink(self):
        before = expert_pool.ExpertPool.__init__
        with B.Instrument(self.probe, self.env.prefill, self.env.ring, self.backend.language):
            self.assertIs(expert_pool.ExpertPool.__init__, before)

    def test_a_diagnostic_override_survives_the_overlay_that_unsets_managed_flags(self):
        extra = {FLAGS_MIN: "64"}
        self.run_condition("F2", B.Phase(), extra=extra)
        flags = self.env.seen[0][1]
        self.assertEqual(flags[FLAGS_MIN], "64")                     # set at the prefill call
        self.assertEqual(flags[T.FLAGS["ring"]], "512")              # the condition is untouched

    def test_a_second_turn_arm_gets_the_override_on_both_turns(self):
        self.run_condition("S1", B.Phase(), extra={FLAGS_MIN: "64"})
        self.assertEqual([flags.get(FLAGS_MIN) for _, flags in self.env.seen], ["64", "64"])

    def test_without_an_override_the_baseline_environment_is_unchanged(self):
        self.run_condition("F0", B.Phase())
        self.assertEqual(self.env.seen[0][1], {})


FLAGS_MIN = H.production_flags()["min_tokens"]


class ChildMainTests(unittest.TestCase):
    """``child_main`` end to end with a fake backend: no model, no GPU, ``os._exit`` mocked."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "results"
        for patcher in (mock.patch.object(results, "RESULTS_ROOT", self.root),
                        mock.patch.dict(os.environ, {}, clear=False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop(results.ENVIRONMENT_KEY, None)
        for name in list(os.environ):
            if name.startswith("FLASHNEXT_"):
                del os.environ[name]
        self.probe = FakeMachineProbe()
        self.store = T.FakeStore()
        self.exits = mock.patch.object(os, "_exit")
        self.exit = self.exits.start()
        self.addCleanup(self.exits.stop)

    def run_child(self, available=8000.0, delay=0.0, **changes):
        self.probe.available = available
        env = T.FakeEnvironment(self.probe, self.store, os.environ)
        backend = T.FakeBackend(env, T.fake_language(self.store))
        backend.tokenizer = None
        if delay:                           # keep the child alive long enough for several 10 ms samples
            opened = backend.open_conversation

            def slow_open(*a, **k):
                time.sleep(delay)
                return opened(*a, **k)

            backend.open_conversation = slow_open
        env.ring, env.prefill = prefill_ring, prefill
        target = self.root / "flashnext" / "child" / "arm-1k-F2-r1.json"
        args = SimpleNamespace(
            condition="F2", size="1k", decode=8, arm_id="1k-F2-r1", round=0, reverse=False,
            json=str(target), expert_pool_gb=None, ring_min_tokens=None,
            min_available_gb=2.0, trace_interval=0.01)
        for key, value in changes.items():
            setattr(args, key, value)
        prompts = {"system": "s", "first": "a", "later": "b", "source_hash": "h", "first_hash": "f",
                   "later_hash": "l", "first_ids": 988, "later_ids": 1001}
        patches = [
            mock.patch.object(B, "RealProbe", lambda: self.probe),
            mock.patch.object(B, "kv_digest", lambda cache: "kv"),
            mock.patch.object(H, "build_prompts", lambda *a, **k: prompts),
            mock.patch("veystrix.backends.flashnext.FlashNextBackend", lambda **k: backend),
            mock.patch("veystrix.checkpoints.resolve_flashnext", lambda name=None: Path("/models/ckpt")),
            mock.patch("models.flashnext.checkpoint_policy.resolve_resident_experts", lambda *a: 8),
            mock.patch("models.flashnext.routing.checkpoint_identity_for_store", lambda store: "identity"),
            mock.patch.object(prefill, "prefill_target", env._prefill),
            mock.patch.object(prefill_ring, "acquire", env._acquire),
            mock.patch.object(prefill_ring, "release", env._release),
            mock.patch.object(Path, "home", lambda: Path(self.directory.name)),
        ]
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            for patcher in patches:
                stack.enter_context(patcher)
            code = B.child_main(args)
        return code, target

    def test_a_prefix_cache_arm_runs_each_request_as_a_fresh_conversation(self):
        def reset(backend):
            backend.tape, backend.pending = [], []
            resets.append(1)

        resets = []
        with mock.patch.object(T.FakeBackend, "reset", reset, create=True), \
                mock.patch.object(T.FakeBackend, "last_prefix", None, create=True):
            code, target = self.run_child(
                condition="F2P", requests="ref,ref,ref", prefix_dir="/cache/prefix", dump_dir=None,
                prefix_min_tokens=16, prefix_ram_gb=0.75, prefix_disk_gb=5.0)
        self.assertEqual(code, 0)
        data = json.loads(target.read_text())
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["requests"], ["ref", "ref", "ref"])
        self.assertEqual([t["name"] for t in data["turns"]], ["1-ref", "2-ref", "3-ref"])
        self.assertEqual(len(resets), 2, "the conversation is reset between requests, not before the first")
        for turn in data["turns"]:
            self.assertEqual(turn["request"], "ref")
            self.assertEqual(turn["prompt_tokens"], 1000)
            self.assertEqual(len(turn["prompt_digest"]), 16)
        self.assertEqual(os.environ["FLASHNEXT_PREFIX_CACHE"], "1")
        self.assertEqual(os.environ["FLASHNEXT_PREFILL_CHUNK_TOKENS"], "512")
        self.assertEqual(os.environ["FLASHNEXT_EVAL_LINEAR_STATE"], "1")
        self.assertEqual(os.environ["FLASHNEXT_PREFIX_CACHE_DIR"], "/cache/prefix")
        self.assertEqual(os.environ["FLASHNEXT_PREFIX_CACHE_MIN_TOKENS"], "16")
        self.assertEqual(data["prefix_settings"]["FLASHNEXT_PREFIX_CACHE_DISK_GB"], "5")
        self.assertEqual(data["prefix_integrity"]["entries_changed_between_requests"], {})

    def test_a_real_prefix_report_object_is_logged_and_written_as_json(self):
        # The real backend sets ``last_prefix`` to a prefix_session.Report, not a dict. A real-model run
        # failed on this after its prefill and decode had finished, so the double must match the real type.
        from models.flashnext.prefix_session import Report

        def reset(backend):
            backend.tape, backend.pending = [], []

        report = Report(prompt_tokens=1000, hit_tokens=14, prefilled_tokens=986, tier="ram",
                        restore_seconds=0.01, snapshots=[14])
        with mock.patch.object(T.FakeBackend, "reset", reset, create=True), \
                mock.patch.object(T.FakeBackend, "last_prefix", report, create=True):
            code, target = self.run_child(
                condition="F2P", requests="ref,ref", prefix_dir="/cache/prefix", dump_dir=None,
                prefix_min_tokens=8, prefix_ram_gb=0.75, prefix_disk_gb=5.0)
        self.assertEqual(code, 0)
        data = json.loads(target.read_text())
        self.assertEqual(data["status"], "completed")
        for turn in data["turns"]:
            self.assertEqual(turn["prefix"]["hit_tokens"], 14)
            self.assertEqual(turn["prefix"]["snapshots"], [14])

    def test_a_failure_in_optional_reporting_cannot_lose_the_measured_turn(self):
        def boom(backend):
            raise RuntimeError("reporting broke")

        with mock.patch.object(B, "prefix_report", boom), \
                mock.patch.object(T.FakeBackend, "last_prefix", None, create=True):
            code, target = self.run_child(
                condition="F2P", requests="ref", prefix_dir="/cache/prefix", dump_dir=None,
                prefix_min_tokens=8, prefix_ram_gb=0.75, prefix_disk_gb=5.0)
        self.assertEqual(code, 0)
        data = json.loads(target.read_text())
        turn = data["turns"][0]
        self.assertEqual(turn["kv_hash"], "kv")
        self.assertEqual(len(turn["digest"]), 16)
        self.assertIsInstance(turn["generated_ids"], list)
        self.assertIn("reporting broke", turn["report_error"])

    def test_the_turn_is_on_disk_before_the_extras_run(self):
        seen = []

        path = self.root / "flashnext" / "child" / "arm-1k-F2-r1.json"

        def spy(backend):
            seen.append(json.loads(path.read_text())["turns"])
            return None

        with mock.patch.object(B, "prefix_report", spy), \
                mock.patch.object(T.FakeBackend, "last_prefix", None, create=True):
            code, target = self.run_child(
                condition="F2P", requests="ref", prefix_dir="/cache/prefix", dump_dir=None,
                prefix_min_tokens=8, prefix_ram_gb=0.75, prefix_disk_gb=5.0)
        self.assertEqual(code, 0)
        self.assertEqual(len(seen[0]), 1)
        self.assertIn("digest", seen[0][0])

    def test_the_later_request_keeps_the_system_message_and_changes_the_question(self):
        prompts = {"system": "S", "first": "Q1", "later": "Q2"}
        self.assertEqual(B.request_text("ref", prompts, None), ("S", "Q1"))
        self.assertEqual(B.request_text("later", prompts, None), ("S", "Q2"))

    def test_prefix_report_accepts_none_dict_and_report(self):
        from types import SimpleNamespace

        from models.flashnext.prefix_session import Report

        self.assertIsNone(B.prefix_report(SimpleNamespace()))
        self.assertIsNone(B.prefix_report(SimpleNamespace(last_prefix=None)))
        self.assertEqual(B.prefix_report(SimpleNamespace(last_prefix={"hit_tokens": 3})), {"hit_tokens": 3})
        self.assertEqual(B.prefix_report(SimpleNamespace(last_prefix=Report(prompt_tokens=5)))["prompt_tokens"], 5)

    def test_a_prefix_cache_arm_without_a_directory_refuses_to_start(self):
        with self.assertRaises(SystemExit):
            self.run_child(condition="F2P", requests="ref", prefix_dir=None, dump_dir=None,
                           prefix_min_tokens=16, prefix_ram_gb=0.75, prefix_disk_gb=5.0)

    def test_a_normal_run_writes_the_result_and_a_trace_with_every_phase(self):
        names = []

        class Recording(B.Phase):
            def set(self, name):
                names.append(name)
                super().set(name)

        with mock.patch.object(B, "Phase", Recording):
            code, target = self.run_child()
        self.assertEqual(code, 0)
        order = ["prompts", "feed", "prefill", "decode", "turn-end", "done"]
        positions = [names.index(n) for n in order]
        self.assertEqual(positions, sorted(positions), names)
        data = json.loads(target.read_text())
        self.assertEqual(data["status"], "completed")
        trace = [json.loads(line) for line in (target.parent / data["trace_file"]).read_text().splitlines()]
        phases = {row["phase"] for row in trace if row.get("phase")}
        self.assertTrue({"load", "done"} <= phases, phases)      # the first and last samples
        self.assertEqual(data["trace"]["file"], data["trace_file"])
        self.assertGreater(data["trace"]["samples"], 0)
        self.assertIsNone(data["trace"]["guard"])
        self.exit.assert_not_called()

    def test_the_diagnostic_overrides_are_applied_and_recorded(self):
        code, target = self.run_child(expert_pool_gb=2.0, ring_min_tokens=64)
        data = json.loads(target.read_text())
        self.assertEqual(os.environ["FLASHNEXT_EXPERT_POOL_GB"], "2")
        self.assertEqual(data["environment"]["FLASHNEXT_EXPERT_POOL_GB"], "2")
        self.assertEqual(data["diagnostic_overrides"]["expert_pool_gb"], 2.0)
        self.assertEqual(data["diagnostic_overrides"]["ring_min_tokens"], 64)
        self.assertEqual(os.environ[FLAGS_MIN], "64")
        self.assertEqual(data["turns"][0]["prefilled_tokens"], 1000)
        self.assertTrue(data["turns"][0]["ring_acquired"])

    def test_without_overrides_the_production_pool_size_is_kept(self):
        _code, target = self.run_child()
        data = json.loads(target.read_text())
        self.assertEqual(data["environment"]["FLASHNEXT_EXPERT_POOL_GB"], "6")
        self.assertIsNone(data["diagnostic_overrides"]["expert_pool_gb"])
        self.assertNotIn(FLAGS_MIN, os.environ)

    def test_low_available_memory_makes_the_child_leave_with_the_guard_code(self):
        _code, target = self.run_child(available=700.0)
        self.exit.assert_called_with(H.GUARD_EXIT_CODE)
        trace = [json.loads(line) for line in (target.parent / "arm-1k-F2-r1.trace.jsonl").read_text().splitlines()]
        guards = [row for row in trace if "guard" in row]
        self.assertTrue(guards)
        self.assertIn("available memory", guards[0]["guard"])

    def test_a_compressor_guard_leaves_a_guard_stop_file_then_exits_with_75(self):
        captured = {}
        target = self.root / "flashnext" / "child" / "arm-1k-F2-r1.json"

        def fake_exit(code):
            captured.setdefault("code", code)
            if "file" not in captured and target.exists():
                captured["file"] = json.loads(target.read_text())
                captured["trace"] = (target.parent / "arm-1k-F2-r1.trace.jsonl").read_text()

        self.exit.side_effect = fake_exit
        self.probe.compressor, self.probe.compressor_step = 310.0, 1000.0
        self.run_child(delay=0.4, available=2500.0, max_compressor_growth_gib=2.0)
        self.assertEqual(captured["code"], H.GUARD_EXIT_CODE)
        stop = captured["file"]
        self.assertEqual(stop["status"], "guard_stop")
        self.assertEqual(stop["classification"], "resource_stop")
        self.assertIn("compressor grew", stop["reason"])
        self.assertIsNotNone(stop["phase"])                         # the precise phase
        self.assertGreater(stop["last_sample"]["compressor_growth_mb"], 2048)
        self.assertEqual(stop["trace_file"], "arm-1k-F2-r1.trace.jsonl")
        rows = [json.loads(line) for line in captured["trace"].splitlines()]
        self.assertTrue(any("guard" in row for row in rows))        # flushed before the exit
        self.assertIn("compressor", [r for r in rows if "guard" in r][0]["guard"])

    def test_the_compressor_limit_reaches_the_child_and_is_recorded(self):
        _code, target = self.run_child(max_compressor_growth_gib=4.0)
        data = json.loads(target.read_text())
        self.assertEqual(data["diagnostic_overrides"]["max_compressor_growth_gib"], 4.0)

    def test_growth_over_the_limit_with_ample_memory_warns_but_the_child_completes(self):
        self.probe.compressor, self.probe.compressor_step = 310.0, 1000.0
        code, target = self.run_child(delay=1.5, available=8000.0)   # long enough for a slow runner to take the samples
        self.assertEqual(code, 0)
        self.exit.assert_not_called()
        data = json.loads(target.read_text())
        self.assertEqual(data["status"], "completed")
        events = [e["event"] for e in data["trace"]["events"]]
        self.assertEqual(events.count("compressor_warning"), 1)
        self.assertEqual(data["diagnostic_overrides"]["compressor_stop_available_gib"], 3.0)
        self.assertEqual(data["diagnostic_overrides"]["fast_trace_interval_s"], H.FAST_TRACE_SECONDS)

    def test_growth_inside_the_limit_does_not_stop_the_child(self):
        self.probe.compressor, self.probe.compressor_step = 310.0, 10.0
        self.run_child(delay=0.1)
        self.exit.assert_not_called()


class OrchestratedStopTests(unittest.TestCase):
    """The orchestrator's side: per-arm baselines, and a resource stop is not a failure."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "results"
        for patcher in (mock.patch.object(results, "RESULTS_ROOT", self.root),
                        mock.patch.dict(os.environ, {}, clear=False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop(results.ENVIRONMENT_KEY, None)
        self.launched = []

    def args(self, **changes):
        base = dict(sizes=["256"], conditions=["F0", "F2"], rounds=1, decode=128, interval=1.0,
                    settle=0.0, final=False, record=None, json=None,
                    max_compressor_growth_gib=2.0, min_available_gb=2.0, trace_interval=1.0,
                    expert_pool_gb=2.0, ring_min_tokens=64)
        base.update(changes)
        return SimpleNamespace(**base)

    def popen(self, codes):
        """Fake ``Popen``; ``codes`` gives each launch's exit code (0 writes the arm result)."""
        queue = list(codes)

        def start(command, env):
            self.launched.append(command)
            code = queue.pop(0)
            if code == 0:
                path = Path(command[command.index("--json") + 1])
                size = command[command.index("--size") + 1]
                condition = command[command.index("--condition") + 1]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(T.make_arm(size, condition, 0)))
            return T.FakeProc(code, polls=2)
        return start

    def run_it(self, args, sampler, popen):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = B.orchestrate(args, sampler=sampler, popen=popen, sleep=lambda s: None)
        return code, out.getvalue()

    def records(self):
        path = next(self.root.rglob("records.jsonl"))
        return [json.loads(line) for line in path.read_text().splitlines()]

    def sampler(self, compressors, available=6000.0):
        """Samples whose compressor follows ``compressors`` (the last value repeats)."""
        queue = list(compressors)
        return SimpleNamespace(sample=lambda: T.good_sample(
            compressor_mb=queue.pop(0) if len(queue) > 1 else queue[0], available_mb=available))

    def test_growth_over_two_gib_with_low_available_is_a_diagnostic_stop_and_f2_does_not_run(self):
        # start sample, pre-arm sample, then the child's first watchdog sample 2.75 GB higher, 2.5 GiB available
        queue = [(310.0, 6000.0), (310.0, 6000.0), (3060.0, 2500.0)]
        sampler = self.stepped(queue)
        code, text = self.run_it(self.args(), sampler, self.popen([0, 0]))
        self.assertEqual(code, 3)
        self.assertEqual(len(self.launched), 1)                      # F2 was never launched
        failure = next(r for r in self.records() if r["type"] == "failure")
        self.assertEqual(failure["details"]["classification"], "resource_stop")
        self.assertIn("compressor grew", failure["message"])
        self.assertEqual(failure["details"]["watchdog"]["max_compressor_growth_mb"], 2750.0)
        summary = self.records()[-1]
        self.assertEqual((summary["status"], summary["summary"]["classification"]), ("stopped", "resource_stop"))
        self.assertIn("diagnostic stop, not a correctness failure", text)

    def stepped(self, pairs):
        """Samples that follow (compressor, available) pairs; the last pair repeats."""
        queue = list(pairs)

        def sample():
            compressor, available = queue.pop(0) if len(queue) > 1 else queue[0]
            return T.good_sample(compressor_mb=compressor, available_mb=available)
        return SimpleNamespace(sample=sample)

    def test_the_same_growth_with_ample_memory_is_a_warning_and_both_arms_complete(self):
        sampler = self.stepped([(310.0, 6000.0), (310.0, 6000.0), (3060.0, 5000.0)])
        code, text = self.run_it(self.args(), sampler, self.popen([0, 0]))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.launched), 2)
        arms = [r for r in self.records() if r["type"] == "arm"]
        self.assertGreater(arms[0]["watchdog"]["compressor_warnings"], 0)
        self.assertEqual(arms[0]["watchdog"]["max_compressor_growth_mb"], 2750.0)

    def test_a_high_but_steady_compressor_does_not_stop_either_arm(self):
        code, _ = self.run_it(self.args(), self.sampler([3000.0]), self.popen([0, 0]))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.launched), 2)

    def test_the_child_memory_guard_exit_is_a_resource_stop_too(self):
        code, text = self.run_it(self.args(), self.sampler([310.0]), self.popen([H.GUARD_EXIT_CODE]))
        self.assertEqual(code, 3)
        failure = next(r for r in self.records() if r["type"] == "failure")
        self.assertEqual((failure["error_type"], failure["details"]["classification"]), ("memory_guard", "resource_stop"))
        self.assertIn("diagnostic stop", text)

    def test_a_plain_crash_is_not_called_a_resource_stop(self):
        code, text = self.run_it(self.args(), self.sampler([310.0]), self.popen([1, 0]))
        failure = next(r for r in self.records() if r["type"] == "failure")
        self.assertEqual(failure["details"]["classification"], "failed")
        self.assertNotIn("diagnostic stop", text)

    def test_the_child_trace_summary_names_the_phase_at_the_stop(self):
        def popen(command, env):
            self.launched.append(command)
            path = Path(command[command.index("--json") + 1])
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {"t": 0.0, "phase": "load", "available_mb": 9000.0, "compressor_growth_mb": 0.0},
                {"t": 4.0, "phase": "prefill", "available_mb": 5000.0, "compressor_growth_mb": 2500.0},
                {"t": 4.0, "phase": None, "guard": "compressor grew 2.44 GiB", "during": "prefill"},
            ]
            (path.parent / f"{path.stem}.trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            return T.FakeProc(H.GUARD_EXIT_CODE, polls=2)

        self.run_it(self.args(), self.sampler([310.0]), popen)
        failure = next(r for r in self.records() if r["type"] == "failure")
        child = failure["details"]["child_trace"]
        self.assertEqual(child["last_sample"]["phase"], "prefill")
        self.assertIn("compressor", child["guard"])
        self.assertEqual(child["phases"]["prefill"]["compressor_growth_mb"], 2500.0)

    def test_the_compressor_limit_is_passed_to_the_children_and_recorded(self):
        self.run_it(self.args(max_compressor_growth_gib=2.0), self.sampler([3000.0]), self.popen([0, 0]))
        command = self.launched[0]
        self.assertEqual(command[command.index("--max-compressor-growth-gib") + 1], "2")
        run = self.records()[0]
        self.assertEqual(run["metadata"]["diagnostic_overrides"]["max_compressor_growth_gib"], 2.0)


class SmallRingTests(unittest.TestCase):
    """F2s: F2 with a 256-expert ring, to isolate the scratch pool's size."""

    def test_the_condition_differs_from_f2_only_in_ring_capacity(self):
        f2, small = H.CONDITIONS["F2"], H.ALL_CONDITIONS["F2s"]
        self.assertEqual((small.ring, small.scratch, small.cache_mb), (256, True, 0))
        self.assertEqual((f2.ring, f2.scratch, f2.cache_mb), (512, True, 0))
        self.assertEqual((small.turn, small.guarded, small.uncached_reads), (f2.turn, f2.guarded, False))
        self.assertFalse(small.is_baseline)

    def test_it_is_diagnostic_only_and_the_planned_matrix_is_unchanged(self):
        self.assertNotIn("F2s", H.CONDITIONS)
        self.assertEqual(set(H.CONDITIONS), {"F0", "F1", "F2", "F3a", "F3b", "F4", "S0", "S1", "S2"})

    def test_it_sets_the_production_ring_flag_to_256_and_no_cache_cap(self):
        overlay = H.condition_overlay(H.ALL_CONDITIONS["F2s"])
        self.assertEqual(overlay[prefill_ring.FLAG_EXPERTS], "256")
        self.assertEqual(overlay[prefill_ring.FLAG_SCRATCH], "1")
        self.assertIsNone(overlay[H.production_flags()["cache_mb"]])        # the MLX cache limit is untouched
        self.assertIsNone(overlay[prefill_ring.FLAG_MIN_TOKENS])            # production minimum (512) is kept
        env = H.apply_overlay({}, overlay)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(prefill_ring.configured_experts(), 256)
            self.assertTrue(prefill_ring.scratch_enabled())
            self.assertEqual(prefill_ring.min_tokens(), 512)

    def test_the_scratch_pool_is_about_half_the_size(self):
        from models.flashnext import expert_pool, pool_ring

        stride, expert_bytes = expert_pool.get_slab_layout(32).record_stride, 3_072_000
        sizes = {}
        for half in (512, 256):
            slots = pool_ring.slots_per_half(half, expert_bytes, stride) + pool_ring.MIN_DECODE_SLOTS
            sizes[half] = prefill_ring.pool_length(expert_pool.HEADER + stride * slots, stride)
        self.assertEqual(sizes[512], 1_622_032_384)                  # the mapped size observed in every F2 run
        self.assertAlmostEqual(sizes[256] / 1e9, 0.836, places=3)
        self.assertLess(sizes[256], 0.52 * sizes[512])

    def test_a_layer_that_routes_more_experts_than_the_ring_holds_falls_back_not_fails(self):
        # stage() counts a layer with more experts than the half as a capacity miss and returns None
        # (the caller then uses the existing read path), so a smaller ring is safe but serves fewer layers
        ring = SimpleNamespace(half_experts=256)
        state = prefill_ring.PrefillRing.__new__(prefill_ring.PrefillRing)
        state.ring, state.disabled, state._stats = ring, None, {"fallback_capacity": 0, "fallback_disabled": 0}
        state._retire = lambda: None
        self.assertIsNone(state.stage(None, "p", list(range(334))))
        self.assertEqual(state._stats["fallback_capacity"], 1)

    def test_it_can_be_scheduled_and_is_not_in_the_default_run(self):
        arms = H.build_schedule(["1k"], ["F2s"], 1)
        self.assertEqual([a.arm_id for a in arms], ["1k-F2s-r1"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            B.main(["--dry-run", "--sizes", "1k", "--rounds", "1"])
        self.assertNotIn("F2s", out.getvalue())
        self.assertIn("FLASHNEXT_PREFILL_RING_EXPERTS': '256'", self._dry("F2s"))

    def _dry(self, condition):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            B.main(["--dry-run", "--sizes", "1k", "--conditions", condition, "--rounds", "1"])
        return out.getvalue()


class UncachedControlTests(unittest.TestCase):
    def test_the_control_is_diagnostic_only_and_not_in_the_planned_matrix(self):
        self.assertIn("F0u", H.DIAGNOSTIC_CONDITIONS)
        self.assertNotIn("F0u", H.CONDITIONS)
        self.assertIn("F0u", H.ALL_CONDITIONS)
        self.assertEqual(set(H.CONDITIONS), {"F0", "F1", "F2", "F3a", "F3b", "F4", "S0", "S1", "S2"})

    def test_the_control_sets_no_environment_flag_so_the_production_baseline_is_untouched(self):
        control = H.condition_overlay(H.ALL_CONDITIONS["F0u"])
        self.assertEqual(control, H.condition_overlay(H.CONDITIONS["F0"]))
        self.assertTrue(all(value is None for value in control.values()))
        self.assertEqual(H.launch_overlay(H.ALL_CONDITIONS["F0u"]), H.launch_overlay(H.CONDITIONS["F0"]))

    def test_the_control_is_not_a_baseline_and_the_baseline_still_is(self):
        self.assertFalse(H.ALL_CONDITIONS["F0u"].is_baseline)
        self.assertTrue(H.CONDITIONS["F0"].is_baseline)
        self.assertEqual(H.BASELINES["first"], "F0")

    def test_the_default_conditions_do_not_include_the_control(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            B.main(["--dry-run", "--sizes", "1k", "--rounds", "1"])
        self.assertNotIn("F0u", out.getvalue())

    def test_the_control_can_be_scheduled_beside_f0_and_f2(self):
        arms = H.build_schedule(["256"], ["F0", "F0u", "F2"], 1)
        self.assertEqual([a.condition for a in arms], ["F0", "F0u", "F2"])
        with self.assertRaises(ValueError):
            H.build_schedule(["256"], ["F9"], 1)

    def test_it_sets_nocache_on_open_and_later_descriptors_of_this_store_only(self):
        import fcntl

        class Store:
            def __init__(self, first):
                self._fds = {"a": first}
                self.opened = []

            def _fd(self, shard):
                if shard not in self._fds:
                    self._fds[shard] = 200 + len(self.opened)
                    self.opened.append(shard)
                return self._fds[shard]

        calls = []
        store, untouched = Store(100), Store(300)
        with mock.patch.object(fcntl, "fcntl", lambda fd, cmd, arg=0: calls.append((fd, cmd, arg)) or 0):
            self.assertEqual(B.make_reads_uncached(store), 1)
            self.assertEqual(calls, [(100, fcntl.F_NOCACHE, 1)])        # already open: set
            self.assertEqual(store._fd("b"), 200)
            self.assertEqual(calls[-1], (200, fcntl.F_NOCACHE, 1))      # opened afterwards: set
            self.assertEqual(store._fd("b"), 200)
            self.assertEqual(len(calls), 2)                              # a known descriptor is not re-set
            self.assertEqual(untouched._fd("a"), 300)
            self.assertEqual(untouched._fd("z"), 200)                    # other store: original behaviour
            self.assertEqual(len(calls), 2)

    def test_the_report_names_any_descriptor_the_flag_did_not_reach(self):
        import fcntl

        store = SimpleNamespace(_fds={"a": 100}, _fd=None)
        store._fd = lambda shard: store._fds.setdefault(shard, 200 + len(store._fds))
        with mock.patch.object(fcntl, "fcntl", lambda *a: 0):
            B.make_reads_uncached(store)
            store._fd("b")
            clean = B.uncached_report(store)
            store._fds["c"] = 999                     # a descriptor opened behind the wrapper's back
            leaky = B.uncached_report(store)
        self.assertEqual((clean["descriptors"], clean["applied"], clean["unprotected"]), (2, 2, []))
        self.assertEqual((leaky["descriptors"], leaky["applied"], leaky["unprotected"]), (3, 2, ["c"]))
        self.assertEqual(leaky["shards"], ["a", "b", "c"])

    def test_the_report_of_an_untouched_store_shows_everything_unprotected(self):
        store = SimpleNamespace(_fds={"a": 5, "b": 6})
        report = B.uncached_report(store)
        self.assertEqual((report["applied"], report["unprotected"]), (0, ["a", "b"]))

    def test_a_real_descriptor_accepts_the_flag(self):
        import fcntl

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "shard.bin"
        path.write_bytes(b"x" * 64)
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        store = SimpleNamespace(_fds={"s": fd}, _fd=lambda shard: fd)
        try:
            B.make_reads_uncached(store)            # raises OSError if the OS refuses F_NOCACHE
        except OSError as error:
            self.fail(f"F_NOCACHE refused: {error}")
        self.assertEqual(os.pread(fd, 8, 0), b"x" * 8)            # and reads still work

    def test_the_production_store_class_is_not_modified(self):
        from models.flashnext import store as production_store

        before = production_store.SafeTensorStore._fd
        fake = SimpleNamespace(_fds={}, _fd=lambda shard: -1)
        B.make_reads_uncached(fake)
        self.assertIs(production_store.SafeTensorStore._fd, before)

    def test_the_child_applies_the_control_only_when_asked_and_records_it(self):
        class FdStore(T.FakeStore):
            def __init__(self):
                self._fds = {"s": -1}

            def _fd(self, shard):
                return -1

        for condition, expected in (("F0u", {"descriptors_open_at_start": 1}), ("F0", None)):
            with self.subTest(condition=condition), \
                    mock.patch.object(B, "make_reads_uncached", return_value=1) as patched, \
                    mock.patch.object(B, "uncached_report", return_value={
                        "descriptors": 1, "applied": 1, "unprotected": [], "shards": ["s"]}):
                directory = tempfile.TemporaryDirectory()
                self.addCleanup(directory.cleanup)
                holder = ChildMainTests("test_a_normal_run_writes_the_result_and_a_trace_with_every_phase")
                holder.setUp()
                self.addCleanup(holder.doCleanups)
                holder.store = FdStore()
                _code, target = holder.run_child(condition=condition)
                data = json.loads(target.read_text())
                if expected is None:
                    self.assertIsNone(data["uncached_reads"])
                else:
                    self.assertEqual(data["uncached_reads"]["descriptors_open_at_start"], 1)
                    self.assertTrue(data["uncached_reads"]["verified"])
                    self.assertEqual(data["uncached_reads"]["unprotected"], [])
                    events = [json.loads(line) for line in
                              (target.parent / data["trace_file"]).read_text().splitlines()]
                    self.assertEqual([e["event"] for e in events if str(e.get("event", "")).startswith("uncached")],
                                     ["uncached_reads_applied", "uncached_reads_final"])
                self.assertEqual(patched.called, expected is not None)


class OptionTests(unittest.TestCase):
    def args(self, **changes):
        base = dict(sizes=["128"], conditions=["F0", "F2"], rounds=1, decode=128, final=False,
                    checkpoint=None, expert_pool_gb=None, ring_min_tokens=None,
                    min_available_gb=H.MIN_AVAILABLE_GB, trace_interval=1.0)
        base.update(changes)
        return SimpleNamespace(**base)

    def test_the_children_always_get_the_guard_and_trace_settings(self):
        out = B.diagnostic_args(self.args())
        self.assertEqual(out, ["--min-available-gb", "2", "--max-compressor-growth-gib", "2",
                               "--compressor-stop-available-gib", "3",
                               "--trace-interval", "1", "--fast-trace-interval", "0.25",
                               "--requests", "ref", "--prefix-min-tokens", "16",
                               "--prefix-ram-gb", "0.75", "--prefix-disk-gb", "5"])

    def test_prefix_cache_options_reach_the_child_only_when_given(self):
        out = B.diagnostic_args(self.args(prefix_dir="/cache/d", dump_dir="/dumps", requests="long:A,long:B"))
        self.assertEqual(out[out.index("--requests") + 1], "long:A,long:B")
        self.assertEqual(out[out.index("--prefix-dir") + 1], "/cache/d")
        self.assertEqual(out[out.index("--dump-dir") + 1], "/dumps")
        self.assertNotIn("--prefix-dir", B.diagnostic_args(self.args()))

    def test_the_overrides_reach_the_child_command(self):
        arm = H.build_schedule(["128"], ["F0"], 1)[0]
        command = B.child_command(arm, 128, Path("/r/arm.json"),
                                  B.diagnostic_args(self.args(expert_pool_gb=2.0, ring_min_tokens=64)))
        self.assertEqual(command[command.index("--expert-pool-gb") + 1], "2")
        self.assertEqual(command[command.index("--ring-min-tokens") + 1], "64")
        self.assertEqual(command[command.index("--size") + 1], "128")

    def test_child_command_without_extras_is_unchanged(self):
        arm = H.build_schedule(["1k"], ["F0"], 1)[0]
        self.assertNotIn("--expert-pool-gb", B.child_command(arm, 128, Path("/r/arm.json")))

    def test_the_parser_accepts_the_options_and_the_short_sizes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = B.main(["--dry-run", "--sizes", "128", "256", "--conditions", "F0", "F2", "--rounds", "1",
                           "--expert-pool-gb", "2", "--ring-min-tokens", "64", "--min-available-gb", "2.5"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("4 fresh-process arms", text)
        self.assertIn("'expert_pool_gb': 2.0", text)
        self.assertIn("--expert-pool-gb 2", text)
        self.assertIn("--min-available-gb 2.5", text)

    def test_the_planned_matrix_is_unchanged_by_the_short_sizes(self):
        self.assertEqual(set(H.SIZES), {"1k", "2k", "4k"})
        self.assertEqual(set(H.DIAGNOSTIC_SIZES), {"128", "256"})
        self.assertEqual(set(H.ALL_SIZES), {"128", "256", "1k", "2k", "4k"})

    def test_every_size_has_a_time_limit_and_a_position(self):
        for size in H.ALL_SIZES:
            self.assertIn(size, H.TIME_LIMIT_SECONDS, size)
            self.assertIn(size, B.SIZE_ORDER, size)
        self.assertEqual(sorted(B.SIZE_ORDER, key=B.SIZE_ORDER.get), ["128", "256", "1k", "2k", "4k"])

    def test_short_sizes_come_first_in_the_schedule(self):
        arms = H.build_schedule(["1k", "256", "128"], ["F0", "F2"], 1)
        self.assertEqual([a.size for a in arms], ["128", "128", "256", "256", "1k", "1k"])

    def test_short_prompts_are_sized_from_the_template_overhead(self):
        class Tokenizer:
            def encode(self, text):
                return list(range(40000)) if len(text) > 100 else [1, 2]

            def decode(self, ids):
                return "x" * len(ids)

        prompts = H.build_prompts(Tokenizer(), "y" * 200, 128)
        self.assertEqual((prompts["first_ids"], prompts["later_ids"]),
                         (128 - H.TEMPLATE_OVERHEAD_FIRST, 128 - H.TEMPLATE_OVERHEAD_LATER))

    def test_the_hard_coded_ring_minimum_matches_production(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(prefill_ring.FLAG_MIN_TOKENS, None)
            self.assertEqual(prefill_ring.min_tokens(), B.PRODUCTION_RING_MIN_TOKENS)


class ReadinessDiagnosticsTests(unittest.TestCase):
    def ready(self, sizes_rows, **changes):
        args = T.ready_args(sizes=[row["size"] for row in sizes_rows], conditions=["F0", "F2"], rounds=1,
                            expert_pool_gb=None, ring_min_tokens=None,
                            min_available_gb=H.MIN_AVAILABLE_GB, trace_interval=1.0)
        for key, value in changes.items():
            setattr(args, key, value)
        with mock.patch.object(B, "prompt_sizes", return_value=sizes_rows), \
                mock.patch("veystrix.checkpoints.resolve_flashnext", return_value=Path("/models/ckpt")):
            return B.check_ready(
                args, sampler=SimpleNamespace(sample=lambda: T.good_sample()),
                preflight_module=T.ready_preflight(), tokenizer_loader=lambda p: T.CharTokenizer(),
                contract=lambda: [], environ={}, free_gb=8.0)

    ROW_128 = [{"size": "128", "target": 128, "first": 128, "later": 128}]
    ROW_1K = [{"size": "1k", "target": 1024, "first": 1025, "later": 1024}]

    def status(self, report, name):
        return next(c.status for c in report.checks if c.name == name)

    def test_a_short_f2_run_warns_that_the_ring_would_not_activate(self):
        report = self.ready(self.ROW_128)
        self.assertEqual(self.status(report, "ring activation"), "WARN")
        detail = next(c.detail for c in report.checks if c.name == "ring activation")
        self.assertIn("short prompt", detail)
        self.assertIn("--ring-min-tokens", detail)

    def test_lowering_the_ring_minimum_clears_the_warning(self):
        report = self.ready(self.ROW_128, ring_min_tokens=64)
        self.assertEqual(self.status(report, "ring activation"), "PASS")
        self.assertEqual(self.status(report, "diagnostic overrides"), "INFO")

    def test_a_1k_run_needs_no_override(self):
        report = self.ready(self.ROW_1K)
        self.assertEqual(self.status(report, "ring activation"), "PASS")
        self.assertNotIn("diagnostic overrides", [c.name for c in report.checks])

    def test_a_baseline_only_run_has_no_ring_to_warn_about(self):
        args = T.ready_args(sizes=["128"], conditions=["F0"], rounds=1, expert_pool_gb=None,
                            ring_min_tokens=None, min_available_gb=2.0, trace_interval=1.0)
        with mock.patch.object(B, "prompt_sizes", return_value=self.ROW_128), \
                mock.patch("veystrix.checkpoints.resolve_flashnext", return_value=Path("/models/ckpt")):
            report = B.check_ready(
                args, sampler=SimpleNamespace(sample=lambda: T.good_sample()),
                preflight_module=T.ready_preflight(), tokenizer_loader=lambda p: T.CharTokenizer(),
                contract=lambda: [], environ={}, free_gb=8.0)
        self.assertNotIn("ring activation", [c.name for c in report.checks])

    def test_low_available_memory_blocks_the_run(self):
        sampler = SimpleNamespace(sample=lambda: T.good_sample(available_mb=1200.0))
        args = T.ready_args(sizes=["1k"], conditions=["F0", "F2"], rounds=1, expert_pool_gb=None,
                            ring_min_tokens=None, min_available_gb=2.0, trace_interval=1.0)
        with mock.patch.object(B, "prompt_sizes", return_value=self.ROW_1K), \
                mock.patch("veystrix.checkpoints.resolve_flashnext", return_value=Path("/models/ckpt")):
            report = B.check_ready(
                args, sampler=sampler, preflight_module=T.ready_preflight(),
                tokenizer_loader=lambda p: T.CharTokenizer(), contract=lambda: [], environ={}, free_gb=8.0)
        self.assertEqual(report.verdict, "BLOCKED")
        self.assertEqual(self.status(report, "machine state"), "FAIL")


class PrefixConditionTests(unittest.TestCase):
    """F2P is F2L plus 512-token chunks and the prefix cache, and only those."""

    def test_f2p_overlay(self):
        overlay = H.condition_overlay(H.ALL_CONDITIONS["F2P"])
        self.assertEqual(overlay["FLASHNEXT_PREFILL_CHUNK_TOKENS"], "512")
        self.assertEqual(overlay["FLASHNEXT_PREFIX_CACHE"], "1")
        self.assertEqual(overlay["FLASHNEXT_EVAL_LINEAR_STATE"], "1")
        self.assertEqual(overlay["FLASHNEXT_PREFILL_RING_EXPERTS"], "512")
        self.assertEqual(overlay["FLASHNEXT_PREFILL_RING_SCRATCH"], "1")

    def test_no_other_condition_turns_them_on(self):
        for name, condition in H.ALL_CONDITIONS.items():
            overlay = H.condition_overlay(condition)
            for flag in ("FLASHNEXT_PREFILL_CHUNK_TOKENS", "FLASHNEXT_PREFIX_CACHE"):
                self.assertIn(flag, overlay)
                chunked = name in ("F2P", "F2B", "F2Pn", "F2Bn", "F2Bnr", "F2Br", "F2Pnr", "F2Bm", "F2Bmu")
                self.assertEqual(overlay[flag] is not None, chunked, (name, flag))

    def test_f2b_sets_the_balanced_2048_planner_and_only_f2b_sets_the_policy(self):
        overlay = H.condition_overlay(H.ALL_CONDITIONS["F2B"])
        self.assertEqual(overlay["FLASHNEXT_PREFILL_CHUNK_TOKENS"], "2048")
        self.assertEqual(overlay["FLASHNEXT_PREFILL_CHUNK_POLICY"], "balanced")
        self.assertEqual(overlay["FLASHNEXT_PREFIX_CACHE"], "1")
        self.assertEqual(overlay["FLASHNEXT_PREFILL_RING_EXPERTS"], "512")
        for name, condition in H.ALL_CONDITIONS.items():
            value = H.condition_overlay(condition)["FLASHNEXT_PREFILL_CHUNK_POLICY"]
            self.assertEqual(value is not None, name in ("F2B", "F2Bn", "F2Bnr", "F2Br", "F2Bm", "F2Bmu"), name)
        self.assertIn("F2B", H.DIAGNOSTIC_CONDITIONS)
        self.assertFalse(H.ALL_CONDITIONS["F2B"].is_baseline)

    def test_same_plan_references_differ_from_cached_conditions_only_by_the_cache(self):
        for cached, plain in (("F2P", "F2Pn"), ("F2B", "F2Bn")):
            a = H.condition_overlay(H.ALL_CONDITIONS[plain])
            b = H.condition_overlay(H.ALL_CONDITIONS[cached])
            differing = {k for k in b if a.get(k) != b.get(k)}
            self.assertEqual(differing, {"FLASHNEXT_PREFIX_CACHE_RESTORE"}, (cached, plain))
            self.assertEqual(a["FLASHNEXT_PREFIX_CACHE"], "1")        # same session, so the same chunk plan
            self.assertEqual(a["FLASHNEXT_PREFIX_CACHE_RESTORE"], "0")
            self.assertIsNone(b["FLASHNEXT_PREFIX_CACHE_RESTORE"])
            self.assertFalse(H.ALL_CONDITIONS[plain].is_baseline)

    def test_only_f2bnr_releases_the_decode_pool_and_differs_from_f2bn_by_that_flag_alone(self):
        flag = "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL"
        for name, condition in H.ALL_CONDITIONS.items():
            overlay = H.condition_overlay(condition)
            self.assertIn(flag, overlay, name)
            self.assertEqual(overlay[flag] is not None, name in ("F2Bnr", "F2Br", "F2Pnr"), name)
        a, b = H.condition_overlay(H.ALL_CONDITIONS["F2Bn"]), H.condition_overlay(H.ALL_CONDITIONS["F2Bnr"])
        self.assertEqual({k for k in b if a.get(k) != b.get(k)}, {flag})
        c = H.condition_overlay(H.ALL_CONDITIONS["F2Br"])               # the cached run: only the cache flag differs
        self.assertEqual({k for k in c if b.get(k) != c.get(k)}, {"FLASHNEXT_PREFIX_CACHE_RESTORE"})
        p = H.condition_overlay(H.ALL_CONDITIONS["F2Pnr"])               # the quality reference: 512 grid, release on
        self.assertEqual((p["FLASHNEXT_PREFILL_CHUNK_TOKENS"], p["FLASHNEXT_PREFILL_CHUNK_POLICY"]), ("512", None))
        self.assertEqual(b["FLASHNEXT_PREFIX_CACHE_RESTORE"], "0")     # B stays a full cold prefill, same plan
        self.assertEqual((b["FLASHNEXT_PREFILL_CHUNK_TOKENS"], b["FLASHNEXT_PREFILL_CHUNK_POLICY"]), ("2048", "balanced"))
        self.assertEqual((b["FLASHNEXT_PREFILL_RING_EXPERTS"], b["FLASHNEXT_PREFILL_RING_SCRATCH"]), ("512", "1"))

    def test_managed_conditions_differ_from_f2b_only_by_the_manager_flags(self):
        flags = ("FLASHNEXT_MEMORY_MANAGER", "FLASHNEXT_MEMORY_SHORT_UNCACHED")
        for name, condition in H.ALL_CONDITIONS.items():
            overlay = H.condition_overlay(condition)
            for flag in flags:
                self.assertIn(flag, overlay, name)
            self.assertEqual(overlay[flags[0]] is not None, name in ("F2Bm", "F2Bmu"), name)
            self.assertEqual(overlay[flags[1]] is not None, name == "F2Bmu", name)
        base = H.condition_overlay(H.ALL_CONDITIONS["F2B"])
        for name, extra in (("F2Bm", {flags[0]}), ("F2Bmu", set(flags))):
            overlay = H.condition_overlay(H.ALL_CONDITIONS[name])
            self.assertEqual({k for k in overlay if overlay.get(k) != base.get(k)}, extra, name)

    def test_f2p_is_a_diagnostic_and_not_a_baseline(self):
        self.assertIn("F2P", H.DIAGNOSTIC_CONDITIONS)
        self.assertFalse(H.ALL_CONDITIONS["F2P"].is_baseline)


class LongPromptTests(unittest.TestCase):
    def test_a_fake_tokenizer_gets_a_long_system_message_and_two_questions(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))

            def decode(self, ids):
                return " ".join(f"w{i}" for i in ids)

            def apply_chat_template(self, messages, **kwargs):
                return " ".join(m["content"] for m in messages) + " <|im_start|>assistant"

        source = " ".join(f"s{i}" for i in range(40_000))
        with mock.patch("veystrix.conversation.Conversation.open_conversation", autospec=True,
                        side_effect=lambda self, system, user, **kw: len(system.split()) + len(user.split())):
            result = H.build_long_prompts(Tok(), source, tokens=1025)
        self.assertEqual(set(result["questions"]), {"A", "B"})
        self.assertNotEqual(result["questions"]["A"], result["questions"]["B"])
        self.assertEqual(result["tokens"]["A"], 1025)
        self.assertTrue(result["system"].startswith(H.SYSTEM_TEXT))

    def test_a_source_that_is_too_short_is_refused(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [1] * 100

        with self.assertRaises(ValueError):
            H.build_long_prompts(Tok(), "x", tokens=1025)


class AgentPromptTests(unittest.TestCase):
    class Tok:
        def __init__(self):
            self.vocab = {}

        def encode(self, text, add_special_tokens=False):
            return [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]

    def build(self):
        tok = self.Tok()

        def open_conversation(conversation, system, user, **kwargs):
            conversation.pending = tok.encode(
                f"<|im_start|>system {system} <|im_end|> <|im_start|>user {user} <|im_end|> <|im_start|>assistant")
            return len(conversation.pending)

        with mock.patch("veystrix.conversation.Conversation.open_conversation", autospec=True,
                        side_effect=open_conversation):
            return H.build_agent_prompts(tok)

    def test_the_fixture_is_a_real_coding_agent_prompt_without_template_markers(self):
        text = H.AGENT_PROMPT_FILE.read_text()
        self.assertGreater(len(text.split()), 1400)
        self.assertNotIn("<|", text)
        for needle in ("tool", "git", "test", "secret"):
            self.assertIn(needle, text.lower())

    def test_counts_are_measured_per_question_and_split_at_the_system_boundary(self):
        result = self.build()
        system_words = len(result["system"].split())
        for name in ("A", "B"):
            counts = result["token_counts"][name]
            self.assertEqual(counts["system_message_tokens"], system_words + 2)      # '<|im_start|>system' is one word here, plus the end marker
            self.assertEqual(counts["prompt_tokens"],
                             counts["system_message_tokens"] + counts["user_and_generation_tokens"])
        self.assertNotEqual(result["questions"]["A"], result["questions"]["B"])
        self.assertEqual(result["system_hash"], H.digest_text(result["system"]))

    def test_agent_requests_share_the_system_message(self):
        agent = {"system": "AS", "questions": {"A": "qa", "B": "qb"}}
        self.assertEqual(B.request_text("agent:A", {"system": "S", "first": "U"}, None, agent), ("AS", "qa"))
        self.assertEqual(B.request_text("agent:B", {"system": "S", "first": "U"}, None, agent), ("AS", "qb"))
        with self.assertRaises(ValueError):
            B.request_text("agent:A", {"system": "S", "first": "U"}, None, None)
        with self.assertRaises(ValueError):
            B.request_text("agent:Z", {"system": "S", "first": "U"}, None, agent)


class RequestSpecTests(unittest.TestCase):
    PROMPTS = {"system": "S", "first": "U"}
    LONG = {"system": "LS", "questions": {"A": "qa", "B": "qb"}}

    def test_each_spec_picks_its_system_and_question(self):
        self.assertEqual(B.request_text("ref", self.PROMPTS, None), ("S", "U"))
        self.assertEqual(B.request_text("long:A", self.PROMPTS, self.LONG), ("LS", "qa"))
        self.assertEqual(B.request_text("long:B", self.PROMPTS, self.LONG), ("LS", "qb"))

    def test_unknown_specs_are_refused(self):
        for spec in ("long:C", "short", "long"):
            with self.assertRaises(ValueError):
                B.request_text(spec, self.PROMPTS, self.LONG)
        with self.assertRaises(ValueError):
            B.request_text("long:A", self.PROMPTS, None)


class LinearStateConditionTests(unittest.TestCase):
    """F2L is F2 plus FLASHNEXT_EVAL_LINEAR_STATE=1, and nothing else changes."""

    FLAG = "FLASHNEXT_EVAL_LINEAR_STATE"

    def test_only_f2l_sets_the_flag(self):
        for name, condition in H.ALL_CONDITIONS.items():
            overlay = H.condition_overlay(condition)
            self.assertIn(self.FLAG, overlay, name)
            self.assertEqual(overlay[self.FLAG], "1" if name in ("F2L", "F2P", "F2B", "F2Pn", "F2Bn", "F2Bnr", "F2Br", "F2Pnr", "F2Bm", "F2Bmu") else None, name)

    def test_f2l_matches_f2_apart_from_the_flag(self):
        f2, f2l = H.ALL_CONDITIONS["F2"], H.ALL_CONDITIONS["F2L"]
        a, b = H.condition_overlay(f2), H.condition_overlay(f2l)
        self.assertEqual({k: v for k, v in b.items() if k != self.FLAG},
                         {k: v for k, v in a.items() if k != self.FLAG})
        self.assertEqual((f2.ring, f2.scratch, f2.cache_mb, f2.turn), (f2l.ring, f2l.scratch, f2l.cache_mb, f2l.turn))

    def test_f2l_is_a_diagnostic_not_part_of_the_planned_matrix(self):
        self.assertIn("F2L", H.DIAGNOSTIC_CONDITIONS)
        self.assertNotIn("F2L", H.CONDITIONS)
        self.assertFalse(H.ALL_CONDITIONS["F2L"].is_baseline)
        self.assertTrue(H.ALL_CONDITIONS["F0"].is_baseline)

    def test_launch_environment_carries_the_flag_for_a_first_turn_arm(self):
        self.assertEqual(H.launch_overlay(H.ALL_CONDITIONS["F2L"])[self.FLAG], "1")
        self.assertIsNone(H.launch_overlay(H.ALL_CONDITIONS["F2"])[self.FLAG])


if __name__ == "__main__":
    unittest.main()


class LaunchGateTests(unittest.TestCase):
    """The launch gate and the child's stop floor are different numbers (the first F2Bm launch used one for both)."""

    def test_the_gate_refuses_a_start_below_it_and_says_so(self):
        from models.flashnext.tests.bench import bench_prefill_memory as B

        self.assertIsNone(B.start_gate_reason({"available_mb": 9216.0}, 9))
        self.assertIn("launch gate", B.start_gate_reason({"available_mb": 9215.0}, 9))
        self.assertIn("cannot be checked", B.start_gate_reason({"available_mb": None}, 9))
        self.assertIn("launch gate", B.start_gate_reason({"available_mb": 8190.0}, 9))
        self.assertIsNone(B.start_gate_reason({"available_mb": 100.0}, None))        # no gate set: unchanged behaviour

    def test_the_gate_is_not_passed_to_the_child_as_its_floor(self):
        from models.flashnext.tests.bench import bench_prefill_memory as B

        args = SimpleNamespace(min_available_gb=2.0, start_available_gb=9.0, max_compressor_growth_gib=2.0,
                               compressor_stop_available_gib=3.0)
        out = B.diagnostic_args(args)
        self.assertEqual(out[out.index("--min-available-gb") + 1], "2")
        self.assertNotIn("9", out[out.index("--min-available-gb") + 1])
        self.assertNotIn("--start-available-gb", out)

    def test_check_ready_displays_separate_startup_and_runtime_thresholds(self):
        from models.flashnext.tests.bench import bench_prefill_memory as B

        args = SimpleNamespace(
            sizes=["1k"], conditions=["F0"], rounds=1, final=False,
            min_available_gb=2.0, start_available_gb=9.0, expert_pool_gb=None, ring_min_tokens=None,
            max_compressor_growth_gib=2.0, compressor_stop_available_gib=3.0,
            trace_interval=1.0, fast_trace_interval=0.25, requests="ref",
            prefix_min_tokens=16, prefix_ram_gb=0.75, prefix_disk_gb=5.0,
            prefix_dir=None, dump_dir=None, suite=None
        )
        sampler = SimpleNamespace(sample=lambda: {"available_mb": 9500.0, "battery_percent": 100,
                                                  "power_source": "ac", "low_power": False,
                                                  "pressure_level": 1, "swap_used_mb": 0.0})
        rows = [{"size": "1k", "target": 1024, "first": 1025, "later": 1024}]
        with mock.patch.object(B, "prompt_sizes", return_value=rows), \
                mock.patch("veystrix.checkpoints.resolve_flashnext", return_value=Path("/models/ckpt")):
            report = B.check_ready(
                args, sampler=sampler, preflight_module=T.ready_preflight(),
                tokenizer_loader=lambda p: T.CharTokenizer(), contract=lambda: [], environ={}, free_gb=8.0)
        self.assertEqual(report.verdict, "READY")
        check_names = [c.name for c in report.checks]
        self.assertIn("memory thresholds", check_names)
        mem_check = next(c for c in report.checks if c.name == "memory thresholds")
        self.assertIn("launch gate: 9 GiB", mem_check.detail)
        self.assertIn("child runtime stop floor: 2 x 1000 MiB", mem_check.detail)

    def test_check_ready_fails_when_below_startup_gate(self):
        from models.flashnext.tests.bench import bench_prefill_memory as B

        args = SimpleNamespace(
            sizes=["1k"], conditions=["F0"], rounds=1, final=False,
            min_available_gb=2.0, start_available_gb=9.0, expert_pool_gb=None, ring_min_tokens=None,
            max_compressor_growth_gib=2.0, compressor_stop_available_gib=3.0,
            trace_interval=1.0, fast_trace_interval=0.25, requests="ref",
            prefix_min_tokens=16, prefix_ram_gb=0.75, prefix_disk_gb=5.0,
            prefix_dir=None, dump_dir=None, suite=None
        )
        # 8.19 GB available (the condition that stopped the previous real-model run)
        sampler = SimpleNamespace(sample=lambda: {"available_mb": 8190.0, "battery_percent": 100,
                                                  "power_source": "ac", "low_power": False,
                                                  "pressure_level": 1, "swap_used_mb": 0.0})
        rows = [{"size": "1k", "target": 1024, "first": 1025, "later": 1024}]
        with mock.patch.object(B, "prompt_sizes", return_value=rows), \
                mock.patch("veystrix.checkpoints.resolve_flashnext", return_value=Path("/models/ckpt")):
            report = B.check_ready(
                args, sampler=sampler, preflight_module=T.ready_preflight(),
                tokenizer_loader=lambda p: T.CharTokenizer(), contract=lambda: [], environ={}, free_gb=8.0)
        self.assertEqual(report.verdict, "BLOCKED")
        state_check = next(c for c in report.checks if c.name == "machine state")
        self.assertEqual(state_check.status, "FAIL")
        self.assertIn("below the 9 GiB launch gate", state_check.detail)
