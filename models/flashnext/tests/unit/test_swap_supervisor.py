"""Swap-growth supervisor, launch supervision and flight recorder (offline, synthetic readings, no real swapping).

Incident 2026-11-09: a DeepSeek Harness request ran on a server launched without the supervisor. These tests pin the repairs:
the launch is refused unsupervised; the +2.5 GiB stop cancels an in-flight job cooperatively and then exits; a worker that
never cancels (a blocked Metal call) is ended by the grace deadline; an idle server exits at once; and /health never says
"ok" for a stopped or stalled worker.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from models.flashnext import flight_recorder, memory_controller as MC, supervision
from models.flashnext.chat_guard import ChatGuard
from models.flashnext.tests.unit.test_chat_guard import Probe, Sampler

LAUNCHER_ENV = {"VEYSTRIX_MODE": "agentic", "FLASHNEXT_CHAT_GUARD": "1", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048"}


class LaunchSupervisionTests(unittest.TestCase):
    def test_a_direct_server_launch_without_the_launcher_environment_is_refused(self):
        message = supervision.check({}, model="flashnext", server=True)
        self.assertIsNotNone(message)
        for needle in ("VEYSTRIX_MODE", "FLASHNEXT_CHAT_GUARD", "FLASHNEXT_PREFILL_CHUNK_TOKENS", "./chat.sh --mode agentic",
                       "--max-cold-prefill 32768", "Nothing was loaded"):
            self.assertIn(needle, message)

    def test_each_missing_safety_setting_is_reported_alone(self):
        for key in LAUNCHER_ENV:
            env = {k: v for k, v in LAUNCHER_ENV.items() if k != key}
            message = supervision.check(env, model="flashnext", server=True)
            self.assertIsNotNone(message, key)
            self.assertIn(key, message)
        self.assertIn("unset or 0", supervision.check({**LAUNCHER_ENV, "FLASHNEXT_PREFILL_CHUNK_TOKENS": "0"},
                                                      model="flashnext", server=True))

    def test_the_launcher_environment_is_accepted(self):
        self.assertIsNone(supervision.check(LAUNCHER_ENV, model="flashnext", server=True))

    def test_only_a_flashnext_server_is_checked(self):
        self.assertIsNone(supervision.check({}, model="flashnext", server=False))
        self.assertIsNone(supervision.check({}, model="qwen27b", server=True))
        self.assertIsNone(supervision.check({}, model="flashnext", server=True, benchmarking=True))

    def test_the_real_launcher_plan_satisfies_the_check(self):
        from models.flashnext.settings import modes

        plan = modes.plan_mode("agentic", pool_gb="0", free_disk_gb=500)
        environment = modes.apply_mode({}, plan)
        self.assertIsNone(supervision.check(environment, model="flashnext", server=True))
        self.assertEqual(environment["VEYSTRIX_MAX_COLD_PREFILL"], "32768")
        self.assertEqual(environment["VEYSTRIX_MAX_INCREMENTAL_PREFILL"], "32768")
        self.assertEqual(environment["VEYSTRIX_CONTEXT_LIMIT"], "262144")
        self.assertEqual(environment["VEYSTRIX_CONTEXT_REQUIRED"], "1")


class GuardCase(unittest.TestCase):
    BASELINE_MB = 1000

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.probe, self.sampler = Probe(), Sampler()
        self.sampler.swap_used = self.BASELINE_MB
        self.exits = []
        self.now = [1000.0]
        self.sleeps = []
        self.guard = ChatGuard(Path(self.tmp.name) / "trace.jsonl", probe=self.probe, sampler=self.sampler,
                               exit_fn=self.exits.append, clock=lambda: self.now[0])
        self.guard.sleep = self.fake_sleep
        self.busy = [False]
        MC.register_worker_probe(lambda: self.busy[0])
        self.addCleanup(MC.register_worker_probe, None)
        patch = mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "agentic", "FLASHNEXT_CHAT_GUARD": "1"})
        patch.start()
        self.addCleanup(patch.stop)
        self.guard.start()
        self.addCleanup(self.guard.close)

    def fake_sleep(self, seconds):
        """Time only moves when the guard waits; a test hook may change the world while it does."""
        self.sleeps.append(seconds)
        self.now[0] += seconds
        hook = getattr(self, "on_sleep", None)
        if hook:
            hook(len(self.sleeps))

    def tick(self, seconds=1.0):
        self.guard.tracer.stop()
        self.now[0] += seconds
        self.guard.last_power_check = 0
        self.guard.tracer.step()

    def cross_the_limit(self):
        self.sampler.swap_used = self.BASELINE_MB + 2560      # +2.5 GiB over the launch baseline
        self.tick()

    def wait_for_grace_thread(self):
        for thread in threading.enumerate():
            if thread.name == "guard-grace":
                thread.join(timeout=5)


class GracefulSwapStopTests(GuardCase):
    def test_the_baseline_is_the_launch_reading_not_zero(self):
        ledger = self.guard.controller.ledger()
        self.assertEqual(ledger["swap_baseline_mb"], self.BASELINE_MB)
        self.sampler.swap_used = self.BASELINE_MB + 2000
        self.tick()
        self.assertEqual(self.exits, [])
        self.assertAlmostEqual(self.guard.controller.swap_increase_mb(), 2000, delta=1)
        self.assertEqual(self.guard.controller.baseline_swap_used_mb, self.BASELINE_MB)

    def test_an_idle_server_exits_at_once_at_the_limit(self):
        self.busy[0] = False
        self.cross_the_limit()
        self.assertEqual(self.exits, [75])
        self.assertEqual(self.guard.stop_mode, "hard")
        self.assertFalse(MC.can_admit_work())

    def test_a_busy_server_closes_admission_and_exits_only_after_the_worker_cancels(self):
        self.busy[0] = True
        self.on_sleep = lambda n: self.busy.__setitem__(0, n < 3)    # the job cancels at its next layer boundary
        self.cross_the_limit()
        self.assertEqual(self.guard.stop_mode, "graceful")
        self.assertFalse(MC.can_admit_work(), "new requests must be refused as soon as the limit is crossed")
        self.wait_for_grace_thread()
        self.assertEqual(self.exits, [75])
        self.assertIn(1.5, self.sleeps, "the flush period must be honoured before the exit")

    def test_a_worker_that_never_cancels_is_ended_by_the_grace_deadline(self):
        """A blocked Metal call cannot be interrupted from Python: the delay is bounded and then the process exits."""
        self.busy[0] = True
        with mock.patch.dict(os.environ, {"FLASHNEXT_GUARD_CANCEL_GRACE_S": "5"}):
            self.cross_the_limit()
            self.wait_for_grace_thread()
        self.assertEqual(self.exits, [75])
        waited = sum(self.sleeps)
        self.assertGreaterEqual(waited, 4.99)
        self.assertLess(waited, 6.0, "the exit must follow the deadline, not wait forever")
        self.assertNotIn(1.5, self.sleeps, "a worker that never cancelled gets no flush period")

    def test_a_second_stop_reason_does_not_restart_the_sequence(self):
        self.busy[0] = True
        with mock.patch.dict(os.environ, {"FLASHNEXT_GUARD_CANCEL_GRACE_S": "3"}):
            self.cross_the_limit()
            first = self.guard.stop_started
            self.guard._stop("another reason")
            self.assertEqual(self.guard.stop_started, first)
            self.wait_for_grace_thread()
        self.assertEqual(self.exits, [75])

    def test_a_zero_grace_period_means_an_immediate_exit_even_when_busy(self):
        self.busy[0] = True
        with mock.patch.dict(os.environ, {"FLASHNEXT_GUARD_CANCEL_GRACE_S": "0"}):
            self.cross_the_limit()
        self.assertEqual(self.exits, [75])

    def test_non_swap_stops_are_never_deferred(self):
        """Battery, thermal, a blind guard and the Metal limit keep the immediate exit even with a job running."""
        self.busy[0] = True
        self.sampler.sample = lambda: {"t": 1, "battery_percent": 20, "power_source": "battery", "low_power": False,
                                       "pressure_level": 1, "swap_used_mb": self.BASELINE_MB,
                                       "thermal_speed_limit": None, "thermal_warning": False}
        self.tick()
        self.assertEqual(self.exits, [75])
        self.assertEqual(self.guard.stop_mode, "hard")

    def test_a_missing_swap_probe_fails_closed_immediately(self):
        self.busy[0] = True
        self.guard.tracer.stop()
        self.sampler.swap_used = None
        self.guard.last_power_check = 0
        self.guard.tracer.step()
        self.assertEqual(self.exits, [75])
        self.assertEqual(self.guard.stop_mode, "hard")

    def test_the_high_watermark_survives_a_swap_decrease(self):
        self.busy[0] = False
        self.sampler.swap_used = self.BASELINE_MB + 2559
        self.tick()
        self.assertEqual(self.exits, [])
        self.sampler.swap_used = self.BASELINE_MB + 100
        self.tick()
        self.assertEqual(self.guard.controller.ledger()["peak_swap_growth_mb"], 2559.0)
        self.assertAlmostEqual(self.guard.controller._peak_swap_growth_mb, 2559.0, delta=0.5)


class FlightRecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "logs", "server.jsonl")

    def test_events_are_flushed_line_by_line_and_private(self):
        recorder = flight_recorder.FlightRecorder(self.path)
        recorder.record("request_begin", path="/v1/chat/completions", tools=32)
        recorder.record("prefill_progress", done=2048, total=8853)
        rows = [json.loads(line) for line in open(self.path)]       # readable before close: it survives os._exit
        self.assertEqual([r["event"] for r in rows], ["request_begin", "prefill_progress"])
        self.assertEqual(rows[1]["done"], 2048)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(os.path.dirname(self.path)).st_mode), 0o700)

    def test_it_rotates_and_never_raises(self):
        recorder = flight_recorder.FlightRecorder(self.path, max_bytes=400)
        for index in range(40):
            recorder.record("tick", index=index)
        self.assertTrue(os.path.exists(self.path + ".1"))
        self.assertLessEqual(os.path.getsize(self.path), 400 + 200)
        broken = flight_recorder.FlightRecorder("/proc/definitely/not/writable/x.jsonl")
        broken.record("x")
        self.assertEqual(broken.events, 0)
        self.assertGreaterEqual(broken.failures, 1)

    def test_it_is_configurable_and_can_be_switched_off(self):
        self.assertIsNone(flight_recorder.resolve_path({"FLASHNEXT_FLIGHT_LOG": "0"}))
        self.assertEqual(flight_recorder.resolve_path({"FLASHNEXT_FLIGHT_LOG": "/x/y.jsonl"}), "/x/y.jsonl")
        self.assertTrue(flight_recorder.resolve_path({}, pid=7).endswith("server-7.jsonl"))
        flight_recorder.FlightRecorder(None).record("ignored")      # no path: a no-op


class ProgressSnapshotTests(unittest.TestCase):
    def test_only_real_work_moves_the_progress_clock(self):
        from models.flashnext.progress import Progress

        p = Progress()
        p.request_begin(None, "/v1/chat/completions")
        first = p.snapshot()
        self.assertTrue(first["in_request"])
        time.sleep(0.05)
        self.assertGreaterEqual(p.snapshot()["seconds_since_progress"], 0.0)
        before = p.snapshot()["progress_ticks"]
        p.tick()
        self.assertEqual(p.snapshot()["progress_ticks"], before + 1)
        p.request_end()
        self.assertFalse(p.snapshot()["in_request"])


if __name__ == "__main__":
    unittest.main()
