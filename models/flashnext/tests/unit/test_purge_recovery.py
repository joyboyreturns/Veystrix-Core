import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from models.flashnext import purge_recovery as P

GIB = P.GIB
REAL_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   492963.
Pages active:                                 166961.
Pages inactive:                               129019.
Pages speculative:                             39502.
Pages throttled:                                   0.
Pages wired down:                             124635.
Pages purgeable:                               25364.
File-backed pages:                            113326.
Anonymous pages:                              222156.
Pages stored in compressor:                   166101.
Pages occupied by compressor:                  56791.
Swapins:                                     3044211.
Swapouts:                                    3908991.
"""


class Machine:
    """A byte-granular fake machine: a purge turns `cache` bytes of file cache into free memory."""

    def __init__(self, available, cache, effect=1.0, pressure=1):
        self.free, self.cache, self.effect, self.pressure = available, cache, effect, pressure
        self.inactive = 0
        self.swapouts = 0
        self.t = 1000.0
        self.purges = 0
        self.capability = 0
        self.purge_rc = 0
        self.calls = []
        self.concurrent = self.max_concurrent = 0
        self.hold = None

    def read(self):
        return P.Reading(page=1, free=self.free, inactive=self.inactive, file_backed=self.cache,
                         anonymous=0, pressure=self.pressure, swapouts=self.swapouts)

    def runner(self, command, timeout):
        self.calls.append((tuple(command), timeout))
        if tuple(command) == P.CAPABILITY_COMMAND:
            return SimpleNamespace(returncode=self.capability, stderr="sudo: a password is required" if self.capability else "")
        assert tuple(command) == P.PURGE_COMMAND
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.hold:
                self.hold()
            if self.purge_rc == 0:
                moved = int(self.cache * self.effect)
                self.free += moved
                self.cache -= moved
                self.purges += 1
            return SimpleNamespace(returncode=self.purge_rc, stderr="sudo: a password is required" if self.purge_rc else "")
        finally:
            self.concurrent -= 1

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def make(machine, **kwargs):
    cfg = P.Config(**kwargs.pop("cfg", {}))
    return P.PurgeRecovery(cfg, read=machine.read, runner=machine.runner, clock=machine.clock, sleep=machine.sleep,
                           activity=kwargs.pop("activity", P.Activity()), disk_read=kwargs.pop("disk", lambda: 0), **kwargs)


class AccountingTests(unittest.TestCase):
    def test_real_vm_stat_identity_and_bounds(self):
        reading = P.parse_vm_stat(REAL_VM_STAT, pressure=1)
        info = P.accounting(reading)
        self.assertTrue(info["file_plus_anonymous_equals_queues"])             # nothing counted twice
        self.assertEqual(info["gate_available"], (492963 + 129019 + 39502) * 16384)
        self.assertEqual(info["activity_monitor_cached_files"], (113326 + 25364) * 16384)
        self.assertLessEqual(info["purge_gain_lower"], info["purge_gain_upper"])
        self.assertEqual(info["purge_gain_upper"], (113326 - 39502) * 16384)
        self.assertEqual(info["purge_gain_lower"], 0)                          # inactive alone could be all file cache

    def test_the_gate_unit_is_a_binary_gib(self):
        self.assertEqual(P.Config().start_gate_bytes, 9_663_676_416)

    def test_cached_files_can_read_high_while_the_gate_reads_low(self):
        # 7 GB cached but mostly active, little inactive: the gate figure misses the active cache.
        reading = P.Reading(page=1, free=int(1.5e9), active=int(7e9), inactive=int(2e9), speculative=int(0.3e9),
                            file_backed=int(7.2e9), anonymous=int(2.1e9), purgeable=0)
        info = P.accounting(reading)
        self.assertLess(info["gate_available"], 9 * GIB)
        self.assertGreater(info["activity_monitor_cached_files"], 7e9)
        self.assertGreater(info["purge_gain_upper"], 6.5e9)


class DecisionTests(unittest.TestCase):
    def test_nothing_happens_when_memory_is_enough(self):
        m = Machine(10 * GIB, 5 * GIB)
        r = make(m)
        self.assertEqual(r.attempt(9 * GIB, "startup").action, "ok")
        self.assertEqual(m.purges, 0)
        self.assertEqual(m.calls, [])

    def test_a_justified_purge_runs_once_and_is_recorded(self):
        m = Machine(int(8.5 * GIB), 4 * GIB)
        r = make(m)
        out = r.attempt(9 * GIB + GIB // 8, "startup")
        self.assertEqual(out.action, "purged")
        self.assertEqual(m.purges, 1)
        self.assertTrue(out.record.effective)
        self.assertEqual(out.record.recovered, 4 * GIB)
        self.assertEqual(r.summary()["purges"], 1)
        self.assertEqual([c[0] for c in m.calls], [P.CAPABILITY_COMMAND, P.PURGE_COMMAND])   # only these two commands, ever

    def test_a_purge_that_cannot_close_the_gap_is_not_run(self):
        m = Machine(6 * GIB, 1 * GIB)
        out = make(m).attempt(9 * GIB, "startup")
        self.assertEqual((out.action, m.purges), ("skipped", 0))
        self.assertIn("cannot close the gap", out.reason)

    def test_critical_pressure_and_ongoing_swapout_are_respected(self):
        m = Machine(8 * GIB, 4 * GIB, pressure=4)
        r = make(m)
        self.assertIn("critical", r.attempt(9 * GIB, "idle").reason)
        m.pressure = 1
        m.swapouts = 10
        r._last_swapouts = 0
        self.assertIn("swap-out in progress", r.attempt(9 * GIB, "idle").reason)
        self.assertEqual(m.purges, 0)
        self.assertEqual(r.attempt(9 * GIB, "idle").action, "purged")          # swapping stopped

    def test_owned_memory_is_reclaimed_first_and_can_make_the_purge_unnecessary(self):
        m = Machine(int(8.9 * GIB), 4 * GIB)
        r = make(m)
        r.add_reclaim(lambda reading: (setattr(m, "free", 10 * GIB), 1)[1])
        out = r.attempt(9 * GIB, "idle")
        self.assertEqual((out.action, m.purges), ("ok", 0))

    def test_many_purges_over_time_have_no_daily_limit_but_a_minimum_spacing(self):
        m = Machine(8 * GIB, 4 * GIB)
        r = make(m)
        done = 0
        for _ in range(40):
            m.free, m.cache = 8 * GIB, 4 * GIB          # memory is short again (cache refilled)
            m.t += 61                                   # past the 60 s spacing
            done += r.attempt(9 * GIB, "idle").action == "purged"
        self.assertEqual(done, 40)
        m.free, m.cache = 8 * GIB, 4 * GIB
        self.assertIn("backoff", r.attempt(9 * GIB, "idle").reason)           # a refill loop cannot run tight

    def test_unavailable_sudo_never_runs_purge_and_is_rechecked_later(self):
        m = Machine(8 * GIB, 4 * GIB)
        m.capability = 1
        r = make(m)
        self.assertIn("not permitted", r.attempt(9 * GIB, "startup").reason)
        self.assertIn("not permitted", r.attempt(9 * GIB, "startup").reason)
        self.assertEqual(m.purges, 0)
        self.assertNotIn(P.PURGE_COMMAND, [c[0] for c in m.calls])
        self.assertEqual(sum(c[0] == P.CAPABILITY_COMMAND for c in m.calls), 1)   # cached, not re-run every tick
        m.capability = 0
        m.t += r.cfg.permission_recheck_s + 1
        self.assertEqual(r.attempt(9 * GIB, "startup").action, "purged")

    def test_a_failing_purge_backs_off_and_is_not_retried_in_a_loop(self):
        m = Machine(8 * GIB, 4 * GIB)
        m.purge_rc = 1
        r = make(m)
        out = r.attempt(9 * GIB, "idle")
        self.assertEqual(out.action, "skipped")
        self.assertIn("purge failed", out.reason)
        self.assertEqual(sum(c[0] == P.PURGE_COMMAND for c in m.calls), 1)
        m.t += 1
        r.attempt(9 * GIB, "idle")
        self.assertEqual(sum(c[0] == P.PURGE_COMMAND for c in m.calls), 1)       # still backing off

    def test_ineffective_purges_back_off_exponentially_to_a_cap_and_one_success_resets(self):
        m = Machine(8 * GIB, 4 * GIB, effect=0.0)             # the purge frees nothing
        r = make(m, cfg={"backoff_max_s": 240.0})
        waits = []
        for _ in range(5):
            m.t = max(m.t, r.not_before) + 0.1
            out = r.attempt(9 * GIB, "idle")
            self.assertEqual(out.action, "purged")
            self.assertFalse(out.record.effective)
            waits.append(round(r.not_before - m.t))
        self.assertEqual(waits[:4], [30, 60, 120, 240])
        self.assertLessEqual(max(waits), 240)
        self.assertIn("backoff", r.attempt(9 * GIB, "idle").reason)
        m.effect = 1.0
        m.t = r.not_before + 1
        self.assertTrue(r.attempt(9 * GIB, "idle").record.effective)
        self.assertEqual(r.consecutive_ineffective, 0)

    def test_refill_and_ssd_cost_are_recorded(self):
        m = Machine(8 * GIB, 4 * GIB)
        disk = [0]
        r = make(m, disk=lambda: disk[0])
        r.attempt(9 * GIB, "idle")
        m.cache += 3 * GIB                                    # the cache refills
        disk[0] = 2 * GIB                                     # and the engine read 2 GiB from the SSD
        summary = r.summary()
        self.assertEqual(summary["refill_file_bytes"], [3 * GIB])
        self.assertEqual(summary["ssd_read_bytes_after_last"], 2 * GIB)

    def test_events_are_logged_with_private_permissions(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "logs", "purge.jsonl")
            m = Machine(8 * GIB, 4 * GIB)
            r = make(m, log_path=path)
            r.attempt(9 * GIB, "idle")
            r.close()
            with open(path) as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([x["event"] for x in rows], ["purge", "refill", "summary"])
            self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")


class RunnerTests(unittest.TestCase):
    def test_the_default_runner_executes_only_the_given_argv_without_a_shell_or_terminal_input(self):
        with mock.patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")) as run:
            P.default_runner(P.PURGE_COMMAND, 30)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["sudo", "-n", "/usr/sbin/purge"])
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(kwargs["timeout"], 30)

    def test_the_flag_is_off_by_default_and_the_environment_builds_the_config(self):
        self.assertFalse(P.enabled({}))
        self.assertIsNone(P.from_environment({}))
        recovery = P.from_environment({"FLASHNEXT_AUTO_PURGE": "1", "FLASHNEXT_PURGE_CHECK_SECONDS": "7",
                                       "FLASHNEXT_PURGE_IDLE_HEADROOM_GB": "6"})
        self.assertEqual((recovery.cfg.interval_s, recovery.cfg.idle_headroom_bytes), (7.0, int(6e9)))
        self.assertEqual(P.Config.from_environment({}).interval_s, 5.0)


class ActivityTests(unittest.TestCase):
    def test_no_purge_during_a_request_a_load_or_another_purge(self):
        for guard in ("request", "loading_model"):
            m = Machine(8 * GIB, 4 * GIB)
            activity = P.Activity()
            r = make(m, activity=activity)
            with getattr(activity, guard)():
                out = r.attempt(9 * GIB, "idle")
                self.assertEqual((out.action, m.purges), ("skipped", 0), guard)
            self.assertEqual(r.attempt(9 * GIB, "idle").action, "purged")

    def test_a_cancelled_or_failed_request_releases_the_gate(self):
        activity = P.Activity()
        with self.assertRaises(KeyboardInterrupt):
            with activity.request():
                raise KeyboardInterrupt
        self.assertEqual((activity.requests, activity.busy_reason()), (0, None))

    def test_a_request_arriving_during_a_purge_waits_for_it(self):
        m = Machine(8 * GIB, 4 * GIB)
        activity = P.Activity()
        r = make(m, activity=activity)
        started, release = threading.Event(), threading.Event()
        m.hold = lambda: (started.set(), release.wait(5))
        worker = threading.Thread(target=lambda: r.attempt(9 * GIB, "idle"))
        worker.start()
        self.assertTrue(started.wait(5))
        entered = []
        requester = threading.Thread(target=lambda: [activity.request().__enter__(), entered.append(m.purges)])
        requester.start()
        time.sleep(0.1)
        self.assertEqual(entered, [])                         # still waiting: the purge is running
        release.set()
        worker.join(5)
        requester.join(5)
        self.assertEqual(entered, [1])                        # entered only after the purge finished

    def test_two_purge_attempts_never_overlap(self):
        m = Machine(8 * GIB, 400 * GIB)
        r = make(m)
        started, release = threading.Event(), threading.Event()
        m.hold = lambda: (started.set(), release.wait(5))
        outcomes = []
        first = threading.Thread(target=lambda: outcomes.append(r.attempt(9 * GIB, "idle")))
        first.start()
        self.assertTrue(started.wait(5))
        outcomes.append(r.attempt(9 * GIB, "idle"))           # second attempt while the first purge runs
        release.set()
        first.join(5)
        self.assertEqual(m.max_concurrent, 1)
        self.assertEqual(sorted(o.action for o in outcomes), ["purged", "skipped"])


class LifecycleTests(unittest.TestCase):
    def test_startup_purges_when_justified_and_then_reports_ready(self):
        m = Machine(int(8.4 * GIB), 4 * GIB)
        r = make(m)
        lines = []
        result = P.ensure_startup(r, out=lines.append)
        self.assertTrue(result.ready)
        self.assertEqual((result.purges, m.purges), (1, 1))
        self.assertGreaterEqual(result.available, 9 * GIB + GIB // 8)

    def test_startup_gives_up_after_the_wait_and_leaves_the_decision_to_the_guard(self):
        m = Machine(6 * GIB, 1 * GIB)                          # a purge cannot help
        r = make(m, cfg={"startup_wait_s": 12.0})
        result = P.ensure_startup(r)
        self.assertFalse(result.ready)
        self.assertEqual(m.purges, 0)
        self.assertGreaterEqual(result.waited_s, 12.0)

    def test_startup_with_unavailable_sudo_does_not_prompt_or_purge(self):
        m = Machine(8 * GIB, 4 * GIB)
        m.capability = 1
        result = P.ensure_startup(make(m, cfg={"startup_wait_s": 6.0}))
        self.assertFalse(result.ready)
        self.assertNotIn(P.PURGE_COMMAND, [c[0] for c in m.calls])

    def test_idle_monitor_purges_when_idle_never_when_busy_and_stops_cleanly(self):
        m = Machine(4 * GIB, 6 * GIB)
        activity = P.Activity()
        r = make(m, activity=activity, cfg={"interval_s": 0.01})
        monitor = P.Monitor(r)
        with activity.request():
            self.assertIsNone(monitor.tick())                  # busy: not even considered
        self.assertEqual(m.purges, 0)
        monitor.start()
        deadline = time.time() + 5
        while m.purges == 0 and time.time() < deadline:
            time.sleep(0.01)
        monitor.stop()
        self.assertEqual(m.purges, 1)
        self.assertFalse(monitor._thread.is_alive())
        before = len(m.calls)
        self.assertEqual(r.attempt(9 * GIB, "idle").action, "skipped")
        self.assertEqual(len(m.calls), before)                 # nothing runs after shutdown

    def test_shutdown_during_a_purge_waits_for_it_and_starts_no_more(self):
        m = Machine(4 * GIB, 60 * GIB)
        r = make(m, cfg={"interval_s": 0.01})
        started, release = threading.Event(), threading.Event()
        m.hold = lambda: (started.set(), release.wait(5))
        monitor = P.Monitor(r).start()
        self.assertTrue(started.wait(5))
        stopper = threading.Thread(target=monitor.stop)
        stopper.start()
        time.sleep(0.1)
        self.assertTrue(stopper.is_alive())                    # stop is waiting for the purge, not killing it
        release.set()
        stopper.join(5)
        self.assertFalse(monitor._thread.is_alive())
        self.assertEqual(m.purges, 1)


class BackendWiringTests(unittest.TestCase):
    def test_generate_marks_the_engine_busy_for_the_whole_call_including_cleanup(self):
        from macqwen.tests.test_flashnext_backend import FlashNextBackendTests

        backend = FlashNextBackendTests().backend()
        seen = {}

        def fake_generate(*args, **kwargs):
            seen["during"] = P.ACTIVITY.busy_reason()
            raise RuntimeError("cancelled")

        def fake_after():
            seen["cleanup"] = P.ACTIVITY.busy_reason()

        backend._generate = fake_generate
        backend._after_request = fake_after
        with self.assertRaises(RuntimeError):
            backend.generate(1)
        self.assertEqual(seen, {"during": "a request is active", "cleanup": "a request is active"})
        self.assertIsNone(P.ACTIVITY.busy_reason())

    def test_reclaim_hook_trims_the_allocator_and_keeps_snapshots_unless_stressed(self):
        from macqwen.tests.test_flashnext_backend import FlashNextBackendTests

        backend = FlashNextBackendTests().backend()
        shrunk = []
        backend._prefix = object()
        backend._prefix_session = lambda: SimpleNamespace(store=lambda: SimpleNamespace(shrink_ram=lambda n: shrunk.append(n) or 5))
        calm = P.Reading(page=1, free=int(10e9), pressure=1)
        backend.reclaim_owned_memory(calm)
        self.assertEqual(shrunk, [])                           # valuable RAM snapshots are kept
        stressed = P.Reading(page=1, free=int(2e9), pressure=2)
        self.assertGreaterEqual(backend.reclaim_owned_memory(stressed), 5)
        self.assertEqual(shrunk, [0])


if __name__ == "__main__":
    unittest.main()
