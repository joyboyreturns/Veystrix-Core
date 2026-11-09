import argparse
import json
import tempfile
import unittest
from pathlib import Path

from models.flashnext import overnight as O


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeChild:
    _pid = 5000

    def __init__(self, clock, exit_after, code):
        FakeChild._pid += 1
        self.pid = FakeChild._pid
        self._clock, self._deadline, self._code = clock, clock.now + exit_after, code
        self.killed = False

    def poll(self):
        return self._code if self._clock.now >= self._deadline else None

    def terminate(self):
        self._deadline = self._clock.now
        self._code = -15

    kill = terminate


def args(**over):
    base = dict(overnight=True, status=False, host="127.0.0.1", port=8080, pool_gb="0", checkpoint="/x", max_restarts=8,
                window_minutes=30.0, max_guard_stops=4, guard_cooldown=180.0, settle_seconds=60.0,
                unreachable_seconds=300.0, poll_seconds=10.0)
    base.update(over)
    return argparse.Namespace(**base)


class PolicyTests(unittest.TestCase):
    def test_clean_exit_owner_stop_and_config_error_never_restart(self):
        p = O.Policy()
        self.assertEqual(p.decide(0, 0)[0], "stop")
        self.assertEqual(p.decide(75, 0, owner_stop=True)[0], "stop")
        action, _, reason = p.decide(O.CONFIG_EXIT, 0)
        self.assertEqual(action, "stop")
        self.assertIn("exit 70", reason)

    def test_a_crash_restarts_with_a_growing_cooldown(self):
        p = O.Policy(base_cooldown_s=20, max_cooldown_s=100)
        cooldowns = [p.decide(1, t)[1] for t in (0, 1, 2, 3, 4)]
        self.assertEqual(cooldowns, [20, 40, 80, 100, 100])

    def test_a_crash_loop_halts_the_supervisor(self):
        p = O.Policy(max_restarts=3, window_s=600)
        self.assertEqual([p.decide(1, t)[0] for t in (0, 1, 2)], ["restart"] * 3)
        action, _, reason = p.decide(1, 3)
        self.assertEqual(action, "stop")
        self.assertIn("crash loop", reason)

    def test_old_crashes_age_out_of_the_window(self):
        p = O.Policy(max_restarts=2, window_s=100)
        p.decide(1, 0)
        p.decide(1, 1)
        self.assertEqual(p.decide(1, 500)[0], "restart")

    def test_guard_stops_wait_longer_and_have_their_own_budget(self):
        p = O.Policy(max_guard_stops=2, guard_cooldown_s=180)
        action, cooldown, reason = p.decide(O.GUARD_EXIT, 0)
        self.assertEqual((action, cooldown), ("restart", 180))
        self.assertIn("memory guard stop", reason)
        self.assertEqual(p.decide(O.ADMISSION_EXIT, 10)[0], "restart")
        action, _, reason = p.decide(O.GUARD_EXIT, 20)
        self.assertEqual(action, "stop")
        self.assertIn("cannot sustain", reason)


class SupervisorTests(unittest.TestCase):
    def build(self, script, **over):
        clock = FakeClock()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        children, launches = list(script), []

        def popen(command, **kw):
            launches.append((command, kw))
            exit_after, code = children.pop(0)
            return FakeChild(clock, exit_after, code)

        sup = O.Supervisor(args(**over), popen=popen, sleep=clock.sleep, clock=clock,
                           health_fn=lambda host, port: {"status": "ok", "ready": True},
                           swap_fn=lambda: 500.0, power_fn=lambda: {"source": "ac", "percent": 100}, log_dir=self.tmp.name,
                           port_fn=lambda host, port: False)
        return sup, launches

    def events(self):
        return [json.loads(l)["event"] for l in (Path(self.tmp.name) / "supervisor.jsonl").read_text().splitlines()]

    def test_launch_command_is_the_supervised_agentic_line(self):
        sup, launches = self.build([(1, 0)], overnight=False)
        sup.run()
        command, kw = launches[0]
        self.assertTrue(command[0].endswith("chat.sh"))
        self.assertEqual(command[1:3], ["--mode", "agentic"])
        for flag in ("--context-limit", "262144", "--max-cold-prefill", "--max-incremental-prefill", "32768"):
            self.assertIn(flag, command)
        self.assertEqual(kw["env"]["VEYSTRIX_API_KEY"], "local")
        self.assertEqual(kw["env"]["VEYSTRIX_INVALID_TOOL_CALLS"], "pass")
        self.assertTrue(kw["start_new_session"])

    def test_without_overnight_the_supervisor_runs_once_and_reports_the_exit(self):
        sup, launches = self.build([(5, 75)], overnight=False)
        self.assertEqual(sup.run(), 75)
        self.assertEqual(len(launches), 1)

    def test_overnight_restarts_after_a_crash_then_after_a_guard_stop_then_stops_on_a_clean_exit(self):
        sup, launches = self.build([(30, 1), (30, O.GUARD_EXIT), (30, 0)])
        self.assertEqual(sup.run(), 0)
        self.assertEqual(len(launches), 3)
        self.assertEqual(self.events().count("server_started"), 3)
        status = json.loads((Path(self.tmp.name) / "supervisor-status.json").read_text())
        self.assertEqual((status["state"], status["starts"]), ("stopped", 3))
        self.assertEqual([h["exit"] for h in status["recent"]], [1, 75, 0])

    def test_a_busy_port_at_first_start_stops_without_launching_anything(self):
        sup, launches = self.build([(1, 0)])
        sup.port_fn = lambda host, port: True
        self.assertEqual(sup.run(), O.PORT_BUSY_EXIT)
        self.assertEqual(launches, [])
        self.assertIn("port_busy", self.events())

    def test_a_port_still_held_after_a_restart_is_waited_for_then_gives_up(self):
        sup, launches = self.build([(10, 1), (10, 0)])
        state = {"calls": 0}

        def busy_after_first(host, port):
            state["calls"] += 1
            return state["calls"] > 1            # free for the first launch, busy forever afterwards

        sup.port_fn = busy_after_first
        self.assertEqual(sup.run(), O.PORT_BUSY_EXIT)
        self.assertEqual(len(launches), 1)

    def test_a_port_that_frees_up_during_the_wait_lets_the_restart_proceed(self):
        sup, launches = self.build([(10, 1), (10, 0)])
        calls = {"n": 0}

        def freeing(host, port):
            calls["n"] += 1
            return calls["n"] in (2, 3)          # busy twice right after the crash, then free

        sup.port_fn = freeing
        self.assertEqual(sup.run(), 0)
        self.assertEqual(len(launches), 2)

    def test_ready_is_not_reported_when_the_child_is_already_dead(self):
        sup, _ = self.build([(0, 1)], overnight=False)    # child exits immediately; /health still answers (someone else's)
        sup.run()
        self.assertNotIn("server_ready", self.events())

    def test_a_config_error_is_not_restarted_even_overnight(self):
        sup, launches = self.build([(1, O.CONFIG_EXIT), (1, 0)])
        self.assertEqual(sup.run(), O.CONFIG_EXIT)
        self.assertEqual(len(launches), 1)

    def test_repeated_guard_stops_end_the_night_with_a_reason(self):
        sup, launches = self.build([(10, O.GUARD_EXIT)] * 6, max_guard_stops=2)
        self.assertEqual(sup.run(), O.GUARD_EXIT)
        self.assertEqual(len(launches), 3)
        status = json.loads((Path(self.tmp.name) / "supervisor-status.json").read_text())
        self.assertIn("cannot sustain", status["reason"])

    def test_the_guard_cooldown_is_honoured_before_a_restart(self):
        sup, launches = self.build([(10, O.GUARD_EXIT), (10, 0)], guard_cooldown=180.0)
        clock_start = sup.clock()
        sup.run()
        self.assertGreaterEqual(sup.clock() - clock_start, 10 + 180 + 10)

    def test_a_restart_waits_for_swap_to_stop_moving(self):
        readings = iter([500.0, 900.0, 1400.0, 1400.0, 1400.0, 1400.0, 1400.0])
        sup, _ = self.build([(10, 1), (10, 0)])
        sup.swap_fn = lambda: next(readings, 1400.0)
        sup.run()
        self.assertIn("settled", self.events())

    def test_lost_health_after_ready_kills_and_restarts(self):
        sup, launches = self.build([(10_000, 0), (10, 0)], unreachable_seconds=60.0)
        calls = {"n": 0}

        def flaky(host, port):
            calls["n"] += 1
            return {"status": "ok", "ready": True} if calls["n"] <= 2 else None

        sup.health_fn = flaky
        sup.run()
        self.assertIn("health_lost", self.events())
        self.assertEqual(len(launches), 2)

    def test_an_owner_stop_terminates_the_child_and_does_not_restart(self):
        sup, launches = self.build([(10_000, 0)])
        orig_poll = sup.health_fn
        sup.health_fn = lambda h, p: (setattr(sup, "owner_stop", True) or orig_poll(h, p))
        self.assertEqual(sup.run(), 0)
        self.assertEqual(len(launches), 1)

    def test_old_launch_logs_are_pruned_and_nothing_else_is_touched(self):
        sup, _ = self.build([(1, 0)], overnight=False)
        logs = Path(self.tmp.name)
        for i in range(20):
            (logs / f"server-launch-old-{i:02d}.log").write_text("x")
        (logs / "keep-me.txt").write_text("x")
        sup.prune_launch_logs(keep=5)
        self.assertEqual(len(list(logs.glob("server-launch-*.log"))), 5)
        self.assertTrue((logs / "keep-me.txt").exists())

    def test_status_printer_handles_a_missing_log(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(O.print_status(d), 0)


class FakeThread:
    """Runs the target synchronously on start()."""

    def __init__(self, target):
        self.target, self.started = target, False

    def start(self):
        self.started = True
        self.target()

    def is_alive(self):
        return False


class LiveThread(FakeThread):
    """A warm-up that is still running."""

    def start(self):
        self.started = True

    def is_alive(self):
        return True


class FakeWarm:
    DEFAULT_EFFORTS = ("low", "medium", "high", "xhigh")

    class WarmAborted(Exception):
        pass

    class WarmStop(Exception):
        pass

    def __init__(self, entries, raises=None):
        self.entries, self.raises, self.calls, self.state = entries, raises, [], {}
        self.saved = 0

    def load(self, directory=None):
        return self.entries

    def load_state(self, directory=None):
        return dict(self.state)

    def save_state(self, state, directory=None):
        self.state = dict(state)
        self.saved += 1

    @staticmethod
    def job_key(entry, effort):
        return f"{entry['family']}:{effort}"

    def due_jobs(self, entries, efforts, state, now, refresh):
        return [(e, f) for e in entries for f in efforts if now - float(state.get(self.job_key(e, f), 0.0)) > refresh]

    def warm_one(self, base, entry, effort, *, key, timeout, abort=None):
        self.calls.append((entry["family"], effort, abort))
        if self.raises:
            raise self.raises
        return {"prompt_tokens": 100, "hit_tokens": 90, "prefilled_tokens": 10, "outcome": "SSD_HIT"}


QUIET = {"status": "ok", "ready": True, "worker": {"stage": "idle", "busy": False, "seconds_in_stage": 1200.0},
         "queue": {"waiting": 0}}


class AutoWarmerTests(unittest.TestCase):
    def build(self, entries=None, raises=None, power="ac", enabled=True, thread=FakeThread, **over):
        self.events = []
        clock = FakeClock()
        clock.now = 1.7e9                      # epoch-sized, like time.time()
        self.fake = FakeWarm(entries if entries is not None else [{"family": "qwencode"}, {"family": "deepseek-harness"}], raises)
        ns = args(auto_warm=enabled, warm_idle_minutes=15.0, warm_refresh_hours=6.0, **over)
        warmer = O.AutoWarmer(ns, lambda event, **f: self.events.append((event, f)), power_fn=lambda: {"source": power, "percent": 90},
                              clock=clock, warm_module=self.fake, thread_factory=thread)
        return warmer

    def names(self):
        return [e for e, _ in self.events]

    def test_disabled_does_nothing(self):
        warmer = self.build(enabled=False)
        warmer.tick(QUIET)
        self.assertEqual(self.fake.calls, [])

    def test_it_waits_for_a_quiet_server(self):
        for report in ({**QUIET, "worker": {**QUIET["worker"], "seconds_in_stage": 60.0}},
                       {**QUIET, "worker": {**QUIET["worker"], "busy": True}},
                       {**QUIET, "worker": {**QUIET["worker"], "stage": "prefill"}},
                       {**QUIET, "queue": {"waiting": 1}},
                       {**QUIET, "status": "stopped"}, {**QUIET, "ready": False}, None):
            warmer = self.build()
            warmer.tick(report)
            self.assertEqual(self.fake.calls, [], report)

    def test_it_never_runs_on_battery_power(self):
        warmer = self.build(power="battery")
        warmer.tick(QUIET)
        self.assertEqual(self.fake.calls, [])

    def test_nothing_remembered_means_nothing_to_do(self):
        warmer = self.build(entries=[])
        warmer.tick(QUIET)
        self.assertEqual(self.fake.calls, [])

    def test_it_warms_one_variant_per_tick_records_it_and_moves_on(self):
        warmer = self.build()
        warmer.tick(QUIET)
        warmer.tick(QUIET)
        warmer.tick(QUIET)
        self.assertEqual([(c[0], c[1]) for c in self.fake.calls], [("qwencode", "low"), ("qwencode", "medium"), ("qwencode", "high")])
        self.assertEqual(self.names().count("auto_warm_done"), 3)
        self.assertEqual(self.fake.saved, 3)

    def test_everything_warmed_recently_is_left_alone(self):
        warmer = self.build()
        for _ in range(8):
            warmer.tick(QUIET)
        self.assertEqual(len(self.fake.calls), 8, "2 clients x 4 efforts")
        warmer.tick(QUIET)
        self.assertEqual(len(self.fake.calls), 8)
        warmer.clock.now += 7 * 3600           # past the refresh interval: touched again so SSD eviction sees them as fresh
        warmer.tick(QUIET)
        self.assertEqual(len(self.fake.calls), 9)

    def test_a_queued_real_request_drops_the_running_warmup(self):
        warmer = self.build(thread=LiveThread)
        warmer.tick(QUIET)
        self.assertTrue(warmer.running)
        abort = self.fake.calls[0][2] if self.fake.calls else warmer.abort
        warmer.tick({**QUIET, "queue": {"waiting": 1}})
        self.assertTrue(warmer.abort.is_set())
        self.assertIn("auto_warm_yield", self.names())

    def test_an_aborted_warmup_is_not_recorded_as_done(self):
        warmer = self.build(raises=FakeWarm.WarmAborted())
        warmer.tick(QUIET)
        self.assertIn("auto_warm_aborted", self.names())
        self.assertEqual(self.fake.saved, 0)

    def test_a_safety_stop_ends_auto_warming_until_the_server_restarts_and_is_never_retried(self):
        warmer = self.build(raises=FakeWarm.WarmStop("memory safety stop"))
        warmer.tick(QUIET)
        warmer.tick(QUIET)
        warmer.tick(QUIET)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn("auto_warm_stopped", self.names())
        self.assertEqual(self.fake.saved, 0)
        warmer.reset_for_new_server()
        self.fake.raises = None
        warmer.tick(QUIET)
        self.assertEqual(len(self.fake.calls), 2)

    def test_any_other_failure_also_stops_it_until_restart(self):
        warmer = self.build(raises=RuntimeError("boom"))
        warmer.tick(QUIET)
        warmer.tick(QUIET)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn("auto_warm_failed", self.names())

    def test_overnight_turns_it_on_by_default_and_a_single_run_does_not(self):
        parser = O.build_parser()
        self.assertIsNone(parser.parse_args([]).auto_warm)
        self.assertFalse(parser.parse_args(["--no-auto-warm"]).auto_warm)
        self.assertTrue(parser.parse_args(["--auto-warm"]).auto_warm)

    def test_the_supervisor_cancels_a_warmup_when_the_server_goes_away(self):
        launches = [(30, 0)]
        clock = FakeClock()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        class SpyWarmer:
            def __init__(self):
                self.cancelled, self.ticks, self.resets = 0, 0, 0

            def cancel(self):
                self.cancelled += 1

            def tick(self, report):
                self.ticks += 1

            def reset_for_new_server(self):
                self.resets += 1

        spy = SpyWarmer()
        sup = O.Supervisor(args(overnight=False), popen=lambda c, **k: FakeChild(clock, *launches.pop(0)), sleep=clock.sleep, clock=clock,
                           health_fn=lambda h, p: {"status": "ok", "ready": True}, swap_fn=lambda: 500.0,
                           power_fn=lambda: {"source": "ac"}, log_dir=tmp.name, port_fn=lambda h, p: False, warmer=spy)
        sup.run()
        self.assertGreaterEqual(spy.cancelled, 1)
        self.assertGreaterEqual(spy.ticks, 1)
        self.assertEqual(spy.resets, 1)


class ParserTests(unittest.TestCase):
    def test_defaults_are_the_conservative_launch(self):
        ns = O.build_parser().parse_args([])
        self.assertEqual((ns.port, ns.pool_gb, ns.overnight), (8080, "0", False))


if __name__ == "__main__":
    unittest.main()
