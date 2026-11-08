"""Autonomous memory preparation, tiered admission and the optional app closer. Fake machines only: nothing is purged,
loaded or closed, and no process is signalled."""
import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from models.flashnext import app_closer as A
from models.flashnext import prefill_readiness as R
from models.flashnext import purge_recovery as P
from models.flashnext import resource_prep as RP

GIB = 2 ** 30
TOTAL = 16 * GIB
ROOT = Path(__file__).resolve().parents[4]


class Machine:
    """Byte-granular fake machine. A purge moves file cache into free memory; anonymous memory is never touched."""

    def __init__(self, free=8.4, cache=4.5, anonymous=3.5, wired=1.96, compressor=0.8, effect=1.0, pressure=1, permit=True):
        g = lambda value: int(value * GIB)                                              # noqa: E731
        self.free, self.cache, self.anonymous = g(free), g(cache), g(anonymous)
        self.wired, self.compressor = g(wired), g(compressor)
        self.effect, self.pressure, self.permit = effect, pressure, permit
        self.t = 1000.0
        self.purge_calls = 0
        self.commands = []
        self.sleeps = []
        self.after_purge = None

    def read(self):
        return P.Reading(page=1, free=self.free, file_backed=self.cache, anonymous=self.anonymous, wired=self.wired,
                         compressor=self.compressor, pressure=self.pressure)

    def runner(self, command, timeout):
        self.commands.append(tuple(command))
        if tuple(command) == P.CAPABILITY_COMMAND:
            return SimpleNamespace(returncode=0 if self.permit else 1, stderr="")
        assert tuple(command) == P.PURGE_COMMAND, command
        self.purge_calls += 1
        moved = int(self.cache * self.effect)
        self.free += moved
        self.cache -= moved
        if self.after_purge:
            self.after_purge(self)
        return SimpleNamespace(returncode=0, stderr="")

    def clock(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds

    def recovery(self, **kwargs):
        cfg = P.Config(settle_s=0.0)
        recovery = P.PurgeRecovery(cfg, read=self.read, runner=self.runner, clock=self.clock, sleep=self.sleep,
                                   disk_read=lambda: 0, **kwargs)
        recovery._blocker = R.purge_blocker_tiered
        return recovery

    def preparer(self, agentic=True, recovery="default", closer=None, **kwargs):
        if recovery == "default":
            recovery = self.recovery()
        return RP.ResourcePreparer(agentic=agentic, recovery=recovery, closer=closer, read=self.read, total=TOTAL,
                                   processes=lambda: [], clock=self.clock, sleep=self.sleep, pause_s=0.0, **kwargs)


class TierTests(unittest.TestCase):
    def test_the_budget_follows_the_measured_peaks_and_never_goes_below_them_or_above_the_standard_figure(self):
        peaks = {tokens: R.cold_prefill_peak_bytes(tokens) / GIB for tokens in (1, 500, 898, 1024, 1536, 2046, 4096, 10000)}
        self.assertEqual(peaks[898], 5.0)                          # 4.62 GiB measured + 0.3 process, rounded up to 0.25
        self.assertEqual(peaks[1], peaks[898])                     # nothing below the smallest measurement is assumed
        self.assertEqual(peaks[2046], 6.0)
        self.assertEqual(peaks[10000], 6.0)
        ordered = [peaks[t] for t in (898, 1024, 1536, 2046)]
        self.assertEqual(ordered, sorted(ordered))
        for tokens in (898, 1024, 1536, 2046):                     # always at or above the measured line plus overhead
            line = 4.62 + (tokens - 898) * (5.40 - 4.62) / (2046 - 898) + 0.3
            self.assertGreaterEqual(peaks[tokens], min(line, 6.0))
        self.assertTrue(all(value % 0.25 == 0 for value in peaks.values()))

    def test_the_tiers_run_from_the_standard_size_down_and_the_standard_tier_is_unchanged(self):
        names = [t.name for t in R.TIERS]
        self.assertEqual(names, ["standard", "compact", "minimal"])
        standard = R.TIERS[0]
        self.assertEqual((standard.context_limit, standard.peak_bytes), (8192, R.COLD_PREFILL_PEAK_BYTES))
        self.assertEqual([t.context_limit for t in R.TIERS], sorted((t.context_limit for t in R.TIERS), reverse=True))
        self.assertTrue(all(t.max_cold_prefill + 256 <= t.context_limit for t in R.TIERS))   # the output reserve fits

    def reading(self, anonymous):
        return P.Reading(page=1, free=11 * GIB, anonymous=int(anonymous * GIB), wired=int(1.96 * GIB),
                         compressor=int(0.8 * GIB))

    def test_ample_memory_admits_the_standard_tier(self):
        verdict = R.evaluate_tiered(self.reading(2.0), TOTAL)
        self.assertTrue(verdict.admitted)
        self.assertEqual(verdict.tier.name, "standard")

    def test_a_smaller_tier_needs_no_less_memory_than_standard_so_tight_memory_is_not_rescued_by_degrading(self):
        """Measured: the gate figure falls 6.1-6.65 GiB in every real run from a 60-token CHAT turn to a 3,935-token prefill, so
        compact and minimal need the same start figure as standard. Tier degradation is for the context a launch must serve,
        not a way to squeeze under a memory gate."""
        needs = {t.name: R.M.required_available_gib(t.max_cold_prefill) for t in R.TIERS}
        self.assertLess(max(needs.values()) - min(needs.values()), 0.05, needs)        # 4,096 vs the 3,935-token reference
        verdict = R.evaluate_tiered(self.reading(3.5), TOTAL)
        self.assertTrue(verdict.admitted)
        self.assertEqual(verdict.tier.name, "standard")
        legacy = verdict.checks[2]
        self.assertTrue(legacy.ok and legacy.values["advisory"])                          # still reported, never a veto
        self.assertEqual(verdict.as_dict()["tier_limits"], {"context_limit": 8192, "max_cold_prefill": 4096})

    def test_required_start_figure_grows_with_the_cold_prefill_above_the_measured_range_and_is_never_below_nine(self):
        gib = R.M.required_available_gib
        self.assertGreaterEqual(gib(1024), 9.0)
        self.assertLessEqual(gib(1024), gib(4096) + 1e-9)
        self.assertLess(gib(4096), gib(10240))
        self.assertLess(gib(10240), gib(14336))
        self.assertAlmostEqual(gib(4096), 9.02, places=1)

    def test_the_mandatory_gate_and_pressure_still_refuse_every_tier(self):
        short = P.Reading(page=1, free=int(8.4 * GIB), anonymous=int(2.0 * GIB), wired=int(1.96 * GIB), compressor=GIB)
        verdict = R.evaluate_tiered(short, TOTAL)
        self.assertFalse(verdict.admitted)
        self.assertIsNone(verdict.tier)
        self.assertEqual(len(verdict.tried), 3)
        self.assertIn("start gate", " ".join(c.name for c in verdict.blocking))
        pressured = P.Reading(page=1, free=11 * GIB, anonymous=2 * GIB, wired=2 * GIB, compressor=GIB, pressure=2)
        self.assertFalse(R.evaluate_tiered(pressured, TOTAL).admitted)

    def test_a_gate_shortfall_refuses_every_tier_and_a_purge_is_blocked_only_when_the_cache_cannot_close_it(self):
        short = P.Reading(page=1, free=int(8.4 * GIB), anonymous=int(5.5 * GIB), wired=int(1.96 * GIB), compressor=int(0.8 * GIB),
                          file_backed=int(0.2 * GIB))
        verdict = R.evaluate_tiered(short, TOTAL)
        self.assertFalse(verdict.admitted)
        self.assertEqual([t["admitted"] for t in verdict.tried], [False, False, False])
        self.assertIn("at most", R.purge_blocker_tiered(short, TOTAL))                    # 0.2 GiB of cache cannot close 0.6 GiB
        recoverable = P.Reading(page=1, free=int(8.4 * GIB), anonymous=int(5.5 * GIB), wired=int(1.96 * GIB),
                                compressor=int(0.8 * GIB), file_backed=int(4.5 * GIB))
        self.assertIsNone(R.purge_blocker_tiered(recoverable, TOTAL))                      # a purge could still help here


class PreparationTests(unittest.TestCase):
    def test_recoverable_file_cache_is_purged_and_the_launch_is_admitted_automatically(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=3.5)
        report = machine.preparer().prepare()
        self.assertTrue(report.admitted, report.failure_report())
        self.assertEqual((report.tier.name, report.rounds, machine.purge_calls), ("standard", 1, 1))
        purge = next(s for s in report.steps if s.name == "file-cache purge")
        self.assertEqual(purge.action, "done")
        self.assertGreater(purge.available_after, purge.available_before + GIB)
        owned = next(s for s in report.steps if s.name == "owned memory")
        self.assertEqual(owned.action, "nothing")                   # nothing is loaded yet: recorded, not pretended

    def test_an_already_admitted_machine_does_nothing(self):
        machine = Machine(free=12.0, cache=1.0, anonymous=2.0)
        report = machine.preparer().prepare()
        self.assertTrue(report.admitted)
        self.assertEqual((report.rounds, report.steps, machine.purge_calls), (0, [], 0))

    def test_high_anonymous_memory_no_longer_blocks_the_purge_that_closes_a_recoverable_gap(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=5.5)
        report = machine.preparer().prepare()
        self.assertTrue(report.admitted, report.failure_report())
        self.assertEqual(machine.purge_calls, 1)
        self.assertEqual(report.tier.name, "standard")

    def test_when_file_cache_cannot_close_the_gap_no_purge_runs_and_the_failure_is_one_actionable_report(self):
        machine = Machine(free=7.0, cache=0.3, anonymous=5.5)
        report = machine.preparer().prepare()
        self.assertFalse(report.admitted)
        self.assertEqual(machine.purge_calls, 0, "a purge that cannot close the gap must not be attempted")
        self.assertEqual(report.rounds, 1)
        self.assertIn("recovery cannot help", report.stop_reason)
        text = report.failure_report()
        for needle in ("REFUSED", "short by", "Nothing was loaded", "nothing was terminated", "close-apps.json"):
            self.assertIn(needle, text)
        self.assertEqual(text.count("Nothing was loaded"), 1)
        self.assertNotIn(P.PURGE_COMMAND, machine.commands)

    def test_an_ineffective_purge_is_tried_once_and_then_backs_off_within_bounded_rounds(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=3.5, effect=0.0)
        report = machine.preparer(max_rounds=3).prepare()
        self.assertFalse(report.admitted)
        self.assertEqual(machine.purge_calls, 1)                    # no endless purge loop
        self.assertLessEqual(report.rounds, 3)
        self.assertLessEqual(len([s for s in machine.sleeps if s]), 3)

    def test_settle_waits_back_off_between_rounds(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=3.5)
        machine.after_purge = lambda m: setattr(m, "pressure", 2)   # the purge helps, then macOS reports pressure
        report = machine.preparer().prepare()
        self.assertFalse(report.admitted)
        self.assertIn("recovery cannot help", report.stop_reason)
        waits = [s for s in machine.sleeps if s]                    # the purge's own zero settle is not a wait
        self.assertEqual(waits[0], 2.0)                             # first settle after a helpful step
        self.assertLessEqual(len(waits), 3)
        self.assertEqual(machine.purge_calls, 1)

    def test_the_deadline_bounds_the_whole_procedure(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=3.5, effect=0.0)
        report = machine.preparer(deadline_s=1.0, settle_s=5.0).prepare()
        self.assertFalse(report.admitted)
        self.assertLessEqual(report.rounds, 2)
        self.assertTrue("deadline" in report.stop_reason or "cannot help" in report.stop_reason)

    def test_no_purge_while_a_request_or_load_is_active(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=3.5)
        activity = P.Activity()
        recovery = machine.recovery(activity=activity)
        with activity.request():
            report = machine.preparer(recovery=recovery).prepare()
        self.assertFalse(report.admitted)
        self.assertEqual(machine.purge_calls, 0)
        self.assertNotIn(P.PURGE_COMMAND, machine.commands)
        with activity.loading_model():
            self.assertEqual(machine.preparer(recovery=machine.recovery(activity=activity)).prepare().admitted, False)
        self.assertEqual(machine.purge_calls, 0)

    def test_no_purge_under_critical_pressure_or_without_permission(self):
        for options in ({"pressure": 4}, {"permit": False}):
            with self.subTest(options):
                machine = Machine(free=8.4, cache=4.5, anonymous=3.5, **options)
                report = machine.preparer().prepare()
                self.assertFalse(report.admitted)
                self.assertEqual(machine.purge_calls, 0)

    def test_chat_prepares_only_for_the_gate_and_never_refuses(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=7.0)       # lots of anonymous memory: irrelevant to CHAT's gate
        report = machine.preparer(agentic=False, recovery=P.PurgeRecovery(
            P.Config(settle_s=0.0), read=machine.read, runner=machine.runner, clock=machine.clock, sleep=machine.sleep,
            disk_read=lambda: 0)).prepare()
        self.assertTrue(report.admitted)
        self.assertEqual(machine.purge_calls, 1)
        short = Machine(free=7.0, cache=0.1, anonymous=3.0)
        report = short.preparer(agentic=False).prepare()             # cannot reach the gate: reported, the guard decides
        self.assertFalse(report.admitted)

    def test_the_log_is_private_and_records_the_tier(self):
        machine = Machine(free=8.4, cache=4.5, anonymous=3.5)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sub", "prep.jsonl")
            with mock.patch.dict(os.environ, {"FLASHNEXT_PREP_LOG": path}):
                machine.preparer().prepare()
            self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
            row = json.loads(Path(path).read_text())
            self.assertEqual((row["admitted"], row["tier"]), (True, "standard"))

    def test_nothing_in_the_preparation_can_signal_or_kill_a_process(self):
        for name in ("resource_prep.py", "app_closer.py"):
            source = (ROOT / "models" / "flashnext" / name).read_text()
            code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"', "'")))
            for forbidden in ("os.kill", "pkill", "killall", "SIGKILL", "SIGTERM", ".terminate(", ".kill(", "signal."):
                self.assertNotIn(forbidden, code, f"{name} must never signal a process ({forbidden})")


class AppCloserTests(unittest.TestCase):
    def closer(self, config=None, running=("Slack", "Mail"), stubborn=(), clock_step=1.0):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "close-apps.json")
        if config is not None:
            Path(path).write_text(json.dumps(config))
        state = {"running": set(running), "calls": [], "t": 0.0}

        def runner(command, timeout=0):
            state["calls"].append(tuple(command))
            if command[0] == "pgrep":
                return SimpleNamespace(returncode=0 if command[2] in state["running"] else 1)
            if command[0] == "osascript":
                name = re.search(r'application "(.+?)" to quit', command[2])[1]
                if name not in stubborn:
                    state["running"].discard(name)
                return SimpleNamespace(returncode=0)
            raise AssertionError(command)

        def clock():
            state["t"] += clock_step
            return state["t"]

        return A.AppCloser(path, runner=runner, sleep=lambda s: None, clock=clock), state

    def test_it_is_off_without_an_explicit_authorization(self):
        for config in (None, {"apps": ["Slack"]}, {"enabled": False, "apps": ["Slack"]}, {"enabled": "yes", "apps": ["Slack"]},
                       {"enabled": 1, "apps": ["Slack"]}, {"enabled": True, "apps": []}, {"enabled": True}):
            with self.subTest(config=config):
                closer, state = self.closer(config)
                self.assertFalse(closer.enabled)
                self.assertEqual(closer.close_allowed(), [])
                self.assertEqual(state["calls"], [])               # nothing at all is run

    def test_only_listed_running_apps_are_asked_to_quit_gracefully(self):
        closer, state = self.closer({"enabled": True, "apps": ["Slack", "Notes"]})
        rows = {r["app"]: r for r in closer.close_allowed()}
        self.assertTrue(rows["Slack"]["requested"] and rows["Slack"]["closed"])
        self.assertEqual(rows["Notes"]["reason"], "not running")
        self.assertIn(("osascript", "-e", 'tell application "Slack" to quit'), state["calls"])
        self.assertNotIn("Mail", " ".join(" ".join(c) for c in state["calls"]))            # unlisted apps are never touched
        self.assertTrue(all(c[0] in ("pgrep", "osascript") for c in state["calls"]))       # no kill, no signal, no shell

    def test_protected_applications_are_never_closed_even_when_listed(self):
        names = ["WindowServer", "Finder", "Siri", "Terminal", "claude", "Dock", "loginwindow"]
        closer, state = self.closer({"enabled": True, "apps": names}, running=names)
        rows = closer.close_allowed()
        self.assertTrue(all(not r["requested"] and "protected" in r["reason"] for r in rows))
        self.assertFalse(any(c[0] == "osascript" for c in state["calls"]))

    def test_names_that_could_inject_applescript_are_rejected(self):
        hostile = ['Slack" to quit\ndo shell script "x', "A;B", 'x"', "../etc", "$(id)"]
        closer, state = self.closer({"enabled": True, "apps": hostile}, running=hostile)
        self.assertTrue(all(r["reason"] == "name not allowed" for r in closer.close_allowed()))
        self.assertEqual(state["calls"], [])

    def test_an_app_that_keeps_running_to_ask_about_unsaved_work_is_left_alone(self):
        closer, state = self.closer({"enabled": True, "apps": ["Slack"], "wait_seconds": 5}, stubborn=("Slack",))
        row = closer.close_allowed()[0]
        self.assertTrue(row["requested"])
        self.assertFalse(row["closed"])
        self.assertIn("not forced", row["reason"])
        self.assertEqual(sum(1 for c in state["calls"] if c[0] == "osascript"), 1)          # asked once, never again

    def test_the_preparer_uses_it_only_when_enabled_and_reports_what_it_did(self):
        closer, _ = self.closer({"enabled": True, "apps": ["Slack"]})
        machine = Machine(free=8.4, cache=0.0, anonymous=5.0, effect=0.0)
        machine.closes = 0

        def close_allowed():
            machine.closes += 1
            machine.anonymous -= int(1.2 * GIB)                      # the closed app's memory goes back
            return [{"app": "Slack", "requested": True, "closed": True, "reason": ""}]

        closer.close_allowed = close_allowed
        report = machine.preparer(recovery=None, closer=closer).prepare()
        step = next(s for s in report.steps if s.name == "close allowed apps")
        self.assertIn("Slack", step.detail)
        self.assertGreaterEqual(machine.closes, 1)
        off, _ = self.closer(None)
        machine2 = Machine(free=8.4, cache=0.0, anonymous=5.0)
        report2 = machine2.preparer(recovery=None, closer=off).prepare()
        self.assertFalse(any(s.name == "close allowed apps" for s in report2.steps))


class SessionStartupTests(unittest.TestCase):
    PREFS = {"model": "flashnext"}

    def report(self, admitted, tier=None):
        reading = P.Reading(page=1, free=11 * GIB, anonymous=3 * GIB, wired=2 * GIB, compressor=GIB)
        verdict = R.evaluate_tiered(reading, TOTAL)
        if tier is not None:
            verdict.tier = tier
        return RP.PrepReport(admitted, verdict, [], 1, 0.1, "test")

    def run_prepare(self, agentic, report):
        from macqwen import session

        env = {"VEYSTRIX_MODE": "agentic"} if agentic else {"VEYSTRIX_MODE": "chat"}
        with mock.patch.dict(os.environ, env), mock.patch.object(RP.ResourcePreparer, "prepare", return_value=report), \
                mock.patch.object(P, "from_environment", return_value=None), mock.patch.object(R, "log"):
            return session._prepare_startup(self.PREFS, False, io.StringIO())

    def test_agentic_that_cannot_be_admitted_exits_69_with_one_report_and_loads_nothing(self):
        report = self.report(False)
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as raised:
            self.run_prepare(True, report)
        self.assertEqual(raised.exception.code, R.ADMISSION_EXIT_CODE)
        self.assertEqual(err.getvalue().count("Nothing was loaded"), 1)

    def test_an_admitted_tier_sets_the_enforced_limits_and_never_raises_them(self):
        from macqwen import session

        compact = R.TIERS[1]
        with mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "agentic", "VEYSTRIX_CONTEXT_LIMIT": "4096"}):
            session._apply_tier(SimpleNamespace(tier=compact), io.StringIO())
            self.assertEqual((os.environ["VEYSTRIX_CONTEXT_LIMIT"], os.environ["VEYSTRIX_MAX_COLD_PREFILL"]), ("2048", "1536"))
        with mock.patch.dict(os.environ, {"VEYSTRIX_CONTEXT_LIMIT": "1024"}):
            session._apply_tier(SimpleNamespace(tier=compact), io.StringIO())
            self.assertEqual(os.environ["VEYSTRIX_CONTEXT_LIMIT"], "1024")        # an owner's smaller limit wins
        with mock.patch.dict(os.environ, {"VEYSTRIX_CONTEXT_LIMIT": "4096"}, clear=False):
            os.environ.pop("VEYSTRIX_MAX_COLD_PREFILL", None)
            session._apply_tier(SimpleNamespace(tier=None), io.StringIO())
            self.assertNotIn("VEYSTRIX_MAX_COLD_PREFILL", os.environ)

    def test_chat_never_exits_here_and_never_gets_a_tier(self):
        recovery, verdict = self.run_prepare(False, self.report(False))
        self.assertIsNone(verdict)

    def test_admission_with_a_prepared_verdict_does_not_read_the_machine_again(self):
        from macqwen import session

        report = self.report(True, R.TIERS[2])
        verdict = report.verdict
        verdict.admitted = True
        with mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "agentic", R.LOG_FLAG: os.devnull}), \
                mock.patch.object(R, "live", side_effect=AssertionError("must not read again")):
            self.assertIs(session._admit_agentic(self.PREFS, False, io.StringIO(), verdict), verdict)
            self.assertEqual(os.environ["VEYSTRIX_MAX_COLD_PREFILL"], "1024")
            os.environ.pop("VEYSTRIX_MAX_COLD_PREFILL", None)
            os.environ.pop("VEYSTRIX_CONTEXT_LIMIT", None)

    def test_the_cli_turns_the_preparation_purge_on_by_default_and_honours_an_opt_out(self):
        from macqwen import cli

        for argv, environ, expected in ((["--mode", "chat"], {}, "1"), (["--mode", "agentic"], {}, "1"),
                                        (["--mode", "chat", "--no-auto-purge"], {}, None),
                                        (["--mode", "chat"], {"FLASHNEXT_AUTO_PURGE": "0"}, "0")):
            with self.subTest(argv=argv, environ=environ), mock.patch.dict(os.environ, environ, clear=False):
                if "FLASHNEXT_AUTO_PURGE" not in environ:
                    os.environ.pop("FLASHNEXT_AUTO_PURGE", None)
                _, env = cli.command(argv)
                self.assertEqual(env.get("FLASHNEXT_AUTO_PURGE"), expected)


if __name__ == "__main__":
    unittest.main()
