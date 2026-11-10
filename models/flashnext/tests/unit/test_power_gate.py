"""Power start gate: battery use is allowed, a battery below the floor refuses the start. Offline; no model."""
import unittest
from unittest import mock

from models.flashnext import overnight, power_gate as G

BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=4653155)\t{pct}%; discharging; 3:10 remaining present: true\n"
AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=4653155)\t{pct}%; charging; 0:40 remaining present: true\n"
DESKTOP = "Now drawing from 'AC Power'\n"


class ParseTests(unittest.TestCase):
    def test_sources_and_percent(self):
        self.assertEqual(G.parse(BATT.format(pct=24)), ("battery", 24))
        self.assertEqual(G.parse(AC.format(pct=80)), ("ac", 80))
        self.assertEqual(G.parse(DESKTOP), ("ac", None))
        self.assertEqual(G.parse(""), ("unknown", None))


class EvaluateTests(unittest.TestCase):
    def test_battery_at_or_above_the_floor_may_start(self):
        for pct in (25, 26, 60, 100):
            self.assertTrue(G.evaluate("battery", pct).ok, pct)

    def test_battery_below_the_floor_is_refused_with_an_actionable_message(self):
        for pct in (0, 10, 24):
            verdict = G.evaluate("battery", pct)
            self.assertFalse(verdict.ok, pct)
            self.assertIn(f"{pct}%", verdict.message)
            self.assertIn("25%", verdict.message)
            self.assertIn("Plug in", verdict.message)

    def test_ac_power_always_passes_even_at_a_low_charge(self):
        self.assertTrue(G.evaluate("ac", 3).ok)
        self.assertTrue(G.evaluate("ac", None).ok)

    def test_unreadable_power_fails_closed(self):
        self.assertFalse(G.evaluate("battery", None).ok)
        self.assertFalse(G.evaluate("unknown", None).ok)

    def test_floor_is_configurable_and_zero_disables_it(self):
        self.assertEqual(G.minimum_percent({}), 25)
        self.assertEqual(G.minimum_percent({G.MIN_BATTERY_ENV: "40"}), 40)
        self.assertEqual(G.minimum_percent({G.MIN_BATTERY_ENV: "junk"}), 25)
        self.assertFalse(G.evaluate("battery", 35, 40).ok)
        self.assertTrue(G.evaluate("battery", 1, 0).ok)

    def test_require_ac_overrides_battery_permission(self):
        self.assertFalse(G.check({"VEYSTRIX_REQUIRE_AC": "1"}, lambda: BATT.format(pct=90)).ok)
        self.assertTrue(G.check({"VEYSTRIX_REQUIRE_AC": "1"}, lambda: AC.format(pct=10)).ok)
        self.assertTrue(G.check({"VEYSTRIX_REQUIRE_AC": "0"}, lambda: BATT.format(pct=90)).ok)

    def test_live_watchdog_honours_power_options(self):
        from models.flashnext.tests.bench import prefill_memory_harness as H
        sample = {"power_source": "battery", "battery_percent": 80, "low_power": True}
        self.assertEqual(H.Watchdog(H.SafetyPolicy(), sample).check(sample), "Low Power Mode is on")
        self.assertIsNone(H.Watchdog(H.SafetyPolicy(allow_low_power=True), sample).check(sample))
        self.assertIn("AC power", H.Watchdog(H.SafetyPolicy(allow_low_power=True, require_ac=True), sample).check(sample))

    def test_check_reads_through_the_runner(self):
        self.assertFalse(G.check({}, lambda: BATT.format(pct=24)).ok)
        self.assertTrue(G.check({}, lambda: BATT.format(pct=25)).ok)
        self.assertTrue(G.check({}, lambda: AC.format(pct=10)).ok)
        self.assertFalse(G.check({}, lambda: (_ for _ in ()).throw(OSError("no pmset"))).ok)


class SupervisorTests(unittest.TestCase):
    def test_power_refusal_is_not_restarted(self):
        action, cooldown, reason = overnight.Policy().decide(G.POWER_EXIT_CODE, 0.0)
        self.assertEqual((action, cooldown), ("stop", 0.0))
        self.assertIn("battery", reason)


class SessionGateTests(unittest.TestCase):
    def run_gate(self, text, prefs=None, benchmarking=False, env=None):
        from veystrix import session
        import io
        display = io.StringIO()
        with mock.patch.object(G, "read_power", lambda runner=None: G.parse(text)), mock.patch.dict("os.environ", env or {}, clear=False):
            session._enforce_power_gate(prefs or {"model": "flashnext"}, benchmarking, display)
        return display.getvalue()

    def test_battery_start_is_announced_and_allowed(self):
        self.assertIn("on battery, 60%", self.run_gate(BATT.format(pct=60)))

    def test_low_battery_exits_with_the_power_code_before_anything_loads(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_gate(BATT.format(pct=24))
        self.assertEqual(raised.exception.code, G.POWER_EXIT_CODE)

    def test_other_runtimes_and_benchmarks_are_not_gated_here(self):
        self.run_gate(BATT.format(pct=5), prefs={"model": "qwen27b"})
        self.run_gate(BATT.format(pct=5), benchmarking=True)


class BundledLaunchTests(unittest.TestCase):
    """The packaged app has no chat.sh and no source files: the supervisor must launch the compiled CLI module."""

    def test_source_checkout_uses_chat_sh(self):
        args = overnight.build_parser().parse_args([])
        command = overnight.server_command(args)
        self.assertTrue(command[0].endswith("chat.sh"), command[:2])

    def test_bundle_launches_the_cli_module_with_the_bundled_interpreter(self):
        import sys
        import tempfile
        from pathlib import Path

        args = overnight.build_parser().parse_args([])
        with tempfile.TemporaryDirectory() as root, mock.patch.object(overnight, "REPO", Path(root) / "lib"), \
             mock.patch.object(sys, "executable", str(Path(root) / "python/bin/python3")):
            self.assertTrue(overnight.bundled())
            command = overnight.server_command(args)
            self.assertEqual(command[:3], [sys.executable, "-m", "veystrix.cli"])
        self.assertEqual(command[3:5], ["--mode", "agentic"])
        self.assertIn("127.0.0.1", command)


if __name__ == "__main__":
    unittest.main()
