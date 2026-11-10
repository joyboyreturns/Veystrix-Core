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


if __name__ == "__main__":
    unittest.main()
