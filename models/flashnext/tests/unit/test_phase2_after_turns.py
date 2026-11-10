import unittest
from unittest import mock

from models.flashnext import purge_recovery as P
from models.flashnext.tests.bench import bench_phase2_after_turns as B


class BoundaryTests(unittest.TestCase):
    def reading(self, free_gib=15, swap_mib=100, pressure=1):
        return P.Reading(page=2**20, free=free_gib * 1024, swap_used=swap_mib * 2**20, pressure=pressure)

    def test_headroom_fails_closed_on_swap_pressure_and_shortfall(self):
        self.assertTrue(B.boundary_headroom(self.reading(), 9000)["admitted"])
        for reading in (self.reading(swap_mib=2304), self.reading(pressure=2), self.reading(free_gib=3)):
            self.assertFalse(B.boundary_headroom(reading, 9000)["admitted"])

    def test_larger_prompt_has_larger_reserve(self):
        small = B.boundary_headroom(self.reading(), 8000)
        large = B.boundary_headroom(self.reading(), 10000)
        self.assertGreater(large["required_bytes"], small["required_bytes"])
        self.assertGreaterEqual(large["snapshot_budget"], small["snapshot_budget"])

    def test_refused_boundary_never_sends_8k_request(self):
        health = {"memory_policy": {"swap_ceiling_mib": 3072.0, "swap_ceiling_margin_mib": 768.0}}
        response = {"choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}], "usage": {}}
        safe = {"power_source": "ac", "battery_percent": 80, "battery_state": "charging", "low_power": False,
                "thermal_readable": True, "thermal_warning": False, "thermal_speed_limit": None,
                "pressure_level": 1, "swap_used_mb": 100, "available_mb": 10000, "compressor_mb": 100}
        report = {}
        with mock.patch.object(B.G, "http_json", side_effect=[health, response, response]) as send, \
             mock.patch.object(B.G, "machine_sample", return_value=safe), \
             mock.patch.object(B.request_admission, "strict_read", return_value=self.reading(free_gib=2)):
            with self.assertRaisesRegex(RuntimeError, "postponed"):
                B.make_driver({"max_tokens": 8}, 8100, 9124, report)("http://localhost")
        self.assertEqual(send.call_count, 3)  # health and two short turns only
        self.assertFalse(report["pre_8k_admission"]["admitted"])

    def test_invalid_guard_refuses_before_any_turn(self):
        with mock.patch.object(B.G, "http_json", return_value={"memory_policy": {}}) as send:
            with self.assertRaisesRegex(RuntimeError, "absolute guard"):
                B.make_driver({"max_tokens": 8}, 8100, 9124, {})("http://localhost")
        self.assertEqual(send.call_count, 1)


class PowerModeTests(unittest.TestCase):
    def sample(self, source="battery", percent=60, state="discharging", **extra):
        row = {"power_source": source, "battery_percent": percent, "battery_state": state, "low_power": False,
               "thermal_readable": True, "thermal_warning": False, "thermal_speed_limit": None,
               "pressure_level": 1, "swap_used_mb": 100, "available_mb": 10000, "compressor_mb": 100}
        row.update(extra)
        return row

    def test_battery_requires_explicit_mode_and_floor(self):
        self.assertIn("AC power", B.stop_reason(self.sample(), None, "ac"))
        self.assertIsNone(B.stop_reason(self.sample(percent=60), None, "battery-ok", 40))
        self.assertIn("40% floor", B.stop_reason(self.sample(percent=39), None, "battery-ok", 40))
        self.assertIn("25%", B.stop_reason(self.sample(percent=24), None, "battery-ok", 25))

    def test_battery_mode_keeps_other_safety_checks(self):
        for extra in ({"low_power": True}, {"thermal_warning": True}, {"thermal_speed_limit": 80},
                      {"pressure_level": 2}, {"swap_used_mb": 2304}, {"available_mb": 2000}):
            self.assertIsNotNone(B.stop_reason(self.sample(**extra), None, "battery-ok", 40), extra)

    def test_ac_mode_is_unchanged_for_charging_sample(self):
        self.assertIsNone(B.stop_reason(self.sample("ac", 80, "charging"), None, "ac"))

    def test_driver_stops_before_8k_when_battery_drops_below_floor(self):
        health = {"memory_policy": {"swap_ceiling_mib": 3072.0, "swap_ceiling_margin_mib": 768.0}}
        response = {"choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}], "usage": {}}
        readings = [self.sample(percent=45), self.sample(percent=44), self.sample(percent=43), self.sample(percent=39)]
        report = {}
        with mock.patch.object(B.G, "http_json", side_effect=[health, response, response]) as send, \
             mock.patch.object(B.G, "machine_sample", side_effect=readings), \
             mock.patch.object(B.request_admission, "strict_read", return_value=P.Reading(
                 page=2**20, free=15 * 1024, swap_used=100 * 2**20, pressure=1)):
            with self.assertRaisesRegex(RuntimeError, "pre-8K safety"):
                B.make_driver({"max_tokens": 8}, 8100, 9124, report, "battery-ok", 40)("http://localhost")
        self.assertEqual(send.call_count, 3)
        self.assertTrue(report["pre_8k_admission"]["admitted"])


if __name__ == "__main__":
    unittest.main()
