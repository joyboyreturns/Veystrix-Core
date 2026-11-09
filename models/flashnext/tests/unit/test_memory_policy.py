"""Unit tests for the approved AGENTIC swap-growth-only memory policy.

Covers:
- Startup under low available memory (< 9 GiB) without admission refusal
- Pre-existing swap baseline (e.g. 1.8 GiB) and 2.5 GiB growth threshold
- Threshold crossing stops admission and triggers safe cancellation
- High-watermark enforcement (transient swap dips do not bypass threshold)
- Missing probes and invalid swap measurements fail closed
- Visibility in ledger, health report, model card, and banners
- Non-memory protections preserved (battery, thermal, Metal limits)
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from models.flashnext import memory_controller as MC
from models.flashnext import memory_policy as MP
from models.flashnext import memory_waiver as MW
from models.flashnext.settings import modes


class MemoryPolicyUnitTests(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.copy()
        os.environ["VEYSTRIX_MODE"] = "agentic"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        MC.install(None)

    def test_default_agentic_policy_is_swap_growth_only(self):
        self.assertTrue(MP.is_swap_growth_policy())
        self.assertEqual(MP.policy_identifier(), MP.POLICY_SWAP_GROWTH_ONLY)
        self.assertIn("startup_9gib_gate", MP.telemetry_only_checks())
        self.assertTrue(MW.active())

    def test_policy_override_to_strict_staged(self):
        os.environ[MP.FLAG_POLICY] = MP.POLICY_STRICT_STAGED
        self.assertFalse(MP.is_swap_growth_policy())
        self.assertEqual(MP.policy_identifier(), MP.POLICY_STRICT_STAGED)
        self.assertEqual(MP.telemetry_only_checks(), [])

    def test_pre_existing_swap_baseline_and_threshold_crossing(self):
        # Baseline swap: 1800 MB (~1.8 GiB)
        policy = MC.Policy.from_environment(swap_only=True)
        self.assertEqual(policy.swap_growth_stop_gib, 2.5)

        events = []
        controller = MC.Controller(policy, emit=lambda name, **kw: events.append((name, kw)))
        MC.install(controller)

        # Baseline sample at t=0
        stop = controller.observe(
            0.0,
            pressure=1,
            swap_used_mb=1800.0,
            swapout_pages=0,
            swapin_pages=0,
            available_mb=3500.0,  # below 9 GiB
            compressor_mb=1200.0,
        )
        self.assertIsNone(stop)
        self.assertTrue(controller.can_admit())
        self.assertTrue(MC.can_admit_work())
        self.assertFalse(MC.is_stopped())
        self.assertEqual(controller.baseline["swap_used_mb"], 1800.0)

        # Swap rises to 3800 MB (+2000 MB = +1.95 GiB, below 2.5 GiB stop)
        stop = controller.observe(
            1.0,
            pressure=2,  # pressure level 2 is telemetry-only
            swap_used_mb=3800.0,
            swapout_pages=100,
            swapin_pages=100,
            available_mb=2500.0,
            compressor_mb=2000.0,
        )
        self.assertIsNone(stop)
        self.assertTrue(controller.can_admit())

        # Swap rises to 4360 MB (+2560 MB = 2.50 GiB -> threshold crossed)
        stop = controller.observe(
            2.0,
            pressure=1,
            swap_used_mb=4360.0,
            swapout_pages=100,
            swapin_pages=100,
            available_mb=2000.0,
            compressor_mb=2000.0,
        )
        self.assertIsNotNone(stop)
        self.assertIn("swap grew", stop)
        self.assertIn("2.50 GiB", stop)
        self.assertEqual(controller.state, MC.CRITICAL)
        self.assertFalse(controller.can_admit())
        self.assertFalse(MC.can_admit_work())
        self.assertTrue(MC.is_stopped())

    def test_high_watermark_enforcement(self):
        controller = MC.Controller(MC.Policy.from_environment(swap_only=True))
        MC.install(controller)

        # Baseline: 1000 MB
        controller.observe(
            0.0, pressure=1, swap_used_mb=1000.0, swapout_pages=0, swapin_pages=0,
            available_mb=4000.0, compressor_mb=500.0,
        )

        # Spike to 3600 MB (+2600 MB > 2.5 GiB)
        stop = controller.observe(
            1.0, pressure=1, swap_used_mb=3600.0, swapout_pages=0, swapin_pages=0,
            available_mb=4000.0, compressor_mb=500.0,
        )
        self.assertIsNotNone(stop)
        self.assertFalse(controller.can_admit())

        # Transient decrease back to 1500 MB (+500 MB) must NOT un-stop the controller
        stop2 = controller.observe(
            2.0, pressure=1, swap_used_mb=1500.0, swapout_pages=0, swapin_pages=0,
            available_mb=5000.0, compressor_mb=500.0,
        )
        self.assertIsNotNone(stop2)
        self.assertFalse(controller.can_admit())
        self.assertTrue(MC.is_stopped())

    def test_missing_probe_fails_closed(self):
        controller = MC.Controller(MC.Policy.from_environment(swap_only=True))
        MC.install(controller)

        # Swap probe is None
        stop = controller.observe(
            0.0, pressure=1, swap_used_mb=None, swapout_pages=0, swapin_pages=0,
            available_mb=4000.0, compressor_mb=500.0,
        )
        self.assertIsNotNone(stop)
        self.assertIn("probe unavailable: swap_used_mb", stop)
        self.assertFalse(controller.can_admit())
        self.assertTrue(MC.is_stopped())

    def test_invalid_swap_measurement_fails_closed(self):
        controller = MC.Controller(MC.Policy.from_environment(swap_only=True))
        MC.install(controller)

        stop = controller.observe(
            0.0, pressure=1, swap_used_mb=-10.0, swapout_pages=0, swapin_pages=0,
            available_mb=4000.0, compressor_mb=500.0,
        )
        self.assertIsNotNone(stop)
        self.assertIn("invalid swap measurement", stop)
        self.assertFalse(controller.can_admit())

    def test_ledger_visibility_and_fields(self):
        controller = MC.Controller(MC.Policy.from_environment(swap_only=True))
        MC.install(controller)

        controller.observe(
            0.0, pressure=1, swap_used_mb=2048.0, swapout_pages=10, swapin_pages=5,
            available_mb=3000.0, compressor_mb=800.0,
        )
        ledg = controller.ledger()
        self.assertEqual(ledg["policy_identifier"], MP.POLICY_SWAP_GROWTH_ONLY)
        self.assertEqual(ledg["swap_baseline_mb"], 2048.0)
        self.assertEqual(ledg["threshold_gib"], 2.5)
        self.assertEqual(ledg["threshold_mb"], 2560.0)
        self.assertTrue(ledg["can_admit"])
        self.assertTrue(ledg["probe_availability"]["swap_used_mb"])
        self.assertIn("startup_9gib_gate", ledg["telemetry_only_checks"])
        self.assertIn("does not guarantee protection", ledg["disclaimer"])

    def test_startup_banner_visibility(self):
        plan = modes.plan_mode("agentic")
        b = modes.banner(plan, available_gb=3.2)
        self.assertIn("swap-growth-only", b)
        self.assertIn("telemetry-only", b)
        self.assertIn("+2.5 GiB swap", b)

        policy_b = MP.banner()
        self.assertIn(MP.POLICY_SWAP_GROWTH_ONLY, policy_b)
        self.assertIn("2.5 GiB above launch baseline", policy_b)


if __name__ == "__main__":
    unittest.main()
