"""Unit test wrapper for the Auto-Reclaim Lifecycle Stress Test."""
import unittest

from models.flashnext.tests.bench.bench_auto_reclaim_lifecycle import run_lifecycle_stress_test


class AutoReclaimLifecycleTests(unittest.TestCase):
    def tearDown(self):
        from models.flashnext import auto_reclaim, expert_pool, memory_manager
        auto_reclaim.reset_for_tests()
        memory_manager.reset_for_tests()
        expert_pool.set_budget_gb(None)

    def test_lifecycle_stress_test_20_iterations(self):
        # Run 2 full cycles (20 requests) covering all 10 transitions
        report = run_lifecycle_stress_test(num_requests=20, seed=123)
        self.assertEqual(report.verdict, "PASS")
        self.assertEqual(len(report.failures), 0)
        self.assertGreater(report.ram_hits + report.ssd_hits, 10)
        self.assertGreaterEqual(report.cancellations, 2)
        self.assertGreaterEqual(report.cancellation_recoveries, 2)
        self.assertGreater(report.pool_releases, 0)
        self.assertGreater(report.ram_evictions, 0)
        self.assertLess(report.memory_growth_bytes, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
