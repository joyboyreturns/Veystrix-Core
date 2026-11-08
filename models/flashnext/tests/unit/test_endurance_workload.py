import tempfile
import unittest
from pathlib import Path

from models.flashnext.tests.bench.bench_endurance_workload import run_endurance


class EnduranceWorkloadTests(unittest.TestCase):
    def test_synthetic_endurance_workload_50_requests(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = run_endurance(num_requests=50, directory=Path(tmp_dir), seed=2026)
            self.assertEqual(report.verdict, "PASS")
            self.assertEqual(report.total_requests, 50)
            self.assertGreater(report.ram_hits + report.ssd_hits, 35)
            self.assertGreater(report.cancellations, 0)
            self.assertGreater(report.restarts, 0)
            self.assertEqual(report.failures, [])
            # Assert memory stability (delta under 10 MB)
            self.assertLess(report.max_memory_growth_bytes, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
