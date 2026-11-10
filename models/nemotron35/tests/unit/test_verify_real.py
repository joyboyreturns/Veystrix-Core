"""The real-checkpoint verifier, exercised end to end on a tiny synthetic checkpoint."""
import shutil
import tempfile
import unittest
from pathlib import Path

from models.nemotron35 import verify_real
from models.nemotron35.tiny_checkpoint import permissive_config_check, write_checkpoint


class VerifyRealToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="nemotron-verify-")
        write_checkpoint(Path(cls.dir) / "ck")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_verify(self, **kw):
        return verify_real.verify(Path(self.dir) / "ck", config_check=permissive_config_check, tokens=21, decode=2,
                                  chunk=8, swap_limit_mb=1e9, **kw)

    def test_streamed_chunk_major_matches_the_layer_at_a_time_reference(self):
        report = self.run_verify()
        self.assertTrue(report["all_greedy_agree"])
        self.assertLess(report["max_abs_diff"], 0.05)
        self.assertEqual([s["phase"] for s in report["steps"]], ["prefill", "decode1", "decode2"])

    def test_streamed_layer_major_matches_the_reference_too(self):
        report = self.run_verify(layer_major=16, pool_gb=0.0)
        self.assertTrue(report["all_greedy_agree"])
        self.assertLess(report["max_abs_diff"], 0.05)


if __name__ == "__main__":
    unittest.main()
