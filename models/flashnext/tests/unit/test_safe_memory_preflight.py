import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "flashnext" / "safe_memory_preflight.sh"

# The script's checkpoint gate only needs a directory holding config.json. A fixture keeps these tests independent of
# whether a real checkpoint is installed on the machine that runs them (CI has none).
_FIXTURE = tempfile.TemporaryDirectory()
(Path(_FIXTURE.name) / "config.json").write_text("{}")
CKPT = _FIXTURE.name


class SafeMemoryPreflightTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), f"Preflight script not found at {SCRIPT}")

    def test_help_option_returns_zero(self):
        proc = subprocess.run([str(SCRIPT), "--checkpoint", CKPT, "--help"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("safe_memory_preflight.sh", proc.stdout)

    def test_missing_checkpoint_fails_with_code_2(self):
        proc = subprocess.run(
            [str(SCRIPT), "--checkpoint", CKPT, "--checkpoint", "/nonexistent/checkpoint/path", "--json"],
            capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 2)
        data = json.loads(proc.stdout)
        self.assertEqual(data["status"], "BLOCKED")
        self.assertEqual(data["code"], 2)
        self.assertIn("Checkpoint directory does not exist", data["reason"])

    def test_threshold_met_skips_purge_and_returns_ready(self):
        # Using a very low threshold (0.1 GB) should always pass on any functioning machine
        proc = subprocess.run(
            [str(SCRIPT), "--checkpoint", CKPT, "--threshold-gb", "0.1", "--json"],
            capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertEqual(data["status"], "READY")
        self.assertTrue(data["gate_passed"])
        self.assertFalse(data["purge"]["attempted"])
        self.assertIn("already meets startup threshold", data["purge"]["skipped_reason"])

    def test_insufficient_memory_in_dry_run_reports_simulated_purge(self):
        # Using an impossibly high threshold (999.0 GB) in dry-run mode
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            proc = subprocess.run(
                [str(SCRIPT), "--checkpoint", CKPT, "--threshold-gb", "999.0", "--force-purge", "--dry-run", "--json", "--state-file", tmp.name],
                capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 1)
            data = json.loads(proc.stdout)
            self.assertEqual(data["status"], "BLOCKED")
            self.assertFalse(data["gate_passed"])
            self.assertTrue(data["purge"]["attempted"])
            self.assertFalse(data["purge"]["success"])
            self.assertIn("dry-run mode", data["purge"]["skipped_reason"])

    def test_purge_state_enforces_max_purges(self):
        # Create a state file that already has 2 purges
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            state = {"count": 2, "last_purge_timestamp": int(time.time()) - 3600, "history": []}
            json.dump(state, tmp)
            tmp_path = Path(tmp.name)

        try:
            proc = subprocess.run(
                [str(SCRIPT), "--checkpoint", CKPT, "--threshold-gb", "999.0", "--force-purge", "--json", "--state-file", str(tmp_path)],
                capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 1)
            data = json.loads(proc.stdout)
            self.assertIn("maximum purge limit reached", data["reason"])
            self.assertFalse(data["purge"]["attempted"])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_purge_state_enforces_settling_interval(self):
        # Create a state file with a recent purge (5 minutes ago)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            state = {"count": 1, "last_purge_timestamp": int(time.time()) - 300, "history": []}
            json.dump(state, tmp)
            tmp_path = Path(tmp.name)

        try:
            proc = subprocess.run(
                [str(SCRIPT), "--checkpoint", CKPT, "--threshold-gb", "999.0", "--force-purge", "--json", "--state-file", str(tmp_path)],
                capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 1)
            data = json.loads(proc.stdout)
            self.assertIn("purge settling constraint", data["reason"])
            self.assertFalse(data["purge"]["attempted"])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_json_structure_contains_all_diagnostic_fields(self):
        proc = subprocess.run([str(SCRIPT), "--checkpoint", CKPT, "--threshold-gb", "0.1", "--json"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        for key in ("status", "gate_passed", "reason", "exit_code", "checkpoint",
                    "threshold_gb", "initial_available_gb", "final_available_gb", "purge",
                    "initial_memory", "final_memory"):
            self.assertIn(key, data)
        init_mem = data["initial_memory"]
        for mem_key in ("available_mb", "available_gb", "free_mb", "inactive_mb", "speculative_mb",
                        "purgeable_mb", "compressor_mb", "wired_mb", "active_mb", "file_backed_mb",
                        "anonymous_mb", "swap_used_mb", "pressure_level"):
            self.assertIn(mem_key, init_mem)

    def test_low_memory_never_purges_without_explicit_request(self):
        proc = subprocess.run([str(SCRIPT), "--checkpoint", CKPT, "--threshold-gb", "999", "--json"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        data = json.loads(proc.stdout)
        self.assertFalse(data["purge"]["attempted"])
        self.assertIn("purge not requested", data["reason"])


if __name__ == "__main__":
    unittest.main()
