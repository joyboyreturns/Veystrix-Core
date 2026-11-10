from __future__ import annotations

from io import StringIO
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from veystrix.measurement import read_records
from veystrix.testsuite.catalog import build_catalog, runtime_directories
from veystrix.testsuite.api import TestContext
from veystrix.testsuite.runner import Runner, _stop_process_group
from veystrix.testsuite.spec import TestSpec
from veystrix.testsuite.terminal import choose_model


class TestSuiteDiscoveryTests(unittest.TestCase):
    def test_discovers_runtime_providers(self):
        runtimes = runtime_directories()
        self.assertIn("flashnext", runtimes)
        self.assertIn("bonsai2", runtimes)
        self.assertIn("k2_horizon", runtimes)
        self.assertIn("qwen27b", runtimes)
        for runtime in runtimes:
            self.assertTrue(build_catalog(runtime))

    def test_explicit_checkpoint_skips_installed_checkpoint_prompt(self):
        checkpoint = Path("/tmp/test-checkpoint")
        with patch("veystrix.testsuite.terminal.installed_checkpoints", return_value=[]):
            self.assertEqual(choose_model("qwen27b", str(checkpoint)),
                             ("qwen27b", checkpoint.resolve()))

    def test_runner_records_explicit_interrupted_outcome(self):
        class InterruptingOutput:
            def __iter__(self):
                yield "partial output\n"
                raise KeyboardInterrupt

            def read(self):
                return ""

            def close(self):
                pass

        process = SimpleNamespace(
            pid=123,
            stdout=InterruptingOutput(),
            wait=lambda timeout=None: 130,
        )
        spec = TestSpec(
            id="cancelled", title="cancelled", category="diagnostic",
            explanation="", why="", script=lambda _context, _path: ["fake"],
        )
        output = StringIO()
        context = TestContext("bonsai2", "/tmp/checkpoint", "/tmp/python")
        runner = Runner(context, output)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cancelled.jsonl"
            with (
                patch.object(runner, "_path", return_value=path),
                patch("veystrix.testsuite.runner.subprocess.Popen", return_value=process),
                patch("veystrix.testsuite.runner.os.killpg") as killpg,
            ):
                result = runner.run(spec)
            records = read_records(path)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["returncode"], 130)
        self.assertEqual([record["type"] for record in records],
                         ["run", "arm", "validation", "summary"])
        self.assertEqual(records[1]["status"], "interrupted")
        self.assertEqual(records[2]["status"], "interrupted")
        self.assertEqual(records[3]["status"], "interrupted")
        killpg.assert_called_once_with(123, signal.SIGINT)

    def test_stop_process_group_escalates_with_bounded_waits(self):
        class StubbornProcess:
            pid = 456

            def __init__(self):
                self.waits = 0

            def wait(self, timeout=None):
                self.waits += 1
                if self.waits < 3:
                    raise subprocess.TimeoutExpired("fake", timeout)
                return -signal.SIGKILL

            def poll(self):
                return -signal.SIGKILL

        process = StubbornProcess()
        with patch("veystrix.testsuite.runner.os.killpg") as killpg:
            self.assertEqual(_stop_process_group(process), -signal.SIGKILL)
        self.assertEqual(
            [call.args[1] for call in killpg.call_args_list],
            [signal.SIGINT, signal.SIGTERM, signal.SIGKILL,
             signal.SIGTERM, signal.SIGKILL],
        )


if __name__ == "__main__":
    unittest.main()
