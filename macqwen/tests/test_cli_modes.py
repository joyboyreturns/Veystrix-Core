import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from macqwen import cli


class ModeLaunchTests(unittest.TestCase):
    def launch(self, *argv):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"MACQWEN_FLASHNEXT_PYTHON": str(python), "FLASHNEXT_EXPERT_POOL_GB": "6",
                                         "FLASHNEXT_PREFIX_CACHE": "1"}, clear=False), \
                    patch("macqwen.cli._supports_python", return_value=True):
                return cli.command(list(argv))

    def test_default_launch_is_safe_chat_without_the_old_six_gb_pool(self):
        command, env = self.launch("--model", "flashnext")
        self.assertEqual(command[command.index("--profile") + 1], "plain")
        self.assertNotIn("--server", command)
        self.assertEqual((env["VEYSTRIX_MODE"], env["FLASHNEXT_EXPERT_POOL_GB"], env["FLASHNEXT_EXPERT_POOL_AUTO"],
                          env["FLASHNEXT_CHAT_GUARD"], env["FLASHNEXT_PREFIX_CACHE"]), ("chat", "2", "1", "1", "0"))

    def test_agentic_starts_the_server_with_the_agent_profile_and_cache(self):
        command, env = self.launch("--mode", "agentic")
        self.assertEqual(command[command.index("--profile") + 1], "agent")
        self.assertEqual(command.count("--server"), 1)
        self.assertEqual((env["FLASHNEXT_PREFIX_CACHE"], env["FLASHNEXT_PREFILL_CHUNK_POLICY"],
                          env["VEYSTRIX_CONTEXT_LIMIT"]), ("1", "balanced", "8192"))

    def test_agentic_does_not_double_a_server_flag(self):
        command, _ = self.launch("--mode", "agentic", "/server")
        self.assertEqual(command.count("--server"), 1)

    def test_pool_override_and_cap(self):
        _, env = self.launch("--mode", "chat", "--pool-gb", "0")
        self.assertEqual((env["FLASHNEXT_EXPERT_POOL_GB"], env["FLASHNEXT_EXPERT_POOL_AUTO"]), ("0", "0"))
        with self.assertRaisesRegex(SystemExit, "cap"):
            self.launch("--pool-gb", "6")

    def test_experimental_feature_needs_an_explicit_flag(self):
        _, env = self.launch("--mode", "agentic", "--enable", "memory-manager")
        self.assertEqual(env["FLASHNEXT_MEMORY_MANAGER"], "1")
        self.assertEqual(self.launch("--mode", "agentic")[1]["FLASHNEXT_MEMORY_MANAGER"], "0")

    def test_a_benchmark_launch_is_not_reconfigured_by_a_mode(self):
        command, env = self.launch("--model", "flashnext", "--benchmark-json", "--benchmark-prompt", "hi")
        self.assertNotIn("VEYSTRIX_MODE", env)
        self.assertEqual(env["FLASHNEXT_EXPERT_POOL_GB"], "6")     # untouched, benchmark conditions rule

    def test_mode_is_for_flashnext_only(self):
        with self.assertRaisesRegex(SystemExit, "flashnext"):
            self.launch("--model", "qwen27b", "--mode", "chat")


if __name__ == "__main__":
    unittest.main()
