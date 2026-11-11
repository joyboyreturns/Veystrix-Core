import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix import cli


class ModeLaunchTests(unittest.TestCase):
    def launch(self, *argv):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"VEYSTRIX_FLASHNEXT_PYTHON": str(python), "FLASHNEXT_EXPERT_POOL_GB": "6",
                                         "FLASHNEXT_PREFIX_CACHE": "1"}, clear=False), \
                    patch("veystrix.cli._supports_python", return_value=True), \
                    patch("models.flashnext.settings.modes._free_disk_gb", return_value=500.0):   # not the runner's free disk
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
                          env["VEYSTRIX_CONTEXT_LIMIT"]), ("1", "balanced", "262144"))
        self.assertEqual((env["VEYSTRIX_MAX_COLD_PREFILL"], env["VEYSTRIX_MAX_INCREMENTAL_PREFILL"]), ("32768", "32768"))
        self.assertEqual((env["FLASHNEXT_CHAT_GUARD"], env["FLASHNEXT_WATCHDOG"], env["VEYSTRIX_MODE"]), ("1", "1", "agentic"))

    def test_supervisor_forwards_recorded_full_routing_threshold_to_worker(self):
        from models.flashnext import overnight

        args = overnight.build_parser().parse_args(["--threshold", "1.0"])
        supervised = overnight.server_command(args)
        command, _ = self.launch(*supervised[1:])
        self.assertEqual(command[command.index("--threshold") + 1], "1.0")

    def test_prefill_ceilings_are_launch_options_and_validated(self):
        _, env = self.launch("--mode", "agentic", "--max-cold-prefill", "16384", "--max-incremental-prefill", "8192")
        self.assertEqual((env["VEYSTRIX_MAX_COLD_PREFILL"], env["VEYSTRIX_MAX_INCREMENTAL_PREFILL"]), ("16384", "8192"))
        with self.assertRaisesRegex(SystemExit, "exceeds the context limit"):
            self.launch("--mode", "agentic", "--context-limit", "8192", "--max-cold-prefill", "16384")
        _, small = self.launch("--mode", "agentic", "--context-limit", "8192")      # defaults never exceed the window
        self.assertEqual(small["VEYSTRIX_MAX_COLD_PREFILL"], "8192")
        _, chat = self.launch("--mode", "chat")
        self.assertNotIn("VEYSTRIX_MAX_COLD_PREFILL", chat)

    def test_tiered4_uses_64k_cold_limit_and_preserves_overrides(self):
        _, env = self.launch("--mode", "agentic", "--enable", "kv-tiered4")
        self.assertEqual(env["VEYSTRIX_MAX_COLD_PREFILL"], "65536")
        self.assertEqual(env["VEYSTRIX_MAX_INCREMENTAL_PREFILL"], "32768")
        _, capped = self.launch("--mode", "agentic", "--enable", "kv-tiered4", "--max-cold-prefill", "4096")
        self.assertEqual(capped["VEYSTRIX_MAX_COLD_PREFILL"], "4096")

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
        self.assertEqual(self.launch("--mode", "agentic")[1]["FLASHNEXT_NGRAM_DEDUP"], "0")
        self.assertEqual(self.launch("--mode", "agentic", "--enable", "ngram-dedup")[1]["FLASHNEXT_NGRAM_DEDUP"], "1")

    def test_a_benchmark_launch_is_not_reconfigured_by_a_mode(self):
        command, env = self.launch("--model", "flashnext", "--benchmark-json", "--benchmark-prompt", "hi")
        self.assertNotIn("VEYSTRIX_MODE", env)
        self.assertEqual(env["FLASHNEXT_EXPERT_POOL_GB"], "6")     # untouched, benchmark conditions rule

    def test_mode_is_for_flashnext_only(self):
        with self.assertRaisesRegex(SystemExit, "flashnext"):
            self.launch("--model", "nemotron35-lightning", "--mode", "chat")


if __name__ == "__main__":
    unittest.main()
