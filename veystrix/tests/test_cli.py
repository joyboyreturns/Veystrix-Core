from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import io
import json

from veystrix import cli


class LauncherTests(unittest.TestCase):
    def test_main_probe_does_not_misparse_mode_as_model(self):
        args = ["veystrix", "--mode", "agentic", "--model", "flashnext", "--checkpoint", "/models/test"]
        with patch.object(cli.sys, "argv", args), patch("veystrix.cli.branch_sync_warning", return_value=""), \
             patch("veystrix.cli._default_checkpoint_args", side_effect=lambda x: x), \
             patch("veystrix.cli.command", return_value=(["python", "-m", "test"], {})) as command, \
             patch("veystrix.cli.os.execvpe") as run:
            self.assertEqual(cli.main(), 0)
        command.assert_called_once_with(args[1:])
        run.assert_called_once()

    def test_no_auto_purge_overrides_inherited_enabled_flag(self):
        with patch.dict(os.environ, {"FLASHNEXT_AUTO_PURGE": "1"}), \
                patch("veystrix.cli._supports_python", return_value=True):
            _, environment = cli.command(["--model", "flashnext", "--no-auto-purge"])
        self.assertEqual(environment["FLASHNEXT_AUTO_PURGE"], "0")

    def test_default_launch_uses_the_only_installed_checkpoint(self):
        checkpoint = Path("/models/only")
        with patch("veystrix.cli.installed_checkpoints", return_value=[
            ("nemotron35-lightning", checkpoint),
        ]):
            self.assertEqual(
                cli._default_checkpoint_args(["--profile", "plain"]),
                [
                    "--profile", "plain", "--model", "nemotron35-lightning",
                    "--checkpoint", str(checkpoint),
                ],
            )

    def test_default_launch_asks_when_multiple_checkpoints_exist(self):
        choices = [
            ("flashnext", Path("/models/oq4")),
            ("nemotron35-lightning", Path("/models/nemotron")),
        ]
        with patch("veystrix.cli.installed_checkpoints", return_value=choices), \
                patch.object(cli.sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="2"):
            selected = cli._default_checkpoint_args([])
        self.assertEqual(
            selected,
            ["--model", "nemotron35-lightning", "--checkpoint", "/models/nemotron"],
        )

    def test_flashnext_child_environment_has_backend_chat_preset(self):
        from models.flashnext.settings.launch import CHAT_ENV

        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"VEYSTRIX_FLASHNEXT_PYTHON": str(python)}, clear=False), \
                    patch("veystrix.cli._supports_python", return_value=True):
                _command, child_env = cli.command(["--model", "flashnext"])
        from models.flashnext.settings.modes import OWNED

        for key, value in CHAT_ENV.items():
            if key not in OWNED:      # the launch mode owns these (the decode pool is 2 GB, not the old 6 GB)
                self.assertEqual(child_env[key], value)

    def test_nemotron_uses_the_shared_runtime_environment(self):
        shared = cli.MANAGED_PYTHON
        with patch.dict(os.environ, {
            "VEYSTRIX_PYTHON": "",
            "VEYSTRIX_NEMOTRON_PYTHON": "",
            "VIRTUAL_ENV": "/unrelated/venv",
        }, clear=False), \
                patch("veystrix.cli._supports_python", side_effect=lambda path, model: path == shared), \
                patch("veystrix.cli.setup_environment"):
            self.assertEqual(cli._interpreter("nemotron35-lightning"), shared)

    def test_first_launch_prepares_the_managed_environment_and_retries(self):
        with patch("veystrix.cli._supports_python", side_effect=(False, True)), \
                patch("veystrix.cli.setup_environment") as setup:
            self.assertEqual(cli._interpreter("flashnext"), cli.MANAGED_PYTHON)
        setup.assert_called_once_with(["--venv", str(cli.MANAGED_ENV)])

    def test_repeated_launch_reuses_the_managed_environment(self):
        with patch("veystrix.cli._supports_python", return_value=True), \
                patch("veystrix.cli.setup_environment") as setup:
            self.assertEqual(cli._interpreter("flashnext"), cli.MANAGED_PYTHON)
            self.assertEqual(cli._interpreter("nemotron35-lightning"), cli.MANAGED_PYTHON)
        setup.assert_not_called()

    def test_explicit_override_is_rejected_when_its_runtime_is_wrong(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"VEYSTRIX_PYTHON": str(python)}, clear=False
            ), patch("veystrix.cli._supports_python", return_value=False), \
                    patch("veystrix.cli._python_runtime_error", return_value="mlx 0.31"):
                with self.assertRaisesRegex(SystemExit, "VEYSTRIX_PYTHON.*mlx 0.31"):
                    cli._interpreter("flashnext")

    def test_explicit_flashnext_environment_override_wins(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ,
                {
                    "VEYSTRIX_FLASHNEXT_PYTHON": str(python),
                    "FLASHNEXT_SLAB_GLOBAL": "56",
                },
                clear=False,
            ), patch("veystrix.cli._supports_python", return_value=True):
                _command, child_env = cli.command(["--model", "flashnext"])
        self.assertEqual(child_env["FLASHNEXT_SLAB_GLOBAL"], "56")

    def test_explicit_metal_runtime_opt_out_wins(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ,
                {
                    "VEYSTRIX_FLASHNEXT_PYTHON": str(python),
                    "FLASHNEXT_METAL_RUNTIME": "0",
                },
                clear=False,
            ), patch("veystrix.cli._supports_python", return_value=True):
                _command, child_env = cli.command(["--model", "flashnext"])
        self.assertEqual(child_env["FLASHNEXT_METAL_RUNTIME"], "0")

    def test_packaged_runtime_skips_git_probe(self):
        with tempfile.TemporaryDirectory() as root, patch("veystrix.cli.subprocess.run") as run:
            self.assertEqual(cli.branch_sync_warning(Path(root)), "")
            run.assert_not_called()

    def test_warns_when_branch_does_not_include_known_main(self):
        stale = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        branch = subprocess.CompletedProcess(
            [], 0, stdout="codex/research\n", stderr=""
        )
        with tempfile.TemporaryDirectory() as root, patch("veystrix.cli.subprocess.run", side_effect=(stale, branch)):
            (Path(root) / ".git").mkdir()
            warning = cli.branch_sync_warning(Path(root))
        self.assertIn("codex/research", warning)
        self.assertIn("origin/main", warning)

    def test_skips_warning_when_branch_includes_known_main(self):
        current = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as root, patch("veystrix.cli.subprocess.run", return_value=current) as run:
            (Path(root) / ".git").mkdir()
            warning = cli.branch_sync_warning(Path(root))
        self.assertEqual(warning, "")
        run.assert_called_once()

    def test_no_argument_launch_always_selects_flashnext(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            preferences_file = Path(root, "preferences.json")
            preferences_file.write_text(json.dumps({"model": "qwen27b"}))
            with patch.dict(
                os.environ,
                {"VEYSTRIX_FLASHNEXT_PYTHON": str(python)},
                clear=False,
            ), patch("veystrix.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--preferences-file", str(preferences_file),
                ])
        model_index = command.index("--model") + 1
        self.assertEqual(command[model_index], "flashnext")

    def test_flashnext_uses_its_environment_and_forwards_profile(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(os.environ, {"VEYSTRIX_FLASHNEXT_PYTHON": str(python)}, clear=False), \
                    patch("veystrix.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--model", "flashnext", "--profile", "agent",
                    "--seed", "17",
                ])
        self.assertEqual(command[2], str(python))
        self.assertIn("flashnext", command)
        self.assertIn("agent", command)
        self.assertEqual(command[-2:], ["--seed", "17"])

    def test_flashnext_forwards_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"VEYSTRIX_FLASHNEXT_PYTHON": str(python)}, clear=False
            ), patch("veystrix.cli._supports_python", return_value=True):
                command, _ = cli.command(["--checkpoint", "oq4"])
        index = command.index("--model-path")
        self.assertEqual(command[index + 1], "oq4")

    def test_flashnext_forwards_benchmark_token_limits_together(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"VEYSTRIX_FLASHNEXT_PYTHON": str(python)}, clear=False
            ), patch("veystrix.cli._supports_python", return_value=True):
                command, _ = cli.command([
                    "--model", "flashnext", "--profile", "plain",
                    "--max-tokens", "32", "--think-budget", "4096",
                    "--benchmark-json", "--benchmark-prompt", "hello",
                ])
        self.assertEqual(command[command.index("--max-tokens") + 1], "32")
        self.assertEqual(command[command.index("--think-budget") + 1], "4096")
        self.assertIn("--benchmark-json", command)

    def test_setup_installs_the_shared_runtime_in_a_local_environment(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, ".venv")
            with patch("veystrix.cli.subprocess.check_call") as check_call:
                cli.setup_environment(["--venv", str(target)])
        commands = check_call.call_args_list
        self.assertEqual(commands[0].args[0][-2:], ["venv", str(target.resolve())])
        self.assertEqual(commands[-1].args[0][-1], str(cli.ROOT))

    def test_slash_server_alias_starts_server_mode(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"VEYSTRIX_FLASHNEXT_PYTHON": str(python)}, clear=False
            ), patch("veystrix.cli._supports_python", return_value=True):
                command, _ = cli.command(["/server"])
        self.assertIn("--server", command)
        self.assertEqual(command[command.index("--model") + 1], "flashnext")

    def test_removed_positional_model_argument_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "unexpected argument"):
            cli.command(["qwen27b"])

    def test_removed_models_are_not_selectable(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            for name in ("bonsai2", "k2-horizon", "qwen27b"):
                with self.subTest(model=name), self.assertRaises(SystemExit):
                    cli.command(["--model", name])
        self.assertEqual(sorted(cli.PYTHON_ENV), ["flashnext", "nemotron35-lightning", "shared"])

    def test_nemotron_server_launch_is_supervised_and_forwards_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            environment = {"VEYSTRIX_NEMOTRON_PYTHON": str(python)}
            with patch.dict(os.environ, environment, clear=False), \
                    patch("veystrix.cli._supports_python", return_value=True):
                command, child_env = cli.command(
                    ["--model", "nemotron35-lightning", "--checkpoint", "/models/n", "--server", "--pool-gb", "1"])
        self.assertEqual(command[command.index("--model") + 1], "nemotron35-lightning")
        self.assertEqual(command[command.index("--model-path") + 1], "/models/n")
        self.assertEqual(child_env["VEYSTRIX_MODE"], "agentic")
        self.assertEqual(child_env["FLASHNEXT_CHAT_GUARD"], "1")
        self.assertEqual(child_env["FLASHNEXT_PREFILL_CHUNK_TOKENS"], "2048")
        self.assertEqual(child_env["VEYSTRIX_NEMOTRON_POOL_GB"], "1")


if __name__ == "__main__":
    unittest.main()
