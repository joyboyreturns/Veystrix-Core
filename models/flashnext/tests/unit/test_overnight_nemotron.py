import os
import unittest
from types import SimpleNamespace
from unittest import mock

from models.flashnext import overnight, supervision


def args(**kw):
    base = dict(model="nemotron35-lightning", host="127.0.0.1", port=8080, pool_gb="2", checkpoint="/m/n",
                enable=[], prefix_dir=None, preferences_file=None, no_auto_purge=False, threshold=None)
    base.update(kw)
    return SimpleNamespace(**base)


class NemotronSupervisorTests(unittest.TestCase):
    def test_nemotron_launch_line_is_manual_loopback_and_has_no_flashnext_options(self):
        command = overnight.server_command(args())
        self.assertIn("nemotron35-lightning", command)
        self.assertEqual(command[command.index("--model") + 1], "nemotron35-lightning")
        self.assertEqual(command[command.index("--context-limit") + 1], "32768")
        self.assertIn("--server", command)
        for flashnext_only in ("--mode", "--enable", "--prefix-dir", "--threshold", "--max-cold-prefill"):
            self.assertNotIn(flashnext_only, command)

    def test_flashnext_launch_line_is_unchanged_by_default(self):
        command = overnight.server_command(SimpleNamespace(host="127.0.0.1", port=8080, pool_gb="0", checkpoint="/m/q"))
        self.assertEqual(command[command.index("--model") + 1], "flashnext")
        self.assertIn("--mode", command)
        self.assertIn("agentic", command)

    def test_refuses_to_start_without_authorization(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VEYSTRIX_NEMOTRON_AUTHORIZED", None)
            with mock.patch.object(overnight, "Supervisor") as supervisor:
                self.assertEqual(overnight.main(["--model", "nemotron35-lightning", "--checkpoint", "/m/n"]), 2)
                supervisor.assert_not_called()

    def test_supervision_requires_guard_for_nemotron_server(self):
        refusal = supervision.check({}, model="nemotron35-lightning", server=True)
        self.assertIn("refusing to start the server unsupervised", refusal)
        ok = {"VEYSTRIX_MODE": "agentic", "FLASHNEXT_CHAT_GUARD": "1", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048"}
        self.assertIsNone(supervision.check(ok, model="nemotron35-lightning", server=True))


if __name__ == "__main__":
    unittest.main()
