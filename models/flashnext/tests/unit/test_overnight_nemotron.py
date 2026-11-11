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

    def test_kill_switch_prevents_start(self):
        with mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_AUTHORIZED": "0"}):
            with mock.patch.object(overnight, "Supervisor") as supervisor:
                self.assertEqual(overnight.main(["--model", "nemotron35-lightning", "--checkpoint", "/m/n"]), 2)
                supervisor.assert_not_called()

    def test_supervision_requires_guard_for_nemotron_server(self):
        refusal = supervision.check({}, model="nemotron35-lightning", server=True)
        self.assertIn("refusing to start the server unsupervised", refusal)
        ok = {"VEYSTRIX_MODE": "agentic", "FLASHNEXT_CHAT_GUARD": "1", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048"}
        self.assertIsNone(supervision.check(ok, model="nemotron35-lightning", server=True))


class NemotronAutoWarmTests(unittest.TestCase):
    def warmer(self, model, fake):
        from models.flashnext.tests.unit.test_overnight import FakeClock, FakeThread
        clock = FakeClock()
        clock.now = 1.7e9
        events = []
        ns = args(model=model, auto_warm=True, warm_idle_minutes=15.0, warm_refresh_hours=6.0)
        return overnight.AutoWarmer(ns, lambda e, **f: events.append(e), power_fn=lambda: {"source": "ac", "percent": 90},
                                    clock=clock, warm_module=fake, thread_factory=FakeThread), events

    def test_overnight_enables_auto_warm_for_nemotron_by_default(self):
        with mock.patch.object(overnight, "Supervisor") as supervisor:
            supervisor.return_value.run.return_value = 0
            overnight.main(["--model", "nemotron35-lightning", "--checkpoint", "/m/n", "--overnight"])
            self.assertTrue(supervisor.call_args[0][0].auto_warm)
            overnight.main(["--model", "nemotron35-lightning", "--checkpoint", "/m/n", "--overnight", "--no-auto-warm"])
            self.assertFalse(supervisor.call_args[0][0].auto_warm)

    def test_nemotron_warms_one_effort_and_keeps_its_state_apart_from_flashnext(self):
        from models.flashnext.tests.unit.test_overnight import FakeWarm, QUIET
        fake = FakeWarm([{"family": "hermes"}])
        warmer, events = self.warmer("nemotron35-lightning", fake)
        warmer.tick(QUIET)
        warmer.tick(QUIET)
        self.assertEqual([(c[0], c[1]) for c in fake.calls], [("hermes", "medium")])
        self.assertEqual(list(fake.state), ["nemotron35-lightning|hermes:medium"])
        self.assertEqual(fake.model_id, "veystrix-nemotron35-lightning")
        # Qwen's view of the same file does not see Nemotron's record, so Qwen still warms its own variants.
        qwen, _ = self.warmer("flashnext", fake)
        qwen.tick(QUIET)
        self.assertEqual([(c[0], c[1]) for c in fake.calls][1], ("hermes", "low"))
        self.assertIn("auto_warm_done", events)


if __name__ == "__main__":
    unittest.main()
