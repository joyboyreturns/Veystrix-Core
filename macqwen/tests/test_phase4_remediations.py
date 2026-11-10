import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from macqwen import cli, server, session
from models.flashnext import auto_reclaim, chat_guard, flight_recorder, overnight, request_admission, supervision
from models.flashnext.settings import modes


class TestPhase4Remediations(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_r1_01_auto_reclaim_isolates_storage_exceptions(self):
        """R1-01: post_request_cleanup catches storage errors without raising or corrupting lifecycle."""
        reclaimer = auto_reclaim.get_reclaimer()
        broken_prefix_store = mock.Mock()
        broken_prefix_store.shrink_ram.side_effect = OSError("Disk I/O error during shrink")

        # In CRITICAL pressure state
        with mock.patch.object(reclaimer, "evaluate_state", return_value=auto_reclaim.PressureState.CRITICAL):
            report = reclaimer.post_request_cleanup(store=None, prefix_store=broken_prefix_store)
            self.assertIsNotNone(report)
            self.assertTrue(any("failed" in r for r in report.reasons))

    def test_r1_02_crash_loop_stop_fires_across_exponential_backoff(self):
        """R1-02: Policy.decide halts on repeated crashes even when backoff cooldowns span > window_s."""
        policy = overnight.Policy(max_restarts=8, window_s=1800.0, base_cooldown_s=20.0, max_cooldown_s=600.0)
        now = 0.0
        decisions = []
        # Simulate 9 consecutive crashes spaced by their cooldowns
        for _ in range(9):
            action, cooldown, reason = policy.decide(1, now)
            decisions.append(action)
            now += max(cooldown, 20.0)

        # 9th crash must trigger stop (crash loop)
        self.assertEqual(decisions[-1], "stop")

    def test_r1_04_flight_recorder_handles_bare_filename(self):
        """R1-04: FlightRecorder handles a bare filename without raising FileNotFoundError."""
        bare_path = os.path.join(self.tmp_dir, "flight.jsonl")
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmp_dir)
            rec = flight_recorder.FlightRecorder("flight.jsonl")
            rec.record("test_event", foo="bar")
            self.assertTrue(Path(bare_path).exists())
        finally:
            os.chdir(orig_cwd)

    def test_r2_01_log_request_failure_is_non_fatal(self):
        """R2-01: _log_request does not raise when writing to log file fails."""
        with mock.patch.dict(os.environ, {"MACQWEN_REQUEST_LOG": "/nonexistent/invalid/dir/req.jsonl"}):
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                # Should catch error and not raise
                server._log_request(backend=None, stats=None, wall=1.0, first_piece=None, calls=0, started=0.0)

    def test_r2_04_drain_payload_on_unknown_post_route(self):
        """R2-04: Unknown POST route drains request body before returning 404."""
        handler = mock.Mock(spec=server.MacqwenHandler)
        handler.headers = {"Content-Length": "15"}
        handler.rfile = io.BytesIO(b"0123456789abcde")
        server.MacqwenHandler._drain_payload(handler)
        self.assertEqual(handler.rfile.read(), b"")

    def test_r3_01_server_without_mode_defaults_to_agentic(self):
        """R3-01: ./chat.sh --server defaults to AGENTIC mode so supervision.check passes."""
        with mock.patch("sys.stdin.isatty", return_value=False):
            with mock.patch("macqwen.cli.installed_checkpoints", return_value=[("flashnext", Path("/models/oq4"))]):
                chat_args, env = cli.command(["--server"])
                self.assertEqual(env.get("VEYSTRIX_MODE"), "agentic")

    def test_r3_02_chat_guard_allows_desktop_mac_without_battery(self):
        """R3-02: ChatGuard allows desktop Macs running on AC power where battery_percent is None."""
        guard = chat_guard.ChatGuard(os.path.join(self.tmp_dir, "guard.jsonl"))
        guard.tracer = mock.Mock()
        guard.tracer.compressor_start_mb = 100
        guard.sampler = mock.Mock()
        guard.sampler.sample.return_value = {
            "power_source": "ac",
            "battery_percent": None,  # Mac mini / Studio / Pro
            "low_power": False,
            "pressure_level": 1,
            "swap_used_mb": 100.0,
            "thermal_speed_limit": 100,
            "thermal_warning": False,
        }
        guard.sampler.runner.return_value = "thermal warning level: 0"
        with mock.patch.object(chat_guard.H, "guard_reason", return_value=None):
            with mock.patch.dict(os.environ, {"FLASHNEXT_CHAT_GUARD_WAIVE_STARTUP_AVAILABLE": "1"}):
                reason = guard._reason({
                    "swapout": 0, "swapin": 0, "available_mb": 10000,
                    "mlx_active": 100, "mlx_cache": 100, "compressor_mb": 100,
                    "metal_limit_bytes": 10**10,
                })
                self.assertIsNone(reason)

    def test_r3_03_cli_does_not_override_saved_profile_when_unspecified(self):
        """R3-03: Launcher marks default profile ephemeral so saved profile is preserved."""
        with mock.patch("sys.stdin.isatty", return_value=False):
            with mock.patch("macqwen.cli.installed_checkpoints", return_value=[("flashnext", Path("/models/oq4"))]):
                chat_args, env = cli.command([])
                self.assertIn("--profile", chat_args)
                self.assertEqual(env.get("MACQWEN_PROFILE_EPHEMERAL"), "1")

    def test_r3_04_chat_mode_refuses_layer_major(self):
        """R3-04: plan_mode refuses layer-major under CHAT mode."""
        with self.assertRaises(modes.ModeError):
            modes.plan_mode(modes.CHAT, enable=["layer-major"])
        # request_admission.enabled requires agentic mode
        self.assertFalse(request_admission.enabled({"VEYSTRIX_MODE": "chat", "FLASHNEXT_PREFILL_ORDER": "layer"}))
        self.assertTrue(request_admission.enabled({"VEYSTRIX_MODE": "agentic", "FLASHNEXT_PREFILL_ORDER": "layer"}))

    def test_r3_07_apply_tier_caps_cold_prefill(self):
        """R3-07: _apply_tier caps cold prefill to admitted tier's max_cold_prefill."""
        verdict = mock.Mock()
        verdict.tier = mock.Mock(context_limit=16384, max_cold_prefill=1024, name="minimal")
        with mock.patch.dict(os.environ, {
            "VEYSTRIX_CONTEXT_LIMIT": "262144",
            "VEYSTRIX_MAX_COLD_PREFILL": "32768",
            "VEYSTRIX_CONTEXT_REQUIRED": "1",
        }):
            session._apply_tier(verdict, io.StringIO())
            self.assertEqual(os.environ["VEYSTRIX_MAX_COLD_PREFILL"], "1024")
            self.assertEqual(os.environ["VEYSTRIX_CONTEXT_LIMIT"], "262144")

    def test_r3_10_server_requested_runs_supervision_check(self):
        """R3-10: In-chat /server runs supervision.check before serve()."""
        refusal = supervision.check({"VEYSTRIX_MODE": "chat"}, model="flashnext", server=True)
        self.assertIsNotNone(refusal)
        self.assertIn("refusing to start the server unsupervised", refusal)


if __name__ == "__main__":
    unittest.main()
