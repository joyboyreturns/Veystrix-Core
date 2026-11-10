import tempfile
import unittest
import subprocess
from unittest import mock
from pathlib import Path

from models.flashnext.chat_guard import ChatGuard


class Probe:
    def __init__(self):
        self.active = 100
        self.available = 10000
        self.swapout = 0

    def metal_limit(self): return 12_000_000_000
    def memory(self): return {"active": self.active, "cache": 100, "peak": 200}
    def machine(self):
        return {"available_mb": self.available, "compressor_mb": 100,
                "counters": {"swapin": 0, "swapout": self.swapout}}
    def rss_bytes(self): return 1
    def swap_used_mb(self): return 0
    def pressure_level(self): return 1


class Sampler:
    def __init__(self):
        self.pressure = 1
        self.swap_used = 0
    def sample(self):
        return {"t": 1, "battery_percent": 80, "power_source": "ac",
                "low_power": False, "pressure_level": self.pressure,
                "swap_used_mb": self.swap_used, "thermal_speed_limit": None, "thermal_warning": False}
    def runner(self, command):
        return "Note: No thermal warning level has been recorded\nNote: No CPU power status has been recorded"


class ChatGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.probe, self.sampler = Probe(), Sampler()
        self.exits = []
        self.now = [1000.0]
        self.guard = ChatGuard(Path(self.tmp.name) / "trace.jsonl", probe=self.probe,
                               sampler=self.sampler, exit_fn=self.exits.append, clock=lambda: self.now[0])

    def start(self):
        self.guard.start()
        self.addCleanup(self.guard.close)

    def test_guard_samples_and_records_watchdog(self):
        self.start()
        self.assertIn('watchdog_sample', self.guard.path.read_text())

    def test_explicit_two_gb_overrides_chat_default(self):
        from models.flashnext.settings.launch import apply_chat_environment
        environment = {"FLASHNEXT_EXPERT_POOL_GB": "2"}
        apply_chat_environment(environment)
        self.assertEqual(environment["FLASHNEXT_EXPERT_POOL_GB"], "2")

    def test_safe_launcher_overrides_inherited_and_env_file_research_flags(self):
        script = (Path(__file__).resolve().parents[4] / "chat.sh").read_text()
        script = script.replace('exec /usr/bin/python3 "$(dirname "$0")/veystrix/cli.py" "$@"',
                                'exec /usr/bin/env')
        with tempfile.TemporaryDirectory() as root:
            dot = Path(root) / ".frankenstein"
            dot.mkdir()
            (dot / "env").write_text("FLASHNEXT_EXPERT_POOL_GB=6\nFLASHNEXT_PREFIX_CACHE=1\n")
            result = subprocess.run(["/bin/bash", "-c", script], check=True, capture_output=True, text=True,
                                    env={"HOME": root, "FLASHNEXT_SAFE_CHAT": "1",
                                         "FLASHNEXT_PREFILL_RING_EXPERTS": "512"})
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        self.assertEqual(values["FLASHNEXT_EXPERT_POOL_GB"], "2")
        self.assertEqual(values["FLASHNEXT_PREFIX_CACHE"], "0")
        self.assertEqual(values["FLASHNEXT_PREFILL_RING_EXPERTS"], "0")
        self.assertEqual(values["FLASHNEXT_CHAT_GUARD"], "1")

    def test_guarded_two_gb_pool_failure_is_not_silently_downgraded(self):
        from types import SimpleNamespace
        from models.flashnext import expert_pool
        expert_pool.set_budget_gb(None)
        self.addCleanup(expert_pool.set_budget_gb, None)
        with mock.patch.dict("os.environ", {"FLASHNEXT_CHAT_GUARD": "1",
                                           "FLASHNEXT_EXPERT_POOL_GB": "2"}), \
                mock.patch.object(expert_pool, "ExpertPool", side_effect=RuntimeError("allocation failed")) as create:
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                expert_pool.for_store(SimpleNamespace())
        self.assertEqual(create.call_count, 1)

    def test_failed_power_command_blocks_before_load(self):
        self.sampler.runner = mock.Mock(side_effect=OSError("probe failed"))
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            self.guard.start()
        self.assertEqual(self.exits, [75])
        self.assertIn('monitor failed', self.guard.path.read_text())
        self.guard.close()

    def test_missing_mlx_probe_fails_closed(self):
        self.probe.active = None
        with self.assertRaisesRegex(RuntimeError, 'mlx_active'):
            self.guard.start()
        self.assertEqual(self.exits, [75])
        self.assertIn('critical probe unavailable', self.guard.path.read_text())
        self.guard.close()

    def test_startup_gate_is_nine_gib(self):
        self.probe.available = 9000
        with self.assertRaisesRegex(RuntimeError, '9 GiB'):
            self.guard.start()
        self.guard.close()

    def tick(self, seconds=1.0):
        """One guard sample ``seconds`` later (the power/pressure branch runs once a second)."""
        self.guard.tracer.stop()
        self.now[0] += seconds
        self.guard.last_power_check = 0
        self.guard.tracer.step()

    def test_one_swapped_out_page_is_elevated_not_a_stop(self):
        """Owner decision 2026-11-08: swap activity below the window limit raises the stage, it does not stop."""
        self.start()
        self.probe.swapout = 512                         # 8 MiB of 16 KiB pages: real swap activity, far below the stop
        self.tick()
        self.assertEqual(self.exits, [])
        self.assertEqual(self.guard.controller.state, "elevated")
        self.assertIn("swap activity", " ".join(self.guard.controller.reasons))

    def test_heavy_active_swapout_stops(self):
        self.start()
        self.tick()
        self.probe.swapout = 32768 + 1               # 512 MiB of 16 KiB pages inside the window
        self.tick()
        self.assertEqual(self.exits, [75])
        self.assertIn("active swap-out", self.guard.path.read_text())

    def test_a_single_pressure_level_two_sample_does_not_stop_and_recovers(self):
        self.start()
        self.sampler.pressure = 2
        self.tick()
        self.assertEqual(self.exits, [])
        self.assertEqual(self.guard.controller.state, "elevated")
        self.sampler.pressure = 1
        for _ in range(7):
            self.tick()
        self.assertEqual(self.exits, [])
        self.assertEqual(self.guard.controller.state, "normal")
        self.assertEqual(self.guard.controller.recoveries, 1)
        self.assertIn('"event":"memory_state"', self.guard.path.read_text())

    def test_persistent_pressure_level_two_stops(self):
        self.start()
        self.sampler.pressure = 2
        for _ in range(60):
            self.tick()
            if self.exits:
                break
        self.assertEqual(self.exits, [75])
        self.assertIn("without a calm spell", self.guard.path.read_text())

    def test_critical_pressure_level_four_stops_after_confirmation(self):
        self.start()
        self.sampler.pressure = 4
        self.tick()
        self.assertEqual(self.exits, [])
        self.tick(2.5)
        self.assertEqual(self.exits, [75])
        self.assertIn("memory pressure level 4", self.guard.path.read_text())

    def test_swap_growth_hard_stop_is_two_and_a_half_gib_over_the_session_baseline(self):
        self.sampler.swap_used = 700
        self.start()
        self.sampler.swap_used = 700 + 2400            # +2.34 GiB: elevated, not stopped
        self.tick()
        self.assertEqual(self.exits, [])
        self.sampler.swap_used = 700 + 2560            # +2.5 GiB exactly: hard stop
        self.tick()
        self.assertEqual(self.exits, [75])
        self.assertIn("swap grew 2.50 GiB", self.guard.path.read_text())

    def test_owner_waiver_ignores_pressure_and_availability_but_not_the_swap_limit(self):
        """FLASHNEXT_MEMORY_WAIVER=swap-only: 'launch anyway, only stop if it hits 2.5 GB swap'."""
        self.sampler.swap_used = 700
        with mock.patch.dict("os.environ", {"FLASHNEXT_MEMORY_WAIVER": "swap-only"}):
            self.probe.available = 4000                  # below the 9 GiB start gate: waived
            self.start()
            self.sampler.pressure = 4                    # critical pressure held: waived (measured, not enforced)
            self.probe.available = 1000                  # below the 2 GB floor: waived
            for _ in range(70):
                self.tick()
            self.assertEqual(self.exits, [])
            self.assertEqual(self.guard.controller.ledger()["peak"]["pressure_level"], 4)
            self.probe.swapout = 40000                   # heavy swap-out rate: waived too
            self.tick()
            self.assertEqual(self.exits, [])
            self.sampler.swap_used = 700 + 2560          # +2.5 GiB over the baseline: the one remaining memory stop
            self.tick()
            self.assertEqual(self.exits, [75])
            self.assertIn("swap grew 2.50 GiB", self.guard.path.read_text())

    def test_owner_waiver_keeps_the_non_memory_protections(self):
        with mock.patch.dict("os.environ", {"FLASHNEXT_MEMORY_WAIVER": "swap-only"}):
            self.start()
            self.sampler.sample = lambda: {"t": 1, "battery_percent": 20, "power_source": "battery", "low_power": False,
                                           "pressure_level": 1, "swap_used_mb": 0, "thermal_speed_limit": None,
                                           "thermal_warning": False}
            self.tick()
            self.assertEqual(self.exits, [75])
            self.assertIn("battery", self.guard.path.read_text())

    def test_owner_waiver_accepts_only_its_one_value(self):
        from models.flashnext import memory_waiver
        self.assertFalse(memory_waiver.active({}))
        self.assertTrue(memory_waiver.active({"FLASHNEXT_MEMORY_WAIVER": "swap-only"}))
        with self.assertRaises(ValueError):
            memory_waiver.active({"FLASHNEXT_MEMORY_WAIVER": "all"})

    def test_elevated_stage_never_overrides_the_available_memory_floor(self):
        self.start()
        self.sampler.pressure = 2
        self.tick()
        self.assertEqual(self.exits, [])
        self.probe.available = 1500
        self.tick()
        self.assertEqual(self.exits, [75])
        self.assertIn("available memory", self.guard.path.read_text())

    def test_pressure_probe_failure_stops(self):
        self.start()
        self.guard.tracer.stop()
        self.sampler.pressure = None
        self.guard.last_power_check = 0
        self.guard.tracer.step()
        self.assertEqual(self.exits, [75])
        self.assertIn('pressure_level', self.guard.path.read_text())

    def test_guard_runs_during_two_real_small_model_turns(self):
        import mlx.core as mx
        from veystrix.backends.base import GenerationCancelled
        from veystrix.tests.test_flashnext_backend import FlashNextBackendTests
        from models.flashnext.tests.unit.test_prefix_cache import build_model
        from models.flashnext import expert_cache

        self.start()
        environment = mock.patch.dict("os.environ", {"FLASHNEXT_CHAT_GUARD": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        backend = FlashNextBackendTests().backend()
        backend.language = build_model()
        backend.cache = backend.language.make_cache()
        backend.tokenizer.decode = lambda ids: "".join(chr(65 + int(i) % 26) for i in ids)
        backend.stops = {1}
        out = []
        active = []
        for tokens in ([12, 13, 14], [15, 16], [17, 18]):
            backend.pending = list(tokens)
            backend.generate(3, out=out.append)
            mx.synchronize()
            active.append(mx.get_active_memory())
        mx.synchronize()
        self.assertTrue(out)
        self.assertGreater(len(backend.tape), 5)
        self.assertFalse(expert_cache._PREFILL_RING[0])
        self.assertFalse(self.exits)
        self.assertLess(max(active) - min(active), 10 * 1024 * 1024)
        backend.pending = [19]
        with self.assertRaises(GenerationCancelled):
            backend.generate(3, should_cancel=lambda: True)
        self.assertFalse(expert_cache._PREFILL_RING[0])
        backend.reset()
        self.assertEqual(backend.tape, [])
        self.assertTrue(self.guard.path.read_text())

    def test_chat_starts_guard_before_loading_backend(self):
        import sys
        from veystrix import session
        events = []

        def start(_guard):
            events.append("guard")
        def load(*args):
            events.append("load")
            raise RuntimeError("stop before checkpoint load")

        with tempfile.TemporaryDirectory() as root, \
                mock.patch.dict("os.environ", {"FLASHNEXT_CHAT_GUARD": "1"}), \
                mock.patch.object(sys, "argv", ["session.py", "--model", "flashnext",
                    "--model-path", root, "--profile", "plain",
                    "--preferences-file", str(Path(root) / "prefs.json")]), \
                mock.patch("veystrix.checkpoints.resolve_flashnext", return_value=Path(root)), \
                mock.patch.object(ChatGuard, "start", start), \
                mock.patch.object(session, "build_backend", side_effect=load):
            with self.assertRaisesRegex(RuntimeError, "stop before checkpoint load"):
                session.main()
        self.assertEqual(events, ["guard", "load"])

    def test_guarded_request_fences_before_cleanup_and_poison_on_failure(self):
        from veystrix.tests.test_flashnext_backend import FlashNextBackendTests
        from models.flashnext import gpu_sync, pool_lifecycle

        backend = FlashNextBackendTests().backend()
        backend._generate = mock.Mock(return_value=("answer", None))
        backend._after_request = mock.Mock()
        self.addCleanup(lambda: pool_lifecycle._LIFECYCLE_FAULT.__setitem__(0, None))
        with mock.patch.dict("os.environ", {"FLASHNEXT_CHAT_GUARD": "1"}), \
                mock.patch.object(gpu_sync, "barrier", side_effect=RuntimeError("GPU fence failed")):
            with self.assertRaisesRegex(RuntimeError, "GPU fence failed"):
                backend.generate(1)
            backend._after_request.assert_called_once()
            self.assertIn("GPU fence failed", pool_lifecycle.lifecycle_fault())
            with self.assertRaisesRegex(RuntimeError, "restart the process"):
                backend.generate(1)



class MetalCacheReleaseTests(unittest.TestCase):
    LIMIT = 12_710_000_000

    def row(self, active, cache):
        return {"available_mb": 6000.0, "mlx_active": active, "mlx_cache": cache, "compressor_mb": 100,
                "swapout": 0, "swapin": 0, "metal_limit_bytes": self.LIMIT}

    def guard(self):
        return ChatGuard(Path(tempfile.mkdtemp()) / "g.jsonl", probe=Probe(), sampler=Sampler())

    def run_recheck(self, row, active_after, cache_after):
        import mlx.core as mx

        with mock.patch.object(mx, "clear_cache") as cleared, \
             mock.patch.object(mx, "get_active_memory", return_value=active_after), \
             mock.patch.object(mx, "get_cache_memory", return_value=cache_after):
            reason = self.guard()._recheck_after_cache_release(row, True)
        cleared.assert_called_once()
        return reason

    def test_reclaimable_cache_does_not_stop(self):
        row = self.row(10_000_000_000, 2_220_000_000)          # 12.22 GB held: the real stop
        self.assertIsNone(self.run_recheck(row, 10_000_000_000, 0))

    def test_active_memory_near_the_limit_still_stops(self):
        row = self.row(12_300_000_000, 100_000_000)
        self.assertIn("Metal limit", self.run_recheck(row, 12_300_000_000, 0))

    def test_unreadable_memory_keeps_the_original_stop(self):
        import mlx.core as mx

        row = self.row(12_300_000_000, 100_000_000)
        with mock.patch.object(mx, "clear_cache", side_effect=RuntimeError("no")):
            self.assertIn("Metal limit", self.guard()._recheck_after_cache_release(row, True))
