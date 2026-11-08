"""Unit tests for Automatic Request-to-Request Memory Reclamation (auto_reclaim.py)."""
import os
import subprocess
import unittest
from unittest import mock

from models.flashnext import auto_reclaim
from models.flashnext.auto_reclaim import (
    AutoReclaimer,
    PressureState,
    ReclaimConfig,
    ReclaimReport,
    auto_purge_enabled,
    auto_reclaim_enabled,
    get_reclaimer,
    reset_for_tests,
)

GB = 10 ** 9
MB = 10 ** 6


class DummyPrefixStore:
    def __init__(self, ram_bytes: int = 200 * MB):
        self.ram_bytes = ram_bytes
        self.evictions = []

    def shrink_ram(self, target_bytes: int) -> int:
        evicted = max(0, self.ram_bytes - target_bytes)
        self.evictions.append((target_bytes, evicted))
        self.ram_bytes = min(self.ram_bytes, target_bytes)
        return evicted


class DummyStore:
    def __init__(self, has_pool: bool = True, slots: int = 32, stride: int = 64 * MB):
        self._uncached = False
        if has_pool:
            self._flashnext_expert_pool = DummyPool(slots, stride)
        else:
            self._flashnext_expert_pool = None

    def set_uncached_reads(self, uncached: bool) -> None:
        self._uncached = uncached


class DummyPool:
    def __init__(self, slots: int = 32, stride: int = 64 * MB):
        self.slots = slots
        self.stride = stride
        self.freed = False


class AutoReclaimStateHysteresisTests(unittest.TestCase):
    def setUp(self):
        reset_for_tests()
        self.cfg = ReclaimConfig(
            constrained_headroom_bytes=5500 * MB,
            critical_headroom_bytes=3500 * MB,
            recovery_margin_bytes=500 * MB,
            cooldown_s=60.0,
        )
        self.clock = 1000.0
        self.system_state = {"available": 8000 * MB, "pressure": 1, "file_backed_mb": 4000.0}
        self.reclaimer = AutoReclaimer(cfg=self.cfg, system=lambda: self.system_state, clock=lambda: self.clock)

    def tearDown(self):
        reset_for_tests()

    def test_initial_state_is_normal(self):
        self.assertEqual(self.reclaimer.current_state, PressureState.NORMAL)

    def test_immediate_transition_to_constrained_when_headroom_drops(self):
        state = self.reclaimer.evaluate_state(5000 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.CONSTRAINED)
        self.assertEqual(self.reclaimer.last_state_change, self.clock)

    def test_immediate_transition_to_constrained_on_pressure_level_2(self):
        state = self.reclaimer.evaluate_state(7000 * MB, 2, self.clock)
        self.assertEqual(state, PressureState.CONSTRAINED)

    def test_immediate_transition_to_critical_on_low_headroom_or_pressure_4(self):
        state = self.reclaimer.evaluate_state(3000 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.CRITICAL)

        self.reclaimer.current_state = PressureState.NORMAL
        state = self.reclaimer.evaluate_state(7000 * MB, 4, self.clock)
        self.assertEqual(state, PressureState.CRITICAL)

    def test_downward_transition_from_critical_to_constrained_requires_cooldown(self):
        self.reclaimer.current_state = PressureState.CRITICAL
        self.reclaimer.last_state_change = self.clock

        # Within cooldown (< 60s), remains CRITICAL even if memory is 4500 MB (CONSTRAINED range)
        self.clock += 30.0
        state = self.reclaimer.evaluate_state(4500 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.CRITICAL)

        # After cooldown (>= 60s), transitions to CONSTRAINED
        self.clock += 35.0  # total 65s
        state = self.reclaimer.evaluate_state(4500 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.CONSTRAINED)
        self.assertEqual(self.reclaimer.last_state_change, self.clock)

    def test_downward_transition_from_constrained_to_normal_requires_cooldown_and_recovery_margin(self):
        self.reclaimer.current_state = PressureState.CONSTRAINED
        self.reclaimer.last_state_change = self.clock

        # 5800 MB is above constrained threshold (5500 MB), but below recovery threshold (5500 + 500 = 6000 MB)
        self.clock += 100.0
        state = self.reclaimer.evaluate_state(5800 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.CONSTRAINED)

        # 6200 MB is above recovery threshold (6000 MB), but within cooldown (< 60s)
        self.reclaimer.last_state_change = self.clock
        self.clock += 20.0
        state = self.reclaimer.evaluate_state(6200 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.CONSTRAINED)

        # 6200 MB after cooldown passes recovery margin -> transitions to NORMAL
        self.clock += 45.0
        state = self.reclaimer.evaluate_state(6200 * MB, 1, self.clock)
        self.assertEqual(state, PressureState.NORMAL)


class AutoReclaimPostRequestTests(unittest.TestCase):
    def setUp(self):
        reset_for_tests()
        self.clock = 2000.0
        self.system_state = {"available": 8000 * MB, "pressure": 1, "file_backed_mb": 4500.0}
        self.cfg = ReclaimConfig(
            constrained_headroom_bytes=5500 * MB,
            critical_headroom_bytes=3500 * MB,
            cache_clear_normal_bytes=256 * MB,
        )
        self.reclaimer = AutoReclaimer(cfg=self.cfg, system=lambda: self.system_state, clock=lambda: self.clock)

    def tearDown(self):
        reset_for_tests()

    def test_failed_gpu_synchronization_preserves_all_owned_resources(self):
        self.system_state["available"] = 2800 * MB
        store = DummyStore()
        prefix = DummyPrefixStore()
        pool = store._flashnext_expert_pool
        clear = mock.Mock()
        with mock.patch("models.flashnext.prefill_ring.fence") as fence, \
             mock.patch("models.flashnext.pool_lifecycle.release_decode_pool") as release:
            with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                self.reclaimer.post_request_cleanup(store, prefix, clear_cache=clear,
                    synchronize=mock.Mock(side_effect=RuntimeError("GPU failure")))
        self.assertIs(store._flashnext_expert_pool, pool)
        self.assertEqual(prefix.evictions, [])
        fence.assert_not_called()
        release.assert_not_called()
        clear.assert_not_called()

    def test_failed_fence_preserves_all_owned_resources(self):
        self.system_state["available"] = 2800 * MB
        store = DummyStore()
        prefix = DummyPrefixStore()
        clear = mock.Mock()
        with mock.patch("models.flashnext.prefill_ring.fence", side_effect=RuntimeError("fence failed")), \
             mock.patch("models.flashnext.pool_lifecycle.release_decode_pool") as release:
            with self.assertRaisesRegex(RuntimeError, "fence failed"):
                self.reclaimer.post_request_cleanup(store, prefix, clear_cache=clear, synchronize=lambda: None)
        self.assertIsNotNone(store._flashnext_expert_pool)
        self.assertEqual(prefix.evictions, [])
        release.assert_not_called()
        clear.assert_not_called()

    def test_auto_reclaim_enables_manager_dependency(self):
        from models.flashnext import memory_manager
        with mock.patch.dict(os.environ, {"FLASHNEXT_AUTO_MEMORY_RECLAIM": "1", memory_manager.FLAG: "0"}):
            self.assertTrue(memory_manager.enabled())
        with mock.patch.dict(os.environ, {"FLASHNEXT_AUTO_MEMORY_RECLAIM": "0", memory_manager.FLAG: "0"}):
            self.assertFalse(memory_manager.enabled())

    def test_normal_state_retains_pool_and_ram_snapshots(self):
        store = DummyStore(has_pool=True)
        prefix_store = DummyPrefixStore(ram_bytes=250 * MB)
        clear_cache_called = []

        report = self.reclaimer.post_request_cleanup(
            store=store,
            prefix_store=prefix_store,
            clear_cache=lambda: clear_cache_called.append(True),
            now=self.clock,
        )

        self.assertEqual(report.state_after, PressureState.NORMAL.value)
        self.assertTrue(report.pool_retained)
        self.assertEqual(report.ram_prefix_evicted_bytes, 0)
        self.assertEqual(len(prefix_store.evictions), 0)
        # Allocator cache was not > 256 MB so clear_cache was not called
        self.assertFalse(clear_cache_called)

    def test_normal_state_clears_mlx_cache_if_above_threshold(self):
        store = DummyStore(has_pool=True)
        clear_cache_called = []

        with mock.patch("mlx.core.get_cache_memory", return_value=300 * MB):
            report = self.reclaimer.post_request_cleanup(
                store=store,
                clear_cache=lambda: clear_cache_called.append(True),
                now=self.clock,
            )

        self.assertEqual(report.state_after, PressureState.NORMAL.value)
        self.assertTrue(clear_cache_called)
        self.assertTrue(any("trimmed MLX cache" in r for r in report.reasons))

    def test_constrained_state_evicts_ram_prefix_and_sets_next_pool_budget(self):
        self.system_state["available"] = 4800 * MB  # < 5500 MB -> CONSTRAINED
        store = DummyStore(has_pool=True)
        prefix_store = DummyPrefixStore(ram_bytes=350 * MB)
        clear_cache_called = []

        with mock.patch("models.flashnext.expert_pool.set_budget_gb") as mock_set_budget:
            report = self.reclaimer.post_request_cleanup(
                store=store,
                prefix_store=prefix_store,
                clear_cache=lambda: clear_cache_called.append(True),
                now=self.clock,
            )
            mock_set_budget.assert_called_once_with(1.0)

        self.assertEqual(report.state_after, PressureState.CONSTRAINED.value)
        self.assertEqual(report.ram_prefix_evicted_bytes, 350 * MB)
        self.assertTrue(report.pool_retained)
        self.assertTrue(clear_cache_called)
        self.assertTrue(any("moved 350.0 MB cold RAM prefix snapshots to SSD" in r for r in report.reasons))

    def test_critical_state_releases_unleased_decode_pool_and_evicts_ram(self):
        self.system_state["available"] = 2800 * MB  # < 3500 MB -> CRITICAL
        store = DummyStore(has_pool=True)
        prefix_store = DummyPrefixStore(ram_bytes=400 * MB)
        clear_cache_called = []

        with mock.patch("models.flashnext.pool_lifecycle.release_decode_pool", return_value=2 * GB) as mock_release, \
             mock.patch("models.flashnext.pool_lifecycle.others_in_flight", return_value=0):
            report = self.reclaimer.post_request_cleanup(
                store=store,
                prefix_store=prefix_store,
                clear_cache=lambda: clear_cache_called.append(True),
                now=self.clock,
            )
            mock_release.assert_called_once()

        self.assertEqual(report.state_after, PressureState.CRITICAL.value)
        self.assertFalse(report.pool_retained)
        self.assertEqual(report.pool_released_bytes, 2 * GB)
        self.assertEqual(report.ram_prefix_evicted_bytes, 400 * MB)
        self.assertTrue(clear_cache_called)

    def test_uncached_descriptors_restored_on_cleanup(self):
        store = DummyStore(has_pool=True)
        store._uncached = True

        report = self.reclaimer.post_request_cleanup(store=store, now=self.clock)
        self.assertFalse(store._uncached)
        self.assertIn("restored buffered file descriptors", report.reasons)

    def test_synchronize_is_invoked_before_reclaiming(self):
        sync_called = []
        self.reclaimer.post_request_cleanup(synchronize=lambda: sync_called.append(True), now=self.clock)
        self.assertTrue(sync_called)


class ExceptionalStartupPurgeTests(unittest.TestCase):
    def setUp(self):
        reset_for_tests()
        self.clock = 5000.0
        self.system_state = {"available": 4000 * MB, "file_backed_mb": 4800.0}
        self.cfg = ReclaimConfig(
            purge_cooldown_s=1800.0,
            max_daily_purges=2,
            min_file_backed_for_purge_mb=3000.0,
        )
        self.reclaimer = AutoReclaimer(cfg=self.cfg, system=lambda: self.system_state, clock=lambda: self.clock)

    def tearDown(self):
        reset_for_tests()

    def test_purge_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(auto_reclaim.FLAG_AUTO_PURGE, None)
            ok, recovered, reason = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertFalse(ok)
            self.assertEqual(recovered, 0)
            self.assertIn("disabled", reason)

    def test_purge_blocked_if_file_backed_cache_is_insufficient(self):
        self.system_state["file_backed_mb"] = 1500.0  # < 3000 MB
        with mock.patch.dict(os.environ, {auto_reclaim.FLAG_AUTO_PURGE: "1"}):
            ok, recovered, reason = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertFalse(ok)
            self.assertIn("too small", reason)

    def test_purge_non_interactive_sudo_failure_returns_gracefully_without_hanging(self):
        with mock.patch.dict(os.environ, {auto_reclaim.FLAG_AUTO_PURGE: "1"}), \
             mock.patch("subprocess.run", return_value=subprocess.CompletedProcess(["sudo", "-n", "/usr/sbin/purge"], returncode=1, stderr="sudo: a password is required")):
            ok, recovered, reason = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertFalse(ok)
            self.assertEqual(recovered, 0)
            self.assertIn("password required or disabled", reason)

    def test_purge_rate_limiting_max_2_in_24h(self):
        with mock.patch.dict(os.environ, {auto_reclaim.FLAG_AUTO_PURGE: "1"}), \
             mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], returncode=0)), \
             mock.patch("time.sleep"):
            # Purge 1
            ok1, _, _ = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertTrue(ok1)

            # Purge 2 after 35 minutes (cooldown passed)
            self.clock += 2100.0
            ok2, _, _ = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertTrue(ok2)

            # Purge 3 after another 35 minutes -> blocked by daily limit (max 2/24h)
            self.clock += 2100.0
            ok3, _, reason3 = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertFalse(ok3)
            self.assertIn("maximum daily purges (2/24h) reached", reason3)

    def test_purge_cooldown_enforced(self):
        with mock.patch.dict(os.environ, {auto_reclaim.FLAG_AUTO_PURGE: "1"}), \
             mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], returncode=0)), \
             mock.patch("time.sleep"):
            ok1, _, _ = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertTrue(ok1)

            # Attempt 10 minutes later (< 30 min cooldown)
            self.clock += 600.0
            ok2, _, reason2 = self.reclaimer.exceptional_startup_purge(self.clock)
            self.assertFalse(ok2)
            self.assertIn("purge cooldown active", reason2)
            self.assertIn("1200s remaining", reason2)


if __name__ == "__main__":
    unittest.main()
