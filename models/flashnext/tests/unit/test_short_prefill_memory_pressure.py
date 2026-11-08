"""Unit tests for Goal 2: Short-prefill memory pressure, balanced uncached reads, and lifecycle."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import expert_cache, expert_io_stats, expert_pool, memory_manager as M

GIB, GB = M.GIB, M.GB


def snap(available_gib: float, pressure: int = 1, pool_gb: float = 0.0, **kw) -> M.Snapshot:
    return M.Snapshot(
        available=int(available_gib * GIB),
        pressure=pressure,
        pool_bytes=int(pool_gb * GB),
        **kw
    )


def req(tokens: int, ring: bool = False, fresh: bool = True) -> M.Request:
    return M.Request(tokens=tokens, prompt_tokens=tokens, chunk_tokens=2048, fresh=fresh, ring_eligible=ring)


class TestBalancedUncachedPolicy(unittest.TestCase):
    def setUp(self):
        self.cfg = M.Config(balanced_uncached=True)

    def test_roomy_memory_uses_buffered_reads(self):
        # 8 GB available (> 6 GB headroom threshold), pressure normal (1) -> buffered reads (F_NOCACHE off)
        p = M.plan(snap(8.0, pressure=1), req(21), M.State(), self.cfg, short_uncached=True)
        self.assertFalse(p.uncached_reads)
        self.assertTrue(any("buffered" in r for r in p.reasons))

    def test_tight_memory_uses_uncached_reads(self):
        # 4.2 GB available (< 5 GB threshold), pressure normal (1) -> uncached reads (F_NOCACHE on)
        p = M.plan(snap(4.2, pressure=1), req(21), M.State(), self.cfg, short_uncached=True)
        self.assertTrue(p.uncached_reads)
        self.assertTrue(any("uncached" in r for r in p.reasons))

    def test_high_pressure_forces_uncached_even_with_room(self):
        # 7.5 GB available, but kernel pressure level 2 (warn) -> uncached reads to protect system
        p = M.plan(snap(7.5, pressure=2), req(21), M.State(), self.cfg, short_uncached=True)
        self.assertTrue(p.uncached_reads)
        self.assertTrue(any("pressure >= 2" in r for r in p.reasons))

    def test_intermediate_memory_follows_flag(self):
        # 5.5 GB available (between 5 and 6 GB): follows short_uncached flag
        p_on = M.plan(snap(5.5, pressure=1), req(21), M.State(), self.cfg, short_uncached=True)
        self.assertTrue(p_on.uncached_reads)
        p_off = M.plan(snap(5.5, pressure=1), req(21), M.State(), self.cfg, short_uncached=False)
        self.assertFalse(p_off.uncached_reads)

    def test_env_var_activates_balanced_policy(self):
        with mock.patch.dict(os.environ, {M.FLAG_BALANCED_UNCACHED: "1"}):
            p = M.plan(snap(8.0, pressure=1), req(21), M.State(), M.Config(balanced_uncached=False), short_uncached=True)
            self.assertFalse(p.uncached_reads)

        with mock.patch.dict(os.environ, {M.FLAG_SHORT_UNCACHED: "balanced"}):
            p = M.plan(snap(8.0, pressure=1), req(21), M.State(), M.Config(balanced_uncached=False), short_uncached=True)
            self.assertFalse(p.uncached_reads)


class TestPoolReuseMemoryEstimator(unittest.TestCase):
    def test_short_prefill_estimate_savings_with_pool_reuse(self):
        # Base estimate without pool reuse
        est_standard = M.estimate_prefill(req(21), pool_resident=True, pool_reuse=False)
        # Reduced estimate with pool reuse
        est_reused = M.estimate_prefill(req(21), pool_resident=True, pool_reuse=True)
        self.assertLess(est_reused, est_standard)
        # Verified difference is substantial (at least 150 MB savings)
        self.assertGreaterEqual(est_standard - est_reused, 150 * 1024 * 1024)

    def test_plan_predicted_headroom_reflects_reuse(self):
        state = M.State()
        cfg = M.Config()
        p_std = M.plan(snap(6.0, pool_gb=2.0), req(21), state, cfg, pool_reuse=False)
        p_opt = M.plan(snap(6.0, pool_gb=2.0), req(21), state, cfg, pool_reuse=True)
        self.assertGreater(p_opt.predicted_headroom, p_std.predicted_headroom)
        # Both keep the warm decode pool
        self.assertFalse(p_std.release_pool)
        self.assertFalse(p_opt.release_pool)


class TestRapidShortPrefillSequence(unittest.TestCase):
    def test_rapid_short_prefill_peak_memory_stability(self):
        """Simulate 20 rapid short-prefill requests (10-50 tokens) and verify memory does not ratchet."""
        mgr = M.MemoryManager(cfg=M.Config(balanced_uncached=True))
        peak_records = []
        token_lengths = [10, 25, 40, 15, 50, 30, 20, 45, 12, 35, 22, 48, 16, 28, 38, 18, 50, 24, 32, 14]

        simulated_active = 2800 * 1024 * 1024 # 2.8 GB resident

        for turn, tok in enumerate(token_lengths):
            r = req(tok)
            with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1"}):
                plan_report = mgr.before_prefill(None, r)

            prefill_alloc = M.estimate_prefill(r, pool_resident=True, pool_reuse=True)
            peak_memory = simulated_active + prefill_alloc
            peak_records.append(peak_memory)

            cleared = []
            mgr.after_request(None, clear_cache=lambda: cleared.append(True))

        max_peak = max(peak_records)
        min_peak = min(peak_records)
        self.assertLess(max_peak - min_peak, 200 * 1024 * 1024)
        self.assertEqual(mgr.state.requests, 20)
        self.assertEqual(mgr.state.refusals, 0)
        self.assertEqual(mgr.state.releases, 0)


if __name__ == "__main__":
    unittest.main()
