"""The adaptive memory manager: policy tables, stress with synthetic budgets, and the request transitions
cold long -> short cached tool call -> another long -> short cached continuation -> cancellation -> recovery.
No model, no real memory: a simulated machine feeds the manager's readers."""
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from models.flashnext import expert_cache, expert_pool, memory_manager as M, pool_lifecycle as L

GIB, GB = M.GIB, M.GB


class SizedBuffer:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class FakePool:
    """Stands in for a decode pool of a given size; releases like the real one (close, drop views)."""

    def __init__(self, size):
        self._mm = SizedBuffer(size)
        self.array = object()
        self.buffer_mx = object()
        self._view = None
        self.closed = False

    def close(self):
        self.closed = True


class FakeStore:
    dir = "."

    def __init__(self):
        self.uncached = False

    def set_uncached_reads(self, enabled):
        self.uncached = bool(enabled)
        self._uncached = bool(enabled)
        return 0


class FakePrefix:
    def __init__(self, ram=0):
        self.ram, self.disk = ram, ram
        self.shrinks = []

    def usage(self):
        return {"ram_bytes": self.ram}

    def shrink_ram(self, target):
        freed = max(0, self.ram - target)
        self.ram -= freed
        self.shrinks.append(target)
        return freed


def snap(available_gib, pool_gb=0.0, **kw):
    return M.Snapshot(available=int(available_gib * GIB), pool_bytes=int(pool_gb * GB), **kw)


def req(tokens, ring=True, fresh=True, chunk=2048):
    return M.Request(tokens=tokens, prompt_tokens=tokens, chunk_tokens=chunk, fresh=fresh, ring_eligible=ring)


class PolicyTests(unittest.TestCase):
    def plan(self, s, r, state=None, now=1000.0, short=False):
        return M.plan(s, r, state or M.State(), M.Config(), now, short)

    def test_off_by_default_and_registered(self):
        os.environ.pop(M.FLAG, None)
        self.assertFalse(M.enabled())
        self.assertFalse(M.short_uncached_enabled())

    def test_a_roomy_machine_keeps_the_warm_pool_through_a_long_prefill(self):
        p = self.plan(snap(12, pool_gb=2.0), req(2000))
        self.assertFalse(p.release_pool)
        self.assertIsNone(p.pool_gb, "the pool already exists, nothing is rebuilt")
        self.assertIsNone(p.refuse)

    def test_a_tight_machine_releases_the_pool_and_rebuilds_it_at_2_gb_when_that_fits(self):
        p = self.plan(snap(6.0, pool_gb=2.0), req(2000))                  # keeping: 6 - 3.86 GiB < comfort plus margin
        self.assertTrue(p.release_pool)
        self.assertEqual(p.pool_gb, 2.0)
        self.assertIsNone(p.refuse)
        self.assertGreaterEqual(p.predicted_decode_headroom, M.Config().comfort_bytes)

    def test_less_room_means_a_smaller_pool_and_less_still_a_refusal_before_allocating(self):
        small = self.plan(snap(3.94, pool_gb=2.0), req(2000))              # 5.8 GiB once the pool is released
        self.assertTrue(small.release_pool)
        self.assertEqual(small.pool_gb, 1.0, "2 GB would break the comfort margin, 1 GB keeps it")
        none = self.plan(snap(3.14, pool_gb=2.0), req(2000))               # 5.0 GiB: the prefill itself crosses the guard
        self.assertIsNotNone(none.refuse)
        self.assertIn("guard line", none.refuse)

    def test_short_and_follow_up_prefills_never_touch_a_warm_pool(self):
        short = self.plan(snap(6.0, pool_gb=2.0), req(21, ring=False))
        self.assertFalse(short.release_pool)
        follow = self.plan(snap(6.0, pool_gb=2.0), req(2000, fresh=False))
        self.assertFalse(follow.release_pool, "a follow-up turn keeps its pool")

    def test_a_busy_engine_is_left_alone(self):
        for kw in ({"ring_leased": True}, {"others_in_flight": 1}):
            p = self.plan(snap(6.0, pool_gb=2.0, **kw), req(2000))
            self.assertFalse(p.release_pool, kw)

    def test_hysteresis_a_shrunk_pool_grows_back_only_after_cooldown_and_with_margin(self):
        state = M.State(pool_gb=1.0, last_change=1000.0)
        soon = self.plan(snap(8.2), req(2000), state, now=1010.0)          # fits 2 GB, but just changed
        self.assertEqual(soon.pool_gb, 1.0)
        self.assertTrue(any("hysteresis" in r for r in soon.reasons))
        later_tight = self.plan(snap(6.2), req(2000), state, now=1500.0)    # cooled down, margin not met
        self.assertEqual(later_tight.pool_gb, 1.0)
        later_roomy = self.plan(snap(8.2), req(2000), state, now=1500.0)
        self.assertEqual(later_roomy.pool_gb, 2.0)

    def test_a_safety_shrink_is_never_delayed_by_hysteresis(self):
        state = M.State(pool_gb=2.0, last_change=1000.0)
        p = self.plan(snap(3.94, pool_gb=2.0), req(2000), state, now=1001.0)
        self.assertTrue(p.release_pool)
        self.assertEqual(p.pool_gb, 1.0)

    def test_cold_ram_snapshots_move_to_ssd_only_when_memory_is_tight(self):
        tight = self.plan(snap(4.5, pool_gb=0.0, prefix_ram_bytes=172_000_000), req(21, ring=False))
        self.assertEqual(tight.evict_ram_to, 0)
        roomy = self.plan(snap(11.0, pool_gb=2.0, prefix_ram_bytes=172_000_000), req(21, ring=False))
        self.assertIsNone(roomy.evict_ram_to)

    def test_uncached_reads_are_for_short_prefills_and_need_their_own_flag(self):
        self.assertTrue(self.plan(snap(9), req(40, ring=False), short=True).uncached_reads)
        self.assertFalse(self.plan(snap(9), req(40, ring=False), short=False).uncached_reads)
        self.assertFalse(self.plan(snap(9), req(2000, ring=True), short=True).uncached_reads)

    def test_the_estimator_matches_the_measured_points(self):
        c = M.CALIBRATION
        ring_512 = M.estimate_prefill(req(512, chunk=512), pool_resident=False)
        ring_2k = M.estimate_prefill(req(2046), pool_resident=False)
        self.assertAlmostEqual(ring_512 / GB, 2.3, delta=0.1)
        self.assertAlmostEqual(ring_2k / GB, 3.0, delta=0.1)
        self.assertAlmostEqual((M.estimate_prefill(req(2046), pool_resident=True) - ring_2k) / GB, 1.15, delta=0.01)
        self.assertLess(M.estimate_prefill(req(21, ring=False), pool_resident=True), GB)

    def test_vm_stat_parsing_matches_the_benchmark_available_definition(self):
        text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 200.\n"
                "Pages speculative: 50.\nPages occupied by compressor: 25.\nPages active: 999.\n")
        parsed = M.parse_vm_stat(text)
        self.assertEqual(parsed["available"], 350 * 16384)
        self.assertEqual(parsed["compressor"], 25 * 16384)


class StressTests(unittest.TestCase):
    def test_random_budgets_never_plan_below_the_guard_and_never_thrash(self):
        rng = random.Random(7)
        cfg, state = M.Config(), M.State()
        now, pool_gb, rebuilds_seen = 0.0, 0.0, 0
        for step in range(2000):
            now += rng.uniform(1, 40)
            available = rng.choice([4.0, 5.5, 7.0, 8.5, 10.0, 12.0]) + rng.uniform(-0.3, 0.3)
            long_ = rng.random() < 0.5
            s = snap(available, pool_gb=pool_gb, ring_leased=rng.random() < 0.05, others_in_flight=int(rng.random() < 0.05))
            r = req(rng.choice([300, 1000, 2046, 3000]) if long_ else rng.choice([8, 21, 60]), ring=long_,
                    fresh=rng.random() < 0.7)
            p = M.plan(s, r, state, cfg, now)
            busy = s.ring_leased or s.others_in_flight
            self.assertFalse(busy and p.release_pool, "never detach a pool that is in use")
            if p.refuse is None:
                self.assertGreaterEqual(p.predicted_headroom, cfg.guard_bytes)
            if p.pool_gb is not None:
                self.assertIn(p.pool_gb, cfg.pools_gb)
                self.assertGreaterEqual(p.predicted_decode_headroom, cfg.guard_bytes)
            if p.refuse is None:                                   # apply it the way the manager does
                if p.release_pool:
                    pool_gb = 0.0
                    state.releases += 1
                    state.last_change = now
                if p.pool_gb is not None:
                    if pool_gb == 0.0:
                        pool_gb = p.pool_gb
                    if p.pool_gb != state.pool_gb and state.pool_gb is not None:
                        rebuilds_seen += 1
                        state.last_change = now
                    state.pool_gb = p.pool_gb
        self.assertLess(rebuilds_seen, 400, "size changes stay far below the request count")

    def test_a_steady_machine_does_not_rebuild_the_pool_at_all(self):
        cfg, state, now = M.Config(), M.State(), 0.0
        pool_gb = 0.0
        for _ in range(100):
            now += 30
            p = M.plan(snap(9.0, pool_gb=pool_gb), req(2000), state, cfg, now)
            if p.release_pool:
                pool_gb = 0.0
            if p.pool_gb is not None:
                pool_gb, state.pool_gb = p.pool_gb, p.pool_gb
        self.assertEqual(state.rebuilds, 0)


class Machine:
    """A 16 GiB machine: other applications, MLX weights, the decode pool and the request's transient."""

    def __init__(self, others_gib=6.6):
        self.others, self.weights, self.transient = others_gib * GIB, int(2.8 * GB), 0
        self.store = FakeStore()
        self.prefix = FakePrefix(ram=172_000_000)
        self.cleared = 0
        self.manager = M.MemoryManager(system=self.system, mlx=self.mlx, clock=self.clock)
        self.now = 0.0

    def clock(self):
        return self.now

    def pool(self):
        return getattr(self.store, "_flashnext_expert_pool", None)

    def used(self):
        pool = self.pool()
        return self.others + self.weights + self.transient + (len(pool._mm) if pool else 0)

    def system(self):
        return {"available": max(0, 16 * GIB - self.used()), "compressor": 0, "swap_used": 0, "pressure": 1}

    def mlx(self):
        pool = self.pool()
        return self.weights + self.transient + (len(pool._mm) if pool else 0), 0

    def prefill_and_decode(self, request, expect_refusal=False):
        """What the engine does around the manager: plan, prefill transient, decode pool creation, clean up."""
        report = self.manager.before_prefill(
            self.store, request, self.prefix, synchronize=lambda: None, clear_cache=lambda: setattr(self, "cleared", self.cleared + 1))
        self.transient = M.estimate_prefill(request, pool_resident=self.pool() is not None)
        peak_available = self.system()["available"]
        self.transient = 0
        if self.pool() is None:                                           # decode creates it lazily at the first token
            self.store._flashnext_expert_pool = FakePool(int(expert_pool.configured_gb() * GB))
        self.manager.after_request(self.store, clear_cache=lambda: setattr(self, "cleared", self.cleared + 1))
        self.now += 20
        return report, peak_available


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(expert_pool.set_budget_gb, None)
        self.addCleanup(lambda: L._IN_FLIGHT.__setitem__(0, 0))
        patch = mock.patch.object(expert_cache, "_PREFILL_RING", [None])
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_whole_sequence_stays_above_the_guard_with_one_pool_and_exact_bookkeeping(self):
        m = Machine(others_gib=6.6)
        guard = M.Config().guard_bytes
        pools_seen = []

        # 1. cold long prompt: no pool yet, a 2 GB pool is built for decode
        r1, low1 = m.prefill_and_decode(req(2068))
        self.assertGreaterEqual(low1, guard)
        self.assertEqual(len(m.pool()._mm), int(r1["plan"].pool_gb * GB))
        pools_seen.append(m.pool())

        # 2. short cached tool call: suffix only, the warm pool is kept untouched
        pool_before = m.pool()
        r2, low2 = m.prefill_and_decode(req(21, ring=False))
        self.assertIs(m.pool(), pool_before)
        self.assertEqual(r2["released_bytes"], 0)
        self.assertGreaterEqual(low2, guard)

        # 3. another long prompt: the pool is released first when keeping it would be tight, then rebuilt
        r3, low3 = m.prefill_and_decode(req(2067))
        self.assertGreaterEqual(low3, guard, "the second cold long prompt no longer crosses the guard line")
        if r3["plan"].release_pool:
            self.assertTrue(pool_before.closed)
            self.assertIsNot(m.pool(), pool_before)
        pools_seen.append(m.pool())

        # 4. short cached continuation: pool kept again
        keep = m.pool()
        r4, _ = m.prefill_and_decode(req(40, ring=False))
        self.assertIs(m.pool(), keep)

        # 5. cancellation in the middle of a prefill: cleanup still runs, nothing is left half-released
        m.manager.before_prefill(m.store, req(2000), m.prefix, synchronize=lambda: None, clear_cache=lambda: None)
        m.store.set_uncached_reads(True)
        m.manager.after_request(m.store, clear_cache=lambda: None)
        self.assertFalse(m.store.uncached, "descriptors go back to buffered after an interrupted request")

        # 6. recovery: the next long request plans normally and the engine ends with exactly one pool
        r6, low6 = m.prefill_and_decode(req(2068))
        self.assertGreaterEqual(low6, guard)
        live = [p for p in pools_seen if not p.closed]
        self.assertLessEqual(len(live), 1, "no leaked pool: every superseded pool was released")
        self.assertIsNotNone(m.pool())
        self.assertEqual(m.manager.state.refusals, 0)

    def test_a_machine_with_no_room_refuses_cleanly_and_recovers_when_memory_returns(self):
        m = Machine(others_gib=11.0)
        with self.assertRaises(M.MemoryBudgetRefused) as ctx:
            m.prefill_and_decode(req(2068))
        self.assertIn("guard line", str(ctx.exception))
        self.assertIsNone(m.pool(), "nothing was allocated by the refused request")
        self.assertEqual(m.manager.state.refusals, 1)
        m.others = int(6.0 * GIB)                                   # other applications quit
        report, low = m.prefill_and_decode(req(2068))
        self.assertGreaterEqual(low, M.Config().guard_bytes)
        self.assertEqual(m.manager.state.refusals, 1)

    def test_a_tight_machine_gets_a_smaller_pool_not_a_refusal(self):
        m = Machine(others_gib=7.56)
        report, _ = m.prefill_and_decode(req(2068))
        self.assertEqual(report["plan"].pool_gb, 1.0)
        self.assertEqual(len(m.pool()._mm), 1 * GB)

    def test_prefix_snapshots_move_to_ssd_under_pressure_and_nothing_is_lost(self):
        m = Machine(others_gib=8.5)
        report, _ = m.prefill_and_decode(req(21, ring=False))
        self.assertEqual(report["evicted_ram_bytes"], 172_000_000)
        self.assertEqual(m.prefix.ram, 0)
        self.assertEqual(m.prefix.disk, 172_000_000, "the SSD copy is untouched")

    def test_short_uncached_reads_apply_during_the_prefill_and_are_undone_after(self):
        m = Machine(others_gib=6.0)
        with mock.patch.dict(os.environ, {M.FLAG_SHORT_UNCACHED: "1"}):
            m.manager.before_prefill(m.store, req(40, ring=False), m.prefix, synchronize=lambda: None, clear_cache=lambda: None)
            self.assertTrue(m.store.uncached)
            m.manager.after_request(m.store, clear_cache=lambda: None)
        self.assertFalse(m.store.uncached)

    def test_cleanup_never_raises_and_trims_the_allocator_cache(self):
        manager = M.MemoryManager(system=lambda: {}, mlx=lambda: (0, 10 * 2 ** 20 * 100))
        cleared = []
        done = manager.after_request(None, clear_cache=lambda: cleared.append(1))
        self.assertTrue(done["cleared_cache"])
        broken = M.MemoryManager(system=lambda: {}, mlx=mock.Mock(side_effect=RuntimeError("no GPU")))
        self.assertIn("error", broken.after_request(None))

    def test_the_manager_never_detaches_a_pool_while_another_request_is_in_flight(self):
        m = Machine(others_gib=7.0)
        m.prefill_and_decode(req(2068))
        pool = m.pool()
        with mock.patch.dict(os.environ, {M.FLAG: "1"}), L.in_flight(), L.in_flight():
            m.manager.before_prefill(m.store, req(40, ring=False), m.prefix, synchronize=lambda: None, clear_cache=lambda: None)
        self.assertIs(m.pool(), pool)
        self.assertFalse(pool.closed)


class RealPieceTests(unittest.TestCase):
    def test_the_pool_budget_override_wins_over_the_environment_and_clears(self):
        self.addCleanup(expert_pool.set_budget_gb, None)
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "2"}):
            self.assertEqual(expert_pool.configured_gb(), 2.0)
            expert_pool.set_budget_gb(1.0)
            self.assertEqual(expert_pool.configured_gb(), 1.0)
            expert_pool.set_budget_gb(None)
            self.assertEqual(expert_pool.configured_gb(), 2.0)

    def test_the_store_toggles_uncached_reads_on_open_and_future_descriptors(self):
        import fcntl

        from models.flashnext.store import SafeTensorStore as Store

        store = object.__new__(Store)
        import threading

        store._fd_lock = threading.Lock()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "shard"
            path.write_bytes(b"x" * 4096)
            fd = os.open(path, os.O_RDONLY)
            store._fds = {"shard": fd}
            calls = []
            real = fcntl.fcntl
            with mock.patch.object(fcntl, "fcntl", side_effect=lambda f, c, a=0: calls.append((f, c, a)) or real(f, c, a)):
                self.assertEqual(store.set_uncached_reads(True), 1)
                self.assertEqual(store.set_uncached_reads(False), 1)
            os.close(fd)
        self.assertEqual([(c, a) for _, c, a in calls], [(fcntl.F_NOCACHE, 1), (fcntl.F_NOCACHE, 0)])
        self.assertFalse(store._uncached)


class BackendWiringTests(unittest.TestCase):
    def backend(self, reusable=2046):
        from veystrix.backends.flashnext import FlashNextBackend

        backend = object.__new__(FlashNextBackend)
        backend.language = object()
        backend._prefix_session = lambda: SimpleNamespace(reusable_tokens=lambda p: reusable, store=lambda: FakePrefix())
        return backend

    def patched(self, store, manager):
        from models.flashnext import prefill_ring

        env = {M.FLAG: "1"}
        return [mock.patch.dict(os.environ, env), mock.patch.object(L, "store_of", return_value=store),
                mock.patch.object(M, "get", return_value=manager),
                mock.patch.object(prefill_ring, "enabled", return_value=True),
                mock.patch.object(prefill_ring, "configured_experts", return_value=512),
                mock.patch.object(prefill_ring, "min_tokens", return_value=512),
                mock.patch.object(prefill_ring, "scratch_enabled", return_value=True),
                mock.patch.object(expert_pool, "enabled", return_value=True),
                mock.patch.object(expert_cache, "_PREFILL_RING", [None])]

    def run_hook(self, prompt_len, *, fresh=True, others=6.6, pool=False, reusable=2046):
        m = Machine(others_gib=others)
        if pool:
            m.store._flashnext_expert_pool = FakePool(2 * GB)
        backend = self.backend(reusable)
        patches = self.patched(m.store, m.manager)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(expert_pool.set_budget_gb, None)
        backend._make_room_for_prefill(list(range(prompt_len)), True, fresh)
        return backend, m

    def test_the_hook_plans_by_the_suffix_after_a_prefix_hit_and_reports_what_it_did(self):
        backend, m = self.run_hook(2067, pool=True)
        report = backend.last_memory_report
        self.assertEqual((report["tokens"], report["prompt_tokens"]), (21, 2067))
        self.assertFalse(report["ring_eligible"], "a 21-token suffix is below the ring minimum")
        self.assertFalse(report["release_pool"])
        self.assertIsNotNone(m.pool())

    def test_a_cold_long_prompt_is_ring_eligible_and_gets_a_pool_budget(self):
        backend, m = self.run_hook(2068, reusable=0)
        report = backend.last_memory_report
        self.assertTrue(report["ring_eligible"])
        self.assertEqual(report["pool_gb"], 2.0)
        self.assertEqual(expert_pool.configured_gb(), 2.0)
        self.assertTrue(any("next pool" in a for a in report["applied"]))

    def test_a_refusal_propagates_and_the_generate_path_turns_it_into_a_replayable_state(self):
        with self.assertRaises(M.MemoryBudgetRefused):
            self.run_hook(2068, others=11.0, reusable=0)
        import inspect

        from veystrix.backends.flashnext import FlashNextBackend

        source = inspect.getsource(FlashNextBackend._generate)
        self.assertIn("except memory_manager.MemoryBudgetRefused:\n                    self._mark_replay_needed()", source)

    def test_cleanup_runs_in_a_finally_so_cancellation_and_errors_are_covered(self):
        import inspect

        from veystrix.backends.flashnext import FlashNextBackend

        source = inspect.getsource(FlashNextBackend.generate)
        self.assertIn("self._after_request()", source)
        backend = self.backend()
        backend._generate = mock.Mock(side_effect=KeyboardInterrupt)
        calls = []
        backend._after_request = lambda: calls.append(1)
        with self.assertRaises(KeyboardInterrupt):
            backend.generate(5)
        self.assertEqual(calls, [1])

    def test_with_the_manager_off_nothing_changes_and_the_old_flag_still_works(self):
        os.environ.pop(M.FLAG, None)
        os.environ.pop(L.FLAG, None)
        backend = self.backend()
        backend._make_room_for_prefill(list(range(2068)), True, True)
        self.assertIsNone(backend.last_pool_release)
        self.assertIsNone(backend.last_memory_report)
        backend._after_request()
        self.assertIsNone(backend.last_memory_after)


class DeepSeekHarnessLifecycleSequenceTests(unittest.TestCase):
    """End-to-end multi-turn DeepSeek Harness lifecycle verification (Prompt, Tool Turns, Pool Shifts, Recovery)."""

    def setUp(self):
        expert_pool.set_budget_gb(None)
        M.reset_for_tests()

    def tearDown(self):
        expert_pool.set_budget_gb(None)
        M.reset_for_tests()

    def test_complete_harness_workload_lifecycle_sequence(self):
        # 16 GB machine: start with 6.5 GiB other apps
        m = Machine(others_gib=6.5)
        with mock.patch.dict(os.environ, {M.FLAG: "1", M.FLAG_BALANCED_UNCACHED: "1"}):
            # Turn 1: Cold system prompt (2068 tokens)
            r1, _ = m.prefill_and_decode(req(2068, ring=True, fresh=True))
            self.assertEqual(r1["plan"].pool_gb, 2.0)
            self.assertEqual(len(m.pool()._mm), 2 * GB)
            self.assertEqual(m.manager.state.releases, 0)
            self.assertEqual(m.manager.state.rebuilds, 0)

            # Turn 2: Cached continuation (21 tokens suffix)
            r2, _ = m.prefill_and_decode(req(21, ring=False, fresh=False))
            self.assertFalse(r2["plan"].release_pool)
            self.assertEqual(len(m.pool()._mm), 2 * GB, "Warm pool must be retained on tool turn")

            # Turn 3: Second tool turn (40 tokens suffix)
            r3, _ = m.prefill_and_decode(req(40, ring=False, fresh=False))
            self.assertFalse(r3["plan"].release_pool)
            self.assertEqual(len(m.pool()._mm), 2 * GB)

            # Turn 4: Memory tightens (external apps allocate 1.06 GiB) -> 2 GB to 1 GB pool downgrade
            # and prefix RAM snapshots evicted to SSD under pressure
            m.others += int(1.06 * GIB)
            r4, _ = m.prefill_and_decode(req(2068, ring=True, fresh=True))
            self.assertTrue(r4["plan"].release_pool)
            self.assertEqual(r4["plan"].pool_gb, 1.0)
            self.assertEqual(len(m.pool()._mm), 1 * GB)
            self.assertEqual(r4["evicted_ram_bytes"], 172_000_000, "RAM snapshot evicted to SSD under pressure")
            self.assertEqual(m.prefix.ram, 0)
            self.assertEqual(m.prefix.disk, 172_000_000)

            # Turn 5: Suffix request continues with 1 GB pool and SSD prefix
            r5, _ = m.prefill_and_decode(req(21, ring=False, fresh=False))
            self.assertFalse(r5["plan"].release_pool)
            self.assertEqual(len(m.pool()._mm), 1 * GB)

            # Turn 6: External memory frees up; cooldown advances; pool grows back 1 GB -> 2 GB
            m.others -= int(2.0 * GIB)
            m.now += 150.0  # past 120s cooldown
            r6, _ = m.prefill_and_decode(req(2068, ring=True, fresh=True))
            self.assertEqual(r6["plan"].pool_gb, 2.0)
            self.assertEqual(len(m.pool()._mm), 2 * GB)

            # Turn 7: Cancellation handling during request
            try:
                m.manager.before_prefill(m.store, req(30, ring=False), m.prefix)
                raise KeyboardInterrupt("Simulated user cancellation")
            except KeyboardInterrupt:
                clean_done = m.manager.after_request(m.store)
                self.assertNotIn("error", clean_done)

            # Turn 8: Next request runs cleanly after cancellation without state corruption
            r8, _ = m.prefill_and_decode(req(21, ring=False, fresh=False))
            self.assertFalse(r8["plan"].release_pool)
            self.assertEqual(len(m.pool()._mm), 2 * GB)


if __name__ == "__main__":
    unittest.main()
