"""The decode-pool release proposal (models/flashnext/pool_lifecycle.py): policy, ordering, safety, recreation.
Real ExpertPool objects on a fake store; no model."""
import gc
import os
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

from models.flashnext import expert_pool, pool_lifecycle as L
from models.flashnext.slab_pack import get_slab_layout


class FakeStore:
    dir = "."
    refs = {}


def small_pool(store):
    stride = get_slab_layout(32).record_stride
    pool = expert_pool.ExpertPool(store, expert_pool.HEADER + 16 * stride, 32)
    store._flashnext_expert_pool = pool
    return pool


class PolicyTests(unittest.TestCase):
    ARGS = dict(has_pool=True, ring_experts=512, ring_min_tokens=512, scratch=True)

    def test_managed_requests_are_exclusive_and_nested_requests_fail(self):
        entered = threading.Event()
        finished = threading.Event()
        with mock.patch.dict(os.environ, {"FLASHNEXT_AUTO_MEMORY_RECLAIM": "1"}):
            def other():
                with L.exclusive_request():
                    entered.set()
                finished.set()

            with L.exclusive_request():
                with self.assertRaisesRegex(RuntimeError, "nested generation"):
                    with L.exclusive_request():
                        pass
                thread = threading.Thread(target=other)
                thread.start()
                self.assertFalse(entered.wait(0.05))
            self.assertTrue(finished.wait(2))
            thread.join()

    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(L.FLAG, None)
            self.assertFalse(L.enabled())
            self.assertFalse(L.should_release(2000, **self.ARGS))

    def test_release_only_when_every_condition_holds(self):
        with mock.patch.dict(os.environ, {L.FLAG: "1"}):
            self.assertTrue(L.should_release(2000, **self.ARGS))
            self.assertTrue(L.should_release(512, **self.ARGS))
            self.assertFalse(L.should_release(511, **self.ARGS), "below the ring minimum the pool stays warm")
            self.assertFalse(L.should_release(2000, **{**self.ARGS, "has_pool": False}))
            self.assertFalse(L.should_release(2000, **{**self.ARGS, "ring_experts": 0}))
            self.assertFalse(L.should_release(2000, **{**self.ARGS, "scratch": False}),
                             "without the scratch option the ring is meant to live in the decode pool")


class ReleaseTests(unittest.TestCase):
    def test_nothing_to_release(self):
        calls = []
        self.assertEqual(L.release_decode_pool(FakeStore(), synchronize=lambda: calls.append("sync"),
                                               ring_active=lambda: False), 0)
        self.assertEqual(calls, [], "no pool, so not even a GPU fence")

    def test_release_fences_first_detaches_frees_and_clears_the_cache(self):
        store = FakeStore()
        pool = small_pool(store)
        size = len(pool._mm)
        ref = weakref.ref(pool)
        order = []
        released = L.release_decode_pool(
            store, synchronize=lambda: order.append(("sync", store._flashnext_expert_pool is not None)),
            clear_cache=lambda: order.append(("clear", store._flashnext_expert_pool is None)),
            ring_active=lambda: False)
        del pool
        gc.collect()
        self.assertEqual(released, size)
        self.assertEqual(order, [("sync", True), ("clear", True)], "fence while attached, clear after detaching")
        self.assertIsNone(store._flashnext_expert_pool)
        self.assertIsNone(ref(), "nothing keeps the pool object alive")

    def test_a_leased_ring_blocks_the_release(self):
        store = FakeStore()
        pool = small_pool(store)
        synced = []
        self.assertEqual(L.release_decode_pool(store, synchronize=lambda: synced.append(1), ring_active=lambda: True), 0)
        self.assertIs(store._flashnext_expert_pool, pool)
        self.assertEqual(synced, [])

    def test_a_failed_fence_keeps_the_pool_attached_and_usable(self):
        self.addCleanup(lambda: L._LIFECYCLE_FAULT.__setitem__(0, None))
        store = FakeStore()
        pool = small_pool(store)

        def broken():
            raise RuntimeError("GPU not idle")

        with self.assertRaises(RuntimeError):
            L.release_decode_pool(store, synchronize=broken, ring_active=lambda: False)
        self.assertIs(store._flashnext_expert_pool, pool)
        self.assertIsNotNone(pool.array)                  # untouched
        slots, misses = pool.assign(0, [1, 2])
        self.assertEqual(len(misses), 2)

    def test_release_twice_is_harmless(self):
        store = FakeStore()
        small_pool(store)
        self.assertGreater(L.release_decode_pool(store, ring_active=lambda: False), 0)
        self.assertEqual(L.release_decode_pool(store, ring_active=lambda: False), 0)

    def test_the_default_ring_check_reads_the_prefill_ring_slot(self):
        from models.flashnext import expert_cache

        store = FakeStore()
        pool = small_pool(store)
        with mock.patch.object(expert_cache, "_PREFILL_RING", [object()]):
            self.assertEqual(L.release_decode_pool(store), 0)
        self.assertIs(store._flashnext_expert_pool, pool)
        with mock.patch.object(expert_cache, "_PREFILL_RING", [None]):
            self.assertGreater(L.release_decode_pool(store), 0)

    def test_an_array_still_viewing_the_pool_keeps_its_memory_valid(self):
        import numpy as np

        store = FakeStore()
        pool = small_pool(store)
        pool.array[:8] = 7
        survivor = pool.array[:8]                          # a view someone still holds
        L.release_decode_pool(store, ring_active=lambda: False)
        gc.collect()
        self.assertEqual(survivor.tolist(), [7] * 8, "released by reference, never unmapped under a reader")
        self.assertIsInstance(survivor, np.ndarray)

    def test_decode_recreates_the_pool_on_demand_after_a_release(self):
        store = FakeStore()
        stride = get_slab_layout(32).record_stride
        small_pool(store)
        old = store._flashnext_expert_pool
        L.release_decode_pool(store, ring_active=lambda: False)
        gb = (expert_pool.HEADER + 16 * stride) / 1e9
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: f"{gb:.9f}"}):
            fresh = expert_pool.for_store(store, 32)
        self.assertIsNotNone(fresh)
        self.assertIsNot(fresh, old)
        self.assertEqual((fresh.hits, fresh.misses), (0, 0), "a new pool starts cold")


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        L._IN_FLIGHT[0] = 0
        self.addCleanup(lambda: L._IN_FLIGHT.__setitem__(0, 0))

    def test_in_flight_counts_only_while_the_flag_is_on(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(L.FLAG, None)
            with L.in_flight():
                self.assertEqual(L._IN_FLIGHT[0], 0)
        with mock.patch.dict(os.environ, {L.FLAG: "1"}):
            with L.in_flight():
                self.assertEqual((L._IN_FLIGHT[0], L.others_in_flight()), (1, 0))
                with L.in_flight():
                    self.assertEqual((L._IN_FLIGHT[0], L.others_in_flight()), (2, 1))
            self.assertEqual(L._IN_FLIGHT[0], 0)

    def test_the_count_is_restored_when_a_request_raises(self):
        with mock.patch.dict(os.environ, {L.FLAG: "1"}):
            with self.assertRaises(KeyboardInterrupt):
                with L.in_flight():
                    raise KeyboardInterrupt
            self.assertEqual(L._IN_FLIGHT[0], 0)

    def test_another_request_in_flight_blocks_the_release(self):
        store = FakeStore()
        pool = small_pool(store)
        synced = []
        with mock.patch.dict(os.environ, {L.FLAG: "1"}):
            with L.in_flight(), L.in_flight():              # this request and one more
                self.assertEqual(L.release_decode_pool(store, synchronize=lambda: synced.append(1),
                                                       ring_active=lambda: False), 0)
        self.assertIs(store._flashnext_expert_pool, pool)
        self.assertEqual(synced, [])
        with mock.patch.dict(os.environ, {L.FLAG: "1"}), L.in_flight():          # alone: released
            self.assertGreater(L.release_decode_pool(store, ring_active=lambda: False), 0)


class MakeRoomTests(unittest.TestCase):
    """make_room with a fake store and the ring settings patched in: every branch reports why."""

    def run_case(self, *, flag="1", pool=True, tokens=2000, ring=512, min_tokens=512, scratch=True,
                 ring_leased=False, others=0):
        from models.flashnext import expert_cache, expert_pool as EP, prefill_ring

        store = FakeStore()
        if pool:
            small_pool(store)
        env = {L.FLAG: flag}
        patches = [
            mock.patch.dict(os.environ, env), mock.patch.object(L, "store_of", return_value=store),
            mock.patch.object(L, "others_in_flight", return_value=others),
            mock.patch.object(prefill_ring, "enabled", return_value=True),
            mock.patch.object(prefill_ring, "configured_experts", return_value=ring),
            mock.patch.object(prefill_ring, "min_tokens", return_value=min_tokens),
            mock.patch.object(prefill_ring, "scratch_enabled", return_value=scratch),
            mock.patch.object(EP, "enabled", return_value=True),
            mock.patch.object(expert_cache, "_PREFILL_RING", [object() if ring_leased else None]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return L.make_room(SimpleNamespace(), tokens), store

    def test_off_returns_none_and_touches_nothing(self):
        report, store = self.run_case(flag="0")
        self.assertIsNone(report)
        self.assertIsNotNone(store._flashnext_expert_pool)

    def test_eligible_prefill_releases_and_reports_the_memory(self):
        report, store = self.run_case()
        self.assertEqual(report["reason"], "released")
        self.assertGreater(report["released_bytes"], 0)
        self.assertIsNone(store._flashnext_expert_pool)
        for key in ("active_before", "active_after", "cache_after", "seconds", "tokens"):
            self.assertIn(key, report)
        self.assertEqual(report["tokens"], 2000)

    def test_each_refusal_keeps_the_pool_and_says_why(self):
        for kwargs, reason in (
                ({"pool": False}, "no decode pool"),
                ({"tokens": 100}, "not eligible"),
                ({"ring": 0}, "not eligible"),
                ({"scratch": False}, "not eligible"),
                ({"others": 1}, "another request is in flight"),
                ({"ring_leased": True}, "a prefill ring is leased")):
            with self.subTest(kwargs):
                report, store = self.run_case(**kwargs)
                self.assertEqual(report["released_bytes"], 0)
                self.assertIn(reason, report["reason"])
                self.assertEqual(getattr(store, "_flashnext_expert_pool", None) is not None, kwargs.get("pool", True))

    def test_a_failed_gpu_fence_stops_the_request(self):
        self.addCleanup(lambda: L._LIFECYCLE_FAULT.__setitem__(0, None))
        from models.flashnext import gpu_sync

        with mock.patch.object(gpu_sync, "barrier", side_effect=RuntimeError("GPU busy")):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                self.run_case()
        self.assertIn("GPU busy", L.lifecycle_fault())

    def test_repeated_requests_release_then_find_nothing(self):
        first, store = self.run_case()
        self.assertEqual(first["reason"], "released")
        with mock.patch.dict(os.environ, {L.FLAG: "1"}), mock.patch.object(L, "store_of", return_value=store):
            again = L.make_room(SimpleNamespace(), 2000)
        self.assertEqual((again["reason"], again["released_bytes"]), ("no decode pool", 0))


class BackendWiringTests(unittest.TestCase):
    def test_generate_keeps_its_signature_and_runs_inside_in_flight(self):
        import inspect

        from macqwen.backends.flashnext import FlashNextBackend

        self.assertEqual(list(inspect.signature(FlashNextBackend.generate).parameters),
                         list(inspect.signature(FlashNextBackend._generate).parameters))
        backend = object.__new__(FlashNextBackend)
        seen = []
        backend._generate = lambda *a: seen.append((L._IN_FLIGHT[0], a)) or ("text", "stats")
        L._IN_FLIGHT[0] = 0
        with mock.patch.dict(os.environ, {L.FLAG: "1"}):
            self.assertEqual(backend.generate(5), ("text", "stats"))
        self.assertEqual(seen[0][0], 1, "counted while it ran")
        self.assertEqual(L._IN_FLIGHT[0], 0)
        self.assertEqual(seen[0][1][0], 5)

    def hooked(self, prompt, *, reusable=None, prefix_cache_runs=True, flag="1", report=None):
        from macqwen.backends.flashnext import FlashNextBackend

        backend = object.__new__(FlashNextBackend)
        backend.language = object()
        session = SimpleNamespace(reusable_tokens=lambda p: reusable)
        if isinstance(reusable, Exception):
            def broken(p):
                raise reusable
            session = SimpleNamespace(reusable_tokens=broken)
        backend._prefix_session = lambda: session
        with mock.patch.dict(os.environ, {L.FLAG: flag}), \
                mock.patch.object(L, "make_room", return_value=report if report is not None else {"reason": "released"}) as room:
            backend._make_room_for_prefill(prompt, prefix_cache_runs)
        return backend, room

    def test_the_backend_hook_sizes_the_prefill_by_the_suffix_after_a_prefix_hit(self):
        backend, room = self.hooked(list(range(2067)), reusable=2046)
        room.assert_called_once_with(backend.language, 21)               # a 21-token suffix never uses the ring
        self.assertEqual(backend.last_pool_release["reusable_tokens"], 2046)
        self.assertEqual(backend.last_pool_release["prompt_tokens"], 2067)

    def test_the_backend_hook_uses_the_whole_prompt_on_a_miss_or_without_the_cache(self):
        _, room = self.hooked(list(range(2068)), reusable=0)
        self.assertEqual(room.call_args.args[1], 2068)
        _, room = self.hooked(list(range(2068)), reusable=2046, prefix_cache_runs=False)
        self.assertEqual(room.call_args.args[1], 2068, "the cache does not run, so nothing is reused")

    def test_an_unreadable_prefix_cache_means_nothing_is_assumed_reused(self):
        backend, room = self.hooked(list(range(2068)), reusable=RuntimeError("corrupt"))
        self.assertEqual(room.call_args.args[1], 2068)
        self.assertEqual(backend.last_pool_release["reusable_tokens"], 0)

    def test_the_backend_hook_does_nothing_when_the_flag_is_off(self):
        backend, room = self.hooked(list(range(2068)), reusable=2046, flag="0")
        room.assert_not_called()
        self.assertIsNone(backend.last_pool_release)

    def test_the_backend_hook_keeps_a_none_report(self):
        backend, _ = self.hooked(list(range(10)), reusable=0, report=None)
        self.assertEqual(backend.last_pool_release["reason"], "released")
        self.assertIsNone(__import__("macqwen.backends.flashnext", fromlist=["x"]).FlashNextBackend.last_pool_release)

    def test_it_runs_after_the_cancel_check_and_before_every_prefill_path(self):
        import inspect

        from macqwen.backends.flashnext import FlashNextBackend

        source = inspect.getsource(FlashNextBackend._generate)
        decide = source.index("fresh_for_pool = ")
        extend = source.index("self.tape.extend(self.pending)")
        cancel = source.index("check_cancel()\n            self.last_pool_release = None")
        hook = source.index("self._make_room_for_prefill(prompt, fresh_prefix_cache, fresh_for_pool)")
        prefix = source.index("if decoder is None and fresh_prefix_cache:")
        self.assertLess(decide, extend, "freshness is decided before the tape changes")
        self.assertLess(cancel, hook)
        self.assertLess(hook, prefix)
        self.assertIn("decoder is None and (self._replay_needed or not self.tape) and self.cache_tokens == 0",
                      source[decide:extend])

    def test_only_the_backend_and_the_bench_import_the_module(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[4]
        files = list((root / "models" / "flashnext").glob("*.py")) + list((root / "macqwen").rglob("*.py"))
        users = sorted(str(p.relative_to(root)) for p in files
                       if p.name != "pool_lifecycle.py" and "pool_lifecycle" in p.read_text())
        self.assertEqual(users, [
            "macqwen/backends/flashnext.py",
            "macqwen/server.py",                  # /ready reports the recorded GPU lifecycle fault (read-only)
            "macqwen/tests/test_server_runtime.py",
            "models/flashnext/auto_reclaim.py",
            "models/flashnext/gpu_sync.py",
            "models/flashnext/memory_manager.py",
            "models/flashnext/prefill.py",        # reads the store only to switch to uncached reads while the controller is ELEVATED
            "models/flashnext/prefill_ring.py",
        ])
        self.assertTrue((root / "macqwen" / "backends" / "flashnext.py").exists(), "the scan really looked at the repo")

    def test_the_flag_is_a_registered_research_setting_and_off_by_default(self):
        from models.flashnext.settings.registry import get_registry

        names = {s.env_key: s for s in get_registry().for_backend("flashnext") if s.env_key}
        self.assertIn(L.FLAG, names)
        self.assertEqual(names[L.FLAG].default, "1")                   # the value the research toggle applies
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(L.FLAG, None)
            self.assertFalse(L.enabled())

    def test_it_does_not_touch_prefix_cache_snapshots(self):
        import pathlib

        text = pathlib.Path(L.__file__).read_text()
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#", '"""')))
        self.assertNotIn("import prefix_cache", code)
        self.assertNotIn("from .prefix_cache", code)

    def test_make_room_is_a_no_op_when_the_flag_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(L.FLAG, None)
            self.assertIsNone(L.make_room(SimpleNamespace(), 5000))


if __name__ == "__main__":
    unittest.main()
