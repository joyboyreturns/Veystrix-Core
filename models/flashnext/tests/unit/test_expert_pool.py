"""Expert pool: LRU slot assignment and byte-exact record fills."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_cache, expert_pool
from models.flashnext.slab_pack import get_slab_layout
from models.flashnext.store import SafeTensorStore

PREFIX = "language_model.model.layers.4.mlp.switch_mlp"
ROWS = {"weight": 819_200, "scales": 102_400, "biases": 102_400}


class _Store:
    dir = "/nonexistent"
    refs = {}


class AssignTests(unittest.TestCase):
    def make(self, slots):
        stride = get_slab_layout(32).record_stride
        return expert_pool.ExpertPool(_Store(), expert_pool.HEADER + slots * stride)

    def test_hits_misses_and_lru_eviction(self):
        pool = self.make(16)
        slots, misses = pool.assign(1, list(range(10)))
        self.assertEqual(len(misses), 10)
        self.assertEqual(len(set(slots.values())), 10)
        slots2, misses2 = pool.assign(1, [3, 4, 20])
        self.assertEqual(slots2[3], slots[3])
        self.assertEqual([e for e, _ in misses2], [20])
        # Fill the pool, then force evictions: the oldest keys go first and
        # keys routed in the same call survive.
        pool.assign(2, list(range(5)))           # 16 slots now used
        slots3, misses3 = pool.assign(1, [3, 4, 30, 31])
        self.assertEqual(slots3[3], slots[3])
        self.assertEqual(slots3[4], slots[4])
        evicted = {slots[0], slots[1]}
        self.assertEqual({slot for _, slot in misses3}, evicted)
        self.assertEqual(pool.hits, 2 + 2)
        self.assertEqual(pool.misses, 10 + 1 + 5 + 2)

    def test_too_small_pool_is_refused(self):
        with self.assertRaises(ValueError):
            self.make(4)

    def test_oversized_assignment_fails_before_evicting_or_claiming_slots(self):
        pool = self.make(16)
        pool.assign(1, [1, 2, 3])
        before = (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses)
        with self.assertRaisesRegex(ValueError, "exceed"):
            pool.assign(1, list(range(17)))
        self.assertEqual(before, (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses))

    def test_leased_ring_capacity_prevents_slot_aliasing(self):
        from models.flashnext import pool_ring

        pool = self.make(40)
        self.assertEqual(pool.usable_slots, 40)
        first = pool_ring.lease_range(pool, 20)
        self.assertEqual(first, 20)
        self.assertEqual(pool.usable_slots, 20)

        # 16 distinct experts fits within the 20 unleased slots
        slots, misses = pool.assign(1, list(range(16)))
        self.assertEqual(len(slots), 16)
        self.assertEqual(len(set(slots.values())), 16)
        self.assertTrue(all(s < 20 for s in slots.values()), "all assigned slots must be unleased (< 20)")

        # 25 distinct experts exceeds the 20 unleased slots (even though total slots is 40)
        before = (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses)
        with self.assertRaisesRegex(ValueError, "exceed"):
            pool.assign(1, list(range(25)))
        # Pool state must be completely unmutated: no aliasing, no partial eviction
        self.assertEqual(before, (list(pool._slot_of.items()), list(pool._free), pool.hits, pool.misses))
        self.assertEqual(len(set(pool._slot_of.values())), 16, "no duplicated live slot assignments")


class FillTests(unittest.TestCase):
    def test_oversized_batch_uses_streaming_path_without_changing_pool(self):
        pool = AssignTests().make(16)
        pool.assign(4, [30])
        prior = list(pool._slot_of.items())
        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        fallback = mock.Mock(return_value=mx.array([1.0], dtype=mx.float32))
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0, _one_pass=fallback)
        routes = list(range(20))
        with mock.patch.object(expert_pool, "for_store", return_value=pool):
            result = expert_cache.StreamingSwitchGLU._pool_pass(layer, "indices", routes, "input", "scores", None, None, None)
        self.assertEqual(result.dtype, mx.bfloat16)
        self.assertEqual(result.tolist(), [1.0])
        fallback.assert_called_once_with(None, "indices", routes, True, "input", "scores", None, None, None)
        self.assertEqual(list(pool._slot_of.items()), prior)

    def test_pool_pass_with_leased_ring_falls_back_to_streamed_path_when_unleased_capacity_exceeded(self):
        from models.flashnext import pool_ring

        pool = AssignTests().make(40)
        pool_ring.lease_range(pool, 20)
        self.assertEqual(pool.usable_slots, 20)

        # Pre-assign 5 experts
        pool.assign(4, [100, 101, 102, 103, 104])
        prior = list(pool._slot_of.items())

        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        fallback = mock.Mock(return_value=mx.array([1.0], dtype=mx.float32))
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0, _one_pass=fallback)

        # 8 tokens with top-k routing 25 distinct experts: exceeds the 20 unleased slots (though 25 < total slots 40)
        routes = list(range(25))
        with mock.patch.object(expert_pool, "for_store", return_value=pool):
            result = expert_cache.StreamingSwitchGLU._pool_pass(layer, "indices", routes, "input", "scores", None, None, None)

        self.assertEqual(result.dtype, mx.bfloat16)
        fallback.assert_called_once_with(None, "indices", routes, True, "input", "scores", None, None, None)
        # Verify pool was not corrupted or aliased
        self.assertEqual(list(pool._slot_of.items()), prior)
        self.assertEqual(len(set(pool._slot_of.values())), len(pool._slot_of))

    def test_parallel_descriptor_initialization_opens_once(self):
        pool = AssignTests().make(16)
        def open_slowly(*args):
            time.sleep(0.02)
            return 123
        with mock.patch.object(expert_pool.os, "open", side_effect=open_slowly) as opened, \
             mock.patch.object(expert_pool.fcntl, "fcntl"), \
             mock.patch.dict(os.environ, {expert_pool.BUFFERED_FILL_FLAG: "1"}), \
             ThreadPoolExecutor(8) as workers:
            self.assertEqual(list(workers.map(pool._fill_fd, ["shard"] * 8)), [123] * 8)
        self.assertEqual(opened.call_count, 1)

    def test_partial_submission_is_drained(self):
        pool = AssignTests().make(16)
        _, misses = pool.assign(4, [3])
        first = mock.Mock()
        submit = mock.Mock(side_effect=[first, RuntimeError("submission failed")])
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            pool.fill(PREFIX, misses, submit)
        first.result.assert_called_once_with()

    def test_failed_pool_pass_drains_workers_and_invalidates_misses(self):
        pool = AssignTests().make(16)
        pool.assign(4, [1])  # existing resident expert survives
        failed = mock.Mock()
        failed.result.side_effect = OSError("read failed")
        later = mock.Mock()
        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0)
        with mock.patch.object(expert_pool, "for_store", return_value=pool), \
             mock.patch.object(pool, "fill", return_value=[failed, later]):
            with self.assertRaisesRegex(OSError, "read failed"):
                expert_cache.StreamingSwitchGLU._pool_pass(layer, None, [1, 2], None, None, None, None, None)
        later.result.assert_called_once_with()
        self.assertIn((4, 1), pool._slot_of)
        self.assertNotIn((4, 2), pool._slot_of)
        _, retry = pool.assign(4, [2])
        self.assertEqual([e for e, _ in retry], [2])

    def test_failed_submission_invalidates_assigned_slot(self):
        pool = AssignTests().make(16)
        first = mock.Mock()
        submit = mock.Mock(side_effect=[first, RuntimeError("submit failed")])
        gate = SimpleNamespace(cache=SimpleNamespace(store=_Store(), prefix=PREFIX + ".gate_proj"), group_size=32)
        layer = SimpleNamespace(layer_id=4, gate_proj=gate, hits=0, misses=0)
        with mock.patch.object(expert_pool, "for_store", return_value=pool), \
             mock.patch.object(expert_cache, "_submit_read", submit):
            with self.assertRaisesRegex(RuntimeError, "submit failed"):
                expert_cache.StreamingSwitchGLU._pool_pass(layer, None, [2], None, None, None, None, None)
        first.result.assert_called_once_with()
        self.assertNotIn((4, 2), pool._slot_of)
        self.assertEqual([e for e, _ in pool.assign(4, [2])[1]], [2])

    def test_fill_places_checkpoint_rows_at_record_offsets(self):
        mx.random.seed(8)
        experts = 5
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part, size in ROWS.items():
                tensors[f"{PREFIX}.{projection}.{part}"] = mx.random.randint(
                    0, 255, (experts, size)
                ).astype(mx.uint8)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            layout = get_slab_layout(32)
            pool = expert_pool.ExpertPool(
                store, expert_pool.HEADER + 16 * layout.record_stride)
            workers = ThreadPoolExecutor(4)
            try:
                slots, misses = pool.assign(4, [3, 0, 4])
                futures = pool.fill(PREFIX, misses, workers.submit, chunk=2)
                for future in futures:
                    future.result()
                for expert, slot in slots.items():
                    base = expert_pool.HEADER + slot * layout.record_stride
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for part, size in ROWS.items():
                            start = base + layout.offset(projection, part)
                            expected = store.rows_np(
                                f"{PREFIX}.{projection}.{part}", [expert]
                            ).tobytes()
                            self.assertEqual(
                                bytes(pool.array[start:start + size]), expected,
                                f"expert {expert} {projection}.{part}",
                            )
            finally:
                workers.shutdown()
                pool.close()
                store.close()


class ConfigTests(unittest.TestCase):
    def test_off_by_default_and_parsed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(expert_pool.SIZE_FLAG, None)
            self.assertFalse(expert_pool.enabled())
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "6"}):
            self.assertEqual(expert_pool.configured_gb(), 6.0)
            self.assertTrue(expert_pool.enabled())

    def test_for_store_allocation_fallback_to_1gb_on_error(self):
        store = _Store()
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "2"}):
            self.addCleanup(expert_pool.set_budget_gb, None)
            calls = []

            def fake_init(st, size_bytes, *args, **kwargs):
                calls.append(size_bytes)
                if size_bytes == 2_000_000_000:
                    raise MemoryError("simulated out of memory at 2 GB")
                return None

            with mock.patch.object(expert_pool.ExpertPool, "__init__", side_effect=fake_init):
                pool = expert_pool.for_store(store)
                self.assertIsNotNone(pool)
                self.assertEqual(calls, [2_000_000_000, 1_000_000_000])
                self.assertEqual(expert_pool.configured_gb(), 1.0)

    def test_for_store_allocation_fallback_to_ssd_when_1gb_fails(self):
        store = _Store()
        with mock.patch.dict(os.environ, {expert_pool.SIZE_FLAG: "2"}):
            self.addCleanup(expert_pool.set_budget_gb, None)

            with mock.patch.object(expert_pool.ExpertPool, "__init__", side_effect=MemoryError("OOM")):
                pool = expert_pool.for_store(store)
                # Falls back to pure SSD (pool is None) without raising
                self.assertIsNone(pool)
                self.assertIsNone(getattr(store, "_flashnext_expert_pool", None))


class BufferedFillTests(unittest.TestCase):
    """The opt-in buffered fill reads the same bytes through separate descriptors; the uncached ones stay for the ring."""

    def _run(self, buffered):
        mx.random.seed(11)
        tensors = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part, size in ROWS.items():
                tensors[f"{PREFIX}.{projection}.{part}"] = mx.random.randint(0, 255, (5, size)).astype(mx.uint8)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            layout = get_slab_layout(32)
            pool = expert_pool.ExpertPool(store, expert_pool.HEADER + 16 * layout.record_stride)
            workers = ThreadPoolExecutor(4)
            env = {expert_pool.BUFFERED_FILL_FLAG: "1"} if buffered else {expert_pool.BUFFERED_FILL_FLAG: "0"}
            try:
                with mock.patch.dict(os.environ, env):
                    slots, misses = pool.assign(4, [4, 0, 1, 3])
                    with mock.patch.object(os, "preadv", wraps=os.preadv) as read:
                        for future in pool.fill(PREFIX, misses, workers.submit, chunk=2):
                            future.result()
                    self.assertEqual(read.call_count, 2 * len(tensors))
                    data = {}
                    for expert, slot in slots.items():
                        base = expert_pool.HEADER + slot * layout.record_stride
                        for projection in ("gate_proj", "up_proj", "down_proj"):
                            for part, size in ROWS.items():
                                start = base + layout.offset(projection, part)
                                row = bytes(pool.array[start:start + size])
                                self.assertEqual(row, np.array(tensors[f"{PREFIX}.{projection}.{part}"][expert]).tobytes())
                                data[(expert, projection, part)] = row
                    return data, dict(pool._fds), dict(pool._fill_fds)
            finally:
                workers.shutdown()
                pool.close()
                store.close()

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(expert_pool.BUFFERED_FILL_FLAG, None)
            self.assertFalse(expert_pool.buffered_fill_enabled())

    def test_buffered_fill_is_byte_identical_and_uses_separate_descriptors(self):
        reference, ref_uncached, ref_buffered = self._run(False)
        candidate, cand_uncached, cand_buffered = self._run(True)
        self.assertEqual(reference, candidate)
        self.assertTrue(ref_uncached and not ref_buffered)       # reference: only the shared F_NOCACHE descriptors
        self.assertTrue(cand_buffered and not cand_uncached)     # opt-in: own descriptors, shared ones untouched

    def test_close_releases_both_descriptor_sets(self):
        stride = get_slab_layout(32).record_stride
        pool = expert_pool.ExpertPool(_Store(), expert_pool.HEADER + 16 * stride)
        read, write = os.pipe()
        os.close(write)
        pool._fill_fds["x"] = read
        pool.close()
        self.assertEqual(pool._fill_fds, {})
        with self.assertRaises(OSError):
            os.fstat(read)


if __name__ == "__main__":
    unittest.main()
