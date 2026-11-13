"""Stage 1 prefill ring: exact tensors, zero-copy views, ownership, fallback.

A small synthetic checkpoint with the real expert row sizes (hidden 2560,
intermediate 640, Q4/G32) stands in for the model. Two real
``StreamingSwitchGLU`` layers run through the real ``SafeTensorStore`` and a real
``ExpertPool``; nothing is downloaded and no model weights are needed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from models.flashnext import expert_cache, expert_pool, prefill, prefill_ring
from models.flashnext.pool_ring import FREE, HazardError
from models.flashnext.store import SafeTensorStore

EXPERTS = 12
HIDDEN = 2560
INTER = 640
SHAPES = {
    "gate_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
    "up_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
    "down_proj": (HIDDEN, INTER // 8, INTER // 32),
}
LAYERS = (1, 2)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ENV = {
    "FLASHNEXT_EXPERT_POOL_GB": "0.15",
    "FLASHNEXT_PREFILL_RING_EXPERTS": "8",
    "FLASHNEXT_PREFILL_RING_MIN_TOKENS": "0",
    "FLASHNEXT_PREFILL_RING_MARGIN_GB": "0.5",
    "FLASHNEXT_PREFILL_RING_TRANSIENT_GB": "0.1",
}


def silu_glu(up, gate):
    return nn.silu(gate) * up


def prefix_of(layer):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp"


def build_checkpoint(directory):
    mx.random.seed(11)
    tensors = {}
    for layer in LAYERS:
        for projection, (rows, packed, groups) in SHAPES.items():
            base = f"{prefix_of(layer)}.{projection}"
            tensors[f"{base}.weight"] = mx.random.randint(
                0, 2**31 - 1, (EXPERTS, rows, packed), dtype=mx.uint32
            )
            tensors[f"{base}.scales"] = (
                mx.random.uniform(0.001, 0.01, (EXPERTS, rows, groups))
            ).astype(mx.bfloat16)
            tensors[f"{base}.biases"] = (
                mx.random.uniform(-0.01, 0.01, (EXPERTS, rows, groups))
            ).astype(mx.bfloat16)
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(os.path.join(directory, shard), tensors)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
        json.dump({"weight_map": {key: shard for key in tensors}}, out)


_REAL_METAL_LIMIT = prefill_ring.metal_working_set_limit
# The ring is only granted when the machine's Metal working-set limit has room for it, so on a small
# runner these tests got no ring at all. Tests that exercise the admission arithmetic patch the limit
# themselves; everything else gets this fixed, generous one and no longer depends on the machine.
TEST_WORKING_SET_LIMIT = 64 * 2**30


class RingCase(unittest.TestCase):
    """Builds the store, pool and two streamed layers once per test."""

    pool_gb = "0.15"
    create_main_pool = True         # False: a first turn, before any decode pool
    extra_env = {}

    def need_ring(self, ring):
        """A ring is granted only when this machine has the memory and Metal limits for it (a small CI runner may not)."""
        if ring is None:
            self.skipTest(f"scratch ring not granted on this machine: {prefill_ring.LAST_REASON[0]}")

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        build_checkpoint(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        patcher = mock.patch.dict(
            os.environ,
            {**ENV, "FLASHNEXT_EXPERT_POOL_GB": self.pool_gb, **self.extra_env})
        patcher.start()
        self.addCleanup(patcher.stop)
        limit_patch = mock.patch.object(
            prefill_ring, "metal_working_set_limit", return_value=TEST_WORKING_SET_LIMIT)
        limit_patch.start()
        self.addCleanup(limit_patch.stop)
        self.store = SafeTensorStore(self.directory.name)
        self.addCleanup(self.store.close)
        self.layers = [
            expert_cache.StreamingSwitchGLU(
                self.store, prefix_of(layer), 32, 4, "affine", silu_glu,
                layer_id=layer,
            )
            for layer in LAYERS
        ]
        self.language = SimpleNamespace(model=SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(switch_mlp=glu))
            for glu in self.layers
        ]))
        self.addCleanup(expert_cache.set_prefill_ring, None)
        self.pool = None
        self.free_before = None
        if self.create_main_pool:
            self.pool = expert_pool.for_store(self.store, 32)
            self.addCleanup(self.pool.close)
            self.free_before = sorted(self.pool._free)

    def inputs(self, tokens=16, width=4, seed=3, experts=8):
        """Routes over the first ``experts`` experts (the ring half holds 8)."""
        mx.random.seed(seed)
        x = mx.random.normal((1, tokens, HIDDEN)).astype(mx.bfloat16)
        indices = mx.random.randint(0, experts, (1, tokens, width)).astype(mx.uint32)
        return x, indices

    def run_layers(self, x, indices):
        outputs = []
        for glu in self.layers:
            y = glu(x, indices)
            mx.eval(y)            # the router sync of the next layer does this
            outputs.append(y)
        return outputs

    def assertRestored(self):
        self.assertEqual(sorted(self.pool._free), self.free_before)
        self.assertIsNone(expert_cache._PREFILL_RING[0])


class TensorCorrectnessTests(RingCase):
    def test_prefill_seeds_the_hottest_complete_record_without_a_decode_read(self):
        routed = [7, 2, 8, 2, 5, 2]
        wanted = list(dict.fromkeys(routed))
        prefix = prefix_of(1)
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring.stage(self.store, prefix, wanted))
            ring.seed_decode(1, {expert: i for i, expert in enumerate(wanted)}, routed)
            self.assertEqual(ring.stats()["seeded_bytes"], self.pool.stride)
        slots, misses = self.pool.assign(1, [2])
        self.assertFalse(misses)
        base = expert_pool.HEADER + slots[2] * self.pool.stride
        for (projection, part), offset in self.pool.offsets.items():
            expected = self.store.rows_np(f"{prefix}.{projection}.{part}", [2]).tobytes()
            self.assertEqual(bytes(self.pool.array[base + offset:base + offset + len(expected)]), expected)

    def test_ring_output_is_bitwise_equal_to_existing_path(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            self.assertIs(expert_cache._PREFILL_RING[0], ring)
            produced = self.run_layers(x, indices)
            stats = ring.stats()
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))
        self.assertEqual(stats["staged_layers"], len(LAYERS))
        self.assertEqual(stats["fallback_error"], 0)
        self.assertRestored()

    def test_staged_views_hold_the_checkpoint_rows_in_first_seen_order(self):
        routed = [7, 2, 8, 11, 2, 5]
        wanted = list(dict.fromkeys(routed))
        self.assertEqual(wanted, [7, 2, 8, 11, 5])
        with prefill_ring.session(self.language, 1024) as ring:
            prefix = prefix_of(1)
            weights = ring.stage(self.store, prefix, wanted)
            self.assertIsNotNone(weights)
            for projection, parts in zip(PROJECTIONS, weights):
                for kind, array in zip(prefill_ring.KINDS, parts):
                    name = f"{prefix}.{projection}.{kind}"
                    expected = self.store.rows_np(name, wanted)
                    got = np.array(array.view(mx.uint16) if kind != "weight" else array)
                    self.assertEqual(got.shape, expected.shape, name)
                    self.assertEqual(got.tobytes(), expected.tobytes(), name)
                    self.assertEqual(
                        array.dtype, mx.uint32 if kind == "weight" else mx.bfloat16
                    )

    def test_views_are_zero_copy(self):
        with prefill_ring.session(self.language, 1024) as ring:
            mx.eval(mx.zeros(1))
            before = mx.get_active_memory()
            weights = ring.stage(self.store, prefix_of(1), list(range(8)))
            mx.eval([array for parts in weights for array in parts])
            grown = mx.get_active_memory() - before
        # Eight experts are ~24.6 MB; a copy of any projection is >= 6 MB.
        self.assertLess(grown, 1_000_000)

    def test_only_the_leased_range_is_written(self):
        header = expert_pool.HEADER
        with prefill_ring.session(self.language, 1024) as ring:
            first = ring.ring.first_slot
            below = header + first * self.pool.stride
            digest = hashlib.sha256(self.pool.array[:below]).digest()
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), list(range(8))))
            self.assertEqual(
                hashlib.sha256(self.pool.array[:below]).digest(), digest
            )
            self.assertGreaterEqual(first, 16)

    def test_decode_sized_calls_never_touch_the_ring(self):
        x, indices = self.inputs(tokens=4)
        with prefill_ring.session(self.language, 1024) as ring:
            y = self.layers[0](x, indices)
            mx.eval(y)
            self.assertEqual(ring.stats()["staged_layers"], 0)
        self.assertRestored()


class HighOffsetTests(RingCase):
    """A 4.6 GB pool puts the ring above 4.29 GB, where a bf16 element index
    passes 2**31. Only the touched ring pages become resident."""

    pool_gb = "4.6"

    def test_views_beyond_the_32_bit_element_range_hold_the_checkpoint_rows(self):
        wanted = [9, 3, 11, 0, 6]
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            top = (expert_pool.HEADER + ring.ring.first_slot * self.pool.stride)
            self.assertGreater(top // 2, 2**31 - 1)   # old bf16 index overflowed
            weights = ring.stage(self.store, prefix_of(1), wanted)
            self.assertIsNotNone(weights, ring.stats()["last_error"])
            for projection, parts in zip(PROJECTIONS, weights):
                for kind, array in zip(prefill_ring.KINDS, parts):
                    name = f"{prefix_of(1)}.{projection}.{kind}"
                    expected = self.store.rows_np(name, wanted)
                    got = np.array(array.view(mx.uint16) if kind != "weight" else array)
                    self.assertEqual(got.tobytes(), expected.tobytes(), name)

    def test_output_is_bitwise_equal_with_the_ring_high_in_a_big_pool(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        with prefill_ring.session(self.language, 1024) as ring:
            produced = self.run_layers(x, indices)
            self.assertEqual(ring.stats()["staged_layers"], len(LAYERS))
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))


class FallbackTests(RingCase):
    def reference_and_ring(self, x, indices, ring):
        reference = self.run_layers(x, indices)
        produced = self.run_layers(x, indices)
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))
        return ring.stats()

    def test_more_experts_than_a_half_falls_back(self):
        x, indices = self.inputs(tokens=64, width=8, seed=9, experts=EXPERTS)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "2"}):
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertEqual(ring.ring.half_experts, 2)
                stats = self.reference_and_ring(x, indices, ring)
        self.assertEqual(stats["staged_layers"], 0)
        self.assertGreaterEqual(stats["fallback_capacity"], 1)
        self.assertRestored()

    def test_read_error_releases_the_half_and_falls_back(self):
        x, indices = self.inputs()
        with prefill_ring.session(self.language, 1024) as ring:
            self.need_ring(ring)
            with mock.patch.object(prefill_ring, "_read_rows", side_effect=OSError("boom")):
                got = self.layers[0](x, indices)
                mx.eval(got)
            self.assertEqual(ring.stats()["fallback_error"], 1)
            self.assertIn("boom", ring.stats()["last_error"])
            self.assertEqual([h.state for h in ring.ring._halves], [FREE])
            # The ring keeps working for the next layer.
            again = self.layers[1](x, indices)
            mx.eval(again)
            self.assertEqual(ring.stats()["staged_layers"], 1)
        want = self.layers[0](x, indices)
        mx.eval(want)
        self.assertTrue(bool(mx.array_equal(got, want)))
        self.assertRestored()

    def test_short_read_is_an_error_not_silent_garbage(self):
        with prefill_ring.session(self.language, 1024) as ring:
            with mock.patch.object(prefill_ring.os, "preadv", return_value=7):
                self.assertIsNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            self.assertIn("short expert read", ring.stats()["last_error"])
            self.assertEqual([h.state for h in ring.ring._halves], [FREE])

    def test_busy_half_falls_back(self):
        with prefill_ring.session(self.language, 1024) as ring:
            held = ring.ring.begin_fill()
            self.assertIsNotNone(held)
            self.assertIsNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            self.assertEqual(ring.stats()["fallback_busy"], 1)
            held.release()
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [0, 1]))

    def test_memory_pressure_disables_the_ring_for_the_rest_of_the_prefill(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            ring._active_memory = lambda: ring._limit      # no room left
            self.assertIsNone(ring.stage(self.store, prefix_of(2), [0, 1]))
            self.assertEqual(ring.disabled, "working set")
            ring._active_memory = lambda: 0                # stays off anyway
            self.assertIsNone(ring.stage(self.store, prefix_of(2), [0, 1]))
            self.assertEqual(ring.stats()["fallback_disabled"], 1)
        self.assertRestored()

    def test_empty_request_falls_back(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNone(ring.stage(self.store, prefix_of(1), []))


class OverlapStagingTests(RingCase):
    """Two-half ring for layer-major prefetch: the next layer's rows are read on the other half while one computes."""

    extra_env = {"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_PREFILL_LAYER_PREFETCH": "1",
                 "FLASHNEXT_PREFILL_RING_EXPERTS": "12"}

    def rows_equal(self, weights, prefix, wanted):
        for projection, parts in zip(PROJECTIONS, weights):
            for kind, array in zip(prefill_ring.KINDS, parts):
                name = f"{prefix}.{projection}.{kind}"
                expected = self.store.rows_np(name, wanted)
                got = np.array(array.view(mx.uint16) if kind != "weight" else array)
                self.assertEqual(got.tobytes(), expected.tobytes(), name)

    def test_two_halves_only_when_layer_major_prefetch_is_on(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            self.assertEqual(ring.ring.halves, 2)
            self.assertTrue(ring.overlap_capable)
        for env in ({"FLASHNEXT_PREFILL_LAYER_PREFETCH": "0"}, {"FLASHNEXT_PREFILL_ORDER": "chunk"}):
            with mock.patch.dict(os.environ, env):
                self.assertEqual(prefill_ring.ring_halves(), 1)
        self.assertRestored()

    def test_overlapped_staging_is_byte_identical_and_uses_the_other_half(self):
        wanted = list(range(12))
        with prefill_ring.session(self.language, 1024) as ring:
            first = ring.stage_begin(self.store, prefix_of(1), wanted)
            self.assertIsNotNone(first)
            w1 = ring.stage_finish(first)                       # layer 1 is computing on half A
            self.rows_equal(w1, prefix_of(1), wanted)
            second = ring.stage_begin(self.store, prefix_of(2), wanted)     # layer 2 reads on half B meanwhile
            self.assertIsNotNone(second)
            self.assertNotEqual(first.lease.index, second.lease.index)
            self.assertEqual(sorted(h.state for h in ring.ring._halves), ["computing", "filling"])
            self.assertIsNone(ring.stage_begin(self.store, prefix_of(1), wanted))   # no free half: no third staging
            self.assertEqual(ring.stats()["fallback_busy"], 1)
            ring._retire()                                      # layer 1's GPU work is done
            w2 = ring.stage_finish(second)
            self.rows_equal(w2, prefix_of(2), wanted)
            self.assertEqual(ring.stats()["staged_layers"], 2)
        self.assertRestored()

    def test_cancel_waits_for_the_workers_and_frees_the_half(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.need_ring(ring)
            staging = ring.stage_begin(self.store, prefix_of(1), list(range(12)))
            self.assertIsNotNone(staging)
            ring.cancel_stage(staging)
            self.assertTrue(all(f.done() for f in staging.futures))
            self.assertEqual([h.state for h in ring.ring._halves], [FREE, FREE])
        self.assertRestored()

    def test_a_failed_read_at_finish_releases_the_half_and_reports_it(self):
        with prefill_ring.session(self.language, 1024) as ring:
            with mock.patch.object(prefill_ring, "_read_rows", side_effect=OSError("boom")):
                staging = ring.stage_begin(self.store, prefix_of(1), list(range(12)))
                self.assertIsNotNone(staging)
                self.assertIsNone(ring.stage_finish(staging))   # the caller then reads the layer the ordinary way
            self.assertIn("boom", ring.stats()["last_error"])
            self.assertEqual(ring.stats()["fallback_error"], 1)
            self.assertEqual([h.state for h in ring.ring._halves], [FREE, FREE])
        self.assertRestored()

    def test_one_half_ring_never_overlaps(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_LAYER_PREFETCH": "0"}):
            with prefill_ring.session(self.language, 1024) as ring:
                self.need_ring(ring)
                self.assertEqual(ring.ring.halves, 1)
                self.assertFalse(ring.overlap_capable)
                self.assertIsNone(ring.stage_begin(self.store, prefix_of(1), [0, 1]))
        self.assertRestored()

    def test_window_prefetches_the_next_layer_and_consumes_it_exactly(self):
        from models.flashnext import layer_major

        wanted = list(range(12))
        with prefill_ring.session(self.language, 1024) as ring:
            window = layer_major.ExpertWindow(ring=ring)
            self.assertTrue(window.prefetch(self.layers[1]))             # layer id 2, begun a layer early
            self.assertFalse(window.prefetch(self.layers[1]))            # never the same layer twice
            weights = window.weights_for(self.layers[1])
            self.rows_equal(weights, prefix_of(2), wanted)
            self.assertEqual(window.stats["prefetch_used"], 1)
            self.assertEqual(window.stats["ring_layers"], 1)
            self.assertEqual(window.stats["fallback_layers"], 0)
            window.end_layer()
            window.drain()
        self.assertRestored()

    def test_window_keeps_one_staging_waiting_while_the_next_begins(self):
        """The steady state at the start of layer l+1: its own staging not yet consumed, layer l+2 begins on the freed half."""
        from models.flashnext import layer_major

        with prefill_ring.session(self.language, 1024) as ring:
            window = layer_major.ExpertWindow(ring=ring)
            self.assertTrue(window.prefetch(self.layers[0]))             # layer id 1 -> half B
            first = window.weights_for(self.layers[0])                   # computing on B
            self.assertTrue(window.prefetch(self.layers[1]))             # layer id 2 -> half A, filling meanwhile
            self.assertEqual(sorted(h.state for h in ring.ring._halves), ["computing", "filling"])
            self.assertFalse(window.prefetch(self.layers[0]))            # no free half for a third
            window.end_layer()                                           # layer 1's GPU work done: B is free again
            self.assertTrue(window.prefetch(self.layers[0]))             # the next layer's read may begin
            second = window.weights_for(self.layers[1])
            self.rows_equal(second, prefix_of(2), list(range(12)))
            self.assertEqual(window.stats["prefetch_used"], 2)
            window.end_layer()
            window.drain()
        self.assertRestored()

    def test_window_drain_cancels_an_unconsumed_staging(self):
        from models.flashnext import layer_major

        with prefill_ring.session(self.language, 1024) as ring:
            window = layer_major.ExpertWindow(ring=ring)
            self.assertTrue(window.prefetch(self.layers[0]))
            window.abort_layer()
            self.assertEqual([h.state for h in ring.ring._halves], [FREE, FREE])
            self.assertEqual(window._staged, {})
        self.assertRestored()

    def test_window_falls_back_to_the_blocking_stage_when_nothing_was_begun(self):
        from models.flashnext import layer_major

        with prefill_ring.session(self.language, 1024) as ring:
            window = layer_major.ExpertWindow(ring=ring)
            weights = window.weights_for(self.layers[0])
            self.rows_equal(weights, prefix_of(1), list(range(12)))
            self.assertEqual(window.stats["prefetch_used"], 0)
            self.assertEqual(window.stats["ring_layers"], 1)
            window.end_layer()
        self.assertRestored()


class AcquireTests(RingCase):
    def reason(self):
        return prefill_ring.LAST_REASON[0]

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "0"}):
            self.assertFalse(prefill_ring.enabled())
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertIsNone(ring)
                self.assertIsNone(expert_cache._PREFILL_RING[0])
        os.environ.pop("FLASHNEXT_PREFILL_RING_EXPERTS")
        self.assertFalse(prefill_ring.enabled())

    def test_short_prompts_do_not_use_the_ring(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_MIN_TOKENS": "512"}):
            self.assertIsNone(prefill_ring.acquire(self.language, 100))
            self.assertEqual(self.reason(), "short prompt")
            state = prefill_ring.acquire(self.language, 512)
            self.assertIsNotNone(state)
            prefill_ring.release(state)
        self.assertRestored()

    def test_requires_the_expert_pool(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_EXPERT_POOL_GB": "0"}):
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            self.assertEqual(self.reason(), "expert pool and scratch ring are off")

    def test_refuses_when_the_working_set_does_not_fit(self):
        # Active memory is patched: reading the real value twice races with
        # MLX freeing earlier tests' arrays asynchronously. It already
        # contains the pool, which must count once.
        active = len(self.pool._mm) + 100_000_000
        limit = active + 400_000_000
        with mock.patch.object(mx, "get_active_memory", return_value=active), \
                mock.patch.object(mx, "get_cache_memory", return_value=0):
            with mock.patch.object(
                prefill_ring, "metal_working_set_limit", return_value=limit
            ):
                # 0.1 GB transient + 0.5 GB margin does not fit in 0.4 GB.
                self.assertIsNone(prefill_ring.acquire(self.language, 1024))
                self.assertTrue(
                    self.reason().startswith("working set: MLX active"), self.reason())
                self.assertIn("exceeds limit", self.reason())
            with mock.patch.object(
                prefill_ring, "metal_working_set_limit",
                return_value=limit + 400_000_000,
            ):
                state = prefill_ring.acquire(self.language, 1024)
                self.assertIsNotNone(state)
                prefill_ring.release(state)
        self.assertRestored()

    def test_the_pool_is_counted_once(self):
        pool_bytes = len(self.pool._mm)
        self.assertEqual(prefill_ring.working_set_bytes(pool_bytes + 5, pool_bytes),
                         pool_bytes + 5)
        self.assertEqual(prefill_ring.working_set_bytes(7, pool_bytes), 7 + pool_bytes)
        # The fact the formula rests on, checked against MLX itself: every live
        # wrapped pool is inside active memory. A lower bound, not a before and
        # after difference, because MLX frees earlier tests' arrays
        # asynchronously and would make a difference unreliable.
        stride = self.pool.stride
        second = expert_pool.ExpertPool(
            self.store, expert_pool.HEADER + 20 * stride, 32)
        try:
            live = len(self.pool._mm) + len(second._mm)
            self.assertGreaterEqual(mx.get_active_memory(), live)
        finally:
            second.close()

    def test_a_pool_sized_to_fit_is_accepted_once_counted_once(self):
        # Room for the pool once plus transient (0.1 GB) and margin (0.5 GB),
        # but not for the pool counted twice.
        pool_bytes = len(self.pool._mm)
        active = pool_bytes + 100_000_000
        limit = active + 100_000_000 + 500_000_000 + 50_000_000
        self.assertLess(limit - active, pool_bytes + 100_000_000 + 500_000_000)
        with mock.patch.object(mx, "get_active_memory", return_value=active), \
                mock.patch.object(mx, "get_cache_memory", return_value=0):
            with mock.patch.object(
                prefill_ring, "metal_working_set_limit", return_value=limit
            ):
                state = prefill_ring.acquire(self.language, 1024)
                self.assertIsNotNone(state, self.reason())
                prefill_ring.release(state)

    def test_refuses_when_the_limit_is_unknown(self):
        with mock.patch.object(prefill_ring, "metal_working_set_limit", return_value=None):
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            self.assertEqual(self.reason(), "working-set limit unknown")

    def test_refuses_a_layout_it_cannot_view(self):
        ref = self.store.refs[f"{prefix_of(1)}.gate_proj.weight"]
        original = ref.dtype
        ref.dtype = "F32"
        try:
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            self.assertIn("unsupported dtype", self.reason())
        finally:
            ref.dtype = original

    def test_no_streamed_layers(self):
        empty = SimpleNamespace(model=SimpleNamespace(layers=[]))
        self.assertIsNone(prefill_ring.acquire(empty, 1024))
        self.assertEqual(self.reason(), "no streamed layers")

    def test_real_device_reports_a_working_set_limit(self):
        limit = _REAL_METAL_LIMIT()
        self.assertTrue(limit is None or limit > 1_000_000_000)

    def test_bad_flag_values_raise(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "lots"}):
            with self.assertRaises(ValueError):
                prefill_ring.configured_experts()


class OwnershipTests(RingCase):
    def test_session_returns_every_slot_on_normal_exit(self):
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertLess(len(self.pool._free), len(self.free_before))
            ring.stage(self.store, prefix_of(1), [0, 1, 2])
        self.assertRestored()

    def test_exception_in_the_prefill_still_returns_the_ring(self):
        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with prefill_ring.session(self.language, 1024) as ring:
                ring.stage(self.store, prefix_of(1), [0, 1, 2])
                raise Boom()
        self.assertRestored()

    def test_exception_waits_for_the_gpu_before_release(self):
        calls = []
        with self.assertRaises(RuntimeError):
            with prefill_ring.session(self.language, 1024) as ring:
                ring._synchronize = lambda: calls.append("sync")
                ring.stage(self.store, prefix_of(1), [0, 1, 2])
                raise RuntimeError("fail")
        self.assertEqual(calls, ["sync"])
        self.assertRestored()

    def test_cancellation_during_a_read_releases_everything(self):
        with self.assertRaises(KeyboardInterrupt):
            with prefill_ring.session(self.language, 1024) as ring:
                ring._resolve = mock.Mock(side_effect=KeyboardInterrupt())
                ring.stage(self.store, prefix_of(1), [0, 1, 2, 3])
        self.assertRestored()

    def test_pool_is_usable_for_decode_after_the_session(self):
        with prefill_ring.session(self.language, 1024) as ring:
            ring.stage(self.store, prefix_of(1), [0, 1, 2])
        slots, misses = self.pool.assign(1, [0, 1, 2, 3])
        self.assertEqual(len(misses), 4)
        self.assertEqual(len(set(slots.values())), 4)

    def test_staged_slots_are_evicted_from_the_decode_cache(self):
        slots, _ = self.pool.assign(1, list(range(10)))
        top = max(slots.values())
        with prefill_ring.session(self.language, 1024) as ring:
            leased_first = ring.ring.first_slot
            for key, slot in self.pool._slot_of.items():
                self.assertLess(slot, leased_first, key)
        self.assertTrue(top >= 0)

    def test_second_session_after_the_first_works(self):
        for _ in range(2):
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [4, 5]))
            self.assertRestored()

    def test_released_half_cannot_be_written(self):
        with prefill_ring.session(self.language, 1024) as ring:
            lease = ring.ring.begin_fill()
            lease.release()
            with self.assertRaises(HazardError):
                with lease.writing():
                    pass


class ScratchBase(RingCase):
    """A first turn: no decode pool exists yet, as in a fresh process."""

    create_main_pool = False
    extra_env = {"FLASHNEXT_PREFILL_RING_SCRATCH": "1"}

    def stride(self):
        return expert_pool.get_slab_layout(32).record_stride

    def scratch_bytes(self, experts=8):
        stride = self.stride()
        slots = experts + 16                      # the half plus 16 unused slots
        return prefill_ring.pool_length(expert_pool.HEADER + slots * stride, stride)

    def stable_active(self, seconds=10.0):
        """Active memory once MLX has finished freeing earlier tests' arrays:
        three reads in a row within 1 MB."""
        deadline = time.time() + seconds
        reads = []
        while time.time() < deadline:
            gc.collect()
            prefill_ring.fence()
            reads.append(mx.get_active_memory())
            reads = reads[-3:]
            if len(reads) == 3 and max(reads) - min(reads) < 1_000_000:
                return reads[-1]
            time.sleep(0.1)
        self.fail("active memory did not settle")

    def settle(self, ceiling, tolerance=16_000_000, seconds=5.0):
        """True once active memory is back at ``ceiling``. MLX frees buffers
        asynchronously, so poll instead of reading once."""
        deadline = time.time() + seconds
        while True:
            gc.collect()
            prefill_ring.fence()
            if mx.get_active_memory() <= ceiling + tolerance:
                return True
            if time.time() > deadline:
                return False
            time.sleep(0.05)

    def no_decode_pool(self):
        return getattr(self.store, "_flashnext_expert_pool", None) is None


class FirstTurnScratchTests(ScratchBase):
    def test_a_first_turn_uses_a_private_pool_and_never_creates_the_decode_pool(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        baseline = self.stable_active()
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            self.assertTrue(ring.owns_pool)
            self.assertTrue(self.no_decode_pool())
            produced = self.run_layers(x, indices)
            private = len(ring.pool._mm)
            temporary = ring.pool
            self.assertTrue(temporary._fds)            # reads opened descriptors
            during = mx.get_active_memory() - baseline
            stats = ring.stats()
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))
        self.assertEqual(stats["staged_layers"], len(LAYERS))
        self.assertTrue(stats["scratch_pool"])
        # Sized for the ring alone: the half plus 16 slots, not the configured pool.
        self.assertEqual(private, self.scratch_bytes())
        self.assertLess(private, 0.6 * expert_pool.configured_gb() * 1e9)
        self.assertGreaterEqual(during, 0.98 * private)
        self.assertLess(during, private + 32_000_000)
        self.assertTrue(self.no_decode_pool())
        # Nothing of the temporary pool is left holding the memory or a file.
        self.assertEqual(temporary._fds, {})
        self.assertIsNone(temporary.buffer_mx)
        self.assertIsNone(temporary.array)
        self.assertIsNone(temporary._view)
        self.assertIsNone(ring.pool)
        self.assertIsNone(ring.ring.pool)
        del produced
        self.assertTrue(self.settle(baseline), "temporary pool memory was not returned")

    def test_the_temporary_pool_is_gone_before_the_decode_pool_is_created(self):
        x, indices = self.inputs()
        self.run_layers(x, indices)
        baseline = self.stable_active()
        with prefill_ring.session(self.language, 1024) as ring:
            self.run_layers(x, indices)
            state = ring
        self.assertTrue(state.disposed)
        self.assertTrue(self.settle(baseline))
        main = expert_pool.for_store(self.store, 32)        # what the first decode does
        self.addCleanup(main.close)
        grown = mx.get_active_memory() - baseline
        # Only the decode pool: a leftover temporary pool would add 74 MB.
        self.assertGreaterEqual(grown, 0.98 * len(main._mm))
        self.assertLess(grown, len(main._mm) + 16_000_000)

    def test_later_turns_put_the_ring_in_the_decode_pool(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        with prefill_ring.session(self.language, 1024) as first:
            self.assertTrue(first.owns_pool)
            self.run_layers(x, indices)
        main = expert_pool.for_store(self.store, 32)
        self.addCleanup(main.close)
        free_before = sorted(main._free)
        with prefill_ring.session(self.language, 1024) as second:
            self.assertIsNotNone(second, prefill_ring.LAST_REASON[0])
            self.assertFalse(second.owns_pool)
            self.assertIs(second.pool, main)
            produced = self.run_layers(x, indices)
        for got, want in zip(produced, reference):
            self.assertTrue(bool(mx.array_equal(got, want)))
        self.assertFalse(second.disposed)
        self.assertEqual(sorted(main._free), free_before)
        self.assertIsNone(expert_cache._PREFILL_RING[0])

    def test_two_first_turn_prefills_in_a_row_each_get_and_free_their_own_pool(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        baseline = self.stable_active()
        for _ in range(2):
            with prefill_ring.session(self.language, 1024) as ring:
                produced = self.run_layers(x, indices)
            for got, want in zip(produced, reference):
                self.assertTrue(bool(mx.array_equal(got, want)))
            self.assertTrue(ring.disposed)
            del produced
            self.assertTrue(self.settle(baseline))
        self.assertTrue(self.no_decode_pool())

    def test_without_the_flag_a_first_turn_behaves_as_before(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_SCRATCH": "0"}):
            with prefill_ring.session(self.language, 1024) as ring:
                self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
                self.assertFalse(ring.owns_pool)
                pool = getattr(self.store, "_flashnext_expert_pool", None)
                self.assertIsNotNone(pool)             # the ring created the pool
                self.assertIs(ring.pool, pool)
        self.addCleanup(pool.close)
        self.assertFalse(ring.disposed)

    def test_the_flag_is_parsed_strictly(self):
        for value, expected in (("0", False), ("1", True)):
            with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_SCRATCH": value}):
                self.assertEqual(prefill_ring.scratch_enabled(), expected)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_SCRATCH": "yes"}):
            with self.assertRaises(ValueError):
                prefill_ring.scratch_enabled()
        os.environ.pop("FLASHNEXT_PREFILL_RING_SCRATCH")
        self.assertFalse(prefill_ring.scratch_enabled())

    def test_pool_length_matches_what_a_real_pool_maps(self):
        stride = self.stride()
        for slots in (16, 24, 33):
            size = expert_pool.HEADER + slots * stride + 12345   # a ragged size
            pool = expert_pool.ExpertPool(self.store, size, 32)
            try:
                self.assertEqual(len(pool._mm), prefill_ring.pool_length(size, stride))
            finally:
                pool.close()


class ScratchMemoryPressureTests(ScratchBase):
    def accounting(self, active, cache, limit):
        return (
            mock.patch.object(mx, "get_active_memory", return_value=active),
            mock.patch.object(mx, "get_cache_memory", return_value=cache),
            mock.patch.object(prefill_ring, "metal_working_set_limit", return_value=limit),
        )

    def test_refused_before_anything_is_allocated(self):
        active = 1_000_000_000
        room = 100_000_000 + 500_000_000                      # transient + margin
        exact = active + self.scratch_bytes() + room
        a, c, l = self.accounting(active, 0, exact - 1)
        with a, c, l, mock.patch.object(expert_pool, "ExpertPool") as made:
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
            made.assert_not_called()
        self.assertTrue(
            prefill_ring.LAST_REASON[0].startswith("working set: MLX active"),
            prefill_ring.LAST_REASON[0])
        self.assertIn("scratch pool", prefill_ring.LAST_REASON[0])
        self.assertTrue(self.no_decode_pool())
        # One byte more room and it is accepted, so the pool is counted once.
        a, c, l = self.accounting(active, 0, exact)
        with a, c, l:
            state = prefill_ring.acquire(self.language, 1024)
            self.assertIsNotNone(state, prefill_ring.LAST_REASON[0])
            prefill_ring.release(state)

    def test_the_allocator_cache_counts_once_beside_the_new_pool(self):
        active, cache = 1_000_000_000, 200_000_000
        exact = active + cache + self.scratch_bytes() + 600_000_000
        a, c, l = self.accounting(active, cache, exact - 1)
        with a, c, l:
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
        a, c, l = self.accounting(active, cache, exact)
        with a, c, l:
            state = prefill_ring.acquire(self.language, 1024)
            self.assertIsNotNone(state, prefill_ring.LAST_REASON[0])
            prefill_ring.release(state)

    def test_without_scratch_a_first_turn_is_refused_before_the_decode_pool_exists(self):
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_SCRATCH": "0"}):
            a, c, l = self.accounting(1_000_000_000, 0, 1_500_000_000)
            with a, c, l, mock.patch.object(expert_pool, "for_store") as made:
                self.assertIsNone(prefill_ring.acquire(self.language, 1024))
                made.assert_not_called()
            self.assertIn("pool created for the ring", prefill_ring.LAST_REASON[0])
            self.assertTrue(self.no_decode_pool())

    def test_pressure_during_the_prefill_disables_the_ring_and_still_frees_the_pool(self):
        baseline = self.stable_active()
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring.stage(self.store, prefix_of(1), [0, 1]))
            ring._active_memory = lambda: ring._limit            # no room left
            self.assertIsNone(ring.stage(self.store, prefix_of(2), [0, 1]))
            self.assertEqual(ring.disabled, "working set")
        self.assertTrue(ring.disposed)
        self.assertTrue(self.settle(baseline))

    def test_a_pool_too_small_for_the_layout_frees_what_it_made(self):
        baseline = self.stable_active()
        with mock.patch.object(prefill_ring, "plan_ring", return_value=0):
            self.assertIsNone(prefill_ring.acquire(self.language, 1024))
        self.assertEqual(prefill_ring.LAST_REASON[0], "pool too small")
        self.assertIsNone(expert_cache._PREFILL_RING[0])
        self.assertTrue(self.settle(baseline))


class ScratchCleanupTests(ScratchBase):
    def test_an_exception_frees_the_temporary_pool(self):
        baseline = self.stable_active()
        with self.assertRaises(RuntimeError):
            with prefill_ring.session(self.language, 1024) as ring:
                state = ring
                ring.stage(self.store, prefix_of(1), [0, 1, 2])
                raise RuntimeError("fail")
        self.assertTrue(state.disposed)
        self.assertIsNone(expert_cache._PREFILL_RING[0])
        self.assertTrue(self.no_decode_pool())
        self.assertTrue(self.settle(baseline))

    def test_cancellation_during_a_read_frees_the_temporary_pool(self):
        baseline = self.stable_active()
        with self.assertRaises(KeyboardInterrupt):
            with prefill_ring.session(self.language, 1024) as ring:
                state = ring
                ring._resolve = mock.Mock(side_effect=KeyboardInterrupt())
                ring.stage(self.store, prefix_of(1), [0, 1, 2, 3])
        self.assertTrue(state.disposed)
        self.assertIsNone(expert_cache._PREFILL_RING[0])
        self.assertTrue(self.settle(baseline))

    def test_releasing_twice_is_safe(self):
        with prefill_ring.session(self.language, 1024) as ring:
            state = ring
            ring.stage(self.store, prefix_of(1), [0, 1])
        self.assertTrue(state.disposed)
        prefill_ring.release(state)                    # a second release
        prefill_ring.release(state, failed=True)       # and an abort after it
        self.assertFalse(state.dispose())              # nothing left to free

    def test_a_ring_that_will_not_close_keeps_its_pool_alive(self):
        with prefill_ring.session(self.language, 1024) as ring:
            state = ring
            ring.stage(self.store, prefix_of(1), [0, 1])
            real_close = ring.ring.close
            ring.ring.close = lambda timeout=None: False
        try:
            self.assertFalse(state.disposed)
            self.assertIn(state, prefill_ring._RETAINED)
            self.assertIn("retained", prefill_ring.LAST_REASON[0])
            self.assertIsNotNone(state.pool)           # not freed under a writer
            self.assertIsNone(expert_cache._PREFILL_RING[0])
        finally:
            state.ring.close = real_close
            prefill_ring._RETAINED.remove(state)
            self.assertTrue(state.close())
            self.assertTrue(state.dispose())


class ScratchGpuSafetyTests(ScratchBase):
    def test_failed_fence_retains_scratch_pool(self):
        with self.assertRaisesRegex(RuntimeError, "fence failed"):
            with prefill_ring.session(self.language, 1024) as state:
                state._synchronize = mock.Mock(side_effect=RuntimeError("fence failed"))
        self.assertFalse(state.disposed)
        self.assertIn(state, prefill_ring._RETAINED)
        from models.flashnext import pool_lifecycle
        self.assertIn("fence failed", pool_lifecycle.lifecycle_fault())
        pool_lifecycle._LIFECYCLE_FAULT[0] = None
        state._synchronize = lambda: None
        prefill_ring._RETAINED.remove(state)
        self.assertTrue(state.dispose())

    def test_a_release_with_gpu_work_in_flight_never_loses_the_data(self):
        x, indices = self.inputs()
        reference = self.run_layers(x, indices)
        for _ in range(3):
            with prefill_ring.session(self.language, 1024):
                lazy = self.layers[0](x, indices)      # stages into the private pool
                mx.async_eval(lazy)                    # the GPU is reading it now
            # The session ended at once: the pool was freed behind a fence.
            self.assertTrue(bool(mx.array_equal(lazy, reference[0])))
            del lazy

    def test_the_gpu_is_fenced_before_the_temporary_pool_is_freed(self):
        log = []
        with prefill_ring.session(self.language, 1024) as ring:
            ring._synchronize = lambda: log.append("fence")
            real_close = ring.pool.close
            ring.pool.close = lambda: (log.append("free"), real_close())[1]
            ring.stage(self.store, prefix_of(1), [0, 1])
        self.assertIn("free", log)
        at = log.index("free")
        self.assertGreaterEqual(at, 1)
        self.assertEqual(log[at - 1], "fence")         # the last thing before the free
        self.assertEqual(log.count("free"), 1)

    def test_a_failed_prefill_fences_twice_before_the_free(self):
        log = []
        with self.assertRaises(RuntimeError):
            with prefill_ring.session(self.language, 1024) as ring:
                ring._synchronize = lambda: log.append("fence")
                real_close = ring.pool.close
                ring.pool.close = lambda: (log.append("free"), real_close())[1]
                ring.stage(self.store, prefix_of(1), [0, 1])
                raise RuntimeError("fail")
        self.assertEqual(log, ["fence", "fence", "free"])   # abort, then dispose


class AccountingTests(RingCase):
    """Memory is counted once: the pool inside active memory, the cache beside it."""

    def test_the_cache_and_the_pool_each_count_once_at_acquire(self):
        pool_bytes = len(self.pool._mm)
        active, cache = pool_bytes + 100_000_000, 200_000_000
        exact = active + cache + 600_000_000
        for limit, accepted in ((exact - 1, False), (exact, True)):
            with mock.patch.object(mx, "get_active_memory", return_value=active), \
                    mock.patch.object(mx, "get_cache_memory", return_value=cache), \
                    mock.patch.object(
                        prefill_ring, "metal_working_set_limit", return_value=limit):
                state = prefill_ring.acquire(self.language, 1024)
                self.assertEqual(state is not None, accepted, prefill_ring.LAST_REASON[0])
                prefill_ring.release(state)

    def test_the_per_layer_guard_counts_the_cache_and_the_pool_once(self):
        pool_bytes = len(self.pool._mm)
        with prefill_ring.session(self.language, 1024) as ring:
            ring._active_memory = lambda: pool_bytes + 10
            ring._cache_memory = lambda: 20
            ring._limit = pool_bytes + 30 + ring._margin
            self.assertFalse(ring._over_limit())        # exactly at the limit
            ring._limit -= 1
            self.assertTrue(ring._over_limit())         # one byte over
            ring._cache_memory = None                   # no cache reader: ignored
            self.assertFalse(ring._over_limit())

    def test_a_cache_alone_can_push_the_ring_over(self):
        with prefill_ring.session(self.language, 1024) as ring:
            ring._active_memory = lambda: 0
            ring._cache_memory = lambda: ring._limit
            self.assertTrue(ring._over_limit())


class PrefillHookTests(RingCase):
    def test_flag_off_calls_the_existing_function_untouched(self):
        os.environ["FLASHNEXT_PREFILL_RING_EXPERTS"] = "0"
        with mock.patch.object(prefill, "_prefill_target", return_value="done") as inner:
            with mock.patch.object(prefill_ring, "session") as session:
                result = prefill.prefill_target(
                    self.language, mx.zeros((1, 4), dtype=mx.uint32), [], True, False, None
                )
        self.assertEqual(result, "done")
        inner.assert_called_once()
        session.assert_not_called()

    def test_flag_on_wraps_the_prefill_and_flushes_before_release(self):
        order = []

        class FakeSession:
            def __enter__(self_inner):
                order.append("enter")
                return object()

            def __exit__(self_inner, *args):
                order.append("exit")
                return False

        def inner(*args):
            order.append("prefill")
            return (None, None, mx.array([5], dtype=mx.uint32))

        with mock.patch.object(prefill, "_prefill_target", side_effect=inner):
            with mock.patch.object(prefill_ring, "session", return_value=FakeSession()):
                with mock.patch.object(
                    prefill_ring, "fence", side_effect=lambda: order.append("sync")
                ):
                    prefill.prefill_target(
                        self.language, mx.zeros((1, 4), dtype=mx.uint32), []
                    )
        self.assertEqual(order, ["enter", "prefill", "sync", "exit"])

    def test_flag_on_without_a_ring_does_not_flush(self):
        class NoRing:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *args):
                return False

        with mock.patch.object(prefill, "_prefill_target", return_value=(None, None, 1)):
            with mock.patch.object(prefill_ring, "session", return_value=NoRing()):
                with mock.patch.object(prefill_ring, "fence") as sync:
                    prefill.prefill_target(
                        self.language, mx.zeros((1, 4), dtype=mx.uint32), []
                    )
        sync.assert_not_called()

    def test_prompt_failure_inside_the_hook_still_releases_the_ring(self):
        with mock.patch.object(prefill, "_prefill_target", side_effect=ValueError("x")):
            with self.assertRaises(ValueError):
                prefill.prefill_target(
                    self.language, mx.zeros((1, 2000), dtype=mx.uint32), []
                )
        self.assertRestored()


if __name__ == "__main__":
    unittest.main()
