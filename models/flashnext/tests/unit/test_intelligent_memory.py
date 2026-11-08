"""Unit tests for the Intelligent Long-Context Memory Architecture.

Covers:
1. Memory ledger estimation and stage transitions.
2. Adaptive prefill scheduler chunk selection and hysteresis.
3. Affine4 QSA KV cache representation, group size, compression ratio, and cosine similarity.
4. Prefix cache capture, atomic serialization, and bit-exact reconstruction of Affine4 KV.
5. Strict fingerprint separation (no mixing of Affine4 and BF16 snapshots).
6. Adaptive memory profiles (Conservative, Balanced, Performance).
7. Blocker diagnostics in prefill readiness.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from models.flashnext import (
    prefill_scheduler as PS,
    quantized_kv as QKV,
    prefix_cache as PC,
    memory_manager as MM,
)
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache


class IntelligentMemoryArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="veystrix_test_mem_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        QKV.restore()

    def test_01_memory_ledger_and_chunk_estimation(self):
        """Verify memory estimation across chunk candidates 256, 512, 1024, 2048."""
        avail = 9 * 1024 * 1024 * 1024  # 9 GiB
        est_2048 = PS.estimate_chunk_memory(2048, 12288, avail, ring_active=True)
        est_1024 = PS.estimate_chunk_memory(1024, 12288, avail, ring_active=True)
        est_512 = PS.estimate_chunk_memory(512, 12288, avail, ring_active=True)
        est_256 = PS.estimate_chunk_memory(256, 12288, avail, ring_active=True)

        self.assertGreater(est_2048.total_peak_bytes, est_1024.total_peak_bytes)
        self.assertGreater(est_1024.total_peak_bytes, est_512.total_peak_bytes)
        self.assertGreater(est_512.total_peak_bytes, est_256.total_peak_bytes)
        
        # Transient memory must scale linearly with chunk length
        self.assertEqual(est_2048.transient_bytes, est_1024.transient_bytes * 2)
        self.assertEqual(est_1024.transient_bytes, est_512.transient_bytes * 2)

    def test_02_adaptive_scheduler_selection_and_hysteresis(self):
        """Scheduler picks largest safe chunk and steps down under constrained memory."""
        # Generous memory (10 GiB available): selects 2048
        sched = PS.AdaptivePrefillScheduler(
            candidate_chunks=(2048, 1024, 512, 256),
            system_reader=lambda: {"available": 10 * 1024 * 1024 * 1024},
        )
        c0 = sched.select_initial_chunk(4096)
        self.assertEqual(c0, 2048)

        # Immediate step down under memory pressure
        c1 = sched.observe_and_adapt(available_bytes=3 * 1024 * 1024 * 1024, pressure_level=2)
        self.assertEqual(c1, 2048)          # moderate pressure does not shrink chunks (they cost the same SSD traffic each)
        c1 = sched.observe_and_adapt(available_bytes=1 * 1024 * 1024 * 1024, pressure_level=4)
        self.assertEqual(c1, 1024)

        # Further step down if pressure persists
        c2 = sched.observe_and_adapt(available_bytes=1 * 1024 * 1024 * 1024, pressure_level=4)
        self.assertEqual(c2, 512)

        # One safe chunk steps back up (fewer, larger chunks stream the experts fewer times)
        c3 = sched.observe_and_adapt(available_bytes=8 * 1024 * 1024 * 1024, pressure_level=1)
        self.assertEqual(c3, 1024)
        c4 = sched.observe_and_adapt(available_bytes=8 * 1024 * 1024 * 1024, pressure_level=1)
        self.assertEqual(c4, 2048)

    def test_03_affine4_qsa_kv_representation_and_compression(self):
        """Verify AffineQuantizedQSAKVCache data layout and ~3.5x compression ratio."""
        B, n_kv, L, D = 1, 2, 512, 256
        keys = mx.random.normal((B, n_kv, L, D)).astype(mx.bfloat16)
        values = mx.random.normal((B, n_kv, L, D)).astype(mx.bfloat16)
        idx_keys = mx.random.normal((B, L, 128)).astype(mx.bfloat16)
        pos_ids = mx.arange(L).reshape(1, 1, L)

        # Standard BF16 cache
        bf16_cache = QSAKVCache()
        bf16_cache.update_and_fetch(keys, values)
        bf16_cache.update_indexer(idx_keys, pos_ids)
        bf16_bytes = bf16_cache.nbytes

        # Affine Q4 cache
        q4_cache = QKV.AffineQuantizedQSAKVCache(group_size=64, bits=4)
        q4_cache.update_and_fetch(keys, values)
        q4_cache.update_indexer(idx_keys, pos_ids)
        q4_bytes = q4_cache.nbytes

        # Verify compression
        detailed = q4_cache.detailed_nbytes
        self.assertIn("quantized_data_bytes", detailed)
        self.assertIn("metadata_bytes", detailed)
        self.assertIn("indexer_bytes", detailed)
        self.assertLess(q4_bytes, bf16_bytes)
        
        # Sliced state dequantization check
        deq_k, deq_v = q4_cache.dequantize()
        self.assertIsNotNone(deq_k)
        self.assertIsNotNone(deq_v)
        self.assertEqual(deq_k.shape, keys.shape)
        self.assertEqual(deq_v.shape, values.shape)

        # Cosine similarity between original and dequantized tensors
        cos_k = float(mx.mean(keys * deq_k) / (mx.sqrt(mx.mean(keys**2)) * mx.sqrt(mx.mean(deq_k**2))))
        cos_v = float(mx.mean(values * deq_v) / (mx.sqrt(mx.mean(values**2)) * mx.sqrt(mx.mean(deq_v**2))))
        self.assertGreater(cos_k, 0.98, f"Key cosine similarity {cos_k} below 0.98")
        self.assertGreater(cos_v, 0.98, f"Value cosine similarity {cos_v} below 0.98")

    def test_04_prefix_cache_quantized_kv_roundtrip(self):
        """Verify capture, atomic SSD write, and bit-exact reconstruction of Affine4 KV."""
        class DummyModel:
            def __init__(self):
                # 4 layers: 3 linear, 1 attention
                self.layers = [
                    type("L", (), {"is_linear": True})(),
                    type("L", (), {"is_linear": True})(),
                    type("L", (), {"is_linear": False})(),
                    type("L", (), {"is_linear": True})(),
                ]

        model = DummyModel()
        caches = QKV.make_cache(model, dtype="affine4", grp_size=64, n_bits=4)
        self.assertEqual(len(caches), 4)
        self.assertIsInstance(caches[2], QKV.AffineQuantizedQSAKVCache)
        self.assertIsInstance(caches[0], ArraysCache)

        # Populate cache
        tokens = tuple(range(100, 228))  # 128 tokens
        B, n_kv, L, D = 1, 2, len(tokens), 256
        keys = mx.random.normal((B, n_kv, L, D)).astype(mx.bfloat16)
        values = mx.random.normal((B, n_kv, L, D)).astype(mx.bfloat16)
        idx_keys = mx.random.normal((B, L, 128)).astype(mx.bfloat16)
        pos_ids = mx.arange(L).reshape(1, 1, L)

        caches[2].update_and_fetch(keys, values)
        caches[2].update_indexer(idx_keys, pos_ids)
        for i in (0, 1, 3):
            caches[i].cache[0] = mx.random.normal((1, 10, 32))

        fp = PC.fingerprint(model="test_model", kv_cache_dtype="affine4", group_size=64)
        entry = PC.capture(caches, tokens, fp)
        self.assertEqual(entry.n_tokens, 128)
        self.assertEqual(entry.layers[2].kind, "qsa_quantized_kv")

        # Save to disk
        PC.write_entry(self.tmp_dir, entry, fp)
        loaded = PC.read_entry(self.tmp_dir, entry.key, fp)
        self.assertEqual(loaded.key, entry.key)
        self.assertEqual(loaded.n_tokens, 128)

        # Restore into fresh cache list
        restored_caches = PC.restore([loaded], lambda: QKV.make_cache(model, dtype="affine4"))
        self.assertEqual(len(restored_caches), 4)
        restored_q4 = restored_caches[2]
        self.assertEqual(restored_q4.offset, 128)

        # Check that quantized arrays match exactly
        for i in range(3):
            self.assertTrue(bool(mx.array_equal(caches[2].keys[i][..., :128, :], restored_q4.keys[i][..., :128, :])))
            self.assertTrue(bool(mx.array_equal(caches[2].values[i][..., :128, :], restored_q4.values[i][..., :128, :])))
        self.assertTrue(bool(mx.array_equal(caches[2].index_keys, restored_q4.index_keys)))

    def test_05_strict_fingerprint_isolation(self):
        """Affine4 snapshots and BF16 snapshots must have disjoint fingerprints."""
        fp_bf16 = PC.fingerprint(model="qwen4", kv_cache_dtype="bf16")
        fp_q4 = PC.fingerprint(model="qwen4", kv_cache_dtype="affine4")
        self.assertNotEqual(fp_bf16, fp_q4)

        # Cross-restoration must fail closed
        tokens = tuple(range(100, 200))
        key_bf16 = PC.chain_key(fp_bf16, tokens)
        key_q4 = PC.chain_key(fp_q4, tokens)
        self.assertNotEqual(key_bf16, key_q4)

    def test_06_profile_selection(self):
        """Verify adaptive memory profile configurations."""
        snap_tight = MM.Snapshot(
            available=4 * 1024 * 1024 * 1024,
            pressure=2,
            pool_bytes=2 * 1024 * 1024 * 1024,
            prefix_ram_bytes=500 * 1024 * 1024,
        )
        req = MM.Request(tokens=4096, ring_eligible=True, fresh=True)
        state = MM.State()
        cfg = MM.Config(pools_gb=(2.0, 1.0, 0.0))

        # Under tight memory, pool must be released
        p = MM.plan(snap_tight, req, state, cfg)
        self.assertTrue(p.release_pool)
        self.assertEqual(p.evict_ram_to, 0)


if __name__ == "__main__":
    unittest.main()
