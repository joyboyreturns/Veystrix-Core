"""Real MLX primitives, tiny random models; no checkpoint quality claims."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache

from models.flashnext import prefix_cache as P, qsa_chunk as Q, quantized_kv as K
from models.flashnext.mixed_kv import MixedPrecisionQSAKVCache, _unpack, _slice
from models.flashnext.tests.unit.test_prefix_cache import build_model


class GatherTests(unittest.TestCase):
    def test_native_context_selected_set_matches_reference_with_ties(self):
        spec = SimpleNamespace(block_topk=512, compress_ratio=4, head_dim=128)
        key_len, queries = 262143, 8
        blocks = key_len // spec.compress_ratio
        # Full checkpoint geometry with tied block scores, including the last
        # incomplete block. The reference scatter is only a test oracle.
        q = mx.zeros((1, 4, queries, 128), dtype=mx.bfloat16)
        pool = mx.zeros((1, 1, blocks, 128), dtype=mx.bfloat16)
        args = (spec, q, pool, key_len, key_len - queries, blocks, 0, queries)
        idx, valid = Q._selected_indices(*args)
        expected = Q._chunk_mask(*args)[:, 0]
        self.assertEqual(idx.shape, (1, 8, 2052))
        matched = mx.take_along_axis(expected, idx, axis=-1)
        self.assertTrue(mx.all(matched | ~valid).item())
        self.assertTrue(mx.array_equal(mx.sum(valid, axis=-1), mx.sum(expected, axis=-1)).item())

    def test_indices_match_original_mask_with_ties_tails_and_dense_transition(self):
        indexer = SimpleNamespace(block_topk=2, compress_ratio=4, head_dim=8)
        for tied in (True, False):
            for past in (0, 7, 8, 11, 16, 97):
                length, key_len = 19, past + 19
                blocks = key_len // 4
                query = mx.zeros((2, 2, length, 8)) if tied else mx.random.normal((2, 2, length, 8))
                pooled = mx.zeros((2, 1, blocks, 8)) if tied else mx.random.normal((2, 1, blocks, 8))
                for start in range(0, length, 3):
                    end = min(start + 3, length)
                    args = (indexer, query, pooled, key_len, past, blocks, start, end)
                    indices, valid = Q._selected_indices(*args)
                    mask = Q._chunk_mask(*args, scatter=False)[:, 0]
                    reconstructed = mx.any((mx.arange(key_len)[None, None, None, :] == indices[..., None])
                                           & valid[..., None], axis=2)
                    self.assertTrue(mx.array_equal(mask, reconstructed).item(), (past, start, tied))
                    self.assertLessEqual(indices.shape[-1], 12)

    def test_gather_attention_matches_dense_mask(self):
        mx.random.seed(17)
        for quantized in (False, True):
            k = mx.random.normal((2, 2, 41, 64))
            v = mx.random.normal((2, 2, 41, 64))
            q = mx.random.normal((2, 4, 3, 64))
            cache = QSAQuantizedKVCache(32, 4) if quantized else QSAKVCache()
            keys, values = cache.update_and_fetch(k, v)
            idx = mx.array([[[0, 9, 39, 40], [2, 11, 40, 40], [0, 15, 29, 40]]] * 2)
            valid = mx.array([[[True, True, True, False], [True, True, False, False], [True] * 4]] * 2)
            gk, gv = Q._gather_kv(keys, values, cache, idx)
            gq = q.transpose(0, 2, 1, 3).reshape(6, 4, 1, 64)
            actual = mx.fast.scaled_dot_product_attention(gq, gk, gv, scale=0.125,
                                                         mask=valid.reshape(6, 1, 1, 4))
            dense_mask = mx.any((mx.arange(41)[None, None, None, :] == idx[..., None]) & valid[..., None], axis=2)
            if quantized:
                k = mx.dequantize(*keys, group_size=32, bits=4)
                v = mx.dequantize(*values, group_size=32, bits=4)
            expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=dense_mask[:, None])
            actual = actual.reshape(2, 3, 4, 64).transpose(0, 2, 1, 3)
            self.assertTrue(mx.allclose(actual, expected, atol=2e-6, rtol=2e-5).item())

    def test_tiny_model_gather_logits_and_indexer_states(self):
        model = build_model()
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention
        original = Qwen4ExpAttention.__call__
        if original is Q._chunked_call:
            original = Q._ORIGINAL_CALL
        reference, candidate = model.make_cache(), model.make_cache()
        with patch.object(Q, "_ORIGINAL_CALL", original), patch.object(Qwen4ExpAttention, "__call__", Q._chunked_call):
            for length in (13, 59, 1, 7):
                ids = mx.random.randint(2, 512, (1, length))
                with patch.dict(os.environ, {"FLASHNEXT_QSA_KERNEL": "reference"}):
                    expected = model(ids, cache=reference).logits
                with patch.dict(os.environ, {"FLASHNEXT_QSA_KERNEL": "gather", "FLASHNEXT_QSA_GATHER_TILE": "3"}):
                    actual = model(ids, cache=candidate).logits
                mx.eval(expected, actual, [c.state for c in candidate], [c.state for c in reference])
                self.assertTrue(mx.allclose(actual, expected, atol=2e-4, rtol=2e-3).item(),
                                mx.max(mx.abs(actual - expected)).item())
                for r, c in zip(reference, candidate):
                    if isinstance(r, QSAKVCache):
                        self.assertTrue(mx.allclose(r.index_keys, c.index_keys, atol=2e-5).item())


class MixedKVTests(unittest.TestCase):
    def append(self, cache, length):
        k = mx.random.normal((1, 2, length, 64)).astype(mx.bfloat16)
        v = mx.random.normal(k.shape).astype(mx.bfloat16)
        start = cache.offset
        cache.update_indexer(mx.random.normal((1, length, 32)).astype(mx.bfloat16),
                             mx.arange(start, start + length)[None])
        cache.update_and_fetch(k, v)
        mx.eval(cache.state)
        return k, v

    def test_mixed_precision_selected_rows(self):
        for mode in ("k16v4", "k8v4"):
            for group in (32, 64):
                cache = MixedPrecisionQSAKVCache(mode, group)
                k, v = self.append(cache, 17)
                self.append(cache, 3)
                idx = mx.array([[[0, 5, 16], [1, 2, 3]]])
                actual_k, actual_v = cache.gather_selected(idx)
                expected_k = k if mode == "k16v4" else mx.dequantize(*mx.quantize(k, group_size=group, bits=8), group_size=group, bits=8)
                expected_v = mx.dequantize(*mx.quantize(v, group_size=group, bits=4), group_size=group, bits=4)
                self.assertTrue(mx.array_equal(actual_k, Q._gather_rows(expected_k, idx)).item())
                self.assertTrue(mx.array_equal(actual_v, Q._gather_rows(expected_v, idx)).item())

    def test_tier_aging_page_boundaries_and_history_immutability(self):
        cache = MixedPrecisionQSAKVCache("tiered4", 32, 256)
        original_keys, _ = self.append(cache, 511)
        self.assertEqual(cache.history_tokens, 0)
        self.append(cache, 1)
        self.assertEqual(cache.history_tokens, 256)
        expected = mx.quantize(original_keys[..., :256, :], group_size=32, bits=4)
        for a, b in zip(expected, cache.keys):
            self.assertTrue(mx.array_equal(a, b[..., :256, :]).item())
        old = tuple(a[..., :256, :] for a in cache.keys)
        self.append(cache, 601)
        self.assertEqual(cache.history_tokens, 768)
        self.assertLess(cache.offset - cache.history_tokens, 512)
        for a, b in zip(old, cache.keys):
            self.assertTrue(mx.array_equal(a, b[..., :256, :]).item())
        indices = mx.array([[[0, 255, 256, 767, 768, 1112]]])
        actual, _ = cache.gather_selected(indices)
        dense = mx.concatenate((_unpack(_slice(cache.keys, 0, cache.history_tokens), 4, 32),
                                _unpack(_slice(cache.recent_keys, 0, cache.offset-cache.history_tokens), 8, 32)), axis=-2)
        self.assertTrue(mx.array_equal(actual, Q._gather_rows(dense, indices)).item())

    def test_ram_ssd_restore_and_same_plan_continuation(self):
        for mode in ("k16v4", "k8v4", "tiered4"):
            cache = MixedPrecisionQSAKVCache(mode, 32, 256)
            self.append(cache, 301)
            fp = P.fingerprint(test=mode)
            parent = P.capture([cache], range(301), fp, plan=(100,))
            parent._tokens = tuple(range(301))
            self.append(cache, 250)
            leaf = P.capture([cache], range(551), fp, parent=parent, plan=(100, 301))
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                P.write_entry(directory, parent, fp)
                P.write_entry(directory, leaf, fp)
                loaded = P.read_entry(directory, leaf.key, fp)
                restored = P.restore([parent, loaded], lambda: [MixedPrecisionQSAKVCache(mode, 32, 256)])[0]
                for name, a in cache.snapshot().items():
                    self.assertTrue(mx.array_equal(a, restored.snapshot()[name]).item(), name)
                # Append identical activations under identical chunk plan.
                mx.random.seed(19)
                self.append(cache, 73)
                mx.random.seed(19)
                self.append(restored, 73)
                for name, a in cache.snapshot().items():
                    self.assertTrue(mx.array_equal(a, restored.snapshot()[name]).item(), name)
            # Prior snapshot survives subsequent mutation and tier migrations.
            restored_parent = P.restore([parent], lambda: [MixedPrecisionQSAKVCache(mode, 32, 256)])[0]
            self.assertEqual(restored_parent.offset, 301)

    def test_restore_rejects_incompatible_or_corrupt_layout(self):
        cache = MixedPrecisionQSAKVCache("tiered4", 32, 256)
        self.append(cache, 513)
        snapshot = cache.snapshot()
        for other in (MixedPrecisionQSAKVCache("k8v4", 32, 256),
                      MixedPrecisionQSAKVCache("tiered4", 64, 256),
                      MixedPrecisionQSAKVCache("tiered4", 32, 512)):
            with self.assertRaises(ValueError):
                other.restore_snapshot(snapshot, 513)
        corrupt = {**snapshot, "keys_0": snapshot["keys_0"].astype(mx.float32)}
        with self.assertRaises(ValueError):
            cache.restore_snapshot(corrupt, 513)

    def test_factory_uses_requested_precision_and_fails_closed(self):
        model = SimpleNamespace(layers=[SimpleNamespace(is_linear=False)])
        with patch.dict(os.environ, {"FLASHNEXT_QSA_KERNEL": "gather"}):
            self.assertEqual(K.make_cache(model, "affine8")[0].bits, 8)
            self.assertEqual(K.make_cache(model, "k8v4")[0].key_bits, 8)
            with self.assertRaises(ValueError):
                K.make_cache(model, "unknown")
        with patch.dict(os.environ, {"FLASHNEXT_QSA_KERNEL": "reference"}):
            with self.assertRaisesRegex(ValueError, "requires"):
                K.make_cache(model, "tiered4")

    def test_per_layer_override_cannot_replace_recurrence(self):
        model = SimpleNamespace(layers=[SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)])
        env = {"FLASHNEXT_QSA_KERNEL": "gather", "FLASHNEXT_KV_CACHE_DTYPE": "bf16",
               "FLASHNEXT_KV_LAYER_MODES": '{"1":"k8v4"}'}
        with patch.dict(os.environ, env):
            layers = K.make_cache(model)
            self.assertEqual(type(layers[0]).__name__, "ArraysCache")
            self.assertEqual(layers[1].key_bits, 8)
            self.assertEqual(K.fingerprint_parts()["kv_layer_modes"], {"1": "k8v4"})
        with patch.dict(os.environ, {**env, "FLASHNEXT_KV_LAYER_MODES": '{"0":"k8v4"}'}):
            with self.assertRaisesRegex(ValueError, "QSA layer"):
                K.make_cache(model)

    def test_ssd_restart_preserves_full_model_continuation(self):
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention
        model = build_model()
        model.set_dtype(mx.bfloat16)
        original = Qwen4ExpAttention.__call__
        if original is Q._chunked_call:
            original = Q._ORIGINAL_CALL
        env = {"FLASHNEXT_QSA_KERNEL": "gather", "FLASHNEXT_KV_RECENT_TOKENS": "256"}
        with (tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env),
              patch.object(Q, "_ORIGINAL_CALL", original),
              patch.object(Qwen4ExpAttention, "__call__", Q._chunked_call)):
            def make_cache():
                return K.make_cache(model, "tiered4", 32)

            def forward(ids, caches):
                result = model(mx.array([ids]), cache=caches).logits
                mx.eval(result, [c.state for c in caches])
                return result

            caches = make_cache()
            fp = P.fingerprint(test="mixed", checkpoint="tiny")
            store = P.PrefixCache(fp, 0, Path(tmp), min_disk_free_bytes=0)
            ids = [2 + (i * 97) % 510 for i in range(523)]
            forward(ids[:301], caches)
            store.put(ids[:301], caches, language=model, plan=())
            forward(ids[301:515], caches)
            store.put(ids[:515], caches, language=model, plan=(301,))
            store._owner_finalizer()
            store = P.PrefixCache(fp, 0, Path(tmp), min_disk_free_bytes=0)
            restored = store.lookup(ids, make_cache, language=model,
                                    accept=lambda n, plan: n == 515 and plan == (301,))
            self.assertEqual(restored.outcome, P.OUTCOME_SSD_HIT)
            pos = P.capture_language(model)
            expected = forward(ids[515:], caches)
            P.apply_language(model, pos)
            actual = forward(ids[515:], restored.caches)
            self.assertTrue(mx.array_equal(expected, actual).item())
            for reference, candidate in zip(caches, restored.caches):
                if hasattr(reference, "snapshot"):
                    for key, value in reference.snapshot().items():
                        self.assertTrue(mx.array_equal(value, candidate.snapshot()[key]).item(), key)
                else:
                    for a, b in zip(reference.cache, candidate.cache):
                        self.assertTrue(mx.array_equal(a, b).item())
            store._owner_finalizer()


if __name__ == "__main__":
    unittest.main()
