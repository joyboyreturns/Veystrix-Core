import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from models.nemotron35 import admission, attention, state
from models.nemotron35.config import EXPECTED, LAYER_ORDER


class Int8AttentionTests(unittest.TestCase):
    def test_full_history_causality_accuracy_and_restore(self):
        mx.random.seed(91)
        q = mx.random.normal((1, 4, 14, 64)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, 14, 64)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, 14, 64)).astype(mx.bfloat16)
        scale = 64 ** -0.5
        with patch.object(attention, "BLOCK_TOKENS", 8), patch.object(attention, "QUERY_TOKENS", 4):
            cache = attention.AttentionCache()
            cache.append(k, v)
            out = attention.attend(q, cache, scale).astype(mx.float32)
            ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal").astype(mx.float32)
            self.assertLess(float(mx.max(mx.abs(out - ref)).item()), 0.025)
            self.assertEqual([block.offset for block in cache.blocks], [8, 6])
            recurrent = ArraysCache(size=2)
            recurrent.state = [mx.ones((1, 2, 64), mx.bfloat16), mx.ones((1, 2, 4, 64), mx.float32)]
            identity, tokens = {"model": "tiny-int8"}, list(range(14))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.safetensors"
                state.save(path, [recurrent, cache], identity, tokens)
                ssm, restored = state.restore(path, identity, tokens)
                self.assertEqual(ssm.state[1].dtype, mx.float32)
                self.assertTrue(mx.array_equal(ssm.state[1], recurrent.state[1]).item())
                self.assertEqual(restored.offset, 14)
                self.assertTrue(mx.array_equal(out, attention.attend(q, restored, scale).astype(mx.float32)).item())
                restored.append(k[..., :3, :], v[..., :3, :])
                self.assertEqual([block.offset for block in restored.blocks], [8, 8, 1])

    def test_admission_estimates_payload_and_keeps_swap_stop(self):
        gib = 2**30
        config = dict(EXPECTED, layers_block_type=LAYER_ORDER, max_position_embeddings=262144)
        args = dict(dense_bytes=gib, context_tokens=1_048_576, int8_kv=True, lightning=True)
        plan = admission.make_plan(config, admission.Machine(16*gib, 10*gib, 0, 1, 12*gib), **args)
        self.assertEqual(plan.context_tokens, 1_048_576)
        self.assertEqual(plan.estimate.attention_kv_bytes, 3_422_552_064)
        with self.assertRaises(admission.AdmissionRefused):
            admission.make_plan(config, admission.Machine(16*gib, 10*gib, 3*gib, 1, 12*gib), **args)
        with self.assertRaises(admission.AdmissionRefused):
            admission.make_plan(config, admission.Machine(16*gib, 2*gib, 0, 1, 12*gib), **args)
