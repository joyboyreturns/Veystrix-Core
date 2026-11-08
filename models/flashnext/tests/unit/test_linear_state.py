"""FLASHNEXT_EVAL_LINEAR_STATE: evaluate the GatedDeltaNet conv state per layer.

The fake-call tests need no MLX arrays. The layer tests build a small
GatedDeltaNet with random weights (no checkpoint) and check two things that
matter: values and state are bit-identical, and the layer's prefill inputs are
released once its state is evaluated.
"""
from __future__ import annotations

import gc
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import linear_state


class Shape:
    def __init__(self, tokens):
        self.shape = (1, tokens, 8)


class FakeCache(list):
    pass


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.saved = (dict(linear_state._ORIGINAL), linear_state.ENABLED[0], linear_state._APPLIED[0])

        def original(layer, inputs, mask=None, cache=None):
            self.calls.append(("layer", int(inputs.shape[1])))
            return "output"

        linear_state._ORIGINAL["call"] = original
        self.evaluated = []
        patcher = mock.patch.object(linear_state.mx, "eval", side_effect=lambda *a: self.evaluated.append(a))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.restore)

    def restore(self):
        linear_state._ORIGINAL.clear()
        linear_state._ORIGINAL.update(self.saved[0])
        linear_state.ENABLED[0], linear_state._APPLIED[0] = self.saved[1], self.saved[2]

    def run_call(self, tokens, cache):
        return linear_state.call(object(), Shape(tokens), mask=None, cache=cache)

    def test_off_never_evaluates(self):
        linear_state.set_enabled(False)
        self.assertEqual(self.run_call(1000, FakeCache(["state", "ssm"])), "output")
        self.assertEqual(self.evaluated, [])

    def test_on_evaluates_only_the_conv_state(self):
        linear_state.set_enabled(True)
        self.assertEqual(self.run_call(1000, FakeCache(["state", "ssm"])), "output")
        self.assertEqual(self.evaluated, [("state",)])

    def test_decode_and_short_calls_are_untouched(self):
        linear_state.set_enabled(True)
        for tokens in (1, 2, linear_state.MIN_TOKENS - 1):
            self.run_call(tokens, FakeCache(["state", "ssm"]))
        self.assertEqual(self.evaluated, [])
        self.run_call(linear_state.MIN_TOKENS, FakeCache(["state", "ssm"]))
        self.assertEqual(len(self.evaluated), 1)

    def test_no_cache_or_empty_state_is_safe(self):
        linear_state.set_enabled(True)
        self.run_call(1000, None)
        self.run_call(1000, FakeCache([None, "ssm"]))
        self.run_call(1000, FakeCache())
        self.assertEqual(self.evaluated, [])

    def test_output_is_whatever_the_layer_returned(self):
        linear_state.set_enabled(True)
        self.assertIs(self.run_call(1000, FakeCache(["s", "t"])), "output")


def small_config():
    from mlx_vlm.models.qwen4_exp.config import TextConfig

    return TextConfig(
        model_type="qwen4_exp_text", hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
        linear_num_value_heads=16, linear_num_key_heads=4, linear_key_head_dim=64,
        linear_value_head_dim=64, linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
        shared_expert_intermediate_size=32, moe_intermediate_size=32, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=4096,
    )


def settle() -> int:
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


class LayerTests(unittest.TestCase):
    TOKENS = 4096

    @classmethod
    def setUpClass(cls):
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet

        mx.random.seed(7)
        cls.config = small_config()
        cls.layer = Qwen4ExpGatedDeltaNet(cls.config)
        mx.eval(cls.layer.parameters())
        cls.conv_dim = 2 * 4 * 64 + 16 * 64

    def setUp(self):
        linear_state.apply()
        self.addCleanup(linear_state.restore)
        self.addCleanup(linear_state.set_enabled, linear_state.ENABLED[0])

    def run_layer(self, enabled: bool):
        from mlx_lm.models.cache import ArraysCache

        linear_state.set_enabled(enabled)
        cache = ArraysCache(size=2)
        x = mx.random.normal((1, self.TOKENS, self.config.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        base = settle()
        out = self.layer(x, mask=None, cache=cache)
        mx.eval(out)                      # what the streaming MoE does
        del out
        held = settle() - base
        mx.eval(cache[0], cache[1])
        return cache, held, x

    def test_state_and_output_are_bit_identical(self):
        mx.random.seed(11)
        x = mx.random.normal((1, self.TOKENS, self.config.hidden_size)).astype(mx.bfloat16)
        results = []
        for enabled in (False, True):
            from mlx_lm.models.cache import ArraysCache

            linear_state.set_enabled(enabled)
            cache = ArraysCache(size=2)
            out = self.layer(x, mask=None, cache=cache)
            mx.eval(out, cache[0], cache[1])
            results.append((out, cache[0], cache[1]))
        for off, on in zip(*results):
            self.assertEqual(off.dtype, on.dtype)
            self.assertTrue(mx.array_equal(off.view(mx.uint8), on.view(mx.uint8)).item())

    def test_prefill_inputs_are_released_when_enabled(self):
        _, held_off, _ = self.run_layer(False)
        _, held_on, _ = self.run_layer(True)
        conv_input = (self.TOKENS + 3) * self.conv_dim * 2
        self.assertGreater(held_off, 0.8 * conv_input, "the lazy state should hold the layer's inputs")
        self.assertLess(held_on, 0.1 * conv_input, "evaluating the state should release them")

    def test_decode_call_is_not_evaluated_early(self):
        from mlx_lm.models.cache import ArraysCache

        linear_state.set_enabled(True)
        cache = ArraysCache(size=2)
        x = mx.random.normal((1, 1, self.config.hidden_size)).astype(mx.bfloat16)
        with mock.patch.object(linear_state.mx, "eval") as evaluate:
            self.layer(x, mask=None, cache=cache)
        evaluate.assert_not_called()


class ApplyTests(unittest.TestCase):
    def test_apply_is_idempotent_and_restores(self):
        from mlx_vlm.models.qwen4_exp import language

        original = language.Qwen4ExpGatedDeltaNet.__call__
        self.addCleanup(setattr, language.Qwen4ExpGatedDeltaNet, "__call__", original)
        was = linear_state._APPLIED[0]
        linear_state.restore()
        try:
            self.assertTrue(linear_state.apply())
            self.assertFalse(linear_state.apply())
            self.assertIs(language.Qwen4ExpGatedDeltaNet.__call__, linear_state.call)
            linear_state.restore()
            self.assertIs(language.Qwen4ExpGatedDeltaNet.__call__, original)
        finally:
            if was:
                linear_state.apply()

    def test_flag_is_registered_and_default_off(self):
        from models.flashnext.settings.registry import get_registry

        keys = {s.env_key for s in get_registry().for_backend("flashnext") if s.env_key}
        self.assertIn(linear_state.FLAG, keys)

    def test_off_when_the_variable_is_unset(self):
        import importlib
        import os

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(linear_state.FLAG, None)
            module = importlib.reload(linear_state)
            self.addCleanup(importlib.reload, linear_state)
            self.assertFalse(module.ENABLED[0])


if __name__ == "__main__":
    unittest.main()
