"""The numerical diagnostics themselves: recorders do not change inference, and the comparisons separate
ordinary floating-point drift from a state hand-off defect (small random model, no checkpoint)."""
from __future__ import annotations

import unittest
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from models.flashnext import prefill
from models.flashnext.tests.bench import numeric_diagnostics as N
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits


def to_bf16(model):
    model.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, model.parameters()))
    mx.eval(model.parameters())
    return model


def healthy_norms(model):
    """Make every RMSNorm gain about 1 under whichever convention is active. patch_rmsnorm.apply() is global and
    other test modules call it: under its one-centered convention the tiny model's zero-initialised norm weights
    mean a gain of zero, flat logits, and diagnostics with nothing to measure, so these tests would depend on
    the order the modules run in. (Unpatched, a zero weight is a gain of one, the model these tests were
    calibrated on.)"""
    norms = [m for _, m in model.named_modules() if "norm" in type(m).__name__.lower() and hasattr(m, "weight")]
    probe = norms[0]
    saved = probe.weight
    probe.weight = mx.zeros_like(saved)
    one_centered = float(mx.abs(probe(mx.ones((1, saved.shape[-1]), dtype=saved.dtype))).max()) == 0.0
    probe.weight = saved
    if one_centered:                                  # a gain of exactly one, as an unpatched zero weight gives
        fixed = [(name, mx.ones_like(value)) for name, value in tree_flatten(model.parameters())
                 if value.ndim == 1 and name.endswith("weight") and "norm" in name]
        model.load_weights(fixed, strict=False)
        mx.eval(model.parameters())
    return model


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(5)
        cls.ids = [int(t) for t in mx.random.randint(2, VOCAB, (700,)).tolist()]
        cls.fp32 = healthy_norms(build_model())
        cls.bf16 = to_bf16(healthy_norms(build_model()))
        cls.whole32 = N.execute(cls.fp32, cls.ids, chunk_tokens=0, decode_tokens=8, label="whole")
        cls.chunked32 = N.execute(cls.fp32, cls.ids, chunk_tokens=128, policy="balanced", anchors=(300,),
                                  decode_tokens=8, label="chunked")
        cls.whole16 = N.execute(cls.bf16, cls.ids, chunk_tokens=0, decode_tokens=8, label="whole16")
        cls.chunked16 = N.execute(cls.bf16, cls.ids, chunk_tokens=128, policy="balanced", anchors=(300,),
                                  decode_tokens=8, label="chunked16")


class RecorderTests(Base):
    def test_margin_sampler_records_the_top_two_and_samples_like_the_parent(self):
        rng = np.random.default_rng(0)
        row = mx.array(rng.normal(size=(1, 500)).astype(np.float32))
        sink = []
        token = N.MarginSampler(N.Sampling.greedy_settings(), sink)(row)
        order = np.argsort(-np.array(row).reshape(-1))
        self.assertEqual(int(token.item()), int(order[0]))
        self.assertEqual(sink[0][:2], (int(order[0]), int(order[1])))
        self.assertAlmostEqual(sink[0][2], float(np.array(row).reshape(-1)[order[0]]), places=6)
        self.assertGreaterEqual(sink[0][2], sink[0][3])

    def test_margin_sampler_context_replaces_and_restores_a_module_sampler(self):
        class Holder:
            Sampler = object

        sink = []
        with N.margin_sampler(Holder, sink):
            self.assertTrue(issubclass(Holder.Sampler, N.MarginSampler))
            Holder.Sampler(N.Sampling.greedy_settings())(mx.zeros((1, 10)))
        self.assertIs(Holder.Sampler, object)
        self.assertEqual(len(sink), 1)

    def test_the_routing_tap_returns_the_same_logits_and_is_removed_afterwards(self):
        before = [(n, tuple(v.shape)) for n, v in tree_flatten(self.fp32.parameters())]
        originals = [layer.mlp["gate"] for layer in self.fp32.layers]
        recorder = N.RoutingRecorder()
        x = mx.random.normal((4, 128))
        expected = originals[0](x)
        with recorder.install(self.fp32):
            self.assertIsNot(self.fp32.layers[0].mlp["gate"], originals[0])
            tapped = self.fp32.layers[0].mlp.gate(x)
        self.assertTrue(bool(mx.array_equal(tapped, expected).item()))
        self.assertTrue(all(layer.mlp["gate"] is original for layer, original in zip(self.fp32.layers, originals)))
        self.assertEqual([(n, tuple(v.shape)) for n, v in tree_flatten(self.fp32.parameters())], before,
                         "every parameter is back after the context (do not save or load while it is installed)")

    def test_recording_does_not_change_the_result(self):
        plain = N.execute(self.bf16, self.ids, chunk_tokens=128, policy="balanced", anchors=(300,),
                          decode_tokens=8, record_routing=False)
        self.assertEqual(plain.tokens, self.chunked16.tokens)
        for name in plain.state:
            self.assertTrue(np.array_equal(plain.state[name], self.chunked16.state[name]), name)

    def test_routing_has_one_row_per_prompt_token_and_per_decoded_token(self):
        recorder = self.chunked16.routing
        indices, probs = recorder.get("prefill", 0)
        self.assertEqual(indices.shape, (len(self.ids), 3))                # top_k = 2, plus one
        self.assertTrue(np.all(np.diff(probs, axis=1) <= 1e-7), "probabilities are sorted descending")
        self.assertEqual(recorder.get("decode", 0)[0].shape[0], 8)


class NoiseFloorTests(unittest.TestCase):
    def ex(self, keys_ref, keys_other, spans):
        make = lambda k: N.Execution("x", 0, spans, {"l0.keys": k}, (0, 0, 0, 0), [], [], [])
        return make(keys_ref), make(keys_other)

    def test_precision_is_read_from_the_low_bits(self):
        rng = np.random.default_rng(0)
        fp32 = rng.standard_normal((1, 1, 64, 4)).astype(np.float32)
        bf16 = np.array(mx.array(fp32).astype(mx.bfloat16).astype(mx.float32))
        self.assertEqual(N.noise_floor(fp32), N.FP32_NOISE)
        self.assertEqual(N.noise_floor(bf16), N.BF16_NOISE)

    def test_an_exactly_matching_prefix_does_not_make_rounding_at_the_edge_a_spike(self):
        rng = np.random.default_rng(1)
        ref = np.array(mx.array(rng.standard_normal((1, 1, 64, 8)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
        other = ref.copy()
        other[..., 32:40, :] *= 1.004                     # rounding-sized change at the edge, nothing elsewhere
        a, b = self.ex(ref, other, [32, 32])
        self.assertLess(N.boundary_spikes(a, b, window=8)["worst"]["ratio"], 10.0)
        broken = ref.copy()
        broken[..., 32:40, :] = 0.0                        # a real defect at the edge is still far above it
        a, c = self.ex(ref, broken, [32, 32])
        self.assertGreater(N.boundary_spikes(a, c, window=8)["worst"]["ratio"], 100.0)


class ComparisonTests(Base):
    def test_the_same_plan_twice_is_bit_identical_everywhere(self):
        again = N.execute(self.bf16, self.ids, chunk_tokens=128, policy="balanced", anchors=(300,),
                          decode_tokens=8, label="again")
        report = N.compare(self.chunked16, again)
        summary = N.summary(report)
        self.assertTrue(summary["bit_identical"])
        self.assertEqual(summary["routing_flip_fraction"], 0.0)
        self.assertEqual(report["routing"]["flips"], 0)
        self.assertEqual(report["boundaries"]["worst"]["ratio"], 0.0)

    def test_chunk_spans_are_recorded_as_planned(self):
        self.assertEqual(self.whole32.chunk_spans, [700])
        self.assertEqual(self.chunked32.chunk_spans, [100, 100, 100, 100, 100, 100, 100])

    def test_fp32_chunking_is_numerically_negligible(self):
        summary = N.summary(N.compare(self.whole32, self.chunked32))
        self.assertTrue(summary["positions_exact"])
        self.assertLess(max(summary["state_worst_rel_l2_by_kind"].values()), 1e-5)
        self.assertEqual(summary["routing_flip_fraction"], 0.0)
        self.assertTrue(summary["tokens_identical"])
        self.assertTrue(summary["logits_decision_safe"])
        self.assertEqual(summary["shape_mismatches"], [])

    def test_bf16_chunking_drifts_smoothly_and_its_routing_flips_are_near_ties(self):
        report = N.compare(self.whole16, self.chunked16)
        summary = N.summary(report)
        self.assertFalse(summary["bit_identical"], "the point: chunking is not bit-identical in bf16")
        self.assertTrue(summary["positions_exact"])
        worst = max(summary["state_worst_rel_l2_by_kind"].values())
        self.assertGreater(worst, 1e-4)
        self.assertLess(worst, 0.2)
        self.assertLess(report["routing"]["flip_fraction"], 0.1)
        if report["routing"]["flips"]:
            self.assertTrue(summary["flips_are_near_ties"], "a changed expert set must have been a near tie")
        self.assertLess(summary["max_boundary_spike_ratio"], 10.0, "drift is smooth across chunk edges")
        generation = report["generation"]
        if not generation["identical"]:                     # a divergence must be explained by a near tie
            self.assertTrue(generation["diverged_to_second_choice"])
            self.assertLess(generation["ref_margin_at_divergence"], 0.5)

    def broken_run(self, model):
        real = prefill._prefill_chunk
        calls = []

        def zero_conv_after_third_chunk(language, ids, cache):
            real(language, ids, cache)
            calls.append(1)
            if len(calls) == 3:                             # the chunk that ends at the anchor (300)
                for entry in cache:
                    if type(entry).__name__ == "ArraysCache" and entry.cache[0] is not None:
                        entry.cache[0] = mx.zeros_like(entry.cache[0])
                mx.eval([entry.state for entry in cache])

        with mock.patch.object(prefill, "_prefill_chunk", zero_conv_after_third_chunk):
            return N.execute(model, self.ids, chunk_tokens=128, policy="balanced", anchors=(300,),
                             decode_tokens=8, label="defect")

    def test_a_conv_state_defect_at_a_chunk_edge_is_caught_by_the_position_profile(self):
        defect = self.broken_run(self.bf16)
        report = N.compare(self.whole16, defect)
        summary = N.summary(report)
        drift = N.summary(N.compare(self.whole16, self.chunked16))
        self.assertGreater(summary["max_boundary_spike_ratio"], 20.0)
        self.assertGreater(summary["max_boundary_spike_ratio"], 10 * max(drift["max_boundary_spike_ratio"], 1.0))
        worst = report["boundaries"]["worst"]
        self.assertEqual(worst["edge"], 300)
        # The aggregate numbers alone would not separate them: this is why the profile is required.
        self.assertLess(max(summary["state_worst_rel_l2_by_kind"].values()),
                        5 * max(drift["state_worst_rel_l2_by_kind"].values()))

    def test_a_missing_kv_token_is_a_shape_mismatch_not_drift(self):
        broken = N.Execution(**{**self.chunked32.__dict__, "state": dict(self.chunked32.state)})
        name = next(n for n in broken.state if n.endswith(".keys"))
        broken.state[name] = broken.state[name][..., :-1, :]
        summary = N.summary(N.compare(self.whole32, broken))
        self.assertEqual(summary["shape_mismatches"], [name])
        self.assertFalse(summary["bit_identical"])

    def test_a_shifted_position_is_reported_as_inexact(self):
        broken = N.Execution(**{**self.chunked32.__dict__, "state": dict(self.chunked32.state)})
        name = next(n for n in broken.state if n.endswith(".index_position_ids"))
        broken.state[name] = broken.state[name] + 1
        summary = N.summary(N.compare(self.whole32, broken))
        self.assertFalse(summary["positions_exact"])

    def test_the_logit_decision_check_uses_the_margin(self):
        report = N.compare(self.whole16, self.chunked16)
        self.assertEqual(report["logits"]["decision_safe"],
                         report["logits"]["ref_margin"] > 2 * max(
                             report["logits"]["top1_logit_diff"], report["logits"]["top2_logit_diff"] or 0.0))


if __name__ == "__main__":
    unittest.main()
