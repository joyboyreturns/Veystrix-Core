"""Layer-major (windowed) prefill: planning, cost model, bitwise equivalence with the chunk-major order, expert reuse, safety.

Checkpoint-free. The model-level tests run a small Qwen4-Exp language model (all layer kinds: linear attention with PLE, sparse
attention) both ways and compare every cache array and the final hidden state byte for byte. The expert-level tests run real
``StreamingSwitchGLU`` layers (real expert row sizes, real store, real pool) through the window and through the existing path.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import expert_cache, layer_major, prefill_ring
from models.flashnext.tests.unit.test_prefill_ring import (
    HIDDEN, LAYERS, PROJECTIONS, RingCase, prefix_of,
)
from models.flashnext.tests.unit.test_prefix_cache import build_model, cache_bits


class FlagTests(unittest.TestCase):
    def test_default_is_chunk_major_and_has_no_identity(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(layer_major.FLAG_ORDER, None)
            self.assertEqual(layer_major.order(), "chunk")
            self.assertFalse(layer_major.enabled())
            self.assertEqual(layer_major.identity(), "")

    def test_layer_order_carries_a_versioned_identity(self):
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_WINDOW: "4096",
                                         layer_major.FLAG_HYBRID: "0"}):
            self.assertTrue(layer_major.enabled())
            self.assertEqual(layer_major.identity(), f"{layer_major.PLAN_VERSION}/w4096")
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "layer", layer_major.FLAG_WINDOW: "4096",
                                         layer_major.FLAG_HYBRID: "1"}):
            self.assertEqual(layer_major.identity(), f"{layer_major.PLAN_VERSION}/w4096/hybrid-small-v1")

    def test_inner_snapshot_flag_defaults_on_and_validates(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(layer_major.FLAG_INNER_SNAPSHOTS, None)
            self.assertTrue(layer_major.inner_snapshots_enabled())
        with mock.patch.dict(os.environ, {layer_major.FLAG_INNER_SNAPSHOTS: "0"}):
            self.assertFalse(layer_major.inner_snapshots_enabled())
        with mock.patch.dict(os.environ, {layer_major.FLAG_INNER_SNAPSHOTS: "x"}):
            with self.assertRaises(ValueError):
                layer_major.inner_snapshots_enabled()

    def test_bad_values_raise(self):
        with mock.patch.dict(os.environ, {layer_major.FLAG_ORDER: "wavefront"}):
            with self.assertRaises(ValueError):
                layer_major.order()
        with mock.patch.dict(os.environ, {layer_major.FLAG_WINDOW: "10"}):
            with self.assertRaises(ValueError):
                layer_major.window_tokens()
        with mock.patch.dict(os.environ, {layer_major.FLAG_HYBRID: "maybe"}):
            with self.assertRaises(ValueError):
                layer_major.hybrid_enabled()


def spans_of(sizes):
    out, position = [], 0
    for size in sizes:
        out.append((position, position + size))
        position += size
    return out


class WindowPlanTests(unittest.TestCase):
    def test_windows_hold_whole_chunks_within_the_limit(self):
        spans = spans_of([2048] * 10)
        windows = layer_major.plan_windows(spans, 0, 8192)
        self.assertEqual([len(w) for w in windows], [4, 4, 2])
        self.assertEqual([s for w in windows for s in w], spans)

    def test_a_chunk_larger_than_the_window_still_runs_alone(self):
        windows = layer_major.plan_windows(spans_of([3000, 3000]), 0, 1024)
        self.assertEqual([len(w) for w in windows], [1, 1])

    def test_window_ends_at_the_required_system_boundary(self):
        spans = spans_of([2048, 2048, 1349, 2048, 2048])       # the system prompt ends at 5445
        windows = layer_major.plan_windows(spans, 0, 8192, required=(5445,))
        self.assertEqual(windows[0][-1][1], 5445)
        self.assertEqual([s for w in windows for s in w], spans)

    def test_required_position_is_offset_aware(self):
        spans = spans_of([2048, 2048, 2048])                    # relative; the cache already holds 1000 tokens
        windows = layer_major.plan_windows(spans, 1000, 8192, required=(1000 + 2048,))
        self.assertEqual(windows[0], spans[:1])

    def test_window_prefers_an_anchor_in_its_second_half(self):
        spans = spans_of([2048] * 6)
        windows = layer_major.plan_windows(spans, 0, 8192, anchors=(6144,))
        self.assertEqual(windows[0][-1][1], 6144)               # not 8192: the message end is the better snapshot point

    def test_anchor_in_the_first_half_is_ignored(self):
        spans = spans_of([2048] * 6)
        windows = layer_major.plan_windows(spans, 0, 8192, anchors=(2048,))
        self.assertEqual(windows[0][-1][1], 8192)

    def test_empty_plan(self):
        self.assertEqual(layer_major.plan_windows([], 0, 8192), [])


class CostModelTests(unittest.TestCase):
    """The real checkpoint: 48 layers, 512 experts, 3,072,000-byte records (docs/flashnext/architecture-overhaul.md)."""

    RECORD = 3_072_000

    def test_one_full_pass_is_75_gb(self):
        self.assertAlmostEqual(layer_major.expert_pass_bytes(512, 48, self.RECORD) / 1e9, 75.5, delta=0.1)

    def test_layer_major_reads_once_per_window_not_once_per_chunk(self):
        tokens = 16384
        chunk = layer_major.chunk_major_bytes(tokens, 2048, 48, 400, self.RECORD)
        window = layer_major.layer_major_bytes(tokens, 8192, 48, 512, self.RECORD)
        self.assertAlmostEqual(chunk / 1e9, 472.4, delta=1.0)
        self.assertAlmostEqual(window / 1e9, 151.0, delta=1.0)
        self.assertGreater(chunk / window, 3.0)

    def test_activations_are_small(self):
        self.assertEqual(layer_major.activation_bytes(8192, 4, 2560), 8192 * 4 * 2560 * 2)
        self.assertLess(layer_major.activation_bytes(32768, 4, 2560), 700_000_000)

    def test_overlap_bound_and_serialized_time(self):
        self.assertEqual(layer_major.predicted_seconds(1000, 100.0, int(3.2e9 * 10), 3.2, True), 10.0)
        self.assertEqual(layer_major.predicted_seconds(1000, 100.0, int(3.2e9 * 10), 3.2, False), 20.0)


class LayerPrefetchTests(RingCase):
    """Opt-in layer prefetch on real StreamingSwitchGLU layers: same bytes, overlapped, drained on every exit."""

    create_main_pool = False
    extra_env = {"FLASHNEXT_PREFILL_LAYER_PREFETCH": "1"}

    def run_two_layers(self, window, chunks):
        """Layer-major over two layers: for each layer, every chunk; the window prefetches the next layer first."""
        expert_cache._LAYER_WINDOW[0] = window
        outputs = []
        try:
            for index, glu in enumerate(self.layers):
                if index + 1 < len(self.layers):
                    window.prefetch(self.layers[index + 1])
                for x, i in chunks:
                    y = glu(x, i)
                    mx.eval(y)
                    outputs.append(y)
                window.end_layer()
        finally:
            expert_cache._LAYER_WINDOW[0] = None
            window.drain()
        return outputs

    def test_prefetched_layers_are_bit_identical_to_blocking_reads_and_are_used(self):
        chunks = [self.inputs(tokens=48, width=4, seed=s, experts=12) for s in (3, 4)]
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_LAYER_PREFETCH": "0"}):
            blocking = layer_major.ExpertWindow(ring=None)
            want = self.run_two_layers(blocking, chunks)
            self.assertEqual(blocking.stats["prefetch_layers"], 0, "off by default: nothing is prefetched")
        window = layer_major.ExpertWindow(ring=None)
        got = self.run_two_layers(window, chunks)
        for g, w in zip(got, want):
            self.assertTrue(bool(mx.array_equal(g, w)))
        self.assertEqual(window.stats["prefetch_layers"], 1)
        self.assertEqual(window.stats["prefetch_used"], 1, "the second layer must come from the prefetch")
        self.assertEqual(window.stats["layers"], 2)
        self.assertEqual(window.stats["bytes"], blocking.stats["bytes"])
        self.assertEqual(window._prefetched, {})

    def test_prefetch_is_a_no_op_with_a_ring_with_the_flag_off_and_for_non_streamed_layers(self):
        window = layer_major.ExpertWindow(ring=object())
        self.assertFalse(window.prefetch(self.layers[1]))
        window = layer_major.ExpertWindow(ring=None)
        self.assertFalse(window.prefetch(None))
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_LAYER_PREFETCH": "0"}):
            self.assertFalse(window.prefetch(self.layers[1]))
        self.assertTrue(window.prefetch(self.layers[1]))
        self.assertFalse(window.prefetch(self.layers[1]), "never the same layer twice")
        self.assertTrue(window.prefetch(self.layers[0]), "the layer being consumed and the next may both be pending")
        window.drain()
        self.assertEqual(window._prefetched, {})

    def test_abort_and_drain_wait_for_queued_reads_and_clear_them(self):
        window = layer_major.ExpertWindow(ring=None)
        self.assertTrue(window.prefetch(self.layers[1]))
        _, pending = window._prefetched[self.layers[1].layer_id]
        window.abort_layer()
        self.assertEqual(window._prefetched, {})
        for projection in pending:                       # every queued read has finished: nothing writes after abort
            for part in projection:
                for future in part.futures:
                    self.assertTrue(future.done())

    def test_a_prefetch_for_another_layer_is_not_consumed_early_and_is_drained_at_the_end(self):
        window = layer_major.ExpertWindow(ring=None)
        window.prefetch(self.layers[1])
        expert_cache._LAYER_WINDOW[0] = window
        try:
            x, i = self.inputs(tokens=48, width=4, experts=12)
            mx.eval(self.layers[0](x, i))              # layer 0 was not prefetched: blocking read, layer 1's entry waits
        finally:
            expert_cache._LAYER_WINDOW[0] = None
            window.end_layer()
        self.assertEqual(window.stats["prefetch_used"], 0)
        self.assertEqual(list(window._prefetched), [self.layers[1].layer_id])
        window.drain()
        self.assertEqual(window._prefetched, {})

    def test_flag_validation_and_mode_wiring(self):
        from models.flashnext.settings import modes
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_LAYER_PREFETCH": "2"}):
            with self.assertRaises(ValueError):
                layer_major.prefetch_enabled()
        self.assertEqual(modes.EXPERIMENTAL_OFF["FLASHNEXT_PREFILL_LAYER_PREFETCH"], "0")
        with self.assertRaises(modes.ModeError):
            modes.plan_mode("agentic", enable=("layer-prefetch",), available_gb=12, free_disk_gb=200)
        plan = modes.plan_mode("agentic", enable=("layer-major-4k", "layer-prefetch"), available_gb=12, free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_LAYER_PREFETCH"], "1")
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_WINDOW_TOKENS"], "4096")
        self.assertEqual(modes.plan_mode("agentic", available_gb=12, free_disk_gb=200).env["FLASHNEXT_PREFILL_LAYER_PREFETCH"], "0")


class ModelEquivalenceTests(unittest.TestCase):
    """The two orders run the same operations on the same inputs, so the results must match byte for byte."""

    model = None

    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        mx.random.seed(11)
        cls.ids = [int(t) for t in mx.random.randint(2, 512, (700,)).tolist()]

    def chunk_major(self, caches, start, spans):
        outs = []
        for a, b in spans:
            result = self.model(mx.array([self.ids[start + a:start + b]]), cache=caches, return_hidden=True, skip_logits=True)
            mx.eval(result.hidden_states[-1], [c.state for c in caches])
            outs.append(result.hidden_states[-1])
        return outs

    def layer_major(self, caches, start, spans):
        ids = mx.array([self.ids[start + spans[0][0]:start + spans[-1][1]]])
        rel = [(a - spans[0][0], b - spans[0][0]) for a, b in spans]
        states, report = layer_major.run_window(self.model, ids, caches, rel, offset=start + spans[0][0])
        return [self.model.model.hyper_connection_mixer(s) for s in states], report

    def check(self, spans, offset=0):
        base = self.model.make_cache()
        if offset:
            self.chunk_major(base, 0, [(0, offset)])
        reference = self.model.make_cache()
        candidate = self.model.make_cache()
        if offset:
            self.chunk_major(reference, 0, [(0, offset)])
            self.chunk_major(candidate, 0, [(0, offset)])
        want = self.chunk_major(reference, offset, spans)
        got, report = self.layer_major(candidate, offset, spans)
        mx.eval(got)
        self.assertEqual(cache_bits(candidate), cache_bits(reference), "cache state differs")
        for g, w in zip(got, want):
            self.assertTrue(bool(mx.array_equal(g, w)), "hidden states differ")
        return report

    def test_equal_chunks_from_an_empty_cache(self):
        report = self.check(spans_of([128, 128, 128, 128]))
        self.assertEqual(report.chunks, 4)
        self.assertEqual(len(report.layer_seconds), 8)

    def test_uneven_chunks(self):
        self.check(spans_of([64, 200, 97, 131]))

    def test_resuming_from_a_restored_prefix(self):
        self.check([(a + 0, b + 0) for a, b in spans_of([150, 150, 150])], offset=160)

    def test_a_single_chunk_window(self):
        self.check(spans_of([192]))

    def test_two_windows_in_sequence_equal_one_chunk_major_run(self):
        reference = self.model.make_cache()
        want = self.chunk_major(reference, 0, spans_of([128, 128, 128, 128]))
        candidate = self.model.make_cache()
        a, _ = self.layer_major(candidate, 0, spans_of([128, 128]))
        b, _ = self.layer_major(candidate, 256, [(0, 128), (128, 256)])
        mx.eval(a, b)
        self.assertEqual(cache_bits(candidate), cache_bits(reference))
        self.assertTrue(bool(mx.array_equal(b[-1], want[-1])))

    def test_window_report_excludes_prior_window_statistics(self):
        from types import SimpleNamespace

        window = SimpleNamespace(stats={"bytes": 100, "read_seconds": 2.0,
                                        "ring_layers": 3, "fallback_layers": 4},
                                 end_layer=lambda: None)

        def record_layer(index, count):
            window.stats["bytes"] += 10
            window.stats["read_seconds"] += 0.25
            window.stats["ring_layers"] += 1

        _, report = layer_major.run_window(
            self.model, mx.array([self.ids[:64]]), self.model.make_cache(),
            [(0, 64)], offset=0, window=window, after_layer=record_layer)
        self.assertEqual(report.expert_bytes, 80)
        self.assertEqual(report.read_seconds, 2.0)
        self.assertEqual(report.ring_layers, 8)
        self.assertEqual(report.fallback_layers, 0)

    def test_window_hook_is_cleared_after_a_failure(self):
        caches = self.model.make_cache()
        ids = mx.array([self.ids[:128]])
        original = expert_cache._LAYER_WINDOW[0]
        sentinel = object()
        expert_cache._LAYER_WINDOW[0] = sentinel
        try:
            with mock.patch("mlx_vlm.models.qwen4_exp.language.Qwen4ExpDecoderLayer.__call__",
                            side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    layer_major.run_window(self.model, ids, caches, [(0, 128)], offset=0, window=None)
            self.assertIs(expert_cache._LAYER_WINDOW[0], sentinel)      # the previous hook is restored on every exit
        finally:
            expert_cache._LAYER_WINDOW[0] = original

    def test_window_failure_aborts_before_releasing_the_ring(self):
        from types import SimpleNamespace

        window = SimpleNamespace(stats={"bytes": 0, "read_seconds": 0.0, "ring_layers": 0,
                                        "fallback_layers": 0}, end_layer=mock.Mock(), abort_layer=mock.Mock())
        previous = expert_cache._LAYER_WINDOW[0]
        with mock.patch("mlx_vlm.models.qwen4_exp.language.Qwen4ExpDecoderLayer.__call__",
                        side_effect=RuntimeError("evaluation failed")):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                layer_major.run_window(self.model, mx.array([self.ids[:128]]), self.model.make_cache(),
                                       [(0, 128)], offset=0, window=window)
        window.abort_layer.assert_called_once_with()
        window.end_layer.assert_not_called()
        self.assertIs(expert_cache._LAYER_WINDOW[0], previous)


class MinimumSizeTests(unittest.TestCase):
    def test_default_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(layer_major.FLAG_MIN_TOKENS, None)
            self.assertEqual(layer_major.min_tokens(), 4096)
        with mock.patch.dict(os.environ, {layer_major.FLAG_MIN_TOKENS: "0"}):
            self.assertEqual(layer_major.min_tokens(), 0)
        for bad in ("-1", "abc"):
            with mock.patch.dict(os.environ, {layer_major.FLAG_MIN_TOKENS: bad}):
                with self.assertRaises(ValueError):
                    layer_major.min_tokens()

    def test_small_multi_chunk_prefill_stays_chunk_major(self):
        from models.flashnext import prefill

        env = {layer_major.FLAG_ORDER: "layer", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64", "FLASHNEXT_PREFILL_CHUNK_POLICY": "grid"}
        ids = mx.array([[int(t) for t in mx.random.randint(2, 512, (200,)).tolist()]])
        for minimum, expected in (("4096", 0), ("0", 1)):
            with mock.patch.dict(os.environ, {**env, layer_major.FLAG_MIN_TOKENS: minimum}), \
                 mock.patch.object(prefill, "_run_layer_major", return_value=None) as layered, \
                 mock.patch.object(prefill, "_prefill_chunk"), mock.patch.object(prefill, "_prefill_target", return_value=None):
                try:
                    prefill._run_chunks(None, ids, None, False, False, None, (), None, 0)
                except Exception:
                    pass
            self.assertEqual(layered.call_count, expected, minimum)


class PrefillIntegrationTests(unittest.TestCase):
    """``prefill_target`` with ``FLASHNEXT_PREFILL_ORDER=layer``: same token, same cache, same plan, coarser snapshots."""

    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        mx.random.seed(5)
        cls.ids = [int(t) for t in mx.random.randint(2, 512, (700,)).tolist()]

    def run_prefill(self, order, window=None, boundaries=(), required=(), anchors=None, prompt=None, offset=0, cache=None,
                    hybrid=False, inner=False, states=None):
        from models.flashnext import prefill

        env = {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "64", "FLASHNEXT_PREFILL_CHUNK_POLICY": "grid",
               layer_major.FLAG_ORDER: order, layer_major.FLAG_HYBRID: "1" if hybrid else "0",
               layer_major.FLAG_INNER_SNAPSHOTS: "1" if inner else "0",
               layer_major.FLAG_MIN_TOKENS: "0"}     # these tiny prompts must still exercise layer-major
        seen, trace = [], prefill.PlanTrace()
        cache = cache if cache is not None else self.model.make_cache()
        prompt = prompt or self.ids

        def on_boundary(position, frozen=None):
            seen.append(position)
            if states is not None:                      # what a snapshot would store at this position
                states[position] = cache_bits(cache if frozen is None else frozen)

        with mock.patch.dict(os.environ, env):
            with mock.patch.object(layer_major, "window_tokens", return_value=window or 128):
                _, _, token = prefill.prefill_target(
                    self.model, mx.array([prompt[offset:]]), cache, boundaries=tuple(boundaries),
                    on_boundary=on_boundary, offset=offset, anchors=anchors, trace=trace, required=required)
        return int(token.item()), cache, seen, trace

    def test_token_and_cache_equal_chunk_major(self):
        want_token, want_cache, _, _ = self.run_prefill("chunk")
        token, cache, _, trace = self.run_prefill("layer", window=256)
        self.assertEqual(token, want_token)
        self.assertEqual(cache_bits(cache), cache_bits(want_cache))
        self.assertEqual(trace.order, f"{layer_major.PLAN_VERSION}/w256")
        self.assertGreater(len(trace.windows), 1)
        self.assertEqual(trace.windows[0][0], 0)
        self.assertEqual(trace.windows[-1][1], len(self.ids))

    def test_trailing_tiny_chunk_runs_chunk_major_after_the_window_and_stays_exact(self):
        prompt = self.ids[:456]                                          # seven 64-token chunks and a tail of 8 tokens
        boundaries = (192, 384, 448)
        want_token, want_cache, want_seen, want_trace = self.run_prefill("chunk", boundaries=boundaries, prompt=prompt)
        token, cache, seen, trace = self.run_prefill("layer", window=512, boundaries=boundaries, prompt=prompt)
        self.assertEqual(token, want_token)
        self.assertEqual(cache_bits(cache), cache_bits(want_cache))
        self.assertEqual([(row["start"], row["end"], row["execution"]) for row in trace.layers],
                         [(0, 448, "layer"), (448, 456, "chunk")])
        self.assertEqual(trace.edges, want_trace.edges)                  # the executed edges are the reference's
        self.assertIn(448, seen)                                         # the window end is a usable snapshot point
        with mock.patch.dict(os.environ, {layer_major.FLAG_TAIL_CHUNK: "0"}):
            token_off, cache_off, _, trace_off = self.run_prefill("layer", window=512, boundaries=boundaries, prompt=prompt)
        self.assertEqual(token_off, want_token)
        self.assertEqual(cache_bits(cache_off), cache_bits(want_cache))
        self.assertEqual([row["execution"] for row in trace_off.layers], ["layer"])   # the switch restores the old schedule

    def test_peel_tail_rules(self):
        self.assertEqual(layer_major.peel_tail([(0, 64), (64, 72)]), ([(0, 64)], (64, 72)))
        self.assertEqual(layer_major.peel_tail([(0, 64), (64, 73)]), ([(0, 64), (64, 73)], None))   # 9 tokens: window path
        self.assertEqual(layer_major.peel_tail([(0, 8)]), ([(0, 8)], None))                         # nothing left to window
        with mock.patch.dict(os.environ, {layer_major.FLAG_TAIL_CHUNK: "0"}):
            self.assertEqual(layer_major.peel_tail([(0, 64), (64, 72)]), ([(0, 64), (64, 72)], None))
        with mock.patch.dict(os.environ, {layer_major.FLAG_TAIL_CHUNK: "2"}):
            with self.assertRaises(ValueError):
                layer_major.tail_chunk_enabled()

    def test_hybrid_windows_match_reference_and_snapshot_safe_chunk_edges(self):
        boundaries = tuple(range(64, 700, 64))
        want_token, want_cache, _, _ = self.run_prefill("chunk", boundaries=boundaries)
        with mock.patch.object(layer_major, "HYBRID_SMALL_WINDOW_TOKENS", 200):
            token, cache, seen, trace = self.run_prefill("layer", window=256, boundaries=boundaries, hybrid=True)
        self.assertEqual(token, want_token)
        self.assertTrue(cache_bits(cache) == cache_bits(want_cache), "hybrid cache differs from reference")
        self.assertEqual([row["execution"] for row in trace.layers], ["layer", "layer", "chunk"])
        self.assertIn(576, seen)  # inside the small window, all layers have reached this edge
        self.assertIn("hybrid-small-v1", trace.order)

    def test_hybrid_cancellation_keeps_each_completed_chunk_boundary(self):
        from models.flashnext import memory_controller, prefill

        seen, trace = [], prefill.PlanTrace()
        env = {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "64", "FLASHNEXT_PREFILL_CHUNK_POLICY": "grid",
               layer_major.FLAG_ORDER: "layer", layer_major.FLAG_HYBRID: "1"}
        with mock.patch.dict(os.environ, env), mock.patch.object(layer_major, "window_tokens", return_value=256), \
                mock.patch.object(layer_major, "HYBRID_SMALL_WINDOW_TOKENS", 256), \
                mock.patch.object(memory_controller, "checkpoint", side_effect=[{"waited_s": 0}, RuntimeError("stop")]):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                prefill.prefill_target(self.model, mx.array([self.ids[:256]]), self.model.make_cache(),
                                       boundaries=(64, 128, 192), on_boundary=seen.append, trace=trace)
        self.assertEqual(seen, [64, 128])
        self.assertEqual(trace.edges, [64, 128])

    def test_snapshots_happen_only_at_window_ends(self):
        every_chunk_edge = tuple(range(64, 700, 64))                    # what snapshot mode "all" asks for
        _, _, seen, trace = self.run_prefill("layer", window=256, boundaries=every_chunk_edge)      # inner snapshots off
        ends = [end for _, end in trace.windows[:-1]]
        self.assertEqual(seen, ends)
        _, _, seen_chunk, _ = self.run_prefill("chunk", boundaries=every_chunk_edge)
        self.assertGreater(len(seen_chunk), len(seen))                  # chunk-major snapshots at every edge
        self.assertTrue(set(seen) <= set(every_chunk_edge))

    def test_window_ends_at_the_required_system_boundary_without_inner_snapshots(self):
        _, _, seen, trace = self.run_prefill("layer", window=512, boundaries=(192,), required=(192,))
        self.assertEqual(trace.windows[0][1], 192)
        self.assertEqual(seen[0], 192)

    def test_inner_snapshots_keep_one_window_across_the_system_boundary_and_store_the_chunk_major_state(self):
        edges = (192, 256)                                              # neither is an anchor in the window's second half
        want_states = {}
        want_token, want_cache, want_seen, _ = self.run_prefill("chunk", boundaries=edges, states=want_states)
        states = {}
        token, cache, seen, trace = self.run_prefill("layer", window=512, boundaries=edges, required=(192,), inner=True,
                                                     states=states)
        self.assertEqual(trace.windows[0], (0, 512))                    # the boundary no longer ends the window
        self.assertEqual(seen, want_seen)
        self.assertEqual(states, want_states)                           # every cache array at both edges, byte for byte
        self.assertEqual(token, want_token)
        self.assertEqual(cache_bits(cache), cache_bits(want_cache))

    def test_inner_snapshot_stays_correct_after_later_chunks_update_the_live_cache(self):
        # The frozen state is taken mid-window; the live cache then moves on (KV buffers are updated in place).
        states = {}
        _, cache, seen, _ = self.run_prefill("layer", window=700, boundaries=(128,), inner=True, states=states)
        want = {}
        self.run_prefill("chunk", boundaries=(128,), states=want)
        self.assertEqual(seen, [128])
        self.assertEqual(states[128], want[128])
        self.assertNotEqual(states[128], cache_bits(cache))

    def test_a_restore_from_an_inner_snapshot_resumes_exactly(self):
        from models.flashnext import prefix_cache

        taken = {}

        def run_and_keep():
            from models.flashnext import prefill

            env = {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "64", "FLASHNEXT_PREFILL_CHUNK_POLICY": "grid",
                   layer_major.FLAG_ORDER: "layer", layer_major.FLAG_INNER_SNAPSHOTS: "1"}
            cache = self.model.make_cache()

            def keep(position, frozen=None):
                entry = prefix_cache.capture(cache if frozen is None else frozen, self.ids[:position], "fp", None, self.model)
                taken[position] = entry

            with mock.patch.dict(os.environ, env), mock.patch.object(layer_major, "window_tokens", return_value=700):
                prefill.prefill_target(self.model, mx.array([self.ids]), cache, boundaries=(192,), on_boundary=keep)

        run_and_keep()
        want_token, want_cache, _, _ = self.run_prefill("chunk")
        restored = prefix_cache.restore([taken[192]], self.model.make_cache)
        token, cache, _, _ = self.run_prefill("layer", window=700, offset=192, cache=restored, inner=True)
        self.assertEqual(token, want_token)
        self.assertEqual(cache_bits(cache), cache_bits(want_cache))

    def test_a_snapshot_at_a_window_end_resumes_exactly(self):
        # State at the window end equals chunk-major's state there, so a restore continues to the same final result.
        want_token, want_cache, _, _ = self.run_prefill("chunk")
        _, partial, _, trace = self.run_prefill("layer", window=256, prompt=self.ids[:256])
        token, cache, _, _ = self.run_prefill("layer", window=256, offset=256, cache=partial)
        self.assertEqual(token, want_token)
        self.assertEqual(cache_bits(cache), cache_bits(want_cache))

    def test_chunk_order_is_untouched_by_the_new_keywords(self):
        token, cache, seen, trace = self.run_prefill("chunk", boundaries=(128, 256))
        self.assertEqual(seen, [128, 256])
        self.assertEqual(trace.order, "")
        self.assertEqual(trace.windows, [])

    def test_fingerprint_is_unchanged_for_chunk_order_and_distinct_for_layer_order(self):
        from models.flashnext import prefix_session

        import tempfile

        def key(order):
            # Never the real cache directory: a running server owns it.
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                    os.environ, {layer_major.FLAG_ORDER: order, "FLASHNEXT_PREFILL_CHUNK_POLICY": "balanced",
                                 prefix_session.FLAG_DIR: directory}):
                session = prefix_session.PrefixSession(lambda text: [1], lambda: {"model": "m"})
                store = session.store()
                try:
                    return store.fp if hasattr(store, "fp") else store.fingerprint
                finally:
                    close = getattr(store, "close", None)
                    if close:
                        close()
        self.assertNotEqual(key("chunk"), key("layer"))


class LaunchWiringTests(unittest.TestCase):
    def test_agentic_default_is_chunk_major(self):
        from models.flashnext.settings import modes

        plan = modes.plan_mode(modes.AGENTIC, free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_ORDER"], "chunk")

    def test_enable_layer_major_sets_the_order_and_is_labelled(self):
        from models.flashnext.settings import modes

        plan = modes.plan_mode(modes.AGENTIC, enable=("layer-major",), free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_ORDER"], "layer")
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_WINDOW_TOKENS"], "8192")
        self.assertIn("layer-major", modes.banner(plan))

    def test_four_k_window_is_explicit_and_cannot_be_combined_with_eight_k(self):
        from models.flashnext.settings import modes

        plan = modes.plan_mode(modes.AGENTIC, enable=("layer-major-4k",), free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_ORDER"], "layer")
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_WINDOW_TOKENS"], "4096")
        hybrid = modes.plan_mode(modes.AGENTIC, enable=("layer-major-hybrid-4k",), free_disk_gb=200)
        self.assertEqual(hybrid.env["FLASHNEXT_PREFILL_HYBRID_SMALL_WINDOWS"], "1")
        with self.assertRaises(modes.ModeError):
            modes.plan_mode(modes.AGENTIC, enable=("layer-major", "layer-major-4k"), free_disk_gb=200)

    def test_supervisor_forwards_enable_to_chat_sh(self):
        from types import SimpleNamespace

        from models.flashnext import overnight

        args = SimpleNamespace(pool_gb="0", host="127.0.0.1", port=8080, checkpoint="/x", enable=["layer-major"])
        command = overnight.server_command(args)
        self.assertEqual(command[command.index("--enable") + 1], "layer-major")
        args.enable = []
        self.assertNotIn("--enable", overnight.server_command(args))

    def test_an_inherited_order_cannot_override_the_mode(self):
        from models.flashnext.settings import modes

        self.assertIn("FLASHNEXT_PREFILL_ORDER", modes.OWNED)
        self.assertIn("FLASHNEXT_PREFILL_WINDOW_TOKENS", modes.OWNED)


class WindowExpertsTests(RingCase):
    """Real StreamingSwitchGLU layers: the window's superset operand is bit-identical and reads each layer once."""

    extra_env = {"FLASHNEXT_PREFILL_RING_EXPERTS": "12"}

    def test_abort_fences_before_releasing_operands_and_preserves_them_on_failure(self):
        operands = object()
        ring = mock.Mock()
        window = layer_major.ExpertWindow(ring=ring)
        window._layer, window._weights = 3, operands

        def fail_fence():
            self.assertIs(window._weights, operands)
            raise RuntimeError("fence failed")

        ring.abort.side_effect = fail_fence
        with self.assertRaisesRegex(RuntimeError, "fence failed"):
            window.abort_layer()
        self.assertIs(window._weights, operands)
        ring._retire.assert_not_called()
        ring.abort.side_effect = lambda: self.assertIs(window._weights, operands)
        window.abort_layer()
        self.assertIsNone(window._weights)

    def test_real_ring_abort_fences_before_retiring_its_staged_layer(self):
        calls = []
        with prefill_ring.session(self.language, 1024) as ring:
            window = layer_major.ExpertWindow(ring=ring)
            window.weights_for(self.layers[0])
            self.assertIsNotNone(ring.pending)
            original_retire = ring._retire
            ring._synchronize = lambda: calls.append("fence")
            ring._retire = lambda: (calls.append("retire"), original_retire())[1]
            window.abort_layer()
            self.assertEqual(calls, ["fence", "retire"])
            self.assertIsNone(ring.pending)

    def reference(self, glu, x, indices):
        y = glu(x, indices)
        mx.eval(y)
        return y

    def test_window_output_equals_the_existing_path_for_several_chunks(self):
        glu = self.layers[0]
        chunks = [self.inputs(tokens=48, width=4, seed=s, experts=e) for s, e in ((3, 12), (4, 5), (5, 9))]
        want = [self.reference(glu, x, i) for x, i in chunks]
        window = layer_major.ExpertWindow(ring=None)
        expert_cache._LAYER_WINDOW[0] = window
        self.addCleanup(expert_cache._LAYER_WINDOW.__setitem__, 0, None)
        try:
            got = []
            for x, i in chunks:
                y = glu(x, i)
                mx.eval(y)
                got.append(y)
        finally:
            expert_cache._LAYER_WINDOW[0] = None
            window.end_layer()
        for g, w in zip(got, want):
            self.assertTrue(bool(mx.array_equal(g, w)))
        self.assertEqual(window.stats["layers"], 1, "three chunks must share one read of the layer")
        self.assertEqual(window.stats["fallback_layers"], 1)

    def test_ring_staged_window_is_bit_identical_and_reads_the_layer_once(self):
        glu = self.layers[0]
        chunks = [self.inputs(tokens=48, width=4, seed=s, experts=12) for s in (3, 4, 5)]
        want = [self.reference(glu, x, i) for x, i in chunks]
        with prefill_ring.session(self.language, 1024) as ring:
            self.assertIsNotNone(ring, prefill_ring.LAST_REASON[0])
            window = layer_major.ExpertWindow(ring=ring)
            expert_cache._LAYER_WINDOW[0] = window
            try:
                got = []
                for x, i in chunks:
                    y = glu(x, i)
                    mx.eval(y)
                    got.append(y)
                window.end_layer()                  # after the evaluations: the half becomes reusable
                staged = ring.stats()["staged_layers"]
                states = ring.ring.stats()["states"]
            finally:
                expert_cache._LAYER_WINDOW[0] = None
        for g, w in zip(got, want):
            self.assertTrue(bool(mx.array_equal(g, w)))
        self.assertEqual(staged, 1)
        self.assertEqual(window.stats["ring_layers"], 1)
        self.assertEqual(states, ["free"])
        self.assertRestored()

    def test_window_bytes_are_the_whole_layer_once(self):
        glu = self.layers[0]
        x, i = self.inputs(tokens=48, width=4, experts=12)
        window = layer_major.ExpertWindow(ring=None)
        expert_cache._LAYER_WINDOW[0] = window
        try:
            mx.eval(glu(x, i))
        finally:
            expert_cache._LAYER_WINDOW[0] = None
        row = sum(self.store.refs[f"{prefix_of(1)}.{p}.{k}"].row_bytes for p in PROJECTIONS for k in ("weight", "scales", "biases"))
        self.assertEqual(window.stats["bytes"], 12 * row)

    def test_decode_sized_calls_ignore_the_window(self):
        glu = self.layers[0]
        x, i = self.inputs(tokens=4, width=4, experts=12)
        window = layer_major.ExpertWindow(ring=None)
        expert_cache._LAYER_WINDOW[0] = window
        try:
            mx.eval(glu(x, i))
        finally:
            expert_cache._LAYER_WINDOW[0] = None
        self.assertEqual(window.stats["layers"], 0)

    def test_progress_observer_runs_on_every_window_call(self):
        glu = self.layers[0]
        seen = []
        expert_cache.set_prefill_progress(seen.append)
        self.addCleanup(expert_cache.set_prefill_progress, None)
        window = layer_major.ExpertWindow(ring=None)
        expert_cache._LAYER_WINDOW[0] = window
        try:
            for seed in (3, 4):
                x, i = self.inputs(tokens=48, width=4, seed=seed, experts=12)
                mx.eval(glu(x, i))
        finally:
            expert_cache._LAYER_WINDOW[0] = None
        self.assertEqual(seen, [glu.layer_id, glu.layer_id])


if __name__ == "__main__":
    unittest.main()
