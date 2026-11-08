"""Opt-in chunked prefill: planning, boundaries, state continuity, restored-versus-cold equality.

Real cache classes and the real continuation path run on a small random-weight model. What these tests
pin down: chunking is numerically equivalent to one call but NOT bit-identical (the single call is the
reference), and a run restored from a snapshot is bit-identical to a cold run that uses the same spans.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import prefill, prefix_cache as P
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits


class PlanTests(unittest.TestCase):
    def test_spans_cover_the_prompt_exactly(self):
        for total, chunk, bounds, offset in ((10_000, 2048, (), 0), (1000, 0, (300,), 0), (500, 256, (), 300),
                                             (700, 128, (130, 400), 100), (1, 64, (), 0), (64, 64, (), 0)):
            spans = prefill.plan_chunks(total, chunk, bounds, offset)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], total)
            for (a, b), (c, d) in zip(spans, spans[1:]):
                self.assertEqual(b, c)
            self.assertTrue(all(b > a for a, b in spans))

    def test_off_means_one_span(self):
        self.assertEqual(prefill.plan_chunks(5000, 0), [(0, 5000)])

    def test_edges_are_absolute_multiples_of_the_chunk(self):
        self.assertEqual(prefill.plan_chunks(560, 256, (), 300), [(0, 212), (212, 468), (468, 560)])

    def test_a_tiny_last_chunk_is_merged_but_a_boundary_never_is(self):
        self.assertEqual(prefill.plan_chunks(1025, 512, (30,)), [(0, 30), (30, 512), (512, 1025)])
        self.assertEqual(prefill.plan_chunks(500, 256, (), 300), [(0, 212), (212, 500)])
        self.assertEqual(prefill.plan_chunks(1025, 512, (1024,)), [(0, 512), (512, 1024), (1024, 1025)])

    def test_the_tail_rule_keeps_cold_and_restored_spans_equal(self):
        for total, chunk, boundary in ((1025, 512, 30), (1100, 256, 700), (10_001, 2048, 3000), (262, 100, 130)):
            cold = prefill.plan_chunks(total, chunk, (boundary,))
            restored = [(a + boundary, b + boundary)
                        for a, b in prefill.plan_chunks(total - boundary, chunk, (), boundary)]
            self.assertEqual([s for s in cold if s[0] >= boundary], restored, (total, chunk, boundary))

    def test_boundaries_are_edges_and_the_end_is_ignored(self):
        self.assertEqual(prefill.plan_chunks(1000, 0, (300, 1000, 0)), [(0, 300), (300, 1000)])

    def test_a_restored_run_uses_the_same_spans_as_the_cold_run_past_the_snapshot(self):
        cold = prefill.plan_chunks(10_000, 2048, (3000,))
        absolute = [(a, b) for a, b in cold]
        restored = [(a + 3000, b + 3000) for a, b in prefill.plan_chunks(7000, 2048, (3000,), 3000)]
        self.assertEqual([s for s in absolute if s[0] >= 3000], restored)

    def test_chunk_size_validation(self):
        for value, ok in (("0", True), ("", True), ("64", True), ("2048", True), ("63", False), ("-5", False), ("x", False)):
            with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_TOKENS: value}):
                if ok:
                    prefill.chunk_tokens()
                else:
                    with self.assertRaises(ValueError):
                        prefill.chunk_tokens()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(prefill.FLAG_CHUNK_TOKENS, None)
            self.assertEqual(prefill.chunk_tokens(), 0)


def legacy_grid_plan(total, chunk, boundaries=(), offset=0):
    """The original absolute-grid planner, kept here verbatim as the reference for the "grid" policy."""
    edges = set()
    if chunk:
        first = (offset // chunk + 1) * chunk
        edges.update(range(first, offset + total, chunk))
    wanted = {b for b in boundaries if offset < b < offset + total}
    edges.update(wanted)
    ordered = sorted(edges)
    if ordered and offset + total - ordered[-1] < prefill.MIN_CHUNK_TOKENS and ordered[-1] not in wanted:
        ordered.pop()
    points = [0] + [e - offset for e in ordered] + [total]
    return [(a, b) for a, b in zip(points, points[1:]) if b > a]


class BalancedPlanTests(unittest.TestCase):
    CHUNKS = (64, 100, 512, 777, 1000, 1500, 2048)
    ENDS = (1, 63, 64, 65, 513, 1037, 2047, 2048, 2049, 2068, 2110, 4118, 5000, 10_022)

    def plan(self, end, chunk, anchors=(), offset=0):
        return prefill.plan_chunks(end - offset, chunk, anchors, offset, policy="balanced", anchors=anchors)

    def test_grid_is_unchanged_for_every_input_the_old_planner_accepted(self):
        for total in (1, 64, 100, 513, 2068, 10_000):
            for chunk in (0, 64, 512, 2048):
                for bounds in ((), (14,), (2046,), (14, 600, 2046)):
                    for offset in (0, 14, 512, 2046):
                        self.assertEqual(prefill.plan_chunks(total, chunk, bounds, offset, policy="grid"),
                                         legacy_grid_plan(total, chunk, bounds, offset),
                                         (total, chunk, bounds, offset))

    def test_the_default_policy_is_grid(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(prefill.FLAG_CHUNK_POLICY, None)
            self.assertEqual(prefill.chunk_policy(), "grid")
            self.assertEqual(prefill.plan_chunks(2068, 512, (2046,), 0), legacy_grid_plan(2068, 512, (2046,), 0))

    def test_spans_cover_the_prompt_never_exceed_the_chunk_and_make_anchors_edges(self):
        for end in self.ENDS:
            for chunk in self.CHUNKS:
                for anchors in ((), (14,), (end // 2,), (14, end // 3, 2 * end // 3), (end - 22,)):
                    spans = self.plan(end, chunk, anchors)
                    self.assertEqual(spans[0][0], 0)
                    self.assertEqual(spans[-1][1], end)
                    self.assertTrue(all(b > a for a, b in spans))
                    self.assertTrue(all(b - a <= chunk for a, b in spans), (end, chunk, anchors, spans))
                    edges = {b for _, b in spans}
                    for anchor in anchors:
                        if 0 < anchor < end:
                            self.assertIn(anchor, edges)
                    for (a, b), (c, d) in zip(spans, spans[1:]):
                        self.assertEqual(b, c)

    def test_each_segment_uses_the_fewest_pieces_and_equal_sizes(self):
        for end in self.ENDS:
            for chunk in self.CHUNKS:
                for anchors in ((), (end // 2,), (14, end // 3)):
                    spans = self.plan(end, chunk, anchors)
                    points = [0] + sorted(a for a in anchors if 0 < a < end) + [end]
                    for start, stop in zip(points, points[1:]):
                        inside = [(a, b) for a, b in spans if start <= a and b <= stop]
                        sizes = [b - a for a, b in inside]
                        self.assertEqual(len(inside), -(-(stop - start) // chunk), (end, chunk, anchors))
                        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_no_tiny_chunk_appears_unless_a_segment_itself_is_tiny(self):
        # The case that motivated the policy: a boundary a few tokens past a grid line.
        grid = legacy_grid_plan(2110, 512, (2088,))
        balanced = self.plan(2110, 512, (2088,))
        self.assertIn(40, [b - a for a, b in grid])
        self.assertEqual(min(b - a for a, b in balanced), 22)           # only the question segment
        self.assertEqual(len(balanced), 6)                              # 5 pieces of the system segment + 1
        self.assertEqual(self.plan(2110, 2048, (2088,)), [(0, 1044), (1044, 2088), (2088, 2110)])
        self.assertEqual(self.plan(2068, 2048, (2046,)), [(0, 2046), (2046, 2068)])

    def test_a_restored_plan_is_the_cold_plan_from_the_reused_boundary_onward(self):
        for end in self.ENDS:
            for chunk in self.CHUNKS:
                for anchors in ((), (14,), (end // 2,), (14, end // 3, 2 * end // 3)):
                    cold = self.plan(end, chunk, anchors)
                    cold_edges = [b for _, b in cold][:-1]
                    for edge in cold_edges + [a for a in anchors if 0 < a < end]:
                        restored = self.plan(end, chunk, anchors, offset=edge)
                        shifted = [(a + edge, b + edge) for a, b in restored]
                        self.assertEqual(shifted, [(a, b) for a, b in cold if a >= edge],
                                         (end, chunk, anchors, edge))

    def test_a_hit_that_is_not_an_edge_shortens_only_the_first_span(self):
        cold = self.plan(5000, 1000, (14,))
        for offset in (100, 777, 1500, 3000):
            restored = [(a + offset, b + offset) for a, b in self.plan(5000, 1000, (14,), offset=offset)]
            self.assertEqual(restored[0][0], offset)
            self.assertEqual(restored[1:], [(a, b) for a, b in cold if a > offset and a >= restored[0][1]])
            self.assertLessEqual(restored[0][1], offset + 1000)

    def test_the_plan_does_not_depend_on_the_cached_length(self):
        # Same prompt, different amounts already cached: every plan is a tail of one cold plan.
        cold = self.plan(8214, 2048, (4096,))
        edges = {b for _, b in cold}
        for offset in (0, 4096, 2048, 100):
            tail = [(a + offset, b + offset) for a, b in self.plan(8214, 2048, (4096,), offset=offset)]
            for _, b in tail:
                self.assertIn(b, edges)

    def test_tool_message_boundaries_are_edges_and_restoring_at_each_matches(self):
        anchors = (2046, 2610, 3320, 3999)               # system, user, assistant, tool result
        cold = self.plan(4300, 2048, anchors)
        self.assertEqual({b for _, b in cold} >= set(anchors), True)
        for anchor in anchors:
            restored = [(a + anchor, b + anchor) for a, b in self.plan(4300, 2048, anchors, offset=anchor)]
            self.assertEqual(restored, [(a, b) for a, b in cold if a >= anchor])

    def test_anchors_outside_the_prompt_are_ignored(self):
        self.assertEqual(self.plan(1000, 512, (0, 1000, 5000)), self.plan(1000, 512, ()))

    def test_off_with_anchors_cuts_only_at_the_anchors(self):
        self.assertEqual(self.plan(3000, 0, (14, 2000)), [(0, 14), (14, 2000), (2000, 3000)])
        self.assertEqual(self.plan(3000, 0), [(0, 3000)])

    def test_default_anchors_are_the_boundaries(self):
        self.assertEqual(prefill.plan_chunks(2068, 2048, (2046,), 0, policy="balanced"),
                         [(0, 2046), (2046, 2068)])

    def test_policy_and_size_validation(self):
        for value, ok in (("grid", True), ("balanced", True), ("BALANCED", True), ("", True), ("fast", False)):
            with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_POLICY: value}):
                if ok:
                    self.assertIn(prefill.chunk_policy(), prefill.CHUNK_POLICIES)
                else:
                    with self.assertRaises(ValueError):
                        prefill.chunk_policy()
        for value, ok in (("2048", True), ("2049", False), ("4096", False), ("512", True), ("63", False)):
            with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_TOKENS: value}):
                if ok:
                    self.assertEqual(prefill.chunk_tokens(), int(value))
                else:
                    with self.assertRaises(ValueError):
                        prefill.chunk_tokens()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        mx.random.seed(21)
        cls.ids = [int(t) for t in mx.random.randint(2, VOCAB, (260,)).tolist()]

    def run_prefill(self, chunk, upto=260, bounds=(), on_boundary=None, model=None, caches=None, start=0, offset=0):
        model = model or self.model
        caches = caches if caches is not None else model.make_cache()
        env = {prefill.FLAG_CHUNK_TOKENS: str(chunk)}
        with mock.patch.dict(os.environ, env):
            logits, hidden, token = prefill.prefill_target(
                model, mx.array([self.ids[start:upto]]), caches, boundaries=bounds,
                on_boundary=on_boundary, offset=offset)
        mx.eval(token, [c.state for c in caches])
        return caches, int(token.item())


class ChunkedPrefillTests(Base):
    def test_off_runs_one_call(self):
        with mock.patch.object(prefill, "_prefill_chunk", side_effect=AssertionError("chunked while off")):
            self.run_prefill(0)

    def test_chunking_matches_one_call_numerically_and_picks_the_same_token(self):
        whole, token = self.run_prefill(0)
        chunked, chunked_token = self.run_prefill(64)
        self.assertEqual(token, chunked_token)
        for a, b in zip(whole, chunked):
            self.assertEqual(P.kind_of(a), P.kind_of(b))
            states_a = a.cache if P.kind_of(a) == "arrays" else [a.keys[..., : a.offset, :], a.values[..., : a.offset, :], a.index_keys]
            states_b = b.cache if P.kind_of(b) == "arrays" else [b.keys[..., : b.offset, :], b.values[..., : b.offset, :], b.index_keys]
            for x, y in zip(states_a, states_b):
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(mx.allclose(x.astype(mx.float32), y.astype(mx.float32), rtol=1e-3, atol=1e-4).item())

    def test_every_cache_holds_the_whole_prompt_afterwards(self):
        caches, _ = self.run_prefill(64)
        for cache in caches:
            if P.kind_of(cache) == "qsa_kv":
                self.assertEqual(cache.offset, 260)
                self.assertEqual(cache.index_keys.shape[1], 260)

    def test_boundaries_report_the_cache_at_the_right_position(self):
        seen = []
        caches = self.model.make_cache()

        def record(position):
            offsets = {c.offset for c in caches if P.kind_of(c) == "qsa_kv"}
            seen.append((position, offsets))

        self.run_prefill(0, bounds=(100, 180), on_boundary=record, caches=caches)
        self.assertEqual(seen, [(100, {100}), (180, {180})])

    def test_boundaries_that_are_not_edges_are_not_reported(self):
        seen = []
        self.run_prefill(64, bounds=(100,), on_boundary=seen.append)
        self.assertEqual(seen, [100])

    def test_no_chunking_when_logits_are_requested(self):
        caches = self.model.make_cache()
        with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_TOKENS: "64"}), \
                mock.patch.object(prefill, "_prefill_chunk", side_effect=AssertionError("chunked")):
            logits, _, _ = prefill.prefill_target(self.model, mx.array([self.ids]), caches, want_logits=True)
        self.assertEqual(logits.shape[1], 260)

    def test_the_last_chunk_decides_full_logits_from_the_whole_prompt(self):
        calls = []
        original = prefill._prefill_target

        def spy(language, ids, cache, *a, **k):
            calls.append((int(ids.shape[1]), k.get("total_tokens")))
            return original(language, ids, cache, *a, **k)

        with mock.patch.object(prefill, "_prefill_target", spy):
            self.run_prefill(100)
        self.assertEqual(calls, [(160, 260)], "the 60-token tail merges into the chunk before it")

    def test_each_chunk_evaluates_every_cache_state_before_the_next(self):
        """Memory stays bounded only if a chunk's temporaries are released when it ends."""
        caches = self.model.make_cache()
        with mock.patch.object(prefill.mx, "eval") as evaluate:
            prefill._prefill_chunk(self.model, mx.array([self.ids[:64]]), caches)
        evaluate.assert_called_once()
        (targets,), _ = evaluate.call_args
        self.assertEqual(len(targets), len(caches))

    def test_decode_after_chunked_prefill_continues(self):
        caches, token = self.run_prefill(64)
        out = self.model(mx.array([[token]]), cache=caches)
        mx.eval(out.logits)
        self.assertEqual(out.logits.shape, (1, 1, VOCAB))
        self.assertTrue(mx.isfinite(out.logits).all().item())


class RestoredEqualsColdTests(Base):
    """The property the prefix cache relies on."""

    def test_restored_run_is_bit_identical_to_a_cold_run_with_the_same_spans(self):
        boundary = 128
        fp = P.fingerprint(model="small", chunk=64)
        store = P.PrefixCache(fp, 1 << 40)

        def snapshot(position):
            store.put(self.ids[:position], cold_caches, language=self.model)

        cold_caches = self.model.make_cache()
        cold, cold_token = self.run_prefill(64, bounds=(boundary,), on_boundary=snapshot, caches=cold_caches)
        fresh = build_model()
        hit, restored = store.lookup(self.ids, fresh.make_cache, language=fresh)
        self.assertEqual(hit.n_tokens, boundary)
        caches, token = self.run_prefill(64, model=fresh, caches=restored, start=hit.n_tokens, offset=hit.n_tokens)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(caches) == cache_bits(cold), "restored continuation differs bitwise")

    def test_chunking_a_restored_suffix_from_the_wrong_offset_would_differ(self):
        """Why ``offset`` exists: spans measured from the suffix start are not the cold run's spans."""
        cold = prefill.plan_chunks(260, 64, (130,))
        wrong = [(a + 130, b + 130) for a, b in prefill.plan_chunks(130, 64)]
        right = [(a + 130, b + 130) for a, b in prefill.plan_chunks(130, 64, (), 130)]
        self.assertNotEqual([s for s in cold if s[0] >= 130], wrong)
        self.assertEqual([s for s in cold if s[0] >= 130], right)


if __name__ == "__main__":
    unittest.main()
