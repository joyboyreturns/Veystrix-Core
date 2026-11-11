"""The four-request workflow on the small model: cold, same system prompt, extended, restart."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

from models.flashnext import prefill, prefix_cache as P, prefix_session as S
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits

MARK = [3, 4]                                   # stands for "<|im_end|>\n"


def compat():
    return {"model": "small", "tokenizer": "t", "engine": "e", "profile": {}}


def encode(text):
    assert text == S.IM_END
    return list(MARK)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(5)
        body = [int(t) for t in mx.random.randint(5, VOCAB, (300,)).tolist()]
        cls.system = body[:150] + MARK                 # system message, ends at 152
        cls.suffix_a = body[150:200]
        cls.suffix_b = body[200:240]
        cls.suffix_c = body[240:300]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = {S.FLAG: "1", S.FLAG_DIR: str(self.tmp / "cache"), S.FLAG_RAM_GB: "1", S.FLAG_DISK_GB: "1",
               S.FLAG_MIN_TOKENS: "64", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def engine(self):
        model = build_model()
        return model, S.PrefixSession(encode, compat)

    def go(self, model, session, prompt):
        token, cache, report = session.prefill(model, prompt, model.make_cache())
        mx.eval(token)
        return int(token.item()), cache, report


class WorkflowTests(Base):
    def test_restored_suffix_uses_incremental_limit_and_miss_uses_cold_limit(self):
        system = [5] * 62 + MARK
        with mock.patch.dict(os.environ, {"VEYSTRIX_MAX_COLD_PREFILL": "100",
                                          "VEYSTRIX_MAX_INCREMENTAL_PREFILL": "128"}):
            model, session = self.engine()
            self.go(model, session, system + [7] * 30)
            _, _, report = self.go(model, session, system + [7] * 120)
            self.assertEqual((report.hit_tokens, report.prefilled_tokens), (64, 120))
            with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "miss")}):
                with self.assertRaises(S.RecoveryRequiredError):
                    self.go(build_model(), S.PrefixSession(encode, compat), system + [7] * 120)

    def test_pressure_skips_snapshot_without_changing_live_result(self):
        model, session = self.engine()
        with mock.patch("models.flashnext.memory_controller.forbid_optional", return_value=True):
            token, cache, report = self.go(model, session, self.system + self.suffix_a)
        self.assertEqual(report.snapshots, [])
        self.assertEqual(report.snapshots_skipped, [{"position": len(self.system), "reason": "memory_pressure"}])
        self.assertEqual(session.store().usage()["ram_entries"], 0)
        self.assertIsInstance(token, int)
        self.assertTrue(cache)

    def test_full_routing_never_restores_a_pruned_routing_snapshot(self):
        from models.flashnext.routing import RoutingProfile
        from models.flashnext.tests.unit.test_routing import FakeStore

        def compatible(threshold):
            # This test needs the compatibility fingerprint, not a global
            # change to the adaptive router used by subsequent tests.
            with mock.patch.object(RoutingProfile, "reset"):
                profile = RoutingProfile("exact-quality", FakeStore(), object(), threshold=threshold)
            return {**compat(), "profile": profile.session_profile(set())}

        model = build_model()
        prompt = self.system + self.suffix_a
        old = S.PrefixSession(encode, lambda: compatible(0.85))
        _, _, before = self.go(model, old, prompt)
        self.assertEqual(before.snapshots, [len(self.system)])
        full = S.PrefixSession(encode, lambda: compatible(1.0))
        _, _, different = self.go(model, full, prompt)
        self.assertEqual(different.hit_tokens, 0)
        resumed_old = S.PrefixSession(encode, lambda: compatible(0.85))
        _, _, same = self.go(model, resumed_old, prompt)
        self.assertEqual(same.hit_tokens, len(self.system))
        self.assertEqual(same.tier, "ssd")

    def test_layer_order_report_records_the_executed_windows(self):
        from models.flashnext import layer_major

        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_ORDER": "layer", layer_major.FLAG_MIN_TOKENS: "0"}), \
                mock.patch.object(layer_major, "window_tokens", return_value=128):
            model, session = self.engine()
            _, _, report = self.go(model, session, self.system + self.suffix_a)
        self.assertEqual(report.execution_order, "layer-major-v1/w128")
        self.assertTrue(report.execution_windows)
        self.assertTrue(all(segment["execution"] == "layer" for segment in report.execution_segments))
        self.assertEqual(report.execution_windows[0][0], 0)
        self.assertEqual(report.execution_windows[-1][1], len(self.system + self.suffix_a))

    def test_cold_prompt_snapshots_at_the_system_boundary_and_hits_nothing(self):
        model, session = self.engine()
        _, _, report = self.go(model, session, self.system + self.suffix_a)
        self.assertEqual((report.hit_tokens, report.snapshots), (0, [152]))
        self.assertEqual(report.prefilled_tokens, 152 + 50)

    def test_a_boundary_inside_the_first_chunk_splits_the_run_and_snapshots_exact_state(self):
        """Stage A shape: a 14-token system message inside a larger chunk. The run is split at 14 (the chunk
        is never rewound), the snapshot is what a fresh 14-token prefill produces, and it reaches the SSD."""
        from models.flashnext.tests.bench import bench_prefill_memory as B

        system = [int(t) for t in mx.random.randint(5, VOCAB, (12,)).tolist()] + MARK     # 14 tokens
        prompt = system + self.suffix_a + self.suffix_b + self.suffix_c                    # 164 tokens
        with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "8", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64"}):
            model, session = self.engine()
            with B.ChunkRecorder() as chunks:
                _, cache, report = self.go(model, session, prompt)
            self.assertEqual(report.snapshots, [14])
            self.assertGreater(report.snapshot_seconds, 0.0)
            self.assertEqual(len(chunks.seconds), len(chunks.spans))
            self.assertTrue(all(x > 0 for x in chunks.seconds))
            self.assertEqual(chunks.spans, [14, 50, 100])      # edge 128 leaves a 36-token tail, merged
            self.assertEqual(chunks.edges(), [14, 64, 164])
            fresh_model = build_model()
            fresh = fresh_model.make_cache()
            mx.eval(fresh_model(mx.array([system]), cache=fresh).logits, [c.state for c in fresh])
            hit, restored = session.store().lookup(system + [7], model.make_cache, language=model)
            self.assertEqual(hit.n_tokens, 14)
            self.assertTrue(cache_bits(restored) == cache_bits(fresh), "snapshot differs from a fresh 14-token prefill")
            self.assertGreater(session.store().usage()["disk_bytes"], 0)
            files = [f for f in (self.tmp / "cache").rglob("*") if f.is_file()]
            self.assertTrue(files, "the snapshot was not written to the SSD tier")
            # A restored run on a fresh engine reuses exactly 14 tokens.
            _, restored_session = self.engine()
            _, _, again = self.go(build_model(), restored_session, system + self.suffix_c)
            self.assertEqual(again.hit_tokens, 14)

    def test_taking_a_snapshot_does_not_change_the_run_that_takes_it(self):
        """Same chunk plan, with and without the snapshot callback: bit-identical caches and token."""
        from models.flashnext import prefill

        system = [int(t) for t in mx.random.randint(5, VOCAB, (12,)).tolist()] + MARK
        prompt = system + self.suffix_a + self.suffix_b + self.suffix_c
        with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "8", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64"}):
            model, session = self.engine()
            token, with_snapshot, report = self.go(model, session, prompt)
            self.assertEqual(report.snapshots, [14])
            plain_model = build_model()
            plain_cache = plain_model.make_cache()
            _, _, plain_token = prefill.prefill_target(
                plain_model, mx.array([prompt]), plain_cache, boundaries=(14,), on_boundary=None)
            mx.eval(plain_token, [c.state for c in plain_cache])
        self.assertEqual(token, int(plain_token.item()))
        self.assertTrue(cache_bits(with_snapshot) == cache_bits(plain_cache), "the snapshot changed the run")

    def test_same_system_prompt_with_a_new_question_reuses_it_and_matches_a_cold_run(self):
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_a)
        token, cache, report = self.go(model, session, self.system + self.suffix_b)
        self.assertEqual((report.hit_tokens, report.tier, report.prefilled_tokens), (152, "ram", 40))
        cold_model, cold_session = self.engine()
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "other")}):
            cold_token, cold_cache, cold_report = self.go(cold_model, S.PrefixSession(encode, compat), self.system + self.suffix_b)
        self.assertEqual(cold_report.hit_tokens, 0)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(cache) == cache_bits(cold_cache), "reused prefix changed the state")

    def test_a_long_suffix_uses_the_cold_runs_chunk_edges(self):
        """The suffix crosses an absolute chunk edge (192): restored and cold runs must split alike."""
        long_suffix = self.suffix_a + self.suffix_b
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_c)          # stores the system prompt
        token, cache, report = self.go(model, session, self.system + long_suffix)
        self.assertEqual(report.hit_tokens, 152)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "other")}):
            cold_token, cold_cache, _ = self.go(build_model(), S.PrefixSession(encode, compat), self.system + long_suffix)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(cache) == cache_bits(cold_cache), "restored run split its chunks differently")

    def test_snapshots_in_one_prefill_are_chained_so_kv_is_stored_once(self):
        model, session = self.engine()
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks"}):
            _, _, report = self.go(model, session, self.system + self.suffix_a + self.suffix_b)
        store = session.store()
        keys = [P.chain_key(store.fp, tuple((self.system + self.suffix_a + self.suffix_b)[:n])) for n in report.snapshots]
        entries = [store._ram[k] for k in keys]
        self.assertEqual(entries[0].parent, None)
        for earlier, later in zip(entries, entries[1:]):
            self.assertEqual(later.parent, earlier.key)
        kv_tokens = sum(e.layers[3].kv["keys"].shape[2] for e in entries)
        self.assertEqual(kv_tokens, report.snapshots[-1], "each token's KV is stored exactly once")

    def test_snapshots_after_a_hit_chain_from_the_hit_and_then_from_each_other(self):
        model, session = self.engine()
        long_prompt = self.system + self.suffix_a + self.suffix_b + self.suffix_c
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks"}):
            self.go(model, session, self.system + self.suffix_c)
            _, _, report = self.go(model, session, long_prompt)
        self.assertEqual(report.hit_tokens, 152)
        self.assertEqual(report.snapshots, [192, 256])
        store = session.store()
        first, second = (store._ram[P.chain_key(store.fp, tuple(long_prompt[:n]))] for n in report.snapshots)
        self.assertEqual(first.parent, P.chain_key(store.fp, tuple(long_prompt[:152])))
        self.assertEqual(second.parent, first.key, "the second snapshot extends the first, not the hit")
        self.assertEqual(second.layers[3].kv["keys"].shape[2], 256 - 192)

    def test_a_longer_conversation_with_a_tool_result_reuses_the_system_prompt(self):
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_a)
        _, _, report = self.go(model, session, self.system + self.suffix_a + self.suffix_b + self.suffix_c)
        self.assertEqual(report.hit_tokens, 152)
        self.assertEqual(report.prefilled_tokens, 50 + 40 + 60)

    def test_a_restart_restores_the_system_prompt_from_ssd(self):
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_a)
        new_model, new_session = self.engine()               # new process: new model object, new cache object
        token, cache, report = self.go(new_model, new_session, self.system + self.suffix_b)
        self.assertEqual((report.hit_tokens, report.tier), (152, "ssd"))
        self.assertGreater(report.bytes_loaded, 0)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "other")}):
            cold_token, cold_cache, _ = self.go(*(lambda m: (m, S.PrefixSession(encode, compat)))(build_model()),
                                                 self.system + self.suffix_b)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(cache) == cache_bits(cold_cache))

    def test_a_changed_system_prompt_is_a_miss(self):
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_a)
        changed = list(self.system)
        changed[10] = (changed[10] + 1) % VOCAB
        _, _, report = self.go(model, session, changed + self.suffix_b)
        self.assertEqual(report.hit_tokens, 0)

    def test_a_prompt_that_is_exactly_the_system_message_leaves_a_token_to_prefill(self):
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_a)
        _, _, report = self.go(model, session, list(self.system))
        self.assertEqual(report.hit_tokens, 0)
        self.assertEqual(report.prefilled_tokens, 152)

    def test_a_different_engine_setting_is_a_different_cache(self):
        model, session = self.engine()
        self.go(model, session, self.system + self.suffix_a)
        other_model, other = self.engine()
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "128"}):
            _, _, report = self.go(other_model, S.PrefixSession(encode, compat), self.system + self.suffix_b)
        self.assertEqual(report.hit_tokens, 0, "chunk size changes numerics, so it is part of the key")

    def test_short_system_prompts_are_not_stored(self):
        model, session = self.engine()
        with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "400"}):
            _, _, report = self.go(model, session, self.system + self.suffix_a)
        self.assertEqual(report.snapshots, [])

    def test_chunks_mode_snapshots_every_chunk_edge_and_a_partial_hit_follows(self):
        model, session = self.engine()
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks"}):
            _, _, report = self.go(model, session, self.system + self.suffix_a + self.suffix_b)
            self.assertEqual(report.snapshots, [64, 128, 152, 192, 242][:len(report.snapshots)])
            self.assertIn(152, report.snapshots)
            changed_tail = self.system + self.suffix_a[:30] + [9] * 40
            _, _, again = self.go(model, session, changed_tail)
        self.assertEqual(again.hit_tokens, 152, "the tail changed after the 192-token edge was not reached")

    def test_boundary_mode_is_validated(self):
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "everything"}):
            with self.assertRaises(ValueError):
                S.boundary_mode()

    def test_flag_is_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(S.FLAG, None)
            self.assertFalse(S.enabled())


class BoundaryTests(Base):
    def test_system_boundary_is_after_the_first_marker_only(self):
        _, session = self.engine()
        prompt = [9, 9] + MARK + [8] + MARK + [7]
        self.assertEqual(session.system_boundary(prompt), 4)
        self.assertIsNone(session.system_boundary([9, 9, 3, 9, 4]))

    def test_boundaries_stay_inside_the_prompt_and_after_the_restored_prefix(self):
        _, session = self.engine()
        prompt = self.system + self.suffix_a
        self.assertEqual(session.boundaries(prompt, 0), (152,))
        self.assertEqual(session.boundaries(prompt, 152), ())
        self.assertEqual(session.boundaries(self.system, 0), (), "no token would be left to prefill")


class BalancedPolicyTests(Base):
    def test_new_plan_never_auto_chains_to_unrelated_old_boundary(self):
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks", S.FLAG_DIR: str(self.tmp / "chain")}):
            model, session = self.engine()
            self.go(model, session, [4] * 300)
            _, _, report = self.go(model, session, [4] * 330)
            self.assertEqual(report.hit_tokens, 0)
            store = session.store()
            first = store._ram[P.chain_key(store.fp, tuple([4] * report.snapshots[0]))]
            self.assertIsNone(first.parent)

    """The boundary-aware planner through the real prefix session on the small model."""

    def setUp(self):
        super().setUp()
        env = {"FLASHNEXT_PREFILL_CHUNK_POLICY": "balanced", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "128",
               S.FLAG_MIN_TOKENS: "8"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def system_of(self, length, seed):
        mx.random.seed(seed)
        body = [int(t) for t in mx.random.randint(5, VOCAB, (length - len(MARK),)).tolist()]
        return body + MARK

    def cold_reference(self, prompt, where="reference"):
        """The same prompt, uncached, under the same plan (an empty cache directory)."""
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / where)}):
            model = build_model()
            return self.go(model, S.PrefixSession(encode, compat), prompt)

    def recorded(self, model, session, prompt):
        from models.flashnext.tests.bench import bench_prefill_memory as B

        with B.ChunkRecorder() as chunks:
            token, cache, report = self.go(model, session, prompt)
        return token, cache, report, chunks

    def test_restored_equals_cold_for_system_messages_that_are_not_multiples_of_anything(self):
        for length in (14, 63, 64, 100, 129, 200, 257):
            with self.subTest(system=length):
                system = self.system_of(length, length)
                suffix_a, suffix_b = system[:0] + [7] * 40 + [9] * 11, [8] * 33 + [6] * 20
                with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / f"cache{length}")}):
                    model, session = self.engine()
                    _, _, cold_report = self.go(model, session, system + suffix_a)
                    self.assertEqual(cold_report.snapshots, [length])
                    token, cache, report, chunks = self.recorded(model, session, system + suffix_b)
                    self.assertEqual((report.hit_tokens, report.tier), (length, "ram"))
                    restart_model = build_model()
                    r_token, r_cache, r_report, r_chunks = self.recorded(
                        restart_model, S.PrefixSession(encode, compat), system + suffix_b)
                    self.assertEqual((r_report.hit_tokens, r_report.tier), (length, "ssd"))
                ref_token, ref_cache, _ = self.cold_reference(system + suffix_b, f"ref{length}")
                self.assertEqual(token, ref_token)
                self.assertTrue(cache_bits(cache) == cache_bits(ref_cache), "RAM restore differs from cold")
                self.assertEqual(r_token, ref_token)
                self.assertTrue(cache_bits(r_cache) == cache_bits(ref_cache), "SSD restore differs from cold")
                # The restored plan is the cold plan's tail, with the same spans.
                total = length + len(suffix_b)
                cold_spans = [b - a for a, b in prefill.plan_chunks(
                    total, 128, (length,), 0, policy="balanced", anchors=(length,))]
                self.assertEqual(chunks.spans, cold_spans[-len(chunks.spans):])
                self.assertEqual(chunks.spans, r_chunks.spans)

    def test_executed_chunks_follow_the_plan_and_the_boundary_is_an_edge(self):
        system = self.system_of(300, 1)
        prompt = system + [5] * 100
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "c")}):
            model, session = self.engine()
            _, _, report, chunks = self.recorded(model, session, prompt)
        self.assertEqual(report.snapshots, [300])
        self.assertEqual(chunks.spans, [100, 100, 100, 100])             # 300 in three pieces, then the question
        self.assertEqual(chunks.edges()[:3], [100, 200, 300])

    def test_the_grid_policy_still_plans_the_old_way_through_the_session(self):
        system = self.system_of(300, 1)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CHUNK_POLICY": "grid",
                                          S.FLAG_DIR: str(self.tmp / "g")}):
            model, session = self.engine()
            _, _, _, chunks = self.recorded(model, session, system + [5] * 100)
        self.assertEqual(chunks.spans, [128, 128, 44, 100])

    def test_tool_message_boundaries_are_snapshots_and_a_longer_conversation_reuses_the_latest(self):
        def message(length, seed):
            mx.random.seed(seed)
            return [int(t) for t in mx.random.randint(5, VOCAB, (length - len(MARK),)).tolist()] + MARK

        system, user, assistant, tool = message(150, 1), message(60, 2), message(45, 3), message(220, 4)
        follow_up, question = message(70, 5), [8] * 25
        first = system + user + assistant + [7] * 30                      # ends with an open message
        longer = system + user + assistant + tool + follow_up + question
        cuts = [150, 210, 255]
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages", S.FLAG_DIR: str(self.tmp / "m")}):
            model, session = self.engine()
            self.assertEqual(session.message_boundaries(longer), [150, 210, 255, 475, 545])
            self.assertEqual(session.anchors(first), tuple(cuts))
            _, _, cold = self.go(model, session, first)
            self.assertEqual(cold.snapshots, cuts)
            token, cache, report, chunks = self.recorded(model, session, longer)
            self.assertEqual(report.hit_tokens, 255, "the latest stored message end is reused")
            self.assertEqual(report.snapshots, [475, 545], "new message ends are stored as the run passes them")
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages"}):
            ref_token, ref_cache, _ = self.cold_reference(longer)        # same anchors, empty cache
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))
        self.assertTrue(all(width <= 128 for width in chunks.spans), chunks.spans)
        self.assertEqual(sum(chunks.spans), len(longer) - 255, "only the uncached suffix was prefilled")

    def test_a_partial_hit_at_an_interior_edge_continues_exactly_and_matches_cold_when_the_plans_agree(self):
        system = self.system_of(100, 3)
        body = [4] * 330
        prompt = system + body                                          # 430 tokens
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks", S.FLAG_DIR: str(self.tmp / "p")}):
            model, session = self.engine()
            _, _, cold = self.go(model, session, prompt)
            self.assertEqual(cold.snapshots[0], 100)
            self.assertEqual(cold.snapshots, [100, 210, 320], "the system boundary and the interior edges")
            # Same system message and same end, different tail: the plan is identical, the hit is the last edge.
            changed = system + body[:200] + [9] * 130
            token, cache, report = self.go(model, session, changed)
            self.assertGreater(report.hit_tokens, 100)
            self.assertLess(report.hit_tokens, 300)
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks"}):
            ref_token, ref_cache, _ = self.cold_reference(changed)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache),
                        "same end and anchors give the same plan, so a partial hit is bit-identical")

    def test_a_prefix_cut_differently_by_its_run_is_refused_and_the_fallback_is_bit_identical(self):
        """Interior edges of a plan depend on the prompt's end. A snapshot from a run with another end was
        computed with different chunk shapes, so it must not continue this prompt (provenance check)."""
        system = self.system_of(100, 3)
        other = system + [4] * 180 + [9] * 77                       # a different end
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks", S.FLAG_DIR: str(self.tmp / "q")}):
            model, session = self.engine()
            self.go(model, session, system + [4] * 330)
            interior = [n for n in session.store()._lengths if n > 100]
            self.assertTrue(interior, "the first run stored interior edges")
            token, cache, report = self.go(model, session, other)
            self.assertEqual(report.hit_tokens, 100, "only the anchor, which every plan cuts at, is reusable")
            self.assertGreaterEqual(report.plan_rejections, 1)
            self.assertGreaterEqual(session.store().stats.rejected_plan, 1)
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "chunks"}):
            ref_token, ref_cache, _ = self.cold_reference(other, "ref2")
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache), "the accepted fallback must match cold exactly")

    def test_the_policy_is_part_of_the_cache_key_and_grid_keeps_the_original_key(self):
        system = self.system_of(150, 9)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "k")}):
            model, session = self.engine()
            self.go(model, session, system + [7] * 50)
            with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CHUNK_POLICY": "grid"}):
                grid_session = S.PrefixSession(encode, compat)
                _, _, report = self.go(build_model(), grid_session, system + [8] * 50)
                self.assertEqual(report.hit_tokens, 0, "a grid prefix is not a balanced prefix")
                expected = P.fingerprint(
                    **compat(), prefix_cache=P.FORMAT_VERSION, snapshot_chain_semantics=2, chunk=128,
                    ring=os.environ.get("FLASHNEXT_PREFILL_RING_EXPERTS", "0"),
                    qsa_chunk=os.environ.get("FLASHNEXT_QSA_QUERY_CHUNK", "2048"))
                self.assertEqual(grid_session.store().fp, expected)
            self.assertNotEqual(session.store().fp, expected)
            with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages"}):
                self.assertNotEqual(S.PrefixSession(encode, compat).store().fp, session.store().fp)
            with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "100"}):
                self.assertNotEqual(S.PrefixSession(encode, compat).store().fp, session.store().fp)

    def test_cancelling_after_the_snapshot_keeps_it_valid_and_the_next_request_matches_cold(self):
        from veystrix.backends.base import GenerationCancelled

        system = self.system_of(200, 11)
        prompt = system + [6] * 90
        real = prefill._prefill_target

        def cancel(*args, **kwargs):
            raise GenerationCancelled

        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "x")}):
            model, session = self.engine()
            with mock.patch.object(prefill, "_prefill_target", cancel):
                with self.assertRaises(GenerationCancelled):
                    session.prefill(model, prompt, model.make_cache())
            files = sorted(p.name for p in session.store().directory.iterdir())
            self.assertEqual(len([f for f in files if f.endswith(".safetensors")]), 1, files)
            self.assertFalse([f for f in files if "tmp" in f or f.endswith(".partial")], files)
            token, cache, report = self.go(model, S.PrefixSession(encode, compat), system + [5] * 70)
            self.assertEqual(report.hit_tokens, 200)
        ref_token, ref_cache, _ = self.cold_reference(system + [5] * 70)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_cancelling_in_the_first_chunk_leaves_no_snapshot_and_no_stale_state(self):
        from veystrix.backends.base import GenerationCancelled

        system = self.system_of(200, 12)
        calls = []
        real = prefill._prefill_chunk

        def cancel_first(*args, **kwargs):
            calls.append(1)
            raise GenerationCancelled

        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "y")}):
            model, session = self.engine()
            with mock.patch.object(prefill, "_prefill_chunk", cancel_first):
                with self.assertRaises(GenerationCancelled):
                    session.prefill(model, system + [6] * 90, model.make_cache())
            self.assertEqual(len(calls), 1)
            self.assertEqual(list(session.store().directory.glob("*.safetensors")), [])
            token, cache, report = self.go(model, S.PrefixSession(encode, compat), system + [6] * 90)
            self.assertEqual(report.hit_tokens, 0)
        ref_token, ref_cache, _ = self.cold_reference(system + [6] * 90)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_a_cancel_between_chunks_does_not_disturb_a_stored_entry(self):
        from veystrix.backends.base import GenerationCancelled

        system = self.system_of(200, 13)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "z")}):
            model, session = self.engine()
            self.go(model, session, system + [6] * 90)
            store = session.store()
            before = {key: entry.checksum() if hasattr(entry, "checksum") else None for key, entry in store._ram.items()}
            files = {p.name: p.stat().st_size for p in (self.tmp / "z").iterdir()}
            real = prefill._prefill_chunk
            count = []

            def cancel_second(language, ids, cache):
                count.append(1)
                if len(count) == 2:
                    raise GenerationCancelled
                return real(language, ids, cache)

            with mock.patch.object(prefill, "_prefill_chunk", cancel_second):
                with self.assertRaises(GenerationCancelled):
                    session.prefill(model, system + [3] * 400, model.make_cache())
            self.assertEqual({p.name: p.stat().st_size for p in (self.tmp / "z").iterdir()}, files)
            after = {key: entry.checksum() if hasattr(entry, "checksum") else None for key, entry in store._ram.items()}
            self.assertEqual(before, after)

    def test_message_boundaries_respect_the_minimum_and_the_prompt_end(self):
        _, session = self.engine()
        prompt = [9, 9] + MARK + [8] * 20 + MARK + [7] * 5 + MARK + [6]
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages", S.FLAG_MIN_TOKENS: "8"}):
            self.assertEqual(session.message_boundaries(prompt), [4, 26, 33])
            self.assertEqual(session.anchors(prompt), (26, 33))
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages", S.FLAG_MIN_TOKENS: "30"}):
            self.assertEqual(session.anchors(prompt), (33,))
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "system", S.FLAG_MIN_TOKENS: "2"}):
            self.assertEqual(session.anchors(prompt), (4,))


class _Killed(Exception):
    """Stands for a cancelled request or a crash in the middle of a prefill."""


class _DiesAfter:
    """Proxy for the small model whose forward pass raises after ``after`` chunks (one call per chunk)."""

    def __init__(self, model, after):
        self._model, self._after, self.calls = model, after, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls > self._after:
            raise _Killed
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class ResumablePrefillTests(Base):
    """AGENTIC boundaries="all": message ends plus every chunk edge, so an interrupted cold prefill resumes."""

    def setUp(self):
        super().setUp()
        env = {"FLASHNEXT_PREFILL_CHUNK_POLICY": "balanced", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "128",
               S.FLAG_MIN_TOKENS: "8", S.FLAG_BOUNDARIES: "all"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def system_of(length, seed):
        mx.random.seed(seed)
        return [int(t) for t in mx.random.randint(5, VOCAB, (length - len(MARK),)).tolist()] + MARK

    def cold_reference(self, prompt):
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "reference")}):
            return self.go(build_model(), S.PrefixSession(encode, compat), prompt)

    def test_all_snapshots_every_message_end_and_every_chunk_edge(self):
        system = self.system_of(300, 1)
        prompt = system + [7] * 100
        model, session = self.engine()
        self.assertEqual(session.anchors(prompt), (300,))
        _, _, report = self.go(model, session, prompt)
        self.assertEqual(report.snapshots, [100, 200, 300], "three equal pieces of the system message, each stored")

    def test_failed_disk_spill_stops_additional_snapshots_over_ram_budget(self):
        prompt = self.system_of(300, 1) + [7] * 100
        with mock.patch.dict(os.environ, {S.FLAG_RAM_GB: "0.000000001"}), \
                mock.patch.object(P, "write_entry", side_effect=OSError("disk unavailable")):
            model, session = self.engine()
            _, _, report = self.go(model, session, prompt)
        self.assertEqual(report.snapshots, [100])
        self.assertEqual([row["reason"] for row in report.snapshots_skipped],
                         ["ram_budget_exceeded", "ram_budget_exceeded"])
        store = session.store()
        self.assertEqual(store.usage()["ram_entries"], 1)
        self.assertEqual(store.usage()["disk_entries"], 0)
        self.assertGreater(store.stats.writes_skipped, 0)

    def test_the_old_modes_are_unchanged(self):
        system = self.system_of(300, 1)
        prompt = system + [7] * 100
        for mode, expected in (("messages", [300]), ("system", [300])):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: mode, S.FLAG_DIR: str(self.tmp / mode)}):
                model, session = self.engine()
                _, _, report = self.go(model, session, prompt)
                self.assertEqual(report.snapshots, expected)

    def test_an_interrupted_prefill_resumes_at_its_last_edge_and_matches_a_cold_run_bit_for_bit(self):
        prompt = self.system_of(300, 1) + [7] * 100
        model, session = self.engine()
        dying = _DiesAfter(model, after=2)
        with self.assertRaises(_Killed):
            session.prefill(dying, prompt, dying.make_cache())
        self.assertEqual(dying.calls, 3)
        # A new process: new model object, new session over the same SSD directory.
        model2, session2 = self.engine()
        token, cache, report = self.go(model2, session2, prompt)
        self.assertEqual(report.hit_tokens, 200, "the work of the two finished chunks was kept")
        self.assertEqual(report.prefilled_tokens, len(prompt) - 200)
        ref_token, ref_cache, _ = self.cold_reference(prompt)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache), "a resumed prefill differs from an uninterrupted one")

    def test_a_different_first_question_after_the_same_system_prompt_also_resumes_from_its_edges(self):
        system = self.system_of(300, 1)
        model, session = self.engine()
        dying = _DiesAfter(model, after=2)
        with self.assertRaises(_Killed):
            session.prefill(dying, system + [7] * 100, dying.make_cache())
        model2, session2 = self.engine()
        question = [9] * 45
        token, cache, report = self.go(model2, session2, system + question)
        self.assertEqual(report.hit_tokens, 200)
        ref_token, ref_cache, _ = self.cold_reference(system + question)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_all_and_messages_share_one_fingerprint_so_stored_entries_survive_the_switch(self):
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages"}):
            before = S.PrefixSession(encode, compat).store().fp
        after = S.PrefixSession(encode, compat).store().fp
        self.assertEqual(before, after)
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "system"}):
            self.assertNotEqual(before, S.PrefixSession(encode, compat).store().fp, "system mode plans differently")

    def test_entries_written_under_messages_are_found_under_all(self):
        prompt = self.system_of(300, 1) + [7] * 100
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages"}):
            model, session = self.engine()
            self.go(model, session, prompt)
        model2, session2 = self.engine()
        _, _, report = self.go(model2, session2, self.system_of(300, 1) + [5] * 60)
        self.assertEqual(report.hit_tokens, 300)


class EffortTurnSharingTests(Base):
    """The effort sentence is its own system turn after the system message: every effort shares the whole system prompt."""

    LOW = [201 + i for i in range(18)] + MARK          # a "<|im_start|>system ... <|im_end|>\n" turn, as tokens
    XHIGH = [221 + i for i in range(30)] + MARK

    def setUp(self):
        super().setUp()
        env = {"FLASHNEXT_PREFILL_CHUNK_POLICY": "balanced", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "128",
               S.FLAG_MIN_TOKENS: "8", S.FLAG_BOUNDARIES: "all"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        mx.random.seed(11)
        self.content = [int(t) for t in mx.random.randint(5, VOCAB, (300,)).tolist()] + MARK     # system message, ends at 302
        self.user = [int(t) for t in mx.random.randint(5, VOCAB, (60,)).tolist()]

    def prompt(self, turn=(), user=None):
        return self.content + list(turn) + (user or self.user)

    def cold_reference(self, prompt):
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "reference")}):
            return self.go(build_model(), S.PrefixSession(encode, compat), prompt)

    def test_the_system_end_is_an_anchor_for_every_effort_and_the_shared_plan_is_identical(self):
        _, session = self.engine()
        medium = session.cold_edges(self.prompt())
        for turn in (self.LOW, self.XHIGH):
            self.assertIn(302, session.anchors(self.prompt(turn)))
            edges = session.cold_edges(self.prompt(turn))
            self.assertEqual([e for e in edges if e <= 302], [e for e in medium if e <= 302])

    def test_low_restores_the_whole_system_prompt_medium_stored_and_matches_a_cold_run_bit_for_bit(self):
        model, session = self.engine()
        self.go(model, session, self.prompt())
        model2, session2 = self.engine()                      # a restart: only the SSD tier remains
        low = self.prompt(self.LOW, user=[9] * 41)
        token, cache, report = self.go(model2, session2, low)
        self.assertEqual(report.hit_tokens, 302, "the entire system message comes from the cache")
        self.assertEqual(report.prefilled_tokens, len(low) - 302)
        ref_token, ref_cache, _ = self.cold_reference(low)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache), "state restored from another effort differs from a cold run")

    def test_every_order_of_efforts_shares_the_system_prompt_and_equals_cold(self):
        steps = (("low", self.LOW, [4] * 31), ("xhigh", self.XHIGH, [5] * 47), ("medium", (), [6] * 52), ("low again", self.LOW, [7] * 29))
        model, session = self.engine()
        for index, (name, turn, user) in enumerate(steps):
            with self.subTest(step=name):
                prompt = self.prompt(turn, user=user)
                token, cache, report = self.go(model, session, prompt)
                if index:
                    self.assertGreaterEqual(report.hit_tokens, 302, "the shared system prompt must come from the cache")
                ref_token, ref_cache, _ = self.cold_reference(prompt)
                self.assertEqual(token, ref_token)
                self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_the_same_effort_in_a_new_session_restores_through_its_effort_turn(self):
        model, session = self.engine()
        self.go(model, session, self.prompt(self.LOW))
        model2, session2 = self.engine()
        _, _, report = self.go(model2, session2, self.prompt(self.LOW, user=[3] * 33))
        self.assertEqual(report.hit_tokens, 302 + len(self.LOW), "system end plus the effort turn")

    def test_medium_prompts_are_planned_exactly_as_before_so_existing_entries_stay_valid(self):
        _, session = self.engine()
        before = session.cold_edges(self.prompt())
        self.assertEqual(session.anchors(self.prompt()), (302,))
        with mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages"}):
            self.assertEqual(session.cold_edges(self.prompt()), before, "all vs messages differ only in extra snapshots")


class ProvenanceSessionTests(Base):
    """A prefix is reused only when its chunk history is the one this prompt's cold run would have made."""

    def system_of(self, length, seed):
        mx.random.seed(seed)
        return [int(t) for t in mx.random.randint(5, VOCAB, (length - len(MARK),)).tolist()] + MARK

    def cold_reference(self, prompt, where):
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / where)}):
            return self.go(build_model(), S.PrefixSession(encode, compat), prompt)

    def test_compatible_is_exact(self):
        c = S.PrefixSession.compatible
        self.assertTrue(c((64, 100, 128), 100, (64,)))
        self.assertTrue(c((100,), 100, ()))
        self.assertFalse(c((64, 128), 100, (64,)), "the cold plan does not cut at 100")
        self.assertFalse(c((64, 100), 100, ()), "the stored prefix skipped the cut at 64")
        self.assertFalse(c((100,), 100, (64,)), "the stored prefix cut where this plan does not")
        self.assertFalse(c((64, 100), 100, None), "unknown history is never compatible")
        self.assertFalse(c((), 100, ()), "n_tokens must itself be an edge")

    def test_a_snapshot_from_a_run_with_other_anchors_is_refused_even_under_the_grid_key(self):
        """The grid cache key has no minimum-token setting, but the minimum decides which boundaries cut."""
        system = self.system_of(100, 21)
        tail_a, tail_b = [7] * 60, [8] * 60
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "g"), S.FLAG_MIN_TOKENS: "8"}):
            _, _, first = self.go(build_model(), S.PrefixSession(encode, compat), system + tail_a)
            self.assertEqual(first.snapshots, [100])
            self.assertEqual(first.plan_edges, [64, 100])
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "g"), S.FLAG_MIN_TOKENS: "300"}):
            token, cache, report = self.go(build_model(), S.PrefixSession(encode, compat), system + tail_b)
            self.assertEqual(report.hit_tokens, 0, "that prefix was executed with an extra cut at 100")
            self.assertEqual(report.plan_rejections, 1)
        with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "300"}):
            ref_token, ref_cache, _ = self.cold_reference(system + tail_b, "ref_g2")
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_restore_off_keeps_the_plan_but_neither_restores_nor_snapshots(self):
        """The same-plan uncached reference: identical chunk edges and bit-identical result to a cold cached run."""
        system = self.system_of(100, 23)
        prompt = system + [8] * 61
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "r"), S.FLAG_MIN_TOKENS: "8"}):
            self.go(build_model(), S.PrefixSession(encode, compat), system + [7] * 60)       # a snapshot exists
            self.assertTrue(S.restore_enabled())
            with mock.patch.dict(os.environ, {S.FLAG_RESTORE: "0"}):
                self.assertFalse(S.restore_enabled())
                token, cache, report = self.go(build_model(), S.PrefixSession(encode, compat), prompt)
            self.assertEqual((report.hit_tokens, report.snapshots, report.prefilled_tokens), (0, [], len(prompt)))
        with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "8"}):
            ref_token, ref_cache, ref = self.cold_reference(prompt, "r_cold")
        self.assertEqual(report.plan_edges, ref.plan_edges)
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_reusable_tokens_predicts_the_hit_and_is_side_effect_free(self):
        system = self.system_of(100, 24)
        prompt = system + [8] * 61
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "u"), S.FLAG_MIN_TOKENS: "8"}):
            session = S.PrefixSession(encode, compat)
            self.assertEqual(session.reusable_tokens(prompt), 0, "nothing stored yet")
            self.go(build_model(), session, system + [7] * 60)
            stats = dict(session.store().stats.__dict__)
            self.assertEqual(session.reusable_tokens(prompt), 100)
            self.assertEqual(session.store().stats.__dict__, stats)
            _, _, report = self.go(build_model(), session, prompt)
            self.assertEqual(report.hit_tokens, 100, "the prediction was the real hit")
            with mock.patch.dict(os.environ, {S.FLAG_RESTORE: "0"}):
                self.assertEqual(session.reusable_tokens(prompt), 0, "restore off reuses nothing")
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "u"), S.FLAG_MIN_TOKENS: "300"}):
            other_plan = S.PrefixSession(encode, compat)         # another minimum: the stored history is not this plan's
            self.assertEqual(other_plan.reusable_tokens(prompt), 0)

    def test_a_matching_history_is_reused_after_a_restart_and_equals_cold(self):
        system = self.system_of(100, 22)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "h"), S.FLAG_MIN_TOKENS: "8"}):
            self.go(build_model(), S.PrefixSession(encode, compat), system + [7] * 60)
            token, cache, report = self.go(build_model(), S.PrefixSession(encode, compat), system + [8] * 61)
            self.assertEqual((report.hit_tokens, report.tier, report.plan_rejections), (100, "ssd", 0))
        with mock.patch.dict(os.environ, {S.FLAG_MIN_TOKENS: "8"}):
            ref_token, ref_cache, _ = self.cold_reference(system + [8] * 61, "ref_h")
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_the_stored_provenance_is_the_cold_plan_before_each_snapshot(self):
        system = self.system_of(100, 23)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "p"), S.FLAG_MIN_TOKENS: "8",
                                          S.FLAG_BOUNDARIES: "chunks"}):
            model, session = self.engine()
            _, _, report = self.go(model, session, system + [7] * 200)
            store = session.store()
            for position in report.snapshots:
                expected = tuple(e for e in report.plan_edges if e < position)
                self.assertEqual(store._plans[next(k for k, t in store._tokens.items() if len(t) == position)], expected)

    def test_a_restored_run_stores_provenance_that_equals_the_cold_runs_for_its_own_snapshots(self):
        system = self.system_of(100, 24)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(self.tmp / "c2"), S.FLAG_MIN_TOKENS: "8",
                                          S.FLAG_BOUNDARIES: "chunks"}):
            model, session = self.engine()
            self.go(model, session, system + [7] * 40)                    # stores the anchor at 100
            _, _, report = self.go(model, session, system + [9] * 300)    # restored at 100, snapshots on
            self.assertEqual(report.hit_tokens, 100)
            store = session.store()
            for position in report.snapshots:
                key = next(k for k, t in store._tokens.items() if len(t) == position)
                self.assertEqual(store._plans[key], tuple(e for e in report.plan_edges if e < position))


if __name__ == "__main__":
    unittest.main()
