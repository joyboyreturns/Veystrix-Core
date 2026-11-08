"""Budgeted chunk plan, live step-down provenance, strict KV settings, and quantized KV through the real attention path.

The model is the small real MLX model of test_prefix_cache (same code path as the checkpoint, tiny weights). What this
proves is plumbing and exactness of state handling. It says nothing about the quality of a quantized KV cache on the real
checkpoint: that needs real KV statistics (see docs/flashnext/long-context-agentic-audit-2026-11-08.md).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

from models.flashnext import prefill, prefill_scheduler as PS, prefix_cache as P, prefix_session as S, quantized_kv as QKV
from models.flashnext.prefill_budget import chunk_increment_bytes
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits

MARK = [3, 4]
BUDGET = int(1000e6)


def compat():
    return {"model": "small", "tokenizer": "t", "engine": "e", "profile": {}}


def encode(text):
    assert text == S.IM_END
    return list(MARK)


class PlanTests(unittest.TestCase):
    def test_chunks_are_candidates_and_shrink_as_context_grows(self):
        edges = prefill.budgeted_edges(12288, 2048, BUDGET)
        sizes = [b - a for a, b in zip([0] + edges, edges + [12288])]
        self.assertTrue(set(sizes) <= set(prefill.CANDIDATE_CHUNKS), sizes)
        self.assertEqual(sizes[0], 2048)
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        position = 0
        for size in sizes:
            self.assertLessEqual(chunk_increment_bytes(size, position), BUDGET)
            position += size

    def test_a_smaller_budget_gives_smaller_or_equal_chunks(self):
        big = prefill.budgeted_edges(12288, 2048, int(1500e6))
        small = prefill.budgeted_edges(12288, 2048, int(600e6))
        self.assertGreater(len(small), len(big))

    def test_plan_before_the_end_does_not_depend_on_the_end(self):
        """What makes a cold run and a restored run take the same chunks: the edges before a point are a function of
        the budget and anchors only (the last edge may be merged into a short tail)."""
        long_edges = prefill.budgeted_edges(12288, 2048, BUDGET, anchors=(900,))
        short_edges = prefill.budgeted_edges(5000, 2048, BUDGET, anchors=(900,))
        common = [e for e in short_edges if e < 5000 - prefill.MIN_CHUNK_TOKENS]
        self.assertTrue(common)
        self.assertEqual([e for e in long_edges if e <= common[-1]], common)

    def test_anchor_is_always_an_edge_and_restored_tail_matches_cold(self):
        with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_POLICY: "budgeted"}):
            cold = prefill.plan_chunks(9000, 2048, anchors=(3100,))
            self.assertIn(3100, [b for _, b in cold])
            edge = next(b for _, b in cold if b > 3100)
            restored = prefill.plan_chunks(9000 - edge, 2048, offset=edge, anchors=(3100,))
            self.assertEqual([(a + edge, b + edge) for a, b in restored], [s for s in cold if s[0] >= edge])

    def test_tiny_tail_is_merged_unless_it_is_an_anchor(self):
        edges = prefill.budgeted_edges(2048 + 10, 2048, BUDGET)
        self.assertEqual(edges, [])
        edges = prefill.budgeted_edges(2048 + 10, 2048, BUDGET, anchors=(2048,))
        self.assertEqual(edges, [2048])

    def test_bad_budget_and_policy_raise(self):
        with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_BUDGET_MB: "12"}):
            with self.assertRaises(ValueError):
                prefill.chunk_budget_bytes()
        with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_POLICY: "elastic"}):
            with self.assertRaises(ValueError):
                prefill.chunk_policy()


class SchedulerTests(unittest.TestCase):
    def test_in_run_estimate_does_not_recount_resident_weights(self):
        """Regression: the first version subtracted weights + ring (4.4 GB, already resident) from host availability, so a
        loaded process with 6 GiB available always fell to the smallest chunk."""
        sched = PS.AdaptivePrefillScheduler(system_reader=lambda: {"available": 6 * 2 ** 30})
        self.assertEqual(sched.select_initial_chunk(4096), 2048)
        old = PS.estimate_chunk_memory(2048, 4096, 6 * 2 ** 30)
        self.assertFalse(old.fits_comfortably)          # the pre-load view is stricter, and only for admission

    def test_unreadable_machine_adapts_nothing(self):
        def broken():
            raise OSError("vm_stat failed")
        sched = PS.AdaptivePrefillScheduler(system_reader=broken)
        self.assertIsNone(sched.read_system())
        self.assertIsNone(sched.observe())

    def test_worst_case_bytes_grow_with_context(self):
        self.assertGreater(PS.plan_worst_case_bytes([], 4096), PS.plan_worst_case_bytes([], 1024))


class RealModelBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(21)
        body = [int(t) for t in mx.random.randint(5, VOCAB, (3200,)).tolist()]
        cls.system = body[:150] + MARK                  # 152 tokens
        cls.tail_a = body[150:2950]
        cls.tail_b = body[1000:3200]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = {S.FLAG: "1", S.FLAG_DIR: str(self.tmp / "cache"), S.FLAG_RAM_GB: "1", S.FLAG_DISK_GB: "1",
               S.FLAG_MIN_TOKENS: "64", prefill.FLAG_CHUNK_TOKENS: "2048", prefill.FLAG_CHUNK_POLICY: "budgeted",
               prefill.FLAG_CHUNK_BUDGET_MB: "4000"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def go(self, model, session, prompt):
        token, cache, report = session.prefill(model, prompt, model.make_cache())
        mx.eval(token)
        return int(token.item()), cache, report


class BudgetedRealModelTests(RealModelBase):
    def test_cold_and_restored_runs_are_bit_identical_under_the_budgeted_plan(self):
        model = build_model()
        session = S.PrefixSession(encode, compat)
        prompt = self.system + self.tail_a
        _, _, first = self.go(model, session, prompt)
        self.assertEqual(first.snapshots, [152])
        self.assertEqual(first.executed_edges, first.plan_edges)
        self.assertFalse(first.plan_deviated)
        token, cache, report = self.go(model, session, self.system + self.tail_b)
        self.assertEqual((report.hit_tokens, report.tier), (152, "ram"))
        other = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, other, True)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(other)}):
            cold_token, cold_cache, cold_report = self.go(build_model(), S.PrefixSession(encode, compat),
                                                          self.system + self.tail_b)
        self.assertEqual(cold_report.hit_tokens, 0)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(cache) == cache_bits(cold_cache), "restored run differs from the cold run")
        self.assertEqual(report.plan_edges, cold_report.plan_edges)

    def test_chunk_trace_has_one_row_per_chunk_with_positions(self):
        model = build_model()
        _, _, report = self.go(model, S.PrefixSession(encode, compat), self.system + self.tail_a)
        rows = report.chunk_trace
        self.assertEqual(len(rows), len(report.executed_edges) + 1)
        self.assertEqual(rows[0]["start"], 0)
        self.assertEqual(rows[-1]["end"], len(self.system + self.tail_a))
        self.assertTrue(all(r["seconds"] > 0 for r in rows))

    def test_budget_is_part_of_the_cache_key(self):
        a = S.PrefixSession(encode, compat).store()
        with mock.patch.dict(os.environ, {prefill.FLAG_CHUNK_BUDGET_MB: "800"}):
            b = S.PrefixSession(encode, compat).store()
        self.assertNotEqual(a.fingerprint if hasattr(a, "fingerprint") else a.fp, b.fingerprint if hasattr(b, "fingerprint") else b.fp)

    def test_live_stepdown_replans_records_it_and_poisons_no_snapshot(self):
        model = build_model()
        session = S.PrefixSession(encode, compat)
        prompt = self.system + self.tail_a
        low = {"available": 1 * 2 ** 30, "pressure": 4}          # critical: the only reading that shrinks chunks
        with mock.patch.dict(os.environ, {PS.FLAG_ADAPTIVE: "1"}), \
                mock.patch.object(PS.AdaptivePrefillScheduler, "read_system", lambda self: dict(low)):
            token, cache, report = self.go(model, session, prompt)
        self.assertTrue(report.plan_deviated)
        self.assertTrue(report.stepdowns)
        self.assertNotEqual(report.executed_edges, report.plan_edges)
        self.assertEqual(report.stepdowns[0]["cap_to"], 1024)
        # The first snapshot (system boundary) was taken before any deviation: it follows the cold plan exactly.
        self.assertEqual(report.snapshots[0], 152)
        # The run still produced a valid token and a cache holding the whole prompt.
        self.assertEqual(int(cache[3].offset if hasattr(cache[3], "offset") else cache[7].offset), len(prompt))


class MessageBoundaryTests(RealModelBase):
    """system / user / tool-result boundaries under the budgeted plan: snapshots at each, restore from the deepest is exact."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages", prefill.FLAG_CHUNK_BUDGET_MB: "300"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def conversation(self, tool_len):
        body = [int(t) for t in mx.random.randint(5, VOCAB, (5000,)).tolist()]
        system = body[:300] + MARK                       # system message
        user = body[300:900] + MARK                      # user message
        tool = body[900:900 + tool_len] + MARK           # tool result
        return system, user, tool, body[4000:4100]

    def test_snapshots_at_each_message_boundary_and_exact_restore_from_the_deepest(self):
        mx.random.seed(33)
        system, user, tool, tail = self.conversation(700)
        model, session = build_model(), S.PrefixSession(encode, compat)
        first = system + user + tool + tail
        _, _, report = self.go(model, session, first)
        boundaries = [len(system), len(system) + len(user), len(system) + len(user) + len(tool)]
        self.assertEqual(report.snapshots, boundaries)
        for boundary in boundaries:                      # a chunk edge sits exactly on every message boundary
            self.assertIn(boundary, report.executed_edges)
        self.assertEqual(report.executed_edges, report.plan_edges)
        # the next request: same conversation plus a new assistant/tool exchange after the tool result
        longer = system + user + tool + tail + [int(t) for t in mx.random.randint(5, VOCAB, (400,)).tolist()]
        token, cache, again = self.go(model, session, longer)
        self.assertGreaterEqual(again.hit_tokens, boundaries[-1])
        other = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, other, True)
        with mock.patch.dict(os.environ, {S.FLAG_DIR: str(other)}):
            cold_token, cold_cache, cold = self.go(build_model(), S.PrefixSession(encode, compat), longer)
        self.assertEqual(cold.hit_tokens, 0)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(cache) == cache_bits(cold_cache), "restore at a message boundary differs from cold")


class QuantizedKVTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(QKV.restore)

    def test_settings_are_strict(self):
        for name, value in ((QKV.FLAG_KV_DTYPE, "affine5"), (QKV.FLAG_KV_GROUP_SIZE, "48")):
            with mock.patch.dict(os.environ, {QKV.FLAG_KV_DTYPE: "affine4", name: value}):
                with self.assertRaises(ValueError, msg=name):
                    QKV.fingerprint_parts()
        with mock.patch.dict(os.environ, {QKV.FLAG_KV_DTYPE: "affine4", QKV.FLAG_KV_BITS: "8"}):
            with self.assertRaises(ValueError):
                QKV.bits()

    def test_fingerprint_separates_dtype_bits_and_group_size_and_leaves_bf16_alone(self):
        def fp(**env):
            with mock.patch.dict(os.environ, env):
                return P.fingerprint(model="m", **QKV.fingerprint_parts())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(QKV.FLAG_KV_DTYPE, None)
            self.assertEqual(QKV.fingerprint_parts(), {})
        base = fp(**{QKV.FLAG_KV_DTYPE: "affine4"})
        self.assertNotEqual(base, fp(**{QKV.FLAG_KV_DTYPE: "affine4", QKV.FLAG_KV_GROUP_SIZE: "32"}))
        self.assertNotEqual(base, fp(**{QKV.FLAG_KV_DTYPE: "affine8"}))
        self.assertNotEqual(base, fp(**{QKV.FLAG_KV_DTYPE: "bf16"}))

    def _run(self, dtype, ids):
        env = {QKV.FLAG_KV_DTYPE: dtype}
        with mock.patch.dict(os.environ, env):
            QKV.apply()
            model = build_model()
            cache = model.make_cache()
            chunks = [ids[:, i:i + 200] for i in range(0, ids.shape[1], 200)]      # chunked prefill, then last logits
            for part in chunks[:-1]:
                model(part, cache=cache)
                mx.eval([c.state for c in cache])
            out = model(chunks[-1], cache=cache)
            mx.eval(out.logits)
            return model, cache, out.logits[:, -1, :].astype(mx.float32)

    def test_quantized_cache_is_the_one_the_attention_consumes_and_no_bf16_copy_is_kept(self):
        mx.random.seed(2)
        ids = mx.random.randint(5, VOCAB, (1, 640))
        model, cache, _ = self._run("affine4", ids)
        attention = [c for c in cache if hasattr(c, "bits")]
        self.assertEqual(len(attention), 2)                       # tiny model: 8 layers, every 4th is full attention
        for c in attention:
            self.assertEqual((c.bits, c.group_size), (4, 64))
            self.assertIsInstance(c.keys, tuple)
            self.assertEqual(c.keys[0].dtype, mx.uint32)          # packed data, not a bf16 tensor
            self.assertEqual(c.offset, 640)
        bf_model, bf_cache, _ = self._run("bf16", ids)
        q_bytes = sum(c.nbytes for c in attention)
        b_bytes = sum(c.nbytes for c in bf_cache if hasattr(c, "index_keys"))
        self.assertLess(q_bytes, b_bytes)

    def test_affine8_tracks_bf16_closely_affine4_less_so_on_the_tiny_model(self):
        mx.random.seed(3)
        ids = mx.random.randint(5, VOCAB, (1, 640))
        _, _, ref = self._run("bf16", ids)

        def cos(x):
            return float(mx.sum(ref * x) / (mx.linalg.norm(ref) * mx.linalg.norm(x)))
        c8 = cos(self._run("affine8", ids)[2])
        c4 = cos(self._run("affine4", ids)[2])
        print(f"\n[tiny model last-token logits cosine vs bf16: affine8={c8:.5f} affine4={c4:.5f}]")
        self.assertGreater(c8, 0.99)
        self.assertGreaterEqual(c8, c4)

    def test_quantized_snapshot_round_trip_through_the_real_model_is_exact(self):
        mx.random.seed(4)
        ids = mx.random.randint(5, VOCAB, (1, 400))
        tokens = [int(t) for t in ids[0].tolist()]
        with mock.patch.dict(os.environ, {QKV.FLAG_KV_DTYPE: "affine4"}):
            QKV.apply()
            model = build_model()
            cache = model.make_cache()
            mx.eval(model(ids, cache=cache).logits, [c.state for c in cache])
            fp = P.fingerprint(model="m", **QKV.fingerprint_parts())
            entry = P.capture(cache, tokens, fp, language=model)
            tmp = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, tmp, True)
            P.write_entry(tmp, entry, fp)
            loaded = P.read_entry(tmp, entry.key, fp)
            restored = P.restore([loaded], model.make_cache)
            more = mx.random.randint(5, VOCAB, (1, 50))
            a = model(more, cache=cache).logits
            b = model(more, cache=restored).logits
            mx.eval(a, b)
            self.assertTrue(bool(mx.array_equal(a, b)), "continuation from a restored quantized snapshot differs")
            # A snapshot made with another group size cannot be read back under this fingerprint.
            with mock.patch.dict(os.environ, {QKV.FLAG_KV_GROUP_SIZE: "32"}):
                other = P.fingerprint(model="m", **QKV.fingerprint_parts())
            self.assertNotEqual(fp, other)
            with self.assertRaises(Exception):
                P.read_entry(tmp, entry.key, other)


if __name__ == "__main__":
    unittest.main()
