"""Prefetch, parallel reads, auto pool and the persistent prefix cache, on tiny synthetic checkpoints."""
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import mlx.core as mx

from models.nemotron35 import admission
from models.nemotron35.prefix_cache import PrefixCache
from models.nemotron35.streamed_model import load_streamed
from models.nemotron35.tests.unit.test_backend import StubTokenizer, machine
from veystrix.backends.nemotron35 import NemotronBackend
from models.nemotron35.tiny_checkpoint import TINY_CONFIG, permissive_config_check, write_checkpoint
from veystrix.backends.base import GenerationCancelled

_dir = None


def setUpModule():
    global _dir
    _dir = tempfile.mkdtemp(prefix="nemotron-speed-")
    write_checkpoint(Path(_dir) / "ck")


def tearDownModule():
    shutil.rmtree(_dir, ignore_errors=True)


def make(**kw):
    args = dict(authorized=True, config_check=permissive_config_check, machine=machine(), tokenizer=StubTokenizer(),
                pool_bytes=64 * 1024, chunk_tokens=4, session_dir=str(Path(_dir) / "sessions"),
                prefix_dir=str(Path(_dir) / "prefix"), snapshot_tokens=8, layer_major_tokens=8)
    args.update(kw)
    backend = NemotronBackend(Path(_dir) / "ck", **args)
    backend._current_machine = lambda: machine()
    return backend


def f32(x):
    return x.astype(mx.float32)


class PrefetchTests(unittest.TestCase):
    def load(self, **kw):
        return load_streamed(Path(_dir) / "ck", pool_bytes=kw.pop("pool_bytes", 64 * 1024),
                             config_check=permissive_config_check, **kw)

    def test_prefetch_never_changes_results(self):
        tokens = mx.array([[1, 5, 9, 3, 7, 2, 11, 4, 6, 8]])
        with_pf, without = self.load(prefetch=True), self.load(prefetch=False)
        try:
            a, b = f32(with_pf.model(tokens)), f32(without.model(tokens))
            self.assertTrue(bool(mx.all(a == b)))
            c1, c2 = with_pf.model.make_cache(), without.model.make_cache()
            for t in (3, 9, 1, 5, 7):
                x = mx.array([[t]])
                self.assertTrue(bool(mx.all(f32(with_pf.model(x, cache=c1)) == f32(without.model(x, cache=c2)))))
        finally:
            with_pf.close()
            without.close()

    def test_prefetch_is_issued_and_consumed_and_staging_is_bounded(self):
        s = self.load(pool_bytes=0, staged_bytes=64 * 1024)
        try:
            cache = s.model.make_cache()
            for t in range(24):
                s.model(mx.array([[t, (t * 5) % 64]]), cache=cache)
            st = s.pool.stats()
            self.assertGreater(st["prefetch_issued"], 0)
            self.assertGreater(st["prefetch_used"], 0)
            self.assertLessEqual(st["staged_bytes"], 64 * 1024)
            self.assertEqual(st["resident_bytes"], 0)       # pool of 0: nothing is retained, prefetch still works
        finally:
            s.close()

    def test_parallel_reads_return_every_expert(self):
        s = self.load(pool_bytes=0)
        try:
            got = s.pool.fetch(1, list(range(8)))
            self.assertEqual(sorted(got), list(range(8)))
            self.assertTrue(all(set(v) == {"fc1.weight", "fc1.scales", "fc1.biases", "fc2.weight", "fc2.scales", "fc2.biases"}
                                for v in got.values()))
        finally:
            s.close()


class AutoPoolTests(unittest.TestCase):
    cfg = dict(TINY_CONFIG, max_position_embeddings=262144)

    def test_auto_pool_uses_room_but_keeps_headroom(self):
        plan = admission.make_plan(self.cfg, machine(), dense_bytes=1_252_009_856, pool_bytes=None)
        self.assertGreater(plan.pool_bytes, 0)
        self.assertLessEqual(plan.pool_bytes, admission.AUTO_POOL_MAX)
        self.assertLessEqual(plan.estimate.total_bytes + admission.AUTO_POOL_HEADROOM, plan.budget_bytes + 1)

    def test_auto_pool_shrinks_when_swap_is_already_in_use(self):
        gib = 2**30
        sizes = [admission.make_plan(self.cfg, machine(swap_used=int(g * gib)), dense_bytes=1_252_009_856,
                                     pool_bytes=None).pool_bytes for g in (0, 1.2, 2.2)]
        self.assertEqual(sizes[0], admission.AUTO_POOL_MAX)
        self.assertLessEqual(sizes[1], gib)
        self.assertLessEqual(sizes[2], gib // 4)
        explicit = admission.make_plan(self.cfg, machine(swap_used=int(2.2 * gib)), dense_bytes=1_252_009_856,
                                       pool_bytes=2 * gib).pool_bytes
        self.assertEqual(explicit, 2 * gib)       # an explicit size is never overridden

    def test_auto_pool_is_zero_when_memory_is_tight(self):
        tight = machine(available=int(2.6 * 2**30))
        plan = admission.make_plan(self.cfg, tight, dense_bytes=1_252_009_856, pool_bytes=None)
        self.assertEqual(plan.pool_bytes, 0)

    def test_env_auto_and_bounds(self):
        self.assertIsNone(admission.pool_bytes_from_env({}))
        self.assertIsNone(admission.pool_bytes_from_env({"VEYSTRIX_NEMOTRON_POOL_GB": "auto"}))
        self.assertEqual(admission.pool_bytes_from_env({"VEYSTRIX_NEMOTRON_POOL_GB": "3"}), 3 * 2**30)
        with self.assertRaises(ValueError):
            admission.pool_bytes_from_env({"VEYSTRIX_NEMOTRON_POOL_GB": "9"})


class PrefixCacheTests(unittest.TestCase):
    PROMPT = list(range(4, 60)) * 2      # 112 tokens, chunk 4 -> several checkpoints at snapshot_tokens=8

    def setUp(self):
        shutil.rmtree(Path(_dir) / "prefix", ignore_errors=True)

    def backend(self, **kw):
        b = make(**kw)
        if b.prefix_cache is not None:
            b.prefix_cache.min_tokens = 8
        return b

    def cold_reference(self, tail=()):
        b = make(prefix_cache=False)
        try:
            b.append_tokens(self.PROMPT + list(tail))
            b.generate(5)
            return b.tape[len(self.PROMPT) + len(tail):]
        finally:
            b.close()

    def test_restart_resumes_from_snapshot_and_matches_cold_run(self):
        a = self.backend()
        a.append_tokens(self.PROMPT)
        a.generate(3)
        a.close()
        b = self.backend()                         # a "restart": new process state, same cache directory
        try:
            tail = [7, 8, 9, 10]
            b.append_tokens(self.PROMPT + tail)
            _, stats = b.generate(5)
            self.assertGreaterEqual(stats.reused_tokens, len(self.PROMPT) - 8)
            self.assertEqual(stats.prompt_tokens, len(self.PROMPT) + len(tail))
            self.assertEqual(b.tape[len(self.PROMPT) + len(tail):], self.cold_reference(tail))
            self.assertTrue(b.check_invariant())
        finally:
            b.close()

    def test_cancelled_prefill_resumes_from_its_last_checkpoint(self):
        a = self.backend()
        seen = []
        a.append_tokens(self.PROMPT)
        with self.assertRaises(GenerationCancelled):
            a.generate(3, on_prefill_progress=lambda d, t: seen.append(d), should_cancel=lambda: bool(seen) and seen[-1] >= 16)
        a.close()
        self.assertTrue(any(m["tokens"] >= 8 for m in a.prefix_cache._disk.values()))
        b = self.backend()
        try:
            b.append_tokens(self.PROMPT)
            _, stats = b.generate(5)
            self.assertGreater(stats.reused_tokens, 0)
            self.assertEqual(b.tape[len(self.PROMPT):], self.cold_reference())
        finally:
            b.close()

    def test_corrupt_snapshot_is_rejected_deleted_and_cold_run_is_correct(self):
        a = self.backend()
        a.append_tokens(self.PROMPT)
        a.generate(2)
        files = list(a.prefix_cache.directory.glob("*.safetensors"))
        a.close()
        self.assertTrue(files)
        for path in files:
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(bytes(data))
        b = self.backend()
        try:
            b.append_tokens(self.PROMPT + [5, 6])
            _, stats = b.generate(5)
            self.assertEqual(stats.reused_tokens, 0)
            self.assertGreater(b.prefix_cache.stats.restore_failures, 0)
            self.assertEqual(b.tape[len(self.PROMPT) + 2:], self.cold_reference([5, 6]))
        finally:
            b.close()

    def test_diverging_prompt_reuses_only_a_true_prefix(self):
        a = self.backend()
        a.append_tokens(self.PROMPT)
        a.generate(2)
        a.close()
        other = list(self.PROMPT)
        other[30] = 63                              # differs inside the snapshots that cover token 30
        b = self.backend()
        try:
            b.append_tokens(other)
            _, stats = b.generate(3)
            self.assertLessEqual(stats.reused_tokens, 30)
            ref = make(prefix_cache=False)
            ref.append_tokens(other)
            ref.generate(3)
            self.assertEqual(b.tape, ref.tape)
            ref.close()
        finally:
            b.close()

    def test_other_model_identity_cannot_read_the_cache(self):
        a = self.backend()
        a.append_tokens(self.PROMPT)
        a.generate(2)
        a.close()
        foreign = PrefixCache(Path(_dir) / "prefix", {"model": "something-else"}, quota_bytes=2**30, min_tokens=8)
        self.assertEqual(foreign.reusable_tokens(self.PROMPT + [1]), 0)

    def test_oversize_snapshot_never_exceeds_global_quota(self):
        a = self.backend(prefix_quota_gb=0.00005)   # ~50 KB: smaller than one snapshot
        a.append_tokens(self.PROMPT)
        a.generate(2)
        self.assertEqual(len(a.prefix_cache._disk), 0)
        a.close()

    def test_server_facing_store_fields(self):
        a = self.backend()
        try:
            store = a._prefix_session().store()
            for attribute in ("stats", "_disk", "disk_budget", "ram_budget", "_ram", "_ram_bytes", "directory"):
                self.assertTrue(hasattr(store, attribute), attribute)
        finally:
            a.close()

    def test_disabled_cache_has_no_prefix_session(self):
        a = make(prefix_cache=False)
        try:
            self.assertFalse(hasattr(a, "_prefix_session"))
        finally:
            a.close()


if __name__ == "__main__":
    unittest.main()


class LayerMajorTests(unittest.TestCase):
    TOKENS = [1, 5, 9, 3, 7, 2, 11, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    def core(self, layer_major, pool_bytes=64 * 1024, chunk=4):
        from models.nemotron35.backend_core import EngineCore
        streamed = load_streamed(Path(_dir) / "ck", pool_bytes=pool_bytes, config_check=permissive_config_check)
        return streamed, EngineCore(streamed, chunk_tokens=chunk, layer_major_tokens=layer_major)

    def test_layer_major_logits_and_state_are_bit_identical_to_chunk_major(self):
        for segment in (8, 12, 16, 100):     # segment sizes are rounded down to whole chunks (min one chunk)
            a_s, a = self.core(0)
            b_s, b = self.core(segment)
            try:
                la, lb = a.prefill(self.TOKENS), b.prefill(self.TOKENS)
                self.assertTrue(bool(mx.all(f32(la) == f32(lb))), segment)
                self.assertEqual(a.cached_tokens, b.cached_tokens)
                for x, y in zip(a.state_arrays(), b.state_arrays()):
                    self.assertTrue(bool(mx.all(f32(x) == f32(y))), segment)
                nxt = [3, 9]
                for t in nxt:       # decoding continues identically from the layer-major state
                    self.assertTrue(bool(mx.all(f32(a.decode(t)) == f32(b.decode(t)))))
            finally:
                a_s.close()
                b_s.close()

    def test_short_anchor_snapshot_matches_separate_prefills(self):
        from models.nemotron35.backend_core import EngineCore

        directory = Path(_dir) / "int8-anchor"
        write_checkpoint(directory, config=dict(TINY_CONFIG, head_dim=64))
        reference_model = load_streamed(directory, pool_bytes=0, config_check=permissive_config_check)
        candidate_model = load_streamed(directory, pool_bytes=0, config_check=permissive_config_check)
        reference = EngineCore(reference_model, chunk_tokens=4, int8_kv=True)
        candidate = EngineCore(candidate_model, chunk_tokens=4, layer_major_tokens=8, int8_kv=True)
        try:
            tokens = self.TOKENS[:7]
            reference.prefill(tokens[:4], emit_logits=False)
            prefix = [mx.array(a) for a in reference.state_arrays()]
            mx.eval(*prefix)
            expected = reference.prefill(tokens[4:])
            captured = []
            actual = candidate.prefill(tokens, anchor=4, save_anchor=lambda _, caches: captured.extend(caches))
            self.assertTrue(mx.array_equal(expected, actual).item())
            self.assertEqual(candidate.cached_tokens, len(tokens))
            self.assertTrue(all(mx.array_equal(a, b).item() for a, b in zip(
                prefix, [a for cache in captured for a in cache.state], strict=True)))
            self.assertLess(candidate_model.reader.bytes_read, reference_model.reader.bytes_read)
            candidate.bind_cache(captured)
            self.assertTrue(mx.array_equal(expected, candidate.prefill(tokens[4:])).item())
        finally:
            reference_model.close()
            candidate_model.close()

    def test_each_layers_experts_are_read_once_per_segment(self):
        a_s, a = self.core(0, pool_bytes=0)
        b_s, b = self.core(len(self.TOKENS) + 4, pool_bytes=0)
        try:
            a.prefill(self.TOKENS)
            b.prefill(self.TOKENS)
            chunk_reads, layer_reads = a_s.reader.reads, b_s.reader.reads
            moe_layers, experts = 2, TINY_CONFIG["n_routed_experts"]
            self.assertLessEqual(layer_reads, moe_layers * experts)
            self.assertLess(layer_reads, chunk_reads)
            self.assertEqual(b_s.pool.stats()["held_bytes"], 0)       # released after every layer
        finally:
            a_s.close()
            b_s.close()

    def test_progress_and_boundaries_are_per_segment_and_state_matches_each_boundary(self):
        s, core = self.core(8)
        try:
            seen, marks = [], []
            core.prefill(self.TOKENS, progress=lambda d, t: seen.append(d), boundary=lambda d: marks.append((d, core.cached_tokens)))
            self.assertEqual(seen, [8, 16, 23])
            self.assertEqual(marks, [(8, 8), (16, 16), (23, 23)])
        finally:
            s.close()

    def test_cancel_mid_layer_major_prefill_resets_and_next_run_is_exact(self):
        from models.nemotron35.runtime import Cancelled
        s, core = self.core(8)
        ref_s, ref = self.core(0)
        try:
            calls = [0]

            def cancel():
                calls[0] += 1
                return calls[0] > 6

            with self.assertRaises(Cancelled):
                core.prefill(self.TOKENS, cancelled=cancel)
            self.assertEqual(core.cached_tokens, 0)
            self.assertIsNone(s.pool._held)
            self.assertTrue(s.pool.speculate)
            self.assertTrue(bool(mx.all(f32(core.prefill(self.TOKENS)) == f32(ref.prefill(self.TOKENS)))))
        finally:
            s.close()
            ref_s.close()

    def test_backend_layer_major_generation_equals_chunk_major(self):
        text = "".join(chr(65 + (i * 7) % 26) for i in range(40))
        outputs = []
        for segment in (0, 8):
            backend = make(layer_major_tokens=segment, prefix_cache=False)
            try:
                backend.append_text(text)
                outputs.append(backend.generate(6)[0])
            finally:
                backend.close()
        self.assertEqual(outputs[0], outputs[1])
        self.assertTrue(outputs[0])


class StockKernelTests(unittest.TestCase):
    """The decode path runs the stock mlx-lm ``gather_qmm`` computation over the selected experts."""

    def test_streamed_model_is_bit_identical_to_the_stock_in_memory_model(self):
        from models.nemotron35.tiny_checkpoint import build_reference
        from models.nemotron35 import streamed_experts as se
        ref = build_reference(TINY_CONFIG)
        streamed = load_streamed(Path(_dir) / "ck", pool_bytes=0, config_check=permissive_config_check)
        patch = unittest.mock.patch.object(se, "GATHER", True)
        patch.start()
        self.addCleanup(patch.stop)
        try:
            c1, c2 = ref.make_cache(), streamed.model.make_cache()
            tokens = mx.array([[3, 9, 1, 5, 7, 2, 11, 4]])
            self.assertTrue(bool(mx.all(ref(tokens, cache=c1) == streamed.model(tokens, cache=c2))))
            for t in (3, 9, 1, 5, 7, 2):
                x = mx.array([[t]])
                self.assertTrue(bool(mx.all(ref(x, cache=c1) == streamed.model(x, cache=c2))))
        finally:
            streamed.close()

    def test_per_expert_fallback_for_large_chunks_matches_within_rounding(self):
        from models.nemotron35 import streamed_experts as se
        streamed = load_streamed(Path(_dir) / "ck", pool_bytes=0, config_check=permissive_config_check)
        try:
            tokens = mx.array([[3, 9, 1, 5, 7, 2, 11, 4, 6, 8, 10, 12]])
            with unittest.mock.patch.object(se, "GATHER", True):
                fast = f32(streamed.model(tokens, cache=streamed.model.make_cache()))
            with unittest.mock.patch.object(se, "GATHER_MAX_EXPERTS", 0):
                slow = f32(streamed.model(tokens, cache=streamed.model.make_cache()))
            self.assertLess(float(mx.max(mx.abs(fast - slow))), 0.05)
        finally:
            streamed.close()
