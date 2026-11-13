"""Prefetch, parallel reads, auto pool and the persistent prefix cache, on tiny synthetic checkpoints."""
import shutil
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import mlx.core as mx

from models.nemotron35 import admission
from models.nemotron35.prefix_cache import PrefixCache
from models.nemotron35.streamed_model import load_streamed
from models.nemotron35.streamed_experts import StreamedSwitchMLP
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

    def test_prefill_seed_moves_the_hottest_held_expert_into_decode_pool(self):
        model = self.load(prefetch=False)
        self.addCleanup(model.close)
        pool = model.pool
        pool.begin_layer(1, seed=True)
        resident = pool.fetch(1, [0, 1])
        pool._held_routes.update([1, 0, 1])
        pool.end_layer(seed=True)
        before = pool.stats()
        reused = pool.fetch(1, [1])
        self.assertIs(reused[1], resident[1])
        self.assertEqual(pool.stats()["bytes_read"], before["bytes_read"])
        self.assertGreater(before["seeded_bytes"], 0)

    def test_explicit_fma_nemotron_projection_matches_stock_within_bf16_tolerance(self):
        model = self.load(prefetch=False)
        self.addCleanup(model.close)
        weights = model.pool.fetch(1, [0])[0]
        group = TINY_CONFIG["quantization"]["group_size"]
        layer = StreamedSwitchMLP(model.pool, 1, group)
        x = mx.random.normal((1, TINY_CONFIG["hidden_size"])).astype(mx.bfloat16)
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_METAL_FMA_DEQUANT": "0"}):
            expected = layer._expert(x, weights)
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_METAL_FMA_DEQUANT": "1"}):
            actual = layer._expert(x, weights)
        self.assertTrue(mx.allclose(f32(actual), f32(expected), rtol=0.02, atol=0.02).item())

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

    def test_decode_overlap_preserves_slot_order(self):
        baseline = self.load(prefetch=False, pool_bytes=64 * 1024)
        candidate = self.load(prefetch=False, pool_bytes=64 * 1024)
        self.addCleanup(baseline.close)
        self.addCleanup(candidate.close)
        x = (mx.arange(TINY_CONFIG["hidden_size"], dtype=mx.float32) / 64).reshape(1, -1)
        routes = mx.array([[2, 0, 1]])
        group = TINY_CONFIG["quantization"]["group_size"]
        for model in (baseline, candidate):
            model.pool.fetch(1, [0])
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_DECODE_OVERLAP": "0"}):
            expected = StreamedSwitchMLP(baseline.pool, 1, group)(x, routes)
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_DECODE_OVERLAP": "1"}), \
                unittest.mock.patch.object(mx, "async_eval", wraps=mx.async_eval) as dispatch:
            actual = StreamedSwitchMLP(candidate.pool, 1, group)(x, routes)
            dispatch.assert_called_once()
        self.assertTrue(mx.array_equal(actual, expected).item())


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

    def test_background_save_restores_original_state_after_decode_advances(self):
        backend = self.backend()
        self.addCleanup(backend.close)
        tokens = self.PROMPT[:12]
        backend.core.prefill(tokens, emit_logits=False)
        expected = [array.tolist() for array in backend.core.state_arrays()]
        self.assertTrue(backend.prefix_cache.save_background(backend.core.cache, tokens))
        backend.core.decode(self.PROMPT[12])
        backend.prefix_cache.flush_writes()
        restored, count = backend.prefix_cache.restore(self.PROMPT)
        self.assertEqual(count, len(tokens))
        self.assertEqual([array.tolist() for entry in restored for array in entry.state], expected)

    def test_lookup_verifier_matches_singleton_decode_with_int8_kv(self):
        from models.nemotron35.backend_core import EngineCore

        directory = Path(_dir) / "int8-lookup"
        write_checkpoint(directory, config=dict(TINY_CONFIG, head_dim=64))
        models = [load_streamed(directory, pool_bytes=0, config_check=permissive_config_check) for _ in range(2)]
        for model in models:
            self.addCleanup(model.close)
        reference, candidate = [EngineCore(model, chunk_tokens=4, int8_kv=True) for model in models]
        prompt = self.PROMPT[:12]
        first = int(mx.argmax(reference.prefill(prompt), axis=-1).item())
        tokens = [first]
        for _ in range(4):
            tokens.append(int(mx.argmax(reference.decode(tokens[-1]), axis=-1).item()))
        for drafts in (tokens[1:4], [(tokens[1] + 1) % 64, *tokens[2:4]]):
            candidate.reset()
            candidate.prefill(prompt)
            committed, next_token = candidate.verify_decode(first, drafts, cancelled=lambda: False)
            self.assertEqual(committed, tokens[:4] if drafts[0] == tokens[1] else tokens[:1])
            self.assertEqual(next_token, tokens[len(committed)])
            self.assertEqual(candidate.cached_tokens, len(prompt) + len(committed))
            reference.reset()
            reference.prefill(prompt)
            for token in committed:
                reference.decode(token)
            self.assertTrue(all(mx.array_equal(a, b).item() for a, b in
                                zip(reference.state_arrays(), candidate.state_arrays(), strict=True)))

    def test_ram_restore_is_independent_and_rejects_replaced_disk_state(self):
        with unittest.mock.patch.dict(os.environ, {"VEYSTRIX_NEMOTRON_PREFIX_RAM_MB": "8"}):
            backend = self.backend()
        self.addCleanup(backend.close)
        cache = backend.prefix_cache
        tokens = self.PROMPT[:12]
        backend.core.prefill(tokens, emit_logits=False)
        self.assertTrue(cache.save(backend.core.cache, tokens))
        first, length = cache.restore(self.PROMPT)
        self.assertEqual(length, len(tokens))
        expected = first[0].state[0].tolist()
        first[0].state[0][:] = 0
        second, length = cache.restore(self.PROMPT)
        self.assertEqual(length, len(tokens))
        self.assertEqual(second[0].state[0].tolist(), expected)
        self.assertEqual((cache.stats.ssd_hits, cache.stats.ram_hits), (1, 1))
        cache._paths(cache._key(tokens))[0].write_bytes(b"replaced")
        self.assertEqual(cache.restore(self.PROMPT), (None, 0))
        self.assertEqual(cache._ram_bytes, 0)

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

    def test_corrupt_longest_snapshot_falls_back_to_shorter_exact_prefix(self):
        backend = self.backend()
        try:
            cache = backend.prefix_cache
            tokens = list(self.PROMPT)
            backend.core.prefill(tokens[:8], emit_logits=False)
            self.assertTrue(cache.save(backend.core.cache, tokens[:8]))
            backend.core.prefill(tokens[8:12], emit_logits=False)
            self.assertTrue(cache.save(backend.core.cache, tokens[:12]))
            state_path, _ = cache._paths(cache._key(tokens[:12]))
            state_path.write_bytes(b"corrupt")

            restored, length = cache.restore(tokens)

            self.assertIsNotNone(restored)
            self.assertEqual(length, 8)
            self.assertEqual(cache.stats.restore_failures, 1)
            self.assertFalse(state_path.exists())
        finally:
            backend.close()

    def test_default_cache_retains_a_short_tool_harness_prefix(self):
        backend = make(prefix_cache=False)
        try:
            prefix = (self.PROMPT * 3)[:320]
            backend.core.prefill(prefix, emit_logits=False)
            cache = PrefixCache(Path(_dir) / "prefix", {"model": "short-tool-prefix"}, quota_bytes=2**30)

            self.assertTrue(cache.save(backend.core.cache, prefix))
            restored, length = cache.restore(prefix + [1, 2, 3])
            self.assertIsNotNone(restored)
            self.assertEqual(length, 320)
        finally:
            backend.close()

    def test_managed_restore_releases_global_lock_and_protects_snapshot(self):
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        from veystrix.universal_cache import CacheManager
        from models.nemotron35 import state

        manager = CacheManager(Path(_dir) / "prefix", quota_bytes=2**30, reserve_bytes=0)
        backend = make(prefix_cache=False)
        self.addCleanup(backend.close)
        cache = PrefixCache(None, {"model": "managed"}, quota_bytes=2**30, min_tokens=8, manager=manager)
        prefix = self.PROMPT[:32]
        backend.core.prefill(prefix)
        self.assertTrue(cache.save(backend.core.cache, prefix))
        suffix = [9, 10]
        expected = backend.core.prefill(suffix)
        real_restore = state.restore

        def clear():
            with manager.transaction() as tx:
                return tx.reclaim(tx.used, apply=True)["snapshots"]

        with ThreadPoolExecutor(max_workers=1) as executor:
            def restore_while_clearing(*args):
                self.assertEqual(executor.submit(clear).result(timeout=5), 0)
                return real_restore(*args)

            with patch.object(state, "restore", side_effect=restore_while_clearing):
                restored, length = cache.restore(prefix + suffix)
        self.assertEqual(length, len(prefix))
        backend.core.bind_cache(restored)
        actual = backend.core.prefill(suffix)
        self.assertTrue(mx.array_equal(actual, expected).item())
        with manager.transaction() as tx:
            self.assertEqual(tx.reclaim(tx.used, apply=True)["snapshots"], 1)

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

    def test_flashnext_and_nemotron_state_lookups_remain_family_scoped(self):
        from models.flashnext import prefix_cache as flashnext
        from veystrix.universal_cache import CacheManager
        import json

        manager = CacheManager(Path(_dir) / "prefix", quota_bytes=2**30, reserve_bytes=0)
        backend = make(prefix_cache=False)
        self.addCleanup(backend.close)
        nem = PrefixCache(None, {"model": "nemotron"}, quota_bytes=2**30, min_tokens=8, manager=manager)
        qwen = flashnext.PrefixCache(nem.directory.name, 0, manager=manager)
        prefix = self.PROMPT[:32]
        query = prefix + [7]
        backend.core.prefill(prefix, emit_logits=False)
        self.assertTrue(nem.save(backend.core.cache, prefix))
        self.assertEqual(nem.reusable_tokens(query), len(prefix))
        self.assertEqual(qwen.peek(query), 0)

        nem.clear()
        key = flashnext.chain_key(qwen.fp, prefix)
        (qwen.directory / f"{key}.safetensors").write_bytes(b"fixture")
        (qwen.directory / f"{key}.json").write_text(json.dumps({
            "key": key, "fingerprint": qwen.fp, "n_tokens": len(prefix), "parent": None, "file_bytes": 7,
        }))
        self.assertEqual(qwen.peek(query), len(prefix))
        self.assertEqual(nem.restore(query), (None, 0))

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
