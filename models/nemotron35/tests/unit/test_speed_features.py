"""Prefetch, parallel reads, auto pool and the persistent prefix cache, on tiny synthetic checkpoints."""
import shutil
import tempfile
import unittest
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
                prefix_dir=str(Path(_dir) / "prefix"), snapshot_tokens=8)
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
            a.generate(3, on_prefill_progress=lambda d, t: seen.append(d), should_cancel=lambda: len(seen) > 20)
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

    def test_quota_evicts_oldest_but_keeps_newest(self):
        a = self.backend(prefix_quota_gb=0.00005)   # ~50 KB: smaller than one snapshot
        a.append_tokens(self.PROMPT)
        a.generate(2)
        self.assertLessEqual(len(a.prefix_cache._disk), 1)
        self.assertGreater(a.prefix_cache.stats.evicted_ssd, 0)
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
