import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

from models.flashnext import prefix_cache as P
from models.flashnext import prefix_profiles as PP
from models.flashnext import prefix_session as S
from models.flashnext import prefill as prefill_module


def make_dummy_entry(key: str, n_tokens: int = 10, parent: str | None = None, parent_tokens: int = 0) -> P.Entry:
    delta_len = max(1, n_tokens - parent_tokens)
    layer = P.Layer(
        kind="qsa_kv",
        kv={
            "keys": mx.zeros((1, 1, delta_len, 8)),
            "values": mx.zeros((1, 1, delta_len, 8)),
        },
    )
    return P.Entry(
        key=key,
        parent=parent,
        n_tokens=n_tokens,
        parent_tokens=parent_tokens,
        token_delta=tuple(range(parent_tokens, n_tokens)),
        layers=[layer],
    )


class TestPhase3Remediations(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.tmp_dir) / "prefix-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_r1_03_boundaries_budgeted_policy_aligns_with_budgeted_edges(self):
        """R1-03: PrefixSession.boundaries uses budgeted_edges when chunk policy is budgeted."""
        encode = lambda s: [ord(c) for c in s]
        compat = lambda: "fingerprint123"
        session = S.PrefixSession(encode, compat)

        prompt = list(range(1000))
        with mock.patch.dict(os.environ, {
            "FLASHNEXT_PREFILL_CHUNK_POLICY": "budgeted",
            S.FLAG_BOUNDARIES: "all",
            S.FLAG_MIN_TOKENS: "100",
        }):
            mock_edges = [250, 500, 750]
            with mock.patch("models.flashnext.prefill.budgeted_edges", return_value=mock_edges) as mock_b_edges:
                bounds = session.boundaries(prompt, 0)
                mock_b_edges.assert_called_once()
                for edge in mock_edges:
                    self.assertIn(edge, bounds)

    def test_r2_06_evict_ram_preserves_parent_with_resident_child_if_persist_fails(self):
        """R2-06: _evict_ram does not drop a parent whose child is resident if persist is skipped."""
        fp = "testfp" + "0" * 26
        # Cache with tiny RAM budget (1 byte) and disk reserve that forces persist to skip
        cache = P.PrefixCache(fp, ram_budget=1, directory=self.cache_dir, min_disk_free_bytes=10**15)

        e_parent = make_dummy_entry("p" * 64, n_tokens=10, parent=None, parent_tokens=0)
        e_child = make_dummy_entry("c" * 64, n_tokens=20, parent="p" * 64, parent_tokens=10)

        with cache._lock:
            with mock.patch("shutil.disk_usage") as mock_usage:
                mock_usage.return_value = mock.Mock(free=100)  # less than min_disk_free_bytes
                cache._admit(e_parent)
                cache._admit(e_child)

                # e_parent has resident child e_child, and was not persisted to disk.
                # It must NOT be dropped from RAM while e_child is resident.
                self.assertIn("p" * 64, cache._ram)

    def test_r2_06_shrink_ram_does_not_drop_unpersisted_entries(self):
        """R2-06: shrink_ram does not drop an entry if _persist cannot write to SSD."""
        fp = "testfp" + "0" * 26
        cache = P.PrefixCache(fp, ram_budget=10**8, directory=self.cache_dir, min_disk_free_bytes=10**15)

        entry = make_dummy_entry("e" * 64, n_tokens=10, parent=None)
        cache._admit(entry)
        self.assertIn("e" * 64, cache._ram)

        with mock.patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = mock.Mock(free=100)  # cannot persist
            freed = cache.shrink_ram(0)
            self.assertEqual(freed, 0)
            self.assertIn("e" * 64, cache._ram)
            self.assertNotIn("e" * 64, cache._disk)

    def test_r2_06_evict_ram_retains_unpersisted_leaf(self):
        cache = P.PrefixCache("testfp" + "0" * 26, ram_budget=1,
                              directory=self.cache_dir, min_disk_free_bytes=10**15)
        leaf = make_dummy_entry("l" * 64, n_tokens=10, parent=None)
        newest = make_dummy_entry("n" * 64, n_tokens=20, parent=None)
        with mock.patch("shutil.disk_usage", return_value=mock.Mock(free=100)):
            cache._admit(leaf)
            cache._admit(newest)
        self.assertIn(leaf.key, cache._ram)
        self.assertNotIn(leaf.key, cache._disk)
        self.assertEqual(cache._ram_bytes, sum(e.nbytes() for e in cache._ram.values()))

    def test_child_is_not_persisted_when_parent_spill_fails(self):
        cache = P.PrefixCache("testfp" + "0" * 26, ram_budget=10**8,
                              directory=self.cache_dir, min_disk_free_bytes=0)
        parent = make_dummy_entry("p" * 64, n_tokens=10, parent=None)
        child = make_dummy_entry("c" * 64, n_tokens=20, parent=parent.key, parent_tokens=10)
        cache._admit(parent)
        cache._admit(child)
        with mock.patch.object(P, "write_entry", side_effect=OSError("disk failure")) as write:
            cache._persist(child)
        self.assertEqual(write.call_count, 1)
        self.assertNotIn(parent.key, cache._disk)
        self.assertNotIn(child.key, cache._disk)
        self.assertIn(child.key, cache._ram)

    def test_pressure_invalidation_removes_only_unleased_unbacked_leaves(self):
        cache = P.PrefixCache("testfp" + "0" * 26, ram_budget=10**8,
                              directory=self.cache_dir, min_disk_free_bytes=10**15)
        parent = make_dummy_entry("p" * 64, n_tokens=10, parent=None)
        child = make_dummy_entry("c" * 64, n_tokens=20, parent=parent.key, parent_tokens=10)
        cache._admit(parent)
        cache._admit(child)
        with cache.lease((child.key,)):
            self.assertEqual(cache.invalidate_unbacked_under_pressure()["invalidated_entries"], 0)
        result = cache.invalidate_unbacked_under_pressure()
        self.assertEqual(result["invalidated_entries"], 2)
        self.assertTrue(result["recompute_required"])
        self.assertEqual(cache._ram_bytes, 0)
        self.assertEqual(cache.inspect()["stats"]["pressure_invalidated"], 2)
        self.assertNotIn(child.key, cache._plans)

    def test_pressure_invalidation_never_discards_durable_snapshot(self):
        cache = P.PrefixCache("testfp" + "0" * 26, ram_budget=10**8, directory=self.cache_dir)
        entry = make_dummy_entry("e" * 64, n_tokens=10, parent=None)
        cache._admit(entry)
        cache._disk[entry.key] = {"n": 10, "parent": None, "bytes": 10, "atime": 0}
        result = cache.invalidate_unbacked_under_pressure()
        self.assertEqual(result["invalidated_entries"], 0)
        self.assertIn(entry.key, cache._ram)
        self.assertIn(entry.key, cache._disk)

    def test_r2_07_prefix_cache_evicts_foreign_leaves_first(self):
        """R2-07: Base PrefixCache tracks foreign manifests and reclaims them first on disk over-budget."""
        fp = "testfp" + "0" * 26
        foreign_key = "f" * 64
        manifest = {
            "key": foreign_key, "n_tokens": 50, "parent": None,
            "fingerprint": "other_fingerprint", "format": P.FORMAT_VERSION,
            "file_bytes": 1000,
        }
        (self.cache_dir / f"{foreign_key}.json").write_text(json.dumps(manifest))
        (self.cache_dir / f"{foreign_key}.safetensors").write_bytes(b"x" * 1000)

        cache = P.PrefixCache(fp, ram_budget=10**6, directory=self.cache_dir, disk_budget=1500)
        cache.min_disk_free_bytes = 0     # the test is about eviction order, not about how much disk the machine has free
        self.assertIn(foreign_key, cache._foreign)

        local_entry = make_dummy_entry("l" * 64, n_tokens=60, parent=None)
        cache._admit(local_entry)
        cache.shrink_ram(0)  # persists local_entry to disk

        # Foreign entry should be evicted first!
        self.assertNotIn(foreign_key, cache._foreign)
        self.assertFalse((self.cache_dir / f"{foreign_key}.safetensors").exists())
        self.assertIn("l" * 64, cache._disk)


if __name__ == "__main__":
    unittest.main()
