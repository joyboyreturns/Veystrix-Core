import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from veystrix import cache_tools
from models.flashnext import prefix_cache as P


class TestCacheTools(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.tmp_dir) / "prefix-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_entry(self, key: str, n_tokens: int, parent: str | None = None, file_bytes: int = 1000, age_days: float = 0):
        manifest = {
            "key": key,
            "n_tokens": n_tokens,
            "parent": parent,
            "file_bytes": file_bytes,
            "created": time.time() - (age_days * 86400),
            "fingerprint": "abc1234567890def",
            "format": P.FORMAT_VERSION,
        }
        manifest_path = self.cache_dir / f"{key}.json"
        manifest_path.write_text(json.dumps(manifest))
        data_path = self.cache_dir / f"{key}.safetensors"
        data_path.write_bytes(b"x" * file_bytes)

    def test_inspect_cache_reports_correct_metrics(self):
        self._create_entry("a" * 64, n_tokens=100, file_bytes=2000)
        self._create_entry("b" * 64, n_tokens=200, file_bytes=3000)

        res = cache_tools.inspect_cache(self.cache_dir)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["total_entries"], 2)
        self.assertEqual(res["total_tokens_cached"], 200)
        self.assertGreater(res["total_bytes"], 5000)  # 2000 + 3000 + json manifests

    def test_clean_cache_iterative_leaf_pruning_on_delta_chains(self):
        # Chain: A -> B -> C
        key_a = "1" * 64
        key_b = "2" * 64
        key_c = "3" * 64
        self._create_entry(key_a, n_tokens=100, parent=None, file_bytes=1000)
        self._create_entry(key_b, n_tokens=200, parent=key_a, file_bytes=1000)
        self._create_entry(key_c, n_tokens=300, parent=key_b, file_bytes=1000)

        # Total is ~3000 bytes. If target_gb is tiny (e.g. 0.000001 GB = 1000 bytes),
        # it should prune C first (leaf), exposing B as a new leaf, then prune B, leaving A.
        res = cache_tools.clean_cache(self.cache_dir, target_gb=0.0000015)
        self.assertEqual(res["status"], "ok")
        # Should have pruned at least 2 entries (C and B) to get below ~1500 bytes
        self.assertGreaterEqual(res["removed_entries"], 2)
        self.assertFalse((self.cache_dir / f"{key_c}.safetensors").exists())
        self.assertFalse((self.cache_dir / f"{key_b}.safetensors").exists())
        # A should still be retained or pruned only if needed
        self.assertTrue((self.cache_dir / f"{key_a}.safetensors").exists())

    def test_clean_cache_respects_directory_ownership_lock(self):
        # Simulate active owner holding lock
        owner = P._claim_directory(self.cache_dir)
        try:
            # Another attempt to clean should hit CacheOwnershipError
            with mock.patch("models.flashnext.prefix_cache._claim_directory", side_effect=P.CacheOwnershipError("Locked")):
                res = cache_tools.clean_cache(self.cache_dir, target_gb=0.1)
                self.assertEqual(res["status"], "error")
                self.assertEqual(res["error"], "CacheOwnershipError")
                self.assertEqual(res["removed_entries"], 0)
        finally:
            P._release_directory(*owner)

    def test_cli_argument_aliases(self):
        with mock.patch("sys.argv", ["cache_tools", "--legacy", "--cleanup", "--target-disk-gb", "20", "--directory", str(self.cache_dir)]):
            with mock.patch("veystrix.cache_tools.clean_cache") as mock_clean:
                mock_clean.return_value = {"status": "ok"}
                with mock.patch("builtins.print"):
                    cache_tools.main()
                mock_clean.assert_called_once_with(str(self.cache_dir), max_age_days=None, target_gb=20.0, dry_run=False)


if __name__ == "__main__":
    unittest.main()
