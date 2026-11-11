import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from veystrix import universal_cache as U


class SharedCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.cache = U.CacheManager(self.directory, quota_bytes=U.DEFAULT_QUOTA_GB * U.GB, reserve_bytes=0)

    def snapshot(self, family, key, tokens, size, parent=None):
        fingerprint = "a" * 64
        path = self.cache.namespace(family, fingerprint)
        key = key * U.FORMATS[family][0]
        data = path / (key + ".safetensors")
        data.write_bytes(bytes(size))
        side = path / (key + U.FORMATS[family][1])
        if family == "flashnext":
            side.write_text(json.dumps({"key": key, "fingerprint": fingerprint, "parent": parent,
                                        "file_bytes": size, "n_tokens": tokens}))
        else:
            side.write_bytes(b"VXT1" + bytes(tokens * 4))
        return key, path, data.stat().st_size + side.stat().st_size

    def test_quota_preview_lease_and_ancestor(self):
        base, qwen, base_size = self.snapshot("flashnext", "b", 100, 150)
        leaf, _, child_size = self.snapshot("flashnext", "c", 10_000, 150, base)
        _, _, other_size = self.snapshot("nemotron35", "d", 100, 200)
        with self.cache.transaction() as tx:
            self.assertEqual(tx.quota, 80 * U.GB)
            self.assertEqual(tx.used, base_size + child_size + other_size)
            lease = tx.lease(qwen, leaf)
        try:
            with self.cache.transaction() as tx:
                preview = tx.reclaim(tx.used)
                self.assertEqual(preview["reclaimed_bytes"], other_size)
                self.assertEqual(preview["shortfall_bytes"], base_size + child_size)
                self.assertEqual(len(tx.entries), 3)
                blocked = tx.set_quota(100)
                self.assertGreater(blocked["shortfall_bytes"], 0)
                self.assertEqual(blocked["reclaimed_bytes"], 0)
                self.assertEqual(len(tx.entries), 3)
                result = tx.reclaim(other_size, apply=True)
                self.assertEqual(result["reclaimed_bytes"], other_size)
                self.assertFalse(tx.make_room(tx.quota))
        finally:
            os.close(lease)
        with self.cache.transaction() as tx:
            self.assertEqual(tx.set_quota(100)["shortfall_bytes"], 0)
            self.assertEqual(tx.used, 0)
        with U.CacheManager(self.directory).transaction() as tx:
            self.assertEqual(tx.quota, 100)

    def test_model_free_process_observes_committed_files_and_skips_unknown_data(self):
        _, path, size = self.snapshot("nemotron35", "a", 8, 200)
        user = path / "conversation.txt"
        user.write_text("preserve")
        orphan = path / ("b" * 24 + ".safetensors")
        orphan.write_bytes(b"interrupted")
        link = path / ("c" * 24 + ".safetensors")
        link.symlink_to(user)
        command = [sys.executable, "-m", "veystrix.universal_cache", "--directory", str(self.directory)]
        status = json.loads(subprocess.check_output(command + ["status"]))
        self.assertEqual(status["used_bytes"], size)
        self.assertFalse(orphan.exists())
        subprocess.check_output(command + ["clear"])
        self.assertEqual(user.read_text(), "preserve")
        self.assertTrue(link.is_symlink())

    def test_external_lease_and_pins_protect_the_same_registry(self):
        key, path, _ = self.snapshot("flashnext", "b", 1000, 100)
        other, _, other_size = self.snapshot("flashnext", "c", 500, 50)
        with self.cache.transaction() as tx:
            self.assertTrue(tx.set_pin(path, other, True))
        script = """
import os, sys
from pathlib import Path
from veystrix.universal_cache import CacheManager
with CacheManager(sys.argv[1]).transaction() as tx:
    fd = tx.lease(Path(sys.argv[2]), sys.argv[3])
print("ready", flush=True)
sys.stdin.readline()
os.close(fd)
"""
        with subprocess.Popen([sys.executable, "-c", script, str(self.directory), str(path), key],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as process:
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                with self.cache.transaction() as tx:
                    self.assertEqual(tx.reclaim(tx.used, apply=True)["snapshots"], 0)
            finally:
                process.communicate("\n", timeout=10)
        with self.cache.transaction() as tx:
            result = tx.reclaim(tx.used, apply=True)
            self.assertEqual(result["shortfall_bytes"], other_size)
            self.assertTrue(tx.set_pin(path, other, False))
            self.assertEqual(tx.reclaim(tx.used, apply=True)["shortfall_bytes"], 0)

    def test_reserve_refuses_growth_even_below_quota(self):
        cache = U.CacheManager(self.directory, reserve_bytes=10**18)
        with cache.transaction() as tx:
            self.assertFalse(tx.make_room(1))
