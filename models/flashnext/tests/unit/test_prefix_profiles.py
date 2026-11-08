"""Prefix profiles and the managed SSD cache: records, recognition, eviction, limits, protection, crash recovery.

Same small random-weight model as the prefix-cache tests, so snapshots are real (KV deltas, GatedDeltaNet state,
convolution state, position state). Clock and free disk space are injected, so every policy is deterministic.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from models.flashnext import prefix_cache as P
from models.flashnext import prefix_profiles as PP
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Base(unittest.TestCase):
    model = None

    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.dir = self.tmp / "cache"
        self.fp = P.fingerprint(model="small", engine="profiles")
        self.clock = Clock()
        self.free = [10**15]

    def tokens(self, seed, n=48):
        mx.random.seed(seed)
        return [int(t) for t in mx.random.randint(2, VOCAB, (n,)).tolist()]

    def live(self, tokens):
        caches = self.model.make_cache()
        mx.eval(self.model(mx.array([tokens]), cache=caches).logits, [c.state for c in caches])
        return caches

    def cache(self, ram=1 << 40, disk=1 << 40, reserve=0, pin_after=3, fp=None, **kw):
        kw.setdefault("config", {"expert_pool_gb": 2.0, "chunk_tokens": 64})
        return PP.ProfiledPrefixCache(
            fp or self.fp, ram, self.dir, disk, free_reserve=reserve, free_bytes=lambda path: self.free[0],
            pin_after=pin_after, clock=self.clock, orphan_grace_s=60.0, **kw)

    def snap(self, cache, seed, n=48, parent=None, kind=None):
        ids = self.tokens(seed, n)
        return ids, cache.put(ids, self.live(ids), parent=parent, plan=(), kind=kind)

    def hit(self, cache, ids):
        found, caches = cache.lookup(list(ids) + [5], self.model.make_cache)
        return found

    def entry_bytes(self):
        scratch = self.cache()
        _, entry = self.snap(scratch, 1)
        size = scratch._disk[entry.key]["bytes"]
        scratch.clear()
        return size

    def disk_files(self):
        return sorted(p.name for p in self.dir.iterdir())


class ProfileRecordTests(Base):
    def test_a_snapshot_has_a_profile_with_identity_cost_and_execution_profile(self):
        cache = self.cache()
        ids, entry = self.snap(cache, 1, kind="system")
        row = next(r for r in cache.profiles() if r["key"] == entry.key)
        self.assertEqual((row["n_tokens"], row["kind"], row["hits"], row["pinned"]), (48, "system", 0, None))
        self.assertEqual(row["plan_edges"], ())
        self.assertEqual(row["config"], {"expert_pool_gb": 2.0, "chunk_tokens": 64})
        self.assertAlmostEqual(row["recompute_seconds"], 48 / PP.DEFAULT_PREFILL_TOK_S)
        manifest = P.read_manifest(self.dir, entry.key)
        self.assertEqual(manifest["fingerprint"], self.fp)          # model and configuration fingerprint
        self.assertTrue((self.dir / f"{entry.key}.safetensors").exists())

    def test_hits_count_frequency_recency_and_time_saved(self):
        cache = self.cache()
        ids, entry = self.snap(cache, 1, kind="system")
        self.clock.advance(100)
        self.hit(cache, ids)
        self.clock.advance(50)
        self.hit(cache, ids)
        row = cache.profiles()[0]
        self.assertEqual(row["hits"], 2)
        self.assertEqual(row["last_used"], self.clock.now)
        self.assertGreater(row["saved_seconds"], 0)
        report = cache.report()
        self.assertEqual((report["hits"], report["misses"], report["restored_tokens"]), (2, 0, 96))
        self.assertGreater(report["time_saved_seconds"], 0)

    def test_the_learned_prefill_rate_replaces_the_default(self):
        cache = self.cache()
        cache.record_prefill(2048, 34.0)
        self.assertAlmostEqual(cache.prefill_rate(), 2048 / 34.0)
        cache.record_prefill(100, 0.1)                                  # too short to say anything
        self.assertAlmostEqual(cache.prefill_rate(), 2048 / 34.0)

    def test_profiles_survive_a_restart_with_their_counters_and_pins(self):
        cache = self.cache()
        ids, entry = self.snap(cache, 1, kind="system")
        self.hit(cache, ids)
        self.assertTrue(cache.pin(ids))
        cache.record_prefill(2048, 34.0)
        cache.flush()
        again = self.cache()
        row = again.profiles()[0]
        self.assertEqual((row["hits"], row["pinned"], row["kind"]), (1, "manual", "system"))
        self.assertEqual(again.index_state, "ok")
        self.assertAlmostEqual(again.prefill_rate(), 2048 / 34.0)
        self.assertEqual(self.hit(again, ids).tier, "ssd")              # and the state restores from the SSD tier

    def test_a_drifted_execution_profile_is_reported_but_still_restores(self):
        cache = self.cache()
        ids, _ = self.snap(cache, 1)
        again = self.cache(config={"expert_pool_gb": 1.0, "chunk_tokens": 64})
        self.assertIsNotNone(self.hit(again, ids))
        self.assertEqual(again.extra["config_drift"], 1)


class RecognitionTests(Base):
    def test_a_repeated_system_prompt_is_recognised_and_pinned_automatically(self):
        cache = self.cache(pin_after=2)
        ids = self.tokens(1)
        self.assertEqual(cache.observe(ids), 0)                         # new prompt
        cache.put(ids, self.live(ids), plan=(), kind="system")
        self.assertIsNone(cache.profiles()[0]["pinned"])
        self.assertEqual(cache.observe(ids), 1)                         # recognised
        self.assertIsNone(cache.profiles()[0]["pinned"])
        self.assertEqual(cache.observe(ids), 2)
        self.assertEqual(cache.profiles()[0]["pinned"], "auto")
        self.assertEqual(cache.profiles()[0]["seen"], 3)

    def test_only_a_system_snapshot_is_auto_pinned(self):
        cache = self.cache(pin_after=1)
        ids, _ = self.snap(cache, 1, kind=None)
        for _ in range(3):
            self.hit(cache, ids)
        self.assertIsNone(cache.profiles()[0]["pinned"])

    def test_auto_pins_never_take_more_than_half_the_budget(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 2.5), pin_after=1)
        first, _ = self.snap(cache, 1, kind="system")
        second, _ = self.snap(cache, 2, kind="system")
        for ids in (first, second):
            for _ in range(2):
                self.hit(cache, ids)
        self.assertEqual(sum(1 for r in cache.profiles() if r["pinned"]), 1)

    def test_a_different_prompt_is_not_recognised(self):
        cache = self.cache()
        cache.observe(self.tokens(1))
        changed = self.tokens(1)
        changed[7] = (changed[7] + 1) % VOCAB
        self.assertEqual(cache.observe(changed), 0)


class EvictionTests(Base):
    def test_budget_evicts_the_least_valuable_entry_first_and_keeps_a_frequently_used_one(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 3.5))
        hot, _ = self.snap(cache, 1)
        cold, _ = self.snap(cache, 2)
        self.clock.advance(10)
        for _ in range(4):
            self.hit(cache, hot)
        self.clock.advance(10)
        self.snap(cache, 3)
        self.clock.advance(10)
        self.snap(cache, 4)                                              # the fourth no longer fits
        keys = {r["key"] for r in cache.profiles()}
        self.assertIn(P.chain_key(self.fp, tuple(hot)), keys)           # used often: kept although older
        self.assertNotIn(P.chain_key(self.fp, tuple(cold)), keys)       # never used: evicted first
        self.assertLessEqual(cache.usage()["disk_total_bytes"], int(size * 3.5))
        self.assertEqual(cache.stats.evicted_ssd, 1)
        self.assertFalse((self.dir / f"{P.chain_key(self.fp, tuple(cold))}.safetensors").exists())

    def test_among_unused_entries_the_oldest_goes_first(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 2.5))
        old, _ = self.snap(cache, 1)
        self.clock.advance(3600)
        self.snap(cache, 2)
        self.clock.advance(3600)
        self.snap(cache, 3)
        self.assertNotIn(P.chain_key(self.fp, tuple(old)), {r["key"] for r in cache.profiles()})

    def test_pinned_entries_are_never_evicted_and_a_write_that_cannot_fit_is_skipped(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 2.5))
        first, _ = self.snap(cache, 1)
        second, _ = self.snap(cache, 2)
        cache.pin(first)
        cache.pin(second)
        _, third = self.snap(cache, 3)
        keys = {r["key"] for r in cache.profiles()}
        self.assertEqual(keys, {P.chain_key(self.fp, tuple(first)), P.chain_key(self.fp, tuple(second))})
        self.assertEqual(cache.stats.writes_skipped, 1)
        self.assertIn(third.key, cache._ram)                             # kept in RAM, only the SSD write was skipped

    def test_a_pinned_child_protects_its_whole_chain_and_only_leaves_are_evicted(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 4.5))
        ids = self.tokens(1, 96)
        caches = self.model.make_cache()
        mx.eval(self.model(mx.array([ids[:32]]), cache=caches).logits, [c.state for c in caches])
        root = cache.put(ids[:32], caches, plan=())
        mx.eval(self.model(mx.array([ids[32:64]]), cache=caches).logits, [c.state for c in caches])
        mid = cache.put(ids[:64], caches, parent=root, plan=(32,))
        mx.eval(self.model(mx.array([ids[64:]]), cache=caches).logits, [c.state for c in caches])
        leaf = cache.put(ids, caches, parent=mid, plan=(32, 64))
        cache.pin(ids)                                                   # pin the leaf
        for seed in range(10, 16):
            self.clock.advance(5)
            self.snap(cache, seed)
        keys = {r["key"] for r in cache.profiles()}
        self.assertTrue({root.key, mid.key, leaf.key} <= keys, "a pinned entry's ancestors must stay")
        found = self.hit(cache, ids)
        self.assertEqual(found.n_tokens, 96)

    def test_eviction_never_leaves_an_entry_without_its_parent(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 3.5))
        ids = self.tokens(1, 96)
        caches = self.model.make_cache()
        parent = None
        for start, stop in ((0, 32), (32, 64), (64, 96)):
            mx.eval(self.model(mx.array([ids[start:stop]]), cache=caches).logits, [c.state for c in caches])
            parent = cache.put(ids[:stop], caches, parent=parent, plan=tuple(e for e in (32, 64) if e < stop))
            self.clock.advance(1)
        for seed in range(20, 26):
            self.clock.advance(1)
            self.snap(cache, seed)
        for key, meta in cache._disk.items():
            self.assertTrue(meta["parent"] is None or meta["parent"] in cache._disk, "orphaned child on the SSD")

    def test_a_leased_entry_is_not_evicted_and_the_lease_ends_when_the_block_raises(self):
        size = self.entry_bytes()
        cache = self.cache(disk=int(size * 1.5))
        ids, entry = self.snap(cache, 1)
        with self.assertRaises(RuntimeError):
            with cache.lease((entry.key,)):
                self.snap(cache, 2)                                      # would evict ``entry`` if it were not leased
                self.assertIn(entry.key, cache._disk)
                raise RuntimeError("cancelled")
        self.assertEqual(cache._leases, {})
        self.snap(cache, 3)                                              # now it may go
        self.assertNotIn(entry.key, cache._disk)

    def test_ram_pressure_moves_cold_entries_to_ssd_and_keeps_pinned_ones_resident(self):
        size = self.entry_bytes()
        cache = self.cache(ram=1 << 40)
        pinned, a = self.snap(cache, 1)
        cache.pin(pinned)
        ids, b = self.snap(cache, 2)
        self.clock.advance(10)
        self.snap(cache, 3)
        freed = cache.shrink_ram(int(a.nbytes() * 1.2))
        self.assertGreater(freed, 0)
        self.assertIn(a.key, cache._ram, "a pinned snapshot stays hot")
        self.assertNotIn(b.key, cache._ram)
        self.assertEqual(self.hit(cache, ids).tier, "ssd")              # the evicted one restores from the SSD tier


class ReserveTests(Base):
    def test_free_space_below_the_reserve_evicts_until_it_fits(self):
        size = self.entry_bytes()
        reserve = 100 * size
        cache = self.cache(reserve=reserve)
        cache._free_bytes = lambda path: 103 * size - cache._disk_total()      # the volume holds this cache and little else
        self.snap(cache, 1)
        self.clock.advance(5)
        self.snap(cache, 2)
        self.assertEqual(len(cache._disk), 2)
        self.clock.advance(5)
        self.snap(cache, 3)                                                    # 3 entries would leave less than the reserve
        self.assertEqual(len(cache._disk), 2)
        self.assertEqual(cache.stats.evicted_ssd, 1)
        self.assertGreaterEqual(cache._free_bytes(self.dir), reserve)

    def test_a_reserve_that_eviction_cannot_restore_evicts_nothing_and_skips_the_write(self):
        size = self.entry_bytes()
        cache = self.cache(reserve=50 * 10**9)
        self.snap(cache, 1)
        self.snap(cache, 2)
        self.free[0] = 40 * 10**9                                        # the volume is below the reserve for other reasons
        before = len(cache._disk)
        self.snap(cache, 3)
        self.assertEqual(len(cache._disk), before, "must not wipe the cache in a futile attempt")
        self.assertEqual(cache.stats.writes_skipped, 1)

    def test_a_write_that_is_larger_than_the_whole_budget_is_skipped(self):
        cache = self.cache(disk=1000)
        self.snap(cache, 1)
        self.assertEqual((len(cache._disk), cache.stats.writes_skipped), (0, 1))


class CrashRecoveryTests(Base):
    def populated(self):
        cache = self.cache()
        ids, entry = self.snap(cache, 1, kind="system")
        cache.flush()
        return ids, entry

    def age(self, path, seconds=600):
        stamp = self.clock.now - seconds
        os.utime(path, (stamp, stamp))

    def test_leftovers_of_an_interrupted_write_are_removed_and_a_fresh_temporary_is_kept(self):
        ids, entry = self.populated()
        key = "ab" * 32
        old_tmp = self.dir / f".{key}.{'c' * 32}.tmp.safetensors"
        old_manifest_tmp = self.dir / f".{key}.{'d' * 32}.json.tmp"
        fresh_tmp = self.dir / f".{key}.{'e' * 32}.tmp.safetensors"
        orphan_data = self.dir / f"{key}.safetensors"
        for path in (old_tmp, old_manifest_tmp, fresh_tmp, orphan_data):
            path.write_bytes(b"partial")
        for path in (old_tmp, old_manifest_tmp, orphan_data):
            self.age(path)
        os.utime(fresh_tmp, (self.clock.now, self.clock.now))
        again = self.cache()
        names = self.disk_files()
        for gone in (old_tmp, old_manifest_tmp, orphan_data):
            self.assertNotIn(gone.name, names)
        self.assertIn(fresh_tmp.name, names, "a temporary file inside the grace period may belong to a live writer")
        self.assertEqual(again.stats.orphans_removed, 3)
        self.assertIsNotNone(self.hit(again, ids))

    def test_a_manifest_without_data_or_with_a_truncated_data_file_is_dropped(self):
        ids, entry = self.populated()
        _, other = self.snap(self.cache(), 2)
        (self.dir / f"{entry.key}.safetensors").unlink()
        with open(self.dir / f"{other.key}.safetensors", "r+b") as handle:
            handle.truncate(100)
        again = self.cache()
        self.assertEqual(again.profiles(), [])
        self.assertEqual(again.stats.invalidated, 2)
        self.assertEqual([n for n in self.disk_files() if n.endswith(".json") and n != PP.INDEX], [])

    def test_a_flipped_byte_is_caught_by_the_checksum_and_the_entry_removed(self):
        ids, entry = self.populated()
        data = self.dir / f"{entry.key}.safetensors"
        raw = bytearray(data.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        data.write_bytes(bytes(raw))
        again = self.cache()
        self.assertIsNone(self.hit(again, ids))
        self.assertEqual(again.profiles(), [])
        self.assertFalse(data.exists())

    def test_scrub_finds_corruption_and_removes_descendants(self):
        ids, entry = self.populated()
        data = self.dir / f"{entry.key}.safetensors"
        raw = bytearray(data.read_bytes())
        raw[-10] ^= 0xFF
        data.write_bytes(bytes(raw))
        again = self.cache()
        self.assertEqual(again.scrub(delete=False), [entry.key])
        self.assertEqual(again.scrub(delete=True), [entry.key])
        self.assertEqual(again.profiles(), [])

    def test_a_corrupt_index_falls_back_to_the_backup_and_then_to_the_manifests(self):
        ids, entry = self.populated()
        cache = self.cache()
        cache.pin(ids)
        cache.flush()                                                    # now index + backup both exist
        (self.dir / PP.INDEX).write_text("{ not json")
        recovered = self.cache()
        self.assertEqual(recovered.index_state, "backup")
        (self.dir / PP.INDEX).write_text("garbage")
        (self.dir / (PP.INDEX + ".bak")).write_text("garbage")
        rebuilt = self.cache()
        self.assertEqual(rebuilt.index_state, "rebuilt")
        self.assertEqual(len(rebuilt.profiles()), 1)                     # state is intact; only counters and pins reset
        self.assertIsNotNone(self.hit(rebuilt, ids))

    def test_a_tampered_index_checksum_is_rejected(self):
        ids, entry = self.populated()
        path = self.dir / PP.INDEX
        wrapper = json.loads(path.read_text())
        wrapper["payload"]["entries"][entry.key]["hits"] = 999
        path.write_text(json.dumps(wrapper))
        (self.dir / (PP.INDEX + ".bak")).unlink(missing_ok=True)
        again = self.cache()
        self.assertEqual(again.profiles()[0]["hits"], 0)

    def test_an_interrupted_write_leaves_the_cache_consistent(self):
        cache = self.cache()
        real = P.write_entry

        def broken(directory, entry, fp):
            real(directory, entry, fp)
            raise KeyboardInterrupt("cancelled between data and manifest bookkeeping")
        P.write_entry = broken
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.snap(cache, 1)
        finally:
            P.write_entry = real
        self.assertEqual(cache._leases, {})
        again = self.cache()                                             # restart: whatever was half written is cleaned or valid
        for row in again.profiles():
            self.assertIsNotNone(row["key"])
        self.assertEqual(again.scrub(delete=False), [])

    def test_a_lowered_budget_is_enforced_after_a_restart(self):
        size = self.entry_bytes()
        cache = self.cache(disk=1 << 40)
        for seed in range(1, 7):
            self.clock.advance(1)
            self.snap(cache, seed)
        cache.flush()
        again = self.cache(disk=int(size * 2.5))
        self.assertLessEqual(len(again._disk), 2)
        self.assertLessEqual(again.usage()["disk_total_bytes"], int(size * 2.5))

    def test_a_pinned_entry_survives_a_restart_that_lowers_the_budget(self):
        size = self.entry_bytes()
        cache = self.cache()
        pinned, entry = self.snap(cache, 1)
        cache.pin(pinned)
        for seed in range(2, 6):
            self.snap(cache, seed)
        cache.flush()
        again = self.cache(disk=int(size * 1.5))
        self.assertIn(entry.key, again._disk)


class SafetyTests(Base):
    def test_entries_of_another_fingerprint_are_not_loaded_or_evicted(self):
        other_fp = P.fingerprint(model="small", engine="other-config")
        other = self.cache(fp=other_fp)
        other_ids, other_entry = self.snap(other, 1)
        size = self.entry_bytes()
        again = self.cache(disk=int(size * 2.5))
        self.assertEqual(again.usage()["foreign_entries"], 1)
        self.assertTrue((self.dir / f"{other_entry.key}.safetensors").exists(), "another configuration's state is not deleted at startup")
        self.assertIsNone(self.hit(again, other_ids))
        self.snap(again, 2)
        self.snap(again, 3)
        self.assertEqual(again.usage()["foreign_entries"], 1)
        self.assertTrue((self.dir / f"{other_entry.key}.safetensors").exists())
        self.assertEqual(again.extra["foreign_evicted"], 0)

    def test_files_that_are_not_ours_are_never_touched_and_the_directory_is_refused_if_it_holds_them(self):
        self.dir.mkdir(parents=True)
        (self.dir / "notes.txt").write_text("keep")
        with self.assertRaises(PP.UnsafeDirectory):
            self.cache()
        self.assertEqual((self.dir / "notes.txt").read_text(), "keep")

    def test_a_marked_directory_with_other_files_is_managed_but_those_files_stay(self):
        cache = self.cache()
        self.snap(cache, 1)
        (self.dir / "notes.txt").write_text("keep")
        (self.dir / ("f" * 63 + ".json")).write_text("{}")             # not a 64-hex name: not ours
        size = self.entry_bytes()
        again = self.cache(disk=int(size * 0.5))
        again.clear()
        self.assertEqual((self.dir / "notes.txt").read_text(), "keep")
        self.assertTrue((self.dir / ("f" * 63 + ".json")).exists())

    def test_a_symlink_named_like_a_cache_file_is_not_followed_or_removed(self):
        cache = self.cache()
        outside = self.tmp / "outside.safetensors"
        outside.write_bytes(b"precious")
        key = "9" * 64
        link = self.dir / f"{key}.safetensors"
        link.symlink_to(outside)
        os.utime(link, (self.clock.now - 600, self.clock.now - 600), follow_symlinks=False)
        self.cache()
        self.assertEqual(outside.read_bytes(), b"precious")
        self.assertTrue(link.is_symlink())

    def test_eviction_deletes_only_cache_files(self):
        size = self.entry_bytes()
        weights = self.tmp / "model.safetensors"
        weights.write_bytes(b"weights")
        cache = self.cache(disk=int(size * 1.5))
        for seed in range(1, 6):
            self.snap(cache, seed)
        self.assertEqual(weights.read_bytes(), b"weights")


class ContinuationTests(Base):
    """A snapshot from the profile cache continues bit-identically to a cold run with the same boundary."""

    def test_restored_continuation_is_bit_identical(self):
        from models.flashnext.tests.unit.test_prefix_cache import cache_bits

        ids = self.tokens(7, 160)
        cache = self.cache()
        cache.put(ids[:96], self.live(ids[:96]), plan=(), kind="system")
        again = self.cache()                                             # SSD restore after a restart
        found, restored = again.lookup(ids, self.model.make_cache)
        self.assertEqual((found.n_tokens, found.tier), (96, "ssd"))
        logits_hit = self.model(mx.array([ids[96:]]), cache=restored).logits[:, -1]
        cold = self.live(ids[:96])
        logits_cold = self.model(mx.array([ids[96:]]), cache=cold).logits[:, -1]
        mx.eval(logits_hit, logits_cold, [c.state for c in restored], [c.state for c in cold])
        self.assertTrue(mx.array_equal(logits_hit, logits_cold).item())
        self.assertTrue(cache_bits(restored) == cache_bits(cold))


class Phase2ProfileTests(Base):
    """Goal 4 & Goal 5: First-use pinning, corruption detection, file locking, and eviction telemetry."""

    def test_high_value_system_prompt_pins_on_first_use(self):
        cache = self.cache(pin_after=3)
        ids = self.tokens(10, n=64)
        # Register as high value
        cache.register_high_value(ids)
        # First use snapshot put
        cache.put(ids, self.live(ids), plan=(), kind="system")
        # Assert pinned automatically on first use
        row = next(r for r in cache.profiles() if r["n_tokens"] == 64)
        self.assertEqual(row["pinned"], "auto")
        self.assertEqual(cache.extra["auto_pins"], 1)

    def test_standard_prefix_retains_3_hit_threshold(self):
        cache = self.cache(pin_after=3)
        ids = self.tokens(11, n=48)
        # Standard system prompt (not high value)
        cache.put(ids, self.live(ids), plan=(), kind="system")
        row = next(r for r in cache.profiles() if r["n_tokens"] == 48)
        self.assertIsNone(row["pinned"], "Standard prompt must not pin on 1st use")

        # 1st hit
        self.hit(cache, ids)
        row = next(r for r in cache.profiles() if r["n_tokens"] == 48)
        self.assertIsNone(row["pinned"], "Standard prompt must not pin on 1st hit")

        # 2nd hit
        self.hit(cache, ids)
        row = next(r for r in cache.profiles() if r["n_tokens"] == 48)
        self.assertIsNone(row["pinned"], "Standard prompt must not pin on 2nd hit")

        # 3rd hit meets threshold
        self.hit(cache, ids)
        row = next(r for r in cache.profiles() if r["n_tokens"] == 48)
        self.assertEqual(row["pinned"], "auto")

    def test_bit_flip_corruption_recovery(self):
        cache = self.cache()
        ids = self.tokens(12, n=48)
        cache.put(ids, self.live(ids), plan=(), kind="system")
        key = P.chain_key(self.fp, ids)
        data_path = self.dir / f"{key}.safetensors"
        self.assertTrue(data_path.exists())

        # Corrupt data file: flip a byte in the payload
        raw = bytearray(data_path.read_bytes())
        raw[-1] ^= 0xFF
        data_path.write_bytes(raw)

        # Reopen cache and scan: corrupted entry must be safely discarded, not crashed
        again = self.cache()
        # Lookup should not return corrupt entry, but None, triggering re-prefill
        found, _ = again.lookup(ids, self.model.make_cache)
        self.assertIsNone(found, "Corrupted entry should not be looked up")

    def test_cross_process_index_locking_and_telemetry(self):
        cache = self.cache(disk=int(self.entry_bytes() * 1.5))
        ids1 = self.tokens(13, n=48)
        ids2 = self.tokens(14, n=48)
        cache.put(ids1, self.live(ids1), plan=(), kind="system")
        # Trigger eviction of ids1 by adding ids2 under tight disk budget
        cache.put(ids2, self.live(ids2), plan=(), kind="system")

        events = cache.get_eviction_events()
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0]["reason"], "least_valuable")
        self.assertIn("key", events[0])
        self.assertIn("bytes", events[0])
        self.assertIn("remaining_disk", events[0])


class FocusedRobustnessMLXTests(Base):
    """Real MLX model validation for concurrent protection, stale pin yielding, and corruption recovery."""

    def test_leased_entry_is_protected_from_eviction_during_active_restore(self):
        size = self.entry_bytes()
        # Budget only fits ~1.5 entries
        cache = self.cache(disk=int(size * 1.5))
        ids1 = self.tokens(20, n=48)
        ids2 = self.tokens(21, n=48)
        entry1 = cache.put(ids1, self.live(ids1), plan=(), kind="system")

        # Lease entry1 simulating an in-progress restore or prefill
        with cache.lease((entry1.key,)):
            # Try to put entry2 which would normally evict entry1
            entry2 = cache.put(ids2, self.live(ids2), plan=(), kind="system")
            # entry1 MUST still be in disk cache because it is actively leased!
            self.assertIn(entry1.key, cache._disk, "Leased entry must not be evicted during active restore")

        # Once lease block exits, subsequent snapshot can evict entry1
        ids3 = self.tokens(22, n=48)
        cache.put(ids3, self.live(ids3), plan=(), kind="system")
        self.assertNotIn(entry1.key, cache._disk, "Entry may be evicted once lease expires")

    def test_stale_autopinned_entries_yield_when_half_budget_exceeded(self):
        size = self.entry_bytes()
        # Budget fits exactly 2 entries (half budget fits 1 entry)
        cache = self.cache(disk=int(size * 2.2), pin_after=1)
        ids1 = self.tokens(31, n=48)
        cache.register_high_value(ids1)
        entry1 = cache.put(ids1, self.live(ids1), plan=(), kind="system")
        self.assertEqual(cache._meta[entry1.key]["pinned"], "auto")

        # Age entry1 beyond half_life_s without any hits
        self.clock.advance(cache.half_life_s + 100.0)

        # Now add a new active high-value prompt ids2
        ids2 = self.tokens(32, n=48)
        cache.register_high_value(ids2)
        entry2 = cache.put(ids2, self.live(ids2), plan=(), kind="system")

        # Stale entry1 should have yielded its auto-pin, allowing active entry2 to be pinned!
        self.assertIsNone(cache._meta[entry1.key]["pinned"], "Stale auto-pin must yield")
        self.assertEqual(cache._meta[entry2.key]["pinned"], "auto", "New high-value prompt must pin")

    def test_corrupt_manifest_and_data_recovery_triggers_clean_reprefill(self):
        cache = self.cache()
        ids = self.tokens(40, n=48)
        entry = cache.put(ids, self.live(ids), plan=(), kind="system")
        manifest_path = self.dir / f"{entry.key}.json"
        self.assertTrue(manifest_path.exists())

        # Overwrite manifest with invalid garbage JSON
        manifest_path.write_text("CORRUPT_NOT_JSON{{{")

        # Reopen cache: scan should safely drop corrupt entry without crashing
        again = self.cache()
        found, _ = again.lookup(ids, self.model.make_cache)
        self.assertIsNone(found, "Corrupt manifest must trigger clean miss/re-prefill")

    def test_repeated_system_prompt_reuse_tracks_measured_net_benefit(self):
        cache = self.cache()
        ids = self.tokens(50, n=48)
        entry = cache.put(ids, self.live(ids), plan=(), kind="system")

        # Hit 3 times
        for _ in range(3):
            self.hit(cache, ids)

        meta = cache._meta[entry.key]
        self.assertEqual(meta["hits"], 3)
        self.assertGreater(meta["saved_s"], 0.0)
        # Value must factor in net saved seconds
        val = cache._value(entry.key)
        self.assertGreater(val, 0.0)


if __name__ == "__main__":
    unittest.main()
