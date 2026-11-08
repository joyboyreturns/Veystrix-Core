"""Exact prefix-state cache: capture and restore, chained deltas, two tiers, corruption.

The model is a small random-weight Qwen4 language model (3 GatedDeltaNet + 1 QSA layers twice, one PLE
layer), so the real cache classes and the real continuation path are exercised without a checkpoint.
The claim under test: a restored prefix continues bit-identically to a cold run that was stopped at
the same boundary, and a stored snapshot can never be changed by later generation.
"""
from __future__ import annotations

import os
import gc
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from models.flashnext import prefix_cache as P

VOCAB = 512


def build_model():
    from mlx_vlm.models.qwen4_exp.config import TextConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    mx.random.seed(3)
    config = TextConfig(
        model_type="qwen4_exp_text", hidden_size=128, num_hidden_layers=8, num_attention_heads=4,
        linear_num_value_heads=8, linear_num_key_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2, shared_expert_intermediate_size=32,
        moe_intermediate_size=32, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
        max_position_embeddings=65536, head_dim=64, full_attention_interval=4, indexer_budget=64,
        indexer_compress_ratio=4, indexer_head_dim=32, indexer_n_heads=2, hc_lowrank=32, ple_layer_ids=[2],
        ple_embed_dim=128, eos_token_id=1, ngram_vocab_size_base=1000, split_ngram_parts=4, heads_per_ngram=2)
    stub = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=-1,
                           video_token_id=-2, vision_start_token_id=-3, text_config=config)
    model = LanguageModel(config, stub)
    mx.eval(model.parameters())
    return model


def bits(array) -> bytes:
    flat = array.reshape(-1)
    return bytes(memoryview(__import__("numpy").array(flat.view(mx.uint8) if flat.dtype != mx.uint8 else flat)))


def cache_bits(caches):
    """Every array in every cache, as bytes, for exact comparison."""
    out = []
    for cache in caches:
        kind = P.kind_of(cache)
        arrays = cache.cache if kind == "arrays" else [
            cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :],
            cache.index_keys, cache.index_position_ids]
        out.append([None if a is None else (tuple(a.shape), str(a.dtype), bits(a)) for a in arrays])
    return out


class Base(unittest.TestCase):
    model = None

    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        mx.random.seed(11)
        cls.ids = [int(t) for t in mx.random.randint(2, VOCAB, (240,)).tolist()]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.fp = P.fingerprint(model="small", engine="test")

    def prefill(self, tokens, caches):
        out = self.model(mx.array([tokens]), cache=caches)
        mx.eval(out.logits, [c.state for c in caches])
        return out.logits[:, -1]

    def cold(self, upto, then=None):
        caches = self.model.make_cache()
        self.prefill(self.ids[:upto], caches)
        return caches

    def make_cache(self):
        return self.model.make_cache()

    def same(self, a, b, message=""):
        """Bitwise equality of two cache bit-dumps, without unittest building a diff of megabytes."""
        self.assertTrue(a == b, message or "cache state differs bitwise")

    def run_chunks(self, cache, bounds):
        """One prefill stopped at each boundary, snapshotting there: how real snapshots are taken.

        Returns the live cache at the last boundary. (Snapshots from separate runs of different
        lengths are not bit-compatible: a token's KV depends on the length of the call that made it.)
        """
        caches = self.model.make_cache()
        start = 0
        for bound in bounds:
            self.prefill(self.ids[start:bound], caches)
            cache.put(self.ids[:bound], caches)
            start = bound
        return caches

    def cache(self, directory=None, ram=1 << 40, disk=0, **kw):
        return P.PrefixCache(self.fp, ram, directory if directory is not None else None, disk, **kw)


class CaptureRestoreTests(Base):
    def test_restore_is_bit_identical_to_the_captured_state(self):
        caches = self.cold(160)
        entry = P.capture(caches, self.ids[:160], self.fp)
        entry._tokens = tuple(self.ids[:160])
        restored = P.restore([entry], self.make_cache)
        self.same(cache_bits(restored), cache_bits(caches))

    def test_state_carries_every_required_part(self):
        caches = self.cold(160)
        entry = P.capture(caches, self.ids[:160], self.fp)
        kinds = [layer.kind for layer in entry.layers]
        self.assertEqual(kinds, ["arrays"] * 3 + ["qsa_kv"] + ["arrays"] * 3 + ["qsa_kv"])
        kv = entry.layers[3].kv
        self.assertEqual(set(kv), set(P.KV_FIELDS))
        self.assertEqual(kv["keys"].shape[2], 160)
        self.assertEqual(kv["index_keys"].shape[1], 160)
        self.assertEqual(kv["index_position_ids"].shape[-1], 160)
        # the PLE layer carries more than the usual two recurrent slots
        self.assertGreater(max(len(layer.states) for layer in entry.layers if layer.kind == "arrays"), 2)

    def test_continuation_matches_a_cold_run_stopped_at_the_same_boundary(self):
        boundary = 128
        cold = self.cold(boundary)
        logits_cold = self.prefill(self.ids[boundary:], cold)
        cache = self.cache()
        cache.put(self.ids[:boundary], self.cold(boundary))
        hit, restored = cache.lookup(self.ids, self.make_cache)
        self.assertEqual(hit.n_tokens, boundary)
        logits_hit = self.prefill(self.ids[hit.n_tokens:], restored)
        self.assertTrue(mx.array_equal(logits_cold, logits_hit).item())
        self.same(cache_bits(restored), cache_bits(cold))

    def test_a_fresh_engine_needs_the_language_position_state(self):
        """A restart builds a new model object. Its first call on a restored cache must keep counting."""
        boundary = 128
        cold = self.cold(boundary)
        cold_logits = self.prefill(self.ids[boundary:200], cold)
        cache = self.cache()
        cache.put(self.ids[:boundary], self.cold(boundary), language=self.model)
        fresh = build_model()                                   # same weights, no position state yet
        self.assertIsNone(fresh._rope_deltas)

        def continue_on(model, apply):
            hit, caches = cache.lookup(self.ids, model.make_cache, language=model if apply else None)
            out = model(mx.array([self.ids[hit.n_tokens:200]]), cache=caches)
            mx.eval(out.logits, [c.state for c in caches])
            return out.logits[:, -1], caches

        def positions(caches):
            layer = next(c for c in caches if P.kind_of(c) == "qsa_kv")
            return layer.index_position_ids[..., -72:].reshape(-1)[-72:].tolist()

        with_state, with_caches = continue_on(fresh, True)
        self.assertTrue(mx.array_equal(cold_logits, with_state).item())
        self.assertEqual(positions(with_caches), list(range(128, 200)), "tokens keep counting from the prefix")
        bare = build_model()
        _, bare_caches = continue_on(bare, False)
        self.assertEqual(positions(bare_caches), list(range(0, 72)),
                         "without the stored position state a fresh engine numbers the suffix from 0")

    def test_language_state_survives_the_ssd_tier(self):
        directory = self.tmp / "c"
        self.cache(directory).put(self.ids[:128], self.cold(128), language=self.model)
        reopened = self.cache(directory)
        fresh = build_model()
        hit, _ = reopened.lookup(self.ids, fresh.make_cache, language=fresh)
        self.assertEqual(hit.tier, "ssd")
        self.assertTrue(mx.array_equal(fresh._rope_deltas, self.model._rope_deltas).item())
        self.assertEqual(fresh._position_ids.ndim, self.model._position_ids.ndim)

    def test_decode_after_restore_matches_decode_after_cold(self):
        boundary = 128
        cold = self.cold(boundary)
        self.prefill(self.ids[boundary:200], cold)
        cache = self.cache()
        cache.put(self.ids[:boundary], self.cold(boundary))
        _, restored = cache.lookup(self.ids, self.make_cache)
        self.prefill(self.ids[boundary:200], restored)
        for step in range(5):
            token = self.ids[200 + step]
            a = self.prefill([token], cold)
            b = self.prefill([token], restored)
            self.assertTrue(mx.array_equal(a, b).item(), f"decode step {step}")
        self.same(cache_bits(restored), cache_bits(cold))

    def test_stored_snapshot_cannot_be_changed_by_later_generation(self):
        cache = self.cache()
        cache.put(self.ids[:128], self.cold(128))
        entry = next(iter(cache._ram.values()))
        before = [[(tuple(a.shape), bits(a)) for a in (*l.states, *l.kv.values()) if a is not None] for l in entry.layers]
        _, restored = cache.lookup(self.ids, self.make_cache)
        self.prefill(self.ids[128:200], restored)
        for token in self.ids[200:210]:
            self.prefill([token], restored)
        after = [[(tuple(a.shape), bits(a)) for a in (*l.states, *l.kv.values()) if a is not None] for l in entry.layers]
        self.assertEqual(before, after)
        _, again = cache.lookup(self.ids[:129], self.make_cache)
        self.same(cache_bits(again), cache_bits(self.cold(128)))

    def test_snapshot_and_restored_caches_never_share_array_objects(self):
        live = self.cold(128)
        entry = P.capture(live, self.ids[:128], self.fp)
        entry._tokens = tuple(self.ids[:128])
        stored = [[bits(a) for a in l.states if a is not None] for l in entry.layers if l.kind == "arrays"]
        for cache, layer in zip(live, entry.layers):
            if layer.kind == "arrays":
                for mine, theirs in zip(cache.cache, layer.states):
                    self.assertIsNot(mine, theirs)
        for cache in live:                       # MLX updates an array object in place: do it to the live one
            if P.kind_of(cache) == "arrays":
                for array in cache.cache:
                    array[...] = 0
        after = [[bits(a) for a in l.states if a is not None] for l in entry.layers if l.kind == "arrays"]
        self.assertTrue(stored == after, "changing the live cache must not change the snapshot")
        restored = P.restore([entry], self.make_cache)
        for cache, layer in zip(restored, entry.layers):
            if layer.kind == "arrays":
                for mine, theirs in zip(cache.cache, layer.states):
                    self.assertIsNot(mine, theirs)

    def test_unsupported_cache_types_fail_closed(self):
        class Rotating:
            pass

        with self.assertRaises(P.UnsupportedCache):
            P.capture([Rotating()], [1, 2, 3], self.fp)

    def test_cache_and_token_count_must_agree(self):
        with self.assertRaises(ValueError):
            P.capture(self.cold(100), self.ids[:99], self.fp)


class ChainTests(Base):
    def test_chained_snapshots_store_only_the_kv_delta(self):
        cache = self.cache()
        self.run_chunks(cache, [64, 128, 200])
        a, b, c = (cache._ram[P.chain_key(self.fp, tuple(self.ids[:n]))] for n in (64, 128, 200))
        self.assertEqual((a.parent, b.parent, c.parent), (None, a.key, b.key))
        self.assertEqual((a.n_tokens, b.n_tokens, c.n_tokens), (64, 128, 200))
        deltas = [e.layers[3].kv["keys"].shape[2] for e in (a, b, c)]
        self.assertEqual(deltas, [64, 64, 72])
        self.assertEqual(c.token_delta, tuple(self.ids[128:200]))
        # recurrent state is a full copy in every snapshot, KV is not
        self.assertEqual(len(c.layers[0].states), len(a.layers[0].states))

    def test_restoring_a_chain_equals_the_live_cache(self):
        cache = self.cache()
        live = self.run_chunks(cache, [64, 128, 200])
        hit, restored = cache.lookup(self.ids, self.make_cache)
        self.assertEqual(hit.n_tokens, 200)
        self.assertEqual(len(hit.chain), 3)
        self.same(cache_bits(restored), cache_bits(live))

    def test_longest_valid_prefix_wins_and_it_is_strictly_shorter(self):
        cache = self.cache()
        for n in (64, 128, 200):
            cache.put(self.ids[:n], self.cold(n))
        hit, _ = cache.lookup(self.ids[:200], self.make_cache)
        self.assertEqual(hit.n_tokens, 128, "an exact match must leave one token to prefill")
        hit, _ = cache.lookup(self.ids[:201], self.make_cache)
        self.assertEqual(hit.n_tokens, 200)
        hit, _ = cache.lookup(self.ids[:65], self.make_cache)
        self.assertEqual(hit.n_tokens, 64)

    def test_a_changed_token_inside_the_prefix_is_a_miss(self):
        cache = self.cache()
        cache.put(self.ids[:128], self.cold(128))
        changed = list(self.ids)
        changed[100] = (changed[100] + 1) % VOCAB
        hit, caches = cache.lookup(changed, self.make_cache)
        self.assertIsNone(hit)
        self.assertIsNone(caches)

    def test_a_change_after_the_boundary_still_hits_the_boundary(self):
        cache = self.cache()
        cache.put(self.ids[:128], self.cold(128))
        other = self.ids[:128] + [7, 8, 9, 10]
        hit, _ = cache.lookup(other, self.make_cache)
        self.assertEqual(hit.n_tokens, 128)

    def test_parent_must_be_a_prefix(self):
        cache = self.cache()
        a = cache.put(self.ids[:64], self.cold(64))
        other = [(t + 1) % VOCAB for t in self.ids[:128]]
        with self.assertRaises(ValueError):
            P.capture(self.cold(128), other, self.fp, a)

    def test_statistics(self):
        cache = self.cache()
        cache.put(self.ids[:128], self.cold(128))
        cache.lookup(self.ids[:200], self.make_cache)
        cache.lookup([5] * 50, self.make_cache)
        s = cache.stats
        self.assertEqual((s.lookups, s.hits, s.ram_hits, s.reused_tokens), (2, 1, 1, 128))
        self.assertEqual(s.missed_tokens, (200 - 128) + 50)


class TierTests(Base):
    def test_ram_eviction_spills_to_disk_and_reloads_bit_identically(self):
        entry_size = P.capture(self.cold(64), self.ids[:64], self.fp).nbytes()
        cache = self.cache(self.tmp / "c", ram=int(entry_size * 2.5), disk=1 << 40)
        live = self.run_chunks(cache, [64, 128, 200])
        self.assertGreater(cache.stats.evicted_ram, 0)
        before = cache.stats.ssd_writes
        hit, restored = cache.lookup(self.ids[:201], self.make_cache)
        self.assertEqual(hit.n_tokens, 200)
        self.same(cache_bits(restored), cache_bits(live))
        self.assertEqual(cache.stats.ssd_writes, before, "an unchanged entry must not be rewritten")

    def test_a_restart_restores_from_ssd(self):
        directory = self.tmp / "c"
        first = self.cache(directory)
        live = self.run_chunks(first, [128, 200])
        second = self.cache(directory)
        hit, restored = second.lookup(self.ids[:230], self.make_cache)
        self.assertEqual((hit.n_tokens, hit.tier), (200, "ssd"))
        self.assertGreater(hit.bytes_loaded, 0)
        self.same(cache_bits(restored), cache_bits(live))
        again, _ = second.lookup(self.ids[:230], self.make_cache)
        self.assertEqual(again.tier, "ram")

    def test_identical_puts_write_once(self):
        cache = self.cache(self.tmp / "c")
        for _ in range(3):
            cache.put(self.ids[:128], self.cold(128))
        self.assertEqual((cache.stats.ssd_writes, cache.stats.deduped_puts), (1, 2))

    def test_files_are_private(self):
        cache = self.cache(self.tmp / "c")
        cache.put(self.ids[:128], self.cold(128))
        directory = self.tmp / "c"
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        files = list(directory.iterdir())
        self.assertEqual(len(files), 3)
        for path in files:
            self.assertEqual(path.stat().st_mode & 0o777, 0o600, path.name)

    def test_disk_budget_evicts_leaves_first_and_keeps_chains_whole(self):
        size = P.capture(self.cold(64), self.ids[:64], self.fp).nbytes()
        cache = self.cache(self.tmp / "c", disk=int(size * 2.2))
        for n in (64, 128, 200):
            cache.put(self.ids[:n], self.cold(n))
        self.assertGreater(cache.stats.evicted_ssd, 0)
        known = {k: m["parent"] for k, m in cache._disk.items()}
        for key, parent in known.items():
            self.assertTrue(parent is None or parent in known, "a child must never outlive its parent")

    def test_incompatible_fingerprint_is_not_used_or_removed(self):
        directory = self.tmp / "c"
        self.cache(directory).put(self.ids[:128], self.cold(128))
        other = P.PrefixCache(P.fingerprint(model="other"), 1 << 40, directory)
        self.assertEqual(other.stats.invalidated, 0)
        self.assertEqual(len(list(directory.glob("*.json"))), 1)
        hit, _ = other.lookup(self.ids, self.make_cache)
        self.assertIsNone(hit)

    def test_leftover_temporary_files_are_removed_at_start(self):
        directory = self.tmp / "c"
        directory.mkdir()
        (directory / ("." + "a" * 64 + "." + "b" * 32 + ".tmp.safetensors")).write_bytes(b"partial")
        (directory / ("." + "a" * 64 + "." + "b" * 32 + ".json.tmp")).write_text("{")
        self.cache(directory)
        self.assertEqual([p.name for p in directory.iterdir()], [P.OWNER_LOCK])

    def test_unrelated_files_are_not_removed_during_scan(self):
        directory = self.tmp / "unrelated"
        directory.mkdir()
        (directory / "notes.json").write_text("not a cache manifest")
        (directory / ".other.tmp").write_text("keep")
        self.cache(directory)
        self.assertEqual((directory / "notes.json").read_text(), "not a cache manifest")
        self.assertEqual((directory / ".other.tmp").read_text(), "keep")


class CorruptionTests(Base):
    def prepared(self):
        directory = self.tmp / "c"
        writer = self.cache(directory)
        writer.put(self.ids[:128], self.cold(128))
        entry = writer.put(self.ids[:200], self.cold(200))
        return directory, entry

    def lookup_after_restart(self, directory, n=230):
        return self.cache(directory).lookup(self.ids[:n], self.make_cache)

    def test_a_flipped_byte_is_a_miss_for_that_entry_and_falls_back(self):
        directory, entry = self.prepared()
        path = directory / f"{entry.key}.safetensors"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
        hit, restored = self.lookup_after_restart(directory)
        self.assertEqual(hit.n_tokens, 128, "the damaged leaf is dropped; its intact parent is used")
        self.same(cache_bits(restored), cache_bits(self.cold(128)))
        self.assertFalse(path.exists())

    def test_a_truncated_file_is_removed_at_start(self):
        directory, entry = self.prepared()
        path = directory / f"{entry.key}.safetensors"
        path.write_bytes(path.read_bytes()[:1000])
        reopened = self.cache(directory)
        self.assertGreaterEqual(reopened.stats.invalidated, 1)
        hit, _ = reopened.lookup(self.ids[:230], self.make_cache)
        self.assertEqual(hit.n_tokens, 128)

    def test_a_damaged_parent_removes_its_children(self):
        directory, entry = self.prepared()
        parent = entry.parent
        (directory / f"{parent}.safetensors").write_bytes(b"x")
        reopened = self.cache(directory)
        hit, _ = reopened.lookup(self.ids[:230], self.make_cache)
        self.assertIsNone(hit)
        self.assertEqual(list(directory.glob("*.json")), [])

    def test_a_missing_manifest_leaves_no_usable_entry(self):
        directory, entry = self.prepared()
        (directory / f"{entry.key}.json").unlink()
        hit, _ = self.lookup_after_restart(directory)
        self.assertEqual(hit.n_tokens, 128)

    def test_tampered_tokens_are_rejected(self):
        directory, entry = self.prepared()
        manifest = directory / f"{entry.key}.json"
        text = manifest.read_text().replace(entry.key[:8], "deadbeef", 1) if False else manifest.read_text()
        import json

        content = json.loads(text)
        content["tokens_sha256"] = "0" * 64
        manifest.write_text(json.dumps(content))
        hit, _ = self.lookup_after_restart(directory)
        self.assertEqual(hit.n_tokens, 128)

    def test_reading_with_the_wrong_fingerprint_is_refused(self):
        directory, entry = self.prepared()
        with self.assertRaises(P.CorruptEntry):
            P.read_entry(directory, entry.key, P.fingerprint(model="other"))
        P.read_entry(directory, entry.key, self.fp)          # the right one still reads

    def test_a_key_collision_cannot_return_the_wrong_prefix(self):
        cache = self.cache()
        other = [(t + 1) % VOCAB for t in self.ids[:128]]
        original = P.chain_key
        P.chain_key = lambda fp, tokens: "k" if len(tokens) == 128 else original(fp, tokens)
        self.addCleanup(setattr, P, "chain_key", original)
        cache.put(self.ids[:128], self.cold(128))
        hit, _ = cache.lookup(other + [3, 4], self.make_cache)
        self.assertIsNone(hit, "same key, different tokens: must be a miss")

    def test_layer_structure_mismatch_is_rejected_not_applied(self):
        cache = self.cache()
        cache.put(self.ids[:128], self.cold(128))
        short = lambda: self.model.make_cache()[:-1]
        hit, caches = cache.lookup(self.ids, short)
        self.assertIsNone(hit)
        self.assertEqual(cache.usage()["ram_entries"], 0, "the incompatible entry is dropped")


class ProvenanceTests(Base):
    """Chunk-plan provenance: stored with every entry, checked by lookup, never silently dropped."""

    def put_plan(self, cache, n, plan, language=None):
        return cache.put(self.ids[:n], self.cold(n), plan=plan)

    def test_the_plan_is_recorded_and_survives_a_restart_in_manifest_and_data_file(self):
        directory = self.tmp / "c"
        writer = self.cache(directory)
        entry = self.put_plan(writer, 128, (64,))
        self.assertEqual(entry.plan, (64,))
        manifest = P.read_manifest(directory, entry.key)
        self.assertEqual(manifest["plan_edges"], [64])
        reopened = self.cache(directory)
        self.assertEqual(reopened._plans[entry.key], (64,))
        seen = []
        hit, _ = reopened.lookup(self.ids[:200], self.make_cache, accept=lambda n, plan: seen.append((n, plan)) or True)
        self.assertEqual((hit.n_tokens, seen), (128, [(128, (64,))]))
        self.assertEqual(P.read_entry(directory, entry.key, self.fp).plan, (64,))

    def test_a_veto_skips_that_entry_for_the_caller_and_keeps_it_stored(self):
        cache = self.cache()
        self.put_plan(cache, 64, ())
        self.put_plan(cache, 128, (64,))
        hit, _ = cache.lookup(self.ids[:200], self.make_cache, accept=lambda n, plan: n != 128)
        self.assertEqual(hit.n_tokens, 64)
        self.assertEqual(cache.stats.rejected_plan, 1)
        hit, _ = cache.lookup(self.ids[:200], self.make_cache)
        self.assertEqual(hit.n_tokens, 128, "a bare lookup, or another plan, can still use it")

    def test_peek_reports_the_reusable_length_without_touching_anything(self):
        directory = self.tmp / "pk"
        cache = self.cache(directory)
        self.put_plan(cache, 64, ())
        self.put_plan(cache, 128, (64,))
        before = dict(cache.stats.__dict__)
        self.assertEqual(cache.peek(self.ids[:200]), 128)
        self.assertEqual(cache.peek(self.ids[:128]), 64, "strictly shorter, like lookup")
        self.assertEqual(cache.peek(self.ids[:200], accept=lambda n, plan: n != 128), 64)
        self.assertEqual(cache.peek([1, 2, 3]), 0)
        self.assertEqual(cache.stats.__dict__, before, "no lookup, hit, veto or byte counted")
        reopened = self.cache(directory)                       # entries only on SSD: still no load
        self.assertEqual(reopened.peek(self.ids[:200]), 128)
        self.assertEqual(reopened.stats.ssd_bytes_read, 0)
        hit, _ = reopened.lookup(self.ids[:200], self.make_cache)
        self.assertEqual(hit.n_tokens, 128, "peek agreed with the lookup that follows")

    def test_shrink_ram_drops_the_ram_copy_only_and_the_ssd_copy_restores_identically(self):
        directory = self.tmp / "sh"
        cache = self.cache(directory)
        self.put_plan(cache, 64, ())
        self.put_plan(cache, 128, (64,))
        hit_ram, caches_ram = cache.lookup(self.ids[:200], self.make_cache)
        self.assertEqual(hit_ram.tier, "ram")
        freed = cache.shrink_ram(0)
        self.assertGreater(freed, 0)
        self.assertEqual(cache.usage()["ram_bytes"], 0)
        self.assertEqual(cache.usage()["disk_entries"], 2, "nothing was discarded")
        hit_ssd, caches_ssd = cache.lookup(self.ids[:200], self.make_cache)
        self.assertEqual((hit_ssd.tier, hit_ssd.n_tokens), ("ssd", 128))
        self.assertTrue(cache_bits(caches_ram) == cache_bits(caches_ssd), "SSD restore is bit-identical to the RAM copy")

    def test_shrink_ram_does_nothing_without_an_ssd_tier(self):
        cache = self.cache(None)
        self.put_plan(cache, 64, ())
        before = cache.usage()["ram_bytes"]
        self.assertEqual(cache.shrink_ram(0), 0)
        self.assertEqual(cache.usage()["ram_bytes"], before, "dropping it would lose the snapshot")

    def test_an_entry_without_a_plan_is_unknown_history(self):
        cache = self.cache()
        entry = cache.put(self.ids[:128], self.cold(128))
        self.assertIsNone(entry.plan)
        seen = []
        cache.lookup(self.ids[:200], self.make_cache, accept=lambda n, plan: seen.append(plan) or False)
        self.assertEqual(seen, [None])

    def test_a_put_at_known_tokens_with_another_plan_keeps_the_old_entry_and_counts_it(self):
        cache = self.cache()
        first = self.put_plan(cache, 128, (64,))
        second = self.put_plan(cache, 128, (32, 64))
        self.assertIs(first, second)
        self.assertEqual(cache.stats.plan_conflicts, 1)
        self.assertEqual(cache._plans[first.key], (64,))
        self.put_plan(cache, 128, (64,))
        self.assertEqual(cache.stats.plan_conflicts, 1, "the same plan is an ordinary dedupe")

    def test_invalid_plans_are_refused(self):
        for plan in ((0,), (128,), (64, 64), (96, 64), (-3,)):
            with self.assertRaises(ValueError, msg=str(plan)):
                P.capture(self.cold(128), self.ids[:128], self.fp, plan=plan)

    def test_a_manifest_that_disagrees_with_the_checksummed_file_is_corrupt(self):
        directory = self.tmp / "c"
        writer = self.cache(directory)
        entry = self.put_plan(writer, 128, (64,))
        path = directory / f"{entry.key}.json"
        manifest = json.loads(path.read_text())
        manifest["plan_edges"] = [32]                              # an attacker or a bug edits the manifest
        path.write_text(json.dumps(manifest, sort_keys=True))
        reopened = self.cache(directory)
        hit, _ = reopened.lookup(self.ids[:200], self.make_cache)
        self.assertIsNone(hit, "the entry is invalidated, not trusted")
        self.assertGreaterEqual(reopened.stats.invalidated, 1)

    def test_execution_critical_manifest_changes_are_rejected(self):
        directory = self.tmp / "binding"
        cache = self.cache(directory)
        entry = self.put_plan(cache, 128, ())
        path = directory / f"{entry.key}.json"
        original = json.loads(path.read_text())
        for field, value in (("parent", "0" * 64), ("n_tokens", 127),
                             ("parent_tokens", 1), ("layers", []), ("lang", ["wrong"]),
                             ("tensor_spec", {}), ("tokens_sha256", "0" * 64),
                             ("fingerprint", "wrong"), ("plan_edges", [32])):
            with self.subTest(field=field):
                changed = dict(original)
                changed[field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(P.CorruptEntry):
                    P.read_entry(directory, entry.key, self.fp)
        path.write_text(json.dumps(original))

    def test_modified_disk_entry_is_reverified_after_an_earlier_restore(self):
        directory = self.tmp / "modified"
        cache = self.cache(directory)
        entry = self.put_plan(cache, 128, ())
        cache.shrink_ram(0)
        hit, _ = cache.lookup(self.ids[:200], self.make_cache)
        self.assertEqual(hit.tier, "ssd")
        cache.shrink_ram(0)
        path = directory / f"{entry.key}.safetensors"
        with path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([byte[0] ^ 1]))
        hit, _ = cache.lookup(self.ids[:200], self.make_cache)
        self.assertIsNone(hit)
        self.assertGreaterEqual(cache.stats.invalidated, 1)

    def test_replacement_during_load_cannot_change_verified_bytes(self):
        directory = self.tmp / "replace-during-load"
        cache = self.cache(directory)
        entry = self.put_plan(cache, 128, ())
        path = directory / f"{entry.key}.safetensors"
        real_load = mx.load

        def replace_original_then_load(handle, *args, **kwargs):
            replacement = directory / "replacement"
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 1
            replacement.write_bytes(blob)
            os.replace(replacement, path)
            return real_load(handle, *args, **kwargs)

        with mock.patch.object(P.mx, "load", side_effect=replace_original_then_load):
            loaded = P.read_entry(directory, entry.key, self.fp)
        self.assertEqual(loaded.key, entry.key)
        with self.assertRaises(P.CorruptEntry):
            P.read_entry(directory, entry.key, self.fp)

    def test_an_older_format_without_provenance_is_discarded_at_start(self):
        directory = self.tmp / "c"
        writer = self.cache(directory)
        entry = self.put_plan(writer, 128, ())
        path = directory / f"{entry.key}.json"
        manifest = json.loads(path.read_text())
        manifest["format"] = "1"
        manifest.pop("plan_edges")
        path.write_text(json.dumps(manifest, sort_keys=True))
        reopened = self.cache(directory)
        self.assertEqual(list(reopened._disk), [])
        self.assertEqual(len(list(directory.glob("*.safetensors"))), 1)


class ProcessOwnershipTests(unittest.TestCase):
    def test_other_process_cannot_open_until_owner_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = P.PrefixCache("owner", 0, directory)
            code = ("import sys; from models.flashnext.prefix_cache import PrefixCache, CacheOwnershipError\n"
                    "try:\n PrefixCache('other', 0, sys.argv[1])\n"
                    "except CacheOwnershipError:\n sys.exit(23)\n")
            child = subprocess.run([sys.executable, "-c", code, directory], capture_output=True, text=True)
            self.assertEqual(child.returncode, 23, child.stderr)
            del cache
            gc.collect()
            child = subprocess.run([sys.executable, "-c", code, directory], capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_process_termination_releases_owner_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ("import sys, time; from models.flashnext.prefix_cache import PrefixCache\n"
                    "cache=PrefixCache('owner', 0, sys.argv[1]); print('ready', flush=True); time.sleep(60)\n")
            child = subprocess.Popen([sys.executable, "-c", code, directory], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaises(P.CacheOwnershipError):
                    P.PrefixCache("other", 0, directory)
            finally:
                child.terminate()
                child.communicate(timeout=10)
            cache = P.PrefixCache("other", 0, directory)
            self.assertIsNotNone(cache)


if __name__ == "__main__":
    unittest.main()
