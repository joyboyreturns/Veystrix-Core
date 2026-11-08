"""Endurance: 100 consecutive requests through the prefix session with the profile manager (small model, real caches).

A mixture of long uncached system prompts, short tool-result continuations, different conversations that share one
system prompt, RAM-to-SSD eviction, an SSD budget that is reached repeatedly, cancellations in the middle of a prefill,
engine restarts and changing memory availability. After every request:

* the result is bit-identical (token and every cache array) to a cold run with the same chunk plan and no cache;
* no lease is left behind, byte accounting matches, every file in the directory is accounted for;
* the SSD tier never exceeds its budget, no entry lacks its parent, no index refers to a stale entry.

Over the run: the frequent system prompt is recognised, pinned and always hits; memory does not grow.
"""
from __future__ import annotations

import gc
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

from models.flashnext import prefill as prefill_module
from models.flashnext import prefix_cache as P
from models.flashnext import prefix_profiles as PP
from models.flashnext import prefix_session as S
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits

MARK = [3, 4]
RAM_GB, DISK_GB = "0.0015", "0.006"          # about 3 resident and 12 stored snapshots of the small model


class Cancelled(BaseException):
    """What a client abort looks like to the prefill: not an Exception, so nothing swallows it."""


def compat():
    return {"model": "small", "tokenizer": "t", "engine": "e", "profile": {}}


def encode(text):
    assert text == S.IM_END
    return list(MARK)


class EnduranceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = {S.FLAG: "1", S.FLAG_DIR: str(self.tmp / "cache"), S.FLAG_RAM_GB: RAM_GB, S.FLAG_MIN_TOKENS: "64", S.FLAG_BOUNDARIES: "chunks",
               PP.FLAG: "1", PP.FLAG_MAX_GB: DISK_GB, PP.FLAG_RESERVE_GB: "0", PP.FLAG_PIN_AFTER: "3",
               "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ---- helpers ----
    def prompt_parts(self, rng):
        def body(n):
            return [rng.randrange(5, VOCAB) for _ in range(n)]
        systems = [body(98) + MARK, body(110) + MARK, body(130) + MARK, body(90) + MARK]
        return systems

    def engine(self):
        return build_model(), S.PrefixSession(encode, compat)

    def go(self, model, session, prompt):
        token, cache, report = session.prefill(model, prompt, model.make_cache())
        mx.eval(token, [c.state for c in cache])
        return int(token.item()), cache, report

    def reference(self, prompt):
        """The cold, uncached run with the same chunk plan: restore off keeps boundaries and anchors."""
        with mock.patch.dict(os.environ, {S.FLAG_RESTORE: "0", S.FLAG_DIR: str(self.tmp / "ref")}):
            session = S.PrefixSession(encode, compat)
            token, cache, report = session.prefill(self.ref_model, prompt, self.ref_model.make_cache())
            mx.eval(token, [c.state for c in cache])
            self.assertEqual(report.hit_tokens, 0)
            return int(token.item()), cache

    def invariants(self, session, step):
        store = session.store()
        where = f"step {step}"
        self.assertEqual(store._leases, {}, f"{where}: a lease leaked")
        self.assertEqual(store._ram_bytes, sum(e.nbytes() for e in store._ram.values()), f"{where}: RAM accounting")
        budget = store.disk_budget
        pinned_bytes = sum(store._disk[k]["bytes"] for k in store._protected_by_pins() if k in store._disk)
        self.assertLessEqual(store._disk_total(), max(budget, pinned_bytes), f"{where}: SSD budget exceeded")
        for key, meta in store._disk.items():
            self.assertTrue(meta["parent"] is None or meta["parent"] in store._disk, f"{where}: orphaned child")
            self.assertTrue((store.directory / f"{key}.json").exists(), f"{where}: manifest missing")
            self.assertTrue((store.directory / f"{key}.safetensors").exists(), f"{where}: data missing")
        names = {p.name for p in store.directory.iterdir()}
        accounted = {PP.MARKER, PP.INDEX, PP.INDEX + ".bak", PP.INDEX_LOCK, P.OWNER_LOCK}
        for key in store._disk:
            accounted |= {f"{key}.json", f"{key}.safetensors"}
        self.assertEqual(names - accounted, set(), f"{where}: unaccounted files {sorted(names - accounted)[:3]}")
        live = set(store._disk) | set(store._ram)
        self.assertLessEqual(set(store._meta), live, f"{where}: stale profile record")
        self.assertLessEqual(set(store._tokens), live | set(store._disk), f"{where}: stale token tape")
        self.assertLessEqual(len(store._seen), PP.SEEN_LIMIT)

    # ---- the run ----
    def test_one_hundred_requests_with_restarts_cancellations_and_pressure(self):
        rng = random.Random(2026)
        systems = self.prompt_parts(rng)
        weights = [0.55, 0.2, 0.15, 0.1]
        self.ref_model = build_model()
        model, session = self.engine()
        conversations: list = []
        totals = {"evicted_ssd": 0, "evicted_ram": 0}
        stats = {"hits": 0, "cancelled": 0, "restarts": 0, "pressure": 0, "frequent_requests": 0, "frequent_hits": 0}
        memory = {}
        for step in range(100):
            if step in (40, 99):                                           # MLX-held memory at two points of the run
                gc.collect()
                mx.clear_cache()
                memory[step] = mx.get_active_memory()
            roll = rng.random()
            if roll < 0.06 and step > 3:                                  # restart: new engine, state only on the SSD
                for name in totals:
                    totals[name] += getattr(session.store().stats, name)
                model, session = self.engine()
                stats["restarts"] += 1
            elif roll < 0.14:                                              # memory availability drops: RAM tier shrinks
                session.store().shrink_ram(rng.choice([0, 400_000, 900_000]))
                stats["pressure"] += 1
            if roll > 0.9 and conversations:                              # a tool result extends an earlier conversation
                base = rng.choice(conversations)
                prompt = base + [rng.randrange(5, VOCAB) for _ in range(rng.randrange(12, 40))]
                which = None
            else:
                which = rng.choices(range(4), weights)[0]
                prompt = systems[which] + [rng.randrange(5, VOCAB) for _ in range(rng.randrange(20, 70))]
            if rng.random() < 0.08:                                       # cancel in the middle of this prefill
                real = prefill_module.prefill_target

                def aborting(language, suffix, cache, **kw):
                    inner = kw.get("on_boundary")
                    if not kw.get("boundaries"):
                        raise Cancelled()                                 # nothing to snapshot: abort before computing

                    def boundary(position):
                        if inner is not None:
                            inner(position)
                        raise Cancelled()
                    kw["on_boundary"] = boundary
                    return real(language, suffix, cache, **kw)
                with mock.patch.object(prefill_module, "prefill_target", aborting):
                    try:
                        session.prefill(model, prompt, model.make_cache())
                    except Cancelled:
                        stats["cancelled"] += 1
                self.invariants(session, step)
                continue
            token, cache, report = self.go(model, session, prompt)
            ref_token, ref_cache = self.reference(prompt)
            self.assertEqual(token, ref_token, f"step {step}: token differs from the uncached same-plan run")
            self.assertTrue(cache_bits(cache) == cache_bits(ref_cache), f"step {step}: state differs")
            stats["hits"] += report.hit_tokens > 0
            if which == 0:
                stats["frequent_requests"] += 1
                stats["frequent_hits"] += report.hit_tokens == len(systems[0])
            conversations = (conversations + [prompt])[-6:]
            self.invariants(session, step)
        store = session.store()
        profile = next(r for r in store.profiles() if r["n_tokens"] == len(systems[0]))
        self.assertEqual(profile["pinned"], "auto", "the frequent system prompt must be recognised and pinned")
        self.assertGreaterEqual(profile["hits"], 10)
        self.assertGreater(stats["hits"], 40)
        self.assertGreater(stats["cancelled"], 0)
        self.assertGreater(stats["restarts"], 0)
        self.assertGreater(stats["pressure"], 0)
        # the pinned system prompt hit every time once it existed
        self.assertGreaterEqual(stats["frequent_hits"], stats["frequent_requests"] - 3)
        self.assertGreater(totals["evicted_ssd"] + store.stats.evicted_ssd, 0,
                           "the SSD budget was never reached: the test lost its point")
        self.assertGreater(totals["evicted_ram"] + store.stats.evicted_ram, 0)
        # no progressive growth in MLX-held memory between step 40 and step 99
        self.assertLess(memory[99] - memory[40], 4_000_000, f"memory grew: {memory}")

    def test_a_restart_in_the_middle_keeps_a_pinned_system_prompt_and_its_counters(self):
        rng = random.Random(1)
        system = self.prompt_parts(rng)[0]
        self.ref_model = build_model()
        model, session = self.engine()
        for _ in range(5):
            self.go(model, session, system + [rng.randrange(5, VOCAB) for _ in range(30)])
        session.store().flush()
        model, session = self.engine()
        token, cache, report = self.go(model, session, system + [9, 10, 11])
        self.assertEqual((report.hit_tokens, report.tier), (len(system), "ssd"))
        row = session.store().profiles()[0]
        self.assertEqual(row["pinned"], "auto")
        self.assertGreaterEqual(row["hits"], 4)
        ref_token, ref_cache = self.reference(system + [9, 10, 11])
        self.assertEqual(token, ref_token)
        self.assertTrue(cache_bits(cache) == cache_bits(ref_cache))

    def test_cancellation_during_a_restore_leaves_nothing_leased_and_the_entry_usable(self):
        rng = random.Random(3)
        system = self.prompt_parts(rng)[0]
        model, session = self.engine()
        self.go(model, session, system + [7] * 20)
        store = session.store()

        def broken_make_cache():
            raise Cancelled()
        with self.assertRaises(Cancelled):
            store.lookup(system + [7] * 20, broken_make_cache)
        self.assertEqual(store._leases, {})
        _, _, report = self.go(model, session, system + [8] * 20)
        self.assertEqual(report.hit_tokens, len(system))

    def test_cancellation_during_eviction_and_write_leaves_a_clean_cache(self):
        rng = random.Random(4)
        systems = self.prompt_parts(rng)
        model, session = self.engine()
        store = session.store()
        real = P.write_entry
        calls = {"n": 0}

        def interrupted(directory, entry, fp):
            calls["n"] += 1
            if calls["n"] == 2:
                real(directory, entry, fp)
                raise Cancelled()
            return real(directory, entry, fp)
        with mock.patch.object(P, "write_entry", interrupted):
            for system in systems:
                try:
                    self.go(model, session, system + [11] * 25)
                except Cancelled:
                    pass
        self.assertEqual(store._leases, {})
        model2, session2 = self.engine()                                    # restart: the interrupted write is recovered
        self.invariants(session2, "after restart")
        self.assertEqual(session2.store().scrub(delete=False), [])

    def test_an_incompatible_configuration_never_reuses_a_profile(self):
        rng = random.Random(5)
        system = self.prompt_parts(rng)[0]
        model, session = self.engine()
        self.go(model, session, system + [7] * 20)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CHUNK_TOKENS": "128"}):          # changes the chunk plan and key
            model2, session2 = self.engine()
            _, _, report = self.go(model2, session2, system + [7] * 20)
            self.assertEqual(report.hit_tokens, 0)
            self.assertGreaterEqual(session2.store().usage()["foreign_entries"], 1)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_RING_EXPERTS": "512"}):         # numerics setting: new fingerprint
            _, session3 = self.engine()
            self.assertNotEqual(session3.store().fp, session.store().fp)


if __name__ == "__main__":
    unittest.main()
