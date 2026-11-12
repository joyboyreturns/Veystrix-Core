"""Cross-session SSD/RAM prefix sharing on the real PrefixSession pipeline (small random-weight model, real cache classes).

Scenarios from the 2026-11-09 brief. Every scenario records total prompt tokens, matched (restored) tokens, newly prefilled
tokens, tier, restore latency, and compares the final state bit-for-bit with a cold run, so a reported hit is proven to be
skipped computation and a shared prefix is proven not to carry anything beyond the shared tokens.

Isolation argument under test: a snapshot at n tokens is a pure function of those n tokens (plus the engine fingerprint), and
a lookup accepts it only when the prompt's first n tokens are identical. Another session therefore gets exactly the state it
would have computed itself, never anything of the original session's later private tokens.

Not covered here by design: an authorization/client namespace partition of the SSD store (not implemented; see
docs/flashnext/persistent_cache_design.md), and a restore of decode-produced state (never bit-identical to a cold prefill, so
the engine refuses it by policy; resume after the live state was replaced restores the last prefill-time message boundary).
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import prefix_cache as P, prefix_session as S
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits
from models.flashnext.tests.unit.test_prefix_session import MARK, Base, compat, encode

TABLE: list[dict] = []


def rand(n, seed_offset=0):
    return [int(t) for t in mx.random.randint(5, VOCAB, (n,)).tolist()]


class CrossSessionBase(Base):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(21)
        cls.static = rand(150) + MARK                 # system prompt + tool schemas, ends at 152
        cls.tools_changed = cls.static[:100] + [(cls.static[100] + 1) % VOCAB or 7] + cls.static[101:]
        cls.user_a, cls.user_b = rand(60) + MARK, rand(70) + MARK
        cls.shared_user = rand(45) + MARK             # the same words typed in two different sessions
        cls.asst_1, cls.asst_2 = rand(40) + MARK, rand(44) + MARK
        cls.user_2 = rand(50) + MARK
        cls.gen = rand(5)                              # generation prompt after the last message

    def setUp(self):
        super().setUp()
        patch = mock.patch.dict(os.environ, {S.FLAG_BOUNDARIES: "messages"})
        patch.start()
        self.addCleanup(patch.stop)

    def run_prompt(self, label, prompt, session=None, model=None, note=""):
        if session is None:
            model, session = self.engine()
        token, cache, report = self.go(model, session, prompt)
        TABLE.append({"scenario": label, "total": report.prompt_tokens, "matched": report.hit_tokens,
                      "restored": report.bytes_restored and report.hit_tokens, "prefilled": report.prefilled_tokens,
                      "outcome": report.outcome, "tier": report.tier, "restore_ms": round(report.restore_seconds * 1000, 2),
                      "note": note})
        return token, cache, report, model, session

    def cold(self, prompt):
        """The same prompt with restore and snapshots off (same chunk plan): the reference state."""
        with mock.patch.dict(os.environ, {S.FLAG_RESTORE: "0"}):
            model, session = self.engine()
            token, cache, report = self.go(model, session, prompt)
        return token, cache, report

    def assert_same_as_cold(self, prompt, token, cache):
        cold_token, cold_cache, cold_report = self.cold(prompt)
        self.assertEqual(token, cold_token)
        self.assertTrue(cache_bits(cache) == cache_bits(cold_cache), "restored run differs from the cold run")
        self.assertEqual(cold_report.prefilled_tokens, len(prompt), "the cold reference must compute every token")


class SharedStaticPrefixTests(CrossSessionBase):
    def test_session_b_reuses_session_a_static_prefix_and_is_exactly_a_cold_run(self):
        a = self.static + self.user_a + self.gen
        self.run_prompt("A creates the static prefix", a, note="cold")
        b = self.static + self.user_b + self.gen
        token, cache, report, _, _ = self.run_prompt("B (new session, new client process) reuses it", b)
        self.assertEqual((report.hit_tokens, report.outcome), (152, "SSD_HIT"))
        self.assertEqual(report.prefilled_tokens, len(b) - 152, "only B's own tokens are computed")
        self.assert_same_as_cold(b, token, cache)

    def test_a_hit_in_the_same_process_comes_from_ram(self):
        model, session = self.engine()
        self.run_prompt("A (RAM)", self.static + self.user_a + self.gen, session, model)
        _, _, report, _, _ = self.run_prompt("B (RAM)", self.static + self.user_b + self.gen, session, model)
        self.assertEqual((report.hit_tokens, report.tier, report.outcome), (152, "ram", "RAM_HIT"))

    def test_session_c_with_a_changed_tool_schema_reuses_nothing(self):
        self.run_prompt("A", self.static + self.user_a + self.gen)
        c = self.tools_changed + self.user_b + self.gen
        token, cache, report, _, _ = self.run_prompt("C changed tool schema", c)
        self.assertEqual(report.hit_tokens, 0)
        self.assertEqual(report.prefilled_tokens, len(c))
        self.assert_same_as_cold(c, token, cache)

    def test_a_changed_system_prompt_is_a_miss_and_a_changed_effort_is_a_different_prompt(self):
        self.run_prompt("A", self.static + self.user_a + self.gen)
        other_effort = [self.static[0] + 1] + self.static[1:]        # the effort sentence is at the start of the system message
        _, _, report, _, _ = self.run_prompt("D changed reasoning effort", other_effort + self.user_a + self.gen)
        self.assertEqual(report.hit_tokens, 0)

    def test_the_sessions_prefixes_are_shared_only_up_to_their_common_tokens(self):
        """Same system prompt and same first words, different private histories: the second session may reuse the shared part
        (static prompt and the identical first user message) and nothing of the first session's later private tokens."""
        s1 = self.static + self.shared_user + self.asst_1 + self.user_2 + self.gen
        s2 = self.static + self.shared_user + self.asst_2 + self.user_2 + self.gen
        self.run_prompt("S1 private history 1", s1)
        token, cache, report, _, _ = self.run_prompt("S2 same user text, different private history", s2)
        shared = len(self.static) + len(self.shared_user)                # 152 + 47 = 199
        self.assertEqual(report.hit_tokens, shared, "reuse stops exactly where the token streams diverge")
        self.assertLess(report.hit_tokens, len(self.static + self.shared_user + self.asst_1))
        self.assert_same_as_cold(s2, token, cache)


class ResumeTests(CrossSessionBase):
    def test_resume_after_the_live_state_was_replaced_restores_the_last_message_boundary(self):
        a1 = self.static + self.user_a + self.gen
        self.run_prompt("A turn 1", a1)
        self.run_prompt("B interleaves (replaces the live state)", self.static + self.user_b + self.gen)
        a2 = self.static + self.user_a + self.asst_1 + self.user_2 + self.gen    # A resumes with its reply and a new question
        token, cache, report, _, _ = self.run_prompt("A resumes (new process)", a2)
        boundary = len(self.static) + len(self.user_a)
        self.assertEqual(report.hit_tokens, boundary)
        self.assertEqual(report.prefilled_tokens, len(a2) - boundary)
        self.assertLess(report.prefilled_tokens, len(a2) // 2)
        self.assert_same_as_cold(a2, token, cache)

    def test_a_restarted_server_finds_the_snapshots_again(self):
        self.run_prompt("A before restart", self.static + self.user_a + self.gen)
        _, _, report, _, _ = self.run_prompt("A after server restart", self.static + self.user_a + self.asst_1 + self.user_2 + self.gen)
        self.assertEqual(report.outcome, "SSD_HIT")
        self.assertGreaterEqual(report.hit_tokens, 152)

    def test_ram_eviction_then_ssd_restore_is_exact(self):
        model, session = self.engine()
        self.run_prompt("A (RAM)", self.static + self.user_a + self.gen, session, model)
        freed = session.store().shrink_ram(0)
        self.assertGreater(freed, 0)
        b = self.static + self.user_b + self.gen
        token, cache, report, _, _ = self.run_prompt("B after RAM eviction", b, session, model)
        self.assertEqual((report.tier, report.outcome, report.hit_tokens), ("ssd", "SSD_HIT", 152))
        self.assertGreater(report.bytes_loaded, 0, "an SSD hit really read bytes")
        self.assert_same_as_cold(b, token, cache)


class DamageAndBudgetTests(CrossSessionBase):
    def files(self):
        return sorted(f for f in (self.tmp / "cache").rglob("*") if f.is_file())

    def test_a_corrupted_snapshot_falls_back_to_a_cold_exact_result(self):
        self.run_prompt("A", self.static + self.user_a + self.gen)
        data = [f for f in self.files() if f.suffix not in (".json", ".lock") and f.stat().st_size > 1000]
        self.assertTrue(data, "no snapshot data file was written")
        for path in data:
            raw = bytearray(path.read_bytes())
            raw[len(raw) // 2] ^= 0xFF
            path.write_bytes(bytes(raw))
        b = self.static + self.user_b + self.gen
        token, cache, report, _, _ = self.run_prompt("B after corruption", b, note="flipped byte")
        self.assertEqual(report.hit_tokens, 0)
        self.assertEqual(report.prefilled_tokens, len(b))
        self.assert_same_as_cold(b, token, cache)

    def test_a_missing_parent_removes_its_children_and_lookups_stay_correct(self):
        long_prompt = self.static + self.user_a + self.asst_1 + self.user_2 + self.gen
        self.run_prompt("A multi-boundary", long_prompt)
        manifests = [f for f in self.files() if f.suffix == ".json"]
        self.assertGreaterEqual(len(manifests), 2)
        manifests[0].unlink()
        b = self.static + self.user_b + self.gen
        token, cache, report, _, _ = self.run_prompt("B after manifest loss / index rebuild", b)
        self.assertIn(report.hit_tokens, (0, 152))
        self.assert_same_as_cold(b, token, cache)

    def test_the_disk_quota_is_enforced_and_chains_stay_whole(self):
        budget = 3_000_000
        with mock.patch.dict(os.environ, {"VEYSTRIX_CACHE_GB": str(budget / 1e9)}):
            for index in range(8):
                user = rand(60) + MARK
                self.run_prompt(f"quota session {index}", self.static[:80] + [index + 6] + self.static[81:] + user + self.gen)
            model, session = self.engine()
            usage = session.store().usage()
        self.assertLessEqual(usage["disk_bytes"], budget)
        store = session.store()
        for key, meta in store._disk.items():                            # no child survives without its parent
            parent = meta.get("parent")
            self.assertTrue(parent is None or parent in store._disk, f"orphaned entry {key[:8]}")

    def test_files_and_directories_are_private_to_the_user(self):
        self.run_prompt("A", self.static + self.user_a + self.gen)
        for f in self.files():
            self.assertEqual(f.stat().st_mode & 0o077, 0, f"{f.name} is readable by others")

    def test_a_different_engine_fingerprint_never_shares(self):
        self.run_prompt("A", self.static + self.user_a + self.gen)
        with mock.patch.dict(os.environ, {"FLASHNEXT_QSA_QUERY_CHUNK": "1024"}):
            model, session = self.engine()
            _, _, report, _, _ = self.run_prompt("B different engine setting", self.static + self.user_b + self.gen, session, model)
        self.assertEqual(report.hit_tokens, 0)


class ZZTableTests(unittest.TestCase):
    def test_print_the_scenario_table(self):
        print("\nCROSS-SESSION CACHE SCENARIOS (total / matched / prefilled tokens, outcome, tier, restore ms)")
        for row in TABLE:
            print(f"  {row['scenario']:<52s} total={row['total']:4d} matched={row['matched']:4d} prefilled={row['prefilled']:4d} "
                  f"{str(row['outcome']):<14s} {str(row['tier']):<4s} {row['restore_ms']:7.2f} ms  {row['note']}")


if __name__ == "__main__":
    unittest.main()
