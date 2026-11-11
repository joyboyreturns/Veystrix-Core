import os
import unittest
from unittest.mock import patch

from models.flashnext.settings import modes as M


class FlagNameTests(unittest.TestCase):
    def test_every_flag_name_exists_in_the_production_modules(self):
        from models.flashnext import (auto_reclaim, expert_io_stats, linear_state, memory_manager, pool_lifecycle,
                                      prefill, prefill_memory, prefill_scheduler, prefix_profiles, prefix_session, prefill_ring, progress,
                                      quantized_kv, qsa_chunk, layer_major, store, expert_pool, ngram, memory_controller)
        known = {prefill_ring.FLAG_EXPERTS, prefill_ring.FLAG_SCRATCH, prefill.FLAG_CHUNK_TOKENS,
                 prefill.FLAG_CHUNK_POLICY, pool_lifecycle.FLAG, prefix_session.FLAG, prefix_session.FLAG_DIR,
                 prefix_session.FLAG_RAM_GB, prefix_session.FLAG_DISK_GB, prefix_session.FLAG_BOUNDARIES,
                 linear_state.FLAG, memory_manager.FLAG, prefix_profiles.FLAG, prefix_profiles.FLAG_MAX_GB,
                 prefix_profiles.FLAG_RESERVE_GB, auto_reclaim.FLAG_AUTO_RECLAIM, auto_reclaim.FLAG_AUTO_PURGE,
                 expert_io_stats.FLAG_POOL_REUSE, "FLASHNEXT_EXPERT_POOL_GB", "FLASHNEXT_EXPERT_POOL_AUTO", "FLASHNEXT_CHAT_GUARD",
                 "FLASHNEXT_NATIVE_MTP", progress.FLAG, prefill.FLAG_CHUNK_BUDGET_MB, prefill_scheduler.FLAG_ADAPTIVE,
                 quantized_kv.FLAG_KV_DTYPE, quantized_kv.FLAG_KV_GROUP_SIZE, quantized_kv.FLAG_KV_BITS,
                 quantized_kv.FLAG_KV_RECENT, quantized_kv.FLAG_KV_LAYERS,
                 qsa_chunk.FLAG_KERNEL, qsa_chunk.FLAG_GATHER_TILE,
                 layer_major.FLAG_ORDER, layer_major.FLAG_WINDOW, layer_major.FLAG_HYBRID,
                 store.FLAG_UNCACHED_EXPERT_READS, layer_major.FLAG_PREFETCH,
                 expert_pool.BUFFERED_FILL_FLAG, ngram.FLAG_DEDUP, memory_controller.FLAG_SWAP_CEILING,
                 memory_controller.FLAG_SWAP_CEILING_MARGIN, "FLASHNEXT_WATCHDOG_MEMORY", prefill_memory.FLAG_CACHE_MB}
        self.assertLessEqual(M.OWNED, known)

    def test_agentic_prefill_matches_the_real_model_f2br_condition(self):
        from models.flashnext.tests.bench import prefill_memory_harness as H
        overlay = H.condition_overlay(H.ALL_CONDITIONS["F2Br"])
        plan = M.plan_mode(M.AGENTIC, free_disk_gb=200)
        for key, value in overlay.items():
            if value is not None:
                self.assertEqual(plan.env.get(key), value, key)


class NewFlagOwnershipTests(unittest.TestCase):
    def test_default_modes_keep_bf16_kv_and_no_adaptive_chunks_and_drop_inherited_values(self):
        for mode in (M.CHAT, M.AGENTIC):
            plan = M.plan_mode(mode, free_disk_gb=200)
            self.assertEqual(plan.env["FLASHNEXT_KV_CACHE_DTYPE"], "bf16")
            self.assertEqual(plan.env["FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS"], "0")
            inherited = {"FLASHNEXT_KV_CACHE_DTYPE": "affine4", "FLASHNEXT_KV_GROUP_SIZE": "32",
                         "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB": "100", "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS": "1"}
            applied = M.apply_mode(dict(inherited), plan)
            self.assertEqual(applied["FLASHNEXT_KV_CACHE_DTYPE"], "bf16")
            for key in ("FLASHNEXT_KV_GROUP_SIZE", "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB"):
                self.assertNotIn(key, applied)

    def test_long_context_is_agentic_only_and_raises_the_default_limit(self):
        with self.assertRaises(M.ModeError):
            M.plan_mode(M.CHAT, enable=("long-context",))
        plan = M.plan_mode(M.AGENTIC, enable=("long-context",), free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_CHUNK_POLICY"], "budgeted")
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS"], "1")
        self.assertEqual(plan.context_limit, M.LONG_CONTEXT_LIMIT)
        self.assertIn("long-context", plan.labels)
        self.assertEqual(M.plan_mode(M.AGENTIC, free_disk_gb=200).context_limit, M.AGENTIC_CONTEXT_LIMIT)

    def test_one_quantized_kv_mode_at_a_time_and_labelled(self):
        with self.assertRaises(M.ModeError):
            M.plan_mode(M.AGENTIC, enable=("kv-affine4", "kv-affine8"), free_disk_gb=200)
        plan = M.plan_mode(M.AGENTIC, enable=("kv-affine8",), free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_KV_CACHE_DTYPE"], "affine8")
        self.assertIn("kv-affine8", plan.labels)


class PoolPolicyTests(unittest.TestCase):
    def test_default_is_auto_up_to_two_gb(self):
        gb, reason = M.choose_pool_gb(None, 10)
        self.assertEqual(gb, 2.0)
        self.assertIn("auto", reason)
        self.assertEqual(M.choose_pool_gb("auto", 10)[0], 2.0)

    def test_default_plan_sets_auto_and_a_fixed_size_does_not(self):
        self.assertEqual(M.plan_mode(M.CHAT).env["FLASHNEXT_EXPERT_POOL_AUTO"], "1")
        fixed = M.plan_mode(M.CHAT, pool_gb="2")
        self.assertEqual((fixed.env["FLASHNEXT_EXPERT_POOL_AUTO"], fixed.env["FLASHNEXT_EXPERT_POOL_GB"]), ("0", "2"))
        self.assertIn("stopped the compressor guard", fixed.pool_reason)
        off = M.plan_mode(M.CHAT, pool_gb="0")
        self.assertEqual((off.env["FLASHNEXT_EXPERT_POOL_AUTO"], off.env["FLASHNEXT_EXPERT_POOL_GB"]), ("0", "0"))
        with self.assertRaises(M.ModeError):
            M.plan_mode(M.CHAT, pool_gb="large")

    def test_the_banner_says_auto(self):
        self.assertIn("auto, up to 2 GB", M.banner(M.plan_mode(M.CHAT)))

    def test_under_the_start_gate_the_pool_is_off(self):
        self.assertEqual(M.choose_pool_gb(None, 7.5)[0], 0.0)

    def test_larger_than_the_cap_is_refused_including_the_old_six_gb(self):
        for value in (2.5, 4, 6):
            with self.assertRaises(M.ModeError):
                M.choose_pool_gb(value)

    def test_zero_and_one_gb_are_allowed_with_the_evidence_attached(self):
        self.assertEqual(M.choose_pool_gb(0)[0], 0.0)
        gb, reason = M.choose_pool_gb(1)
        self.assertEqual(gb, 1.0)
        self.assertIn("near zero", reason)

    def test_negative_is_refused(self):
        with self.assertRaises(M.ModeError):
            M.choose_pool_gb(-1)


class PlanTests(unittest.TestCase):
    def test_agentic_absolute_swap_guard_cannot_be_disabled(self):
        with patch.dict(os.environ, {"FLASHNEXT_SWAP_CEILING_MIB": "0",
                                          "FLASHNEXT_SWAP_CEILING_MARGIN_MIB": "0",
                                          "FLASHNEXT_WATCHDOG_MEMORY": "0"}):
            plan = M.plan_mode(M.AGENTIC, free_disk_gb=200)
        self.assertEqual(float(plan.env["FLASHNEXT_SWAP_CEILING_MIB"]), 3072)
        self.assertEqual(float(plan.env["FLASHNEXT_SWAP_CEILING_MARGIN_MIB"]), 256)
        self.assertEqual(plan.env["FLASHNEXT_WATCHDOG_MEMORY"], "1")

    def test_agentic_preserves_stricter_swap_guard(self):
        with patch.dict(os.environ, {"FLASHNEXT_SWAP_CEILING_MIB": "2800",
                                          "FLASHNEXT_SWAP_CEILING_MARGIN_MIB": "768"}):
            plan = M.plan_mode(M.AGENTIC, free_disk_gb=200)
        self.assertEqual(float(plan.env["FLASHNEXT_SWAP_CEILING_MIB"]), 2800)
        self.assertEqual(float(plan.env["FLASHNEXT_SWAP_CEILING_MARGIN_MIB"]), 768)

    def test_chat_is_the_exact_reference_path_with_live_state_only(self):
        plan = M.plan_mode(M.CHAT)
        self.assertEqual((plan.profile, plan.serve, plan.context_limit), ("plain", False, None))
        env = plan.env
        for key in ("FLASHNEXT_PREFIX_CACHE", "FLASHNEXT_PREFILL_CHUNK_TOKENS", "FLASHNEXT_PREFILL_RING_EXPERTS",
                    "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL", "FLASHNEXT_MEMORY_MANAGER",
                    "FLASHNEXT_PREFIX_PROFILES", "FLASHNEXT_AUTO_MEMORY_RECLAIM", "FLASHNEXT_SHORT_PREFILL_POOL_REUSE"):
            self.assertEqual(env[key], "0", key)
        self.assertEqual((env["FLASHNEXT_CHAT_GUARD"], env["FLASHNEXT_EVAL_LINEAR_STATE"],
                          env["FLASHNEXT_EXPERT_POOL_GB"]), ("1", "1", "2"))

    def test_agentic_serves_and_caches_with_no_experimental_feature_by_default(self):
        plan = M.plan_mode(M.AGENTIC, free_disk_gb=200)
        self.assertEqual((plan.profile, plan.serve, plan.context_limit), ("agent", True, M.AGENTIC_CONTEXT_LIMIT))
        self.assertEqual(plan.env["FLASHNEXT_PREFIX_CACHE"], "1")
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_CHUNK_POLICY"], "balanced")
        self.assertEqual((plan.env["FLASHNEXT_MEMORY_MANAGER"], plan.env["FLASHNEXT_PREFIX_PROFILES"]), ("0", "0"))
        self.assertEqual(plan.labels, {})

    def test_auto_purge_is_the_users_choice_and_passes_through_a_mode(self):
        env = {"FLASHNEXT_AUTO_PURGE": "1", "FLASHNEXT_PURGE_CHECK_SECONDS": "5"}
        M.apply_mode(env, M.plan_mode(M.CHAT))
        self.assertEqual((env["FLASHNEXT_AUTO_PURGE"], env["FLASHNEXT_PURGE_CHECK_SECONDS"]), ("1", "5"))
        self.assertIn("auto purge   ON", M.banner(M.plan_mode(M.CHAT), auto_purge=True))
        self.assertIn("auto purge   off", M.banner(M.plan_mode(M.CHAT), auto_purge=False))

    def test_free_disk_does_not_change_the_shared_quota(self):
        self.assertEqual(M.plan_mode(M.AGENTIC, free_disk_gb=500).prefix_disk_gb, M.PREFIX_DISK_GB)
        self.assertEqual(M.plan_mode(M.AGENTIC, free_disk_gb=60).prefix_disk_gb, M.PREFIX_DISK_GB)

    def test_the_default_ssd_quota_is_80_gb_and_the_owner_can_override_it(self):
        self.assertEqual(M.PREFIX_DISK_GB, 80.0)
        self.assertEqual(M.plan_mode(M.AGENTIC, free_disk_gb=500).prefix_disk_gb, 80.0)
        self.assertEqual(M.cache_quota_gb({"VEYSTRIX_CACHE_GB": "80"}), 80.0)
        for bad in ("", "abc", "0", "-5"):
            self.assertEqual(M.cache_quota_gb({"VEYSTRIX_CACHE_GB": bad}), 80.0, bad)
        with patch.dict(os.environ, {"VEYSTRIX_CACHE_GB": "30"}):
            self.assertEqual(M.plan_mode(M.AGENTIC, free_disk_gb=500).prefix_disk_gb, 30.0)
            self.assertEqual(M.plan_mode(M.AGENTIC, free_disk_gb=70).prefix_disk_gb, 30.0)

    def test_low_free_disk_leaves_reads_available(self):
        plan = M.plan_mode(M.AGENTIC, free_disk_gb=49)
        self.assertEqual(plan.env["FLASHNEXT_PREFIX_CACHE"], "1")
        self.assertEqual(plan.prefix_disk_gb, M.PREFIX_DISK_GB)
        self.assertTrue(any("SSD reserve" in w for w in plan.warnings))

    def test_experimental_features_are_explicit_and_labelled(self):
        plan = M.plan_mode(M.AGENTIC, enable=["memory-manager", "prefix-profiles"], free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_MEMORY_MANAGER"], "1")
        self.assertEqual(plan.env["FLASHNEXT_PREFILL_RELEASE_DECODE_POOL"], "0")   # the manager releases the pool
        self.assertEqual(set(plan.labels), {"memory-manager", "prefix-profiles"})
        self.assertIn("EXPERIMENTAL", M.banner(plan))

    def test_bad_requests_are_refused(self):
        for call in (lambda: M.plan_mode("fast"), lambda: M.plan_mode(M.CHAT, enable=["nope"]),
                     lambda: M.plan_mode(M.CHAT, enable=["prefix-profiles"]),
                     lambda: M.plan_mode(M.AGENTIC, context_limit=0, free_disk_gb=200)):
            with self.assertRaises(M.ModeError):
                call()

    def test_context_limit_above_the_validated_path_warns(self):
        plan = M.plan_mode(M.AGENTIC, free_disk_gb=200)
        self.assertTrue(any("real-checkpoint evidence" in w for w in plan.warnings))   # 32K cold prefill is unvalidated
        self.assertFalse(M.plan_mode(M.AGENTIC, context_limit=2048, free_disk_gb=200).warnings)


class ApplyTests(unittest.TestCase):
    def test_inherited_owned_flags_are_replaced_and_unrelated_ones_kept(self):
        env = {"FLASHNEXT_EXPERT_POOL_GB": "6", "FLASHNEXT_PREFIX_CACHE": "1", "FLASHNEXT_MEMORY_MANAGER": "1",
               "FLASHNEXT_SLAB_GLOBAL": "56", "FLASHNEXT_PREFIX_CACHE_DIR": "/elsewhere"}
        M.apply_mode(env, M.plan_mode(M.CHAT))
        self.assertEqual((env["FLASHNEXT_EXPERT_POOL_GB"], env["FLASHNEXT_PREFIX_CACHE"],
                          env["FLASHNEXT_MEMORY_MANAGER"]), ("2", "0", "0"))
        self.assertNotIn("FLASHNEXT_PREFIX_CACHE_DIR", env)
        self.assertEqual(env["FLASHNEXT_SLAB_GLOBAL"], "56")
        self.assertEqual(env["VEYSTRIX_MODE"], "chat")

    def test_environment_round_trip_reflects_what_actually_runs(self):
        env = {}
        M.apply_mode(env, M.plan_mode(M.AGENTIC, pool_gb=0, enable=["memory-manager"], free_disk_gb=200))
        plan = M.plan_from_environment(env)
        self.assertEqual((plan.mode, plan.pool_gb, plan.enabled, plan.context_limit),
                         ("agentic", 0.0, ("memory-manager",), M.AGENTIC_CONTEXT_LIMIT))
        text = M.banner(plan, checkpoint="/m", available_gb=10.2)
        for needle in ("AGENTIC", "/m", "off", "exact RAM+SSD", "balanced", "READY", "supervisor", "NOT verified"):
            self.assertIn(needle, text)

    def test_no_mode_means_no_plan_and_a_low_reading_says_below_gate(self):
        self.assertIsNone(M.plan_from_environment({}))
        self.assertIn("BELOW GATE", M.banner(M.plan_mode(M.CHAT), available_gb=7))


if __name__ == "__main__":
    unittest.main()
