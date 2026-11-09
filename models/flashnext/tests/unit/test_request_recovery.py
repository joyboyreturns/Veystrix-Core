"""Automatic recovery at every launch and at every request boundary. Fake machines only: nothing is purged or loaded."""
import inspect
import os
import unittest
from unittest import mock

from macqwen.backends import flashnext as backend_module
from models.flashnext import prefill_readiness as R
from models.flashnext import purge_recovery as P
from models.flashnext import resource_prep as RP
from models.flashnext.settings import modes as M
from models.flashnext.tests.unit.test_resource_prep import GIB, Machine, TOTAL

CTX = {"VEYSTRIX_CONTEXT_LIMIT": "8192", "VEYSTRIX_CONTEXT_REQUIRED": "1", "FLASHNEXT_PREFILL_CHUNK_POLICY": "budgeted"}


def machine_for(**kwargs):
    m = Machine(**kwargs)
    m.activity = P.Activity()
    m.recovery_for = lambda: P.PurgeRecovery(P.Config(settle_s=0.0), read=m.read, runner=m.runner, clock=m.clock,
                                             sleep=m.sleep, disk_read=lambda: 0, activity=m.activity)
    return m


class RequestBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(P.register, None)

    def test_short_ram_with_recoverable_cache_is_recovered_before_the_request(self):
        m = machine_for(free=3.0, cache=6.0, anonymous=3.0)
        recovery = m.recovery_for()
        P.register(recovery)
        outcome = P.prepare_for_request(prefill_tokens=3900)
        self.assertEqual(outcome.action, "purged")
        self.assertEqual(outcome.record.trigger, "pre-request")
        self.assertGreater(outcome.record.available_after, outcome.record.available_before + 5 * GIB)
        self.assertIsNone(m.activity.busy_reason(), "the purge slot must be released before the request starts")

    def test_enough_ram_means_no_purge(self):
        m = machine_for(free=9.0, cache=4.0)
        P.register(m.recovery_for())
        self.assertEqual(P.prepare_for_request(1000).action, "ok")
        self.assertEqual(m.purge_calls, 0)

    def test_the_requirement_grows_with_the_prefill_but_is_bounded(self):
        cfg = P.Config()
        small, big, huge = (P.request_required_bytes(cfg, n) for n in (0, 500, 14000))
        self.assertEqual(small, cfg.idle_headroom_bytes)
        self.assertGreater(big, small)
        self.assertEqual(P.request_required_bytes(cfg, 1024), huge)       # capped at one 1,024-token chunk

    def test_layer_major_adds_its_measured_working_set_only_for_a_large_prefill(self):
        import os
        from unittest import mock

        cfg = P.Config()
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_ORDER": "chunk"}):
            chunk_large, chunk_small = P.request_required_bytes(cfg, 8000), P.request_required_bytes(cfg, 500)
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_ORDER": "layer"}):
            self.assertEqual(P.request_required_bytes(cfg, 8000), chunk_large + P.LAYER_MAJOR_EXTRA_BYTES)
            self.assertEqual(P.request_required_bytes(cfg, 500), chunk_small)      # a short prefill never runs a window

    def test_anonymous_memory_shortage_never_causes_a_purge_loop(self):
        m = machine_for(free=2.0, cache=0.05, anonymous=9.0)             # nothing to recover: other apps hold the memory
        P.register(m.recovery_for())
        outcomes = [P.prepare_for_request(3000) for _ in range(5)]
        self.assertTrue(all(o.action == "skipped" for o in outcomes))
        self.assertEqual(m.purge_calls, 0)
        self.assertIn("cannot close the gap", outcomes[0].reason)

    def test_an_ineffective_purge_triggers_backoff_so_the_next_request_does_not_purge_again(self):
        m = machine_for(free=3.0, cache=6.0, effect=0.0)                 # the purge runs but frees nothing
        P.register(m.recovery_for())
        first = P.prepare_for_request(3000)
        self.assertEqual(first.action, "purged")
        self.assertFalse(first.record.effective)
        second = P.prepare_for_request(3000)
        self.assertEqual((second.action, m.purge_calls), ("skipped", 1))
        self.assertIn("backoff", second.reason)

    def test_an_effective_purge_is_not_repeated_within_the_minimum_interval(self):
        m = machine_for(free=3.0, cache=6.0)
        P.register(m.recovery_for())
        self.assertEqual(P.prepare_for_request(3000).action, "purged")
        m.free = 3 * GIB                                                  # short again immediately
        m.cache = 4 * GIB
        self.assertEqual(P.prepare_for_request(3000).action, "skipped")  # useful caches are not flushed repeatedly
        self.assertEqual(m.purge_calls, 1)
        m.t += 61                                                         # after the interval it may act again
        self.assertEqual(P.prepare_for_request(3000).action, "purged")

    def test_a_purge_never_overlaps_a_request_a_model_load_or_another_purge(self):
        for guard, expect in ((lambda a: a.request(), "request"), (lambda a: a.loading_model(), "loading")):
            m = machine_for(free=3.0, cache=6.0)
            P.register(m.recovery_for())
            with guard(m.activity):
                outcome = P.prepare_for_request(3000)
            self.assertEqual((outcome.action, m.purge_calls), ("skipped", 0), expect)
        m = machine_for(free=3.0, cache=6.0)
        P.register(m.recovery_for())
        self.assertIsNone(m.activity.try_begin_purge())                  # another purge holds the slot
        self.assertEqual(P.prepare_for_request(3000).action, "skipped")
        self.assertEqual(m.purge_calls, 0)

    def test_critical_pressure_and_swapout_skip_the_purge(self):
        m = machine_for(free=2.0, cache=6.0, pressure=4)
        P.register(m.recovery_for())
        self.assertIn("critical", P.prepare_for_request(3000).reason)
        self.assertEqual(m.purge_calls, 0)

    def test_nothing_registered_or_a_failing_recovery_never_stops_a_request(self):
        P.register(None)
        self.assertIsNone(P.prepare_for_request(100))
        boom = mock.Mock(closed=False, cfg=P.Config())
        boom.attempt.side_effect = RuntimeError("probe broke")
        P.register(boom)
        self.assertEqual(P.prepare_for_request(100).action, "skipped")

    def test_backend_recovers_before_it_enters_the_activity_record(self):
        source = inspect.getsource(backend_module.FlashNextBackend.generate if hasattr(backend_module, "FlashNextBackend")
                                   else next(c for c in vars(backend_module).values()
                                             if inspect.isclass(c) and hasattr(c, "generate")).generate)
        self.assertLess(source.index("prepare_for_request"), source.index("ACTIVITY.request()"))


class StatedContextTests(unittest.TestCase):
    def attempt_one_machine(self, **kw):
        # the shape of Gate A attempt 1: 6.1 GiB free + 3.1 GiB recoverable cache, 4.27 GiB other anonymous memory
        return machine_for(free=6.1, cache=3.1, anonymous=4.27, wired=1.59, compressor=0.79, **kw)

    def test_without_a_stated_requirement_one_purge_admits_the_attempt_one_machine_at_the_standard_tier(self):
        # Gate A attempt 1's state (6.1 GiB free, 3.1 GiB of recoverable file cache, 4.27 GiB other anonymous memory) was
        # refused by the retired anonymous budget, which also blocked the purge. The gate is the whole rule now: a purge that
        # returns the cache closes the 2.9 GiB gap, and it is re-measured before admission.
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_CHUNK_POLICY": "budgeted"}, clear=False):
            os.environ.pop("VEYSTRIX_CONTEXT_REQUIRED", None)
            m = self.attempt_one_machine()
            report = m.preparer(recovery=m.recovery_for()).prepare()
        self.assertTrue(report.admitted)
        self.assertEqual(m.purge_calls, 1)
        self.assertEqual(report.tier.name, "standard")      # every tier needs the same start figure below 4K tokens

    def test_a_stated_context_requirement_is_admitted_after_the_purge_and_refused_with_no_purge_when_the_cache_cannot_help(self):
        with mock.patch.dict(os.environ, CTX):
            m = self.attempt_one_machine()
            recovery = m.recovery_for()
            recovery._blocker = R.purge_blocker_tiered
            report = m.preparer(recovery=recovery).prepare()
            self.assertTrue(report.admitted, report.failure_report())
            self.assertEqual(m.purge_calls, 1)
            self.assertEqual(report.tier.name, "standard")
            poor = machine_for(free=6.1, cache=0.3, anonymous=4.27, wired=1.59, compressor=0.79)
            recovery = poor.recovery_for()
            recovery._blocker = R.purge_blocker_tiered
            refused = poor.preparer(recovery=recovery).prepare()
            text = refused.failure_report()
            required = refused.as_dict()["required_context"]
        self.assertFalse(refused.admitted)
        self.assertEqual(poor.purge_calls, 0, "0.3 GiB of file cache cannot close a 2.9 GiB gap; no purge")
        self.assertIn("requires a context of 8192", text)
        self.assertIn("short by", text)
        self.assertIn("Nothing was loaded", text)
        self.assertEqual(required, 8192)

    def test_if_closing_things_frees_enough_the_stated_requirement_is_admitted_at_standard_or_long(self):
        with mock.patch.dict(os.environ, CTX):
            m = machine_for(free=6.1, cache=3.1, anonymous=3.0, wired=1.59, compressor=0.79)
            recovery = m.recovery_for()
            recovery._blocker = R.purge_blocker_tiered
            report = m.preparer(recovery=recovery).prepare()
        self.assertTrue(report.admitted, report.failure_report() if not report.admitted else "")
        self.assertIn(report.tier.name, ("long", "standard"))

    def test_recoverable_file_cache_restores_the_start_gate_and_the_values_are_recorded(self):
        with mock.patch.dict(os.environ, CTX):
            m = machine_for(free=6.1, cache=3.1, anonymous=3.0, wired=1.59, compressor=0.79)
            recovery = m.recovery_for()
            recovery._blocker = R.purge_blocker_tiered
            report = m.preparer(recovery=recovery).prepare()
        self.assertEqual(m.purge_calls, 1)
        before, after = report.readings[0], report.readings[-1]
        self.assertEqual((before["stage"], after["stage"]), ("before", "after round 1"))
        self.assertGreater(after["available_gib"], before["available_gib"] + 2.5)
        self.assertEqual(before["anonymous_gib"], after["anonymous_gib"])        # a purge never touches anonymous memory
        self.assertFalse(before["admitted"])
        self.assertTrue(after["admitted"])
        self.assertIn("headroom_gib", after)

    def test_a_large_anonymous_footprint_with_enough_available_memory_is_admitted_without_any_purge(self):
        with mock.patch.dict(os.environ, CTX):
            m = machine_for(free=10.0, cache=3.0, anonymous=5.5, wired=1.6, compressor=0.8)   # gate fine; the retired budget refused
            recovery = m.recovery_for()
            recovery._blocker = R.purge_blocker_tiered
            report = m.preparer(recovery=recovery).prepare()
        self.assertTrue(report.admitted)
        self.assertEqual(m.purge_calls, 0)
        self.assertEqual(report.rounds, 0)

    def test_explicit_context_limit_becomes_a_requirement_a_default_does_not_and_inherited_values_are_dropped(self):
        explicit = M.plan_mode(M.AGENTIC, context_limit=8192, free_disk_gb=200)
        default = M.plan_mode(M.AGENTIC, free_disk_gb=200)
        self.assertTrue(explicit.context_required)
        # Owner decision 2026-11-09: the AGENTIC default is the native 262,144 window and is a stated requirement, so memory
        # tiers may not silently shrink it. A mode without a context limit (CHAT) is never a requirement, and an inherited
        # VEYSTRIX_CONTEXT_REQUIRED does not turn it into one.
        self.assertTrue(default.context_required)
        self.assertEqual(default.context_limit, 262144)
        applied = M.apply_mode({"VEYSTRIX_CONTEXT_REQUIRED": "1"}, M.plan_mode(M.CHAT, free_disk_gb=200))
        self.assertNotIn("VEYSTRIX_CONTEXT_REQUIRED", applied)
        applied = M.apply_mode({}, explicit)
        self.assertEqual((applied["VEYSTRIX_CONTEXT_LIMIT"], applied["VEYSTRIX_CONTEXT_REQUIRED"]), ("8192", "1"))
        self.assertTrue(M.plan_from_environment(applied).context_required)

    def test_tiers_follow_the_requirement_for_4k_8k_and_12k_and_chat_is_untouched(self):
        def names(limit, policy="budgeted"):
            env = {"VEYSTRIX_CONTEXT_LIMIT": str(limit), "VEYSTRIX_CONTEXT_REQUIRED": "1",
                   "FLASHNEXT_PREFILL_CHUNK_POLICY": policy}
            return [t.name for t in R.active_tiers(env)]
        self.assertEqual(names(4096), ["long", "8k", "standard"])
        self.assertEqual(names(8192), ["long", "8k", "standard"])
        self.assertEqual(names(12288), ["long", "8k"])
        self.assertEqual(names(2048), ["long", "8k", "standard", "compact"])
        self.assertEqual(names(40000), ["long"])                               # beyond every tier: report the largest gap
        self.assertEqual([t.name for t in R.active_tiers({})], ["standard", "compact", "minimal"])
        chat = RP.ResourcePreparer(agentic=False, read=Machine(free=10.0).read, total=TOTAL, processes=lambda: [], pause_s=0.0,
                                   sleep=lambda s: None).prepare()
        self.assertTrue(chat.admitted)                                          # CHAT still prepares only for the 9 GiB gate

    def test_thresholds_were_not_loosened(self):
        self.assertEqual((R.REQUIRED_HEADROOM_BYTES, R.MANDATORY_START_GATE_BYTES, R.COMPRESSION_RATIO),
                         (4 * GIB, 9 * GIB, 0.75))


if __name__ == "__main__":
    unittest.main()


class WaiverAdmissionTests(unittest.TestCase):
    """FLASHNEXT_MEMORY_WAIVER=swap-only lets an AGENTIC launch proceed past a refused admission, at the smallest tier that
    serves the stated context; without it the same verdict still exits 69 with nothing loaded."""

    ENV = {"VEYSTRIX_CONTEXT_REQUIRED": "1", "VEYSTRIX_CONTEXT_LIMIT": "12288",
           "FLASHNEXT_PREFILL_CHUNK_POLICY": "budgeted"}

    def refused(self):
        check = R.Check("headroom", False, "memory", "other anonymous 3.2 GiB leaves 3.85 GiB; 4 GiB needed", {})
        return R.Verdict(admitted=False, checks=[check], categories={})

    def run_admit(self, extra):
        import io
        from unittest import mock
        from macqwen import session as S
        env = {**self.ENV, **extra}
        with mock.patch.dict("os.environ", env), mock.patch.object(S, "_agentic_mode", return_value=True), \
                mock.patch.object(R, "log"), mock.patch.object(R, "render", return_value="refused"):
            out = io.StringIO()
            try:
                S._admit_agentic({"model": "flashnext"}, False, out, self.refused())
                code = None
            except SystemExit as stop:
                code = stop.code
            return code, dict(os.environ)

    def test_without_the_waiver_a_refusal_still_exits_69_with_nothing_loaded(self):
        code, _ = self.run_admit({})
        self.assertEqual(code, R.ADMISSION_EXIT_CODE)

    def test_with_the_waiver_the_launch_proceeds_at_the_largest_tier(self):
        """Owner decision 2026-11-09 (swap-growth-only policy): the largest tier is chosen, memory gates are telemetry-only.
        This test expected the 8K tier (10240) from before that decision; the approved behaviour is the standard tier."""
        code, env = self.run_admit({"FLASHNEXT_MEMORY_WAIVER": "swap-only"})
        self.assertIsNone(code)
        self.assertEqual((env["VEYSTRIX_CONTEXT_LIMIT"], env["VEYSTRIX_MAX_COLD_PREFILL"]), ("12288", "14336"))
