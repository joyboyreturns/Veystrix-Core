"""Staged memory controller (owner decision 2026-11-08): transient pressure is survivable, persistent or critical is not.

The scenarios are the numbers of Gate A attempt 2: pressure level 2 for ONE sample with 3.4 GiB available, zero swap-outs,
compressor +5.5 GiB. The old guard stopped there; the controller must not, and must still stop for the real failures."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from models.flashnext import memory_controller as M
from models.flashnext import prefill, prefill_scheduler as S

PAGE = 16384
MIB_PAGES = 2 ** 20 // PAGE


class Clock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


class Feed:
    """Drives a controller one second at a time."""

    def __init__(self, **policy):
        self.events = []
        self.c = M.Controller(M.Policy(**policy), page_bytes=PAGE, emit=lambda name, **f: self.events.append((name, f)))
        self.t = 0.0
        self.out = 0
        self.inn = 0

    def step(self, pressure=1, swap=700.0, out_mib=0, in_mib=0, available=4000.0, compressor=1000.0, dt=1.0):
        self.t += dt
        self.out += int(out_mib * MIB_PAGES)
        self.inn += int(in_mib * MIB_PAGES)
        return self.c.observe(self.t, pressure=pressure, swap_used_mb=swap, swapout_pages=self.out, swapin_pages=self.inn,
                              available_mb=available, compressor_mb=compressor)


class StageTests(unittest.TestCase):
    def test_absolute_swap_only_has_no_growth_stop_or_margin(self):
        telemetry = dict(pressure=4, swapout_pages=None, swapin_pages=None,
                         available_mb=100, compressor_mb=None)
        for override, ceiling in (({}, 3072), ({"FLASHNEXT_SWAP_CEILING_MIB": "1000"}, 1000)):
            with self.subTest(ceiling=ceiling):
                policy = M.Policy.from_environment({
                    "FLASHNEXT_MEMORY_POLICY": "absolute-swap-only",
                    "FLASHNEXT_SWAP_CEILING_MARGIN_MIB": "256",
                    **override,
                })
                controller = M.Controller(policy)
                self.assertIsNone(controller.observe(0, swap_used_mb=10, **telemetry))
                self.assertIsNone(controller.observe(60, swap_used_mb=ceiling - 0.1, **telemetry))
                self.assertTrue(controller.can_admit())
                self.assertIn(f"limit {ceiling} MiB", controller.observe(61, swap_used_mb=ceiling, **telemetry))
                self.assertFalse(controller.can_admit())

    def test_attempt_two_scenario_one_level_two_sample_is_survived(self):
        f = Feed()
        for _ in range(3):
            self.assertIsNone(f.step(available=4300, compressor=4400))
        self.assertIsNone(f.step(pressure=2, available=3460, compressor=6342))        # the sample that killed attempt 2
        self.assertEqual(f.c.state, M.ELEVATED)
        self.assertTrue(f.c.elevated())
        self.assertEqual(f.c.scale(), 0.5)
        for _ in range(3):
            f.step(available=3600, compressor=5000)
        self.assertEqual(f.c.state, M.RECOVERY)
        for _ in range(6):
            self.assertIsNone(f.step(available=4000))
        self.assertEqual(f.c.state, M.NORMAL)
        self.assertEqual(f.c.recoveries, 1)
        self.assertFalse(f.c.elevated())
        self.assertEqual([e[1]["state"] for e in f.events if e[0] == "memory_state"], [M.ELEVATED, M.RECOVERY, M.NORMAL])

    def test_sustained_level_two_is_unrecoverable_and_dips_do_not_reset_it(self):
        f = Feed(persist_s=30, recover_s=5)
        reason = None
        for i in range(60):
            level = 1 if i % 10 == 9 else 2          # a one-second dip every ten seconds is not a calm spell
            reason = f.step(pressure=level)
            if reason:
                break
        self.assertIsNotNone(reason)
        self.assertIn("without a calm spell", reason)
        self.assertEqual(f.c.state, M.CRITICAL)

    def test_a_real_calm_spell_resets_the_persistence_clock(self):
        f = Feed(persist_s=30, recover_s=5)
        for _ in range(25):
            self.assertIsNone(f.step(pressure=2))
        for _ in range(6):
            f.step(pressure=1)
        for _ in range(25):
            self.assertIsNone(f.step(pressure=2))
        self.assertNotEqual(f.c.state, M.CRITICAL)

    def test_pressure_level_four_stops_only_after_confirmation(self):
        f = Feed(critical_confirm_s=2)
        self.assertIsNone(f.step(pressure=4))
        self.assertIsNone(f.step(pressure=1))                # one critical sample that clears is not held
        self.assertIsNone(f.step(pressure=4))
        self.assertIsNone(f.step(pressure=4))
        self.assertIn("level 4", f.step(pressure=4))

    def test_swap_increase_stops_at_two_and_a_half_gib_over_the_baseline_not_the_absolute_total(self):
        f = Feed()                                           # the growth rule alone: the absolute guard is off by default
        self.assertIsNone(f.step(swap=3000.0))               # baseline 3000 MB already in swap from before the session
        self.assertIsNone(f.step(swap=3000.0 + 2559.0))
        self.assertEqual(f.c.state, M.ELEVATED)              # fresh swap growth is a signal ...
        reason = f.step(swap=3000.0 + 2560.0)                # ... and a hard stop at +2.5 GiB
        self.assertIn("swap grew 2.50 GiB", reason)
        self.assertEqual(f.c.ledger()["peak"]["swap_increase_mib"], 2560.0)

    def test_absolute_swap_guard_stops_below_the_owner_ceiling_whatever_the_baseline(self):
        f = Feed(swap_ceiling_mib=3072.0)                    # ceiling 3072 MiB, default margin 256 MiB -> stop at 2816
        self.assertIsNone(f.step(swap=1500.0))               # baseline 1.5 GiB: the 2.5 GiB growth rule alone would allow 4 GiB
        self.assertIsNone(f.step(swap=2815.0))
        reason = f.step(swap=2816.0)
        self.assertIn("absolute guard 2816 MiB", reason)
        self.assertIn("ceiling 3072 MiB", reason)
        self.assertTrue(f.c.stop_reason)

    def test_absolute_swap_guard_is_configurable_and_can_be_disabled(self):
        self.assertEqual(M.Policy.from_environment({}).swap_ceiling_mib, 0.0)            # off unless asked for
        self.assertEqual(M.Policy.from_environment({M.FLAG_SWAP_CEILING: "3072"}).swap_ceiling_mib, 3072.0)
        self.assertEqual(M.Policy.from_environment({M.FLAG_SWAP_CEILING_MARGIN: "512"}).swap_ceiling_margin_mib, 512.0)
        with self.assertRaises(ValueError):
            M.Policy.from_environment({M.FLAG_SWAP_CEILING: "-5"})
        f = Feed(swap_ceiling_mib=0, swap_growth_stop_gib=64)    # disabled: only the (here huge) growth rule is left
        f.step(swap=500.0)
        self.assertIsNone(f.step(swap=3500.0))
        ledger = Feed(swap_ceiling_mib=3072.0).c.ledger()
        self.assertEqual((ledger["swap_ceiling_mib"], ledger["swap_ceiling_margin_mib"]), (3072.0, 256.0))

    def test_swap_growth_limit_is_configurable_and_strict(self):
        f = Feed(swap_growth_stop_gib=1.0)
        f.step(swap=0.0)
        self.assertIn("swap grew 1.00 GiB", f.step(swap=1024.0))
        with self.assertRaises(ValueError):
            M.Policy.from_environment({M.FLAG_SWAP_GROWTH: "lots"})
        with self.assertRaises(ValueError):
            M.Policy.from_environment({M.FLAG_SWAP_GROWTH: "-1"})
        self.assertEqual(M.Policy.from_environment({M.FLAG_SWAP_GROWTH: "3"}).swap_growth_stop_gib, 3.0)
        self.assertEqual(M.Policy.from_environment({}).swap_growth_stop_gib, 2.5)

    def test_active_swap_activity_is_a_rate_not_the_total(self):
        f = Feed(swap_window_s=10, swap_activity_stop_mib=512)
        f.step()
        self.assertIsNone(f.step(out_mib=100))               # swapping began: elevated, not a stop
        self.assertEqual(f.c.state, M.ELEVATED)
        f2 = Feed(swap_window_s=10, swap_activity_stop_mib=512)
        f2.step()
        reasons = [f2.step(out_mib=60) for _ in range(12)]   # 60 MiB/s: 600 MiB inside ten seconds
        self.assertTrue(any(r and "active swap-out" in r for r in reasons))
        f3 = Feed(swap_window_s=10, swap_activity_stop_mib=512)
        f3.step()
        self.assertTrue(any(r and "active swap-in" in r for r in [f3.step(in_mib=600)]))

    def test_slow_swap_out_below_the_window_limit_never_stops(self):
        f = Feed(swap_window_s=10, swap_activity_stop_mib=512)
        f.step()
        for _ in range(120):
            self.assertIsNone(f.step(out_mib=5))             # 50 MiB per window for two minutes: noticed, not fatal
        self.assertEqual(f.c.state, M.ELEVATED)

    def test_compressor_surge_only_counts_while_memory_is_short(self):
        f = Feed()
        f.step(available=9000.0, compressor=1000.0)
        f.step(available=9000.0, compressor=4000.0)
        self.assertEqual(f.c.state, M.NORMAL)
        f.step(available=3000.0, compressor=7000.0)
        self.assertEqual(f.c.state, M.ELEVATED)
        self.assertIn("compressor", " ".join(f.c.reasons))

    def test_a_missing_probe_is_a_stop_not_a_guess(self):
        c = M.Controller()
        reason = c.observe(1.0, pressure=None, swap_used_mb=0, swapout_pages=0, swapin_pages=0, available_mb=1, compressor_mb=1)
        self.assertIn("probe unavailable: pressure", reason)
        self.assertEqual(c.state, M.CRITICAL)

    def test_ledger_reports_swap_at_launch_peak_increase_and_recoveries(self):
        f = Feed()
        f.step(swap=717.5)
        f.step(pressure=2, swap=900.0)
        for _ in range(20):                                  # the 10 s window must slide past the swap growth, then 5 s calm
            f.step(swap=900.0)
        ledger = f.c.ledger()
        self.assertEqual(ledger["swap_at_launch_mb"], 717.5)
        self.assertAlmostEqual(ledger["peak"]["swap_increase_mib"], 182.5)
        self.assertEqual(ledger["recovery_events"], 1)
        self.assertEqual(ledger["swap_growth_stop_gib"], 2.5)
        self.assertGreater(ledger["elevated_seconds"], 0)


class ReplayTests(unittest.TestCase):
    """The recorded samples of Gate A attempt 2 (about 1 s apart), fed to the controller exactly as the guard fed them."""

    def rows(self):
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "bench/fixtures/gate_a_attempt2_samples.json"
        return json.loads(path.read_text())["rows"]

    def test_the_run_that_the_old_guard_stopped_is_survived_and_ends_elevated_not_critical(self):
        rows = self.rows()
        c = M.Controller(M.Policy(), page_bytes=PAGE)
        t = 0.0
        for row in rows:
            t += row["dt"]
            stop = c.observe(t, pressure=row["pressure"], swap_used_mb=row["swap_used_mb"], swapout_pages=row["swapout"],
                             swapin_pages=row["swapin"], available_mb=row["available_mb"], compressor_mb=row["compressor_mb"])
            self.assertIsNone(stop, f"stopped at sample {t}: {stop}")
        self.assertEqual(rows[-1]["pressure"], 2)                    # the sample that killed attempt 2
        self.assertEqual(c.state, M.ELEVATED)
        ledger = c.ledger()
        self.assertEqual(ledger["swap_at_launch_mb"], 717.5)
        self.assertEqual(ledger["peak"]["swap_increase_mib"], 0.0)   # swap never moved in that run
        self.assertEqual(ledger["peak"]["swapout_window_mib"], 0.0)
        self.assertGreater(ledger["peak"]["compressor_growth_mib"], 5000)    # the +5.5 GiB the old guard only warned about
        self.assertEqual(ledger["peak"]["pressure_level"], 2)
        self.assertLess(ledger["peak"]["pressure_seconds"], 2)

    def test_the_same_run_with_pressure_that_never_clears_is_stopped(self):
        c = M.Controller(M.Policy(persist_s=45), page_bytes=PAGE)
        t, stop = 0.0, None
        for row in self.rows():
            t += row["dt"]
            stop = c.observe(t, pressure=2, swap_used_mb=row["swap_used_mb"], swapout_pages=row["swapout"],
                             swapin_pages=row["swapin"], available_mb=row["available_mb"], compressor_mb=row["compressor_mb"])
            if stop:
                break
        self.assertIsNotNone(stop)
        self.assertIn("without a calm spell", stop)
        self.assertGreaterEqual(t, 45)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_waits_only_while_elevated_and_is_bounded(self):
        f = Feed(throttle_max_wait_s=3, throttle_request_cap_s=5)
        f.step(pressure=2)
        clock = Clock()
        done = f.c.checkpoint("test", sleep=clock.sleep, clock=clock)
        self.assertGreaterEqual(done["waited_s"], 3.0)
        self.assertLess(done["waited_s"], 4.0)
        again = f.c.checkpoint("test", sleep=clock.sleep, clock=clock)
        self.assertLessEqual(again["waited_s"], 2.0 + 0.5)         # the per-request cap (5 s) is shared
        third = f.c.checkpoint("test", sleep=clock.sleep, clock=clock)
        self.assertEqual(third["waited_s"], 0.0)
        f.c.begin_request()
        self.assertGreater(f.c.checkpoint("test", sleep=clock.sleep, clock=clock)["waited_s"], 0)

    def test_checkpoint_returns_at_once_when_normal(self):
        f = Feed()
        f.step()
        clock = Clock()
        self.assertEqual(f.c.checkpoint("x", sleep=clock.sleep, clock=clock)["waited_s"], 0.0)
        self.assertEqual(clock.slept, [])

    def test_process_handle_defaults_to_normal_when_no_guard_runs(self):
        M.install(None)
        self.assertEqual(M.state(), M.NORMAL)
        self.assertFalse(M.elevated())
        self.assertFalse(M.forbid_optional())
        self.assertEqual(M.scale(), 1.0)
        self.assertEqual(M.checkpoint("x")["waited_s"], 0.0)
        self.assertIsNone(M.ledger())

    def test_installed_controller_drives_the_handle(self):
        f = Feed()
        f.step(pressure=2)
        M.install(f.c)
        self.addCleanup(M.install, None)
        self.assertTrue(M.elevated())
        self.assertTrue(M.forbid_optional())
        self.assertEqual(M.scale(), 0.5)


class ConsumerTests(unittest.TestCase):
    def test_scheduler_does_not_shrink_chunks_for_moderate_pressure_or_an_elevated_controller(self):
        """The 8K gate (2026-11-08): shrinking chunks multiplies SSD streaming, so only a critical reading may do it."""
        sched = S.AdaptivePrefillScheduler()
        sched.start_at(2048)
        gib = 2 ** 30
        for available, pressure, elevated in ((3.9 * gib, 1, False), (3.0 * gib, 2, False), (4.0 * gib, 1, True),
                                              (2.5 * gib, 2, True)):
            self.assertEqual(sched.observe_and_adapt(int(available), pressure, elevated=elevated), 2048)
        self.assertEqual(sched.observe_and_adapt(int(1.9 * gib), 1), 1024)            # under the 2 GiB guard floor
        self.assertEqual(sched.observe_and_adapt(int(5 * gib), 4), 512)                # macOS critical
        self.assertEqual(sched.observe_and_adapt(int(5 * gib), 1), 1024)               # one safe chunk steps back up

    def test_the_recorded_8k_trace_no_longer_collapses_the_chunk_plan(self):
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "bench/fixtures/gate_8k_attempt1_available.json"
        series = json.loads(path.read_text())["available_gib_at_each_edge"]
        sched = S.AdaptivePrefillScheduler(max_chunk=1024)        # the plan's own cap at this size
        sched.start_at(1024)
        caps = [sched.observe_and_adapt(int(a * 2 ** 30), 1) for a in series]
        self.assertEqual(set(caps), {1024})                                           # the old rule produced 256 and 512

    def test_uncached_reads_are_switched_on_only_while_elevated_and_only_once(self):
        store = mock.Mock(_uncached=False)
        trace = prefill.PlanTrace()
        with mock.patch("models.flashnext.pool_lifecycle.store_of", return_value=store), \
                mock.patch.dict(os.environ, {"FLASHNEXT_UNCACHE_WHEN_ELEVATED": "1"}):
            f = Feed()
            f.step()
            M.install(f.c)
            self.addCleanup(M.install, None)
            prefill._uncache_reads_when_elevated(object(), trace)
            store.set_uncached_reads.assert_not_called()
            f.step(pressure=2)
            prefill._uncache_reads_when_elevated(object(), trace)
            store.set_uncached_reads.assert_called_once_with(True)
            store._uncached = True
            prefill._uncache_reads_when_elevated(object(), trace)
            store.set_uncached_reads.assert_called_once()
        self.assertEqual(len(trace.throttles), 1)
        self.assertIn("throttles", trace.as_dict())

    def test_uncached_reads_are_off_unless_explicitly_enabled(self):
        store = mock.Mock(_uncached=False)
        f = Feed()
        f.step(pressure=2)
        M.install(f.c)
        self.addCleanup(M.install, None)
        with mock.patch("models.flashnext.pool_lifecycle.store_of", return_value=store), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_UNCACHE_WHEN_ELEVATED", None)
            prefill._uncache_reads_when_elevated(object(), prefill.PlanTrace())
        store.set_uncached_reads.assert_not_called()

    def test_a_trace_of_swap_in_is_not_swap_activity(self):
        f = Feed()
        f.step()
        f.step(in_mib=1.4)                                   # what the real 8K run recorded as an episode
        self.assertEqual(f.c.state, M.NORMAL)
        f.step(in_mib=9)
        self.assertEqual(f.c.state, M.ELEVATED)

    def test_the_optional_decode_pool_is_declined_while_elevated(self):
        from models.flashnext import expert_pool
        ample = 20 * 2 ** 30
        self.assertTrue(expert_pool.headroom_decision(ample, 2 ** 30)[0])
        f = Feed()
        f.step(pressure=2)
        M.install(f.c)
        self.addCleanup(M.install, None)
        allowed, reason = expert_pool.headroom_decision(ample, 2 ** 30)
        self.assertFalse(allowed)
        self.assertIn("ELEVATED", reason)

    def test_categories_keep_the_five_groups_apart(self):
        from models.flashnext import purge_recovery as P
        pages = lambda gib: int(gib * 2 ** 30 / PAGE)
        reading = P.Reading(page=PAGE, free=pages(0.1), inactive=pages(3.0), speculative=0, wired=pages(5.0),
                            compressor=pages(4.0), anonymous=pages(5.4), file_backed=pages(5.0), pressure=2,
                            swap_used=int(0.7 * 2 ** 30))
        c = M.categorize(reading, mlx_active=int(3.2 * 2 ** 30), mlx_cache=int(0.5 * 2 ** 30), rss=int(2.7 * 2 ** 30))
        gib = {k: round(v / 2 ** 30, 2) for k, v in c.items() if k != "pressure"}
        self.assertEqual(gib["active_computation"], 3.2)
        self.assertEqual(gib["reclaimable_own"], 0.5)
        self.assertEqual(gib["reclaimable_file_cache"], 3.0)      # file-backed pages that are inactive
        self.assertEqual(gib["other_applications"], 2.7)          # anonymous 5.4 minus this process's 2.7
        self.assertEqual(gib["unreleasable"], 5.0)
        self.assertEqual(gib["compressed"], 4.0)
        self.assertEqual(c["pressure"], 2)


class ChunkLogTests(unittest.TestCase):
    def test_every_finished_chunk_is_appended_live_and_logging_never_raises(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            with mock.patch.dict(os.environ, {prefill.CHUNK_LOG_FLAG: str(path)}):
                prefill._log_chunk(0, 1024, 3.25, {"available": 4 * 2 ** 30, "pressure": 1})
                prefill._log_chunk(1024, 2048, 4.0, None)
                prefill._log_chunk(2048, 3072, 1.0, {"bad": "data"})          # a malformed reading must not raise
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([(r["start"], r["end"], r["tokens"]) for r in rows[:2]], [(0, 1024, 1024), (1024, 2048, 1024)])
            self.assertEqual(rows[0]["available_gb"], 4.0)
            self.assertIsNone(rows[1]["available_gb"])
            self.assertIn(rows[0]["memory_state"], ("normal", "elevated", "recovery"))
        with mock.patch.dict(os.environ, {prefill.CHUNK_LOG_FLAG: "/nonexistent-dir/x.jsonl"}):
            prefill._log_chunk(0, 1, 1.0, None)                                # unwritable path: silent
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(prefill.CHUNK_LOG_FLAG, None)
            prefill._log_chunk(0, 1, 1.0, None)                                # not set: nothing happens


class BackendReleaseTests(unittest.TestCase):
    def test_release_between_turns_clears_cache_restores_reads_keeps_state_and_never_raises(self):
        from veystrix.backends import flashnext as B
        store = mock.Mock(_uncached=True)
        calls = []
        stub = SimpleNamespace(language=object(), _prefix=None, turn_memory_before={"rss": 1},
                               _turn_memory=lambda: {"mlx_cache": 0})
        with mock.patch("models.flashnext.pool_lifecycle.store_of", return_value=store), \
                mock.patch.object(B.gpu_sync, "synchronize", side_effect=lambda: calls.append("sync")), \
                mock.patch("models.flashnext.prefill_ring.fence", side_effect=lambda: calls.append("fence")), \
                mock.patch.object(B.mx, "get_cache_memory", return_value=B.RELEASE_CACHE_BYTES + 1), \
                mock.patch.object(B.mx, "clear_cache", side_effect=lambda: calls.append("clear")):
            B.FlashNextBackend._release_between_turns(stub)
        self.assertEqual(calls, ["sync", "fence", "clear"])        # the GPU barrier comes before anything is freed
        store.set_uncached_reads.assert_called_once_with(False)
        self.assertTrue(stub.last_turn_memory["released"]["cache_cleared"])
        self.assertEqual(stub.last_turn_memory["before"], {"rss": 1})
        broken = SimpleNamespace(language=object(), _prefix=None, _turn_memory=lambda: 1 / 0)
        with mock.patch.object(B.gpu_sync, "synchronize"):
            B.FlashNextBackend._release_between_turns(broken)       # must not raise
        self.assertIn("error", broken.last_turn_memory)

    def test_small_cache_is_left_alone(self):
        from veystrix.backends import flashnext as B
        stub = SimpleNamespace(language=object(), _prefix=None, turn_memory_before=None, _turn_memory=lambda: {})
        with mock.patch("models.flashnext.pool_lifecycle.store_of", return_value=None), \
                mock.patch.object(B.gpu_sync, "synchronize"), mock.patch("models.flashnext.prefill_ring.fence"), \
                mock.patch.object(B.mx, "get_cache_memory", return_value=1024), \
                mock.patch.object(B.mx, "clear_cache") as clear:
            B.FlashNextBackend._release_between_turns(stub)
        clear.assert_not_called()
        self.assertFalse(stub.last_turn_memory["released"]["cache_cleared"])

    def test_release_never_calls_a_gpu_synchronize_other_than_the_safe_one(self):
        import inspect
        from veystrix.backends import flashnext as B
        source = inspect.getsource(B.FlashNextBackend._release_between_turns)
        self.assertNotIn("mx.synchronize(", source)
        self.assertIn("gpu_sync.synchronize()", source)


if __name__ == "__main__":
    unittest.main()
