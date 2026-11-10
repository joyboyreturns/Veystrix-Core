"""AGENTIC prefill readiness: the measured traces, the category accounting and the purge interplay. No model is loaded."""
import io
import json
import os
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from models.flashnext import prefill_readiness as R
from models.flashnext import purge_recovery as P
from models.flashnext.tests.bench import prefill_memory_harness as H
from models.flashnext.tests.unit.test_purge_recovery import Machine, make

GIB = 2 ** 30
MIB = 2 ** 20
ROOT = Path(__file__).resolve().parents[4]
FAILED_AGENTIC = ROOT / "results/flashnext/20261108-first-agentic-smoke-guard-stop/veystrix-chat-7444.jsonl"
CHAT_SUCCESS = ROOT / "results/flashnext/20261108-chat-pool0-success/veystrix-chat-87188.jsonl"
CHAT_POOL2_STOP = ROOT / "results/flashnext/20261108-first-chat-guard-stop/veystrix-chat-85653.jsonl"
# The traces record available, anonymous, file-backed and compressor memory but not wired memory. Live readings on this
# machine were 1.88 to 2.0 GiB, so the tests assume 1.9 GiB and say so.
ASSUMED_WIRED = int(1.9 * GIB)
TOTAL = 16 * GIB


class ConfiguredAgenticTierTests(unittest.TestCase):
    def test_configured_context_and_cold_limit_are_assessed(self):
        env = {"VEYSTRIX_MODE": "agentic", "VEYSTRIX_CONTEXT_LIMIT": "262144",
               "VEYSTRIX_MAX_COLD_PREFILL": "32768", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048"}
        tier, = R.active_tiers(env)
        self.assertEqual((tier.context_limit, tier.max_cold_prefill), (262144, 32768))
        self.assertEqual(tier.peak_bytes, R.cold_prefill_peak_bytes(32768, budgeted=True))

    def test_unbounded_chunks_do_not_get_configured_tier(self):
        env = {"VEYSTRIX_MODE": "agentic", "VEYSTRIX_CONTEXT_LIMIT": "262144",
               "VEYSTRIX_MAX_COLD_PREFILL": "32768", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "0"}
        self.assertNotEqual(R.active_tiers(env)[0].name, "configured")

LIVE_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   254557.
Pages active:                                 248297.
Pages inactive:                               269727.
Pages speculative:                             48240.
Pages throttled:                                   0.
Pages wired down:                             121619.
Pages purgeable:                                7283.
File-backed pages:                            427513.
Anonymous pages:                              138751.
Pages stored in compressor:                   203786.
Pages occupied by compressor:                  67431.
Swapins:                                     3044371.
Swapouts:                                    3908991.
"""


def start_row(path):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "mlx_active" in row:
            return row
    raise AssertionError("no sample rows")


def reading_from_trace(path, pressure=1) -> P.Reading:
    """The guard's first sample, as a byte-granular reading (free carries the whole gate figure)."""
    row = start_row(path)
    return P.Reading(page=1, free=int(row["available_mb"] * MIB), anonymous=int(row["anonymous_mb"] * MIB),
                     file_backed=int(row["file_backed_mb"] * MIB), compressor=int(row["compressor_mb"] * MIB),
                     wired=ASSUMED_WIRED, swap_used=int(row["swap_used_mb"] * MIB), pressure=pressure)


def traces_present():
    return all(path.exists() for path in (FAILED_AGENTIC, CHAT_SUCCESS, CHAT_POOL2_STOP))


@unittest.skipUnless(traces_present(), "preserved result traces are not present")
class MeasuredTraceTests(unittest.TestCase):
    def test_the_old_gate_admits_all_three_measured_starts(self):
        for path in (FAILED_AGENTIC, CHAT_SUCCESS, CHAT_POOL2_STOP):
            with self.subTest(trace=path.parent.name):
                self.assertTrue(R.check_gate(reading_from_trace(path)).ok)

    def test_the_failed_agentic_start_is_admitted_by_the_measured_gate_and_the_retired_budget_is_only_advisory(self):
        """The 898-token run began with 9.59 GiB available and 6.31 GiB of other anonymous memory and was stopped in its cold
        prefill by the compressor rule of that day (retired since). Replayed through the staged controller it is never stopped
        and its lowest gate figure (3.26 GiB) stays above the 2 GiB floor, so the budget no longer refuses it; the gate adds
        the anonymous caution (+0.5 GiB) because that is the only memory-driven failure on record at this anonymous level."""
        reading = reading_from_trace(FAILED_AGENTIC)
        verdict = R.evaluate(reading, TOTAL, tokens=898)
        self.assertTrue(verdict.admitted, R.render(verdict))
        gate = verdict.checks[0]
        self.assertAlmostEqual(gate.values["required"] / GIB, 9.51, places=1)      # 9.01 + 0.5 caution
        advisory = verdict.checks[2]
        self.assertTrue(advisory.ok)
        self.assertTrue(advisory.values["would_refuse_under_legacy_rule"])
        self.assertAlmostEqual(advisory.values["other_anonymous"] / GIB, 6.31, places=1)

    def test_the_first_chat_stop_is_still_refused_by_the_gate_because_of_the_caution(self):
        verdict = R.evaluate(reading_from_trace(CHAT_POOL2_STOP), TOTAL)           # 9.15 GiB available, 6.36 GiB anonymous
        self.assertFalse(verdict.admitted)
        self.assertEqual([check.name for check in verdict.blocking], ["start gate (mandatory, measured)"])

    def test_the_successful_chat_start_is_admitted_the_former_false_positive_is_gone(self):
        # CHAT finished five answers from 5.58 GiB of anonymous memory and 9.70 GiB available. The retired budget refused this
        # state for AGENTIC; the measured gate admits it (its lowest gate figure was 3.24 GiB, above the 2 GiB floor).
        verdict = R.evaluate(reading_from_trace(CHAT_SUCCESS), TOTAL)
        self.assertTrue(verdict.admitted, R.render(verdict))
        self.assertTrue(verdict.checks[2].values["would_refuse_under_legacy_rule"])

    def test_headroom_ranks_the_runs_by_how_far_they_were_from_safe(self):
        headroom = {path.parent.name: R.check_committed(reading_from_trace(path), TOTAL).values["headroom"]
                    for path in (FAILED_AGENTIC, CHAT_SUCCESS, CHAT_POOL2_STOP)}
        self.assertLess(headroom["20261108-first-agentic-smoke-guard-stop"], headroom["20261108-chat-pool0-success"])

    def test_the_traced_runs_did_not_differ_in_the_figure_the_old_gate_uses(self):
        gates = [reading_from_trace(path).available / GIB for path in (FAILED_AGENTIC, CHAT_SUCCESS, CHAT_POOL2_STOP)]
        self.assertTrue(all(9.1 < gate < 9.8 for gate in gates), gates)

    def test_the_compression_ratio_matches_the_failed_trace(self):
        rows = [json.loads(line) for line in FAILED_AGENTIC.read_text().splitlines() if '"mlx_active"' in line]
        before = [row for row in rows if row["phase"] == "init"][-1]        # loaded, idle, just before the prefill
        after = next(row for row in rows if row["t"] >= before["t"] + 2.5)  # the first seconds of the prefill
        anonymous_drop = before["anonymous_mb"] - after["anonymous_mb"]
        compressor_rise = after["compressor_mb"] - before["compressor_mb"]
        self.assertGreater(anonymous_drop, 4000)                             # 4.2 GiB moved out of anonymous memory
        self.assertAlmostEqual(compressor_rise / anonymous_drop, 0.69, places=1)
        self.assertLessEqual(compressor_rise / anonymous_drop, R.COMPRESSION_RATIO)


class LiveSampleTests(unittest.TestCase):
    def reading(self, **changes):
        base = P.parse_vm_stat(LIVE_VM_STAT, pressure=1, swap_used=int(773.5 * MIB))
        return P.Reading(**{**base.__dict__, **changes})

    def test_a_quiet_machine_passes_the_budget_and_only_the_gate_needs_the_purge(self):
        verdict = R.evaluate(self.reading(), TOTAL)
        names = {check.name: check.ok for check in verdict.checks}
        self.assertTrue(names["legacy anonymous-memory budget (advisory)"])
        self.assertTrue(names["pressure and swap-out"])
        self.assertFalse(names["start gate (mandatory, measured)"])           # 8.79 GiB
        self.assertIn("AUTO_PURGE", " ".join(verdict.advice))

    def test_after_the_file_cache_is_purged_the_same_machine_is_admitted(self):
        base = self.reading()
        purged = self.reading(free=base.free + base.file_backed - base.speculative, file_backed=base.speculative)
        verdict = R.evaluate(purged, TOTAL)
        self.assertTrue(verdict.admitted, R.render(verdict))

    def test_file_cache_never_moves_the_budget(self):
        base = self.reading()
        purged = self.reading(free=base.free + base.file_backed, file_backed=0)
        self.assertEqual(R.check_committed(base, TOTAL).values["headroom"],
                         R.check_committed(purged, TOTAL).values["headroom"])

    def test_budget_arithmetic(self):
        reading = self.reading()
        values = R.check_committed(reading, TOTAL).values
        other = reading.anonymous * reading.page
        expected = TOTAL - (reading.wired * reading.page + reading.compressor * reading.page
                            + R.COLD_PREFILL_PEAK_BYTES + R.HARNESS_ALLOWANCE_BYTES + int(0.75 * other))
        self.assertEqual(values["headroom"], expected)

    def test_each_extra_anonymous_gib_costs_three_quarters_of_a_gib_of_headroom(self):
        low = R.check_committed(self.reading(), TOTAL).values["headroom"]
        more = R.check_committed(self.reading(anonymous=self.reading().anonymous + GIB // 16384), TOTAL).values["headroom"]
        self.assertAlmostEqual((low - more) / GIB, 0.75, places=2)

    def test_category_bounds_follow_from_the_vm_stat_identities(self):
        reading = self.reading()
        cats = R.categories(reading, TOTAL)
        self.assertEqual(cats["free"], reading.free * 16384)
        self.assertLessEqual(cats["inactive_anonymous_min"], cats["inactive_anonymous_max"])
        self.assertLessEqual(cats["inactive_anonymous_max"], cats["anonymous"])
        self.assertLessEqual(cats["inactive_file_min"], cats["inactive_file_max"])
        self.assertEqual(cats["inactive_file_max"] + cats["inactive_anonymous_min"], cats["inactive"])
        self.assertGreaterEqual(cats["reusable_floor"], cats["free"] + cats["speculative_file"])
        self.assertEqual(cats["compressor"], 67431 * 16384)
        self.assertEqual(cats["wired"], 121619 * 16384)

    def test_swap_out_in_progress_or_pressure_refuses_without_suggesting_a_purge(self):
        for changes, delta in (({"pressure": 2}, 0), ({}, 12)):
            with self.subTest(changes=changes, delta=delta):
                verdict = R.evaluate(self.reading(**changes), TOTAL, swapout_delta=delta)
                self.assertFalse(verdict.admitted)
                text = " ".join(verdict.advice)
                self.assertIn("do not purge", text)

    def test_a_crowded_machine_is_refused_by_the_gate_names_the_big_processes_and_does_not_hide_the_purge_option(self):
        crowded = self.reading(anonymous=int(6.3 * GIB) // 16384)
        verdict = R.evaluate(crowded, TOTAL, processes=[{"pid": 1, "command": "Safari", "resident": GIB, "compressed": GIB // 2}])
        text = R.render(verdict)
        self.assertIn("REFUSED", text)
        self.assertIn("Safari", text)
        self.assertEqual([check.name for check in verdict.blocking], ["start gate (mandatory, measured)"])
        self.assertIn("The gate is short by", " ".join(verdict.advice))

    def test_the_mandatory_gate_is_unchanged(self):
        self.assertEqual(R.MANDATORY_START_GATE_BYTES, 9 * 1024 * 1024 * 1024)
        self.assertEqual(P.Config().start_gate_bytes, R.MANDATORY_START_GATE_BYTES)
        below = self.reading(free=0, inactive=(9 * GIB - GIB) // 16384, speculative=0)
        self.assertFalse(R.check_gate(below).ok)

    def test_the_runtime_guard_thresholds_are_untouched(self):
        policy = H.SafetyPolicy()
        self.assertEqual(H.MIN_AVAILABLE_GB, 2.0)
        self.assertEqual(H.MAX_COMPRESSOR_GROWTH_GIB, 2.0)
        self.assertEqual(H.COMPRESSOR_STOP_AVAILABLE_GIB, 3.0)
        self.assertEqual(policy.pressure_stop_level, 2)
        self.assertEqual(policy.min_battery_percent, 25)
        self.assertEqual(policy.swap_growth_mb, 64.0)

    def test_live_reads_twice_and_reports_a_swapout_between_the_readings(self):
        base = self.reading()
        readings = iter([base, P.Reading(**{**base.__dict__, "swapouts": base.swapouts + 5})])
        verdict = R.live(read=lambda: next(readings), total=TOTAL, processes=lambda: [], pause=0, sleep=lambda _: None)
        self.assertFalse(verdict.admitted)
        self.assertEqual([c for c in verdict.checks if c.name == "pressure and swap-out"][0].values["swapout_delta"], 5)


class TopParsingTests(unittest.TestCase):
    def test_rows_with_spaces_units_and_suffixes(self):
        text = """PID    COMMAND          MEM   CMPRS
601    WindowServer     592M  231M
1299   Activity Monitor 97M   34M
78251  VTDecoderXPCServ 85M+  5008K
1315   Siri AI          1.5G  0B
garbage line
"""
        rows = R.parse_top(text)
        self.assertEqual([row["command"] for row in rows], ["WindowServer", "Activity Monitor", "VTDecoderXPCServ", "Siri AI"])
        self.assertEqual(rows[0]["resident"], 592 * MIB)
        self.assertEqual(rows[2]["compressed"], 5008 * 1024)
        self.assertEqual(rows[3]["resident"], int(1.5 * GIB))
        self.assertEqual(rows[3]["compressed"], 0)


class PurgeInterplayTests(unittest.TestCase):
    def machine(self):
        # 8 GiB available, 6 GiB of purgeable-looking file cache: the existing rule would purge.
        return Machine(available=8 * GIB, cache=6 * GIB)

    def test_without_a_blocker_the_gate_shortfall_is_purged_as_before(self):
        m = self.machine()
        result = P.ensure_startup(make(m))
        self.assertTrue(result.ready)
        self.assertEqual(m.purges, 1)

    def test_a_refusal_that_file_cache_cannot_fix_runs_no_purge_and_does_not_wait(self):
        m = self.machine()
        reasons = []
        recovery = make(m, blocker=lambda reading: "cold-prefill memory budget: anonymous memory")
        result = P.ensure_startup(recovery, out=reasons.append)
        self.assertFalse(result.ready)
        self.assertEqual(m.purges, 0)
        self.assertNotIn(P.PURGE_COMMAND, [call[0] for call in m.calls])
        self.assertLess(result.waited_s, 1.0)                        # returned at once, not after the 90 s wait
        self.assertIn("a purge cannot admit this launch", result.reason)

    def test_repeated_attempts_never_purge_while_the_blocker_holds(self):
        m = self.machine()
        recovery = make(m, blocker=lambda reading: "anonymous")
        for _ in range(50):
            recovery.attempt(9 * GIB, "startup")
            m.t += 31
        self.assertEqual(m.purges, 0)
        self.assertEqual(sum(recovery.skips.values()), 50)

    def test_the_blocker_applies_to_startup_only(self):
        m = self.machine()
        recovery = make(m, blocker=lambda reading: "anonymous")
        outcome = recovery.attempt(9 * GIB, "idle")
        self.assertEqual(outcome.action, "purged")

    def test_the_real_blocker_reports_pressure_only_and_no_longer_vetoes_on_anonymous_memory(self):
        crowded = P.Reading(page=1, free=8 * GIB, anonymous=int(6.3 * GIB), wired=int(1.9 * GIB),
                            compressor=GIB, file_backed=6 * GIB)
        self.assertIsNone(R.purge_blocker(crowded, TOTAL))              # the file cache can close the gap, so a purge may run
        quiet = P.Reading(page=1, free=8 * GIB, anonymous=2 * GIB, wired=int(1.9 * GIB), compressor=GIB, file_backed=6 * GIB)
        self.assertIsNone(R.purge_blocker(quiet, TOTAL))
        pressured = P.Reading(page=1, free=8 * GIB, anonymous=2 * GIB, wired=int(1.9 * GIB), compressor=GIB, pressure=2)
        self.assertIn("pressure", R.purge_blocker(pressured, TOTAL))


class SessionAdmissionTests(unittest.TestCase):
    PREFS = {"model": "flashnext"}

    def refused(self):
        verdict = R.evaluate(P.Reading(page=1, free=7 * GIB, anonymous=7 * GIB, wired=2 * GIB, compressor=GIB), TOTAL)
        self.assertFalse(verdict.admitted)
        return verdict

    def test_chat_mode_never_reads_the_machine(self):
        from macqwen import session

        with mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "chat"}), \
                mock.patch.object(R, "live", side_effect=AssertionError("CHAT must not be gated")):
            self.assertIsNone(session._admit_agentic(self.PREFS, False, io.StringIO()))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VEYSTRIX_MODE", None)
            with mock.patch.object(R, "live", side_effect=AssertionError("no mode must not be gated")):
                self.assertIsNone(session._admit_agentic(self.PREFS, False, io.StringIO()))

    def test_benchmarks_and_other_backends_are_not_gated(self):
        from macqwen import session

        with mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "agentic"}), \
                mock.patch.object(R, "live", side_effect=AssertionError("not gated")):
            self.assertIsNone(session._admit_agentic(self.PREFS, True, io.StringIO()))
            self.assertIsNone(session._admit_agentic({"model": "bonsai2"}, False, io.StringIO()))

    def test_agentic_refusal_exits_before_loading_and_explains(self):
        from macqwen import session

        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "agentic", "FLASHNEXT_MEMORY_POLICY": "strict-staged", R.LOG_FLAG: os.devnull}), \
                mock.patch.object(R, "live", return_value=self.refused()), redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                session._admit_agentic(self.PREFS, False, io.StringIO())
        self.assertEqual(raised.exception.code, R.ADMISSION_EXIT_CODE)
        text = err.getvalue()
        self.assertIn("REFUSED", text)
        self.assertIn("Nothing was loaded", text)
        self.assertIn("No guard was weakened", text)

    def test_agentic_admission_returns_the_verdict_and_prints_the_headroom(self):
        from macqwen import session

        ok = R.evaluate(P.Reading(page=1, free=11 * GIB, anonymous=2 * GIB, wired=int(1.9 * GIB), compressor=GIB), TOTAL)
        self.assertTrue(ok.admitted)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"VEYSTRIX_MODE": "agentic", "FLASHNEXT_MEMORY_POLICY": "strict-staged", R.LOG_FLAG: os.devnull}), \
                mock.patch.object(R, "live", return_value=ok):
            self.assertIs(session._admit_agentic(self.PREFS, False, out), ok)
        self.assertIn("predicted lowest", out.getvalue())

    def test_the_admission_log_is_private(self):
        import tempfile

        ok = R.evaluate(P.Reading(page=1, free=11 * GIB, anonymous=2 * GIB, wired=int(1.9 * GIB), compressor=GIB), TOTAL)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sub", "admission.jsonl")
            R.log(ok, {R.LOG_FLAG: path})
            self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
            self.assertTrue(json.loads(Path(path).read_text())["admitted"])


if __name__ == "__main__":
    unittest.main()
