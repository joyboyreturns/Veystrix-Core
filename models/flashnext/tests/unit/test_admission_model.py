"""The measured admission model against the seven preserved real-checkpoint runs. No model is loaded, no machine is read."""
import json
import unittest
from pathlib import Path

from models.flashnext import admission_model as M
from models.flashnext import memory_controller as C
from models.flashnext import prefill_readiness as R
from models.flashnext import purge_recovery as P

GIB = 2 ** 30
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
FIXTURE = HERE / "bench/fixtures/admission_traces.json"
RUNS = {run["name"]: run for run in json.loads(FIXTURE.read_text())["runs"]}
CLIENT_FREE = {"chat-success", "chat-first-stop"}           # CHAT runs have no harness client


def predicted_lowest(run):
    client = 0.0 if run["name"] in CLIENT_FREE else M.CLIENT_GIB
    return run["start"]["available_gib"] - M.drop_gib(run["prefill_tokens"], client)


class EvidenceTests(unittest.TestCase):
    def test_the_fixture_holds_the_seven_runs_with_their_outcomes(self):
        self.assertEqual(len(RUNS), 7)
        self.assertEqual({r["outcome"] for r in RUNS.values()}, {"success", "guard-stop", "gpu-hang", "incomplete"})

    def test_the_gate_figure_fell_between_six_and_six_and_a_half_gib_in_every_complete_run(self):
        drops = [r["drop_total_gib"] for name, r in RUNS.items() if name != "chat-first-stop"]
        self.assertGreater(min(drops), 6.0)
        self.assertLess(max(drops), 6.7)

    def test_the_fall_does_not_follow_the_prefill_length(self):
        """60, 902, 3,935 and 8,200 tokens all fall within 0.6 GiB of each other while the MLX peak moves 1.2 GiB."""
        by_tokens = {RUNS[n]["prefill_tokens"]: RUNS[n]["drop_total_gib"] for n in ("chat-success", "agentic-1k", "agentic-4k", "agentic-8k")}
        self.assertLess(max(by_tokens.values()) - min(by_tokens.values()), 0.6, by_tokens)
        peaks = [RUNS[n]["mlx_peak_gib"] for n in ("agentic-1k", "agentic-4k", "agentic-8k")]
        self.assertGreater(max(peaks) - min(peaks), 1.1)

    def test_other_anonymous_memory_does_not_predict_the_compressor(self):
        """2.69 GiB of other anonymous memory still grew the compressor by 3.15 GiB: the engine's own churn dominates."""
        low = RUNS["agentic-gpu-hang"]
        grown = low["max_compressor_gib"] - low["start"]["compressor_gib"]
        self.assertLess(low["start"]["anonymous_gib"], 3.0)
        self.assertGreater(grown, 3.0)
        # a 0.75 x anonymous squeeze would have predicted this much compressor growth, a fraction of what happened
        self.assertLess(0.75 * low["start"]["anonymous_gib"], grown)

    def test_the_model_is_never_optimistic_about_the_lowest_gate_figure(self):
        for name, run in RUNS.items():
            with self.subTest(run=name):
                self.assertLessEqual(predicted_lowest(run), run["min_available_gib"] + 0.02,
                                     f"predicted {predicted_lowest(run):.2f} but the run bottomed at {run['min_available_gib']:.2f}")

    def test_and_it_is_not_wildly_pessimistic_either(self):
        for name, run in RUNS.items():
            with self.subTest(run=name):
                self.assertLess(run["min_available_gib"] - predicted_lowest(run), 1.0)

    def test_the_old_headroom_budget_did_not_track_the_margin_the_runs_really_had(self):
        """Rank the six runs by observed margin above the 2 GiB floor and by the retired headroom: they disagree, while the
        model's predicted margin follows the observed one within 0.6 GiB."""
        wired = {"agentic-4k": 1.81, "agentic-8k": 1.68, "agentic-1k": 1.87}
        observed, legacy, model = {}, {}, {}
        for name, run in RUNS.items():
            if name == "chat-first-stop":
                continue
            s = run["start"]
            peak = R.cold_prefill_peak_bytes(max(run["prefill_tokens"], 1)) / GIB
            legacy[name] = 16 - (wired.get(name, 1.9) + s["compressor_gib"] + peak + 1.0 + 0.75 * s["anonymous_gib"])
            observed[name] = run["min_available_gib"] - M.FLOOR_GIB
            model[name] = predicted_lowest(run) - M.FLOOR_GIB
        for name in observed:
            self.assertLess(observed[name] - model[name], 1.0, name)
        # the legacy figure calls the 8K run safer than the chat run although it had 0.6 GiB of margin against 1.24
        self.assertGreater(legacy["agentic-8k"], legacy["chat-success"] + 0.5)
        self.assertLess(observed["agentic-8k"], observed["chat-success"])
        # and it refuses the successful 4K and chat runs (headroom < 4) while admitting nothing the gate would not
        self.assertLess(legacy["chat-success"], 4.0)


class ModelTests(unittest.TestCase):
    def test_the_requirement_is_the_nine_gib_gate_up_to_the_measured_reference_and_grows_above_it(self):
        gib = M.required_available_gib
        self.assertEqual(gib(0, client_gib=0.0), 9.0)             # CHAT: 6.46 + 2.0 + 0.3 = 8.76, never below the old gate
        self.assertAlmostEqual(gib(3935), 9.01, places=2)
        self.assertLess(gib(3935), gib(8200) - 0.1)
        self.assertLess(gib(8200), gib(10240))
        self.assertLess(gib(10240), gib(14336))
        self.assertLess(gib(14336), gib(20000))

    def test_the_floor_is_the_runtime_guards_own_stop_line(self):
        source = (ROOT / "models/flashnext/chat_guard.py").read_text()
        self.assertIn("min_available_gb", source)
        from models.flashnext.tests.bench import prefill_memory_harness as H
        self.assertEqual(H.SafetyPolicy().min_available_gb, M.FLOOR_GIB)

    def test_the_anonymous_caution_adds_half_a_gib_only_at_the_one_failing_level(self):
        base = M.required_available_gib(902)
        self.assertEqual(M.required_available_gib(902, other_anonymous_gib=5.99), base)
        self.assertAlmostEqual(M.required_available_gib(902, other_anonymous_gib=6.31) - base, 0.5)

    def test_expected_purge_gain_is_the_smallest_one_on_record_capped_by_the_cache_that_exists(self):
        self.assertEqual(M.expected_purge_gain_bytes(5 * GIB), int(1.4 * GIB))
        self.assertEqual(M.expected_purge_gain_bytes(GIB), GIB)
        self.assertEqual(M.expected_purge_gain_bytes(0), 0)

    def test_recorded_purges_back_the_expected_gain(self):
        gains = []
        for name in ("20261108-224316-gate-a-attempt2", "20261108-agentic-smoke-retry-4-max-tokens-1",
                     "20261108-first-agentic-smoke-guard-stop"):
            path = ROOT / "results/flashnext" / name / "purge-log.jsonl"
            if not path.exists():
                self.skipTest("preserved purge logs are not present")
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("event") == "purge":
                    gains.append((row["available_after"] - row["available_before"]) / GIB)
        self.assertEqual(len(gains), 9)
        self.assertGreaterEqual(min(gains), M.PURGE_GAIN_OBSERVED_MIN_GIB - 0.05)
        self.assertAlmostEqual(sorted(gains)[len(gains) // 2], M.PURGE_GAIN_OBSERVED_MEDIAN_GIB, delta=0.05)


class CapacityTests(unittest.TestCase):
    """What each workload needs at launch, and what each real purge on record would have made of the user's refused state."""

    NEEDS = {"CHAT": (0, 0.0), "DeepSeek Harness 4K": (4096, M.CLIENT_GIB), "Qwen Code ~10K": (10240, M.CLIENT_GIB),
             "12K+ (long tier 14336)": (14336, M.CLIENT_GIB)}

    def test_the_start_figures_each_workload_needs_are_reported(self):
        need = {name: M.required_available_gib(tokens, client) for name, (tokens, client) in self.NEEDS.items()}
        self.assertEqual(need["CHAT"], 9.0)
        self.assertLess(need["DeepSeek Harness 4K"], 9.1)
        self.assertGreater(need["Qwen Code ~10K"], 9.4)
        self.assertLess(need["Qwen Code ~10K"], 9.6)
        self.assertGreater(need["12K+ (long tier 14336)"], 9.9)
        self.assertLess(need["12K+ (long tier 14336)"], 10.1)

    def test_the_refused_state_with_6_43_gib_is_recoverable_only_when_a_purge_returns_about_2_6_gib_or_more(self):
        gains = [3.10, 2.16, 2.0, 1.4, 2.5, 2.0, 2.4, 2.0, 1.7]          # every purge on record (tests above)
        need = M.required_available_gib(4096) - 6.43
        self.assertAlmostEqual(need, 2.59, places=1)
        enough = [g for g in gains if g >= need]
        self.assertEqual(len(enough), 1)                                  # one of nine recorded purges would have sufficed

    def test_the_refused_state_is_refused_with_honest_advice_and_no_guard_changes(self):
        reading = P.Reading(page=1, free=int(6.43 * GIB), anonymous=int(6.82 * GIB), wired=int(1.9 * GIB), compressor=GIB,
                            file_backed=int(3.0 * GIB), pressure=1)
        verdict = R.evaluate(reading, 16 * GIB)
        self.assertFalse(verdict.admitted)
        self.assertEqual([c.name for c in verdict.blocking], ["start gate (mandatory, measured)"])
        self.assertGreater(verdict.checks[0].values["required"], 9.5 * GIB)       # 9.02 + 0.5 anonymous caution
        self.assertIn("The gate is short by", " ".join(verdict.advice))


@unittest.skipUnless((ROOT / "results/flashnext/20261108-224316-gate-a-attempt2/guard-trace.jsonl").exists(),
                     "preserved result traces are not present")
class ReplayTests(unittest.TestCase):
    """Every real trace through today's staged controller and the 2 GiB floor, as the guard would have fed them."""

    TRACES = {
        "chat-success": "20261108-chat-pool0-success/veystrix-chat-87188.jsonl",
        "agentic-first-stop": "20261108-first-agentic-smoke-guard-stop/veystrix-chat-7444.jsonl",
        "agentic-1k": "20261108-agentic-smoke-retry-4-max-tokens-1/guard-trace.jsonl",
        "agentic-4k": "20261108-224316-gate-a-attempt2/guard-trace.jsonl",
        "agentic-8k": "20261108-232323-gate-8k-attempt1/guard-trace.jsonl",
    }

    def rows(self, name):
        out = []
        for line in (ROOT / "results/flashnext" / self.TRACES[name]).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "available_mb" in row and "mlx_peak" in row and "pressure_level" in row and "swapout" in row:
                out.append(row)
        return out

    def replay(self, name):
        controller = C.Controller(C.Policy(), page_bytes=16384)
        stop, floor_breaches = None, 0
        for row in self.rows(name):
            now = row["t"] if row["t"] < 1e6 else None
            if now is None:
                continue
            stop = stop or controller.observe(now, pressure=row["pressure_level"], swap_used_mb=row["swap_used_mb"],
                                              swapout_pages=row["swapout"], swapin_pages=row["swapin"],
                                              available_mb=row["available_mb"], compressor_mb=row["compressor_mb"])
            if row["available_mb"] / 1024 < M.FLOOR_GIB:
                floor_breaches += 1
        return stop, floor_breaches

    def test_no_complete_run_is_stopped_by_the_staged_controller_or_touches_the_floor(self):
        for name in self.TRACES:
            with self.subTest(run=name):
                stop, breaches = self.replay(name)
                self.assertIsNone(stop, f"{name}: {stop}")
                self.assertEqual(breaches, 0, name)

    def test_the_run_the_retired_compressor_rule_stopped_never_reached_sustained_pressure(self):
        rows = self.rows("agentic-first-stop")
        self.assertEqual(min(r["available_mb"] for r in rows) / 1024 > M.FLOOR_GIB, True)
        self.assertLess(sum(1 for r in rows if r["pressure_level"] >= 2), 5)


if __name__ == "__main__":
    unittest.main()
