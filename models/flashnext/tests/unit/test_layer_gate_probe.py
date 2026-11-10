"""The live gate must notice changes and reject incomplete state."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx

from models.flashnext.tests.bench import layer_gate_probe as probe
from models.flashnext.tests.bench import bench_layer_gate as gate
from models.flashnext.tests.unit.test_prefix_cache import build_model


class StateProbeTests(unittest.TestCase):
    def test_exact_state_and_prompt_changes_are_visible(self):
        model = build_model()
        ids = [2, 3, 4, 5]
        caches = model.make_cache()
        output = model(mx.array([ids]), cache=caches)
        mx.eval(output.logits)
        first = probe.state_record(caches, ids, model)
        self.assertEqual(first, probe.state_record(caches, ids, model))
        self.assertTrue(any(layer["kind"] == "arrays" for layer in first["layers"]))
        self.assertTrue(any(layer["kind"] == "qsa_kv" for layer in first["layers"]))
        arrays = next(c for c in caches if type(c).__name__ == "ArraysCache")
        slot = next(i for i, a in enumerate(arrays.cache) if a is not None)
        arrays.cache[slot] = arrays.cache[slot] + 1
        self.assertNotEqual(first, probe.state_record(caches, ids, model))
        with self.assertRaises(ValueError):
            probe.state_record(caches, ids[:-1], model)

    def test_unknown_or_empty_cache_fails_closed(self):
        with self.assertRaises(ValueError):
            probe.state_record([], [1], object())
        with self.assertRaises(Exception):
            probe.state_record([object()], [1], object())

    def test_probe_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            path.write_text("preserve")
            with mock.patch("veystrix.results.ensure_inside", return_value=path):
                with self.assertRaises(FileExistsError):
                    probe.Probe(path)
            self.assertEqual(path.read_text(), "preserve")

    def test_prefill_probe_captures_actual_final_logits_without_changing_token(self):
        import json
        import os
        from veystrix.sampling import Sampler, Sampling
        from models.flashnext import prefill

        model = build_model()
        ids = mx.array([[2, 3, 4, 5]])
        sampler = Sampler(Sampling.greedy_settings())
        with mock.patch.object(prefill, "PREFILL_FULL_LOGITS_MAX_TOKENS", 2):
            _, _, reference = prefill.prefill_target(model, ids, model.make_cache(), sampler=sampler)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "probe.jsonl"
                path.touch()
                with mock.patch.dict(os.environ, {probe.FLAG: str(path)}), \
                        mock.patch("veystrix.results.ensure_inside", return_value=path):
                    _, _, observed = prefill.prefill_target(model, ids, model.make_cache(),
                                                           sampler=Sampler(Sampling.greedy_settings()))
                records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(int(reference.item()), int(observed.item()))
        self.assertEqual([r["event"] for r in records], ["logits"])
        self.assertTrue(records[0]["value"]["sha256"])


class GateValidationTests(unittest.TestCase):
    def evidence(self, layer=False):
        prefix = {"hit_tokens": 0, "prefilled_tokens": 3000, "executed_edges": [1000, 2000],
                  "execution_order": "layer-major-v1/w4096" if layer else "",
                  "execution_segments": [{"start": 0, "end": 3000, "execution": "layer"}] if layer else []}
        array = {"sha256": "a" * 64, "shape": [1], "dtype": "mlx.core.float32"}
        layers = [{"kind": "arrays", "states": [array]}] * 36 + [
            {"kind": "qsa_kv", "kv": {"keys": array, "values": array}}] * 12
        return {"logits": {"value": array},
                "prefilled": {"prefix": prefix, "state": {"token_count": 3000, "layers": layers,
                                                             "language": {"position_ids": array, "rope_deltas": array}},
                              "prompt_sha256": "x", "first_token": 7, "physical_bytes": 10, "peak_bytes": 20},
                "finished": {"token_ids": [7] * 32}}

    def test_matching_pair_passes_and_missing_or_different_evidence_fails(self):
        import copy
        reference, candidate = self.evidence(), self.evidence(True)
        self.assertIsNone(gate.validate_pair(reference, candidate))
        for mutate in (lambda x: x.pop("finished"),
                       lambda x: x["logits"]["value"].update(sha256="different"),
                       lambda x: x["prefilled"].update(first_token=99),
                       lambda x: x["prefilled"].update(physical_bytes=None),
                       lambda x: x["prefilled"]["state"].update(layers=[2]),
                       lambda x: x["finished"].update(token_ids=[7, 9]),
                       lambda x: x["prefilled"]["prefix"].update(execution_segments=[{"start": 0, "end": 3000, "execution": "chunk"}])):
            changed = copy.deepcopy(candidate)
            mutate(changed)
            self.assertIsNotNone(gate.validate_pair(reference, changed))

    def test_safety_preflight_fails_closed(self):
        good = {"power_source": "ac", "battery_percent": 80, "battery_state": "charging",
                "low_power": False, "thermal_readable": True, "thermal_warning": False,
                "thermal_speed_limit": None, "pressure_level": 1, "swap_used_mb": 100,
                "available_mb": 8000, "compressor_mb": 100}
        self.assertIsNone(gate.safety_reason(good))
        for field, value in (("power_source", "battery"), ("low_power", None),
                             ("thermal_readable", False), ("pressure_level", 2),
                             ("swap_used_mb", None), ("compressor_mb", None), ("available_mb", 2999)):
            sample = {**good, field: value}
            self.assertIsNotNone(gate.safety_reason(sample), field)
        self.assertIsNotNone(gate.safety_reason({**good, "available_mb": 4000, "compressor_mb": 2300}, good))
        self.assertIsNone(gate.safety_reason({**good, "swap_used_mb": gate.STRICT_SWAP_STOP_MB - 1}))
        self.assertIn("safety stop", gate.safety_reason({**good, "swap_used_mb": gate.STRICT_SWAP_STOP_MB}))
        self.assertIn("safety stop", gate.safety_reason({**good, "swap_used_mb": gate.SWAP_CEILING_MB + 100}))

    def test_headroom_budget_is_a_stricter_start_condition_not_a_weaker_floor(self):
        # The stopped run began at 8,540.2 MiB and bottomed at 2,693.5 MiB: that start must now be postponed.
        self.assertIsNotNone(gate.headroom_reason({"available_mb": 8540.2}))
        self.assertLess(8540.2 - gate.OBSERVED_FOOTPRINT_MB, gate.FLOOR_MB)
        self.assertIsNone(gate.headroom_reason({"available_mb": gate.REQUIRED_START_MB}))
        self.assertIsNotNone(gate.headroom_reason({"available_mb": gate.REQUIRED_START_MB - 1}))
        self.assertIsNotNone(gate.headroom_reason({}))
        # The unmeasured candidate arm is budgeted above the reference arm, never at it.
        between = gate.REQUIRED_START_MB + gate.CANDIDATE_EXTRA_MB - 1
        self.assertIsNone(gate.headroom_reason({"available_mb": between}))
        self.assertIsNotNone(gate.headroom_reason({"available_mb": between}, candidate=True))
        self.assertIsNone(gate.headroom_reason({"available_mb": between + 1}, candidate=True))
        self.assertIsNotNone(gate.headroom_reason(None))
        # The in-flight floors are untouched and the budget never sits below them.
        self.assertEqual(gate.FLOOR_MB, 3 * 1024)
        self.assertGreater(gate.REQUIRED_START_MB, gate.FLOOR_MB + 5846.7)
        self.assertIn("never lower the floor", gate.headroom_reason({"available_mb": 1}))

    def test_postponed_gate_starts_nothing(self):
        low = {"power_source": "ac", "battery_percent": 80, "battery_state": "charging", "low_power": False,
               "thermal_readable": True, "thermal_warning": False, "thermal_speed_limit": None,
               "pressure_level": 1, "swap_used_mb": 100, "available_mb": 8300, "compressor_mb": 400}
        with mock.patch.object(gate, "machine_sample", return_value=low), \
                mock.patch.object(gate, "_run_purge", return_value=mock.Mock(returncode=1, stderr="no")), \
                mock.patch.object(gate, "make_prompt") as prompt, \
                mock.patch.object(gate, "run_arm") as arm, \
                mock.patch.object(gate, "checkpoint_fingerprint"), \
                mock.patch("sys.argv", ["bench_layer_gate", "--memory-rules", "strict"]), \
                mock.patch("builtins.print"), \
                mock.patch("shutil.disk_usage") as disk:
            disk.return_value = mock.Mock(free=10**12)
            code = gate.main()
        self.assertEqual(code, 2)
        prompt.assert_not_called()
        arm.assert_not_called()

    GOOD = {"power_source": "ac", "battery_percent": 80, "battery_state": "charging", "low_power": False,
            "thermal_readable": True, "thermal_warning": False, "thermal_speed_limit": None,
            "pressure_level": 1, "swap_used_mb": 100, "compressor_mb": 400}

    def test_short_headroom_is_purged_once_then_rechecked_against_the_same_budget(self):
        readings = iter([{**self.GOOD, "available_mb": 8300},
                    {**self.GOOD, "available_mb": gate.REQUIRED_CANDIDATE_START_MB}])
        calls = []
        sample, reason, record = gate.prepare_headroom(
            sampler=lambda: next(readings), candidate=True,
            runner=lambda: calls.append(1) or mock.Mock(returncode=0, stderr=""), sleep=lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(reason)
        self.assertEqual((record["available_before_mb"], record["available_after_mb"]),
                         (8300, gate.REQUIRED_CANDIDATE_START_MB))
        # A purge that frees too little is still refused: the budget is not relaxed.
        readings = iter([{**self.GOOD, "available_mb": 8300},
                    {**self.GOOD, "available_mb": gate.REQUIRED_CANDIDATE_START_MB - 1}])
        _, reason, record = gate.prepare_headroom(
            sampler=lambda: next(readings), candidate=True,
            runner=lambda: mock.Mock(returncode=0, stderr=""), sleep=lambda _: None)
        self.assertIsNotNone(reason)
        self.assertTrue(record["purged"])

    def test_under_swap_only_rules_a_short_budget_purges_but_never_blocks(self):
        short = {**self.GOOD, "available_mb": 8000}
        calls = []
        runner = lambda: calls.append(1) or mock.Mock(returncode=0, stderr="")
        sample, reason, record = gate.prepare_headroom(sampler=lambda: short, candidate=True, runner=runner,
                                                       sleep=lambda _: None, memory_rules="swap-only")
        self.assertIsNone(reason)
        self.assertEqual(len(calls), 1)                     # purged once, then not blocked although still short
        self.assertTrue(record["headroom_short"])
        # Strict rules still postpone on the same reading.
        _, reason, _ = gate.prepare_headroom(sampler=lambda: short, candidate=True, runner=runner,
                                             sleep=lambda _: None, memory_rules="strict")
        self.assertIsNotNone(reason)
        # Swap-only still refuses an unsafe machine (battery) and a swap ceiling breach.
        for bad in ({**short, "power_source": "battery"}, {**short, "swap_used_mb": gate.SWAP_CEILING_MB}):
            calls.clear()
            _, reason, _ = gate.prepare_headroom(sampler=lambda b=bad: b, candidate=True, runner=runner,
                                                 sleep=lambda _: None, memory_rules="swap-only")
            self.assertIsNotNone(reason)
            self.assertEqual(calls, [])

    def test_purge_is_not_attempted_when_unneeded_unsafe_or_failing(self):
        def run(sample, **kw):
            calls = []
            out = gate.prepare_headroom(sampler=lambda: sample, runner=lambda: calls.append(1) or mock.Mock(
                returncode=kw.get("rc", 0), stderr="denied"), sleep=lambda _: None, candidate=True)
            return calls, out
        calls, (_, reason, _) = run({**self.GOOD, "available_mb": 20000})
        self.assertEqual((calls, reason), ([], None))                      # enough memory: no purge
        for field, value in (("pressure_level", 2), ("power_source", "battery"), ("low_power", True)):
            calls, (_, reason, _) = run({**self.GOOD, "available_mb": 8000, field: value})
            self.assertEqual(calls, [], field)                              # unsafe machine: no purge, still refused
            self.assertIsNotNone(reason)
        calls, (_, reason, record) = run({**self.GOOD, "available_mb": 8000}, rc=1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(record["purged"])                                  # passwordless purge unavailable: refused
        self.assertIsNotNone(reason)

    def test_summary_keeps_every_arm_and_reports_the_spread(self):
        def arm(seconds, gb):
            return {"prefilled": {"prefill_seconds": seconds, "physical_bytes": gb * 10**9, "peak_bytes": 5 * 10**9,
                                  "state": {"token_count": 8000}}}
        arms = {"r1-chunk": arm(200.0, 400), "r1-layer": arm(100.0, 160), "r2-layer": arm(110.0, 160),
                "r2-chunk": arm(210.0, 400), "r3-chunk": arm(190.0, 400), "r3-layer": arm(105.0, 160)}
        summary = gate.summarize_arms(arms)
        self.assertEqual(summary["chunk"]["arms"], 3)
        self.assertEqual(summary["chunk"]["prefill_seconds"], [200.0, 210.0, 190.0])
        self.assertEqual(summary["chunk"]["median_tok_s"], 40.0)
        self.assertEqual(summary["chunk"]["spread_pct"], 10.0)
        self.assertEqual(summary["layer"]["median_tok_s"], round(8000 / 105.0, 2))
        self.assertEqual(summary["layer_over_chunk_median_tok_s"], round(round(8000 / 105.0, 2) / 40.0, 3))

    def test_candidates_map_to_the_exact_plan_identities_and_wider_prompts_need_a_wider_range(self):
        from models.flashnext.settings import modes
        from models.flashnext import layer_major
        for name, expected in gate.ORDERS.items():
            flags = modes.EXPERIMENTAL[name][0]
            with mock.patch.dict("os.environ", flags, clear=False):
                self.assertEqual(layer_major.identity(), expected)

    def test_swap_only_rules_have_one_memory_limit_and_keep_the_non_memory_stops(self):
        good = {**self.GOOD, "available_mb": 3500, "pressure_level": 2, "compressor_mb": 9000, "swap_used_mb": 950}
        # The three rules that stopped the real runs no longer apply...
        self.assertIsNotNone(gate.safety_reason(good))                              # strict: pressure level 2 / low available
        self.assertIsNone(gate.safety_reason(good, {"compressor_mb": 100}, "swap-only"))
        self.assertIsNone(gate.safety_reason({**good, "available_mb": 800}, None, "swap-only"))
        # ...the owner's swap ceiling and a critical-pressure stop do.
        self.assertIsNone(gate.safety_reason({**good, "swap_used_mb": gate.SWAP_CEILING_MB - 1}, None, "swap-only"))
        self.assertIn("ceiling", gate.safety_reason({**good, "swap_used_mb": gate.SWAP_CEILING_MB}, None, "swap-only"))
        self.assertIn("critical", gate.safety_reason({**good, "pressure_level": 4}, None, "swap-only"))
        self.assertIsNotNone(gate.safety_reason({**good, "swap_used_mb": None}, None, "swap-only"))
        # Power, Low Power Mode and thermal stops are not memory rules and stay.
        for field, value in (("power_source", "battery"), ("low_power", True), ("thermal_readable", False)):
            self.assertIsNotNone(gate.safety_reason({**good, field: value}, None, "swap-only"), field)
        self.assertEqual(gate.SWAP_CEILING_MB, 3 * 1024)

    def _decode_row(self, seconds=20.0, tokens=None, logit="a" * 64):
        tokens = tokens or list(range(32))
        return {"logits": {"value": {"shape": [1, 8], "dtype": "bfloat16", "sha256": logit}},
                "prefilled": {"prompt_sha256": "p", "first_token": tokens[0], "state": {"token_count": 2739},
                              "prefix": {}, "physical_bytes": 100 * 10**9, "prefill_seconds": 100.0,
                              "peak_bytes": 5 * 10**9},
                "finished": {"token_ids": tokens, "generation_seconds": seconds, "tail_tokens": 23,
                             "tail_seconds": seconds * 0.7, "physical_bytes_total": 130 * 10**9}}

    def test_decode_pair_requires_exact_equality_and_a_chunk_major_reference(self):
        left, right = self._decode_row(20.0), self._decode_row(15.0)
        self.assertIsNone(gate.validate_decode_pair(left, right))
        for mutate, text in ((lambda r: r["finished"].update(token_ids=list(range(1, 33))), "token"),
                             (lambda r: r["logits"]["value"].update(sha256="b" * 64), "logits"),
                             (lambda r: r["prefilled"].update(state={"token_count": 2739, "x": 1}), "state"),
                             (lambda r: r["prefilled"]["prefix"].update(execution_order="layer-major-v1/w4096"), "chunk-major"),
                             (lambda r: r["finished"].pop("generation_seconds"), "telemetry")):
            changed = self._decode_row(15.0)
            mutate(changed)
            reason = gate.validate_decode_pair(left, changed)
            self.assertIsNotNone(reason, text)

    def test_decode_summary_reports_every_arm_median_spread_and_bytes_per_token(self):
        arms = {"r1-chunk": self._decode_row(20.0), "r1-layer": self._decode_row(10.0), "r2-layer": self._decode_row(12.4),
                "r2-chunk": self._decode_row(22.0), "r3-chunk": self._decode_row(31.0), "r3-layer": self._decode_row(10.0)}
        summary = gate.summarize_decode(arms)
        self.assertEqual(summary["chunk"]["arms"], 3)
        self.assertEqual(summary["chunk"]["median_prefill_tok_s"], 27.39)
        self.assertEqual(summary["candidate_over_reference_median_prefill"], 1.0)
        self.assertEqual(summary["chunk"]["median_horizon_tok_s"], round(31 / 22.0, 3))
        self.assertEqual(summary["layer"]["median_horizon_tok_s"], round(31 / 10.0, 3))
        self.assertEqual(summary["chunk"]["decode_physical_gb_per_token"], [round(30 / 31, 3)] * 3)
        self.assertEqual(summary["candidate_over_reference_median_horizon"], round(round(31 / 10.0, 3) / round(31 / 22.0, 3), 3))

    def test_battery_power_is_allowed_down_to_25_percent_only_under_the_owner_rule(self):
        base = {**self.GOOD, "available_mb": 9000}
        battery = {**base, "power_source": "battery", "battery_state": "discharging", "battery_percent": 80}
        self.assertIsNotNone(gate.safety_reason(battery))                                 # default: AC required
        self.assertIsNone(gate.safety_reason(battery, power_rules="battery-ok", memory_rules="swap-only"))
        self.assertIsNone(gate.safety_reason({**battery, "battery_percent": 25}, power_rules="battery-ok", memory_rules="swap-only"))
        self.assertIsNotNone(gate.safety_reason({**battery, "battery_percent": 24}, power_rules="battery-ok", memory_rules="swap-only"))
        self.assertIsNotNone(gate.safety_reason({**battery, "power_source": "unknown"}, power_rules="battery-ok", memory_rules="swap-only"))
        # Low Power Mode and thermal still stop a battery run; AC reported while discharging is still refused.
        self.assertIsNotNone(gate.safety_reason({**battery, "low_power": True}, power_rules="battery-ok", memory_rules="swap-only"))
        self.assertIsNotNone(gate.safety_reason({**battery, "thermal_warning": True}, power_rules="battery-ok", memory_rules="swap-only"))
        self.assertIsNotNone(gate.safety_reason({**base, "battery_state": "discharging"}, power_rules="battery-ok", memory_rules="swap-only"))

    def test_low_power_mode_is_read_from_the_active_power_profile(self):
        custom = ("Battery Power:\n lowpowermode 1\n sleep 1\nAC Power:\n lowpowermode 0\n sleep 1\n")
        for source, expected in (("battery", True), ("ac", False)):
            with mock.patch.object(gate.H, "Sampler") as sampler, mock.patch.object(gate.H, "_run_text",
                    side_effect=lambda cmd: custom if "custom" in cmd else "No thermal warning level has been recorded"):
                sampler.return_value.sample.return_value = {"power_source": source}
                self.assertEqual(gate.machine_sample()["low_power"], expected)

    def test_thermal_pressure_is_recorded_and_trapping_stops_a_run(self):
        base = {"power_source": "ac", "battery_state": "charging", "battery_percent": 80, "low_power": False,
                "thermal_readable": True, "thermal_warning": False, "thermal_speed_limit": None,
                "swap_used_mb": 900.0, "pressure_level": 1}
        for level in (None, 0, 1, 2):
            self.assertIsNone(gate.safety_reason({**base, "thermal_pressure": level}, memory_rules="swap-only"), level)
        for level in (3, 4):
            self.assertIsNotNone(gate.safety_reason({**base, "thermal_pressure": level}, memory_rules="swap-only"), level)

        def fake(cmd):
            if "custom" in cmd:
                return "AC Power:\n lowpowermode 0\n"
            if cmd[0] == "notifyutil":
                return "com.apple.system.thermalpressurelevel 2\n"
            return "No thermal warning level has been recorded"
        with mock.patch.object(gate.H, "Sampler") as sampler, mock.patch.object(gate.H, "_run_text", side_effect=fake):
            sampler.return_value.sample.return_value = {"power_source": "ac"}
            self.assertEqual(gate.machine_sample()["thermal_pressure"], 2)
        with mock.patch.object(gate.H, "Sampler") as sampler, mock.patch.object(gate.H, "_run_text", return_value=""):
            sampler.return_value.sample.return_value = {"power_source": "ac"}
            self.assertIsNone(gate.machine_sample()["thermal_pressure"])


if __name__ == "__main__":
    unittest.main()


class RestoreCheckValidationTests(unittest.TestCase):
    """``bench_restore_check.validate_restore`` fails closed on anything but an exact restored arm."""

    @staticmethod
    def row(hit, prefilled, tokens=(1, 2, 3) + (0,) * 29, logits="a" * 64, layers=("x",), cold_tokens=2739):
        return {"logits": {"value": {"shape": [1], "dtype": "float32", "sha256": logits}},
                "prefilled": {"prompt_sha256": "p", "first_token": 1, "state": {"token_count": cold_tokens, "layers": list(layers)},
                              "prefix": {"hit_tokens": hit, "prefilled_tokens": prefilled}},
                "finished": {"token_ids": list(tokens)}}

    def test_exact_restore_passes_and_each_difference_is_named(self):
        from models.flashnext.tests.bench import bench_restore_check as R

        ref = self.row(0, 2739)
        self.assertIsNone(R.validate_restore(ref, self.row(2107, 632)))
        self.assertIn("did not restore", R.validate_restore(ref, self.row(0, 2739)))
        self.assertIn("wrong number", R.validate_restore(ref, self.row(2107, 700)))
        self.assertIn("logits differ", R.validate_restore(ref, self.row(2107, 632, logits="b" * 64)))
        self.assertIn("token IDs differ", R.validate_restore(ref, self.row(2107, 632, tokens=(1, 9) + (0,) * 30)))
        self.assertIn("cache arrays differ", R.validate_restore(ref, self.row(2107, 632, layers=("y",))))
        self.assertIn("not cold", R.validate_restore(self.row(5, 2734), self.row(2107, 632)))
