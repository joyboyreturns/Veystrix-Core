"""Offline tests of the prefill-memory benchmark harness.

Fakes only: no checkpoint, no model, no GPU benchmark, no child process that
loads anything. Real MLX is not needed by these tests either, except for the
production flag constants they compare against.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from macqwen import results
from models.flashnext import prefill_memory, prefill_ring
from models.flashnext.settings.launch import CHAT_ENV, apply_chat_environment
from models.flashnext.tests.bench import bench_prefill_memory as B
from models.flashnext.tests.bench import prefill_memory_harness as H

GB = 1_000_000_000
FLAGS = H.production_flags()
ALL_FLAG_NAMES = set(FLAGS.values())

BATT_AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=1234)\t88%; charging; 0:42 remaining present: true
"""
BATT_DISCHARGING = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=1234)\t24%; discharging; 3:10 remaining present: true
"""
PMSET_LOW = "System-wide power settings:\n lowpowermode         1\n sleep                1\n"
PMSET_NORMAL = "System-wide power settings:\n lowpowermode         0\n sleep                1\n"
THERM_OK = "Note: No thermal warning level has been recorded\nCPU_Speed_Limit = 100\n"
THERM_HOT = "Note: A thermal warning level has been recorded\nCPU_Speed_Limit = 70\n"


def good_sample(**changes):
    sample = {
        "t": 0.0, "battery_percent": 80, "power_source": "ac", "battery_state": "charging",
        "low_power": False, "thermal_speed_limit": 100, "thermal_warning": False,
        "swap_used_mb": 100.0, "pressure_level": 1, "vm": {"swapin": 0, "swapout": 0},
    }
    sample.update(changes)
    return sample


def make_turn(*, prefill_tok_s=20.0, tok_1=10.0, tok_2=9.5, peak=7 * GB, cache=GB // 2,
              active=6 * GB, digest="d1", first=11, kv="k1", ring=None, acquired=False,
              pool_exists=False, retained=0, vm=None, ssd=30 * GB):
    return {
        "name": "measured", "pending_tokens": 1000, "prefilled_tokens": 1000,
        "prefill_tok_s": prefill_tok_s,
        "mem_end_prefill": {"active": active, "peak": peak, "cache": cache},
        "mem_end_decode": {"active": active, "peak": peak, "cache": cache},
        "cache_max": cache, "digest": digest, "first_token": first, "kv_hash": kv,
        "ring": ring, "ring_acquired": acquired, "ring_reason": None if acquired else "off",
        "prefill_calls": [{"tokens": 1000, "after": {
            "pool_exists": pool_exists, "retained": retained}}],
        "decode": {"tok_s_1_64": tok_1, "tok_s_65_128": tok_2},
        "ssd_prefill_bytes": ssd, "vm": vm or {"swapin": 0, "swapout": 0, "pageout": 0},
    }


def make_arm(size, condition, round_index, **changes):
    cond = H.CONDITIONS[condition]
    turn = make_turn(**changes)
    turns = [turn] if cond.turn == "first" else [dict(turn, name="setup"), turn]
    return {"arm_id": f"{size}-{condition}-r{round_index + 1}", "size": size,
            "turn": cond.turn, "condition": condition, "round": round_index,
            "reverse": round_index % 2 == 1, "metal_limit_bytes": H.METAL_LIMIT_BYTES,
            "turns": turns}


class ConditionTests(unittest.TestCase):
    def test_the_overlay_uses_the_production_flag_names(self):
        self.assertEqual(FLAGS["ring"], prefill_ring.FLAG_EXPERTS)
        self.assertEqual(FLAGS["scratch"], prefill_ring.FLAG_SCRATCH)
        self.assertEqual(FLAGS["cache_mb"], prefill_memory.FLAG_CACHE_MB)
        overlay = H.condition_overlay(H.CONDITIONS["F3a"])
        self.assertEqual(overlay[prefill_ring.FLAG_EXPERTS], "512")
        self.assertEqual(overlay[prefill_ring.FLAG_SCRATCH], "1")
        self.assertEqual(overlay[prefill_memory.FLAG_CACHE_MB], "512")

    def test_production_reads_each_condition_as_intended(self):
        for cond in H.CONDITIONS.values():
            env = H.apply_overlay({}, H.condition_overlay(cond))
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(prefill_ring.configured_experts(), cond.ring, cond.id)
                self.assertEqual(prefill_ring.scratch_enabled(), cond.scratch, cond.id)
                self.assertEqual(prefill_memory.cache_limit_mb(), cond.cache_mb, cond.id)

    def test_every_overlay_names_every_flag_so_nothing_leaks_in(self):
        for cond in H.CONDITIONS.values():
            self.assertEqual(set(H.condition_overlay(cond)), ALL_FLAG_NAMES)

    def test_the_baseline_is_exactly_the_production_chat_environment(self):
        leaked = {name: "1" for name in ALL_FLAG_NAMES}
        env = H.apply_overlay(dict(leaked), H.condition_overlay(H.CONDITIONS["F0"]))
        self.assertEqual(env, {})
        apply_chat_environment(env)
        self.assertEqual(env, CHAT_ENV)
        for name in ("F0", "S0"):
            self.assertTrue(H.CONDITIONS[name].is_baseline)

    def test_candidates_are_not_baselines_and_set_only_what_they_name(self):
        for name in ("F1", "F2", "F3a", "F3b", "F4", "S1", "S2"):
            self.assertFalse(H.CONDITIONS[name].is_baseline, name)
        f4 = {k: v for k, v in H.condition_overlay(H.CONDITIONS["F4"]).items() if v is not None}
        self.assertEqual(f4, {FLAGS["cache_mb"]: "512"})

    def test_scratch_conditions_always_carry_the_ring(self):
        for cond in H.CONDITIONS.values():
            if cond.scratch:
                self.assertGreater(cond.ring, 0, cond.id)

    def test_second_turn_arms_launch_with_baseline_flags(self):
        for name in ("S0", "S1", "S2"):
            launch = H.launch_overlay(H.CONDITIONS[name])
            self.assertTrue(all(v is None for v in launch.values()), name)
        launch = H.launch_overlay(H.CONDITIONS["F2"])
        self.assertEqual(launch[FLAGS["scratch"]], "1")

    def test_the_hand_off_to_the_child_command_carries_the_overlay(self):
        base = {"PATH": "x", FLAGS["scratch"]: "1", FLAGS["ring"]: "9"}
        env = B.child_environment(base, H.CONDITIONS["F4"])
        self.assertNotIn(FLAGS["scratch"], env)
        self.assertNotIn(FLAGS["ring"], env)
        self.assertEqual(env[FLAGS["cache_mb"]], "512")
        self.assertEqual(env["PATH"], "x")
        self.assertIn(FLAGS["scratch"], base)       # the caller's dict is untouched


class ScheduleTests(unittest.TestCase):
    def test_arm_ids_are_unique_and_indices_follow_order(self):
        arms = H.build_schedule(["1k", "2k"], list(H.CONDITIONS), 3)
        self.assertEqual(len({a.arm_id for a in arms}), len(arms))
        self.assertEqual([a.index for a in arms], list(range(len(arms))))

    def test_odd_rounds_are_reversed(self):
        arms = [a for a in H.build_schedule(["1k"], ["F0", "F2", "F4"], 4)]
        by_round = {}
        for arm in arms:
            by_round.setdefault(arm.round, []).append(arm.condition)
        self.assertEqual(by_round[0], ["F0", "F2", "F4"])
        self.assertEqual(by_round[1], ["F4", "F2", "F0"])
        self.assertEqual(by_round[2], by_round[0])
        self.assertEqual(by_round[3], by_round[1])
        self.assertEqual([a.reverse for a in arms if a.condition == "F0"],
                         [False, True, False, True])

    def test_every_condition_runs_in_every_round(self):
        arms = H.build_schedule(["1k"], ["F0", "F2", "S0", "S1"], 3)
        for name in ("F0", "F2", "S0", "S1"):
            self.assertEqual(sorted(a.round for a in arms if a.condition == name), [0, 1, 2])

    def test_sizes_run_in_ascending_order_whatever_the_request_order(self):
        arms = H.build_schedule(["4k", "1k", "2k"], ["F0"], 1)
        self.assertEqual([a.size for a in arms], ["1k", "2k", "4k"])

    def test_first_turn_block_then_second_turn_block_then_guarded(self):
        arms = H.build_schedule(["1k"], list(H.CONDITIONS), 2)
        kinds = [("guarded" if H.CONDITIONS[a.condition].guarded else H.CONDITIONS[a.condition].turn)
                 for a in arms]
        self.assertEqual(kinds, sorted(kinds, key=["first", "second", "guarded"].index))
        self.assertEqual(arms[-1].condition, "F1")

    def test_the_guarded_condition_runs_at_1k_only(self):
        arms = H.build_schedule(["1k", "2k", "4k"], list(H.CONDITIONS), 3)
        self.assertEqual({a.size for a in arms if a.condition == "F1"}, {"1k"})

    def test_a_guarded_only_request_at_a_larger_size_schedules_nothing_for_it(self):
        self.assertEqual(H.build_schedule(["2k"], ["F1"], 3), [])

    def test_bad_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            H.build_schedule(["1k"], ["F0"], 0)
        with self.assertRaises(ValueError):
            H.build_schedule(["1k"], ["nope"], 1)
        with self.assertRaises(ValueError):
            H.build_schedule(["8k"], ["F0"], 1)

    def test_the_default_matrix_covers_the_required_sizes_and_conditions(self):
        self.assertEqual(set(H.SIZES), {"1k", "2k", "4k"})
        self.assertEqual(set(H.CONDITIONS),
                         {"F0", "F1", "F2", "F3a", "F3b", "F4", "S0", "S1", "S2"})

    def test_dry_run_text_names_every_arm(self):
        arms = H.build_schedule(["1k"], ["F0", "S1"], 2)
        lines = B.describe_schedule(arms, 128)
        self.assertEqual(len(lines), len(arms))
        self.assertTrue(any("flags_for_turn_2" in line for line in lines))


class PromptTests(unittest.TestCase):
    class Tokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    def test_prompts_are_reproducible_and_distinct(self):
        source = "".join(chr(97 + i % 26) for i in range(30000))
        one = H.build_prompts(self.Tokenizer(), source, 1024)
        two = H.build_prompts(self.Tokenizer(), source, 1024)
        self.assertEqual(one, two)
        self.assertNotEqual(one["first_hash"], one["later_hash"])
        self.assertEqual(one["first_ids"], 1024 - H.TEMPLATE_OVERHEAD_FIRST)
        self.assertTrue(one["first"].startswith(H.FIRST_PREFACE))
        self.assertEqual(one["first"][len(H.FIRST_PREFACE):], source[:one["first_ids"]])
        self.assertEqual(one["later"][len(H.LATER_PREFACE):],
                         source[H.FOLLOW_UP_OFFSET:H.FOLLOW_UP_OFFSET + one["later_ids"]])

    def test_a_different_size_changes_the_text_and_a_short_source_is_refused(self):
        source = "x" * 30000
        small = H.build_prompts(self.Tokenizer(), source, 1024)
        large = H.build_prompts(self.Tokenizer(), source, 4096)
        self.assertNotEqual(small["first_hash"], large["first_hash"])
        with self.assertRaises(ValueError):
            H.build_prompts(self.Tokenizer(), "x" * 100, 1024)

    def test_the_source_file_is_long_enough_for_every_size(self):
        text = B.SOURCE_TEXT.read_text()
        self.assertGreater(len(text) // 4, H.FOLLOW_UP_OFFSET + 4096)   # at least ~4 chars per token

    def test_digests_are_stable(self):
        self.assertEqual(H.digest_ids([1, 2, 3]), H.digest_ids((1, 2, 3)))
        self.assertNotEqual(H.digest_ids([1, 2, 3]), H.digest_ids([1, 2, 4]))
        self.assertEqual(len(H.digest_text("a")), 16)


class ParserTests(unittest.TestCase):
    def test_battery(self):
        self.assertEqual(H.parse_battery(BATT_AC),
                         {"percent": 88, "source": "ac", "state": "charging"})
        self.assertEqual(H.parse_battery(BATT_DISCHARGING),
                         {"percent": 24, "source": "battery", "state": "discharging"})
        self.assertEqual(H.parse_battery(""), {"percent": None, "source": None, "state": None})

    def test_low_power_mode(self):
        self.assertTrue(H.parse_low_power(PMSET_LOW))
        self.assertFalse(H.parse_low_power(PMSET_NORMAL))
        self.assertIsNone(H.parse_low_power("nothing here"))

    def test_thermal(self):
        self.assertEqual(H.parse_thermal(THERM_OK), {"speed_limit": 100, "warning": False})
        self.assertEqual(H.parse_thermal(THERM_HOT), {"speed_limit": 70, "warning": True})
        self.assertEqual(H.parse_thermal(""), {"speed_limit": None, "warning": False})

    def test_swap_and_pressure(self):
        text = "vm.swapusage: total = 3072.00M  used = 1023.25M  free = 2048.75M  (encrypted)"
        self.assertAlmostEqual(H.parse_swap_used_mb(text), 1023.25)
        self.assertAlmostEqual(H.parse_swap_used_mb("used = 2.00G"), 2048.0)
        self.assertIsNone(H.parse_swap_used_mb(""))
        self.assertEqual(H.parse_pressure_level("4\n"), 4)
        self.assertIsNone(H.parse_pressure_level("error"))

    def test_the_sampler_assembles_one_reading_from_the_commands(self):
        answers = {
            ("pmset", "-g", "batt"): BATT_DISCHARGING, ("pmset", "-g"): PMSET_NORMAL,
            ("pmset", "-g", "therm"): THERM_OK,
            ("sysctl", "vm.swapusage"): "used = 10.00M",
            ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1",
        }
        sampler = H.Sampler(runner=lambda cmd: answers[tuple(cmd)], vm=lambda: {"swapin": 7},
                            clock=lambda: 5.0, load=lambda: 1.5)
        sample = sampler.sample()
        self.assertEqual(sample["battery_percent"], 24)
        self.assertEqual(sample["power_source"], "battery")
        self.assertFalse(sample["low_power"])
        self.assertEqual(sample["thermal_speed_limit"], 100)
        self.assertEqual(sample["swap_used_mb"], 10.0)
        self.assertEqual(sample["pressure_level"], 1)
        self.assertEqual((sample["vm"], sample["load"], sample["t"]), ({"swapin": 7}, 1.5, 5.0))

    def test_a_failing_command_reads_as_unknown_not_as_a_crash(self):
        sample = H.Sampler(runner=lambda cmd: "").sample()
        self.assertIsNone(sample["battery_percent"])
        self.assertIsNone(sample["low_power"])

    def test_vm_delta(self):
        self.assertEqual(H.vm_delta({"a": 1, "b": 5}, {"a": 4, "b": 5}), {"a": 3, "b": 0})
        self.assertEqual(H.vm_delta(None, {"a": 1}), {})


class WatchdogTests(unittest.TestCase):
    def dog(self, **policy):
        return H.Watchdog(H.SafetyPolicy(**policy), good_sample())

    def test_a_healthy_sample_does_not_stop(self):
        self.assertIsNone(self.dog().check(good_sample()))

    def test_battery_below_25_on_battery_stops(self):
        reason = self.dog().check(good_sample(power_source="battery", battery_percent=24))
        self.assertIn("24%", reason)
        self.assertIsNone(self.dog().check(good_sample(power_source="battery", battery_percent=25)))

    def test_a_low_charge_on_ac_does_not_stop(self):
        self.assertIsNone(self.dog().check(good_sample(power_source="ac", battery_percent=10)))

    def test_low_power_mode_always_stops(self):
        self.assertIn("Low Power", self.dog().check(good_sample(low_power=True)))
        self.assertIn("Low Power", self.dog(final=True).check(good_sample(low_power=True)))

    def test_final_runs_need_ac_and_a_readable_power_mode(self):
        self.assertIn("AC", self.dog(final=True).check(good_sample(power_source="battery")))
        self.assertIn("discharging", self.dog(final=True).check(good_sample(battery_state="discharging")))
        self.assertIn("AC", self.dog(final=True).check(good_sample(power_source=None)))
        self.assertIn("unreadable", self.dog(final=True).check(good_sample(low_power=None)))
        self.assertIsNone(self.dog(final=True).check(good_sample()))
        self.assertIsNone(self.dog().check(good_sample(power_source="battery", battery_percent=60)))

    def test_pressure_level_2_or_more_stops(self):
        self.assertIn("pressure", self.dog().check(good_sample(pressure_level=2)))
        self.assertIn("pressure", self.dog().check(good_sample(pressure_level=4)))
        self.assertIsNone(self.dog().check(good_sample(pressure_level=1)))

    PAGE = 16384                      # bytes per vm_stat page on Apple silicon
    LIMIT_PAGES = 64 * 1024 * 1024 // PAGE   # 4096 pages is exactly 64 MiB

    def test_swapout_always_stops(self):
        self.assertIn("swapout", self.dog().check(good_sample(), {"swapout": 1}))
        self.assertIn("swapout", self.dog().check(good_sample(), {"swapin": 0, "swapout": 3}))
        self.assertIsNone(self.dog().check(good_sample(), {"swapin": 0, "swapout": 0, "pageout": 9}))

    def test_tiny_historical_swapin_does_not_stop(self):
        # The first real run was stopped by a 4-page (64 KiB) swap-in.
        dog = self.dog(page_bytes=self.PAGE)
        self.assertIsNone(dog.check(good_sample(), {"swapin": 4, "swapout": 0}))
        self.assertIsNone(dog.check(good_sample(), {"swapin": 1000, "swapout": 0}))

    def test_swapin_up_to_64_mib_is_allowed_and_more_stops(self):
        dog = self.dog(page_bytes=self.PAGE)
        self.assertIsNone(dog.check(good_sample(), {"swapin": self.LIMIT_PAGES, "swapout": 0}))
        reason = self.dog(page_bytes=self.PAGE).check(
            good_sample(), {"swapin": self.LIMIT_PAGES + 1, "swapout": 0})
        self.assertIn("swap-in", reason)
        self.assertIn("64 MiB", reason)

    def test_swapin_limit_uses_the_page_size(self):
        # The same page count is four times the bytes on a 64 KiB-page system.
        pages = self.LIMIT_PAGES // 2
        self.assertIsNone(self.dog(page_bytes=self.PAGE).check(good_sample(), {"swapin": pages}))
        self.assertIn("swap-in", self.dog(page_bytes=4 * self.PAGE).check(good_sample(), {"swapin": pages}))

    def test_swapin_limit_is_configurable(self):
        dog = self.dog(page_bytes=self.PAGE, swapin_limit_mib=1.0)
        self.assertIn("swap-in", dog.check(good_sample(), {"swapin": 65}))

    def test_swapin_alone_does_not_hide_pressure_or_swapout(self):
        dog = self.dog(page_bytes=self.PAGE)
        self.assertIn("pressure", dog.check(good_sample(pressure_level=2), {"swapin": 4}))
        self.assertIn("swapout", dog.check(good_sample(), {"swapin": 4, "swapout": 1}))

    def test_the_check_records_what_it_saw(self):
        dog = self.dog(page_bytes=self.PAGE)
        dog.check(good_sample(pressure_level=1, swap_used_mb=120.0), {"swapin": 512, "swapout": 0})
        self.assertEqual(dog.last["swapin_mib"], 8.0)
        self.assertEqual((dog.last["pressure_level"], dog.last["swap_used_mb"]), (1, 120.0))
        self.assertEqual(dog.last["swapout_pages"], 0)

    def test_swap_growth_over_64_mb_stops(self):
        self.assertIn("swap grew", self.dog().check(good_sample(swap_used_mb=165.0)))
        self.assertIsNone(self.dog().check(good_sample(swap_used_mb=164.0)))

    def test_throttling_needs_two_samples_in_a_row(self):
        dog = self.dog()
        self.assertIsNone(dog.check(good_sample(thermal_speed_limit=80)))
        self.assertIn("thermal", dog.check(good_sample(thermal_speed_limit=80)))
        dog = self.dog()
        dog.check(good_sample(thermal_speed_limit=80))
        self.assertIsNone(dog.check(good_sample(thermal_speed_limit=100)))
        self.assertIsNone(dog.check(good_sample(thermal_speed_limit=80)))

    def test_unreadable_values_are_not_a_stop_outside_final_runs(self):
        blank = good_sample(battery_percent=None, power_source=None, low_power=None,
                            thermal_speed_limit=None, swap_used_mb=None, pressure_level=None)
        self.assertIsNone(self.dog().check(blank))


class FakeProc:
    """A child that runs for ``polls`` polls, then exits with ``code``. A
    stubborn one ignores terminate."""

    def __init__(self, code=0, polls=2, stubborn=False):
        self.code, self.polls, self.stubborn = code, polls, stubborn
        self.terminated = self.killed = False
        self._exit = None

    def poll(self):
        if self._exit is not None:
            return self._exit
        if self.killed or (self.terminated and not self.stubborn):
            self._exit = -9 if self.killed else -15
            return self._exit
        self.polls -= 1
        if self.polls < 0:
            self._exit = self.code
        return self._exit

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RunChildTests(unittest.TestCase):
    def run_it(self, proc, samples, timeout=600, grace=1.0):
        clock = Clock()
        queue = list(samples)
        sampler = SimpleNamespace(sample=lambda: queue.pop(0) if len(queue) > 1 else queue[0])
        dog = H.Watchdog(H.SafetyPolicy(), good_sample())
        result = H.run_child(["x"], {}, timeout=timeout, sampler=sampler, watchdog=dog,
                             interval=5, popen=lambda cmd, env: proc, sleep=clock.sleep,
                             clock=clock, grace=grace)
        return result, proc

    def test_a_clean_exit_is_ok(self):
        result, proc = self.run_it(FakeProc(0, polls=2), [good_sample()])
        self.assertEqual((result.status, result.returncode), ("ok", 0))
        self.assertFalse(proc.terminated)
        self.assertGreaterEqual(len(result.samples), 1)

    def test_exit_137_is_recorded_as_a_memory_kill(self):
        result, _ = self.run_it(FakeProc(137, polls=1), [good_sample()])
        self.assertEqual(result.status, "exit_137")
        self.assertIn("SIGKILL", result.message)
        self.assertEqual(H.classify_exit(-9)[0], "exit_137")

    def test_other_exits(self):
        self.assertEqual(H.classify_exit(1)[0], "failed")
        self.assertEqual(H.classify_exit(-11)[0], "signal")
        self.assertEqual(H.classify_exit(0)[0], "ok")

    def test_a_stop_rule_terminates_the_child_and_keeps_the_samples(self):
        result, proc = self.run_it(
            FakeProc(0, polls=100), [good_sample(), good_sample(pressure_level=4)])
        self.assertEqual(result.status, "stopped")
        self.assertIn("pressure", result.stop_reason)
        self.assertTrue(proc.terminated)
        self.assertFalse(proc.killed)
        self.assertEqual(len(result.samples), 2)

    def test_a_child_that_ignores_terminate_is_killed(self):
        result, proc = self.run_it(
            FakeProc(0, polls=100, stubborn=True), [good_sample(low_power=True)])
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        self.assertEqual(result.status, "stopped")

    def test_a_child_over_its_time_limit_is_stopped(self):
        result, proc = self.run_it(FakeProc(0, polls=10_000), [good_sample()], timeout=30)
        self.assertEqual(result.status, "timeout")
        self.assertTrue(proc.terminated)

    def child(self, vm, vm_start, polls=100):
        clock = Clock()
        dog = H.Watchdog(H.SafetyPolicy(page_bytes=16384), good_sample())
        sampler = SimpleNamespace(sample=lambda: good_sample(vm=vm))
        return H.run_child(["x"], {}, timeout=600, sampler=sampler, watchdog=dog,
                           vm_start=vm_start, popen=lambda c, env: FakeProc(0, polls=polls),
                           sleep=clock.sleep, clock=clock, grace=1.0)

    def test_swapout_since_the_arm_began_stops_the_child(self):
        result = self.child({"swapin": 0, "swapout": 10}, {"swapin": 0, "swapout": 0})
        self.assertEqual(result.status, "stopped")
        self.assertIn("swapout", result.stop_reason)

    def test_large_swapin_since_the_arm_began_stops_the_child(self):
        result = self.child({"swapin": 5000, "swapout": 0}, {"swapin": 0, "swapout": 0})
        self.assertEqual(result.status, "stopped")
        self.assertIn("swap-in", result.stop_reason)

    def test_tiny_swapin_does_not_stop_the_child_and_is_traced(self):
        # Historical counters are huge; only growth since the arm's own start counts.
        result = self.child({"swapin": 2_952_269, "swapout": 3_798_763},
                            {"swapin": 2_952_265, "swapout": 3_798_763}, polls=3)
        self.assertEqual(result.status, "ok")
        summary = H.trace_summary(result.trace)
        self.assertGreater(summary["samples"], 0)
        self.assertAlmostEqual(summary["max_swapin_mib"], 4 * 16384 / 1048576, places=2)
        self.assertEqual((summary["max_swapout_pages"], summary["max_pressure_level"]), (0, 1))

    def test_a_stopped_child_still_carries_its_trace(self):
        result = self.child({"swapin": 5000, "swapout": 0}, {"swapin": 0, "swapout": 0})
        self.assertGreater(H.trace_summary(result.trace)["max_swapin_mib"], 64)

    def test_the_trace_of_an_empty_run_is_empty(self):
        self.assertEqual(H.trace_summary([]), {"samples": 0, "max_swapin_mib": 0.0,
                                               "max_swapout_pages": 0,
                                               "max_pressure_level": None,
                                               "min_available_mb": None,
                                               "max_compressor_growth_mb": None,
                                               "compressor_warnings": 0, "trace": []})


class AnalysisTests(unittest.TestCase):
    def test_exactness_passes_when_every_digest_matches(self):
        arms = [make_arm("1k", c, 0) for c in ("F0", "F2", "F4")]
        report = H.exactness(arms)
        self.assertEqual(report[("1k", "first")]["status"], "pass")

    def test_exactness_names_the_field_and_arm_that_differ(self):
        arms = [make_arm("1k", "F0", 0), make_arm("1k", "F2", 0, digest="other")]
        report = H.exactness(arms)[("1k", "first")]
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["mismatches"][0][:2], ("1k-F2-r1", "digest"))

    def test_a_different_kv_hash_or_first_token_fails(self):
        for change in ({"kv": "z"}, {"first": 99}):
            arms = [make_arm("1k", "F0", 0), make_arm("1k", "F2", 0, **change)]
            self.assertEqual(H.exactness(arms)[("1k", "first")]["status"], "fail", change)

    def test_a_missing_kv_hash_is_noted_but_the_digest_still_gates(self):
        arms = [make_arm("1k", "F0", 0, kv="unavailable"), make_arm("1k", "F2", 0, kv="unavailable")]
        report = H.exactness(arms)[("1k", "first")]
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["notes"])
        arms = [make_arm("1k", "F0", 0, kv="unavailable"),
                make_arm("1k", "F2", 0, kv="unavailable", digest="x")]
        self.assertEqual(H.exactness(arms)[("1k", "first")]["status"], "fail")

    def test_a_missing_digest_is_a_mismatch_not_a_pass(self):
        arms = [make_arm("1k", "F0", 0), make_arm("1k", "F2", 0, digest=None)]
        self.assertEqual(H.exactness(arms)[("1k", "first")]["status"], "fail")

    def test_groups_and_turns_are_compared_separately(self):
        arms = [make_arm("1k", "F0", 0), make_arm("2k", "F0", 0, digest="two"),
                make_arm("1k", "S0", 0, digest="second"), make_arm("1k", "S1", 0, digest="second")]
        report = H.exactness(arms)
        self.assertTrue(all(v["status"] == "pass" for v in report.values()))
        self.assertEqual(len(report), 3)

    def test_a_group_without_its_baseline_is_not_a_pass(self):
        self.assertEqual(H.exactness([make_arm("1k", "F2", 0)])[("1k", "first")]["status"],
                         "no_baseline")

    def test_ring_health(self):
        cond = H.CONDITIONS["F2"]
        good = make_arm("1k", "F2", 0, acquired=True, ring={"fallback_capacity": 0, "reads": 40})
        self.assertEqual(H.ring_health(good, cond)["status"], "pass")
        bad = make_arm("1k", "F2", 0, acquired=True, ring={"fallback_capacity": 2})
        self.assertEqual(H.ring_health(bad, cond)["status"], "invalid")
        self.assertEqual(H.ring_health(make_arm("1k", "F2", 0, acquired=False), cond)["status"],
                         "invalid")
        for ring in ({"last_error": "boom"}, {"disabled": True}):
            arm = make_arm("1k", "F2", 0, acquired=True, ring=ring)
            self.assertEqual(H.ring_health(arm, cond)["status"], "invalid", ring)
        self.assertEqual(H.ring_health(make_arm("1k", "F0", 0), H.CONDITIONS["F0"])["status"], "n/a")

    def test_memory_gate_counts_peak_and_cache_against_the_limit(self):
        limit = H.METAL_LIMIT_BYTES
        near = make_arm("1k", "F0", 0, peak=limit - GB, cache=GB // 2 - 1)
        self.assertEqual(H.memory_gate(near)["status"], "pass")
        self.assertEqual(H.memory_gate(near)["headroom"], H.HEADROOM_BYTES + 1)
        over = make_arm("1k", "F0", 0, peak=limit - GB, cache=GB // 2 + 1)
        self.assertEqual(H.memory_gate(over)["status"], "fail")           # one byte under the headroom
        tight = make_arm("1k", "F0", 0, peak=limit - GB // 4, cache=GB // 2)
        self.assertEqual(H.memory_gate(tight)["status"], "fail")
        self.assertLess(H.memory_gate(tight)["headroom"], H.HEADROOM_BYTES)

    def test_memory_gate_takes_the_larger_cache_reading(self):
        arm = make_arm("1k", "F0", 0, peak=10 * GB, cache=GB // 10)
        arm["turns"][-1]["cache_max"] = 3 * GB
        self.assertEqual(H.memory_gate(arm)["status"], "fail")

    def test_scratch_cleanup(self):
        cond = H.CONDITIONS["F2"]
        base = make_arm("1k", "F0", 0, active=6 * GB)
        good = make_arm("1k", "F2", 0, active=6 * GB + 100_000_000)
        self.assertEqual(H.scratch_cleanup(good, cond, base)["status"], "pass")
        for change, text in (({"pool_exists": True}, "pool existed"),
                             ({"retained": 1}, "retained"),
                             ({"active": 6 * GB + 300_000_000}, "above baseline")):
            arm = make_arm("1k", "F2", 0, **change)
            out = H.scratch_cleanup(arm, cond, base)
            self.assertEqual(out["status"], "fail", change)
            self.assertIn(text, " ".join(out["reasons"]))
        self.assertEqual(H.scratch_cleanup(good, H.CONDITIONS["F0"], base)["status"], "n/a")

    def test_window_rate(self):
        stamps = [i * 0.1 for i in range(128)]
        self.assertAlmostEqual(H.window_rate(stamps, 1, 64), 10.0)
        self.assertAlmostEqual(H.window_rate(stamps, 65, 128), 10.0)
        self.assertIsNone(H.window_rate(stamps[:100], 65, 128))
        self.assertIsNone(H.window_rate([1.0] * 128, 1, 64))


class PairedEffectTests(unittest.TestCase):
    def test_a_consistent_large_gain_is_resolved(self):
        out = H.paired_effect({0: 20.0, 1: 20.2, 2: 19.8}, {0: 22.0, 1: 22.1, 2: 21.9}, {1})
        self.assertEqual(out["verdict"], "resolved")
        self.assertAlmostEqual(out["median"], 2.0, places=6)

    def test_an_effect_inside_the_band_is_unresolved(self):
        out = H.paired_effect({0: 20.0, 1: 22.0, 2: 21.0}, {0: 21.2, 1: 21.0, 2: 21.8}, {1})
        self.assertEqual(out["verdict"], "unresolved")
        self.assertIn("band", out["reason"])

    def test_fewer_than_three_pairs_is_unresolved(self):
        out = H.paired_effect({0: 20.0, 1: 20.1}, {0: 30.0, 1: 30.0}, {1})
        self.assertEqual(out["verdict"], "unresolved")
        self.assertIn("need 3", out["reason"])

    def test_a_sign_that_follows_the_order_is_unresolved(self):
        base = {0: 20.0, 1: 20.0, 2: 20.0, 3: 20.0}
        cand = {0: 25.0, 1: 15.0, 2: 25.0, 3: 15.0}
        out = H.paired_effect(base, cand, {1, 3})
        self.assertEqual(out["verdict"], "unresolved")
        out = H.paired_effect({0: 20.0, 1: 20.5, 2: 19.5, 3: 20.0},
                              {0: 26.0, 1: 22.0, 2: 26.0, 3: 22.0}, {1, 3})
        self.assertEqual(out["verdict"], "resolved")       # same sign in both orders

    def test_mixed_signs_are_unresolved(self):
        out = H.paired_effect({0: 20.0, 1: 20.2, 2: 19.8}, {0: 25.0, 1: 26.0, 2: 10.0}, {1})
        self.assertEqual(out["verdict"], "unresolved")

    def test_band_is_half_the_baseline_range(self):
        self.assertEqual(H.resolution_band([10.0, 12.0, 11.0]), 1.0)
        self.assertEqual(H.resolution_band([10.0]), float("inf"))

    def test_only_rounds_both_conditions_ran_are_paired(self):
        out = H.paired_effect({0: 1.0, 1: 1.1, 2: 0.9}, {1: 5.0, 2: 5.0}, set())
        self.assertEqual(out["pairs"], 2)

    def test_effects_by_group_pairs_each_candidate_with_its_own_rounds_baseline(self):
        arms = []
        for r, (base, cand) in enumerate([(20.0, 24.0), (20.4, 24.5), (19.6, 23.5)]):
            arms.append(make_arm("1k", "F0", r, prefill_tok_s=base))
            arms.append(make_arm("1k", "F2", r, prefill_tok_s=cand))
        effects = H.effects_by_group(arms, ("prefill_tok_s",))
        self.assertEqual(list(effects), [("1k", "first", "F2", "prefill_tok_s")])
        self.assertEqual(effects[("1k", "first", "F2", "prefill_tok_s")]["verdict"], "resolved")

    def test_decode_regression_fails_a_resolved_slowdown_only(self):
        arms = []
        for r in range(3):
            arms.append(make_arm("1k", "F0", r, tok_1=10.0 + 0.1 * r, tok_2=10.0 + 0.1 * r))
            arms.append(make_arm("1k", "F2", r, tok_1=8.0 + 0.1 * r, tok_2=10.0 + 0.1 * r))
            arms.append(make_arm("1k", "F4", r, tok_1=10.1 + 0.1 * r, tok_2=10.0 + 0.1 * r))
        verdicts = H.decode_regression(arms)
        self.assertEqual(verdicts[("1k", "first", "F2")]["status"], "fail")
        self.assertEqual(verdicts[("1k", "first", "F4")]["status"], "pass")

    def test_decode_regression_flags_more_swap_than_the_baseline(self):
        arms = [make_arm("1k", "F0", 0), make_arm("1k", "F2", 0, vm={"swapout": 5})]
        self.assertEqual(H.decode_regression(arms)[("1k", "first", "F2")]["status"], "fail")

    def test_worst_headroom(self):
        arms = [make_arm("1k", "F0", 0, peak=7 * GB), make_arm("1k", "F2", 0, peak=9 * GB),
                make_arm("2k", "F0", 0, peak=12 * GB)]
        worst = H.worst_headroom(arms, "1k")
        self.assertEqual(worst, H.METAL_LIMIT_BYTES - 9 * GB - GB // 2)
        self.assertIsNone(H.worst_headroom(arms, "4k"))


# -- the child, with a fake backend ----------------------------------------------

class FakeProbe:
    def __init__(self):
        self.clock = 0.0
        self.active = 4 * GB
        self.peak = 4 * GB
        self.disk = 0
        self.vm_counts = {"swapin": 0, "swapout": 0, "compress": 0}
        self.minor = 0

    def memory(self):
        return {"active": self.active, "peak": self.peak, "cache": GB // 4}

    def reset_peak(self):
        self.peak = self.active

    def metal_limit(self):
        return H.METAL_LIMIT_BYTES

    def vm(self):
        return dict(self.vm_counts)

    def disk_bytes(self):
        return self.disk

    def free_mb(self):
        return 1000.0

    def faults(self):
        return {"minor": self.minor, "major": 0}

    def rss_bytes(self):
        return 5 * GB

    def now(self):
        self.clock += 0.01
        return self.clock


class FakeStore:
    pass


def fake_language(store):
    glu = SimpleNamespace(gate_proj=SimpleNamespace(cache=SimpleNamespace(store=store)))
    return SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(switch_mlp=glu))]))


class FakeRing:
    def stats(self):
        return {"fallback_capacity": 0, "reads": 12, "last_error": None, "disabled": False}


class FakeEnvironment:
    """Fake prefill and ring modules wired the way production calls them."""

    def __init__(self, probe, store, environ):
        self.probe, self.store, self.environ = probe, store, environ
        self.seen = []                       # (turn tokens, flags in environ at call time)
        self.ring = SimpleNamespace(LAST_REASON=[None], _RETAINED=[])
        self.ring.acquire = self._acquire
        self.ring.release = self._release
        self.prefill = SimpleNamespace(prefill_target=self._prefill)
        self.released = []

    def flags(self):
        return {name: self.environ.get(name) for name in sorted(ALL_FLAG_NAMES)
                if self.environ.get(name) is not None}

    def _acquire(self, language, tokens):
        self.ring.LAST_REASON[0] = None
        if int(self.environ.get(FLAGS["ring"], "0")) <= 0:
            self.ring.LAST_REASON[0] = "disabled"
            return None
        self.probe.active += GB          # a scratch pool
        self.probe.minor += 250_000
        return FakeRing()

    def _release(self, state, failed=False):
        if state is not None:
            self.probe.active -= GB
        self.released.append((state is not None, failed))

    def _prefill(self, language, ids, cache, *a, **k):
        state = self.ring.acquire(language, int(ids.shape[1]))
        self.seen.append((int(ids.shape[1]), self.flags()))
        self.probe.active += 2 * GB
        self.probe.peak = max(self.probe.peak, self.probe.active)
        self.probe.active -= 2 * GB
        self.probe.disk += 7 * GB
        self.ring.release(state)
        return (None, None, 5)


class FakeBackend:
    """Prefills only pending tokens, as the real backend does without a replay."""

    def __init__(self, env, language, decode_s=0.02, replay=False):
        self.env, self.language, self.decode_s = env, language, decode_s
        self.replay = replay
        self.pending, self.tape, self.cache = [], [], [SimpleNamespace(state=None)]
        self.calls = []

    def open_conversation(self, system, user, enable_thinking=True):
        self.calls.append(("open", enable_thinking))
        self.pending.extend(range(1000))

    def append_user(self, text, enable_thinking=True):
        self.calls.append(("append", enable_thinking))
        self.pending.extend(range(500))

    def generate(self, max_tokens, on_prefilled=None, on_decode_token=None):
        length = len(self.tape) + len(self.pending) if self.replay else len(self.pending)
        ids = SimpleNamespace(shape=(1, length))
        self.tape.extend(self.pending)
        self.pending = []
        self.env.prefill.prefill_target(self.language, ids, self.cache)
        if on_prefilled:
            on_prefilled()
        out = []
        for index in range(max_tokens):
            self.env.probe.clock += self.decode_s
            out.append(100 + index % 7)
            if on_decode_token:
                on_decode_token(out[-1], "x")
        self.tape.extend(out)
        return "", SimpleNamespace(tokens=len(out), prefill_seconds=2.0, rate=1 / self.decode_s,
                                   finish="length")


class ChildScenarioTests(unittest.TestCase):
    def setUp(self):
        self.probe = FakeProbe()
        self.store = FakeStore()
        self.environ = {}
        self.env = FakeEnvironment(self.probe, self.store, self.environ)
        self.backend = FakeBackend(self.env, fake_language(self.store))
        self.prompts = {"system": "s", "first": "a", "later": "b"}

    def run_condition(self, name, decode=128):
        cond = H.CONDITIONS[name]
        with B.Instrument(self.probe, self.env.prefill, self.env.ring, self.backend.language) as inst:
            return B.run_scenario(self.backend, self.probe, inst, self.store, cond, self.prompts,
                                  decode, self.environ, hasher=lambda cache: "kv")

    def test_a_first_turn_arm_measures_one_turn_with_its_flags(self):
        turns = self.run_condition("F2")
        self.assertEqual([t["name"] for t in turns], ["measured"])
        self.assertEqual(self.env.seen, [(1000, {FLAGS["ring"]: "512", FLAGS["scratch"]: "1"})])
        measured = turns[0]
        self.assertEqual(measured["prefilled_tokens"], 1000)
        self.assertTrue(measured["ring_acquired"])
        self.assertEqual(measured["ring"]["reads"], 12)
        self.assertGreater(measured["scratch"]["acquire_s"], 0)
        self.assertGreater(measured["scratch"]["dispose_s"], 0)
        self.assertEqual(measured["scratch"]["acquire_faults"]["minor"], 250_000)

    def test_a_second_turn_arm_runs_turn_one_under_baseline_flags(self):
        turns = self.run_condition("S2")
        self.assertEqual([t["name"] for t in turns], ["setup", "measured"])
        setup_flags, measured_flags = self.env.seen[0][1], self.env.seen[1][1]
        self.assertEqual(setup_flags, {})
        self.assertEqual(measured_flags, {FLAGS["ring"]: "512", FLAGS["cache_mb"]: "512"})
        self.assertFalse(turns[0]["ring_acquired"])
        self.assertTrue(turns[1]["ring_acquired"])

    def test_the_measured_second_turn_counts_only_the_tokens_actually_prefilled(self):
        turns = self.run_condition("S1")
        self.assertEqual(turns[0]["prefilled_tokens"], 1000)
        self.assertEqual(turns[1]["prefilled_tokens"], 500)     # KV reuse: not 1,500, not 1,000
        self.assertEqual(turns[1]["pending_tokens"], 500)
        self.assertAlmostEqual(turns[1]["prefill_tok_s"], 500 / 2.0)

    def test_a_replayed_turn_is_counted_by_what_was_prefilled_not_by_what_was_appended(self):
        self.backend.replay = True          # the backend re-feeds the whole tape
        turns = self.run_condition("S1")
        self.assertEqual(turns[1]["pending_tokens"], 500)
        self.assertEqual(turns[1]["prefilled_tokens"], 1000 + 128 + 500)
        self.assertAlmostEqual(turns[1]["prefill_tok_s"], (1000 + 128 + 500) / 2.0)

    def test_turn_semantics_use_open_then_append_with_thinking_closed(self):
        self.run_condition("S0")
        self.assertEqual(self.backend.calls, [("open", False), ("append", False)])
        self.backend.calls.clear()
        self.run_condition("F0")
        self.assertEqual(self.backend.calls[0], ("open", False))

    def test_the_baseline_never_touches_the_ring(self):
        turns = self.run_condition("F0")
        self.assertFalse(turns[0]["ring_acquired"])
        self.assertEqual(turns[0]["ring_reason"], "disabled")
        self.assertIsNone(turns[0]["ring"])
        self.assertEqual(self.env.released, [(False, False)])

    def test_memory_ssd_and_vm_are_recorded_per_turn(self):
        turn = self.run_condition("F2")[0]
        self.assertGreater(turn["mem_end_prefill"]["peak"], turn["mem_start"]["active"])
        self.assertEqual(turn["ssd_prefill_bytes"], 7 * GB)
        self.assertEqual(turn["vm"], {"swapin": 0, "swapout": 0, "compress": 0})
        self.assertEqual(turn["rss_end_prefill"], 5 * GB)
        self.assertEqual(turn["prefill_calls"][0]["after"], {
            "active": self.probe.active, "peak": turn["prefill_calls"][0]["after"]["peak"],
            "cache": GB // 4, "pool_exists": False, "retained": 0})

    def test_the_decode_windows_and_digest_are_recorded(self):
        turn = self.run_condition("F0")[0]
        step = 0.02 + 0.01        # decode time plus one fake-probe clock tick per token
        self.assertAlmostEqual(turn["decode"]["tok_s_1_64"], 1 / step, places=3)
        self.assertAlmostEqual(turn["decode"]["tok_s_65_128"], 1 / step, places=3)
        self.assertEqual(turn["generated_tokens"], 128)
        self.assertEqual(len(turn["digest"]), 16)
        self.assertEqual(turn["first_token"], 100)
        self.assertEqual(turn["kv_hash"], "kv")

    def test_the_same_scenario_gives_the_same_digest_whatever_the_flags(self):
        digests = {name: self.fresh(name)["digest"] for name in ("F0", "F2", "F4")}
        self.assertEqual(len(set(digests.values())), 1)

    def fresh(self, name):
        self.setUp()
        return self.run_condition(name)[-1]

    def test_the_wrappers_return_the_original_result_and_restore_the_originals(self):
        before = (self.env.prefill.prefill_target, self.env.ring.acquire, self.env.ring.release)
        with B.Instrument(self.probe, self.env.prefill, self.env.ring, None):
            self.assertIsNot(self.env.prefill.prefill_target, before[0])
            ids = SimpleNamespace(shape=(1, 8))
            self.assertEqual(self.env.prefill.prefill_target(self.backend.language, ids, []),
                             (None, None, 5))
        self.assertEqual((self.env.prefill.prefill_target, self.env.ring.acquire,
                          self.env.ring.release), before)

    def test_the_originals_are_restored_after_an_exception(self):
        before = self.env.prefill.prefill_target
        with self.assertRaises(RuntimeError):
            with B.Instrument(self.probe, self.env.prefill, self.env.ring, None):
                raise RuntimeError("boom")
        self.assertIs(self.env.prefill.prefill_target, before)

    def test_a_pool_left_behind_is_visible_to_the_scratch_gate(self):
        self.store._flashnext_expert_pool = SimpleNamespace(
            slots=100, _slot_of={}, hits=0, misses=0)
        turn = self.run_condition("F2")[0]
        self.assertTrue(turn["prefill_calls"][-1]["after"]["pool_exists"])
        arm = {"turns": [turn]}
        self.assertEqual(H.scratch_cleanup(arm, H.CONDITIONS["F2"], None)["status"], "fail")

    def test_pool_state_reports_hits_and_slots(self):
        self.assertEqual(B.pool_state(self.store), {"exists": False})
        self.store._flashnext_expert_pool = SimpleNamespace(
            slots=1900, _slot_of={1: 1, 2: 2}, hits=9, misses=3)
        self.assertEqual(B.pool_state(self.store),
                         {"exists": True, "slots": 1900, "used": 2, "hits": 9, "misses": 3})
        self.assertEqual(B.pool_state(None), {"exists": False})

    def test_find_store_walks_the_streamed_layers(self):
        self.assertIs(B.find_store(self.backend.language), self.store)
        self.assertIsNone(B.find_store(SimpleNamespace(model=SimpleNamespace(layers=[]))))


# -- the orchestrator --------------------------------------------------------------

class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "results"
        patcher = mock.patch.object(results, "RESULTS_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(results.ENVIRONMENT_KEY, None)
        self.launched = []

    def args(self, **changes):
        base = dict(sizes=["1k"], conditions=["F0", "F2"], rounds=2, decode=128, interval=1.0,
                    settle=0.0, final=False, record=None, json=None)
        base.update(changes)
        return SimpleNamespace(**base)

    def sampler(self, **changes):
        return SimpleNamespace(sample=lambda: good_sample(**changes))

    def popen(self, make=None, code=0, polls=1):
        """A fake ``Popen`` that writes the arm's result file like the child would."""
        def start(command, env):
            self.launched.append((command, env))
            path = Path(command[command.index("--json") + 1])
            size = command[command.index("--size") + 1]
            condition = command[command.index("--condition") + 1]
            index = int(command[command.index("--round") + 1]) - 1
            if code == 0:
                arm = (make or (lambda s, c, r: make_arm(s, c, r)))(size, condition, index)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(arm))
            return FakeProc(code, polls=polls)
        return start

    def records(self):
        path = next(self.root.rglob("records.jsonl"))
        return [json.loads(line) for line in path.read_text().splitlines()]

    def run_it(self, args=None, sampler=None, popen=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return B.orchestrate(args or self.args(), sampler=sampler or self.sampler(),
                                 popen=popen or self.popen(), sleep=lambda s: None)

    def test_a_complete_run_records_the_lifecycle_and_validations(self):
        self.assertEqual(self.run_it(), 0)
        records = self.records()
        kinds = [r["type"] for r in records]
        self.assertEqual(kinds[0], "run")
        self.assertEqual(kinds.count("arm"), 4)
        self.assertEqual(kinds[-1], "summary")
        self.assertEqual(records[-1]["status"], "completed")
        self.assertIn("validation", kinds)
        arm = next(r for r in records if r["type"] == "arm")
        for key in ("environment_before", "environment_after", "command", "metrics", "size",
                    "reverse", "order_index"):
            self.assertIn(key, arm)
        start = records[0]["metadata"]
        self.assertIn("sources", start)
        self.assertTrue(start["sources"])
        self.assertEqual(start["schedule"][0], "1k-F0-r1")

    def test_the_record_says_whether_the_run_can_support_a_speed_claim(self):
        self.run_it(self.args(rounds=1))
        meta = self.records()[0]["metadata"]
        self.assertFalse(meta["speed_evidence"])
        self.assertIn("not a speed measurement", meta["cache_warning"])
        self.setUp()
        self.run_it(self.args(rounds=3, conditions=["F0"]))
        meta = self.records()[0]["metadata"]
        self.assertTrue(meta["speed_evidence"])
        self.assertIsNone(meta["cache_warning"])

    def test_a_short_run_prints_the_cache_warning(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            B.orchestrate(self.args(rounds=1), sampler=self.sampler(), popen=self.popen(), sleep=lambda s: None)
        self.assertIn("not a speed measurement", out.getvalue())

    def test_children_run_in_the_scheduled_reversed_order(self):
        self.run_it()
        ids = [c[c.index("--arm-id") + 1] for c, _ in self.launched]
        self.assertEqual(ids, ["1k-F0-r1", "1k-F2-r1", "1k-F2-r2", "1k-F0-r2"])

    def test_each_child_gets_only_its_own_flags_in_a_fresh_environment(self):
        with mock.patch.dict(os.environ, {FLAGS["cache_mb"]: "999"}):
            self.run_it()
        for command, env in self.launched:
            condition = command[command.index("--condition") + 1]
            expected = {k: v for k, v in H.launch_overlay(H.CONDITIONS[condition]).items()
                        if v is not None}
            got = {k: v for k, v in env.items() if k in ALL_FLAG_NAMES}
            self.assertEqual(got, expected, condition)

    def test_every_child_command_is_a_fresh_module_process_with_a_results_path(self):
        self.run_it()
        for command, _ in self.launched:
            self.assertEqual(command[1:3], ["-m", B.MODULE])
            self.assertIn("--child", command)
            self.assertTrue(Path(command[command.index("--json") + 1]).is_relative_to(self.root.resolve()))

    def test_a_preflight_stop_starts_nothing_and_says_why(self):
        status = self.run_it(sampler=self.sampler(low_power=True))
        self.assertEqual(status, 3)
        self.assertEqual(self.launched, [])
        records = self.records()
        self.assertEqual(records[-1]["status"], "stopped")
        self.assertIn("Low Power", records[-1]["summary"]["reason"])
        self.assertEqual(records[1]["error_type"], "safety_stop")

    def test_final_runs_refuse_battery_power(self):
        status = self.run_it(self.args(final=True), sampler=self.sampler(power_source="battery"))
        self.assertEqual(status, 3)
        self.assertEqual(self.launched, [])

    def test_battery_runs_are_marked_preliminary(self):
        self.run_it(sampler=self.sampler(power_source="battery", battery_percent=60))
        self.assertTrue(self.records()[0]["metadata"]["preliminary"])
        self.setUp()
        self.run_it(self.args(final=True))
        self.assertFalse(self.records()[0]["metadata"]["preliminary"])

    def test_a_stop_rule_during_a_child_halts_the_run_and_keeps_partial_records(self):
        calls = {"n": 0}

        def sample():
            calls["n"] += 1
            return good_sample(pressure_level=4 if calls["n"] > 6 else 1)

        proc_factory = self.popen(polls=50)
        status = self.run_it(sampler=SimpleNamespace(sample=sample), popen=proc_factory)
        self.assertEqual(status, 3)
        records = self.records()
        self.assertEqual(records[-1]["status"], "stopped")
        self.assertIn("pressure", records[-1]["summary"]["reason"])
        self.assertTrue(any(r["type"] == "failure" and r["error_type"] == "stopped" for r in records))
        self.assertLess(len([r for r in records if r["type"] == "arm"]), 4)

    def test_exit_137_stops_the_run_and_is_recorded(self):
        status = self.run_it(popen=self.popen(code=137))
        self.assertEqual(status, 3)
        failure = next(r for r in self.records() if r["type"] == "failure")
        self.assertEqual(failure["error_type"], "exit_137")
        self.assertEqual(failure["details"]["returncode"], 137)
        self.assertEqual(len(self.launched), 1)

    def test_an_ordinary_child_failure_is_recorded_and_stops_the_run(self):
        outcomes = iter([1, 0, 0, 0])
        base = self.popen()

        def start(command, env):
            code = next(outcomes)
            if code:
                self.launched.append((command, env))
                return FakeProc(code, polls=1)
            return base(command, env)

        self.assertEqual(self.run_it(popen=start), 1)
        records = self.records()
        self.assertEqual([r["error_type"] for r in records if r["type"] == "failure"], ["failed"])
        self.assertEqual(len([r for r in records if r["type"] == "arm"]), 0)
        self.assertEqual(len(self.launched), 1)

    def test_a_digest_mismatch_is_reported_as_a_failed_validation(self):
        def make(size, condition, index):
            return make_arm(size, condition, index, digest="bad" if condition == "F2" else "good")

        self.assertEqual(self.run_it(popen=self.popen(make)), 1)
        validations = {r["arm_id"]: r["status"] for r in self.records() if r["type"] == "validation"}
        self.assertEqual(validations["exactness-1k-first"], "fail")

    def test_the_next_size_is_skipped_when_headroom_was_under_2_gb(self):
        def make(size, condition, index):
            return make_arm(size, condition, index, peak=H.METAL_LIMIT_BYTES - GB)

        status = self.run_it(self.args(sizes=["1k", "2k"], conditions=["F0"], rounds=1),
                             popen=self.popen(make))
        self.assertEqual(status, 0)
        self.assertEqual(len(self.launched), 1)
        failure = next(r for r in self.records() if r["type"] == "failure")
        self.assertEqual(failure["error_type"], "size_gate")

    def test_the_next_size_runs_when_headroom_was_enough(self):
        self.run_it(self.args(sizes=["1k", "2k"], conditions=["F0"], rounds=1))
        self.assertEqual(len(self.launched), 2)

    def test_the_guarded_condition_is_skipped_after_a_failed_gate(self):
        def make(size, condition, index):
            return make_arm(size, condition, index,
                            peak=H.METAL_LIMIT_BYTES - GB // 4 if condition == "F2" else 7 * GB)

        self.run_it(self.args(conditions=["F0", "F2", "F1"], rounds=1), popen=self.popen(make))
        ran = [c[c.index("--condition") + 1] for c, _ in self.launched]
        self.assertNotIn("F1", ran)
        self.assertTrue(any(r["type"] == "failure" and r["error_type"] == "validation"
                            for r in self.records()))

    def test_the_guarded_condition_runs_last_when_all_gates_pass(self):
        self.run_it(self.args(conditions=["F0", "F2", "F1"], rounds=1))
        ran = [c[c.index("--condition") + 1] for c, _ in self.launched]
        self.assertEqual(ran[-1], "F1")

    def test_the_battery_is_checked_before_every_arm(self):
        levels = iter([80, 80, 60, 20, 20, 20, 20, 20, 20])

        def sample():
            return good_sample(power_source="battery", battery_percent=next(levels))

        status = self.run_it(sampler=SimpleNamespace(sample=sample))
        self.assertEqual(status, 3)
        self.assertLess(len(self.launched), 4)

    def test_dry_run_starts_nothing_and_writes_nothing(self):
        boom = mock.Mock(side_effect=AssertionError("a child was started"))
        out = io.StringIO()
        with mock.patch.object(B.subprocess, "Popen", boom), \
                mock.patch.object(H.subprocess, "Popen", boom), \
                mock.patch.object(B, "child_main", boom), \
                contextlib.redirect_stdout(out):
            status = B.main(["--dry-run", "--sizes", "1k", "2k", "--rounds", "2"])
        self.assertEqual(status, 0)
        boom.assert_not_called()
        self.assertFalse(self.root.exists())
        text = out.getvalue()
        self.assertIn("1k-F0-r1", text)
        self.assertIn("2k-S2-r2", text)
        self.assertIn("example command", text)

    def test_the_child_needs_its_size_and_condition(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            B.main(["--child"])

    def test_destinations_outside_results_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_it(self.args(record=str(Path(self.directory.name) / "elsewhere.jsonl")))


class CharTokenizer:
    """One token per character: enough for the conversation framing code."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False,
                            enable_thinking=True, reasoning_effort=None):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def ready_args(**changes):
    base = dict(checkpoint="/models/ckpt", sizes=["1k"], conditions=["F0", "F2"], rounds=1, final=False)
    base.update(changes)
    return SimpleNamespace(**base)


def ready_preflight(verdict="READY"):
    return SimpleNamespace(check_checkpoint=lambda path: SimpleNamespace(
        verdict=verdict, by_status=lambda status: [1] if (status == "FAIL" and verdict != "READY") else []))


class ReadinessTests(unittest.TestCase):
    """``--check-ready``: everything the smoke test needs, checked without starting it."""

    def run_ready(self, args=None, sizes=None, contract=lambda: [], sampler=None, preflight=None,
                  environ=None, free_gb=8.0, resolved="/models/ckpt"):
        rows = sizes if sizes is not None else [
            {"size": "1k", "target": 1024, "first": 1025, "later": 1024}]
        with mock.patch.object(B, "prompt_sizes", return_value=rows), \
                mock.patch("macqwen.checkpoints.resolve_flashnext", return_value=Path(resolved)):
            return B.check_ready(
                args or ready_args(), sampler=sampler or SimpleNamespace(sample=lambda: good_sample()),
                preflight_module=preflight or ready_preflight(), tokenizer_loader=lambda path: CharTokenizer(),
                contract=contract, environ=environ if environ is not None else {}, free_gb=free_gb)

    def status(self, report, name):
        return next(c.status for c in report.checks if c.name == name)

    def test_a_good_machine_and_checkpoint_is_ready(self):
        report = self.run_ready()
        self.assertEqual(report.verdict, "READY", report.text())
        for name in ("checkpoint preflight", "backend API", "prompt size 1k", "machine state", "schedule"):
            self.assertEqual(self.status(report, name), "PASS", name)

    def test_the_schedule_lists_the_smoke_arms(self):
        text = self.run_ready().text()
        self.assertIn("2 fresh-process arms", text)
        self.assertIn("1k-F0-r1", text)
        self.assertIn("1k-F2-r1", text)

    def test_a_checkpoint_that_is_not_ready_blocks_the_smoke_test(self):
        for verdict in ("INCOMPLETE", "INCOMPATIBLE", "MISSING"):
            report = self.run_ready(preflight=ready_preflight(verdict))
            self.assertEqual(report.verdict, "BLOCKED", verdict)
            self.assertEqual(self.status(report, "checkpoint preflight"), "FAIL", verdict)

    def test_a_changed_backend_api_blocks_before_a_model_loads(self):
        report = self.run_ready(contract=lambda: ["FlashNextBackend.generate lacks parameter on_prefilled"])
        self.assertEqual(report.verdict, "BLOCKED")
        self.assertIn("on_prefilled", next(c.detail for c in report.checks if c.name == "backend API"))

    def test_the_machine_state_rules_are_the_watchdogs(self):
        for changes in ({"low_power": True}, {"power_source": "battery", "battery_percent": 20},
                        {"pressure_level": 2}):
            report = self.run_ready(sampler=SimpleNamespace(sample=lambda c=changes: good_sample(**c)))
            self.assertEqual(report.verdict, "BLOCKED", changes)
            self.assertEqual(self.status(report, "machine state"), "FAIL", changes)

    def test_battery_power_is_only_a_warning_unless_the_run_is_final(self):
        battery = SimpleNamespace(sample=lambda: good_sample(power_source="battery", battery_percent=60))
        report = self.run_ready(sampler=battery)
        self.assertEqual(report.verdict, "READY")
        self.assertEqual(self.status(report, "power"), "WARN")
        final = self.run_ready(ready_args(final=True), sampler=battery)
        self.assertEqual(final.verdict, "BLOCKED")

    def test_prompt_sizes_off_target_warn_then_fail(self):
        close = [{"size": "1k", "target": 1000, "first": 1050, "later": 1000}]
        far = [{"size": "1k", "target": 1000, "first": 1200, "later": 1000}]
        self.assertEqual(self.status(self.run_ready(sizes=close), "prompt size 1k"), "WARN")
        report = self.run_ready(sizes=far)
        self.assertEqual(self.status(report, "prompt size 1k"), "FAIL")
        self.assertEqual(report.verdict, "BLOCKED")

    def test_a_smoke_run_is_flagged_as_not_a_speed_measurement(self):
        report = self.run_ready(ready_args(rounds=1))
        self.assertEqual(self.status(report, "cache warming"), "WARN")
        self.assertIn("not a speed measurement", next(c.detail for c in report.checks if c.name == "cache warming"))
        self.assertEqual(report.verdict, "READY")          # a warning, not a block
        three = self.run_ready(ready_args(rounds=3))
        self.assertEqual(self.status(three, "cache warming"), "INFO")

    def test_low_free_memory_warns_without_blocking(self):
        report = self.run_ready(free_gb=1.5)
        self.assertEqual(self.status(report, "free memory"), "WARN")
        self.assertEqual(report.verdict, "READY")

    def test_shell_variables_that_would_change_the_baseline_are_flagged(self):
        report = self.run_ready(environ={"FLASHNEXT_EXPERT_POOL_GB": "0", "PATH": "x"})
        self.assertEqual(self.status(report, "environment"), "WARN")
        self.assertIn("FLASHNEXT_EXPERT_POOL_GB", next(c.detail for c in report.checks if c.name == "environment"))
        flags = {name: "1" for name in ALL_FLAG_NAMES}          # these are cleared per arm, so not a leak
        self.assertEqual(self.status(self.run_ready(environ=flags), "environment"), "PASS")

    def test_a_checkpoint_the_child_would_not_resolve_is_a_failure(self):
        with mock.patch("macqwen.checkpoints.resolve_flashnext", side_effect=ValueError("choose a checkpoint\nx")):
            report = B.check_ready(
                ready_args(), sampler=SimpleNamespace(sample=lambda: good_sample()),
                preflight_module=ready_preflight(), tokenizer_loader=lambda p: CharTokenizer(),
                contract=lambda: [], environ={}, free_gb=8.0)
        self.assertEqual(self.status(report, "child resolution"), "FAIL")

    def test_prompt_sizes_are_computed_from_the_real_conversation_framing(self):
        source = "".join(chr(97 + i % 26) for i in range(30000))
        rows = B.prompt_sizes(CharTokenizer(), ["1k"], source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], 1024)
        self.assertGreater(rows[0]["first"], 1024 - 100)
        self.assertGreater(rows[0]["later"], 1024 - 100)

    def test_the_overhead_constants_are_the_ones_measured_with_the_real_tokenizer(self):
        # measured 2026-11-07 with the checkpoint's tokenizer: turn 1 prefills 1025/2049/4097 and
        # turn 2 1024/2048/4096 tokens for targets 1024/2048/4096
        self.assertEqual((H.TEMPLATE_OVERHEAD_FIRST, H.TEMPLATE_OVERHEAD_LATER), (36, 23))

    def test_the_real_backend_still_offers_what_the_harness_calls(self):
        self.assertEqual(B.backend_contract(), [])

    def test_the_contract_check_notices_a_removed_parameter(self):
        from macqwen.backends.flashnext import FlashNextBackend

        def generate(self, max_tokens, on_decode_token=None):
            return None

        with mock.patch.object(FlashNextBackend, "generate", generate):
            problems = B.backend_contract()
        self.assertTrue(any("on_prefilled" in p for p in problems), problems)

    def test_check_ready_starts_nothing_and_writes_nothing(self):
        boom = mock.Mock(side_effect=AssertionError("a child was started"))
        report = SimpleNamespace(verdict="READY", text=lambda: "report text")
        out = io.StringIO()
        with mock.patch.object(B.subprocess, "Popen", boom), mock.patch.object(B, "child_main", boom), \
                mock.patch.object(B, "orchestrate", boom), mock.patch.object(B, "check_ready", return_value=report), \
                contextlib.redirect_stdout(out):
            status = B.main(["--check-ready", "--sizes", "1k", "--conditions", "F0", "F2", "--rounds", "1"])
        self.assertEqual(status, 0)
        boom.assert_not_called()
        self.assertIn("report text", out.getvalue())

    def test_check_ready_exit_status_follows_the_verdict(self):
        report = SimpleNamespace(verdict="BLOCKED", text=lambda: "blocked")
        with mock.patch.object(B, "check_ready", return_value=report), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(B.main(["--check-ready"]), 1)

    def test_the_checkpoint_option_reaches_the_children_through_the_environment(self):
        report = SimpleNamespace(verdict="READY", text=lambda: "")
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(B, "check_ready", return_value=report), \
                contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop("MACQWEN_FLASHNEXT_MODEL", None)
            B.main(["--check-ready", "--checkpoint", "/models/ckpt"])
            self.assertEqual(os.environ["MACQWEN_FLASHNEXT_MODEL"], "/models/ckpt")
            os.environ.pop("MACQWEN_FLASHNEXT_MODEL", None)


class PolicyTests(unittest.TestCase):
    def test_the_case_card_is_valid_and_not_runnable_by_accident(self):
        from models.flashnext.tests.cases import case_prefill_memory as card

        self.assertEqual(card.TEST.id, "prefill-memory")
        self.assertEqual(card.TEST.status, "deferred")
        self.assertFalse(card.TEST.runnable)
        self.assertIn("section 14", card.TEST.controls["plan"])

    def test_the_harness_leaves_production_code_alone(self):
        for name in ("prefill.py", "prefill_ring.py", "prefill_memory.py", "expert_pool.py",
                     "expert_cache.py", "pool_ring.py"):
            text = (Path(B.ROOT) / "models" / "flashnext" / name).read_text()
            self.assertNotIn("prefill_memory_harness", text)
            self.assertNotIn("bench_prefill_memory", text)

    def test_the_harness_modules_do_not_import_mlx_at_load(self):
        for path in (Path(B.__file__), Path(H.__file__)):
            top = [line for line in path.read_text().splitlines()
                   if line.startswith(("import mlx", "from mlx"))]
            self.assertEqual(top, [], path.name)

    def test_the_orchestrator_has_no_default_output_path(self):
        parser_text = Path(B.__file__).read_text()
        self.assertNotIn('"--json", default', parser_text)
        self.assertIn("output_path(", parser_text)


if __name__ == "__main__":
    unittest.main()
