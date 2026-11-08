"""Offline tests for the quality suite: checkers, case definitions, the runner (fake backend), the comparison gate,
and the bench hook. No model is loaded."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

from models.flashnext.tests.bench import quality_compare as C
from models.flashnext.tests.bench import quality_suite as Q


class CheckerTests(unittest.TestCase):
    def test_number_prefers_labelled_answer_then_last_number(self):
        self.assertTrue(Q.check_number("Answer: 160", 160)[0])
        self.assertTrue(Q.check_number("60 km in 45 min is 80 km/h so 160", 160)[0])
        self.assertFalse(Q.check_number("Answer: 150", 160)[0])
        self.assertFalse(Q.check_number("no idea", 1)[0])
        self.assertTrue(Q.check_number("Answer: 1,200", 1200)[0])

    def test_yes_no(self):
        self.assertTrue(Q.check_yes_no("Answer: yes", "yes")[0])
        self.assertFalse(Q.check_yes_no("Answer: no", "yes")[0])

    def test_contains_all_none_any(self):
        self.assertTrue(Q.check_contains("7390, heron-2026-q4", all_of=["7390"], none_of=["7319"])[0])
        self.assertFalse(Q.check_contains("7390 or 7319", all_of=["7390"], none_of=["7319"])[0])
        self.assertTrue(Q.check_contains("x b", any_of=["a", "b"])[0])
        self.assertFalse(Q.check_contains("x", any_of=["a", "b"])[0])

    def test_bullets(self):
        good = "\n".join(f"- point {i}" for i in range(5))
        self.assertTrue(Q.check_bullets(good, 5, 12)[0])
        self.assertFalse(Q.check_bullets(good, 4, 12)[0])
        self.assertFalse(Q.check_bullets("Intro\n" + good, 5, 12)[0])
        self.assertFalse(Q.check_bullets("- " + "word " * 12, 1, 12)[0])

    def test_sentence_json_caps(self):
        self.assertTrue(Q.check_sentence("The blue sea rolls in!", "blue", 1, "!")[0])
        self.assertFalse(Q.check_sentence("The blue blue sea!", "blue", 1, "!")[0])
        self.assertFalse(Q.check_sentence("The blue sea.", "blue", 1, "!")[0])
        keys = {"name": "string", "age": "integer"}
        self.assertTrue(Q.check_json('{"name": "A", "age": 3}', keys)[0])
        self.assertFalse(Q.check_json('{"name": "A", "age": "3"}', keys)[0])
        self.assertFalse(Q.check_json('{"name": "A", "age": true}', keys)[0])
        self.assertFalse(Q.check_json('{"name": "A", "age": 3, "x": 1}', keys)[0])
        self.assertFalse(Q.check_json("not json", keys)[0])
        self.assertTrue(Q.check_caps("BRIGHT GREEN GRASS.", 3)[0])
        self.assertFalse(Q.check_caps("Bright green grass", 3)[0])

    CALL = ("<tool_call>\n<function=list_dir>\n<parameter=path>\nsrc\n</parameter>\n</function>\n"
            "</tool_call>")

    def test_tool_call_parsing_and_validation(self):
        self.assertEqual(Q.parse_tool_calls(self.CALL), [("list_dir", {"path": "src"})])
        self.assertEqual(Q.check_tool(self.CALL, "list_dir", {"path": r"\bsrc\b"}), (True, "ok"))
        self.assertFalse(Q.check_tool(self.CALL, "read_file")[0])
        self.assertFalse(Q.check_tool("no call here", "list_dir")[0])
        unknown = self.CALL.replace("list_dir", "make_coffee")
        self.assertFalse(Q.check_tool(unknown, "make_coffee")[0])          # the parser drops unknown tools
        missing = self.CALL.replace("path", "depth")
        self.assertFalse(Q.check_tool(missing.replace("list_dir", "read_file"), "read_file")[0])
        self.assertFalse(Q.check_tool(self.CALL + "\n" + self.CALL, "list_dir")[0])
        self.assertFalse(Q.check_tool(self.CALL, "list_dir", {"path": "^tests$"})[0])
        open_ended = self.CALL.replace("</tool_call>", "").replace("</function>\n", "")
        self.assertEqual(Q.check_tool(open_ended, "list_dir")[1], "ok (not closed canonically)")

    def test_a_parameter_the_schema_does_not_define_fails_the_call_even_though_the_parser_drops_it(self):
        call = ("<tool_call>\n<function=run_command>\n<parameter=command>\npwd && ls\n</parameter>\n"
                "<parameter=working_directory>\n/\n</parameter>\n<parameter=timeout_seconds>\n30\n</parameter>\n"
                "</function>\n</tool_call>")
        self.assertEqual(Q.parse_tool_calls(call), [("run_command", {"command": "pwd && ls", "timeout_seconds": 30})])
        self.assertEqual(Q.tool_schema_issues(call), ["run_command: unsupported parameter 'working_directory'"])
        passed, detail = Q.check_tool(call, "run_command", {"command": "unittest|pwd"})
        self.assertFalse(passed)
        self.assertIn("working_directory", detail)
        clean = call.replace("<parameter=working_directory>\n/\n</parameter>\n", "")
        self.assertEqual(Q.tool_schema_issues(clean), [])
        self.assertTrue(Q.check_tool(clean, "run_command", {"command": "pwd"})[0])

    def test_schema_issues_cover_unknown_tools_missing_and_mistyped_values(self):
        self.assertEqual(Q.tool_schema_issues("plain text"), [])
        self.assertIn("unknown tool 'make_coffee'", Q.tool_schema_issues("<tool_call>\n<function=make_coffee>\n</function>"))
        self.assertIn("read_file: missing required parameter 'path'",
                      Q.tool_schema_issues("<tool_call>\n<function=read_file>\n<parameter=start_line>\n3\n</parameter>\n</function>"))
        bad = "<tool_call>\n<function=read_file>\n<parameter=path>\nx\n</parameter>\n<parameter=start_line>\nabc\n</parameter>\n</function>"
        self.assertTrue(any(i.startswith("read_file:") and "start_line" in i for i in Q.tool_schema_issues(bad)))
        self.assertEqual(Q.tool_schema_issues("<tool_call>junk"), ["tool_call block without a <function=...> name"])

    def test_code_checker_runs_and_restricts(self):
        good = "```python\ndef f(x):\n    return x + 1\n```"
        self.assertTrue(Q.check_code(good, ["f(1) == 2"], "f")[0])
        self.assertFalse(Q.check_code(good, ["f(1) == 3"], "f")[0])
        self.assertFalse(Q.check_code("def g(): pass", ["True"], "f")[0])
        self.assertIn("imports", Q.check_code("import os\ndef f(): pass", ["True"], "f")[1])
        self.assertIn("not allowed", Q.check_code("def f():\n    return open('x')", ["True"], "f")[1])
        self.assertIn("dunder", Q.check_code("def f():\n    return ().__class__", ["True"], "f")[1])
        self.assertFalse(Q.check_code("def f(:\n", ["True"], "f")[0])
        self.assertFalse(Q.check_code("def f():\n    while True: pass", ["f() is None"], "f")[0])
        self.assertIn("definition failed", Q.check_code("def f():\n    pass\nraise ValueError('x')", ["True"], "f")[1])


class CaseTests(unittest.TestCase):
    def test_every_category_and_tier_is_covered(self):
        cases = Q.build_cases()
        categories = {c.category for c in cases}
        self.assertEqual(categories, {"coding", "reasoning", "instruction", "tool_call", "multi_turn",
                                      "retrieval_early", "retrieval_middle", "retrieval_late"})
        smoke = Q.select(cases, "smoke")
        self.assertEqual({c.category for c in smoke}, categories - {"retrieval_middle"})
        self.assertGreater(len(Q.select(cases, "full")), len(smoke))
        self.assertEqual(len({c.id for c in cases}), len(cases))
        self.assertEqual(Q.select(cases, ids=["code_fizzbuzz"])[0].id, "code_fizzbuzz")
        with self.assertRaises(ValueError):
            Q.select(cases, ids=["nope"])
        with self.assertRaises(ValueError):
            Q.select(cases, tier="huge")

    def test_cases_reference_known_systems_and_checkers(self):
        systems = Q.system_prompts()
        for case in Q.build_cases():
            self.assertIn(case.system, systems)
            for turn in case.turns:
                self.assertIn(turn.check[0], Q.CHECKERS)
                self.assertTrue(turn.user or turn.tool_results)

    def test_needles_sit_early_middle_late_and_are_unique(self):
        text = Q.needle_system()
        at = {k: text.index(v[2]) / len(text) for k, v in Q.NEEDLES.items()}
        self.assertLess(at["early"], 0.2)
        self.assertTrue(0.3 < at["middle"] < 0.7)
        self.assertGreater(at["late"], 0.9)
        for _, value, note in Q.NEEDLES.values():
            self.assertEqual(text.count(note), 1)
            self.assertIn(value, note)

    def test_tool_cases_use_the_production_tools(self):
        names = {t["function"]["name"] for t in Q.TOOLS}
        for case in Q.build_cases():
            if case.category == "tool_call":
                self.assertTrue(case.tools)
                self.assertEqual(case.system, "tooling")
                for turn in case.turns:
                    self.assertIn(turn.check[1]["tool"], names)

    def test_digest_is_stable(self):
        self.assertEqual(Q.digest([1, 2, 3]), Q.digest([1, 2, 3]))
        self.assertNotEqual(Q.digest([1, 2, 3]), Q.digest([1, 2, 4]))


class FakeBackend:
    """Answers from a script keyed by the last user text; records the calls."""

    def __init__(self, answers=None, default="OK"):
        self.answers, self.default = answers or {}, default
        self.tape, self.pending, self.calls, self.last_prefix = [], [], [], None
        self._last = ""

    def reset(self):
        self.calls.append("reset")
        self.tape, self.pending = [], []

    def _feed(self, text):
        self._last = text
        self.pending = list(range(len(text.split()) + 3))

    def open_conversation(self, system, user, tools=None, enable_thinking=True, reasoning_effort="xhigh"):
        self.calls.append(("open", enable_thinking))
        self.tools_seen = tools
        self._feed(user)

    def append_user(self, text, enable_thinking=True):
        self.calls.append(("user", enable_thinking))
        self._feed(text)

    def append_tool_results(self, results, enable_thinking=True):
        self.calls.append(("tool", enable_thinking))
        self._feed(" ".join(results))

    def generate(self, max_tokens, **kwargs):
        text = next((v for k, v in self.answers.items() if k in self._last), self.default)
        ids = [(ord(ch) % 50) + 100 for ch in text][:max_tokens]
        self.tape += ids
        self.pending = []
        return text, SimpleNamespace(tokens=len(ids), finish="stop", prefill_seconds=0.0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cases = Q.select(Q.build_cases(), ids=["reason_1", "multi_1"])

    def test_runner_uses_fresh_conversations_greedy_no_thinking_and_records_ids(self):
        backend = FakeBackend({"train": "Answer: 160", "name is Priya": "OK", "cats": "16", "live": "Priya, Lisbon"})
        saved = []
        results = Q.run_suite(backend, self.cases, Q.system_prompts(), save=lambda r: saved.append(len(r)))
        self.assertEqual(saved, [1, 2])
        self.assertEqual(backend.calls.count("reset"), 1)
        self.assertEqual([c for c in backend.calls if c != "reset"],
                         [("open", False), ("open", False), ("user", False), ("user", False)])
        self.assertTrue(all(r["passed"] for r in results))
        turn = results[0]["turns"][0]
        self.assertEqual(turn["ids"], [(ord(ch) % 50) + 100 for ch in "Answer: 160"])
        self.assertEqual(turn["digest"], Q.digest(turn["ids"]))
        self.assertEqual(len(results[1]["turns"]), 3)

    def test_every_turn_records_its_tool_calls_and_schema_issues(self):
        case = Q.select(Q.build_cases(), ids=["tool_3"])
        bad = ("<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n<parameter=mode>\nx\n</parameter>\n"
               "<parameter=start_line>\n10\n</parameter>\n<parameter=end_line>\n20\n</parameter>\n</function>\n</tool_call>")
        turn = Q.run_suite(FakeBackend(default=bad), case, Q.system_prompts())[0]["turns"][0]
        self.assertEqual(turn["tool_calls"][0]["name"], "read_file")
        self.assertEqual(turn["schema_issues"], ["read_file: unsupported parameter 'mode'"])
        self.assertFalse(turn["passed"])

    def test_failures_are_recorded_not_raised(self):
        results = Q.run_suite(FakeBackend(default="no"), self.cases, Q.system_prompts())
        self.assertFalse(any(r["passed"] for r in results))
        self.assertEqual(results[1]["turns"][0]["detail"], "missing ['ok'] banned []")

    def test_tool_result_turn_uses_append_tool_results(self):
        case = Q.select(Q.build_cases(), ids=["multi_3"])
        backend = FakeBackend({"directory": TOOL_REPLY}, default="prefix_cache.py is there")
        results = Q.run_suite(backend, case, Q.system_prompts())
        self.assertIn(("tool", False), backend.calls)
        self.assertTrue(results[0]["passed"])
        self.assertIs(backend.tools_seen, Q.TOOLS)                       # offered through the chat template

    def test_prefix_report_is_taken_on_the_first_turn_only(self):
        backend = FakeBackend({"train": "Answer: 160"})
        backend.last_prefix = {"hit_tokens": 7}
        results = Q.run_suite(backend, self.cases, Q.system_prompts(),
                              prefix_report=lambda b: dict(b.last_prefix))
        self.assertEqual(results[0]["turns"][0]["prefix"], {"hit_tokens": 7})
        self.assertIsNone(results[1]["turns"][1]["prefix"])


TOOL_REPLY = ("<tool_call>\n<function=list_dir>\n<parameter=path>\nsrc\n</parameter>\n</function>\n</tool_call>")


def result_set(passed, ids=None, categories=None):
    cases = []
    for number, (name, ok) in enumerate(passed.items()):
        tokens = (ids or {}).get(name, [number, 2, 3])
        cases.append({"case": name, "category": (categories or {}).get(name, "coding"), "passed": ok,
                      "turns": [{"ids": tokens, "passed": ok, "generated": len(tokens)}]})
    return {"cases": cases}


class CompareTests(unittest.TestCase):
    def test_same_plan_requires_identical_tokens(self):
        ref = result_set({"a": True, "b": True})
        same = C.compare(ref, result_set({"a": True, "b": True}), "same-plan")
        self.assertEqual(same["verdict"], "PASS")
        self.assertTrue(same["bit_identical"])
        drift = C.compare(ref, result_set({"a": True, "b": True}, ids={"b": [1, 2, 9]}), "same-plan")
        self.assertEqual(drift["verdict"], "FAIL")
        self.assertEqual(drift["diverged_cases"], [{"case": "b", "turn": 0, "token": 2}])
        self.assertTrue(drift["task_level"]["regressions"] == [])        # still passes the task, still fails the gate

    def test_reordered_allows_token_drift_but_not_task_regressions(self):
        ref = result_set({"a": True, "b": True})
        drift = C.compare(ref, result_set({"a": True, "b": True}, ids={"b": [1, 2, 9]}), "reordered")
        self.assertEqual(drift["verdict"], "PASS")
        self.assertFalse(drift["bit_identical"])
        worse = C.compare(ref, result_set({"a": True, "b": False}), "reordered")
        self.assertEqual(worse["verdict"], "FAIL")
        self.assertEqual(worse["task_level"]["regressions"], ["b"])
        better = C.compare(result_set({"a": True, "b": False}), result_set({"a": True, "b": True}), "reordered")
        self.assertEqual(better["verdict"], "PASS")
        self.assertEqual(better["task_level"]["fixes"], ["b"])

    def test_missing_cases_and_length_differences_fail(self):
        ref = result_set({"a": True, "b": True})
        self.assertEqual(C.compare(ref, result_set({"a": True}), "reordered")["verdict"], "FAIL")
        shorter = C.compare(result_set({"a": True}, ids={"a": [1, 2, 3]}), result_set({"a": True}, ids={"a": [1, 2]}),
                            "same-plan")
        self.assertEqual(shorter["diverged_cases"][0]["token"], 2)
        self.assertEqual(C.compare({"cases": []}, {"cases": []})["verdict"], "FAIL")

    def test_category_pass_rate_drop_is_reported(self):
        cats = {"a": "coding", "b": "coding", "c": "reasoning"}
        ref = result_set({"a": True, "b": False, "c": True}, categories=cats)
        cand = result_set({"a": False, "b": True, "c": True}, categories=cats)
        result = C.compare(ref, cand, "reordered")
        self.assertEqual(result["verdict"], "FAIL")                    # a regressed even though the rate is level
        self.assertEqual(result["task_level"]["reference"]["coding"], {"passed": 1, "total": 2})

    def test_absolute_failures_and_schema_flags_are_separate_from_regressions(self):
        def with_issue(results, case, issue):
            out = json.loads(json.dumps(results))
            for c in out["cases"]:
                if c["case"] == case:
                    c["turns"][0]["schema_issues"] = [issue]
            return out

        ref = with_issue(result_set({"a": False, "b": True}), "a", "run_command: unsupported parameter 'working_directory'")
        cand = with_issue(result_set({"a": False, "b": True}), "a", "run_command: unsupported parameter 'working_directory'")
        result = C.compare(ref, cand, "reordered")
        self.assertEqual(result["verdict"], "PASS", "a failure on both paths is not a regression")
        self.assertEqual(result["task_level"]["absolute_failures"], ["a"])
        self.assertEqual(set(result["task_level"]["schema_issues"]["a"]), {"reference", "candidate"})
        self.assertEqual(result["task_level"]["regressions"], [])
        self.assertIn("unsupported parameter 'working_directory'", C.text(result))
        self.assertIn("a model failure, not a regression", C.text(result))

    def test_text_and_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps({"quality": result_set({"a": True})}))
            b.write_text(json.dumps(result_set({"a": True})))             # bare quality object also loads
            out = Path(d) / "out.json"
            with mock.patch("macqwen.results.output_path", return_value=out) as routed:
                self.assertEqual(C.main([str(a), str(b), "--json", "results/x.json"]), 0)
            routed.assert_called_once()                                   # never written to a path of its own
            self.assertEqual(json.loads(out.read_text())["verdict"], "PASS")
            b.write_text(json.dumps(result_set({"a": True}, ids={"a": [9]})))
            self.assertEqual(C.main([str(a), str(b)]), 1)
            b.write_text("{}")
            with self.assertRaises(ValueError):
                C.main([str(a), str(b)])
        self.assertIn("verdict: PASS", C.text(C.compare(result_set({"a": True}), result_set({"a": True}))))


class BenchHookTests(unittest.TestCase):
    def test_run_quality_wraps_the_suite_and_the_conditions_exist(self):
        from models.flashnext.tests.bench import bench_prefill_memory as B
        from models.flashnext.tests.bench import prefill_memory_harness as H

        backend = FakeBackend({"train": "Answer: 160"})
        out = B.run_quality(backend, "reason_1")
        self.assertEqual((out["passed"], out["total"]), (1, 1))
        self.assertEqual(out["suite"], "reason_1")
        self.assertIn("agent", out["systems"])
        for name, policy, cache in (("F2Bn", "balanced", True), ("F2Pn", "", True), ("F2B", "balanced", True)):
            cond = H.ALL_CONDITIONS[name]
            self.assertEqual((cond.chunk_policy, cond.prefix_cache), (policy, cache))
        # the same-plan reference differs from the cached condition only by the cache
        a, b = H.ALL_CONDITIONS["F2Bn"], H.ALL_CONDITIONS["F2B"]
        self.assertEqual((a.ring, a.scratch, a.chunk_tokens, a.linear_state), (b.ring, b.scratch, b.chunk_tokens, b.linear_state))

    def test_suite_is_passed_to_children(self):
        from models.flashnext.tests.bench import bench_prefill_memory as B

        args = SimpleNamespace(suite="smoke", prefix_dir=None, dump_dir=None)
        self.assertEqual(B.diagnostic_args(args)[-2:], ["--suite", "smoke"])
        self.assertEqual(B.diagnostic_overrides(args)["suite"], "smoke")


if __name__ == "__main__":
    unittest.main()
