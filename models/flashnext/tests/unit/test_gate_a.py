"""Gate A orchestrator regressions. Attempt 1 (2026-11-08 22:25) was admitted only at the 'minimal' tier because the
orchestrator itself had loaded a tokenizer (0.52 GiB of 'other anonymous memory') before the server started."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from models.flashnext import prefill_readiness as R, purge_recovery as P
from models.flashnext.tests.bench import gate_a as G

GIB = 2 ** 30
PAGE = 16384


def reading(anonymous_gib, wired=1.59, compressor=0.79, available=8.0, cache=1.0):
    pages = lambda gib: int(gib * GIB / PAGE)
    return P.Reading(page=PAGE, free=pages(1.0), inactive=pages(available - 1.0), speculative=0, wired=pages(wired),
                     compressor=pages(compressor), anonymous=pages(anonymous_gib), file_backed=pages(cache), pressure=1)


class OrchestratorFootprintTests(unittest.TestCase):
    def test_orchestrator_loads_no_tokenizer_or_model_library(self):
        code = ("import sys; import models.flashnext.tests.bench.gate_a as g; g.load_task(); "
                "g.admission_precheck(); "
                "bad = [m for m in ('transformers', 'tokenizers', 'mlx', 'mlx_vlm') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        done = subprocess.run([sys.executable, "-c", code], cwd=str(G.REPO), capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_task_file_is_present_and_about_the_recorded_size(self):
        text = G.load_task()
        meta = json.loads(G.TASK_FILE.with_suffix(".json").read_text())
        self.assertIn("NIMBUS GATEWAY", text)
        self.assertNotIn("1536", text)                      # the facts are only in tool results
        self.assertGreater(len(text), 10000)
        self.assertEqual(meta["target_first_request"], 4450)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(G.CHECKPOINT))
        except Exception:
            self.skipTest("tokenizer not present")
        self.assertEqual(len(tokenizer.encode(text, add_special_tokens=False)), meta["task_text_tokens"])


class DryAdmissionTests(unittest.TestCase):
    TOTAL = 16 * GIB

    def test_the_first_attempts_memory_state_is_refused_before_launch_because_the_gate_figure_was_short(self):
        # 8.0 GiB available and only 1 GiB of file cache to recover: 9.0 GiB counted, the 4K tier needs 9.02 and the launch states
        # --context-limit 8192, so no tier serves it. (The retired anonymous budget used to be what refused this state.)
        ok, detail, values = G.admission_precheck(reading(4.27), self.TOTAL)
        self.assertFalse(ok)
        self.assertIsNone(values["admitted"])
        self.assertIn("context 8192", detail)

    def test_enough_available_memory_admits_the_standard_tier_whatever_the_anonymous_footprint(self):
        for anonymous in (3.0, 4.27, 5.5):
            with self.subTest(anonymous=anonymous):
                ok, _, values = G.admission_precheck(reading(anonymous, available=10.0), self.TOTAL)
                self.assertTrue(ok)
                self.assertIn(values["admitted"], ("long", "standard"))

    def test_precheck_does_not_leak_the_policy_env_it_sets_temporarily(self):
        import os
        keys = ("FLASHNEXT_PREFILL_CHUNK_POLICY", "VEYSTRIX_CONTEXT_REQUIRED", "VEYSTRIX_CONTEXT_LIMIT")
        before = {k: os.environ.get(k) for k in keys}
        G.admission_precheck(reading(3.0, available=10.0), self.TOTAL)
        self.assertEqual({k: os.environ.get(k) for k in keys}, before)

    def test_thresholds_were_not_touched(self):
        self.assertEqual(R.REQUIRED_HEADROOM_BYTES, 4 * GIB)
        self.assertEqual(R.MANDATORY_START_GATE_BYTES, 9 * GIB)
        self.assertEqual(R.COMPRESSION_RATIO, 0.75)
        self.assertEqual(R.cold_prefill_peak_bytes(4096, budgeted=True), int(6.0 * GIB))


class AnalysisTests(unittest.TestCase):
    def test_a_refused_request_is_a_failed_gate_with_named_criteria_and_no_crash(self):
        """Evidence shaped like attempt 1: server ready, harness got a context-limit error, nothing generated."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            error = {"kind": "error", "error": {"message": "context limit exceeded: the prompt is 3935 tokens ..."}}
            (out / "task.jsonl").write_text("\n".join(json.dumps(e) for e in (
                {"type": "session", "sessionId": "s"}, {"type": "final", "sessionId": "s", "reason": error})))
            (out / "summary.json").write_text(json.dumps({
                "stage": {"ready": {"ok": True}, "shutdown": {"ok": True}}, "task": {"exit": 1},
                "leftover_processes": ["watchdog"], "health_after_follow": {}}))
            (out / "server.log").write_text("server stopped\n")
            result = G.analyze(out)
        self.assertFalse(result["passed"])
        c = result["criteria"]
        self.assertTrue(c["server ready"])
        for name in ("harness task exited 0", "glob then grep generated and executed", "follow-up exited 0 with an answer",
                     "no leftover processes", "first request about 4K real tokens"):
            self.assertFalse(c[name], name)

    def test_a_server_that_died_leaves_a_string_health_reading_and_analysis_still_runs(self):
        """Attempt 2 (22:43): the safety guard stopped the server (exit 75); /health then returned an error string."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "task.jsonl").write_text(json.dumps({"type": "session", "sessionId": "s"}))
            (out / "server.log").write_text("FlashNext safety guard stopped the process: memory pressure level 2\n")
            (out / "summary.json").write_text(json.dumps({
                "stage": {"ready": {"ok": True}, "shutdown": {"ok": False, "returncode": 75}},
                "task": {"exit": 0, "ended_by": "server exited"}, "follow_up": None, "leftover_processes": [],
                "health_before": "URLError: connection refused", "health_after_task": "URLError: connection refused",
                "health_after_follow": "URLError: connection refused"}))
            result = G.analyze(out)
        self.assertFalse(result["passed"])
        self.assertEqual(result["measured"]["cache_state"], {})

    def test_a_complete_run_passes(self):
        from models.flashnext.tests.bench.fixtures import agentic_fixture as F
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            call = lambda i, tool, inp: [{"type": "tool_call", "callId": i, "tool": tool, "input": inp},
                                         {"type": "tool_result", "callId": i, "status": "completed", "result": "x"}]
            events = [{"type": "session", "sessionId": "s"}] + call("1", "glob", F.GLOB_CALL["arguments"]) \
                + call("2", "grep", F.GREP_CALL["arguments"]) + [{"type": "final", "text": F.final_answer()}]
            (out / "task.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (out / "follow.jsonl").write_text(json.dumps({"type": "final", "text": "Lower the token expiry first."}))
            (out / "requests.jsonl").write_text(json.dumps({"prompt_tokens": 3935}) + "\n")
            (out / "server.log").write_text("server stopped\n")
            (out / "summary.json").write_text(json.dumps({
                "stage": {"ready": {"ok": True}, "shutdown": {"ok": True}}, "task": {"exit": 0}, "follow_up": {"exit": 0},
                "leftover_processes": [], "health_after_follow": {"cache": {"rebuilt": 1, "reused": 3, "evictions": 0}}}))
            result = G.analyze(out)
        self.assertTrue(result["passed"], result["criteria"])


class EightKProfileTests(unittest.TestCase):
    """The authorized direct 8K AGENTIC gate: same orchestrator, its own profile."""

    def setUp(self):
        self.addCleanup(G.use_profile, "4k")
        G.use_profile("8k")

    def test_profile_arguments_and_client_window_follow_the_budgeting_rule(self):
        args = G.server_args()
        self.assertEqual(args[args.index("--context-limit") + 1], "12288")
        self.assertIn("long-context", args)
        self.assertEqual(args[args.index("--pool-gb") + 1], "0")                  # 0 GB decode pool fallback kept
        patch = Path(G.PROFILE["patch"]).read_text()
        self.assertIn("contextWindow: 16384", patch)                              # server window 12288 + the harness's 4096
        self.assertIn("maxTokens: 12288", patch)

    def test_task_is_about_8k_real_tokens_guided_and_has_no_facts_in_it(self):
        text = G.load_task()
        meta = json.loads(Path(G.PROFILE["task"]).with_suffix(".json").read_text())
        self.assertIn("glob tool", text)
        self.assertIn("grep tool", text)
        self.assertNotIn("1536", text)
        self.assertNotIn("5400", text)
        self.assertTrue(G.PROFILE["band"][0] <= meta["expected_first_request"] <= G.PROFILE["band"][1])
        self.assertAlmostEqual(meta["expected_first_request"], 8200, delta=100)

    def test_dry_admission_for_8k_uses_the_unchanged_rule_and_the_stated_context_requirement(self):
        ok_roomy, _, roomy = G.admission_precheck(reading(2.9, available=11.0), 16 * GIB)
        self.assertTrue(ok_roomy)
        self.assertIn(roomy["admitted"], ("long", "8k"))
        ok_tight, detail, tight = G.admission_precheck(reading(3.6, available=9.2, cache=0.0), 16 * GIB)   # standard would pass; 8K needs 9.47
        self.assertFalse(ok_tight)
        self.assertIsNone(tight["admitted"])
        self.assertIn("context 12288", detail)

    def test_glob_then_grep_is_judged_on_order_and_completion_not_on_how_many_globs(self):
        calls = [{"tool": "glob", "callId": "a"}, {"tool": "glob", "callId": "b"}, {"tool": "grep", "callId": "c"}]
        done = [{"callId": i, "status": "completed"} for i in "abc"]
        self.assertTrue(G._glob_then_grep(calls, done, strict=False))
        self.assertFalse(G._glob_then_grep(calls, done, strict=True))               # Gate A wanted exactly glob, grep
        self.assertFalse(G._glob_then_grep(calls[:2], done[:2], strict=False))      # glob only: the attempt-2 behaviour
        self.assertFalse(G._glob_then_grep(list(reversed(calls)), done, strict=False))
        self.assertFalse(G._glob_then_grep(calls, done[:2] + [{"callId": "c", "status": "failed"}], strict=False))

    def test_phase_churn_reads_the_compress_decompress_pagein_counters(self):
        page = 16384
        gib_pages = int(2 ** 30 / page)
        rows = [{"t": 0, "phase": "prefill", "compress": 0, "decompress": 0, "pagein": 0, "available_mb": 4096, "compressor_mb": 1024},
                {"t": 10, "phase": "prefill", "compress": 5 * gib_pages, "decompress": 2 * gib_pages, "pagein": 8 * gib_pages,
                 "available_mb": 3072, "compressor_mb": 5120},
                {"t": 11, "phase": "decode", "compress": 5 * gib_pages, "decompress": 2 * gib_pages, "pagein": 8 * gib_pages,
                 "available_mb": 3500, "compressor_mb": 4000},
                {"t": 20, "phase": "decode", "compress": 5 * gib_pages, "decompress": 3 * gib_pages, "pagein": 9 * gib_pages,
                 "available_mb": 3500, "compressor_mb": 3000}]
        churn = G.phase_churn(rows)
        self.assertEqual(churn["prefill"]["compress_gib"], 5.0)
        self.assertEqual(churn["prefill"]["pagein_gib"], 8.0)
        self.assertEqual(churn["prefill"]["min_available_gib"], 3.0)
        self.assertEqual(churn["decode"]["decompress_gib"], 1.0)


if __name__ == "__main__":
    unittest.main()
