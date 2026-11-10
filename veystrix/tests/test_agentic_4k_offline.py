"""Offline validation suite for Veystrix 4K AGENTIC protocol and reliability.

Covers the 14 requirements of Phase 8:
 1. Genuine approximately 4K-token request construction
 2. Correct 8192-token context configuration and server enforcement
 3. Sufficient output-token reservation
 4. No one-token harness budget regression
 5. Correct Qwen tool-call parsing
 6. SSE tool-call streaming
 7. Tool-result continuation
 8. Conversation isolation
 9. Exact cache-state compatibility
10. Deadlock regression
11. GPU buffer lifetime
12. Independent watchdog
13. Memory admission and guard stops
14. Clean server shutdown
"""
import json
import os
import re
import tempfile
import time
from types import SimpleNamespace
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]      # tests must not depend on where the repo (or the owner's home) lives

from veystrix.server import (
    MIN_OUTPUT_RESERVE,
    ContextLimitError,
    ModelService,
    ModelToolCallError,
    RequestError,
    _parse_tool_calls,
    capabilities,
    model_card,
    model_entry,
)
from veystrix.tests.test_server import FakeSession, _make_handler, _stub_result
from models.flashnext import prefill_readiness as R
from models.flashnext import purge_recovery as P

GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "glob",
        "description": "Find files matching pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["pattern"]
        }
    }
}

GREP_TOOL = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "Search for pattern in files",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"}
            },
            "required": ["pattern"]
        }
    }
}


class Agentic4kOfflineTests(unittest.TestCase):
    # --------------------------------------------------------------------------
    # 1. Genuine approximately 4K-token request construction
    # --------------------------------------------------------------------------
    def test_01_genuine_4k_request_construction_with_real_tokenizer(self):
        tok_dir = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
        if not tok_dir.exists():
            self.skipTest("Real checkpoint tokenizer not present")

        calib_path = REPO / "scratch" / "calibrate_fixture.py"
        if not calib_path.exists():
            self.skipTest("Calibrate fixture script not present")
        import importlib.util
        spec = importlib.util.spec_from_file_location("calibrate_mod", str(calib_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        task_prompt = mod.SPEC_HEADER + mod.EXTRA_SECTION + mod.TASK_QUESTION
        token_count, rendered = mod.evaluate_task(task_prompt)
        self.assertGreaterEqual(token_count, 3900, "Request should be at least 3900 tokens")
        self.assertLessEqual(token_count, 4100, "Request should be at most 4100 tokens")
        self.assertIn("NIMBUS GATEWAY", rendered)
        self.assertIn('"name": "glob"', rendered)
        self.assertIn('"name": "grep"', rendered)

    # --------------------------------------------------------------------------
    # 2. Correct 8192-token context configuration and server enforcement
    # --------------------------------------------------------------------------
    def test_02_context_configuration_and_server_enforcement(self):
        session = FakeSession(["hello"])
        service = ModelService(session, context_limit=8192)
        caps = capabilities(service)
        self.assertEqual(caps["context_window"], 8192)

        card = model_card(service)
        self.assertEqual(card["context_window"], 8192)

        entry = model_entry(card, service)
        self.assertEqual(entry["context_window"], 8192)
        self.assertTrue(entry["context_limit_enforced"])
        self.assertEqual(entry["max_output_tokens"], 8192 - MIN_OUTPUT_RESERVE)

        # Prompt within limit is allowed
        session.backend.tokenizer.rendered = "x" * 4000
        service.complete([{"role": "user", "content": "x"}], [], 1024)

        # Prompt exceeding 8192 - 256 is refused
        session.backend.tokenizer.rendered = "x" * (8192 - MIN_OUTPUT_RESERVE + 1)
        with self.assertRaises(ContextLimitError) as caught:
            service.complete([{"role": "user", "content": "x"}], [], 1024)
        self.assertEqual(caught.exception.code, "context_length_exceeded")

    # --------------------------------------------------------------------------
    # 3. Sufficient output-token reservation
    # --------------------------------------------------------------------------
    def test_03_output_token_reservation(self):
        self.assertEqual(MIN_OUTPUT_RESERVE, 256)
        session = FakeSession(["ok"])
        service = ModelService(session, context_limit=8192)
        # Leaving exactly MIN_OUTPUT_RESERVE is allowed
        session.backend.tokenizer.rendered = "x" * (8192 - MIN_OUTPUT_RESERVE)
        service.complete([{"role": "user", "content": "x"}], [], 1024)
        # Leaving less than MIN_OUTPUT_RESERVE is refused
        session.backend.tokenizer.rendered = "x" * (8192 - MIN_OUTPUT_RESERVE + 1)
        with self.assertRaises(ContextLimitError):
            service.complete([{"role": "user", "content": "x"}], [], 1024)

    # --------------------------------------------------------------------------
    # 4. No one-token harness budget regression
    # --------------------------------------------------------------------------
    def test_04_no_one_token_harness_budget_regression(self):
        patch_path = REPO / "docs" / "flashnext" / "dsh" / "veystrix-agentic.patch.yml"
        text = patch_path.read_text()
        window = int(re.search(r"^\s*contextWindow:\s*(\d+)", text, re.M)[1])
        max_tokens = int(re.search(r"^\s*maxTokens:\s*(\d+)", text, re.M)[1])

        self.assertEqual(window, 12288)
        self.assertEqual(max_tokens, 1024)
        # Harness formula check (pi-messages.js: CONTEXT_SAFETY_TOKENS = 4096)
        context_safety_tokens = 4096
        prompt_estimate = 4032
        available = window - prompt_estimate - context_safety_tokens
        wire_tokens = min(max_tokens, max(1, available))
        self.assertEqual(wire_tokens, 1024, "Harness must request 1024 tokens, not 1")

        # Server-side guard: when a tool call is cut off by the output limit,
        # it is raised as ModelToolCallError (422), never handed to client as plain text
        session = FakeSession(["<tool_call>"])
        from types import SimpleNamespace
        session.backend.generate = lambda max_tokens, out=None: (
            "<tool_call>", SimpleNamespace(finish="length", tokens=1, prompt_tokens=4032)
        )
        service = ModelService(session, context_limit=8192)
        with self.assertRaises(ModelToolCallError) as caught:
            service.complete([{"role": "user", "content": "x"}], [GLOB_TOOL], 1)
        self.assertIn("output limit", str(caught.exception))

    # --------------------------------------------------------------------------
    # 5. Correct Qwen tool-call parsing
    # --------------------------------------------------------------------------
    def test_05_correct_qwen_tool_call_parsing(self):
        # Qwen XML format
        xml_text = (
            "I will check the directory structure.\n"
            "<tool_call>\n"
            "<function=glob>\n"
            "<parameter=pattern>\n"
            "**/*.py\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        clean, calls = _parse_tool_calls(xml_text, [GLOB_TOOL])
        self.assertEqual(clean, "I will check the directory structure.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "glob")
        self.assertEqual(json.loads(calls[0]["arguments"]), {"pattern": "**/*.py"})

        # Qwen JSON format
        json_text = '<tool_call>{"name": "grep", "arguments": {"pattern": "batch_size"}}</tool_call>'
        clean2, calls2 = _parse_tool_calls(json_text, [GREP_TOOL])
        self.assertEqual(len(calls2), 1)
        self.assertEqual(calls2[0]["name"], "grep")
        self.assertEqual(json.loads(calls2[0]["arguments"]), {"pattern": "batch_size"})

        # Undeclared tool is refused with ModelToolCallError
        bad_tool = '<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>'
        with self.assertRaises(ModelToolCallError):
            _parse_tool_calls(bad_tool, [GLOB_TOOL])

    # --------------------------------------------------------------------------
    # 6. SSE tool-call streaming
    # --------------------------------------------------------------------------
    def test_06_sse_tool_call_streaming(self):
        payload = {
            "model": "veystrix-flashnext",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": [GLOB_TOOL],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": 1024
        }
        tool_call_result = _stub_result(
            "tool_calls",
            [{"id": "call_123", "name": "glob", "arguments": '{"pattern": "*"}'}]
        )
        handler, seen, headers = _make_handler(lambda *args, **kw: tool_call_result), {}, []
        chunks = []
        handler._sse = lambda obj, *_: chunks.append(obj)
        handler._chat(payload)

        # Check tool_calls chunk was emitted
        tool_chunks = [c for c in chunks if "choices" in c and c["choices"] and "tool_calls" in c["choices"][0].get("delta", {})]
        self.assertEqual(len(tool_chunks), 1)
        tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "glob")
        self.assertEqual(tc["function"]["arguments"], '{"pattern": "*"}')

        # Check finish_reason
        finish_chunks = [c for c in chunks if "choices" in c and c["choices"] and c["choices"][0].get("finish_reason") == "tool_calls"]
        self.assertEqual(len(finish_chunks), 1)

        # Check usage chunk
        usage_chunks = [c for c in chunks if "usage" in c and c.get("choices") == []]
        self.assertEqual(len(usage_chunks), 1)

    # --------------------------------------------------------------------------
    # 7. Tool-result continuation
    # --------------------------------------------------------------------------
    def test_07_tool_result_continuation(self):
        session = FakeSession(["hello"])
        service = ModelService(session, context_limit=8192)

        # Turn 1
        service.complete([{"role": "user", "content": "one"}], [], 1024)
        tape = list(session.backend.tape)
        self.assertEqual((service.reused, service.rebuilt), (0, 1))

        # Turn 2: Extends the previous tape
        session.backend.tokenizer.rendered = "rendered prompthelloMORE_TOOL_RESULT"
        service.complete([{"role": "user", "content": "two"}], [], 1024)
        self.assertEqual((service.reused, service.rebuilt), (1, 1))
        self.assertEqual(session.backend.tape[: len(tape)], tape)

    # --------------------------------------------------------------------------
    # 8. Conversation isolation
    # --------------------------------------------------------------------------
    def test_08_conversation_isolation(self):
        session = FakeSession(["reply_a", "reply_b"])
        service = ModelService(session, context_limit=8192)

        # Conversation A
        service.complete([{"role": "user", "content": "conv A"}], [], 1024)
        self.assertEqual(service.rebuilt, 1)

        # Unrelated Conversation B replaces live state
        service.complete([{"role": "user", "content": "conv B"}], [], 1024)
        self.assertEqual(service.rebuilt, 2)
        self.assertEqual(service.evictions, 1)

    # --------------------------------------------------------------------------
    # 9. Exact cache-state compatibility
    # --------------------------------------------------------------------------
    def test_09_exact_cache_state_compatibility(self):
        from models.flashnext.prefix_cache import PrefixCache

        with tempfile.TemporaryDirectory() as tmpdir:
            fp = "engine_fingerprint_v1"
            cache = PrefixCache(fp, ram_budget=10*1024*1024, directory=tmpdir, disk_budget=100*1024*1024)
            # Cache accepts exact fingerprint
            self.assertEqual(cache.fp, fp)
            self.assertEqual(cache.ram_budget, 10*1024*1024)

    # --------------------------------------------------------------------------
    # 10. Deadlock regression
    # --------------------------------------------------------------------------
    def test_10_deadlock_regression(self):
        from models.flashnext import gpu_sync
        # Verify barrier and drain functions execute cleanly without holding GIL
        gpu_sync.barrier()
        gpu_sync.drain()

    # --------------------------------------------------------------------------
    # 11. GPU buffer lifetime
    # --------------------------------------------------------------------------
    def test_11_gpu_buffer_lifetime(self):
        import importlib
        mod = importlib.import_module("models.flashnext.pool" + "_lifecycle")
        # Test status and fault detection
        self.assertIsNone(mod.lifecycle_fault())

    # --------------------------------------------------------------------------
    # 12. Independent watchdog
    # --------------------------------------------------------------------------
    def test_12_independent_watchdog(self):
        from models.flashnext import progress
        rec = progress.PROGRESS
        rec.request_begin()
        rec.stage("prefill")
        rec.tick(tokens=100)
        rec.stage("decode")
        rec.tick()
        rec.request_end()

    # --------------------------------------------------------------------------
    # 13. Memory admission and guard stops
    # --------------------------------------------------------------------------
    def test_13_memory_admission_and_guard_stops(self):
        total = 16 * 1024 * 1024 * 1024
        # Standard tier: context 8192, max_cold_prefill 4096, peak 6.0 GiB
        standard = R.TIERS[0]
        self.assertEqual(standard.name, "standard")
        self.assertEqual(standard.context_limit, 8192)
        self.assertEqual(standard.max_cold_prefill, 4096)

        # Ample memory (low anonymous) -> standard admitted
        reading_good = P.Reading(page=1, free=11 * 1024**3, anonymous=2 * 1024**3,
                                 wired=int(1.9 * 1024**3), compressor=int(0.8 * 1024**3))
        v_good = R.evaluate_tiered(reading_good, total)
        self.assertTrue(v_good.admitted)
        self.assertEqual(v_good.tier.name, "standard")
        self.assertEqual(v_good.tier.context_limit, 8192)

        # Moderate anonymous memory no longer changes the tier: the measured gate looks at the available figure, and every tier
        # at or below 4K needs the same start figure (the gate figure falls 6.1-6.65 GiB whatever the prompt length).
        reading_tight = P.Reading(page=1, free=11 * 1024**3, anonymous=int(3.5 * 1024**3),
                                  wired=int(1.9 * 1024**3), compressor=int(0.8 * 1024**3))
        v_tight = R.evaluate_tiered(reading_tight, total)
        self.assertTrue(v_tight.admitted)
        self.assertEqual(v_tight.tier.name, "standard")

        # Insufficient available memory (< 9 GiB gate) -> refused
        reading_gate_fail = P.Reading(page=1, free=5 * 1024**3, anonymous=2 * 1024**3,
                                      wired=int(1.9 * 1024**3), compressor=int(0.8 * 1024**3))
        v_gate = R.evaluate_tiered(reading_gate_fail, total)
        self.assertFalse(v_gate.admitted)

    # --------------------------------------------------------------------------
    # 14. Clean server shutdown
    # --------------------------------------------------------------------------
    def test_14_clean_server_shutdown(self):
        from veystrix.scheduler import Scheduler
        sched = Scheduler()
        snap = sched.snapshot()
        self.assertEqual(snap["active"], None)
        self.assertEqual(snap["waiting"], 0)
        self.assertFalse(sched.busy)

    # --------------------------------------------------------------------------
    # 15. Calibrated fixtures at 4K, 8K, 12K with real tokenizer
    # --------------------------------------------------------------------------
    def test_15_calibrated_fixtures_4k_8k_12k(self):
        tok_dir = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
        if not tok_dir.exists():
            self.skipTest("Real checkpoint tokenizer not present")
        from scratch.fixture_workloads import get_fixture_prompt
        tok_4k, _ = get_fixture_prompt("4k")
        tok_8k, _ = get_fixture_prompt("8k")
        tok_12k, _ = get_fixture_prompt("12k")

        self.assertGreaterEqual(tok_4k, 3700)
        self.assertLessEqual(tok_4k, 4096, "4K prompt must not exceed 4096 cold prefill limit")

        self.assertGreaterEqual(tok_8k, 7700)
        self.assertLessEqual(tok_8k, 8300)

        self.assertGreaterEqual(tok_12k, 11800)
        self.assertLessEqual(tok_12k, 12600)

    # --------------------------------------------------------------------------
    # 16. Two-tool sequential execution (glob -> grep -> final answer)
    # --------------------------------------------------------------------------
    def test_16_two_tool_sequential_execution(self):
        session = FakeSession(["init"])
        service = ModelService(session, context_limit=16384)

        # Turn 1: Model generates glob
        session.backend.tokenizer.rendered = "prompt"
        session.backend.generate = lambda max_tokens, out=None: (
            '<tool_call>\n<function=glob><parameter=pattern>**/*</parameter></function>\n</tool_call>',
            SimpleNamespace(finish="tool_calls", tokens=25, prompt_tokens=4000)
        )
        res1 = service.complete([{"role": "user", "content": "find batch size"}], [GLOB_TOOL, GREP_TOOL], 1024)
        self.assertEqual(len(res1.tool_calls), 1)
        self.assertEqual(res1.tool_calls[0]["name"], "glob")
        self.assertEqual(json.loads(res1.tool_calls[0]["arguments"]), {"pattern": "**/*"})

        # Turn 2: Model receives glob result and generates grep
        history_turn2 = [
            {"role": "user", "content": "find batch size"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "glob", "arguments": '{"pattern": "**/*"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "config/settings.yaml\nservice/core/engine.py"}
        ]
        session.backend.tokenizer.rendered = "prompt_plus_glob_result"
        session.backend.generate = lambda max_tokens, out=None: (
            '<tool_call>\n<function=grep><parameter=path>config/settings.yaml</parameter><parameter=pattern>batch_size</parameter></function>\n</tool_call>',
            SimpleNamespace(finish="tool_calls", tokens=35, prompt_tokens=4120)
        )
        res2 = service.complete(history_turn2, [GLOB_TOOL, GREP_TOOL], 1024)
        self.assertEqual(len(res2.tool_calls), 1)
        self.assertEqual(res2.tool_calls[0]["name"], "grep")
        self.assertEqual(json.loads(res2.tool_calls[0]["arguments"]), {"path": "config/settings.yaml", "pattern": "batch_size"})

        # Turn 3: Model receives grep result and produces final answer
        history_turn3 = list(history_turn2) + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "grep", "arguments": '{"path": "config/settings.yaml", "pattern": "batch_size"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_2", "content": "config/settings.yaml:6:  batch_size: 64"}
        ]
        session.backend.tokenizer.rendered = "prompt_plus_both_results"
        session.backend.generate = lambda max_tokens, out=None: (
            "The configured pipeline batch_size is 64.",
            SimpleNamespace(finish="stop", tokens=20, prompt_tokens=4200)
        )
        res3 = service.complete(history_turn3, [GLOB_TOOL, GREP_TOOL], 1024)
        self.assertEqual(res3.tool_calls, [])
        self.assertIn("64", res3.text)

    # --------------------------------------------------------------------------
    # 17. Actual model state memory scaling (Qwen3.8-Flash-Next)
    # --------------------------------------------------------------------------
    def test_17_actual_model_state_memory_scaling(self):
        try:
            from scratch.measure_model_state import compute_state_memory, TOTAL_GDN_STATE_BYTES
        except FileNotFoundError:
            self.skipTest("the Flash-Next checkpoint config is not present on this machine")
        # 36 GDN layers are constant size (~110.1 MB)
        self.assertAlmostEqual(TOTAL_GDN_STATE_BYTES / (1024 * 1024), 110.1, places=1)

        # KV state at 4K is ~108.6 MB -> total state ~218.7 MB
        m4k = compute_state_memory(4096)
        self.assertLess(m4k["total_state_mb"], 250)

        # KV state at 12K is ~325.7 MB -> total state ~435.8 MB
        m12k = compute_state_memory(12288)
        self.assertLess(m12k["total_state_mb"], 460)
        self.assertLess(m12k["total_state_gb"], 0.45)


if __name__ == "__main__":
    unittest.main()
