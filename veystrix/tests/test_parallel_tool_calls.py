"""The exact raw model outputs of Gate A attempt 2 (real checkpoint, 2026-11-08) through the server's tool-call parser and
its streamed Chat Completions shape. The model emitted two ``glob`` calls in one reply, then repeated globs and never
called ``grep``. These tests pin that the protocol layer handled all of it correctly (so the behaviour is the model's, not
the parser's): both calls are returned, in order, with distinct ids and arguments that match the declared schema."""
from __future__ import annotations

import json
import unittest

from veystrix.server import _parse_tool_calls
from veystrix.tests.test_server import _make_handler, _stub_result
from veystrix.tests.test_server_harness_protocol import GLOB

GREP = {"type": "function", "function": {
    "name": "grep", "description": "search file contents",
    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                   "required": ["pattern"]}}}
TOOLS = [GLOB, GREP]

ATTEMPT2_STEP1 = ("I'll start by discovering the configuration files under `services/<name>/config/`.\n\n<tool_call>\n"
                  "<function=glob>\n<parameter=pattern>\nservices/**/config/*\n</parameter>\n</function>\n</tool_call>\n"
                  "<tool_call>\n<function=glob>\n<parameter=pattern>\nservices/**/*.yaml\n</parameter>\n</function>\n</tool_call>")
ATTEMPT2_STEP2 = ("<tool_call>\n<function=glob>\n<parameter=pattern>\nservices/**/*\n</parameter>\n</function>\n</tool_call>\n"
                  "<tool_call>\n<function=glob>\n<parameter=pattern>\n**/*.{yml,yaml,toml,ini,json}\n</parameter>\n"
                  "</function>\n</tool_call>")
GREP_STEP = ("<tool_call>\n<function=grep>\n<parameter=pattern>\nbatch_size|token_expiry\n</parameter>\n"
             "<parameter=path>\nservices/gateway/config/gateway.yaml\n</parameter>\n</function>\n</tool_call>")


class ParallelToolCallTests(unittest.TestCase):
    def test_both_globs_of_the_first_reply_are_parsed_in_order_with_distinct_ids(self):
        visible, calls = _parse_tool_calls(ATTEMPT2_STEP1, TOOLS)
        self.assertEqual([c["name"] for c in calls], ["glob", "glob"])
        self.assertEqual([json.loads(c["arguments"]) for c in calls],
                         [{"pattern": "services/**/config/*"}, {"pattern": "services/**/*.yaml"}])
        self.assertEqual(len({c["id"] for c in calls}), 2)
        self.assertTrue(all(c["id"].startswith("call_") for c in calls))
        self.assertIn("discovering the configuration files", visible)
        self.assertNotIn("<tool_call>", visible)

    def test_braces_and_commas_in_a_glob_pattern_survive(self):
        _, calls = _parse_tool_calls(ATTEMPT2_STEP2, TOOLS)
        self.assertEqual(json.loads(calls[1]["arguments"]), {"pattern": "**/*.{yml,yaml,toml,ini,json}"})

    def test_a_grep_call_with_a_path_is_parsed_against_its_schema(self):
        _, calls = _parse_tool_calls(GREP_STEP, TOOLS)
        self.assertEqual(calls[0]["name"], "grep")
        self.assertEqual(json.loads(calls[0]["arguments"]),
                         {"pattern": "batch_size|token_expiry", "path": "services/gateway/config/gateway.yaml"})

    def test_parallel_calls_stream_as_two_indexed_tool_call_deltas_and_one_tool_calls_finish(self):
        _, calls = _parse_tool_calls(ATTEMPT2_STEP1, TOOLS)
        result = _stub_result("stop", calls, text="")
        handler = _make_handler(lambda *a, **k: result)
        handler._chat({"model": "veystrix-flashnext", "stream": True, "tools": TOOLS, "max_completion_tokens": 1024,
                       "messages": [{"role": "user", "content": "find the config"}]})
        body = handler.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: {")]
        deltas = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                  if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
        self.assertEqual([d["index"] for d in deltas], [0, 1])
        self.assertEqual([d["id"] for d in deltas], [c["id"] for c in calls])
        self.assertEqual([json.loads(d["function"]["arguments"]) for d in deltas],
                         [{"pattern": "services/**/config/*"}, {"pattern": "services/**/*.yaml"}])
        finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(finish, ["tool_calls"])
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))


if __name__ == "__main__":
    unittest.main()
