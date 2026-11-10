"""Wire-level checks for the DeepSeek Harness (pi-ai ``openai-completions``) request shape.

The request bodies below are the shapes captured from the real ``dsh headless``
client (v0.2.2-rc.1): streamed Chat Completions with ``stream_options.include_usage``,
``store: false``, ``max_completion_tokens``, assistant ``content: null`` with
``tool_calls`` whose ``arguments`` is a JSON *string*, and ``role: tool`` results.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from veystrix.server import (
    RequestError,
    _normalize_messages,
    _parse_tool_calls,
)
from veystrix.tests.test_server import _make_handler, _stub_result

GLOB = {"type": "function", "function": {
    "name": "glob", "description": "find files",
    "parameters": {"type": "object", "properties": {
        "pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"]}}}
READ = {"type": "function", "function": {
    "name": "read", "description": "read a file",
    "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string"}, "offset": {"type": "number"},
        "limit": {"type": "number"}, "mode": {"type": "string", "enum": ["text", "hex"]}},
        "required": ["file_path"]}}}

HARNESS_HISTORY = [
    {"role": "system", "content": "You are an AI agent."},
    {"role": "user", "content": "list the files in the current directory"},
    {"role": "user", "content": "Current runtime context. ..."},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_abc123", "type": "function",
         "function": {"name": "glob", "arguments": "{\"pattern\":\"*\"}"}}]},
    {"role": "tool", "content": "alpha.txt\nbeta.md", "tool_call_id": "call_abc123"},
]


class HistoryTests(unittest.TestCase):
    def test_string_arguments_become_the_mapping_the_template_iterates(self):
        messages = _normalize_messages(HARNESS_HISTORY)
        call = messages[3]["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], {"pattern": "*"})
        self.assertEqual(call["id"], "call_abc123")
        self.assertEqual(messages[3]["content"], "")
        self.assertEqual(messages[4]["tool_call_id"], "call_abc123")

    def test_empty_arguments_are_an_empty_mapping(self):
        history = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "glob", "arguments": ""}}]}]
        self.assertEqual(
            _normalize_messages(history)[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_bad_arguments_are_errors_not_repairs(self):
        for arguments in ("{not json", "[1, 2]", "\"text\"", "3"):
            with self.subTest(arguments=arguments):
                history = [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "type": "function",
                     "function": {"name": "glob", "arguments": arguments}}]}]
                with self.assertRaises(RequestError):
                    _normalize_messages(history)

    def test_unsupported_message_content_and_fields_are_named(self):
        cases = [
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            [{"role": "user", "content": "hi", "weird": 1}],
            [{"role": "user", "content": "hi", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "n", "arguments": "{}"}}]}],
            [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "n", "arguments": "{}"}, "extra": 1}]}],
        ]
        for history in cases:
            with self.subTest(history=history):
                with self.assertRaises(RequestError):
                    _normalize_messages(history)

    def test_harmless_empty_fields_and_tool_name_are_accepted(self):
        history = [
            {"role": "user", "content": "hi", "name": "bob", "refusal": None},
            {"role": "tool", "content": "x", "tool_call_id": "c", "name": "glob"},
            {"role": "developer", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        ]
        result = _normalize_messages(history)
        self.assertEqual(result[2], {"role": "system", "content": "ab"})

    def test_reasoning_content_reaches_the_template(self):
        history = [{"role": "assistant", "content": "x", "reasoning_content": "thought"}]
        self.assertEqual(_normalize_messages(history)[0]["reasoning_content"], "thought")

    def test_hermes_assistant_reasoning_is_preserved(self):
        for message in (
            {"role": "assistant", "content": "answer", "reasoning": "thought"},
            {"role": "assistant", "content": "answer", "reasoning": "thought", "reasoning_content": "thought"},
            {"role": "assistant", "content": "answer", "reasoning": "thought", "reasoning_content": ""},
        ):
            self.assertEqual(_normalize_messages([message])[0]["reasoning_content"], "thought")

    def test_hermes_reasoning_rejects_invalid_or_conflicting_values(self):
        for message in (
            {"role": "user", "content": "x", "reasoning": "thought"},
            {"role": "assistant", "content": "x", "reasoning": {"text": "thought"}},
            {"role": "assistant", "content": "x", "reasoning": "a", "reasoning_content": "b"},
        ):
            with self.assertRaises(RequestError):
                _normalize_messages([message])

    @unittest.skipUnless(
        Path(os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP/chat_template.jinja")).exists(),
        "checkpoint chat template not present")
    def test_real_qwen_template_renders_the_harness_history(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        path = os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP/chat_template.jinja")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
        env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)

        def raise_exception(message):
            raise RuntimeError(message)

        template = env.from_string(Path(path).read_text())
        text = template.render(
            messages=_normalize_messages(HARNESS_HISTORY), tools=[GLOB],
            add_generation_prompt=True, enable_thinking=False,
            raise_exception=raise_exception)
        self.assertIn("<function=glob>\n<parameter=pattern>\n*\n</parameter>\n</function>", text)
        self.assertIn("<tool_response>\nalpha.txt\nbeta.md\n</tool_response>", text)


class ToolCallSchemaTests(unittest.TestCase):
    @staticmethod
    def xml(name, **params):
        body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params.items())
        return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"

    def test_a_valid_call_keeps_its_arguments(self):
        _, calls = _parse_tool_calls(self.xml("glob", pattern="*.py"), [GLOB, READ])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"pattern": "*.py"})

    def test_numbers_are_numbers(self):
        _, calls = _parse_tool_calls(self.xml("read", file_path="a", offset="3"), [READ])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"file_path": "a", "offset": 3.0})

    def test_violations_are_reported_by_name(self):
        cases = {
            "undeclared": self.xml("bash", command="ls"),
            "no parameter": self.xml("glob", pattern="*", recursive="yes"),
            "missing required": self.xml("glob", path="."),
            "enum": self.xml("read", file_path="a", mode="binary"),
        }
        for label, text in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(RequestError):
                    _parse_tool_calls(text, [GLOB, READ])

    def test_json_form_is_validated_too(self):
        text = '<tool_call>{"name": "glob", "arguments": {"pattern": "*", "bogus": 1}}</tool_call>'
        with self.assertRaises(RequestError):
            _parse_tool_calls(text, [GLOB])

    def test_without_declared_tools_nothing_is_validated(self):
        _, calls = _parse_tool_calls(self.xml("anything", a="1"), None)
        self.assertEqual(calls[0]["name"], "anything")


class HarnessPatchTests(unittest.TestCase):
    """The harness requests max_completion_tokens = contextWindow - about 4,830 (its fixed prompt estimate plus headroom,
    measured with the real harness), floored at 1, unless the model entry sets maxTokens. A 4096 window gave 1 token and the
    first real round trip could not finish a tool call."""

    HARNESS_RESERVE = 4830

    def values(self):
        import re
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "docs" / "flashnext" / "dsh" / "veystrix-agentic.patch.yml").read_text()
        window = int(re.search(r"^\s*contextWindow:\s*(\d+)", text, re.M)[1])
        found = re.search(r"^\s*maxTokens:\s*(\d+)", text, re.M)
        return window, int(found[1]) if found else None

    def test_the_shipped_patch_leaves_the_harness_room_to_ask_for_a_real_answer(self):
        window, max_tokens = self.values()
        self.assertIsNotNone(max_tokens, "the patch must cap the requested output with maxTokens")
        self.assertGreaterEqual(max_tokens, 512)
        self.assertGreaterEqual(window - self.HARNESS_RESERVE, max_tokens, "the window must leave room for maxTokens")
        self.assertLessEqual(max_tokens, 2048 - 256)          # never more than the smallest enforced context can hold

    def test_the_real_harness_wire_value_is_what_the_patch_implies(self):
        window, max_tokens = self.values()
        self.assertEqual(min(max_tokens, max(1, window - self.HARNESS_RESERVE)), max_tokens)
        self.assertEqual(max(1, 4096 - self.HARNESS_RESERVE), 1)      # the configuration that failed


class ChatRouteTests(unittest.TestCase):
    BASE = {"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}],
            "stream": True, "store": False, "max_completion_tokens": 64}

    def run_chat(self, payload, result=None):
        seen = {}

        def complete(messages, tools, max_tokens, on_text=None, gone=None):
            seen.update(messages=messages, tools=tools, max_tokens=max_tokens)
            return result or _stub_result("stop", [])

        handler = _make_handler(complete)
        headers = []
        handler.send_header = lambda name, value: headers.append((name, value))
        handler._chat(payload)
        return handler, seen, headers

    def chunks(self, handler):
        body = handler.wfile.getvalue().decode()
        return [json.loads(line[6:]) for line in body.split("\n\n")
                if line.startswith("data: {")], body

    def test_hermes_reasoning_history_reaches_completion(self):
        payload = dict(self.BASE, reasoning_effort="medium", messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning": "thought"},
            {"role": "user", "content": "continue"},
        ])
        handler, seen, _ = self.run_chat(payload)
        self.assertEqual(seen["messages"][1]["reasoning_content"], "thought")
        self.assertIn("data: [DONE]", handler.wfile.getvalue().decode())

    def test_usage_chunk_only_when_requested(self):
        handler, _, _ = self.run_chat(dict(self.BASE, stream_options={"include_usage": True}))
        chunks, body = self.chunks(handler)
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], {
            "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 0}, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 7})
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))
        handler, _, _ = self.run_chat(dict(self.BASE))
        chunks, _ = self.chunks(handler)
        self.assertTrue(all("usage" not in chunk for chunk in chunks))

    def test_tool_call_stream_shape(self):
        result = _stub_result("stop", [{"id": "call_1", "name": "glob", "arguments": '{"pattern": "*"}'}], text="")
        handler, _, _ = self.run_chat(dict(self.BASE, tools=[GLOB]), result)
        chunks, _ = self.chunks(handler)
        calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks
                 if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
        self.assertEqual(calls, [{"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "glob", "arguments": '{"pattern": "*"}'}}])
        finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(finish, ["tool_calls"])

    def test_first_model_delta_is_recorded_only_after_a_nonempty_sse_chunk(self):
        handler = _make_handler(lambda *a, **k: _stub_result("stop", []))
        handler._sse_start()
        with mock.patch("veystrix.server.flight_recorder.record") as record:
            handler._sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
            record.assert_not_called()
            handler._sse({"choices": [{"delta": {"content": "OK"}}]})
            handler._sse({"choices": [{"delta": {"content": "again"}}]})
            record.assert_called_once_with("first_model_delta_forwarded", transport="sse")

    def test_harness_fields_are_consumed(self):
        handler, seen, _ = self.run_chat(dict(self.BASE, stream_options={"include_usage": True}))
        self.assertEqual(seen["max_tokens"], 64)
        self.assertEqual(handler.status_calls, [200])

    def test_unsupported_fields_are_refused_before_any_stream(self):
        for field, value in (("stop", ["x"]), ("n", 2), ("response_format", {"type": "json_object"}),
                             ("logprobs", True), ("tool_choice", "required"), ("functions", [])):
            with self.subTest(field=field):
                handler = _make_handler(lambda *a, **k: _stub_result("stop", []))
                with self.assertRaises(RequestError):
                    handler._chat(dict(self.BASE, **{field: value}))
                self.assertEqual(handler.status_calls, [])
                self.assertEqual(handler.wfile.getvalue(), b"")

    def test_sampling_fields_are_listed_not_swallowed(self):
        _, _, headers = self.run_chat(dict(self.BASE, temperature=0.7, top_p=0.9))
        self.assertIn(("X-Veystrix-Ignored-Fields", "temperature,top_p"), headers)
        _, _, headers = self.run_chat(dict(self.BASE))
        self.assertFalse([name for name, _ in headers if name == "X-Veystrix-Ignored-Fields"])

    def test_unknown_stream_option_is_refused(self):
        handler = _make_handler(lambda *a, **k: _stub_result("stop", []))
        with self.assertRaises(RequestError):
            handler._chat(dict(self.BASE, stream_options={"include_usage": True, "x": 1}))

    def test_a_client_that_disconnects_mid_stream_ends_the_request_quietly(self):
        from veystrix.server import ModelService
        from veystrix.tests.test_server import FakeSession

        class Gone(io.BytesIO):
            def write(self, data):
                if b"hello" in data:
                    raise BrokenPipeError
                return super().write(data)

        session = FakeSession(["hello ", "world"])
        service = ModelService(session)
        handler = _make_handler(service.complete)
        handler.server.service = service
        service.model = "veystrix-test"
        handler.wfile = Gone()
        handler._chat(dict(self.BASE))
        self.assertEqual(handler.status_calls, [200])
        self.assertEqual(handler.error_calls, [])
        # the half-finished generation must not survive as reusable state
        self.assertGreaterEqual(session.backend.reset_count, 1)
        # and the next request is served normally
        handler = _make_handler(service.complete)
        handler.server.service = service
        handler._chat(dict(self.BASE))
        self.assertIn("data: [DONE]", handler.wfile.getvalue().decode())


class RequestLogTests(unittest.TestCase):
    def test_log_is_off_by_default_and_writes_jsonl_when_asked(self):
        import tempfile
        from unittest import mock
        from veystrix.server import ModelService
        from veystrix.tests.test_server import FakeSession

        service = ModelService(FakeSession(["hello"]))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VEYSTRIX_REQUEST_LOG", None)
            service.complete([{"role": "user", "content": "hi"}], [], 8, lambda text: None)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "requests.jsonl")
            with mock.patch.dict(os.environ, {"VEYSTRIX_REQUEST_LOG": path}):
                service.complete([{"role": "user", "content": "hi"}], [], 8, lambda text: None)
            row = json.loads(open(path).read())
        self.assertEqual(row["prompt_tokens"], 4)
        self.assertEqual(row["finish"], "stop")
        self.assertIsNotNone(row["ttft_seconds"])
        self.assertEqual(row["tool_calls"], 0)


if __name__ == "__main__":
    unittest.main()
