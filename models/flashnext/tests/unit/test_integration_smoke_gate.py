import io
import json
import unittest
from unittest import mock

from models.flashnext.tests.bench import bench_integration_smoke as smoke


class Response(io.BytesIO):
    status = 200


class SmokeGateTests(unittest.TestCase):
    def report(self):
        return {
            "tool_turn": {"status": 200, "finish_reason": "tool_calls", "tool_calls": 1,
                          "first_call": {"name": "read_file", "arguments": '{"path":"models/flashnext/layer_major.py"}'}},
            "follow_up": {"status": 200, "done": True, "events": 2, "errors": [], "text": "summary",
                          "finish_reasons": ["stop"]},
            "cancel": {"ready": True}, "after_cancel": {"status": 200},
            "anthropic": {"status": 200, "content": []},
            "responses": {"status": 200, "object": "response", "response_status": "completed", "output": []},
        }

    def test_no_tool_call_cannot_pass(self):
        report = self.report()
        self.assertEqual(smoke.verdict(report), [])
        report["tool_turn"]["tool_calls"] = 0
        self.assertTrue(smoke.verdict(report))

    def test_stream_error_cannot_pass_even_if_done(self):
        report = self.report()
        report["follow_up"]["errors"] = ["memory safety stop"]
        self.assertTrue(smoke.verdict(report))

    def test_invalid_tool_arguments_and_protocol_envelopes_fail(self):
        report = self.report()
        report["tool_turn"]["first_call"]["arguments"] = "not json"
        report["anthropic"]["content"] = None
        report["responses"]["response_status"] = "failed"
        self.assertEqual(len(smoke.verdict(report)), 3)

    def test_stream_parser_detects_error_and_finish(self):
        lines = [
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"error": {"code": "memory_safety_stop"}},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        payload = b"".join(b"data: " + json.dumps(line).encode() + b"\n\n" for line in lines) + b"data: [DONE]\n\n"
        with mock.patch.object(smoke.urllib.request, "urlopen", return_value=Response(payload)):
            result = smoke.stream("http://localhost", "/v1/chat/completions", {})
        self.assertEqual(result["text"], "ok")
        self.assertEqual(result["finish_reasons"], ["stop"])
        self.assertTrue(result["done"])
        self.assertIn("memory_safety_stop", result["errors"][0])

    def test_stream_parser_rejects_malformed_event(self):
        with mock.patch.object(smoke.urllib.request, "urlopen", return_value=Response(b"data: {bad\n\ndata: [DONE]\n\n")):
            result = smoke.stream("http://localhost", "/v1/chat/completions", {})
        self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
