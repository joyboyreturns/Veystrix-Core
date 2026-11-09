"""The usage every API returns says how much of the prompt came from cache.

DeepSeek Harness (pi-ai) shows "Cache hit" from ``prompt_tokens_details.cached_tokens`` / ``prompt_cache_hit_tokens`` and
counts uncached input as ``prompt_tokens - cached``. The engine's ``stats.prompt_tokens`` is only what this request prefilled,
so the full prompt length (``last_prompt_tokens``) is what ``prompt_tokens`` must carry.
"""
import json
import unittest
from types import SimpleNamespace

from macqwen import server
from macqwen.tests.test_server import _make_handler, _stub_result


def stats(prefilled, completion=5):
    return SimpleNamespace(finish="stop", tokens=completion, prompt_tokens=prefilled)


class UsageHelperTests(unittest.TestCase):
    def test_cached_is_prompt_minus_prefilled(self):
        service = SimpleNamespace(last_prompt_tokens=9569)
        self.assertEqual(server.usage_counts(service, stats(110, 7)), {"prompt": 9569, "cached": 9459, "completion": 7})

    def test_snapshot_restore_counts_the_matched_prefix_as_cached(self):
        # Real DSH run, 2026-11-09: a request restored from an SSD snapshot (matched 5,392 of 5,602 tokens) reported
        # stats.prompt_tokens == 5,602 and so "0 cached" until the adopted prefix was taken into account.
        service = SimpleNamespace(last_prompt_tokens=5602, last_cached_tokens=5392)
        self.assertEqual(server.usage_counts(service, stats(5602)), {"prompt": 5602, "cached": 5392, "completion": 5})

    def test_matched_prefix_never_exceeds_the_prompt(self):
        service = SimpleNamespace(last_prompt_tokens=100, last_cached_tokens=500)
        self.assertEqual(server.usage_counts(service, stats(10))["cached"], 100)

    def test_unknown_prompt_length_reports_no_cache(self):
        self.assertEqual(server.usage_counts(SimpleNamespace(), stats(7))["cached"], 0)
        self.assertEqual(server.usage_counts(SimpleNamespace(last_prompt_tokens=None), stats(7))["prompt"], 7)

    def test_prompt_never_below_prefilled(self):
        self.assertEqual(server.usage_counts(SimpleNamespace(last_prompt_tokens=3), stats(7)), {"prompt": 7, "cached": 0, "completion": 5})

    def test_chat_usage_has_every_field_clients_read(self):
        usage = server.chat_usage(SimpleNamespace(last_prompt_tokens=1000), stats(40, 10))
        self.assertEqual(usage, {
            "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
            "prompt_tokens_details": {"cached_tokens": 960},
            "prompt_cache_hit_tokens": 960, "prompt_cache_miss_tokens": 40})

    def test_anthropic_input_excludes_cache_reads(self):
        usage = server.anthropic_usage(SimpleNamespace(last_prompt_tokens=1000), stats(40, 10))
        self.assertEqual(usage, {"input_tokens": 40, "cache_read_input_tokens": 960,
                                 "cache_creation_input_tokens": 0, "output_tokens": 10})

    def test_responses_usage_carries_cached_tokens(self):
        usage = server.responses_usage(SimpleNamespace(last_prompt_tokens=1000), stats(40, 10))
        self.assertEqual(usage, {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 960},
                                 "output_tokens": 10, "total_tokens": 1010})


class UsageOverHttpShapeTests(unittest.TestCase):
    def handler(self, prompt_tokens=1000, prefilled=40):
        result = SimpleNamespace(text="hi", tool_calls=[], stats=stats(prefilled))
        handler = _make_handler(lambda *a, **k: result)
        handler.server.service.last_prompt_tokens = prompt_tokens
        handler.server.service.reasoning_summary = None
        return handler

    def test_streamed_chat_usage_chunk_reports_cache(self):
        handler = self.handler()
        handler._chat({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True,
                       "stream_options": {"include_usage": True}, "max_completion_tokens": 64})
        body = handler.wfile.getvalue().decode()
        chunks = [json.loads(line[6:]) for line in body.split("\n\n") if line.startswith("data: {")]
        usage = chunks[-1]["usage"]
        self.assertEqual(usage["prompt_tokens"], 1000)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 960)
        self.assertEqual(usage["prompt_cache_hit_tokens"], 960)

    def test_plain_chat_usage_reports_cache(self):
        handler = self.handler()
        handler._chat({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 64})
        usage = handler.json_calls[-1][0]["usage"]
        self.assertEqual((usage["prompt_tokens"], usage["prompt_tokens_details"]["cached_tokens"]), (1000, 960))

    def test_responses_usage_reports_cache(self):
        handler = self.handler()
        handler._responses({"model": "m", "input": "hi", "max_output_tokens": 64})
        usage = handler.json_calls[-1][0]["usage"]
        self.assertEqual((usage["input_tokens"], usage["input_tokens_details"]["cached_tokens"]), (1000, 960))

    def test_anthropic_usage_reports_cache(self):
        handler = self.handler()
        handler._anthropic({"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
        usage = handler.json_calls[-1][0]["usage"]
        self.assertEqual((usage["input_tokens"], usage["cache_read_input_tokens"]), (40, 960))


if __name__ == "__main__":
    unittest.main()
