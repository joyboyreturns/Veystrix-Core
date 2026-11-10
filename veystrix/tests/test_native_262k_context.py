"""Tests for native 262,144-token context capacity and end-to-end token accounting."""
import unittest
from unittest.mock import MagicMock

from veystrix import context_budget, preferences
from veystrix.server import ModelService, ContextLimitError, model_entry, model_card
from veystrix.tests.test_server import FakeSession


class Native262KContextAccountingTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession([])
        self.backend = MagicMock()
        self.backend.pending = []
        self.backend.tape = []
        self.backend._replay_needed = False
        self.backend._prefix_session = None
        self.backend.encode = lambda text: [1] * len(text)
        self.backend.common_prefix = lambda ids: 0
        self.backend.model_path = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
        self.backend.generate = MagicMock(return_value=("output", MagicMock(tokens=10, finish="stop")))
        self.session.backend = self.backend
        self.session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False, effort="medium")

    def test_native_positional_capability_reported_as_262144(self):
        service = ModelService(self.session, context_limit=262144)
        card = model_card(service)
        entry = model_entry(card, service)
        self.assertEqual(entry["context"]["native_positional_capability"], 262144)
        self.assertEqual(entry["context"]["architectural_context_maximum"], 262144)
        self.assertEqual(entry["context_window"], 262144)
        self.assertTrue(entry["context_limit_enforced"])

    def test_prompt_within_262k_limit_accepted(self):
        service = ModelService(self.session, context_limit=262144)
        # 200,000 token prompt + 256 reserve <= 262144
        prompt_ids = [1] * 200000
        service._check_context(len(prompt_ids))
        self.assertEqual(service.last_prompt_tokens, 200000)

    def test_prompt_exceeding_262k_refused_with_explicit_guidance(self):
        service = ModelService(self.session, context_limit=262144)
        # 261,900 tokens + 256 reserve = 262,156 > 262,144 -> refuses
        prompt_ids = [1] * 261900
        with self.assertRaises(ContextLimitError) as cm:
            service._check_context(len(prompt_ids))
        err_msg = str(cm.exception)
        self.assertIn("context limit exceeded", err_msg)
        self.assertIn("262144", err_msg)
        self.assertIn("Nothing was truncated", err_msg)

    def test_clamping_oversized_requested_output_without_error(self):
        service = ModelService(self.session, context_limit=262144)
        # Mock tape and common_prefix to simulate an accumulated conversation (e.g. 258,000 warm tokens)
        self.backend.tape = [1] * 258000
        self.backend.common_prefix = lambda ids: 258000
        # Mock encode to produce a 260,000-token prompt (2,000 incremental tokens <= 16384 limit)
        self.backend.encode = lambda text: [1] * 260000
        # Client requests 10,000 tokens of output (260,000 + 10,000 = 270,000 > 262,144)
        # Complete should clamp max_tokens to 2,144 (262,144 - 260,000) instead of HTTP 400
        messages = [{"role": "user", "content": "test"}]
        service.complete(messages, [], max_tokens=10000)

        # Verify backend.generate was called with clamped max_tokens = 2144
        call_kwargs = self.backend.generate.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 2144)
        self.assertEqual(service.last_budget["reserved_output"], 2144)
        self.assertEqual(service.last_budget["remaining"], 0)

    def test_simulated_context_scaling_across_tiers(self):
        tiers = [16384, 32768, 65536, 131072, 262144]
        for limit in tiers:
            service = ModelService(self.session, context_limit=limit)
            half = limit // 2
            budget = context_budget.evaluate(limit, half, 4096)
            self.assertEqual(budget.server_window, limit)
            self.assertEqual(budget.input_tokens, half)
            self.assertEqual(budget.reserved_output, 4096)
            self.assertEqual(budget.remaining, half - 4096)
            self.assertTrue(budget.continuation_ok)


if __name__ == "__main__":
    unittest.main()
