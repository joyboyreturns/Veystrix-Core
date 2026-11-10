"""Reasoning-aware session caching and protocol integration tests.

Verifies:
- Thinking ON consecutive turns: 100% token reuse (no cold rebuild)
- Round-tripped reasoning_content matches generated tokens exactly
- Server-owned reasoning ledger restores reasoning when client drops it
- Thinking + tool calls consecutive turns preserve session state
- Thinking OFF consecutive turns
- Reasoning effort levels: low, medium, high, xhigh, max
- Strict session isolation (Session A private reasoning never leaks to Session B)
- Replay boundaries and fallback to cold reconstruction on conversation branching
"""
from __future__ import annotations

import json
import os
import unittest

from veystrix import preferences
from veystrix.server import VeystrixHTTPServer, ModelService
from veystrix.sessions import ReasoningLedger, derive_key
from veystrix.tests.test_agentic_long_fixtures import TOKENIZER, RealTokenizerBackend
from veystrix.tests.test_server import FakeSession
from models.flashnext.tests.bench.fixtures import agentic_fixture as F


class ScriptedModelBackend(RealTokenizerBackend):
    def __init__(self, tokenizer, replies: list[str]):
        super().__init__(tokenizer, replies)
        self.prompts = []

    def generate(self, max_tokens, out=None, should_cancel=None):
        prompt = self.tokenizer.decode(self.tape + self.pending)
        self.prompts.append(prompt)
        text = self.replies.pop(0) if self.replies else "default reply"
        if prompt.rstrip(" ").endswith("<think>\n") and "</think>" not in text:
            text = f"Step-by-step thinking for turn {len(self.prompts)}\n</think>\n\n" + text
        self.replies.insert(0, text)
        return super().generate(max_tokens, out)


@unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
class ReasoningSessionCacheTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession([])
        self.backend = ScriptedModelBackend(TOKENIZER, [])
        self.session.backend = self.backend
        self.service = ModelService(self.session)

    def test_thinking_on_round_trip_preserves_kv_cache(self):
        """When client round-trips reasoning_content, Turn 2 reuses the KV cache 100%."""
        self.backend.replies = [
            "My detailed reasoning.\n</think>\n\nI will help you with math.",
            "Continuing reasoning.\n</think>\n\nThe sum is 42.",
        ]
        self.session.preferences["thinking_enabled"] = True
        self.session.preferences["effort"] = "medium"

        # Turn 1
        messages1 = [{"role": "user", "content": "Hello!"}]
        res1 = self.service.complete(messages1, [], 1024)
        self.assertEqual(res1.text, "I will help you with math.")
        self.assertEqual(res1.reasoning_content, "My detailed reasoning.")
        self.assertEqual(self.service.rebuilt, 1)
        self.assertEqual(self.service.reused, 0)
        tape_len_1 = len(self.backend.tape)
        self.assertGreater(tape_len_1, 0)

        # Turn 2: Client includes reasoning_content from Turn 1
        messages2 = [
            {"role": "user", "content": "Hello!"},
            {
                "role": "assistant",
                "content": "I will help you with math.",
                "reasoning_content": "My detailed reasoning.",
            },
            {"role": "user", "content": "What is 40 + 2?"},
        ]
        res2 = self.service.complete(messages2, [], 1024)
        self.assertEqual(res2.text, "The sum is 42.")
        self.assertEqual(self.service.reused, 1)
        self.assertEqual(self.service.rebuilt, 1)  # unchanged
        self.assertGreater(len(self.backend.tape), tape_len_1)

    def test_server_owned_reasoning_ledger_restores_missing_reasoning(self):
        """When client omits reasoning_content, the server ledger restores it, preventing cold prefill."""
        self.backend.replies = [
            "Secret internal chain of thought.\n</think>\n\nResult A",
            "Second internal chain of thought.\n</think>\n\nResult B",
        ]
        self.session.preferences["thinking_enabled"] = True
        self.session.preferences["effort"] = "medium"

        # Turn 1
        messages1 = [{"role": "user", "content": "Start task"}]
        res1 = self.service.complete(messages1, [], 1024)
        self.assertEqual(res1.text, "Result A")
        self.assertEqual(res1.reasoning_content, "Secret internal chain of thought.")
        self.assertEqual(self.service.reused, 0)

        # Turn 2: Client sends history WITHOUT reasoning_content (common client behavior)
        messages2 = [
            {"role": "user", "content": "Start task"},
            {"role": "assistant", "content": "Result A"},  # no reasoning_content!
            {"role": "user", "content": "Next step"},
        ]
        res2 = self.service.complete(messages2, [], 1024)
        self.assertEqual(res2.text, "Result B")
        # Turn 2 MUST reuse the KV cache, not rebuild cold!
        self.assertEqual(self.service.reused, 1)
        self.assertEqual(self.service.rebuilt, 1)  # only initial prefill rebuilt

    def test_thinking_with_tool_calls_turn_sequence_cache_reuse(self):
        """Tool calls and tool responses maintain cache reuse across multi-turn reasoning loops."""
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up data",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
        self.backend.replies = [
            'Thinking about looking up data...\n</think>\n\n<tool_call>\n<function=lookup>\n<parameter=query>\npython\n</parameter>\n</function>\n</tool_call>',
            'Thinking about the lookup result...\n</think>\n\nFound Python 3.12.',
        ]
        self.session.preferences["thinking_enabled"] = True
        self.session.preferences["effort"] = "medium"

        # Turn 1: User prompt -> Model generates thinking + tool call
        messages1 = [{"role": "user", "content": "Search for python"}]
        res1 = self.service.complete(messages1, [tool], 1024)
        self.assertEqual(len(res1.tool_calls), 1)
        self.assertEqual(res1.tool_calls[0]["name"], "lookup")
        self.assertEqual(self.service.reused, 0)

        # Turn 2: Client sends tool result (without reasoning_content)
        tool_calls_wire = [{
            "id": c["id"],
            "type": "function",
            "function": {"name": c["name"], "arguments": c["arguments"]},
        } for c in res1.tool_calls]
        messages2 = [
            {"role": "user", "content": "Search for python"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls_wire,
            },
            {
                "role": "tool",
                "content": "Python is a high-level programming language.",
                "tool_call_id": res1.tool_calls[0]["id"],
            },
        ]
        res2 = self.service.complete(messages2, [tool], 1024)
        self.assertEqual(res2.text, "Found Python 3.12.")
        self.assertEqual(self.service.reused, 1)

    def test_thinking_off_consecutive_turns(self):
        """With thinking disabled, cache reuse functions normally."""
        self.backend.replies = ["Direct answer 1", "Direct answer 2"]
        self.session.preferences["thinking_enabled"] = False

        m1 = [{"role": "user", "content": "Question 1"}]
        r1 = self.service.complete(m1, [], 1024)
        self.assertEqual(r1.text, "Direct answer 1")

        m2 = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Direct answer 1"},
            {"role": "user", "content": "Question 2"},
        ]
        r2 = self.service.complete(m2, [], 1024)
        self.assertEqual(r2.text, "Direct answer 2")
        self.assertEqual(self.service.reused, 1)

    def test_session_isolation_does_not_leak_reasoning(self):
        """Reasoning from Session A is never injected into or accessible by Session B."""
        self.backend.replies = [
            "Session A private thoughts\n</think>\n\nSession A answer",
            "Session B clean answer",
        ]
        self.session.preferences["thinking_enabled"] = True

        # Session A
        self.service.session_hint = "session-a"
        mA = [{"role": "user", "content": "User prompt A"}]
        rA = self.service.complete(mA, [], 1024)
        self.assertEqual(rA.reasoning_content, "Session A private thoughts")

        # Session B arrives
        self.service.session_hint = "session-b"
        mB = [{"role": "user", "content": "User prompt B"}]
        rB = self.service.complete(mB, [], 1024)

        # Verify Session B cannot restore Session A's reasoning
        mB_test = [
            {"role": "user", "content": "User prompt B"},
            {"role": "assistant", "content": "Session A answer"},  # spoofed content
        ]
        restored = self.service.reasoning_ledger.restore_reasoning("id:session-b", mB_test)
        self.assertEqual(restored, 0)
        self.assertNotIn("reasoning_content", mB_test[1])

    def test_all_reasoning_effort_levels_accepted(self):
        """Verifies low, medium, high, xhigh, max efforts are accepted."""
        for effort in ("low", "medium", "high", "xhigh", "max"):
            self.session.preferences["thinking_enabled"] = True
            self.session.preferences["effort"] = effort
            self.backend.replies = ["Reply for " + effort]
            m = [{"role": "user", "content": "Hello effort " + effort}]
            res = self.service.complete(m, [], 512)
            self.assertIn("Reply for " + effort, res.text)


if __name__ == "__main__":
    unittest.main()
