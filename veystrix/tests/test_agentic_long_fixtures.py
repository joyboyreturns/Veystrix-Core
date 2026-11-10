"""Long agentic conversations (4K, 8K, 12K real tokens) through the real server logic, scripted model.

SCOPE, stated plainly: the model is SCRIPTED. These tests prove protocol, tokenization with the real tokenizer and chat
template, live-state reuse across tool turns, session isolation, context accounting and error text. They prove nothing
about a real model's tool intelligence or about memory on the real checkpoint.
"""
import json
import os
import unittest
from types import SimpleNamespace

from veystrix import preferences
from veystrix.server import ContextLimitError, ModelService
from veystrix.tests.test_server import FakeBackend, FakeSession
from models.flashnext.tests.bench.fixtures import agentic_fixture as F

CHECKPOINT = os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
SIZES = (4096, 8192, 12288)


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(CHECKPOINT)
    except Exception:
        return None


TOKENIZER = load_tokenizer() if os.path.isdir(CHECKPOINT) else None


class RealTokenizerBackend(FakeBackend):
    """The FakeBackend tape/pending logic with the real tokenizer, scripted replies and a prefill log."""

    def __init__(self, tokenizer, replies):
        super().__init__([])
        self.tokenizer = tokenizer
        self.replies = list(replies)
        self.appended = []                      # size of each append_tokens call = tokens that need prefill
        self.im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def append_tokens(self, ids):
        self.appended.append(len(ids))
        super().append_tokens(ids)

    def generate(self, max_tokens, out=None):
        text = self.replies.pop(0)
        if out:
            out(text)
        self.tape.extend(self.pending)
        self.pending = []
        self.tape.extend(self.encode(text) + self.im_end)       # the exact tokens "generated"
        return text, SimpleNamespace(finish="stop", tokens=len(self.encode(text)), prompt_tokens=len(self.tape))


def assistant_call(call_id, call):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}


def make_service(replies, limit=None, tokenizer=None):
    session = FakeSession([])
    session.backend = RealTokenizerBackend(tokenizer or TOKENIZER, replies)
    session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False)
    return ModelService(session, context_limit=limit), session.backend


@unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
class LongFixtureTests(unittest.TestCase):
    fixtures = {}

    @classmethod
    def setUpClass(cls):
        for size in SIZES:
            cls.fixtures[size] = F.build(TOKENIZER, size)

    def test_fixture_sizes_are_within_three_percent_of_target(self):
        for size, fixture in self.fixtures.items():
            self.assertLessEqual(abs(fixture["tokens"] - size), 0.03 * size, (size, fixture["tokens"]))
            self.assertIn("NIMBUS GATEWAY", fixture["messages"][1]["content"])
            self.assertNotIn(str(F.BATCH_SIZE), fixture["messages"][1]["content"])     # facts only in tool results
            self.assertNotIn(str(F.TOKEN_EXPIRY), fixture["messages"][1]["content"])

    def test_two_sequential_tool_calls_final_answer_and_follow_up_reuse_the_live_state(self):
        for size, fixture in self.fixtures.items():
            with self.subTest(size=size):
                replies = [F.tool_call_text(F.GLOB_CALL), F.tool_call_text(F.GREP_CALL), F.final_answer(),
                           "Lower the token expiry first: shorter lifetimes are the cheaper knob here."]
                service, backend = make_service(replies, limit=size + 4096)
                messages = list(fixture["messages"])
                tools = fixture["tools"]
                # 1. model generates glob
                first = service.complete(messages, tools, 1024)
                self.assertEqual([(c["name"], json.loads(c["arguments"])) for c in first.tool_calls],
                                 [("glob", F.GLOB_CALL["arguments"])])
                # cold: the whole prompt. The server's own rendering (thinking off here, reasoning-effort system text)
                # differs from the fixture's count by a constant 36 tokens, so assert a band, not equality.
                self.assertAlmostEqual(backend.appended[0], fixture["tokens"], delta=64)
                self.assertTrue(first.tool_calls[0]["id"].startswith("call_"))
                # 2. harness runs glob, result returns; 3. model generates grep
                messages += [assistant_call("call_1", F.GLOB_CALL),
                             {"role": "tool", "tool_call_id": "call_1", "content": F.glob_result()}]
                second = service.complete(messages, tools, 1024)
                self.assertEqual(second.tool_calls[0]["name"], "grep")
                self.assertEqual(json.loads(second.tool_calls[0]["arguments"]), F.GREP_CALL["arguments"])
                self.assertEqual(service.reused, 1, "the tool turn must extend the live state, not rebuild it")
                self.assertLess(backend.appended[1], 200, "only the tool result tokens should be prefilled")
                # 4. grep result returns; 5. final answer
                messages += [assistant_call("call_2", F.GREP_CALL),
                             {"role": "tool", "tool_call_id": "call_2", "content": F.grep_result()}]
                final = service.complete(messages, tools, 1024)
                self.assertEqual(final.tool_calls, [])
                verdict = F.check_answer(final.text)
                self.assertTrue(verdict["batch_size"] and verdict["token_expiry"] and not verdict["decoy_claimed"], verdict)
                self.assertEqual(service.reused, 2)
                # 6. related follow-up, retained context
                messages += [{"role": "assistant", "content": final.text},
                             {"role": "user", "content": F.FOLLOW_UP}]
                follow = service.complete(messages, tools, 1024)
                self.assertTrue(follow.text)
                self.assertEqual((service.reused, service.rebuilt), (3, 1))
                self.assertLess(backend.appended[3], 200)
                self.assertTrue(service.last_budget["continuation_ok"])

    def test_a_title_request_in_the_middle_evicts_and_the_conversation_is_rebuilt_exactly_not_corrupted(self):
        fixture = self.fixtures[4096]
        replies = [F.tool_call_text(F.GLOB_CALL), "Nimbus gateway config", F.tool_call_text(F.GREP_CALL)]
        service, backend = make_service(replies, limit=8192)
        messages, tools = list(fixture["messages"]), fixture["tools"]
        service.complete(messages, tools, 1024)
        messages += [assistant_call("call_1", F.GLOB_CALL), {"role": "tool", "tool_call_id": "call_1", "content": F.glob_result()}]
        title = service.complete([{"role": "user", "content": "Write a 4 word title for: find the config"}], [], 32)
        self.assertEqual(title.text, "Nimbus gateway config")
        self.assertEqual(service.sessions.switches, 1)
        again = service.complete(messages, tools, 1024)                 # rebuilt from its full exact token sequence
        self.assertEqual(again.tool_calls[0]["name"], "grep")
        self.assertEqual(service.rebuilt, 3)
        self.assertGreater(backend.appended[-1], fixture["tokens"] - 100)
        self.assertEqual(service.cache_state()["evictions"], 2)

    def test_overflow_refuses_with_client_side_guidance_and_runs_nothing(self):
        fixture = self.fixtures[12288]
        service, backend = make_service([], limit=12288)
        with self.assertRaises(ContextLimitError) as raised:
            service.complete(fixture["messages"], fixture["tools"], 1024)
        message = str(raised.exception)
        for needle in ("12288", "never truncates", "Compact on the client"):
            self.assertIn(needle, message)
        self.assertEqual((backend.tape, backend.pending, backend.appended), ([], [], []))

    def test_cold_prefill_limit_of_the_memory_tier_refuses_before_any_state_changes(self):
        fixture = self.fixtures[12288]
        session = FakeSession([])
        session.backend = RealTokenizerBackend(TOKENIZER, [])
        session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False)
        service = ModelService(session, context_limit=16384, max_cold_prefill=4096)
        with self.assertRaises(ContextLimitError) as raised:
            service.complete(fixture["messages"], fixture["tools"], 1024)
        self.assertEqual(raised.exception.code, "prefill_budget_exceeded")
        self.assertEqual(session.backend.appended, [])


if __name__ == "__main__":
    unittest.main()
