"""Reasoning-effort wire shapes (DeepSeek Harness and Qwen Code) and what each effort switch costs the cache.

Real checkpoint tokenizer and the real chat template; the backend is scripted (no model). Facts pinned here were measured on
2026-11-09 with the real DeepSeek Harness 0.2.2-rc.1 CLI against a stub server and with the real tokenizer on its 23- and
32-tool requests (see docs/flashnext/reasoning_compatibility_matrix.md):

* DSH offers off/low/medium/high/xhigh/max and puts ``reasoning_effort: none|low|medium|high|xhigh|max`` on the wire.
* The checkpoint's template knows xhigh, medium and low plus thinking on/off. ``high`` adds one repository sentence late in the
  system prompt; ``max`` is clamped to ``xhigh`` and reported. There is no fifth level.
* The effort sentence sits at the START of the system message, ahead of the tool schemas, so low and xhigh share no useful
  prefix with each other or with medium; off and medium differ only in the generation prompt and share the whole prefix.
"""
from __future__ import annotations

import unittest

from veystrix import reasoning as R
from veystrix.conversation import reasoning_system_text
from veystrix.server import ModelService, _check_chat_fields, _openai_tools
from veystrix.sessions import derive_key
from veystrix.tests.test_reasoning_cache import ScriptedModelBackend, TOKENIZER
from veystrix.tests.test_server import FakeSession

# The reasoning-related keys of each request captured from the real `dsh headless` CLI (profile with reasoningEfforts).
DSH_WIRE = {
    "off": {"reasoning_effort": "none"}, "low": {"reasoning_effort": "low"}, "medium": {"reasoning_effort": "medium"},
    "high": {"reasoning_effort": "high"}, "xhigh": {"reasoning_effort": "xhigh"}, "max": {"reasoning_effort": "max"},
}
# Qwen Code 0.27.0 nested form, measured by the previous session (docs/flashnext/qwen-code-integration-2026-11-09.md).
QWEN_WIRE = {name: {"reasoning": {"effort": value}} for name, value in
             (("off", "none"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"), ("max", "max"))}

EXPECTED = {   # name -> (thinking, applied effort)
    "off": (False, None), "low": (True, "low"), "medium": (True, "medium"), "high": (True, "high"),
    "xhigh": (True, "xhigh"), "max": (True, "xhigh"),
}


def tools(count=23):
    return [{"type": "function", "function": {"name": f"tool_{i}", "description": "Tool number %d. " % i + "x" * 600,
             "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "y" * 200},
                                                            "mode": {"type": "string", "enum": ["a", "b", "c"]}},
                            "required": ["path"]}}} for i in range(count)]


class WireShapeTests(unittest.TestCase):
    def check(self, table):
        for name, wire in table.items():
            _check_chat_fields({"model": "m", "messages": [], **wire})          # the server must not 400 on the field
            r = R.resolve(wire, False, "medium")
            self.assertEqual((r.thinking, r.effort if r.thinking else None), EXPECTED[name], name)
        self.assertTrue(any("clamped" in n or "using xhigh" in n for n in R.resolve(table["max"], False, "medium").notes))

    def test_every_deepseek_harness_effort_is_accepted_and_mapped(self):
        self.check(DSH_WIRE)

    def test_every_qwen_code_effort_is_accepted_and_mapped(self):
        self.check(QWEN_WIRE)

    def test_both_clients_agree_on_the_applied_reasoning(self):
        for name in DSH_WIRE:
            a, b = R.resolve(DSH_WIRE[name], False, "medium"), R.resolve(QWEN_WIRE[name], False, "medium")
            self.assertEqual((a.thinking, a.effort), (b.thinking, b.effort), name)

    def test_an_unknown_effort_is_a_400_never_silently_ignored(self):
        with self.assertRaises(R.ReasoningError):
            R.resolve({"reasoning_effort": "ultra"}, False, "medium")

    def test_the_server_default_is_thinking_off_when_a_client_sends_nothing(self):
        r = R.resolve({}, False, "medium")
        self.assertEqual((r.thinking, r.source), (False, "server-default"))


@unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
class PromptPrefixTests(unittest.TestCase):
    SYSTEM = "You are a coding agent powered by the veystrix-flashnext model. " * 40

    def render(self, effort_name):
        r = R.resolve(DSH_WIRE[effort_name], False, "medium")
        text, template_effort = reasoning_system_text(self.SYSTEM, r.effort)
        rendered = TOKENIZER.apply_chat_template(
            [{"role": "system", "content": text}, {"role": "user", "content": "hello"}], tools=_openai_tools(tools()) or None,
            add_generation_prompt=True, tokenize=False, enable_thinking=r.thinking, reasoning_effort=template_effort)
        return TOKENIZER.encode(rendered, add_special_tokens=False)

    @staticmethod
    def common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def test_off_and_medium_share_the_whole_static_prefix(self):
        medium, off = self.render("medium"), self.render("off")
        self.assertGreater(len(medium), 5000)
        self.assertGreaterEqual(self.common(medium, off), len(medium) - 4)

    def test_high_shares_a_long_prefix_with_medium(self):
        medium, high = self.render("medium"), self.render("high")
        self.assertGreater(self.common(medium, high), len(medium) // 2)

    def test_low_and_xhigh_change_the_prompt_from_the_start(self):
        """The template puts the effort sentence first: switching to or from low/xhigh invalidates the whole static prefix.
        Honest cost, not normalised away: each effort keeps its own snapshot after its first use."""
        medium = self.render("medium")
        for name in ("low", "xhigh"):
            self.assertLess(self.common(medium, self.render(name)), 64, name)
        self.assertLess(self.common(self.render("low"), self.render("xhigh")), 64)

    def test_max_is_exactly_xhigh_on_the_token_level(self):
        self.assertEqual(self.render("max"), self.render("xhigh"))

    def test_the_six_efforts_give_five_distinct_prompts(self):
        prompts = {name: tuple(self.render(name)) for name in DSH_WIRE}
        self.assertEqual(len({prompts[n] for n in ("low", "medium", "high", "xhigh")}), 4)
        self.assertEqual(prompts["max"], prompts["xhigh"])
        self.assertNotEqual(prompts["off"], prompts["medium"])


@unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
class EffortSwitchCacheTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession([])
        self.backend = ScriptedModelBackend(TOKENIZER, [])
        self.session.backend = self.backend
        self.service = ModelService(self.session)

    def turn(self, messages, name, tool_list=()):
        self.service._request.reasoning = R.resolve(DSH_WIRE[name], False, "medium")
        try:
            return self.service.complete(messages, list(tool_list), 512)
        finally:
            self.service._request.reasoning = None

    def test_off_to_medium_to_off_inside_one_conversation_keeps_the_state(self):
        self.backend.replies = ["A1", "Thinking\n</think>\n\nA2", "A3"]
        m = [{"role": "user", "content": "q1"}]
        self.turn(m, "off")
        m += [{"role": "assistant", "content": "A1"}, {"role": "user", "content": "q2"}]
        self.turn(m, "medium")
        m += [{"role": "assistant", "content": "A2", "reasoning_content": "Thinking"}, {"role": "user", "content": "q3"}]
        self.turn(m, "off")
        self.assertEqual((self.service.reused, self.service.rebuilt), (2, 1), "no cold rebuild for off<->medium")

    def test_low_to_xhigh_inside_one_conversation_rebuilds_because_the_prompt_differs(self):
        self.backend.replies = ["Think\n</think>\n\nB1", "Think2\n</think>\n\nB2"]
        m = [{"role": "user", "content": "q1"}]
        self.turn(m, "low")
        m += [{"role": "assistant", "content": "B1", "reasoning_content": "Think"}, {"role": "user", "content": "q2"}]
        self.turn(m, "xhigh")
        self.assertEqual((self.service.reused, self.service.rebuilt), (0, 2))
        self.assertIsNotNone(self.service.last_divergence)
        self.assertLess(self.service.last_divergence["first_difference"], 64)

    def test_the_full_ladder_then_max_to_low_never_reuses_a_state_built_under_another_effort(self):
        prompts = {}
        for name in ("off", "low", "medium", "high", "xhigh", "max", "low"):
            self.backend.replies = [f"R-{name}"] if name == "off" else [f"t\n</think>\n\nR-{name}"]
            self.turn([{"role": "user", "content": "same question"}], name)
            prompts[len(prompts)] = self.backend.prompts[-1]
        self.assertEqual(self.service.reused, 0, "an effort change must never extend a state built under another effort")
        self.assertEqual(self.service.rebuilt, 7)
        # max clamps to xhigh: the same text reaches the model for both
        self.assertEqual(prompts[4], prompts[5])
        self.assertEqual(prompts[1], prompts[6])
        self.assertNotEqual(prompts[1], prompts[4])

    def test_the_ledger_key_does_not_depend_on_effort(self):
        """Reasoning the model really produced in a conversation may be restored whatever effort the next request names."""
        base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertEqual(derive_key(base, None, tools=tools(2)), derive_key(list(base), None, tools=tools(2)))

    def test_a_multi_tool_thinking_turn_survives_a_client_that_drops_reasoning(self):
        call = ('<tool_call>\n<function=tool_0>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>\n'
                '<tool_call>\n<function=tool_1>\n<parameter=path>\nb.py\n</parameter>\n</function>\n</tool_call>')
        self.backend.replies = [f"plan the reads\n</think>\n\n{call}", "digest\n</think>\n\nBoth files read."]
        spec = tools(3)
        m = [{"role": "user", "content": "read two files"}]
        first = self.turn(m, "medium", spec)
        self.assertEqual(len(first.tool_calls), 2)
        wire = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in first.tool_calls]
        m2 = m + [{"role": "assistant", "content": "", "tool_calls": wire},          # reasoning omitted by the client
                  {"role": "tool", "tool_call_id": wire[0]["id"], "content": "file a"},
                  {"role": "tool", "tool_call_id": wire[1]["id"], "content": "file b"}]
        second = self.turn(m2, "medium", spec)
        self.assertEqual(second.text, "Both files read.")
        self.assertEqual((self.service.reused, self.service.rebuilt), (1, 1))
        self.assertGreaterEqual(self.service.reasoning_ledger.restored_turns_total, 1)

    def test_returned_reasoning_is_not_duplicated_or_replaced_by_a_foreign_one(self):
        self.backend.replies = ["own thought\n</think>\n\nA1", "again\n</think>\n\nA2"]
        m = [{"role": "user", "content": "q1"}]
        self.turn(m, "medium")
        m2 = m + [{"role": "assistant", "content": "A1", "reasoning_content": "own thought"}, {"role": "user", "content": "q2"}]
        self.turn(m2, "medium")
        prompt = self.backend.prompts[-1]
        self.assertEqual(prompt.count("own thought"), 1)
        self.assertEqual(self.service.reused, 1)

    def test_cancellation_during_reasoning_leaves_no_state_and_no_ledger_entry(self):
        from veystrix.backends.base import GenerationCancelled

        def cancelled(max_tokens, out=None, should_cancel=None):
            raise GenerationCancelled

        original = self.backend.generate
        self.backend.generate = cancelled
        with self.assertRaises(GenerationCancelled):
            self.turn([{"role": "user", "content": "q1"}], "high")
        self.assertEqual((self.backend.tape, self.backend.pending), ([], []))
        self.assertEqual(self.service.reasoning_ledger.sessions, {})
        self.backend.generate = original
        self.backend.replies = ["r\n</think>\n\nOK"]
        result = self.turn([{"role": "user", "content": "q1"}], "high")
        self.assertEqual(result.text, "OK")

    def test_two_interleaved_sessions_never_see_each_others_reasoning(self):
        self.backend.replies = ["secret-A\n</think>\n\nanswer A", "secret-B\n</think>\n\nanswer B", "x\n</think>\n\nA again"]
        a = [{"role": "user", "content": "task A"}]
        b = [{"role": "user", "content": "task B"}]
        self.turn(a, "medium")
        self.turn(b, "medium")
        a2 = a + [{"role": "assistant", "content": "answer A"}, {"role": "user", "content": "more A"}]   # reasoning omitted
        self.turn(a2, "medium")
        self.assertNotIn("secret-B", self.backend.prompts[-1])
        self.assertIn("secret-A", self.backend.prompts[-1], "A's own reasoning is restored from its own ledger")


if __name__ == "__main__":
    unittest.main()
