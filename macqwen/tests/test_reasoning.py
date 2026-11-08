"""Per-request reasoning controls: the resolver, the wire contract Qwen Code 0.22.0 produces, and the real chat template."""
import os
import unittest

from macqwen import preferences, reasoning as R
from macqwen.conversation import EXTRA_REASONING
from macqwen.server import ModelService, RequestError, _check_chat_fields
from macqwen.tests.test_agentic_long_fixtures import RealTokenizerBackend, TOKENIZER
from macqwen.tests.test_server import FakeSession

LOW = "Reasoning effort is set to low."
XHIGH = "Reasoning effort is set to xhigh."


def resolve(payload, thinking=False, effort="medium"):
    return R.resolve(payload, thinking, effort)


class ResolverTests(unittest.TestCase):
    def test_qwen_code_slash_effort_wire_shape_for_each_tier(self):
        """Measured from the installed client: /effort low|medium|high|xhigh -> reasoning.effort; max is clamped to xhigh by it."""
        for tier in ("low", "medium", "high", "xhigh"):
            r = resolve({"reasoning": {"effort": tier}, "max_tokens": 4000})
            self.assertEqual((r.thinking, r.effort, r.source), (True, tier, "reasoning.effort"), tier)
            self.assertEqual(r.wire, {"reasoning": {"effort": tier}})

    def test_no_reasoning_field_means_the_server_default_and_says_so(self):
        r = resolve({})
        self.assertEqual((r.thinking, r.effort, r.source), (False, "medium", "server-default"))
        r = resolve({}, thinking=True, effort="high")
        self.assertEqual((r.thinking, r.effort, r.source), (True, "high", "server-default"))

    def test_max_is_clamped_to_xhigh_and_reported(self):
        r = resolve({"reasoning": {"effort": "max"}})
        self.assertEqual((r.thinking, r.effort, r.requested), (True, "xhigh", "xhigh"))
        self.assertTrue(any("max has no distinct level" in n for n in r.notes))

    def test_every_off_spelling_turns_thinking_off(self):
        for payload in ({"reasoning": {"effort": "none"}}, {"reasoning_effort": "none"}, {"reasoning": False},
                        {"reasoning": {"enabled": False}}, {"chat_template_kwargs": {"enable_thinking": False}},
                        {"enable_thinking": False}, {"reasoning": {"effort": "off"}}):
            r = resolve(payload, thinking=True)
            self.assertFalse(r.thinking, payload)
            self.assertNotEqual(r.source, "server-default", payload)

    def test_flat_openai_field_and_native_template_switch_are_honoured(self):
        self.assertEqual(resolve({"reasoning_effort": "low"}).effort, "low")
        r = resolve({"chat_template_kwargs": {"enable_thinking": True}}, effort="high")
        self.assertEqual((r.thinking, r.effort, r.source), (True, "high", "chat_template_kwargs.enable_thinking"))
        self.assertTrue(resolve({"enable_thinking": True}).thinking)

    def test_an_explicit_disable_beats_an_effort_in_the_same_request_and_is_reported(self):
        r = resolve({"reasoning": {"effort": "medium"}, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertFalse(r.thinking)
        self.assertTrue(any("explicit thinking disable" in n for n in r.notes))

    def test_unknown_levels_and_unsupported_controls_are_errors_not_ignored(self):
        for payload in ({"reasoning": {"effort": "ludicrous"}}, {"reasoning_effort": "turbo"},
                        {"reasoning": {"max_tokens": 2000}}, {"chat_template_kwargs": {"preserve_thinking": True}},
                        {"chat_template_kwargs": "x"}, {"reasoning": ["high"]}, {"reasoning": {"summary": "auto"}}):
            with self.assertRaises(R.ReasoningError, msg=payload):
                resolve(payload)

    def test_the_four_levels_are_distinct_after_translation(self):
        from macqwen.conversation import reasoning_system_text
        seen = {tier: reasoning_system_text("SYSTEM", tier) for tier in R.LEVELS}
        self.assertEqual(len(set(seen.values())), 4)
        self.assertEqual(seen["high"][1], "medium")              # high rides on the medium template plus its own sentence
        self.assertIn(EXTRA_REASONING["high"], seen["high"][0])
        self.assertEqual(seen["low"], ("SYSTEM", "low"))

    def test_server_accepts_the_fields_instead_of_refusing_them(self):
        ignored = _check_chat_fields({"model": "m", "messages": [], "reasoning": {"effort": "low"}, "reasoning_effort": "low",
                                      "chat_template_kwargs": {"enable_thinking": True}, "enable_thinking": True})
        self.assertEqual(ignored, [])
        with self.assertRaises(RequestError):
            _check_chat_fields({"model": "m", "messages": [], "thinking_budget": 100})


@unittest.skipIf(TOKENIZER is None, "the checkpoint tokenizer is not available")
class RealTemplateTests(unittest.TestCase):
    """The scripted model, the REAL tokenizer and chat template: what the model would be shown for each level."""

    def setUp(self):
        self.session = FakeSession([])
        self.session.backend = RealTokenizerBackend(TOKENIZER, [f"answer {i}" for i in range(20)])
        self.session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False, effort="medium")
        self.service = ModelService(self.session, context_limit=16384)
        self.messages = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "fix it"}]

    def ask(self, payload):
        self.service._request.reasoning = R.resolve(payload, self.session.preferences["thinking_enabled"],
                                                    self.session.preferences["effort"])
        try:
            self.service.complete(self.messages, [], 64)
        finally:
            self.service._request.reasoning = None
        return TOKENIZER.decode(self.session.backend.tape)

    def test_each_level_changes_the_rendered_prompt_the_way_the_template_defines(self):
        low = self.ask({"reasoning": {"effort": "low"}})
        self.assertIn(LOW, low)
        self.assertTrue(low.rstrip().split("answer")[0].rstrip().endswith("<think>") or "<think>\n" in low)
        self.setUp()
        xhigh = self.ask({"reasoning": {"effort": "xhigh"}})
        self.assertIn(XHIGH, xhigh)
        self.setUp()
        medium = self.ask({"reasoning": {"effort": "medium"}})
        self.assertNotIn("Reasoning effort is set to", medium)
        self.assertIn("<think>\n", medium)
        self.setUp()
        high = self.ask({"reasoning": {"effort": "high"}})
        self.assertIn(EXTRA_REASONING["high"], high)
        self.assertNotIn("Reasoning effort is set to", high)
        self.setUp()
        off = self.ask({"reasoning": {"effort": "none"}})
        self.assertIn("<think>\n\n</think>", off)
        self.assertEqual(len({low, xhigh, medium, high, off}), 5)

    def test_switching_low_high_medium_xhigh_low_in_one_session_follows_each_request_and_stores_nothing(self):
        before = dict(self.session.preferences)
        sequence = ("low", "high", "medium", "xhigh", "low")
        rendered, applied = [], []
        for level in sequence:
            rendered.append(self.ask({"reasoning": {"effort": level}}))
            applied.append(self.service.last_reasoning.effort)
        self.assertEqual(applied, list(sequence))
        self.assertEqual(self.session.preferences, before)              # nothing was written to the shared session
        self.assertIn(LOW, rendered[0])
        self.assertIn(EXTRA_REASONING["high"], rendered[1])
        self.assertIn(XHIGH, rendered[3])
        self.assertIn(LOW, rendered[4])
        # every switch changes the prompt prefix (the template puts the instruction first), so the live tape is rebuilt,
        # never reused across a level change: exact token identity is the cache identity
        self.assertEqual(self.service.reused, 0)
        self.assertEqual(self.service.rebuilt, 5)

    def test_the_same_setting_twice_in_a_row_reuses_the_live_state_when_thinking_is_off(self):
        self.ask({"reasoning": {"effort": "none"}})
        self.messages = self.messages + [{"role": "assistant", "content": "answer 0"}, {"role": "user", "content": "again"}]
        self.ask({"reasoning": {"effort": "none"}})
        self.assertEqual((self.service.rebuilt, self.service.reused), (1, 1))

    def test_with_thinking_on_a_resent_history_rebuilds_but_the_divergence_is_the_think_tail_not_the_effort_text(self):
        """Known property, not an effort bug: the client echoes the reply without its reasoning, so the template re-renders the
        past turn as an empty think block and differs from the tape the model really produced. The system prompt (which holds
        the effort sentence) is identical, and the recorded first difference is after it."""
        self.ask({"reasoning": {"effort": "low"}})
        system_tokens = len(TOKENIZER.encode(self.session.backend.render_prefix)) if hasattr(self.session.backend, "render_prefix") else 0
        self.messages = self.messages + [{"role": "assistant", "content": "answer 0"}, {"role": "user", "content": "again"}]
        self.ask({"reasoning": {"effort": "low"}})
        self.assertEqual(self.service.rebuilt, 2)
        divergence = self.service.last_divergence
        self.assertGreater(divergence["first_difference"], system_tokens)
        window = TOKENIZER.decode(divergence["prompt_window"])
        self.assertIn("think", window)

    def test_an_auxiliary_request_with_other_controls_does_not_change_the_next_main_request(self):
        main = self.ask({"reasoning": {"effort": "xhigh"}})
        self.ask({"reasoning": {"effort": "none"}})                     # e.g. a title/summary side request
        again = self.ask({"reasoning": {"effort": "xhigh"}})
        self.assertIn(XHIGH, main)
        self.assertIn(XHIGH, again)
        self.assertEqual(self.session.preferences["effort"], "medium")
        self.assertFalse(self.session.preferences["thinking_enabled"])


if __name__ == "__main__":
    unittest.main()
