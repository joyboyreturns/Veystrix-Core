"""Effort turn placement: same tokens for every effort through the end of the system message; medium and off unchanged."""
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from veystrix import effort_slot as E
from veystrix import reasoning as R
from veystrix.server import ModelService
from veystrix.tests.test_server import FakeSession
from models.flashnext import prefix_session

TEMPLATE = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
RENDER = "<|im_start|>system\n# Tools\n...SYSTEM PROMPT<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
END = "<|im_end|>\n"


def ids(text):
    return [ord(c) for c in text]


class PlacementTests(unittest.TestCase):
    def test_late_is_the_default_and_template_restores_the_checkpoints_layout(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(E.PLACEMENT_ENV, None)
            self.assertEqual(E.placement(), "late")
        for value, expected in (("template", "template"), (" TEMPLATE ", "template"), ("late", "late"), ("nonsense", "late"), ("", "late")):
            with patch.dict(os.environ, {E.PLACEMENT_ENV: value}):
                self.assertEqual(E.placement(), expected, value)

    def test_only_effort_sentences_make_a_turn_and_off_or_medium_make_none(self):
        self.assertEqual(E.block_text("medium", True), "")
        for effort in ("low", "high", "xhigh"):
            block = E.block_text(effort, True)
            self.assertTrue(block.startswith("<|im_start|>system\n"), effort)
            self.assertTrue(block.endswith("<|im_end|>\n"), effort)
            self.assertEqual(E.block_text(effort, False), "", "thinking off has no effort sentence")
        self.assertNotEqual(E.block_text("low", True), E.block_text("xhigh", True))
        self.assertEqual(E.sentence("max", True), "", "max is clamped to xhigh before it gets here")

    @unittest.skipUnless(os.path.exists(f"{TEMPLATE}/chat_template.jinja"), "checkpoint template not present")
    def test_the_sentences_are_verbatim_from_the_checkpoints_template(self):
        with open(f"{TEMPLATE}/chat_template.jinja") as handle:
            template = handle.read()
        for sentence in E.TEMPLATE_SENTENCES.values():
            self.assertIn(sentence, template)


class ServerInsertionTests(unittest.TestCase):
    def service(self):
        session = FakeSession(["answer"])
        session.backend.tokenizer.rendered = RENDER
        service = ModelService(session)
        service._request = threading.local()
        return service, session

    def run_request(self, effort_payload, tools=None):
        service, session = self.service()
        service._request.reasoning = R.resolve(effort_payload, False, "medium")
        service.complete([{"role": "system", "content": "SYSTEM PROMPT"}, {"role": "user", "content": "hello"}], tools or [], 10)
        return service, session

    def fed(self, session):
        """The tokens the backend was fed for the prompt (the fake backend moves them to the tape on generate)."""
        text = "".join(session.backend.pieces)
        tape = list(session.backend.tape)
        return tape[:len(tape) - len(ids(text))]

    def test_medium_and_thinking_off_feed_exactly_the_rendered_text(self):
        for payload in ({"reasoning_effort": "medium"}, {"reasoning_effort": "none"}, {}):
            service, session = self.run_request(payload)
            self.assertEqual(self.fed(session), ids(RENDER), payload)
            self.assertEqual(session.backend.tokenizer.options["reasoning_effort"], "medium")

    def test_every_effort_renders_the_template_as_medium_and_gets_its_own_system_turn_after_the_system_message(self):
        for effort in ("low", "high", "xhigh"):
            service, session = self.run_request({"reasoning_effort": effort})
            self.assertEqual(session.backend.tokenizer.options["reasoning_effort"], "medium", effort)
            self.assertTrue(session.backend.tokenizer.options["enable_thinking"])
            turn = ids(E.block_text(effort, True))
            cut = RENDER.index(END) + len(END)
            expected = ids(RENDER[:cut]) + turn + ids(RENDER[cut:])
            self.assertEqual(self.fed(session), expected, effort)

    def test_high_no_longer_prepends_its_sentence_to_the_system_prompt(self):
        service, session = self.run_request({"reasoning_effort": "high"})
        content = next(m for m in session.backend.tokenizer.messages if m["role"] == "system")["content"]
        self.assertEqual(content, "SYSTEM PROMPT")

    def test_template_placement_is_the_old_behaviour_exactly(self):
        with patch.dict(os.environ, {E.PLACEMENT_ENV: "template"}):
            service, session = self.run_request({"reasoning_effort": "low"})
            self.assertEqual(session.backend.tokenizer.options["reasoning_effort"], "low")
            self.assertEqual(self.fed(session), ids(RENDER))
            service, session = self.run_request({"reasoning_effort": "high"})
            self.assertEqual(session.backend.tokenizer.options["reasoning_effort"], "medium")
            content = next(m for m in session.backend.tokenizer.messages if m["role"] == "system")["content"]
            self.assertIn("Think carefully", content)

    def test_a_prompt_without_a_system_end_marker_is_left_alone(self):
        service, session = self.service()
        session.backend.tokenizer.rendered = "no marker here"
        service._request.reasoning = R.resolve({"reasoning_effort": "low"}, False, "medium")
        service.complete([{"role": "user", "content": "x"}], [], 10)
        self.assertEqual(self.fed(session), ids("no marker here"))

    def test_changing_effort_in_a_live_conversation_rebuilds_instead_of_extending_a_wrong_state(self):
        service, session = self.service()
        service._request.reasoning = R.resolve({"reasoning_effort": "low"}, False, "medium")
        messages = [{"role": "system", "content": "SYSTEM PROMPT"}, {"role": "user", "content": "hello"}]
        service.complete(messages, [], 10)
        before = (service.reused, service.rebuilt)
        service._request.reasoning = R.resolve({"reasoning_effort": "xhigh"}, False, "medium")
        service.complete(messages, [], 10)
        self.assertEqual(service.reused, before[0], "a different effort must not extend the old state")
        self.assertEqual(service.rebuilt, before[1] + 1)


@unittest.skipUnless(os.path.exists(f"{TEMPLATE}/tokenizer.json") or os.path.exists(f"{TEMPLATE}/tokenizer_config.json"), "tokenizer not present")
class RealTokenizerTests(unittest.TestCase):
    """The claim the whole design rests on, checked with the real tokenizer and chat template."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(TEMPLATE)
        cls.tools = [{"type": "function", "function": {"name": "bash", "description": "run a command",
                                                       "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
        cls.system = "You are a coding agent.\nWorking directory: /Users/x/repo.\nBe concise."
        cls.messages = [{"role": "system", "content": cls.system}, {"role": "user", "content": "list the files"}]

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def render(self, effort, thinking):
        return self.tok.apply_chat_template(self.messages, tools=self.tools, add_generation_prompt=True, tokenize=False,
                                            enable_thinking=thinking, reasoning_effort=effort)

    def late(self, effort, thinking=True):
        ids = self.encode(self.render("medium", thinking))
        text = E.block_text(effort, thinking)
        if not text:
            return ids
        marker = self.encode(prefix_session.IM_END)
        cut = next(i for i in range(len(ids)) if ids[i:i + len(marker)] == marker) + len(marker)
        return ids[:cut] + self.encode(text) + ids[cut:]

    def system_end(self, ids):
        marker = self.encode(prefix_session.IM_END)
        return next(i for i in range(len(ids)) if ids[i:i + len(marker)] == marker) + len(marker)

    @staticmethod
    def common(a, b):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    def test_medium_and_off_are_token_for_token_what_they_always_were(self):
        self.assertEqual(self.late("medium"), self.encode(self.render("medium", True)))
        self.assertEqual(self.late("medium", thinking=False), self.encode(self.render("medium", False)))

    def test_off_and_medium_share_everything_but_the_generation_prompt(self):
        a, b = self.late("medium"), self.late("medium", thinking=False)
        self.assertGreater(self.common(a, b), len(a) - 8)

    def test_with_the_late_layout_every_effort_shares_the_whole_system_message(self):
        medium = self.late("medium")
        head = self.system_end(medium)
        self.assertGreater(head, 50)
        for effort in ("low", "high", "xhigh"):
            tokens = self.late(effort)
            self.assertEqual(tokens[:head], medium[:head], effort)
            self.assertEqual(self.system_end(tokens), head, "the system end, where the cache anchors, is the same position")
            self.assertGreater(len(tokens), len(medium))

    def test_the_inserted_turn_tokenizes_the_same_standalone_as_in_place(self):
        """Special tokens on both ends mean no BPE merge can differ between the standalone and the joint encoding."""
        for effort in ("low", "high", "xhigh"):
            text = E.block_text(effort, True)
            standalone = self.encode(text)
            self.assertEqual(standalone[0], self.tok.convert_tokens_to_ids("<|im_start|>"))
            self.assertEqual(standalone[-2], self.tok.convert_tokens_to_ids("<|im_end|>"))

    def test_the_checkpoints_own_layout_shared_almost_nothing(self):
        """What this change fixes: with the sentence first, low and xhigh diverge after three tokens."""
        medium = self.encode(self.render("medium", True))
        for effort in ("low", "xhigh"):
            self.assertLess(self.common(self.encode(self.render(effort, True)), medium), 10, effort)


if __name__ == "__main__":
    unittest.main()
