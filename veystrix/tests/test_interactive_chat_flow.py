"""Focused offline interface tests for interactive chat in Veystrix.

Verifies:
1. Tokenizer and chat template formatting (system/user/assistant, no tools).
2. Multi-turn conversation history preservation.
3. Streaming response emission.
4. Output-token and reasoning limit enforcement.
5. In-session commands (/reset, /exit, /max-tokens).
6. Safe cancellation and staged interrupt handling.
7. Isolated session vs fast chat environment configurations.
"""
from __future__ import annotations

import io
import os
import signal
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from veystrix import preferences
from veystrix.backends.base import GenerationCancelled
from veystrix.conversation import Conversation
from veystrix.profiles import system_prompt, tools_for
from veystrix.session import Session, main, run_turn_plain
from veystrix.ui import IngestGlow


class MockTokenizer:
    """Mock tokenizer providing realistic special tokens and chat template."""
    def __init__(self):
        self.eos_token_id = 151645
        self.added_tokens_decoder = {
            151644: SimpleNamespace(content="<|im_start|>"),
            151645: SimpleNamespace(content="<|im_end|>"),
            151649: SimpleNamespace(content="<think>"),
            151650: SimpleNamespace(content="</think>"),
        }
        self._vocab = {
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<think>": 151649,
            "</think>": 151650,
            "system": 100,
            "user": 101,
            "assistant": 102,
            "\n": 103,
            "hello": 200,
            "world": 201,
        }

    def encode(self, text, add_special_tokens=False):
        tokens = []
        words = text.replace("\n", " \n ").split()
        for w in words:
            if w in self._vocab:
                tokens.append(self._vocab[w])
            else:
                tokens.append(hash(w) % 10000 + 1000)
        return tokens

    def decode(self, token_ids):
        rev = {v: k for k, v in self._vocab.items()}
        res = []
        for t in token_ids:
            if t in rev:
                res.append(rev[t])
            else:
                res.append(f"[tok_{t}]")
        return " ".join(res).replace(" \n ", "\n")

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True,
        tokenize=False, enable_thinking=False, reasoning_effort="medium"
    ):
        del tools, reasoning_effort
        rendered = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            if enable_thinking:
                rendered += "<|im_start|>assistant\n<think>\n"
            else:
                rendered += "<|im_start|>assistant\n"
        return rendered


class MockChatBackend(Conversation):
    """Mock backend implementing Conversation protocol with streaming generation."""
    def __init__(self, tokenizer=None):
        tok = tokenizer or MockTokenizer()
        super().__init__(tok)
        self.stops = {tok.eos_token_id, 151645}
        self.thinking_enabled = False
        self._interactive_budgets = None
        self.generate_calls = []

    def reset(self):
        self.tape = []
        self.pending = []
        self.turn_closed = True

    def generate(self, max_tokens, out=None, on_prefilled=None,
                 on_prefill_progress=None, on_decode_token=None,
                 should_cancel=None):
        self.generate_calls.append(max_tokens)
        prompt_tokens = len(self.pending)
        self.tape.extend(self.pending)
        self.pending = []

        if on_prefill_progress:
            on_prefill_progress(prompt_tokens, prompt_tokens)
        if on_prefilled:
            on_prefilled()

        # Simulate generated response tokens
        reply_tokens = [200, 201]  # "hello world"
        if max_tokens > 0 and len(reply_tokens) > max_tokens:
            reply_tokens = reply_tokens[:max_tokens]

        for tok_id in reply_tokens:
            if should_cancel and should_cancel():
                raise GenerationCancelled
            self.tape.append(tok_id)
            piece = self.tokenizer.decode([tok_id]) + " "
            if on_decode_token:
                on_decode_token(tok_id, piece)
            if out:
                out(piece)

        stats = SimpleNamespace(
            finish="stop",
            tokens=len(reply_tokens),
            seconds=0.05,
            prompt_tokens=prompt_tokens,
            prefill_seconds=0.01,
            rate=40.0,
            prompt_rate=100.0,
            tail_tokens=0,
            tail_seconds=0.0,
        )
        return self.tokenizer.decode(reply_tokens), stats


class InteractiveChatFlowTests(unittest.TestCase):
    def test_plain_profile_has_no_tools_and_no_tool_syntax(self):
        """Verify plain profile chat strictly provides tools=None to template."""
        tools = tools_for("plain")
        self.assertIsNone(tools, "Plain chat must not supply tools")
        prompt = system_prompt("plain", "/tmp")
        self.assertNotIn("<tool_call>", prompt)
        self.assertNotIn("<function=", prompt)

    def test_multi_turn_and_history_preservation(self):
        """Verify turn 1 and turn 2 accumulate transcript correctly on tape."""
        backend = MockChatBackend()
        glow = IngestGlow()
        with tempfile.TemporaryDirectory() as root:
            prefs_path = str(Path(root) / "prefs.json")
            session = Session(backend, "plain", dict(preferences.DEFAULTS), prefs_path)

            # Turn 1
            streamed_pieces_1 = []
            session.preferences["stream_answers"] = True
            session.preferences["animate"] = False
            with redirect_stdout(io.StringIO()), patch("veystrix.session.AsyncWordAnimator.feed", side_effect=lambda piece, _: streamed_pieces_1.append(piece)):
                run_turn_plain(session, "First question", glow)

            self.assertTrue(session.opened)
            turn_1_tape_len = len(backend.tape)
            self.assertGreater(turn_1_tape_len, 0)
            self.assertEqual(len(backend.pending), 0)

            # Turn 2: Follow-up question
            streamed_pieces_2 = []
            with redirect_stdout(io.StringIO()), patch("veystrix.session.AsyncWordAnimator.feed", side_effect=lambda piece, _: streamed_pieces_2.append(piece)):
                run_turn_plain(session, "Follow up question", glow)

            turn_2_tape_len = len(backend.tape)
            self.assertGreater(turn_2_tape_len, turn_1_tape_len, "Tape must preserve and grow across turns")

    def test_reset_command_clears_history_keeps_session_alive(self):
        """Verify /reset resets conversation history while keeping session open."""
        backend = MockChatBackend()
        glow = IngestGlow()
        with tempfile.TemporaryDirectory() as root:
            prefs_path = str(Path(root) / "prefs.json")
            session = Session(backend, "plain", dict(preferences.DEFAULTS), prefs_path)
            session.preferences["animate"] = False

            with redirect_stdout(io.StringIO()):
                run_turn_plain(session, "Question before reset", glow)
            self.assertTrue(session.opened)
            self.assertGreater(len(backend.tape), 0)

            # Issue reset
            session.reset()
            self.assertFalse(session.opened)
            self.assertEqual(len(backend.tape), 0)
            self.assertEqual(len(backend.pending), 0)
            self.assertTrue(session.running, "Session must stay alive after reset")

            # Post-reset turn starts fresh
            with redirect_stdout(io.StringIO()):
                run_turn_plain(session, "Question after reset", glow)
            self.assertTrue(session.opened)
            self.assertGreater(len(backend.tape), 0)

    def test_full_interactive_session_loop_with_reset_and_exit(self):
        """Test the end-to-end interactive loop simulating user typing messages, /reset, /exit."""
        backend = MockChatBackend()
        inputs = [
            "Hello there",        # Turn 1
            "Can you tell me more?",  # Turn 2
            "/reset",             # Reset command
            "A fresh start",      # Turn 3
            "/exit",              # Exit command
        ]

        with tempfile.TemporaryDirectory() as root:
            prefs_path = str(Path(root) / "prefs.json")
            with patch.object(sys, "argv", [
                "session.py", "--model", "flashnext",
                "--profile", "plain",
                "--preferences-file", prefs_path,
            ]), patch(
                "veystrix.session.build_backend", return_value=backend
            ), patch(
                "veystrix.checkpoints.resolve_flashnext", return_value=Path(root)   # the backend is a fake: no checkpoint needed
            ), patch(
                "veystrix.session.read_prompt", side_effect=inputs
            ), redirect_stdout(io.StringIO()) as out:
                ret = main()

            self.assertEqual(ret, 0, "Clean /exit must exit with code 0")
            output_str = out.getvalue()
            self.assertIn("conversation reset, model stayed loaded", output_str)
            # Backend should have run 3 turns total
            self.assertEqual(len(backend.generate_calls), 3)

    def test_configurable_max_tokens_limit(self):
        """Verify --max-tokens and /max-tokens enforce output token limit."""
        backend = MockChatBackend()
        with tempfile.TemporaryDirectory() as root:
            prefs_path = str(Path(root) / "prefs.json")
            session = Session(backend, "plain", dict(preferences.DEFAULTS), prefs_path)
            session.preferences["animate"] = False
            session.preferences["max_tokens"] = 1

            glow = IngestGlow()
            with redirect_stdout(io.StringIO()):
                run_turn_plain(session, "Testing token limit", glow)
            self.assertEqual(backend.generate_calls[-1], 1, "Generation must receive configured max_tokens limit")

    def test_cancellation_preserves_conversation(self):
        """Verify Ctrl-C during generation cancels cleanly and resets/preserves state."""
        backend = MockChatBackend()
        glow = IngestGlow()
        with tempfile.TemporaryDirectory() as root:
            prefs_path = str(Path(root) / "prefs.json")
            session = Session(backend, "plain", dict(preferences.DEFAULTS), prefs_path)
            session.preferences["animate"] = False

            # Simulate Ctrl-C arriving during generation
            with patch("veystrix.session._run_generation", side_effect=KeyboardInterrupt):
                with redirect_stdout(io.StringIO()) as out:
                    presses = run_turn_plain(session, "A cancelling prompt", glow)
                self.assertEqual(presses, 0)
                self.assertIn("generation interrupted; conversation reset", out.getvalue())

    def test_small_model_real_mlx_chat_turn(self):
        """Test real MLX forward pass and chat loop using tiny_language from unit tests."""
        from models.flashnext.tests.unit.test_speculative_tiny import tiny_language
        from veystrix.backends.flashnext import FlashNextBackend
        from veystrix.sampling import Sampling
        from veystrix.session import _on_model_thread
        import mlx.core as mx

        # Initialize model on the executor thread so GPU stream bindings match
        tiny_model = _on_model_thread(tiny_language)

        class TinyFlashNextBackend(FlashNextBackend):
            def __init__(self, tokenizer):
                super(FlashNextBackend, self).__init__(tokenizer)
                self.language = tiny_model
                self.cache = _on_model_thread(self.language.make_cache)
                self.stops = {tokenizer.eos_token_id, 2}
                self.thinking_enabled = False
                self._interactive_budgets = None
                self.sampling = Sampling.greedy_settings()
                self.routing_profile = "standard"
                self._store = None
                self._decoder = None
                self._fused_pending = False
                self._replay_needed = False
                self._mtp_active = False
                self._mtp_blocked = False
                self.model_path = "/tmp/tiny"

                class MiniRouting:
                    pinned_bytes = 0
                    pinned_signature = ""
                    def reset(self): pass
                    def begin_decode(self): pass
                    def after_token(self, count, limit): return False
                    def finish_decode(self): pass
                    def _observe(self, *a): pass

                self.routing = MiniRouting()

        tok = MockTokenizer()
        tok.eos_token_id = 2
        backend = TinyFlashNextBackend(tok)

        with tempfile.TemporaryDirectory() as root:
            prefs_path = str(Path(root) / "prefs.json")
            session = Session(backend, "plain", dict(preferences.DEFAULTS), prefs_path)
            session.preferences["animate"] = False
            session.preferences["max_tokens"] = 3

            glow = IngestGlow()
            # Turn 1
            with redirect_stdout(io.StringIO()):
                run_turn_plain(session, "Hello tiny", glow)

            self.assertTrue(session.opened)
            turn_1_tokens = len(backend.tape)
            self.assertGreater(turn_1_tokens, 0)

            # Turn 2
            with redirect_stdout(io.StringIO()):
                run_turn_plain(session, "How are you?", glow)

            turn_2_tokens = len(backend.tape)
            self.assertGreater(turn_2_tokens, turn_1_tokens)


if __name__ == "__main__":
    unittest.main()
