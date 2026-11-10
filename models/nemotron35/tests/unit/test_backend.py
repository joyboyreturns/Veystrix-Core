"""Backend tests on tiny synthetic checkpoints with a stub tokenizer (no real weights or tokenizer)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from models.nemotron35 import admission
from models.nemotron35.tiny_checkpoint import permissive_config_check, write_checkpoint
from veystrix.backends.base import GenerationCancelled
from veystrix.backends.nemotron35 import NemotronBackend

_dir = _ref = None


class StubTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [4 + (ord(c) % 56) for c in text]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(self, messages, **kw):
        return "".join(m["content"] for m in messages)


def machine(**kw):
    base = dict(total=16 * 2**30, available=12 * 2**30, swap_used=0, pressure=1, gpu_limit=10 * 2**30)
    base.update(kw)
    return admission.Machine(**base)


def setUpModule():
    global _dir, _ref
    _dir = tempfile.mkdtemp(prefix="nemotron-backend-")
    _ref = write_checkpoint(Path(_dir) / "ck")


def tearDownModule():
    shutil.rmtree(_dir, ignore_errors=True)


def make(**kw):
    args = dict(authorized=True, config_check=permissive_config_check, machine=machine(), tokenizer=StubTokenizer(),
                pool_bytes=64 * 1024, chunk_tokens=4, session_dir=str(Path(_dir) / "sessions"),
                prefix_dir=str(Path(_dir) / "prefix"), snapshot_tokens=8)
    args.update(kw)
    backend = NemotronBackend(Path(_dir) / "ck", **args)
    backend._current_machine = lambda: machine()   # no live vm_stat in model-free tests
    return backend


def reference_greedy(tokens, count):
    cache = _ref.make_cache()
    logits = _ref(mx.array([tokens]), cache=cache)
    out = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = _ref(mx.array([[token]]), cache=cache)
    return out


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.b = make()

    def tearDown(self):
        self.b.close()

    def run_turn(self, text, n=5, **kw):
        self.b.append_text(text)
        before = len(self.b.tape)
        _, stats = self.b.generate(n, **kw)
        return self.b.tape[before + 0:], stats

    def test_requires_authorization(self):
        with self.assertRaises(SystemExit):
            NemotronBackend(Path(_dir) / "ck", authorized=False, config_check=permissive_config_check,
                            machine=machine(), tokenizer=StubTokenizer())

    def test_strict_config_refuses_tiny(self):
        with self.assertRaises(Exception):
            NemotronBackend(Path(_dir) / "ck", authorized=True, machine=machine(), tokenizer=StubTokenizer())

    def test_greedy_matches_reference_and_tape_is_consistent(self):
        prompt = self.b.encode("hello nemotron")
        self.b.append_tokens(prompt)
        _, stats = self.b.generate(6)
        produced = self.b.tape[len(prompt):]
        self.assertEqual(len(produced), stats.tokens)
        self.assertEqual(produced, reference_greedy(prompt, len(produced))[:len(produced)])
        self.assertTrue(self.b.check_invariant())
        self.assertEqual(self.b.cache_tokens, len(self.b.tape))
        self.assertGreater(stats.expert_bytes_read + stats.pool_hits, 0)

    def test_incremental_turn_equals_cold_replay(self):
        self.b.append_text("first question")
        self.b.generate(4)
        self.b.append_text(" and then more")
        self.b.generate(4)
        tape = list(self.b.tape)
        other = make()
        try:
            other.append_tokens(tape[:-4])
            other.generate(4)
            self.assertEqual(other.tape, tape)
        finally:
            other.close()

    def test_cancel_during_decode_keeps_state_usable(self):
        self.b.append_text("cancel me")
        seen = []
        with self.assertRaises(GenerationCancelled):
            self.b.generate(20, on_decode_token=lambda t, p: seen.append(t), should_cancel=lambda: len(seen) >= 2)
        self.assertEqual(self.b.cache_tokens, len(self.b.tape))
        self.b.append_text(" go on")
        self.b.generate(3)
        self.assertTrue(self.b.check_invariant())

    def test_cancel_during_prefill_forces_clean_replay(self):
        self.b.append_text("a fairly long prompt to span several chunks of four tokens")
        with self.assertRaises(GenerationCancelled):
            self.b.generate(3, should_cancel=lambda: True)
        self.assertTrue(self.b._replay_needed)
        self.assertEqual(self.b.cache_tokens, 0)
        prompt = list(self.b.tape)
        self.b.append_text("")  # nothing new; replay must rebuild from the tape
        self.b.pending = self.b.encode("?")
        self.b.generate(3)
        self.assertEqual(self.b.tape[:len(prompt)], prompt)
        self.assertEqual(self.b.cache_tokens, len(self.b.tape))

    def test_admission_refusals(self):
        self.b._current_machine = lambda: machine(swap_used=3 * 2**30)
        self.b.append_text("x")
        with self.assertRaises(admission.AdmissionRefused):
            self.b.generate(2)
        self.assertEqual(self.b.cache_tokens, 0)

    def test_context_limit_enforced(self):
        self.b.append_tokens([5] * (self.b.context_limit + 1))
        with self.assertRaises(admission.AdmissionRefused):
            self.b.generate(1)

    def test_session_roundtrip_and_isolation(self):
        self.b.append_text("persist this")
        self.b.generate(3)
        self.assertIn("saved", self.b.save_session("s1"))
        self.b.append_text(" next")
        self.b.generate(3)
        live = list(self.b.tape)
        other = make()
        try:
            self.assertIn("loaded", other.load_session("s1"))
            other.append_text(" next")
            other.generate(3)
            self.assertEqual(other.tape, live)
            self.assertIn("s1", other.list_sessions())
        finally:
            other.close()

    def test_session_rejects_corruption_bad_names_and_other_identity(self):
        self.b.append_text("persist this")
        self.b.generate(2)
        self.b.save_session("bad")
        path = next(Path(_dir, "sessions").rglob("bad.safetensors"))
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))
        other = make()
        try:
            self.assertIn("cannot load", other.load_session("bad"))
            self.assertEqual(other.tape, [])
            with self.assertRaises(ValueError):
                other.save_session("../escape")
            other._identity = dict(other._session_identity(), tokenizer="different")
            self.assertIn("cannot load", other.load_session("bad"))
        finally:
            other.close()

    def test_think_budget_forces_close_token(self):
        self.b.thinking_enabled = True
        self.b._interactive_budgets = (None, 3)
        self.b.encode = lambda text: [13] if text == "</think>" else StubTokenizer().encode(text)
        self.b.append_text("think about it")
        pieces = []
        self.b.generate(10, on_decode_token=lambda t, p: pieces.append((t, p)))
        self.assertIn(13, [t for t, _ in pieces])
        self.assertEqual(next(p for t, p in pieces if t == 13).endswith("</think>"), True)


if __name__ == "__main__":
    unittest.main()
