"""Synthetic-only Nemotron adapter; production model loading is not implemented.

This deliberately does not register a real model in the CLI or API. It exercises
real MLX hybrid state and SSD expert execution without pretending that the toy
fixture has the model's tokenizer, chat template, tool format, or weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from macqwen.backends.base import GenerationCancelled
from models.nemotron35.fixture import load_fixture
from models.nemotron35.runtime import SyntheticRuntime, Cancelled

@dataclass
class SyntheticStats:
    prompt_tokens: int
    tokens: int
    prefill_seconds: float
    seconds: float
    finish: str = 'length'
    @property
    def rate(self):
        return self.tokens / self.seconds if self.seconds else 0.0
    @property
    def prompt_rate(self):
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

class SyntheticBackend:
    """Token-level test adapter. No real checkpoint, model name, or chat API."""
    def __init__(self, fixture_dir):
        self.runtime = SyntheticRuntime(load_fixture(fixture_dir))
        self.pending = []
        self.tape = self.runtime.tokens
        self.thinking_enabled = False

    def append_tokens(self, tokens):
        tokens = list(tokens)
        if any(not isinstance(t, int) or not 0 <= t < 32 for t in tokens):
            raise ValueError('synthetic fixture token outside vocabulary')
        self.pending.extend(tokens)

    def common_prefix(self, tokens):
        return next((i for i, pair in enumerate(zip(self.tape, tokens)) if pair[0] != pair[1]),
                    min(len(self.tape), len(tokens)))

    def reset(self):
        self.runtime.reset()
        self.tape = self.runtime.tokens
        self.pending = []

    def generate(self, max_tokens, out=None, on_prefilled=None, on_prefill_progress=None,
                 on_decode_token=None, should_cancel=None):
        if not self.pending or max_tokens < 0:
            raise ValueError('synthetic generation requires pending input and a finite output limit')
        pending = list(self.pending)
        try:
            started = time.monotonic()
            if on_prefill_progress:
                on_prefill_progress(0, len(pending))
            logits = self.runtime.prefill(pending, cancelled=should_cancel or (lambda: False))
            self.pending.clear()
            prefill = time.monotonic() - started
            if on_prefill_progress:
                on_prefill_progress(len(pending), len(pending))
            if on_prefilled:
                on_prefilled()
            started = time.monotonic()
            ids, pieces = [], []
            for token in self.runtime.iter_greedy(logits, max_tokens, cancelled=should_cancel or (lambda: False)):
                piece = (" " if ids else "") + str(token)
                ids.append(token)
                pieces.append(piece)
                if on_decode_token:
                    on_decode_token(token, piece)
                if out:
                    out(piece)
            return ''.join(pieces), SyntheticStats(len(pending), len(ids), prefill, time.monotonic() - started)
        except Cancelled as exc:
            self.reset()
            raise GenerationCancelled() from exc
        except BaseException:
            self.reset()
            raise
