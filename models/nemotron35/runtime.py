"""Synthetic MLX hybrid inference runtime, not a production checkpoint loader."""
from __future__ import annotations
import mlx.core as mx

class Cancelled(Exception):
    pass

class SyntheticRuntime:
    def __init__(self, model):
        self.model = model
        self.cache = model.make_cache()
        self.tokens = []

    def reset(self):
        self.cache = self.model.make_cache()
        self.tokens = []

    def prefill(self, tokens, chunk_size=16, cancelled=lambda: False):
        if chunk_size <= 0 or not tokens:
            raise ValueError('prefill needs tokens and a positive chunk size')
        logits = None
        try:
            for start in range(0, len(tokens), chunk_size):
                if cancelled():
                    raise Cancelled()
                chunk = tokens[start:start + chunk_size]
                logits = self.model(mx.array([chunk], dtype=mx.int32), cache=self.cache)
                mx.eval(logits)
                self.tokens.extend(chunk)
            return logits
        except BaseException:
            # Neither partial recurrent nor attention state may be resumed after interruption.
            self.reset()
            raise

    def iter_greedy(self, last_logits, count, cancelled=lambda: False):
        if count < 0:
            raise ValueError('negative generation length')
        logits = last_logits
        try:
            for _ in range(count):
                if cancelled():
                    raise Cancelled()
                token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                logits = self.model(mx.array([[token]], dtype=mx.int32), cache=self.cache)
                mx.eval(logits)
                self.tokens.append(token)
                yield token
        except BaseException:
            self.reset()
            raise

    def greedy_decode(self, last_logits, count, cancelled=lambda: False):
        return list(self.iter_greedy(last_logits, count, cancelled=cancelled))
