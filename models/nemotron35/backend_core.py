"""Nemotron engine core: chunked prefill, decode, cancellation and state, no chat framing.

Separated from the shared-backend adapter so it can be tested with tiny
synthetic checkpoints. It owns one live conversation state (Mamba + KV caches).
"""
from __future__ import annotations

import mlx.core as mx

from .runtime import Cancelled


class EngineCore:
    def __init__(self, streamed, *, chunk_tokens: int):
        if chunk_tokens <= 0:
            raise ValueError('chunk_tokens must be positive')
        self.streamed, self.model = streamed, streamed.model
        self.chunk_tokens = chunk_tokens
        self.cache = self.model.make_cache()

    # -- state -------------------------------------------------------------------------------
    def reset(self) -> None:
        self.cache = self.model.make_cache()
        mx.clear_cache()

    @property
    def cached_tokens(self) -> int:
        for entry in self.cache:
            offset = getattr(entry, 'offset', None)
            if offset is not None:
                return int(offset)
        return 0

    def state_arrays(self):
        return [a for entry in self.cache for a in entry.state if a is not None]

    # -- forward -----------------------------------------------------------------------------
    def _step(self, tokens):
        """One forward over ``tokens``; returns last-position logits only (never [T, vocab])."""
        out = self.model.backbone(mx.array([tokens], dtype=mx.int32), cache=self.cache)
        return self.model.lm_head(out[:, -1:, :])[:, -1, :]

    def prefill(self, tokens, *, cancelled=lambda: False, progress=None):
        """Feed ``tokens`` in bounded chunks and return logits for the next token.

        Any interruption leaves recurrent/KV state partially advanced and
        unusable, so the caller must ``reset`` (this method does it itself).
        """
        if not tokens:
            raise ValueError('prefill needs tokens')
        self.streamed.pool.cancel_check = cancelled
        done, logits = 0, None
        try:
            for start in range(0, len(tokens), self.chunk_tokens):
                if cancelled():
                    raise Cancelled()
                chunk = tokens[start:start + self.chunk_tokens]
                last = start + len(chunk) >= len(tokens)
                if last:
                    logits = self._step(chunk)
                    mx.eval(logits, *self.state_arrays())
                else:
                    out = self.model.backbone(mx.array([chunk], dtype=mx.int32), cache=self.cache)
                    mx.eval(out, *self.state_arrays())
                done += len(chunk)
                if progress:
                    progress(done, len(tokens))
            return logits
        except BaseException:
            self.reset()
            raise
        finally:
            self.streamed.pool.cancel_check = lambda: False

    def decode(self, token: int, *, cancelled=lambda: False):
        self.streamed.pool.cancel_check = cancelled
        try:
            logits = self._step([int(token)])
            mx.eval(logits, *self.state_arrays())
            return logits
        except BaseException:
            self.reset()
            raise
        finally:
            self.streamed.pool.cancel_check = lambda: False
