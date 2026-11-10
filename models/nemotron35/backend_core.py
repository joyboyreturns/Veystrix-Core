"""Nemotron engine core: chunked prefill, decode, cancellation and state, no chat framing.

Separated from the shared-backend adapter so it can be tested with tiny
synthetic checkpoints. It owns one live conversation state (Mamba + KV caches).
"""
from __future__ import annotations

import mlx.core as mx

from .runtime import Cancelled


class EngineCore:
    def __init__(self, streamed, *, chunk_tokens: int, layer_major_tokens: int = 0):
        if chunk_tokens <= 0 or layer_major_tokens < 0:
            raise ValueError('chunk_tokens must be positive and layer_major_tokens non-negative')
        self.streamed, self.model = streamed, streamed.model
        self.chunk_tokens = chunk_tokens
        self.layer_major_tokens = layer_major_tokens     # 0: chunk-major prefill; else segment size in tokens
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

    def _segment_layer_major(self, tokens, *, last: bool, cancelled):
        """Run ``tokens`` through the network one layer at a time: every layer processes all chunks of the segment
        before the next layer starts, so a layer's routed experts are read from SSD once per segment instead of once
        per chunk. Per token the arithmetic is identical to chunk-major prefill with the same chunk size: every layer
        sees the same chunks in the same order against the same per-layer state. Returns the last-position logits
        when ``last`` (else None). The per-chunk hidden states of the segment are the only extra resident data."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        backbone, pool = self.model.backbone, self.streamed.pool
        size = self.chunk_tokens
        hidden = [backbone.embeddings(mx.array([tokens[i:i + size]], dtype=mx.int32))
                  for i in range(0, len(tokens), size)]
        moe = [i for i, layer in enumerate(backbone.layers) if layer.block_type == 'E']
        following = dict(zip(moe, moe[1:]))
        counter = 0
        pool.speculate = False
        try:
            for index, layer in enumerate(backbone.layers):
                if cancelled():
                    raise Cancelled()
                cache = None
                if layer.block_type in ('M', '*'):
                    cache = self.cache[counter]
                    counter += 1
                if layer.block_type == 'E':
                    pool.begin_layer(index)
                    if index in following:
                        pool.prefetch_layer(following[index])     # overlap the next layer's reads with this compute
                for j, h in enumerate(hidden):
                    if layer.block_type == '*':
                        mask = create_attention_mask(h, cache)
                    elif layer.block_type == 'M':
                        mask = create_ssm_mask(h, cache)
                    else:
                        mask = None
                    hidden[j] = layer(h, mask=mask, cache=cache)
                    mx.eval(hidden[j])
                if cache is not None:
                    mx.eval(*[a for a in cache.state if a is not None])
                if layer.block_type == 'E':
                    pool.end_layer()
            if not last:
                return None
            out = backbone.norm_f(hidden[-1][:, -1:, :])
            logits = self.model.lm_head(out)[:, -1, :]
            mx.eval(logits)
            return logits
        finally:
            pool.end_layer()
            pool.speculate = True

    def prefill(self, tokens, *, cancelled=lambda: False, progress=None, boundary=None):
        """Feed ``tokens`` in bounded chunks and return logits for the next token.

        Any interruption leaves recurrent/KV state partially advanced and
        unusable, so the caller must ``reset`` (this method does it itself).
        """
        if not tokens:
            raise ValueError('prefill needs tokens')
        self.streamed.pool.cancel_check = cancelled
        done, logits = 0, None
        try:
            if self.layer_major_tokens and len(tokens) > self.chunk_tokens:
                # Segments are whole numbers of chunks so each layer sees exactly the chunk-major chunk boundaries.
                seg = max(self.chunk_tokens, self.layer_major_tokens // self.chunk_tokens * self.chunk_tokens)
                for start in range(0, len(tokens), seg):
                    if cancelled():
                        raise Cancelled()
                    part = tokens[start:start + seg]
                    last = start + len(part) >= len(tokens)
                    result = self._segment_layer_major(part, last=last, cancelled=cancelled)
                    if last:
                        logits = result
                    done += len(part)
                    if progress:
                        progress(done, len(tokens))
                    if boundary:
                        boundary(done)
                return logits
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
                if boundary:          # state now holds exactly the tokens processed so far
                    boundary(done)
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
