"""Nemotron engine core: chunked prefill, decode, cancellation and state, no chat framing.

Separated from the shared-backend adapter so it can be tested with tiny
synthetic checkpoints. It owns one live conversation state (Mamba + KV caches).
"""
from __future__ import annotations

from copy import copy
import os

import mlx.core as mx

from .runtime import Cancelled
from .attention import AttentionCache


class EngineCore:
    def __init__(self, streamed, *, chunk_tokens: int, layer_major_tokens: int = 0, int8_kv: bool = False):
        if chunk_tokens <= 0 or layer_major_tokens < 0:
            raise ValueError('chunk_tokens must be positive and layer_major_tokens non-negative')
        self.streamed, self.model = streamed, streamed.model
        self.chunk_tokens = chunk_tokens
        self.layer_major_tokens = layer_major_tokens     # 0: chunk-major prefill; else segment size in tokens
        self.int8_kv = int8_kv
        self.mtp = None
        self._mtp_anchor = None
        self._capture_mtp = False
        self.bind_cache(self._make_cache())

    def _make_cache(self):
        from mlx_lm.models.cache import KVCache
        return [AttentionCache() if self.int8_kv and isinstance(entry, KVCache) else entry
                for entry in self.model.make_cache()]

    # -- state -------------------------------------------------------------------------------
    def reset(self) -> None:
        self.bind_cache(self._make_cache())
        mx.clear_cache()

    @property
    def cached_tokens(self) -> int:
        for entry in self.cache:
            offset = getattr(entry, 'offset', None)
            if offset is not None:
                return int(offset)
        return 0

    def bind_cache(self, cache):
        self.cache = cache
        for entry in cache:
            if isinstance(entry, AttentionCache):
                entry.cancelled = lambda: self.streamed.pool.cancel_check()

    def state_arrays(self):
        return [a for entry in self.cache for a in entry.state if a is not None]

    # -- forward -----------------------------------------------------------------------------
    def _step(self, tokens):
        """One forward over ``tokens``; returns last-position logits only (never [T, vocab])."""
        out = self._forward(tokens)
        return self.model.lm_head(out[:, -1:, :])[:, -1, :]

    def begin_mtp_turn(self, enabled):
        self._capture_mtp = enabled and self.mtp is not None
        self._mtp_anchor = None
        if self.mtp is not None:
            self.mtp.reset()

    def _forward(self, tokens):
        if not self._capture_mtp:
            return self.model.backbone(mx.array([tokens], dtype=mx.int32), cache=self.cache)
        rows = self._prenorm(tokens)
        self._advance_mtp(rows, tokens)
        return self.model.backbone.norm_f(rows)

    def _prenorm(self, tokens):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        backbone = self.model.backbone
        rows = backbone.embeddings(mx.array([tokens], dtype=mx.int32))
        attention = create_attention_mask(rows, self.cache[backbone.fa_idx])
        ssm = create_ssm_mask(rows, self.cache[backbone.ssm_idx])
        entries = iter(self.cache)
        for layer in backbone.layers:
            if layer.block_type == "*":
                rows = layer(rows, mask=attention, cache=next(entries))
            elif layer.block_type == "M":
                rows = layer(rows, mask=ssm, cache=next(entries))
            else:
                rows = layer(rows, mask=ssm)
        return rows

    def _advance_mtp(self, rows, tokens):
        from .mtp import HISTORY_LIMIT

        for start in range(0, len(tokens), 128):
            part = rows[0, start:start + 128]
            ids = tokens[start:start + 128]
            if self._mtp_anchor is None:
                history, followers = part[:-1], ids[1:]
            else:
                history = mx.concatenate((self._mtp_anchor, part[:-1]), axis=0)
                followers = ids
            if followers:
                if self.mtp.kv.offset + len(followers) > HISTORY_LIMIT:
                    self.mtp.reset()
                self.mtp.forward(history, mx.array(followers, dtype=mx.int32))
            self._mtp_anchor = part[-1:]
            mx.eval(self._mtp_anchor)

    def _segment_layer_major(self, chunks, *, last: bool, cancelled, anchor_cache=None, verify=False, keep_rows=False):
        """Run ``tokens`` through the network one layer at a time: every layer processes all chunks of the segment
        before the next layer starts, so a layer's routed experts are read from SSD once per segment instead of once
        per chunk. Per token the arithmetic is identical to chunk-major prefill with the same chunk size: every layer
        sees the same chunks in the same order against the same per-layer state. Returns the last-position logits
        when ``last`` (else None). The per-chunk hidden states of the segment are the only extra resident data."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        backbone, pool = self.model.backbone, self.streamed.pool
        hidden = [backbone.embeddings(mx.array([chunk], dtype=mx.int32)) for chunk in chunks]
        moe = [i for i, layer in enumerate(backbone.layers) if layer.block_type == 'E']
        following = dict(zip(moe, moe[1:]))
        counter = 0
        seed = not verify and os.environ.get("VEYSTRIX_NEMOTRON_PREFILL_SEED_DECODE") == "1"
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
                    pool.begin_layer(index, seed=seed)
                    if not verify and anchor_cache is None and index in following:
                        pool.prefetch_layer(following[index])     # overlap the next layer's reads with this compute
                for j, h in enumerate(hidden):
                    if layer.block_type == '*':
                        mask = create_attention_mask(h, cache)
                    elif layer.block_type == 'M':
                        mask = create_ssm_mask(h, cache)
                    else:
                        mask = None
                    hidden[j] = layer(h, mask=mask, cache=cache)
                    if cache is not None and j == len(hidden) - 1:
                        mx.eval(hidden[j], *[a for a in cache.state if a is not None])
                    else:
                        mx.eval(hidden[j])
                    if j == 0 and cache is not None and anchor_cache is not None:
                        snapshot = copy(cache)
                        snapshot.state = [mx.array(a) for a in cache.state]
                        mx.eval(*snapshot.state)
                        anchor_cache.append(snapshot)
                if layer.block_type == 'E':
                    pool.end_layer(seed=seed)
            if self._capture_mtp and not verify:
                for rows, tokens in zip(hidden, chunks, strict=True):
                    if cancelled():
                        raise Cancelled()
                    self._advance_mtp(rows, tokens)
            if not last:
                return None
            if verify:
                # Preserve singleton head arithmetic and avoid a [width, vocab] allocation.
                predictions = [int(mx.argmax(self.model.lm_head(backbone.norm_f(row)), axis=-1).item())
                               for row in hidden]
                return (predictions, hidden) if keep_rows else predictions
            out = backbone.norm_f(hidden[-1][:, -1:, :])
            logits = self.model.lm_head(out)[:, -1, :]
            mx.eval(logits)
            return logits
        finally:
            pool.end_layer()
            pool.speculate = True

    def prefill(self, tokens, *, cancelled=lambda: False, progress=None, boundary=None, emit_logits=True, anchor=None, save_anchor=None):
        """Feed ``tokens`` in bounded chunks and return logits for the next token.

        Any interruption leaves recurrent/KV state partially advanced and
        unusable, so the caller must ``reset`` (this method does it itself).
        """
        if not tokens:
            raise ValueError('prefill needs tokens')
        self.streamed.pool.cancel_check = cancelled
        done, logits = 0, None
        try:
            if anchor is not None:
                if not (save_anchor and 0 < anchor <= self.chunk_tokens
                        and 0 < len(tokens) - anchor <= self.chunk_tokens):
                    raise ValueError("anchored prefill requires two bounded chunks and a checkpoint callback")
                frozen = []
                logits = self._segment_layer_major(
                    (tokens[:anchor], tokens[anchor:]), last=emit_logits, cancelled=cancelled, anchor_cache=frozen)
                save_anchor(anchor, frozen)
                if progress:
                    progress(len(tokens), len(tokens))
                if boundary:
                    boundary(len(tokens))
                return logits
            if self.layer_major_tokens and len(tokens) > self.chunk_tokens:
                # Segments are whole numbers of chunks so each layer sees exactly the chunk-major chunk boundaries.
                seg = max(self.chunk_tokens, self.layer_major_tokens // self.chunk_tokens * self.chunk_tokens)
                for start in range(0, len(tokens), seg):
                    if cancelled():
                        raise Cancelled()
                    part = tokens[start:start + seg]
                    last = start + len(part) >= len(tokens)
                    chunks = [part[i:i + self.chunk_tokens] for i in range(0, len(part), self.chunk_tokens)]
                    result = self._segment_layer_major(chunks, last=last and emit_logits, cancelled=cancelled)
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
                if last and emit_logits:
                    logits = self._step(chunk)
                    mx.eval(logits, *self.state_arrays())
                else:
                    out = self._forward(chunk)
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

    def decode(self, token: int, *, cancelled=lambda: False, sampler=None, emit_logits=True):
        self.streamed.pool.cancel_check = cancelled
        try:
            if not emit_logits:
                hidden = self._forward([int(token)])
                mx.eval(hidden, *self.state_arrays())
                return None
            logits = self._step([int(token)])
            result = logits if sampler is None else sampler(logits)
            mx.eval(result, *self.state_arrays())
            return result
        except BaseException:
            self.reset()
            raise
        finally:
            self.streamed.pool.cancel_check = lambda: False

    def verify_decode(self, token, drafts, *, cancelled):
        """Verify a short greedy draft with singleton operations and layer-local expert reuse."""
        from .state import clone_cache

        snapshot = clone_cache(self.cache)
        self.streamed.pool.cancel_check = cancelled
        try:
            values = [token, *drafts]
            verified = self._segment_layer_major(
                [[value] for value in values], last=True, cancelled=cancelled, verify=True,
                keep_rows=self._capture_mtp,
            )
            predictions, rows = verified if self._capture_mtp else (verified, None)
            if predictions[:-1] == drafts:
                if rows is not None:
                    self._advance_mtp(mx.concatenate(rows, axis=1), values)
                return values, predictions[-1]
            self.bind_cache(snapshot)
            committed = [token]
            following = int(mx.argmax(self.decode(token, cancelled=cancelled), axis=-1).item())
            for draft in drafts:
                if draft != following:
                    break
                committed.append(draft)
                following = int(mx.argmax(self.decode(draft, cancelled=cancelled), axis=-1).item())
            return committed, following
        except BaseException:
            self.bind_cache(snapshot)
            raise
        finally:
            self.streamed.pool.cancel_check = lambda: False

    def native_mtp_tokens(self, first, horizon, stops, cancelled, depth):
        produced, value = 0, int(first.item())
        while produced < horizon and value not in stops:
            if cancelled():
                raise Cancelled()
            drafts = []
            offset = self.mtp.kv.offset
            hidden, follower = self._mtp_anchor, value
            try:
                for _ in range(min(depth, horizon - produced - 1)):
                    follower, hidden = self.mtp.forward(hidden, mx.array([follower], dtype=mx.int32), predict=True)
                    if follower in stops:
                        break
                    drafts.append(follower)
            finally:
                self.mtp.rewind(offset)
            if drafts:
                committed, value = self.verify_decode(value, drafts, cancelled=cancelled)
                for token in committed:
                    if cancelled():
                        raise Cancelled()
                    yield token
                produced += len(committed)
            else:
                yield value
                produced += 1
                logits = self.decode(value, cancelled=cancelled, emit_logits=produced < horizon)
                if logits is not None:
                    value = int(mx.argmax(logits, axis=-1).item())

    def prompt_lookup_tokens(self, history, first, horizon, stops, cancelled):
        from models.shared.prompt_lookup import PromptLookup, ThroughputGate
        from time import perf_counter

        lookup, gate = PromptLookup(history), ThroughputGate()
        produced, value = 0, int(first.item())
        while produced < horizon and value not in stops:
            if cancelled():
                raise Cancelled()
            lookup.append(value)
            drafts = lookup.propose(min(4, horizon - produced - 1)) if gate.allow() else []
            for index, draft in enumerate(drafts):
                if draft in stops:
                    drafts = drafts[:index]
                    break
            if not drafts:
                yield value
                produced += 1
                began = perf_counter()
                logits = self.decode(value, cancelled=cancelled, emit_logits=produced < horizon)
                if logits is not None:
                    value = int(mx.argmax(logits, axis=-1).item())
                gate.observe(False, 1, perf_counter() - began)
                continue
            began = perf_counter()
            committed, value = self.verify_decode(value, drafts, cancelled=cancelled)
            gate.observe(True, len(committed), perf_counter() - began)
            for output in committed:
                if cancelled():
                    raise Cancelled()
                yield output
            produced += len(committed)
            for output in committed[1:]:
                lookup.append(output)
