"""Experimental asymmetric/tiered affine KV. Only gather QSA may consume it.

No model weights, raw indexer keys or GDN states are transformed. History is
never dequantized by attention: only the selected rows are unpacked. INT4 is
written directly from original activations, avoiding an INT8-to-INT4 cascade.
Only the bounded recent window has a second representation.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_vlm.models.qwen4_exp.language import QSAKVCache

PAGE = 256
HISTORY_GROWTH = 4096
SCHEMA = 1


def _arrays(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _slice(value, start, end):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(a[..., start:end, :] for a in value)
    return value[..., start:end, :]


def _unpack(value, bits, group):
    if bits == 16:
        return value
    return mx.dequantize(*[mx.contiguous(a) for a in value], group_size=group, bits=bits)


def _append(storage, value, offset, bits, group, growth=PAGE):
    packed = value if bits == 16 else mx.quantize(value, group_size=group, bits=bits)
    end = offset + value.shape[-2]
    capacity = ((end + growth - 1) // growth) * growth
    parts, incoming = _arrays(storage), _arrays(packed)
    if not parts:
        parts = [mx.zeros((*a.shape[:-2], capacity, a.shape[-1]), a.dtype) for a in incoming]
    elif parts[0].shape[-2] < end:
        parts = [mx.concatenate((a[..., :offset, :],
                 mx.zeros((*a.shape[:-2], capacity - offset, a.shape[-1]), a.dtype)), axis=-2)
                 for a in parts]
    for a, new in zip(parts, incoming):
        a[..., offset:end, :] = new
    return parts[0] if bits == 16 else tuple(parts)


class MixedPrecisionQSAKVCache(QSAKVCache):
    """K16/V4, K8/V4, or history K4/V4 + recent K8/V8.

    Recent capacity is window + at most 255 tokens of page-alignment slack.
    There is no full-length shadow BF16 cache. All incoming chunks must be
    bounded by the prefill planner; attention only receives selected rows.
    """
    def __init__(self, mode="tiered4", group_size=64, recent_tokens=8192):
        super().__init__()
        if mode not in ("k8v4", "k16v4", "tiered4"):
            raise ValueError("unsupported mixed KV mode")
        if group_size not in (32, 64):
            raise ValueError("mixed KV group size must be 32 or 64")
        if recent_tokens < PAGE or recent_tokens % PAGE:
            raise ValueError("recent-token window must be a positive multiple of 256")
        self.mode, self.group_size, self.recent_tokens = mode, group_size, recent_tokens
        self.key_bits = {"k16v4": 16, "k8v4": 8, "tiered4": 4}[mode]
        self.value_bits = 4
        self.recent_keys = self.recent_values = None
        self.history_tokens = 0

    def update_and_fetch(self, keys, values):
        end = self.offset + keys.shape[-2]
        self.keys = _append(self.keys, keys, self.offset, self.key_bits, self.group_size, HISTORY_GROWTH)
        self.values = _append(self.values, values, self.offset, 4, self.group_size, HISTORY_GROWTH)
        if self.mode != "tiered4":
            self.history_tokens = end
        else:
            recent = self.offset - self.history_tokens
            self.recent_keys = _append(self.recent_keys, keys, recent, 8, self.group_size)
            self.recent_values = _append(self.recent_values, values, recent, 8, self.group_size)
            recent += keys.shape[-2]
            aged = max(0, (recent - self.recent_tokens) // PAGE * PAGE)
            if aged:
                # Drop the expired INT8 rows. Their already-written INT4
                # rows become visible; no historical row is requantized.
                self.history_tokens += aged
                keep = mx.arange(aged, recent)
                self.recent_keys = tuple(mx.take(a, keep, axis=-2) for a in self.recent_keys)
                self.recent_values = tuple(mx.take(a, keep, axis=-2) for a in self.recent_values)
        self.offset = end
        # Projection helper requires a pair, but only gather_selected consumes
        # this cache. Returning incoming tensors avoids a history-sized view.
        return keys, values

    def gather_selected(self, indices):
        from .qsa_chunk import _gather_rows

        def gather(storage, at, bits):
            if bits == 16:
                return _gather_rows(storage, at)
            return _unpack(tuple(_gather_rows(a, at) for a in storage), bits, self.group_size)

        if self.mode != "tiered4":
            return gather(self.keys, indices, self.key_bits), gather(self.values, indices, 4)
        recent_n = self.offset - self.history_tokens
        if recent_n == 0:
            return gather(self.keys, indices, 4), gather(self.values, indices, 4)
        at = mx.clip(indices - self.history_tokens, 0, recent_n - 1)
        rk, rv = gather(self.recent_keys, at, 8), gather(self.recent_values, at, 8)
        if self.history_tokens == 0:
            return rk, rv
        old_at = mx.minimum(indices, self.history_tokens - 1)
        hk, hv = gather(self.keys, old_at, 4), gather(self.values, old_at, 4)
        old = (indices < self.history_tokens).reshape(-1, 1, indices.shape[-1], 1)
        return mx.where(old, hk, rk), mx.where(old, hv, rv)

    @property
    def state(self):
        # MLX evaluation / generation lifecycle can traverse this nested state.
        return (self.keys, self.values, self.recent_keys, self.recent_values,
                self.index_keys, self.index_position_ids)

    @state.setter
    def state(self, value):
        raise ValueError("mixed KV restoration requires its versioned snapshot")

    @property
    def nbytes(self):
        return sum(a.nbytes for item in self.state for a in _arrays(item))

    def trim(self, n):
        n = min(max(0, n), self.offset)
        self.offset -= n
        history = min(self.history_tokens, self.offset)
        self.keys = _slice(self.keys, 0, self.offset)
        self.values = _slice(self.values, 0, self.offset)
        self.recent_keys = _slice(self.recent_keys, 0, self.offset - history)
        self.recent_values = _slice(self.recent_values, 0, self.offset - history)
        self.history_tokens = history
        if self.index_keys is not None:
            self.index_keys = self.index_keys[:, :self.offset]
            self.index_position_ids = self.index_position_ids[..., :self.offset]
        return n

    def empty(self):
        return self.offset == 0

    def extract(self, idx):
        raise ValueError("mixed KV does not support batched cache extraction")

    def filter(self, batch_indices):
        raise ValueError("mixed KV does not support batched cache filtering")

    def to_quantized(self, *args, **kwargs):
        raise ValueError("mixed KV is already compressed")

    def snapshot(self, history_start=0, index_start=0, *, own=True):
        """Packed immutable primary delta, raw indexer delta, and full recent tier.

        The INT4 primary representation is written once, so its delta starts
        at the parent's end token. Only the leaf's recent tier is used.
        """
        if not 0 <= history_start <= self.offset or not 0 <= index_start <= self.offset:
            raise ValueError("invalid mixed KV snapshot delta start")
        result = {"config": mx.array([SCHEMA, self.key_bits, 4, self.group_size,
                   self.recent_tokens, self.offset, self.history_tokens,
                   int(self.mode == "tiered4")], dtype=mx.int64)}
        for name in ("keys", "values", "recent_keys", "recent_values"):
            n = self.offset if name in ("keys", "values") else self.offset - self.history_tokens
            start = history_start if name in ("keys", "values") else 0
            if start != n:
                for i, a in enumerate(_arrays(getattr(self, name))):
                    # take owns its small delta buffer even when a contiguous
                    # slice would retain a view of the whole live allocation.
                    result[f"{name}_{i}"] = (mx.take(a, mx.arange(start, n), axis=-2)
                                               if own else a[..., start:n, :])
        for name in ("index_keys", "index_position_ids"):
            a = getattr(self, name)
            if a is not None:
                if own:
                    result[name] = mx.take(a, mx.arange(index_start, self.offset),
                                           axis=1 if name == "index_keys" else -1)
                else:
                    result[name] = (a[:, index_start:self.offset] if name == "index_keys"
                                    else a[..., index_start:self.offset])
        return result

    def restore_snapshot(self, tensors, tokens):
        config = tensors.get("config")
        if config is None or config.dtype != mx.int64 or config.shape != (8,):
            raise ValueError("invalid mixed KV configuration")
        version, kb, vb, group, recent, offset, history, tiered = config.tolist()
        if (version, kb, vb, group, recent, tiered) != (
                SCHEMA, self.key_bits, 4, self.group_size, self.recent_tokens, int(self.mode == "tiered4")):
            raise ValueError("incompatible mixed KV configuration")
        if (offset != tokens or not 0 <= history <= offset or
                (not tiered and history != offset) or (tiered and offset-history >= recent+PAGE)):
            raise ValueError("invalid mixed KV offsets")
        expected = {"config", "index_keys", "index_position_ids"}
        restored = {}
        for name in ("keys", "values", "recent_keys", "recent_values"):
            bits = (kb if name == "keys" else 4) if name in ("keys", "values") else 8
            n = offset if name in ("keys", "values") else offset - history
            count = 1 if bits == 16 else 3
            names = [f"{name}_{i}" for i in range(count)]
            present = any(k in tensors for k in names)
            if not n and not present:
                restored[name] = None
                continue
            expected.update(names)
            parts = [tensors.get(k) for k in names]
            if any(a is None or a.ndim != 4 or a.shape[-2] != n for a in parts):
                raise ValueError(f"invalid mixed KV tensor: {name}")
            if bits != 16:
                data, scales, biases = parts
                if (data.dtype != mx.uint32 or scales.dtype != biases.dtype or
                    scales.shape != biases.shape or data.shape[:-1] != scales.shape[:-1] or
                    data.shape[-1] * (32 // bits) != scales.shape[-1] * group):
                    raise ValueError(f"invalid packed mixed KV layout: {name}")
            restored[name] = parts[0][...] if bits == 16 else tuple(a[...] for a in parts)
        if set(tensors) != expected:
            raise ValueError("unexpected or missing mixed KV fields")
        ik, positions = tensors["index_keys"], tensors["index_position_ids"]
        if ik.ndim != 3 or ik.shape[1] != offset or positions.shape[-1] != offset:
            raise ValueError("invalid mixed KV indexer lengths")
        for name, value in restored.items():
            setattr(self, name, value)
        self.offset, self.history_tokens = offset, history
        self.index_keys, self.index_position_ids = ik[...], positions[...]
