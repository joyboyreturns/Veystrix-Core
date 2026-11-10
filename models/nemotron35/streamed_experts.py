"""SSD-streamed routed experts for the stacked mlx-community affine-4-bit layout.

``switch_mlp.fc1`` / ``fc2`` store all 128 experts stacked along the leading
axis. A read fetches only the six contiguous per-expert slices (weight, scales,
biases for each projection) with ``pread``; the full stacked tensor is never
mapped. Exactly the router-selected experts are computed (top-6 of 128) and
the caller adds the shared expert, so no expert is skipped or approximated.

Residency is bounded by a byte-budgeted LRU shared by all layers. Each
expert's result is evaluated before the next is loaded, so an evicted expert
is released rather than kept alive by a lazy graph. The bound covers pooled
experts plus one in-flight expert; it does not cover activations or the MLX
allocator cache, which the memory admission module budgets separately.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .runtime import Cancelled

_NP = {'U32': np.uint32, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}
_PARTS = ('weight', 'scales', 'biases')


class ExpertReadError(RuntimeError):
    """A shard changed, was truncated, or returned a short read."""


class StackedExpertReader:
    """Validated per-expert slice reads from stacked ``switch_mlp`` tensors."""

    def __init__(self, directory, index: dict, *, layers, expert_count: int, hidden: int,
                 moe_dim: int, group_size: int):
        self.directory = Path(directory)
        self.expert_count = expert_count
        self.group_size = group_size
        self._fds: dict[str, int] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.reads = 0
        # Resolve and validate every descriptor once, up front (fail closed).
        self._slices: dict[int, list] = {}
        for layer in layers:
            base = f'backbone.layers.{layer}.mixer.switch_mlp'
            entries = []
            for projection, rows, cols in (('fc1', moe_dim, hidden), ('fc2', hidden, moe_dim)):
                for part in _PARTS:
                    desc = index.get(f'{base}.{projection}.{part}')
                    if desc is None:
                        raise ExpertReadError(f'missing tensor {base}.{projection}.{part}')
                    last = cols // 8 if part == 'weight' else cols // group_size
                    if tuple(desc['shape']) != (expert_count, rows, last):
                        raise ExpertReadError(f'unexpected shape for {base}.{projection}.{part}')
                    per_expert = desc['length'] // expert_count
                    if per_expert * expert_count != desc['length'] or desc['dtype'] not in _NP:
                        raise ExpertReadError(f'unsupported stacked tensor {base}.{projection}.{part}')
                    entries.append((f'{projection}.{part}', desc['shard'], desc['offset'], per_expert,
                                    desc['dtype'], (rows, last)))
            self._slices[layer] = entries
        self.expert_bytes = sum(e[3] for e in next(iter(self._slices.values()))) if self._slices else 0

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds.clear()

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(self.directory / shard, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            self._fds[shard] = fd
            self._sizes[shard] = os.fstat(fd).st_size
        return fd

    def read(self, layer: int, expert: int) -> dict:
        if layer not in self._slices or type(expert) is not int or not 0 <= expert < self.expert_count:
            raise ExpertReadError('invalid layer or expert id')
        out = {}
        with self._lock:
            for name, shard, offset, length, dtype, shape in self._slices[layer]:
                fd = self._fd(shard)
                start = offset + expert * length
                if start + length > self._sizes[shard]:
                    raise ExpertReadError('shard truncated')
                data = os.pread(fd, length, start)
                if len(data) != length:
                    raise ExpertReadError('short expert read')
                self.bytes_read += length
                array = mx.array(np.frombuffer(data, dtype=_NP[dtype]).reshape(shape))
                out[name] = array.view(mx.bfloat16) if dtype == 'BF16' else array
            self.reads += 1
        return out


class ExpertPool:
    """Byte-budgeted LRU of loaded experts, shared by every MoE layer."""

    def __init__(self, reader: StackedExpertReader, budget_bytes: int):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('invalid expert pool budget')
        self.reader, self.budget = reader, budget_bytes
        self._items: OrderedDict = OrderedDict()
        self.resident_bytes = 0
        self.hits = self.misses = 0
        self.cancel_check = lambda: False

    def get(self, layer: int, expert: int) -> dict:
        key = (layer, expert)
        if key in self._items:
            self.hits += 1
            self._items.move_to_end(key)
            return self._items[key][0]
        if self.cancel_check():
            raise Cancelled()
        self.misses += 1
        arrays = self.reader.read(layer, expert)
        size = sum(a.nbytes for a in arrays.values())
        if size <= self.budget:
            while self._items and self.resident_bytes + size > self.budget:
                _, (_, old) = self._items.popitem(last=False)
                self.resident_bytes -= old
            self._items[key] = (arrays, size)
            self.resident_bytes += size
        return arrays

    def clear(self) -> None:
        self._items.clear()
        self.resident_bytes = 0

    def stats(self) -> dict:
        return {'hits': self.hits, 'misses': self.misses, 'resident_bytes': self.resident_bytes,
                'budget_bytes': self.budget, 'bytes_read': self.reader.bytes_read}


class StreamedSwitchMLP(nn.Module):
    """Drop-in for mlx-lm ``SwitchMLP`` (relu2 MLP): returns ``[..., K, hidden]``."""

    def __init__(self, pool: ExpertPool, layer: int, group_size: int):
        super().__init__()
        self.pool, self.layer, self.group_size = pool, layer, group_size

    def __call__(self, x, indices):
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        ids = np.asarray(indices).reshape(flat.shape[0], -1)
        rows_n, k = ids.shape
        hidden = x.shape[-1]
        result = mx.zeros((rows_n, k, hidden), dtype=x.dtype)
        for expert in np.unique(ids):
            rows, slots = np.nonzero(ids == expert)
            w = self.pool.get(self.layer, int(expert))
            sub = flat[mx.array(rows)] if len(rows) != rows_n else flat
            h = mx.quantized_matmul(sub, w['fc1.weight'], w['fc1.scales'], w['fc1.biases'],
                                    group_size=self.group_size, bits=4)
            y = mx.quantized_matmul(nn.relu2(h), w['fc2.weight'], w['fc2.scales'], w['fc2.biases'],
                                    group_size=self.group_size, bits=4)
            result[mx.array(rows), mx.array(slots)] = y.astype(x.dtype)
            mx.eval(result)  # release this expert before the next load
        return result.reshape(*lead, k, hidden)
