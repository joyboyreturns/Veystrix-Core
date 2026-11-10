"""Experimental per-expert affine-4bit streaming for tiny synthetic fixtures.

The tensor schema below is deliberately explicit and is NOT NVIDIA NVFP4 or a
real checkpoint importer. Synchronous routing and host copies are correctness
baselines, not a performance claim. No prefetch or Metal residency is promised.
"""
from __future__ import annotations
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Each expert record contains six concatenated arrays, in this exact order.
# Shapes and types must come from a validated index, not inferred from a checkpoint.
_FIELDS = ('up_w', 'up_scales', 'up_biases', 'down_w', 'down_scales', 'down_biases')

class StreamedExperts(nn.Module):
    def __init__(self, store, specs, group_size=32):
        super().__init__()
        self.store, self.specs, self.group_size = store, specs, group_size
        if group_size not in (32, 64, 128):
            raise ValueError('unsupported affine group size')

    def _load(self, expert):
        data = self.store.get(expert)
        spec = self.specs[expert]
        if set(spec) != set(_FIELDS):
            raise ValueError('invalid expert tensor index')
        result, offset = {}, 0
        for name in _FIELDS:
            shape, dtype = spec[name]
            dtype = np.dtype(dtype)
            expected_dtype = np.dtype('uint32' if name.endswith('_w') else 'float32')
            if dtype != expected_dtype or not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(not isinstance(d, int) or d <= 0 for d in shape):
                raise ValueError('invalid expert tensor shape or dtype')
            count = math.prod(shape)
            size = count * dtype.itemsize
            if offset + size > len(data):
                raise ValueError('truncated expert tensor')
            array = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape)
            result[name] = mx.array(array)
            offset += size
        if offset != len(data):
            raise ValueError('unexpected trailing expert data')
        return result

    def __call__(self, x, indices):
        shape = x.shape
        if len(shape) != 3 or indices.shape[:2] != shape[:2]:
            raise ValueError('expected batched sequence and top-k expert indices')
        flat_x = x.reshape(-1, shape[-1])
        ids = np.asarray(indices).reshape(flat_x.shape[0], -1)
        output = []
        for row, selected in enumerate(ids):
            experts = []
            for expert in selected:
                arrays = self._load(int(expert))
                hidden = mx.quantized_matmul(flat_x[row:row+1], arrays['up_w'], arrays['up_scales'], arrays['up_biases'], group_size=self.group_size, bits=4)
                hidden = nn.relu2(hidden)
                y = mx.quantized_matmul(hidden, arrays['down_w'], arrays['down_scales'], arrays['down_biases'], group_size=self.group_size, bits=4)
                experts.append(y[0])
            output.append(mx.stack(experts))
        return mx.stack(output).reshape(shape[0], shape[1], ids.shape[1], -1)

class IndexedAffineExperts(nn.Module):
    """Selectively load MLX affine experts from a validated safetensors index.

    This prototype budgets cached array sizes, not total transient/GPU memory;
    lazy graphs may retain evicted arrays. Production admission is not provided.
    """
    def __init__(self, index, layer, max_resident_bytes):
        super().__init__()
        from collections import OrderedDict
        if type(max_resident_bytes) is not int or max_resident_bytes < 0:
            raise ValueError('invalid expert pool budget')
        self.index, self.layer = index, layer
        self.max_resident_bytes = max_resident_bytes
        self._pool = OrderedDict()
        self.resident_bytes = self.hits = self.misses = 0

    def _get(self, expert):
        if expert in self._pool:
            self.hits += 1
            self._pool.move_to_end(expert)
            return self._pool[expert][0]
        self.misses += 1
        arrays = self.index.read(self.layer, expert)
        size = sum(array.nbytes for array in arrays.values())
        if size <= self.max_resident_bytes:
            while self._pool and self.resident_bytes + size > self.max_resident_bytes:
                _, (_, old_size) = self._pool.popitem(last=False)
                self.resident_bytes -= old_size
            self._pool[expert] = (arrays, size)
            self.resident_bytes += size
        return arrays

    def __call__(self, x, indices):
        shape = x.shape
        if len(shape) != 3 or indices.shape[:2] != shape[:2]:
            raise ValueError('expected batched sequence and top-k expert indices')
        flat = x.reshape(-1, shape[-1])
        ids = np.asarray(indices).reshape(flat.shape[0], -1)
        rows = []
        for row, selected in enumerate(ids):
            outputs = []
            for expert in selected:
                w = self._get(int(expert))
                hidden = mx.quantized_matmul(flat[row:row+1], w['up_proj.weight'], w['up_proj.scales'], w['up_proj.biases'], group_size=self.index.group_size, bits=4)
                y = mx.quantized_matmul(nn.relu2(hidden), w['down_proj.weight'], w['down_proj.scales'], w['down_proj.biases'], group_size=self.index.group_size, bits=4)
                outputs.append(y[0])
            rows.append(mx.stack(outputs))
        return mx.stack(rows).reshape(shape[0], shape[1], ids.shape[1], -1)
