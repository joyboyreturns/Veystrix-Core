"""Explicit per-expert MLX affine-4bit tensor map (experimental, read-only).

This is an opt-in tensor layout for converted checkpoints, not NVIDIA NVFP4.
It does not convert weights or establish provenance. Validate checkpoint identity,
full tensor inventory and numerical equivalence before production use.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import mlx.core as mx
from .safetensors_index import index_checkpoint

class AffineExpertIndex:
    def __init__(self, directory, *, hidden_dim, intermediate_dim, group_size=64, max_tensor_bytes=64 * 1024 * 1024):
        if any(type(x) is not int or x <= 0 for x in (hidden_dim, intermediate_dim, group_size, max_tensor_bytes)):
            raise ValueError('invalid dimensions or limit')
        if group_size not in (32, 64, 128) or hidden_dim % group_size or intermediate_dim % group_size:
            raise ValueError('invalid affine quantization group size')
        self.directory = Path(directory)
        self.hidden_dim, self.intermediate_dim = hidden_dim, intermediate_dim
        self.group_size, self.max_tensor_bytes = group_size, max_tensor_bytes
        self.index = index_checkpoint(directory)

    def descriptors(self, layer, expert):
        if type(layer) is not int or type(expert) is not int or layer < 0 or expert < 0:
            raise ValueError('invalid layer or expert id')
        base = f'backbone.layers.{layer}.mixer.experts.{expert}'
        found = {}
        for name, rows, cols in (('up_proj', self.intermediate_dim, self.hidden_dim),
                                 ('down_proj', self.hidden_dim, self.intermediate_dim)):
            for part, expected_shape in (('weight', (rows, cols // 8)),
                                         ('scales', (rows, cols // self.group_size)),
                                         ('biases', (rows, cols // self.group_size))):
                key = f'{base}.{name}.{part}'
                desc = self.index.get(key)
                if desc is None or desc['shape'] != expected_shape or desc['length'] > self.max_tensor_bytes:
                    raise ValueError(f'missing or incompatible tensor: {key}')
                allowed_dtypes = ('U32',) if part == 'weight' else ('F32', 'F16', 'BF16')
                if desc['dtype'] not in allowed_dtypes:
                    raise ValueError(f'unsupported dtype for {key}')
                found[f'{name}.{part}'] = desc
        return found

    def validate_inventory(self, layers, expert_count):
        """Validate every required descriptor without reading tensor payloads."""
        if type(expert_count) is not int or expert_count <= 0:
            raise ValueError('invalid expert count')
        layers = list(layers)
        if not layers or len(layers) != len(set(layers)):
            raise ValueError('invalid MoE layer list')
        total_bytes = 0
        for layer in layers:
            for expert in range(expert_count):
                total_bytes += sum(desc['length'] for desc in self.descriptors(layer, expert).values())
        return {'layers': len(layers), 'experts': len(layers) * expert_count,
                'tensors': len(layers) * expert_count * 6, 'bytes': total_bytes}

    def read(self, layer, expert):
        """Read only the six validated tensors of the requested expert."""
        result = {}
        for name, desc in self.descriptors(layer, expert).items():
            fd = os.open(self.directory / desc['shard'], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            try:
                if desc['offset'] + desc['length'] > os.fstat(fd).st_size:
                    raise ValueError('shard changed or is truncated')
                data = os.pread(fd, desc['length'], desc['offset'])
            finally:
                os.close(fd)
            if len(data) != desc['length']:
                raise ValueError('short tensor read')
            dtype = {'U32': np.uint32, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}[desc['dtype']]
            value = mx.array(np.frombuffer(data, dtype=dtype).reshape(desc['shape']))
            result[name] = value.view(mx.bfloat16) if desc['dtype'] == 'BF16' else value
        return result
