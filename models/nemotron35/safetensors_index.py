"""Bounded read-only safetensors metadata indexing; never loads tensor payloads.

This validates file layout and path safety only. It does not prove tensor names,
shapes or quantization correspond to the NVIDIA checkpoint.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import struct

MAX_HEADER_BYTES = 64 * 1024 * 1024
_DTYPES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
           'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}

class IndexError(ValueError):
    pass

def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise IndexError('duplicate JSON key')
        out[key] = value
    return out

def index_shard(directory, filename):
    """Return tensor offset, length, dtype and shape without mapping any weights."""
    if not isinstance(filename, str) or not filename.endswith('.safetensors') or Path(filename).name != filename or filename.startswith('.') or '\\' in filename:
        raise IndexError('invalid shard filename')
    path = Path(directory) / filename
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            size = os.fstat(fd).st_size
            prefix = os.pread(fd, 8, 0)
            if len(prefix) != 8:
                raise IndexError('truncated header length')
            length = struct.unpack('<Q', prefix)[0]
            if length <= 1 or length > MAX_HEADER_BYTES or 8 + length > size:
                raise IndexError('invalid header length')
            raw = os.pread(fd, length, 8)
            if len(raw) != length:
                raise IndexError('truncated header')
            header = json.loads(raw, object_pairs_hook=_unique_object)
        finally:
            os.close(fd)
        if not isinstance(header, dict):
            raise IndexError('header must be an object')
        result, ranges = {}, []
        for name, desc in header.items():
            if name == '__metadata__':
                continue
            if not isinstance(name, str) or not name or not isinstance(desc, dict):
                raise IndexError('invalid tensor descriptor')
            dtype, shape, offsets = desc.get('dtype'), desc.get('shape'), desc.get('data_offsets')
            if dtype not in _DTYPES or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise IndexError(f'invalid tensor dtype or shape: {name}')
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
                raise IndexError(f'invalid tensor offsets: {name}')
            start, end = offsets
            if start < 0 or end < start or 8 + length + end > size or end - start != math.prod(shape) * _DTYPES[dtype]:
                raise IndexError(f'tensor outside shard or has wrong size: {name}')
            ranges.append((start, end))
            result[name] = {'offset': 8 + length + start, 'length': end - start, 'dtype': dtype, 'shape': tuple(shape)}
        ordered = sorted(ranges)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise IndexError('overlapping tensor ranges')
        return result
    except IndexError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        raise IndexError(f'cannot index shard: {exc}') from exc

def index_checkpoint(directory, index_filename='model.safetensors.index.json'):
    """Validate shard index consistency and return descriptors for indexed tensors.

    Reads bounded JSON headers only; does not load or dequantize any weights.
    """
    if index_filename != 'model.safetensors.index.json':
        raise IndexError('unsupported checkpoint index filename')
    path = Path(directory) / index_filename
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            size = os.fstat(fd).st_size
            if size <= 0 or size > MAX_HEADER_BYTES:
                raise IndexError('invalid checkpoint index size')
            data = os.pread(fd, size, 0)
        finally:
            os.close(fd)
        index = json.loads(data, object_pairs_hook=_unique_object)
        weights = index.get('weight_map')
        if not isinstance(weights, dict) or not weights or any(not isinstance(k, str) or not isinstance(v, str) for k, v in weights.items()):
            raise IndexError('missing or invalid weight map')
        shard_indexes = {filename: index_shard(directory, filename) for filename in set(weights.values())}
        result = {}
        for name, filename in weights.items():
            descriptor = shard_indexes[filename].get(name)
            if descriptor is None:
                raise IndexError(f'indexed tensor missing from shard: {name}')
            result[name] = {'shard': filename, **descriptor}
        return result
    except IndexError:
        raise
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise IndexError(f'cannot index checkpoint: {exc}') from exc
