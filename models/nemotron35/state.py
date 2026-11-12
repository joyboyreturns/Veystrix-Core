"""Experimental synthetic hybrid-state snapshots; not a production cache format."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache

from .attention import AttentionCache

class SnapshotError(ValueError):
    pass

def clone_cache(cache):
    """Give each continuation fresh cache objects and independent MLX array handles."""
    result = []
    for entry in cache:
        fresh = ArraysCache(size=len(entry.state)) if isinstance(entry, ArraysCache) else type(entry)()
        fresh.state = [value[...] for value in entry.state]
        result.append(fresh)
    return result

def fingerprint(identity: dict, tokens: list[int]) -> str:
    if not isinstance(identity, dict) or not identity or not all(isinstance(x, int) and x >= 0 for x in tokens):
        raise SnapshotError('invalid identity or tokens')
    encoded = json.dumps({'identity': identity, 'tokens': tokens}, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()

def _digest(tensors):
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key]
        dtype, shape = str(value.dtype), list(value.shape)
        if value.dtype == mx.bfloat16:      # numpy has no bfloat16: hash the raw 16-bit patterns
            value = value.view(mx.uint16)
        a = np.ascontiguousarray(np.asarray(value))
        digest.update(key.encode())
        digest.update(dtype.encode())
        digest.update(json.dumps(shape).encode())
        digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()

def save(path, cache, identity, tokens):
    """Write one atomic safetensors snapshot; only fully initialized, single-session caches."""
    tensors, types = {}, []
    for i, entry in enumerate(cache):
        if isinstance(entry, ArraysCache):
            if entry.lengths is not None or entry.left_padding is not None or len(entry.state) != 2:
                raise SnapshotError('batched or unsupported Mamba state')
            values, kind = entry.state, 'mamba'
        elif isinstance(entry, KVCache):
            values, kind = entry.state, 'attention'
        elif isinstance(entry, AttentionCache):
            if entry.offset != len(tokens):
                raise SnapshotError('attention offset does not match token tape')
            values, kind = entry.state, 'attention-int8'
        else:
            raise SnapshotError('unknown state type')
        if not values or any(v is None for v in values):
            raise SnapshotError('state is not initialized')
        types.append(kind)
        for j, value in enumerate(values):
            tensors[f'{i}.{j}'] = value
    if not types:
        raise SnapshotError('empty cache')
    mx.eval(*tensors.values())
    meta = {'fingerprint': fingerprint(identity, tokens), 'types': json.dumps(types), 'sha256': _digest(tensors)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.nemotron-', suffix='.safetensors', dir=path.parent)
    os.close(fd)
    try:
        mx.save_safetensors(tmp, tensors, metadata=meta)
        os.chmod(tmp, 0o600)
        with open(tmp, 'rb') as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def restore(path, identity, tokens):
    """Read and validate before constructing fresh cache objects."""
    try:
        tensors, meta = mx.load(str(path), return_metadata=True)
        if meta.get('fingerprint') != fingerprint(identity, tokens):
            raise SnapshotError('snapshot identity or prefix mismatch')
        types = json.loads(meta['types'])
        if not isinstance(types, list) or not types or len(types) > 52:
            raise SnapshotError('invalid cache layout')
        if _digest(tensors) != meta.get('sha256'):
            raise SnapshotError('snapshot integrity check failed')
        result, expected = [], set()
        for i, kind in enumerate(types):
            count = sum(key.startswith(f'{i}.') for key in tensors) if kind == 'attention-int8' else 2
            keys = [f'{i}.{j}' for j in range(count)]
            expected.update(keys)
            values = [tensors[key] for key in keys]
            if kind == 'mamba':
                entry = ArraysCache(size=2)
            elif kind == 'attention':
                if values[0].ndim != 4 or values[1].shape != values[0].shape:
                    raise SnapshotError('invalid KV state shape')
                if values[0].shape[-2] != len(tokens):
                    raise SnapshotError('invalid KV position')
                entry = KVCache()
            elif kind == 'attention-int8':
                entry = AttentionCache()
            else:
                raise SnapshotError('unknown cache type')
            entry.state = values
            if kind == 'attention-int8' and entry.offset != len(tokens):
                raise SnapshotError('invalid INT8 KV position')
            result.append(entry)
        if set(tensors) != expected:
            raise SnapshotError('invalid snapshot tensor names')
        return result
    except SnapshotError:
        raise
    except Exception as exc:
        raise SnapshotError(f'cannot read snapshot: {exc}') from exc
