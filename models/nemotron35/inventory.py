"""Exact tensor inventory for the mlx-community Nemotron affine-4-bit layout.

Everything here is derived from ``config.json`` and checked against the shard
headers (never tensor payloads). Verification is fail-closed: a missing,
extra, mis-shaped or mis-typed tensor, a missing or growing shard, trailing
bytes, or a size mismatch is an error. Passing proves the file *layout* only;
it does not prove numerical equivalence with NVIDIA's weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from .safetensors_index import IndexError as ShardIndexError, MAX_HEADER_BYTES, index_checkpoint

FLOAT_DTYPES = ('BF16', 'F16', 'F32')
SCALE_DTYPES = ('BF16', 'F16')
BITS = 4
PACK = 32 // BITS  # 4-bit values packed per uint32
DTYPE_BYTES = {'U32': 4, 'BF16': 2, 'F16': 2, 'F32': 4}


class CheckpointError(ValueError):
    """The checkpoint is incomplete, damaged or not the supported layout."""


def _quantized(prefix, rows, cols, group, lead=()):
    """Weight/scales/biases triple of an affine-quantized matrix [*lead, rows, cols]."""
    if cols % PACK or cols % group:
        raise CheckpointError(f'{prefix}: width {cols} is not compatible with group size {group}')
    return {
        f'{prefix}.weight': ((*lead, rows, cols // PACK), ('U32',)),
        f'{prefix}.scales': ((*lead, rows, cols // group), SCALE_DTYPES),
        f'{prefix}.biases': ((*lead, rows, cols // group), SCALE_DTYPES),
    }


def expected_tensors(config: dict) -> dict:
    """Return {tensor name: (shape, allowed dtypes)} for the whole checkpoint."""
    q = config.get('quantization') or {}
    if q.get('mode') != 'affine' or q.get('bits') != BITS or not isinstance(q.get('group_size'), int):
        raise CheckpointError('only MLX affine 4-bit quantization is supported (not NVFP4)')
    group = q['group_size']
    hidden, vocab = config['hidden_size'], config['vocab_size']
    heads, head_dim = config['mamba_num_heads'], config['mamba_head_dim']
    groups, state, kernel = config['n_groups'], config['ssm_state_size'], config['conv_kernel']
    inner = heads * head_dim
    conv_dim = inner + 2 * groups * state
    attn_heads, kv_heads, hd = config['num_attention_heads'], config['num_key_value_heads'], config['head_dim']
    experts, moe_dim = config['n_routed_experts'], config['moe_intermediate_size']
    shared_dim = config['moe_shared_expert_intermediate_size']
    if config.get('moe_latent_size') is not None:
        raise CheckpointError('latent MoE projections are not part of this layout')
    out = {}
    out.update(_quantized('backbone.embeddings', vocab, hidden, group))
    out['backbone.norm_f.weight'] = ((hidden,), FLOAT_DTYPES)
    out.update(_quantized('lm_head', vocab, hidden, group))
    for layer, kind in enumerate(config['layers_block_type']):
        base = f'backbone.layers.{layer}'
        out[f'{base}.norm.weight'] = ((hidden,), FLOAT_DTYPES)
        mixer = f'{base}.mixer'
        if kind == 'mamba':
            out.update(_quantized(f'{mixer}.in_proj', inner + conv_dim + heads, hidden, group))
            out.update(_quantized(f'{mixer}.out_proj', hidden, inner, group))
            out[f'{mixer}.conv1d.weight'] = ((conv_dim, kernel, 1), FLOAT_DTYPES)
            out[f'{mixer}.conv1d.bias'] = ((conv_dim,), FLOAT_DTYPES)
            for name in ('A_log', 'D', 'dt_bias'):
                out[f'{mixer}.{name}'] = ((heads,), FLOAT_DTYPES)
            out[f'{mixer}.norm.weight'] = ((inner,), FLOAT_DTYPES)
        elif kind == 'attention':
            out.update(_quantized(f'{mixer}.q_proj', attn_heads * hd, hidden, group))
            out.update(_quantized(f'{mixer}.k_proj', kv_heads * hd, hidden, group))
            out.update(_quantized(f'{mixer}.v_proj', kv_heads * hd, hidden, group))
            out.update(_quantized(f'{mixer}.o_proj', hidden, attn_heads * hd, group))
        elif kind == 'moe':
            out[f'{mixer}.gate.weight'] = ((experts, hidden), FLOAT_DTYPES)
            out[f'{mixer}.gate.e_score_correction_bias'] = ((experts,), FLOAT_DTYPES)
            out.update(_quantized(f'{mixer}.shared_experts.up_proj', shared_dim, hidden, group))
            out.update(_quantized(f'{mixer}.shared_experts.down_proj', hidden, shared_dim, group))
            out.update(_quantized(f'{mixer}.switch_mlp.fc1', moe_dim, hidden, group, lead=(experts,)))
            out.update(_quantized(f'{mixer}.switch_mlp.fc2', hidden, moe_dim, group, lead=(experts,)))
        else:
            raise CheckpointError(f'unsupported layer kind: {kind!r}')
    return out


def verify_inventory(index: dict, config: dict) -> dict:
    """Compare an indexed checkpoint with the exact expected inventory."""
    expected = expected_tensors(config)
    missing = sorted(set(expected) - set(index))
    extra = sorted(set(index) - set(expected))
    if missing or extra:
        raise CheckpointError(
            f'tensor inventory mismatch: {len(missing)} missing (e.g. {missing[:3]}), '
            f'{len(extra)} unexpected (e.g. {extra[:3]})')
    expert_bytes = dense_bytes = 0
    for name, (shape, dtypes) in expected.items():
        desc = index[name]
        if tuple(desc['shape']) != tuple(shape):
            raise CheckpointError(f'{name}: shape {tuple(desc["shape"])} != expected {tuple(shape)}')
        if desc['dtype'] not in dtypes:
            raise CheckpointError(f'{name}: dtype {desc["dtype"]} not in {dtypes}')
        if '.switch_mlp.' in name:
            expert_bytes += desc['length']
        else:
            dense_bytes += desc['length']
    return {'tensors': len(expected), 'dense_bytes': dense_bytes, 'routed_expert_bytes': expert_bytes,
            'total_bytes': dense_bytes + expert_bytes}


def _read_json(path: Path, limit: int, what: str):
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as exc:
        raise CheckpointError(f'cannot open {what}: {exc}') from exc
    try:
        size = os.fstat(fd).st_size
        if size <= 0 or size > limit:
            raise CheckpointError(f'{what} has an invalid size')
        data = os.pread(fd, size, 0)
    finally:
        os.close(fd)
    try:
        return json.loads(data)
    except ValueError as exc:
        raise CheckpointError(f'{what} is not valid JSON: {exc}') from exc


def _shard_files(root: Path):
    return sorted(p.name for p in root.iterdir() if p.name.endswith('.safetensors'))


def _check_exact_shards(root: Path, weight_map: dict) -> None:
    """Every referenced shard exists, nothing else claims to be one, no partial downloads."""
    referenced = set(weight_map.values())
    for name in referenced:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise CheckpointError(f'missing or non-regular shard: {name}')
    unreferenced = set(_shard_files(root)) - referenced
    if unreferenced:
        raise CheckpointError(f'shards not named by the index: {sorted(unreferenced)}')
    partial = [p.name for p in root.iterdir() if p.name.endswith(('.incomplete', '.part', '.tmp'))]
    if partial:
        raise CheckpointError(f'partial download files present: {partial}')


def _check_shard_extent(path: Path) -> None:
    """The shard must be exactly header + tensors: no truncation, no trailing bytes."""
    with open(path, 'rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        length = int.from_bytes(prefix, 'little') if len(prefix) == 8 else 0
        if not 1 < length <= MAX_HEADER_BYTES or 8 + length > size:
            raise CheckpointError(f'{path.name}: invalid header')
        header = json.loads(handle.read(length))
    ends = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
    if 8 + length + (max(ends) if ends else 0) != size:
        raise CheckpointError(f'{path.name}: file size does not match its tensor extent')


def hf_sha256(root: Path, filename: str) -> str | None:
    """SHA-256 recorded by huggingface_hub for a downloaded LFS file, if present."""
    meta = root / '.cache' / 'huggingface' / 'download' / f'{filename}.metadata'
    try:
        lines = meta.read_text().splitlines()
    except OSError:
        return None
    etag = lines[1].strip() if len(lines) > 1 else ''
    return etag if len(etag) == 64 and all(c in '0123456789abcdef' for c in etag) else None


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path, *, config_check=None, hashes: bool = False, require_hashes: bool = False) -> dict:
    """Fail-closed structural verification of a complete local checkpoint.

    Reads the config, the index and shard headers only. With ``hashes`` each
    shard is streamed through SHA-256 and compared with the digest recorded by
    huggingface_hub (reads every shard once; no weights are loaded).
    ``config_check`` defaults to the strict public-config validator.
    """
    from .config import validate_config, ConfigurationError
    root = Path(path).expanduser()
    if not root.is_dir():
        raise CheckpointError(f'checkpoint folder does not exist: {root}')
    config = _read_json(root / 'config.json', 32 * 1024 * 1024, 'config.json')
    try:
        (config_check or validate_config)(config)
    except ConfigurationError as exc:
        raise CheckpointError(str(exc)) from exc
    raw_index = _read_json(root / 'model.safetensors.index.json', MAX_HEADER_BYTES, 'checkpoint index')
    if not isinstance(raw_index, dict) or not isinstance(raw_index.get('weight_map'), dict):
        raise CheckpointError('checkpoint index has no weight map')
    _check_exact_shards(root, raw_index['weight_map'])
    for name in sorted(set(raw_index['weight_map'].values())):
        _check_shard_extent(root / name)
    try:
        index = index_checkpoint(root)
    except ShardIndexError as exc:
        raise CheckpointError(f'shard index invalid: {exc}') from exc
    summary = verify_inventory(index, config)
    declared = (raw_index.get('metadata') or {}).get('total_size')
    if declared is not None and declared != summary['total_bytes']:
        raise CheckpointError(f'index total_size {declared} != tensor bytes {summary["total_bytes"]}')
    for required in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        if not (root / required).is_file():
            raise CheckpointError(f'missing {required}')
    verified = {}
    if hashes:
        for name in sorted(set(raw_index['weight_map'].values())):
            expected = hf_sha256(root, name)
            if expected is None:
                if require_hashes:
                    raise CheckpointError(f'no recorded SHA-256 for {name}')
                continue
            if sha256_file(root / name) != expected:
                raise CheckpointError(f'SHA-256 mismatch for {name}: shard is damaged')
            verified[name] = expected
        if require_hashes and len(verified) != len(set(raw_index['weight_map'].values())):
            raise CheckpointError('not every shard has a verified SHA-256')
    summary.update(shards=len(set(raw_index['weight_map'].values())), hashes_verified=len(verified),
                   hashes_checked=bool(hashes))
    return summary


def inventory_names_only(index_path, config: dict) -> dict:
    """Check just the names in model.safetensors.index.json (usable on a partial download)."""
    raw = _read_json(Path(index_path), MAX_HEADER_BYTES, 'checkpoint index')
    weight_map = raw.get('weight_map') if isinstance(raw, dict) else None
    if not isinstance(weight_map, dict):
        raise CheckpointError('checkpoint index has no weight map')
    for shard in weight_map.values():
        if (not isinstance(shard, str) or not shard.endswith('.safetensors') or Path(shard).name != shard
                or shard.startswith('.') or '\\' in shard):
            raise CheckpointError('checkpoint index contains an unsafe shard name')
    expected = expected_tensors(config)
    missing, extra = sorted(set(expected) - set(weight_map)), sorted(set(weight_map) - set(expected))
    if missing or extra:
        raise CheckpointError(f'index names mismatch: {len(missing)} missing, {len(extra)} unexpected')
    return {'names': len(expected), 'shards': sorted(set(weight_map.values())),
            'declared_total_size': (raw.get('metadata') or {}).get('total_size')}


def estimate_bytes(config: dict) -> dict:
    """Byte sizes implied by the expected inventory (no files read)."""
    dense = routed = 0
    for name, (shape, dtypes) in expected_tensors(config).items():
        size = math.prod(shape) * DTYPE_BYTES[dtypes[0]]
        if '.switch_mlp.' in name:
            routed += size
        else:
            dense += size
    return {'dense_bytes': dense, 'routed_expert_bytes': routed, 'total_bytes': dense + routed}
