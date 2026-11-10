"""Build the Nemotron-H model with SSD-streamed routed experts.

The dense core (embeddings, Mamba, attention, router, shared experts, lm_head;
about 1.17 GiB for the 4-bit checkpoint) is read from the shards and held
resident. The stacked routed experts are never loaded: each ``switch_mlp`` is
replaced with ``StreamedSwitchMLP``. Parameter names are those of the installed
mlx-lm ``nemotron_h.Model``, so the same weights load unchanged into the
in-memory reference (used by the synthetic equivalence tests).

``load_streamed`` reads real weights. It is only called after the explicit
authorization/admission gates in the backend; tests exercise it on tiny
synthetic checkpoints written in the identical layout.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import nemotron_h

from .config import validate_config
from .inventory import CheckpointError, verify_checkpoint, verify_inventory, expected_tensors, _read_json
from .safetensors_index import index_checkpoint
from .streamed_experts import ExpertPool, StackedExpertReader, StreamedSwitchMLP

_NP = {'U32': np.uint32, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}


def _read_tensor(directory: Path, desc: dict):
    fd = os.open(directory / desc['shard'], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if desc['offset'] + desc['length'] > os.fstat(fd).st_size:
            raise CheckpointError('shard truncated')
        data = os.pread(fd, desc['length'], desc['offset'])
    finally:
        os.close(fd)
    if len(data) != desc['length']:
        raise CheckpointError('short tensor read')
    array = mx.array(np.frombuffer(data, dtype=_NP[desc['dtype']]).reshape(desc['shape']))
    return array.view(mx.bfloat16) if desc['dtype'] == 'BF16' else array


@contextlib.contextmanager
def _stub_switch_mlp(factory):
    """Make mlx-lm construct streamed experts instead of 128 resident ones."""
    original = nemotron_h.SwitchMLP
    nemotron_h.SwitchMLP = factory
    try:
        yield
    finally:
        nemotron_h.SwitchMLP = original


class StreamedModel:
    """The model plus the objects that own its resources."""

    def __init__(self, model, pool, reader, config, dense_bytes):
        self.model, self.pool, self.reader = model, pool, reader
        self.config, self.dense_bytes = config, dense_bytes

    def close(self) -> None:
        self.pool.clear()
        self.reader.close()


def load_streamed(directory, *, pool_bytes: int, config_check=None, verify: bool = True) -> StreamedModel:
    """Load the dense core resident and stream the routed experts from SSD.

    ``verify`` runs the full structural verification (headers, inventory,
    extents) first. ``config_check`` defaults to the strict public-config check;
    tests of tiny synthetic checkpoints pass a permissive one explicitly.
    """
    root = Path(directory).expanduser()
    check = config_check or validate_config
    if verify:
        verify_checkpoint(root, config_check=check)
    config = _read_json(root / 'config.json', 32 * 1024 * 1024, 'config.json')
    check(config)
    index = index_checkpoint(root)
    verify_inventory(index, config)
    group = config['quantization']['group_size']
    args = nemotron_h.ModelArgs.from_dict(config)
    moe_layers = [i for i, kind in enumerate(config['layers_block_type']) if kind == 'moe']
    reader = StackedExpertReader(root, index, layers=moe_layers, expert_count=config['n_routed_experts'],
                                 hidden=config['hidden_size'], moe_dim=config['moe_intermediate_size'],
                                 group_size=group)
    pool = ExpertPool(reader, pool_bytes)
    order = iter(moe_layers)
    with _stub_switch_mlp(lambda *a, **k: StreamedSwitchMLP(pool, next(order), group)):
        model = nemotron_h.Model(args)
    dense = {name: _read_tensor(root, desc) for name, desc in index.items() if '.switch_mlp.' not in name}
    nn.quantize(model, group_size=group, bits=4, mode='affine',
                class_predicate=lambda path, module: hasattr(module, 'to_quantized') and f'{path}.scales' in dense)
    model.load_weights(list(dense.items()), strict=True)
    mx.eval(model.parameters())
    dense_bytes = sum(v.nbytes for v in dense.values())
    return StreamedModel(model, pool, reader, config, dense_bytes)
