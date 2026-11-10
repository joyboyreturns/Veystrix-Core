"""Equivalence check of the streamed engine against an independent reference on the REAL checkpoint.

``python -m models.nemotron35.verify_real CHECKPOINT [--tokens N] [--decode K]``

The 17 GB checkpoint cannot be held in memory, so the reference is built one layer at a time from the checkpoint files
with the stock mlx-lm blocks (``NemotronHBlock`` with the stock all-experts-resident ``SwitchMLP`` / ``gather_qmm``
path): load a layer's weights, run every token through it, discard it. The engine under test is the production
``load_streamed`` + ``EngineCore`` path (SSD-streamed experts, prefetch, pool, optionally layer-major prefill).

Both paths process the same prompt and then the same forced continuation tokens. Reported per step: max/mean absolute
logit difference, whether the greedy token agrees, and the top-5 overlap. Nothing here proves quality; it shows the
streaming/prefetch/pool/layer-major machinery does not change what the stock model computes on these weights.

Memory: one layer at a time plus the streamed engine (pool 0 by default). Run it alone, plugged in, with other
heavy apps closed. It stops if swap in use passes ``--swap-limit-mb`` (default 2000).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import nemotron_h
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from .config import validate_config
from .inventory import _read_json
from .safetensors_index import index_checkpoint
from .streamed_model import _read_tensor, load_streamed


def swap_mb() -> float:
    out = subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True, text=True).stdout
    return float(re.search(r'used = ([\d.]+)M', out).group(1))


class Reference:
    """Layer-at-a-time stock implementation. Holds only the small per-layer caches between passes."""

    def __init__(self, root: Path, config: dict, swap_limit_mb: float):
        self.root, self.config, self.swap_limit = root, config, swap_limit_mb
        self.index = index_checkpoint(root)
        self.args = nemotron_h.ModelArgs.from_dict(config)
        self.group = config['quantization']['group_size']
        self.types = config['layers_block_type']
        self.caches = {i: (nemotron_h.ArraysCache(size=2) if t == 'mamba' else nemotron_h.KVCache())
                       for i, t in enumerate(self.types) if t in ('mamba', 'attention')}

    def _load(self, module, prefix: str):
        weights = {n[len(prefix):]: _read_tensor(self.root, d) for n, d in self.index.items() if n.startswith(prefix)}
        nn.quantize(module, group_size=self.group, bits=4, mode='affine',
                    class_predicate=lambda path, m: hasattr(m, 'to_quantized') and f'{path}.scales' in weights)
        module.load_weights(list(weights.items()), strict=True)
        mx.eval(module.parameters())

    def _load_leaf(self, module, prefix: str):
        """Load a stand-alone layer (embedding, norm, head): nn.quantize cannot replace a root module, so wrap it."""
        class Holder(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.m = inner
        holder = Holder(module)
        weights = {'m.' + n[len(prefix):]: _read_tensor(self.root, d) for n, d in self.index.items() if n.startswith(prefix)}
        nn.quantize(holder, group_size=self.group, bits=4, mode='affine',
                    class_predicate=lambda path, m: hasattr(m, 'to_quantized') and f'{path}.scales' in weights)
        holder.load_weights(list(weights.items()), strict=True)
        mx.eval(holder.parameters())
        return holder.m

    def _guard(self):
        if swap_mb() > self.swap_limit:
            raise RuntimeError(f'swap guard: {swap_mb():.0f} MB in use')

    def forward(self, tokens, chunk: int = 0) -> mx.array:
        """Logits for the last token of ``tokens``, advancing the per-layer caches. ``chunk`` > 0 feeds each layer the
        tokens in chunks of that size (the engine's chunk boundaries), else all at once."""
        embed = self._load_leaf(nn.Embedding(self.args.vocab_size, self.args.hidden_size), 'backbone.embeddings.')
        size = chunk or len(tokens)
        parts = [embed(mx.array([tokens[i:i + size]], dtype=mx.int32)) for i in range(0, len(tokens), size)]
        mx.eval(*parts)
        kinds = {'mamba': 'M', 'attention': '*', 'moe': 'E', 'mlp': '-'}
        for i, kind in enumerate(self.types):
            block = nemotron_h.NemotronHBlock(self.args, kinds[kind])
            self._load(block, f'backbone.layers.{i}.')
            cache = self.caches.get(i)
            for j, h in enumerate(parts):
                mask = create_attention_mask(h, cache) if kind == 'attention' else (
                    create_ssm_mask(h, cache) if kind == 'mamba' else None)
                parts[j] = block(h, mask=mask, cache=cache)
                mx.eval(parts[j])
            mx.eval(*([a for a in cache.state if a is not None] if cache is not None else []))
            del block
            mx.clear_cache()
            self._guard()
        norm = self._load_leaf(nn.RMSNorm(self.args.hidden_size, eps=self.args.layer_norm_epsilon), 'backbone.norm_f.')
        head = self._load_leaf(nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False), 'lm_head.')
        logits = head(norm(parts[-1][:, -1:, :]))[:, -1, :]
        mx.eval(logits)
        return logits


def compare(a: mx.array, b: mx.array) -> dict:
    x, y = np.asarray(a.astype(mx.float32)).reshape(-1), np.asarray(b.astype(mx.float32)).reshape(-1)
    top_x, top_y = np.argsort(-x)[:5], np.argsort(-y)[:5]
    diff = np.abs(x - y)
    return {'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean()),
            'logit_range': float(x.max() - x.min()), 'greedy_agree': bool(top_x[0] == top_y[0]),
            'top5_overlap': int(len(set(top_x.tolist()) & set(top_y.tolist()))), 'greedy_token': int(top_x[0]),
            'bit_identical': bool(np.array_equal(x, y))}


def verify(root, *, tokens=48, decode=3, layer_major=0, chunk=16, pool_gb=0.0, swap_limit_mb=2000.0,
           config_check=None, reference_chunk=None) -> dict:
    """``reference_chunk``: chunk size the reference feeds each layer (default: the engine's ``chunk``, so the two
    differ only in how routed experts are computed; 0 feeds the whole prompt at once)."""
    root = Path(root).expanduser()
    config = _read_json(root / 'config.json', 32 * 1024 * 1024, 'config.json')
    (config_check or validate_config)(config)
    rng = np.random.default_rng(1234)
    low = 100 if config['vocab_size'] > 1000 else 3
    prompt = [int(t) for t in rng.integers(low, min(config['vocab_size'], 30000), size=tokens)]
    report = {'checkpoint': str(root), 'prompt_tokens': tokens, 'reference_chunk': chunk if reference_chunk is None else reference_chunk, 'layer_major_tokens': layer_major,
              'chunk_tokens': chunk, 'pool_gb': pool_gb, 'steps': []}
    began = time.time()
    from .backend_core import EngineCore
    streamed = load_streamed(root, pool_bytes=int(pool_gb * 2 ** 30), verify=False, config_check=config_check)
    try:
        core = EngineCore(streamed, chunk_tokens=chunk, layer_major_tokens=layer_major)
        ref = Reference(root, config, swap_limit_mb)
        ours = core.prefill(prompt)
        rc = chunk if reference_chunk is None else reference_chunk
        theirs = ref.forward(prompt, rc)
        step = compare(ours, theirs)
        step['phase'] = 'prefill'
        report['steps'].append(step)
        token = step['greedy_token']
        for k in range(decode):
            ours = core.decode(token)
            theirs = ref.forward([token])
            step = compare(ours, theirs)
            step['phase'] = f'decode{k + 1}'
            report['steps'].append(step)
            token = step['greedy_token']
    finally:
        streamed.close()
    report['seconds'] = round(time.time() - began, 1)
    report['all_greedy_agree'] = all(s['greedy_agree'] for s in report['steps'])
    report['all_bit_identical'] = all(s['bit_identical'] for s in report['steps'])
    report['max_abs_diff'] = max(s['max_abs_diff'] for s in report['steps'])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--tokens', type=int, default=48)
    parser.add_argument('--decode', type=int, default=3)
    parser.add_argument('--layer-major', type=int, default=0, help='segment tokens for layer-major prefill (0: chunk-major)')
    parser.add_argument('--chunk', type=int, default=16, help='prefill chunk tokens of the engine under test')
    parser.add_argument('--reference-chunk', type=int, default=None, help='chunk size of the reference (default: --chunk; 0 = whole prompt)')
    parser.add_argument('--pool-gb', type=float, default=0.0)
    parser.add_argument('--swap-limit-mb', type=float, default=2000.0)
    ns = parser.parse_args(argv)
    report = verify(ns.checkpoint, tokens=ns.tokens, decode=ns.decode, layer_major=ns.layer_major, chunk=ns.chunk,
                    pool_gb=ns.pool_gb, swap_limit_mb=ns.swap_limit_mb, reference_chunk=ns.reference_chunk)
    print(json.dumps(report, indent=2))
    return 0 if report['all_greedy_agree'] else 1


if __name__ == '__main__':
    sys.exit(main())
