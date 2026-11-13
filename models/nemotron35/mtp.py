"""Standalone affine-Q4 Nemotron Lightning draft head for the streamed target."""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.nemotron_h import NemotronHBlock

DEFAULT_DIRECTORY = Path.home() / "Models" / "Nemotron-3.5-Lightning-MTP-MLX-4bit"
HISTORY_LIMIT = 8192


def memory_reserve(config, directory):
    # BF16 draft K/V plus a block-allocation margin and transient MoE buffers.
    kv = (HISTORY_LIMIT + 512) * config["num_key_value_heads"] * config["head_dim"] * 4
    return (directory / "mtp_head.safetensors").stat().st_size + kv + 128 * 2**20


def resolve_directory():
    path = Path(os.environ.get("VEYSTRIX_NEMOTRON_MTP_PATH") or DEFAULT_DIRECTORY).expanduser()
    if not (path / "mtp_head.safetensors").is_file() or not (path / "config.json").is_file():
        raise ValueError(f"Nemotron MTP sidecar is missing at {path}")
    return path


class DraftHead:
    def __init__(self, target, directory):
        config = json.loads((directory / "config.json").read_text())
        if (config.get("model_type") != "nemotron_h_mtp_head" or config.get("hidden_input") !=
                "residual stream before norm_f (prenorm)" or config.get("layers") != ["attention", "moe"]
                or config.get("quantization") != {"bits": 4, "group_size": 64, "mode": "affine"}):
            raise ValueError("incompatible Nemotron Lightning MTP sidecar")
        self.target = target
        self.attention = NemotronHBlock(target.args, "*")
        self.experts = NemotronHBlock(target.args, "E")
        tensors = mx.load(str(directory / "mtp_head.safetensors"))
        for block, prefix in ((self.attention, "attn."), (self.experts, "moe.")):
            nn.quantize(block, bits=4, group_size=64, mode="affine")
            block.load_weights([(key.removeprefix(prefix), value) for key, value in tensors.items()
                                if key.startswith(prefix)], strict=True)
        self.enorm, self.hnorm, self.final_norm = (tensors[key] for key in ("enorm", "hnorm", "final_norm"))
        self.project = tensors["eh_proj"]
        size = target.args.hidden_size
        if self.project.shape != (size, 2 * size) or any(
            array.shape != (size,) for array in (self.enorm, self.hnorm, self.final_norm)
        ):
            raise ValueError("Nemotron MTP tensor shapes do not match the target")
        mx.eval(self.attention.parameters(), self.experts.parameters(), self.project, self.enorm, self.hnorm, self.final_norm)
        self.reset()

    def reset(self):
        self.kv = KVCache()

    def rewind(self, offset):
        if self.kv.offset > offset:
            self.kv.trim(self.kv.offset - offset)

    def forward(self, rows, tokens, *, predict=False):
        epsilon = self.target.args.layer_norm_epsilon
        embedded = self.target.backbone.embeddings(tokens)
        pair = mx.concatenate((mx.fast.rms_norm(embedded, self.enorm, epsilon),
                               mx.fast.rms_norm(rows, self.hnorm, epsilon)), axis=-1)
        hidden = (pair @ self.project.T)[None]
        hidden = self.attention(hidden, mask=create_attention_mask(hidden, self.kv), cache=self.kv)
        # Only attention has persistent state. Warmup never needs the MoE or vocabulary projection.
        if not predict:
            mx.eval(*self.kv.state)
            return None
        hidden = mx.fast.rms_norm(self.experts(hidden)[0], self.final_norm, epsilon)
        token = int(mx.argmax(self.target.lm_head(hidden[-1:]), axis=-1).item())
        return token, hidden[-1:]
