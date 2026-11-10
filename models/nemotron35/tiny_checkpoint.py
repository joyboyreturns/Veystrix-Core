"""Tiny synthetic checkpoints in the exact mlx-community stacked affine-4-bit layout.

For tests and diagnostics only. The weights are random; this validates loader,
index, streaming and state logic end to end without any real weights. It is not
evidence of numerical equivalence with the real model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import nemotron_h

TINY_CONFIG = {
    "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "vocab_size": 64,
    "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 4, "max_position_embeddings": 256,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "attention_bias": False,
    "mamba_num_heads": 4, "mamba_head_dim": 16, "mamba_proj_bias": False, "ssm_state_size": 32,
    "conv_kernel": 4, "n_groups": 2, "mlp_bias": False, "layer_norm_epsilon": 1e-5,
    "use_bias": False, "use_conv_bias": True,
    "layers_block_type": ["mamba", "moe", "attention", "moe"],
    "moe_intermediate_size": 64, "moe_shared_expert_intermediate_size": 64, "moe_latent_size": None,
    "n_group": 1, "n_routed_experts": 8, "n_shared_experts": 1, "topk_group": 1,
    "num_experts_per_tok": 3, "norm_topk_prob": True, "routed_scaling_factor": 2.5,
    "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}, "tie_word_embeddings": False,
    "eos_token_id": [2, 11],
}


def permissive_config_check(config: dict) -> None:
    """Accept the tiny config. Production code never uses this; it is a test seam."""
    if config.get("model_type") != "nemotron_h":
        raise ValueError("not a nemotron_h config")


def build_reference(config: dict, seed: int = 3):
    """In-memory quantized reference (all experts resident), bf16 like the real checkpoint."""
    mx.random.seed(seed)
    args = nemotron_h.ModelArgs.from_dict(config)
    model = nemotron_h.Model(args)
    for layer in model.layers:
        if layer.block_type == "E":
            layer.mixer.gate.weight = mx.random.normal(layer.mixer.gate.weight.shape) * 0.2
            layer.mixer.gate.e_score_correction_bias = mx.random.normal((config["n_routed_experts"],)) * 0.05
    q = config["quantization"]
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], mode="affine")
    for layer in model.layers:
        if layer.block_type == "E":  # the real checkpoint keeps the correction bias in float32
            layer.mixer.gate.e_score_correction_bias = layer.mixer.gate.e_score_correction_bias.astype(mx.float32)
    mx.eval(model.parameters())
    return model


def write_checkpoint(directory, config: dict | None = None, shards: int = 2, seed: int = 3):
    """Write config, shards, index, tokenizer placeholders; return the in-memory reference model."""
    config = dict(config or TINY_CONFIG)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    model = build_reference(config, seed)
    weights = dict(tree_flatten(model.parameters()))
    names = sorted(weights)
    weight_map, total = {}, 0
    for shard in range(shards):
        filename = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        part = {n: weights[n] for n in names[shard::shards]}
        mx.save_safetensors(str(root / filename), part)
        weight_map.update({n: filename for n in part})
        total += sum(v.nbytes for v in part.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (root / name).write_text("{}")
    return model
