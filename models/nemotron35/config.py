"""Offline, fail-closed validation of NVIDIA's Nemotron 3.5 Lightning metadata."""
from __future__ import annotations
import json
from pathlib import Path

MODEL_ID = "veystrix-nemotron35-lightning"
# Exact public BF16 config. The 52-element layer order is checked independently.
EXPECTED = {"model_type": "nemotron_h", "num_hidden_layers": 52, "hidden_size": 2688,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8,
            "ssm_state_size": 128, "conv_kernel": 4, "n_routed_experts": 128,
            "num_experts_per_tok": 6, "n_shared_experts": 1,
            "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712,
            "moe_latent_size": None, "norm_topk_prob": True, "routed_scaling_factor": 2.5}

LAYER_ORDER = ('mamba','moe','mamba','moe','mamba','attention','moe','mamba','moe','mamba','moe','mamba','attention','moe','mamba','moe','mamba','moe','mamba','attention','moe','mamba','moe','mamba','moe','mamba','attention','moe','mamba','moe','mamba','moe','mamba','attention','moe','mamba','moe','mamba','moe','mamba','moe','mamba','attention','moe','mamba','moe','mamba','moe','mamba','moe','mamba','moe')

class ConfigurationError(ValueError):
    pass

def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ConfigurationError("config.json must be an object")
    for name, wanted in EXPECTED.items():
        if name not in config or config[name] != wanted:
            raise ConfigurationError(f"incompatible {name}: expected {wanted!r}, got {config.get(name)!r}")
    layers = config.get("layers_block_type")
    if not isinstance(layers, list) or len(layers) != 52 or layers.count("mamba") != 23 or layers.count("moe") != 23 or layers.count("attention") != 6:
        raise ConfigurationError("expected 52 hybrid layers: 23 mamba, 23 moe, 6 attention")
    # The reference order is authoritative; accepting only counts can silently execute the wrong architecture.

    if tuple(layers) != LAYER_ORDER:
        raise ConfigurationError("layer order differs from NVIDIA Lightning configuration")
    q = config.get("quantization_config", {})
    if isinstance(q, dict) and ("nvfp4" in json.dumps(q).lower()):
        raise ConfigurationError("NVFP4 requires conversion; Apple MLX affine 4-bit is not NVFP4")

def inspect_checkpoint(path: str | Path, *, hashes: bool = False) -> dict:
    """Fail-closed metadata check. Never loads weights.

    A partial download is reported as incomplete (names-only check); a complete
    one goes through the full header/inventory/extent verification.
    """
    from .inventory import CheckpointError, inventory_names_only, verify_checkpoint
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ConfigurationError(f"checkpoint folder does not exist: {root} (no download attempted)")
    config_path = root / "config.json"
    try:
        if config_path.stat().st_size > 32 * 1024 * 1024:
            raise ConfigurationError("config.json is unexpectedly large")
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read checkpoint config: {exc}") from exc
    validate_config(config)
    base = {"model": MODEL_ID, "layers": len(config["layers_block_type"]),
            "architecture": config["model_type"], "quantization": config.get("quantization")}
    try:
        names = inventory_names_only(root / "model.safetensors.index.json", config)
    except CheckpointError as exc:
        raise ConfigurationError(str(exc)) from exc
    missing = sorted(n for n in names["shards"] if not (root / n).is_file())
    partial = sorted(p.name for p in root.iterdir() if p.name.endswith((".incomplete", ".part", ".tmp")))
    if missing or partial:
        return {**base, "checkpoint_complete": False, "missing_shards": missing, "partial_files": len(partial),
                "inventory_names_verified": True, "inference_ready": False,
                "reason": "download incomplete; shard headers, sizes and hashes not verified"}
    try:
        summary = verify_checkpoint(root, hashes=hashes)
    except CheckpointError as exc:
        raise ConfigurationError(f"checkpoint verification failed: {exc}") from exc
    return {**base, "checkpoint_complete": True, "missing_shards": [], "verified": summary,
            "inventory_names_verified": True, "inference_ready": False,
            "reason": "layout verified; real-checkpoint numerical validation not yet performed"}
