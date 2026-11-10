"""Tiny MLX Nemotron-H reference execution, for synthetic validation only.

Uses the installed mlx-lm Nemotron-H implementation rather than copying or approximating
its architecture. It holds all weights in memory and MUST NOT be used for the real model.
"""
from mlx_lm.models.nemotron_h import Model, ModelArgs


def make_synthetic_model(seed=1, width=16):
    """Return a deterministic 3-block Mamba/attention/MoE model; never loads a checkpoint."""
    import mlx.core as mx
    mx.random.seed(seed)
    if width not in (16, 32):
        raise ValueError("synthetic width must be 16 or 32")
    args = ModelArgs(model_type="nemotron_h", vocab_size=32, hidden_size=width,
        intermediate_size=width * 2, num_hidden_layers=3, max_position_embeddings=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=width // 4, attention_bias=False,
        mamba_num_heads=4, mamba_head_dim=width // 4, mamba_proj_bias=False,
        ssm_state_size=32, conv_kernel=4, n_groups=2, mlp_bias=False,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        layers_block_type=["mamba", "attention", "moe"],
        moe_intermediate_size=width, moe_shared_expert_intermediate_size=width,
        moe_latent_size=None, n_group=1, n_routed_experts=4, n_shared_experts=1,
        topk_group=1, num_experts_per_tok=2, norm_topk_prob=True, routed_scaling_factor=2.5)
    model = Model(args)
    # Untrained upstream router weights default to zero; tied experts can be
    # chosen differently for different batch shapes. Use deterministic,
    # non-degenerate synthetic scores instead of comparing tie-breaking.
    model.backbone.layers[2].mixer.gate.weight = mx.random.normal((4, width)) * 0.1
    return model
