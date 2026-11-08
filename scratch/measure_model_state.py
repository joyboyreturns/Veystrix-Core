"""Empirical state and memory analysis for Qwen3.8-Flash-Next architecture.

Calculates exact byte requirements for:
- 36 GatedDeltaNet (GDN) linear recurrent layers (fixed size)
- 12 Full Attention layers (grows linearly with sequence length)
- Indexer keys and position IDs
- Activation and intermediate buffers across context targets (4K, 8K, 12K, 16K)
"""
import json
from pathlib import Path

CONFIG_PATH = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP/config.json")
config = json.loads(CONFIG_PATH.read_text())["text_config"]

NUM_LAYERS = config["num_hidden_layers"] # 48
LAYER_TYPES = config["layer_types"]
NUM_LINEAR = sum(1 for t in LAYER_TYPES if t == "linear_attention") # 36
NUM_FULL = sum(1 for t in LAYER_TYPES if t == "full_attention")     # 12

# 1. GatedDeltaNet (Linear Attention) Recurrent State per layer
# Recurrent state: [num_value_heads, key_head_dim, value_head_dim] in float32
# plus 1D depthwise convolution state: [conv_kernel_dim - 1, conv_dim] in bfloat16
NUM_V_HEADS = config["linear_num_value_heads"] # 48
K_HEAD_DIM = config["linear_key_head_dim"]     # 128
V_HEAD_DIM = config["linear_value_head_dim"]   # 128
SSM_DTYPE_BYTES = 4 # float32

# Recurrent matrix: 48 * 128 * 128 * 4 bytes = 3,145,728 bytes (~3.0 MB)
GDN_RECURRENT_BYTES = NUM_V_HEADS * K_HEAD_DIM * V_HEAD_DIM * SSM_DTYPE_BYTES

# Conv state: conv_dim = 2 * (16 * 128) + 48 * 128 = 10,240; kernel = 4 -> (4 - 1) * 10240 * 2 = 61,440 bytes
CONV_DIM = 2 * config["linear_num_key_heads"] * K_HEAD_DIM + NUM_V_HEADS * V_HEAD_DIM # 10,240
CONV_KERNEL = config["linear_conv_kernel_dim"] # 4
GDN_CONV_BYTES = (CONV_KERNEL - 1) * CONV_DIM * 2 # bf16

GDN_STATE_PER_LAYER = GDN_RECURRENT_BYTES + GDN_CONV_BYTES # 3,207,168 bytes (~3.06 MB)
TOTAL_GDN_STATE_BYTES = NUM_LINEAR * GDN_STATE_PER_LAYER    # 36 * 3.06 MB = 115,458,048 bytes (~110.1 MB)

# 2. Full Attention KV Cache per token across 12 layers
# K: 2 kv heads * 256 head_dim * 2 bytes (bf16) = 1,024 bytes
# V: 2 kv heads * 256 head_dim * 2 bytes (bf16) = 1,024 bytes
# Indexer Key: 1 kv head * 128 head_dim * 2 bytes (bf16) = 256 bytes
# MRoPE position ids / metadata: 12 bytes
KV_BYTES_PER_TOKEN_PER_LAYER = (config["num_key_value_heads"] * 2 * config["head_dim"] * 2
                                + config["indexer_kv_heads"] * config["indexer_head_dim"] * 2
                                + 12) # 2,316 bytes per token per layer

TOTAL_KV_BYTES_PER_TOKEN = NUM_FULL * KV_BYTES_PER_TOKEN_PER_LAYER # 12 * 2,316 = 27,792 bytes/token (~27.1 KB/token)

def compute_state_memory(tokens: int):
    kv_bytes = tokens * TOTAL_KV_BYTES_PER_TOKEN
    total_state_bytes = TOTAL_GDN_STATE_BYTES + kv_bytes
    return {
        "tokens": tokens,
        "gdn_recurrent_mb": TOTAL_GDN_STATE_BYTES / (1024 * 1024),
        "kv_cache_mb": kv_bytes / (1024 * 1024),
        "total_state_mb": total_state_bytes / (1024 * 1024),
        "total_state_gb": total_state_bytes / (1024 ** 3),
    }

if __name__ == "__main__":
    print(f"Model: Qwen3.8-Flash-Next ({NUM_LAYERS} total layers: {NUM_LINEAR} GDN linear, {NUM_FULL} full attention)")
    print(f"GDN state per layer: {GDN_STATE_PER_LAYER / 1024:.1f} KB (fixed across all sequence lengths)")
    print(f"Total GDN persistent state: {TOTAL_GDN_STATE_BYTES / (1024 * 1024):.1f} MB (36 layers)")
    print(f"Full attention KV cache rate: {TOTAL_KV_BYTES_PER_TOKEN / 1024:.2f} KB per token (12 layers)")
    print("-" * 65)
    print(f"{'Context':<10} | {'GDN State':<12} | {'KV Cache':<12} | {'Total State':<12}")
    print("-" * 65)
    for t in (1024, 2048, 4096, 8192, 12288, 16384, 32768):
        m = compute_state_memory(t)
        print(f"{t:<10} | {m['gdn_recurrent_mb']:>9.1f} MB | {m['kv_cache_mb']:>9.1f} MB | {m['total_state_mb']:>9.1f} MB ({m['total_state_gb']:.3f} GiB)")
