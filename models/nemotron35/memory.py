"""Conservative arithmetic for planning, not a measured residency guarantee."""
from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryEstimate:
    non_expert_bytes: int
    mamba_state_bytes: int
    attention_kv_bytes: int
    expert_pool_bytes: int
    activation_bytes: int
    reserve_bytes: int
    @property
    def total_bytes(self):
        return sum(vars(self).values())

def estimate(*, config, context_tokens, non_expert_bytes, expert_pool_bytes,
             activation_bytes, reserve_bytes, kv_element_bytes=2, state_element_bytes=4):
    """Estimate single-session allocation; caller must measure non-expert and temporary bytes.

    KV elements hold K and V. Mamba stores [heads, head_dim, state_dim] and the
    depthwise convolution tail [kernel-1, heads*head_dim+2*groups*state_dim].
    """
    inputs = (context_tokens, non_expert_bytes, expert_pool_bytes, activation_bytes, reserve_bytes)
    if any(not isinstance(x, int) or x < 0 for x in inputs) or kv_element_bytes <= 0 or state_element_bytes <= 0:
        raise ValueError('memory inputs must be nonnegative integers')
    blocks = config['layers_block_type']
    heads, dim, state = config['mamba_num_heads'], config['mamba_head_dim'], config['ssm_state_size']
    conv = heads * dim + 2 * config['n_groups'] * state
    recurrent = blocks.count('mamba') * (heads * dim * state * state_element_bytes +
        (config['conv_kernel'] - 1) * conv * kv_element_bytes)
    kv = blocks.count('attention') * context_tokens * 2 * config['num_key_value_heads'] * config['head_dim'] * kv_element_bytes
    return MemoryEstimate(non_expert_bytes, recurrent, kv, expert_pool_bytes, activation_bytes, reserve_bytes)

def require_budget(estimate, limit_bytes):
    if limit_bytes <= 0 or estimate.total_bytes > limit_bytes:
        raise MemoryError(f'estimated allocation {estimate.total_bytes} exceeds allowed budget {limit_bytes}')
