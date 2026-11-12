"""Block-backed affine INT8 attention for the streamed Nemotron-H mixer."""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache
from mlx_lm.models.nemotron_h import NemotronHAttention

from .runtime import Cancelled

FORMAT = "affine-int8-g64"
GROUP_SIZE = 64
BLOCK_TOKENS = 4096
QUERY_TOKENS = 128


class AttentionCache:
    def __init__(self):
        self.blocks = []
        self.offset = 0
        self.cancelled = lambda: False

    def make_mask(self, n, *, return_array=False, window_size=None):
        if return_array or window_size is not None:
            raise ValueError("Nemotron INT8 attention uses a causal mask")
        return "causal" if n > 1 else None

    def append(self, keys, values):
        consumed = 0
        while consumed < keys.shape[-2]:
            if not self.blocks or self.blocks[-1].offset == BLOCK_TOKENS:
                block = QuantizedKVCache(group_size=GROUP_SIZE, bits=8)
                block.step = BLOCK_TOKENS
                self.blocks.append(block)
            block = self.blocks[-1]
            take = min(keys.shape[-2] - consumed, BLOCK_TOKENS - block.offset)
            end = consumed + take
            block.update_and_fetch(keys[..., consumed:end, :], values[..., consumed:end, :])
            self.offset += take
            consumed = end

    @property
    def state(self):
        return [array for block in self.blocks for group in block.state for array in group]

    @state.setter
    def state(self, arrays):
        if not arrays or len(arrays) % 6:
            raise ValueError("Invalid INT8 attention block count")
        blocks = []
        layout = None
        for start in range(0, len(arrays), 6):
            keys, values = tuple(arrays[start:start + 3]), tuple(arrays[start + 3:start + 6])
            packed = keys[0]
            if packed.ndim != 4:
                raise ValueError("Invalid packed key shape")
            b, heads, n, words = packed.shape
            if b != 1 or not 0 < n <= BLOCK_TOKENS or words % (GROUP_SIZE // 4):
                raise ValueError("Invalid INT8 block dimensions")
            if start + 6 < len(arrays) and n != BLOCK_TOKENS:
                raise ValueError("Partial nonterminal INT8 block")
            if layout is not None and layout != (heads, words, keys[1].dtype):
                raise ValueError("INT8 blocks have inconsistent layouts")
            layout = (heads, words, keys[1].dtype)
            for group in (keys, values):
                if group[0].shape != packed.shape or group[0].dtype != mx.uint32:
                    raise ValueError("Invalid packed INT8 payload")
                if any(a.shape != (b, heads, n, words * 4 // GROUP_SIZE) or a.dtype != keys[1].dtype
                       for a in group[1:]) or keys[1].dtype not in (mx.bfloat16, mx.float16, mx.float32):
                    raise ValueError("Invalid INT8 scales or biases")
            block = QuantizedKVCache(group_size=GROUP_SIZE, bits=8)
            block.step = BLOCK_TOKENS
            block.state = (keys, values)
            block.offset = n
            blocks.append(block)
        self.blocks = blocks
        self.offset = sum(block.offset for block in blocks)


def attend(queries, cache, scale):
    """Exact full-history softmax over quantized K/V, with bounded score scratch."""
    if len(cache.blocks) == 1:
        keys, values = cache.blocks[0].state
        return mx.fast.scaled_dot_product_attention(
            queries, mx.dequantize(*keys, group_size=GROUP_SIZE, bits=8),
            mx.dequantize(*values, group_size=GROUP_SIZE, bits=8),
            scale=scale, mask="causal" if queries.shape[-2] > 1 else None)
    batch, heads, length, dim = queries.shape
    kv_heads = cache.blocks[0].keys[0].shape[-3]
    dtype = queries.dtype
    offset = cache.offset - length
    outputs = []
    for start in range(0, length, QUERY_TOKENS):
        end = min(length, start + QUERY_TOKENS)
        q = queries[..., start:end, :].astype(mx.float32) * scale
        q = q.reshape(batch, kv_heads, heads // kv_heads, end - start, dim)
        positions = mx.arange(offset + start, offset + end)
        previous_max = normalizer = weighted = None
        for block_index, block in enumerate(cache.blocks):
            first = block_index * BLOCK_TOKENS
            if first >= offset + end:
                break
            if cache.cancelled():
                raise Cancelled()
            keys, values = block.state
            keys = tuple(a[..., None, :, :] for a in keys)
            values = tuple(a[..., None, :, :] for a in values)
            scores = mx.quantized_matmul(q, *keys, transpose=True, group_size=GROUP_SIZE, bits=8)
            if first + block.offset > offset + start + 1:
                valid = positions[:, None] >= mx.arange(first, first + block.offset)[None, :]
                scores = mx.where(valid, scores, -float("inf"))
            maximum = mx.max(scores, axis=-1, keepdims=True)
            if previous_max is not None:
                maximum = mx.maximum(maximum, previous_max)
            # Every query has at least one visible key in the first block.
            weights = mx.exp(scores - maximum)
            total = mx.sum(weights, axis=-1, keepdims=True)
            result = mx.quantized_matmul(weights, *values, transpose=False, group_size=GROUP_SIZE, bits=8)
            if previous_max is not None:
                correction = mx.exp(previous_max - maximum)
                total = total + normalizer * correction
                result = result + weighted * correction
            previous_max, normalizer, weighted = maximum, total, result
            mx.eval(previous_max, normalizer, weighted)
        outputs.append((weighted / normalizer).reshape(batch, heads, end - start, dim).astype(dtype))
    return mx.concatenate(outputs, axis=2)


class StreamedAttention(NemotronHAttention):
    def __call__(self, x, mask=None, cache=None):
        if not isinstance(cache, AttentionCache):
            return super().__call__(x, mask, cache)
        if mask is not None and (not isinstance(mask, str) or mask != "causal"):
            raise ValueError("Unsupported Nemotron INT8 attention mask")
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(batch, length, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch, length, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        cache.append(keys, values)
        output = attend(queries, cache, self.scale)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))
