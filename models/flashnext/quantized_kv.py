"""Opt-in Affine 4-bit (INT4) Attention KV Cache for Flash-Next (Apple Silicon M4).

Quantizes only the 12 full attention layers (QSA), compressing key/value tensors by ~3.5x
using packed affine INT4 representation with per-group scales and zero-point biases (group size 64).
Preserves all 36 GatedDeltaNet linear recurrent layers in their exact original floating point precision.
Attention computation executes natively via Metal-accelerated quantized matrix multiplication
(`quantized_scaled_dot_product_attention` / `mx.quantized_matmul`).
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import mlx.core as mx
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache

FLAG_KV_DTYPE = "FLASHNEXT_KV_CACHE_DTYPE"       # "bf16" (default) or "affine4"
FLAG_KV_GROUP_SIZE = "FLASHNEXT_KV_GROUP_SIZE"   # default 64
FLAG_KV_BITS = "FLASHNEXT_KV_BITS"               # default 4

DEFAULT_GROUP_SIZE = 64
DEFAULT_BITS = 4


KV_DTYPES = ("bf16", "affine4", "affine8")


def kv_dtype() -> str:
    """Configured KV representation. An unknown value raises: silently falling back to bf16 (or to a default group
    size) would let a snapshot or a quality result be attributed to a configuration that never ran."""
    value = (os.environ.get(FLAG_KV_DTYPE, "bf16") or "bf16").strip().lower()
    if value not in KV_DTYPES:
        raise ValueError(f"{FLAG_KV_DTYPE} must be one of {', '.join(KV_DTYPES)}, not {value!r}")
    return value


def fingerprint_parts() -> dict:
    """What a KV snapshot's numerics depend on. Empty for bf16 so existing bf16 fingerprints are unchanged."""
    if kv_dtype() == "bf16":
        return {}
    return {"kv_dtype": kv_dtype(), "kv_bits": bits(), "kv_group_size": group_size()}


def is_affine4_enabled() -> bool:
    """True for any quantized KV mode ("affine4" and "affine8"); the name is kept for callers."""
    return kv_dtype() in ("affine4", "affine8")


def group_size() -> int:
    raw = os.environ.get(FLAG_KV_GROUP_SIZE, str(DEFAULT_GROUP_SIZE))
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"{FLAG_KV_GROUP_SIZE} must be 32, 64 or 128, not {raw!r}") from None
    if val not in (32, 64, 128):
        raise ValueError(f"{FLAG_KV_GROUP_SIZE} must be 32, 64 or 128, got {val}")
    return val


def bits() -> int:
    """Bits per element: the mode decides ("affine8" is 8, "affine4" is 4); FLASHNEXT_KV_BITS may only agree."""
    mode_bits = 8 if kv_dtype() == "affine8" else DEFAULT_BITS
    raw = os.environ.get(FLAG_KV_BITS)
    if raw in (None, ""):
        return mode_bits
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"{FLAG_KV_BITS} must be 4 or 8, not {raw!r}") from None
    if val not in (4, 8) or (kv_dtype() != "bf16" and val != mode_bits):
        raise ValueError(f"{FLAG_KV_BITS}={val} contradicts {FLAG_KV_DTYPE}={kv_dtype()}")
    return val


class AffineQuantizedQSAKVCache(QSAQuantizedKVCache):
    """Packed affine INT4/INT8 quantized KV cache for Qwen Sparse Attention (QSA).

    Keys and values are quantized into:
    - `data`: packed uint32 array (8 elements per uint32 for 4-bit, 4 for 8-bit)
    - `scales`: floating-point per-group scale factor
    - `biases`: floating-point per-group zero-point offset
    Indexer keys and positions remain in unquantized floating point to ensure exact
    block selection and routing.
    """

    def __init__(self, group_size: int = DEFAULT_GROUP_SIZE, bits: int = DEFAULT_BITS):
        super().__init__(group_size=group_size, bits=bits)

    @property
    def detailed_nbytes(self) -> dict[str, int]:
        """Breakdown of memory consumption in bytes."""
        keys_data = self.keys[0].nbytes if self.keys is not None else 0
        keys_scales = self.keys[1].nbytes if self.keys is not None else 0
        keys_biases = self.keys[2].nbytes if self.keys is not None else 0
        
        vals_data = self.values[0].nbytes if self.values is not None else 0
        vals_scales = self.values[1].nbytes if self.values is not None else 0
        vals_biases = self.values[2].nbytes if self.values is not None else 0
        
        idx_keys = self.index_keys.nbytes if self.index_keys is not None else 0
        idx_pos = self.index_position_ids.nbytes if self.index_position_ids is not None else 0
        
        return {
            "quantized_data_bytes": keys_data + vals_data,
            "metadata_bytes": keys_scales + keys_biases + vals_scales + vals_biases,
            "indexer_bytes": idx_keys + idx_pos,
            "total_bytes": keys_data + keys_scales + keys_biases + vals_data + vals_scales + vals_biases + idx_keys + idx_pos,
        }

    def dequantize(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Dequantize stored keys and values to floating point (reference / verification)."""
        if self.keys is None or self.values is None or self.offset == 0:
            return None, None
        keys = mx.dequantize(
            mx.contiguous(self.keys[0][..., :self.offset, :]),
            mx.contiguous(self.keys[1][..., :self.offset, :]),
            mx.contiguous(self.keys[2][..., :self.offset, :]),
            group_size=self.group_size,
            bits=self.bits,
        )
        values = mx.dequantize(
            mx.contiguous(self.values[0][..., :self.offset, :]),
            mx.contiguous(self.values[1][..., :self.offset, :]),
            mx.contiguous(self.values[2][..., :self.offset, :]),
            group_size=self.group_size,
            bits=self.bits,
        )
        return keys, values


def make_cache(
    language_model: Any,
    dtype: Optional[str] = None,
    grp_size: Optional[int] = None,
    n_bits: Optional[int] = None,
) -> List[Any]:
    """Construct the layer cache list.

    All 36 linear GatedDeltaNet layers use ArraysCache.
    The 12 full attention layers use AffineQuantizedQSAKVCache when dtype=="affine4",
    or standard QSAKVCache when dtype=="bf16".
    """
    use_affine4 = (dtype in ("affine4", "affine8")) if dtype is not None else is_affine4_enabled()
    g_size = grp_size if grp_size is not None else group_size()
    b = n_bits if n_bits is not None else bits()

    caches = []
    for layer in language_model.layers:
        if getattr(layer, "is_linear", False):
            size = 4 if hasattr(layer, "ple") and layer.ple is not None else 2
            caches.append(ArraysCache(size=size))
        else:
            if use_affine4:
                caches.append(AffineQuantizedQSAKVCache(group_size=g_size, bits=b))
            else:
                caches.append(QSAKVCache())
    return caches


def is_quantized_cache(cache_list: Sequence[Any]) -> bool:
    return any(isinstance(c, (QSAQuantizedKVCache, AffineQuantizedQSAKVCache)) for c in cache_list)


_ORIGINAL_MAKE_CACHE = [None]


def apply() -> bool:
    """Wrap LanguageModel.make_cache to return AffineQuantizedQSAKVCache when enabled."""
    if _ORIGINAL_MAKE_CACHE[0] is not None:
        return False
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    _ORIGINAL_MAKE_CACHE[0] = LanguageModel.make_cache

    def wrapped_make_cache(self):
        if is_affine4_enabled():
            return make_cache(self)
        return _ORIGINAL_MAKE_CACHE[0](self)

    LanguageModel.make_cache = wrapped_make_cache
    return True


def restore() -> bool:
    """Restore original LanguageModel.make_cache (unit tests)."""
    if _ORIGINAL_MAKE_CACHE[0] is None:
        return False
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    LanguageModel.make_cache = _ORIGINAL_MAKE_CACHE[0]
    _ORIGINAL_MAKE_CACHE[0] = None
    return True
