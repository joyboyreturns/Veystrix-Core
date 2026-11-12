"""Logical live cache attribution; MLX allocator counters remain authoritative.

Array extents exclude allocator overhead. The breakdown is not added to MLX
active bytes, nor are mmap extents added to process RSS or macOS file pages.
Snapshot bytes are accounted for by PrefixCache, outside this breakdown.
"""
from __future__ import annotations


def cache_breakdown(caches):
    totals = dict(attention_k_bytes=0, attention_v_bytes=0, quantization_metadata_bytes=0,
                  qsa_indexer_bytes=0, qsa_pooled_bytes=0, gdn_state_bytes=0)
    seen = set()
    workspace = None
    offsets = []

    def count(name, array):
        if array is not None and id(array) not in seen:
            seen.add(id(array))
            totals[name] += int(array.nbytes)

    for c in caches or ():
        if not hasattr(c, "index_keys"):
            for array in getattr(c, "cache", ()):
                count("gdn_state_bytes", array)
            continue
        offsets.append(int(c.offset))
        for name in ("keys", "values", "recent_keys", "recent_values"):
            value = getattr(c, name, None)
            category = "attention_k_bytes" if name.endswith("keys") else "attention_v_bytes"
            if isinstance(value, (tuple, list)):
                count(category, value[0])
                for array in value[1:]:
                    count("quantization_metadata_bytes", array)
            else:
                count(category, value)
        count("qsa_indexer_bytes", c.index_keys)
        count("qsa_indexer_bytes", c.index_position_ids)
        pooled = getattr(c, "_flashnext_pooled_keys", None)
        if pooled is not None:
            # The first raw arrays usually alias the primary indexer state.
            count("qsa_indexer_bytes", pooled[1])
            count("qsa_indexer_bytes", pooled[2])
            count("qsa_pooled_bytes", pooled[-1])
        current = getattr(c, "_flashnext_qsa_workspace", None)
        if current and (workspace is None or current["largest_indexer_score_tensor_bytes"] >
                        workspace["largest_indexer_score_tensor_bytes"]):
            workspace = current
    return {**totals, "logical_cache_bytes": sum(totals.values()),
            "min_attention_offset": min(offsets, default=0), "max_attention_offset": max(offsets, default=0),
            "last_qsa_workspace_shape": workspace}


def projected_cache(tokens, mode="tiered4", group_size=64, recent_tokens=8192):
    """Logical populated rows at native Flash-Next geometry, before capacity slack."""
    from .prefill_budget import FULL_LAYERS, INDEX_BYTES_PER_TOKEN, GDN_STATE_BYTES, LINEAR_LAYERS

    kb, vb = {"bf16": (16, 16), "affine4": (4, 4), "affine8": (8, 8),
              "k16v4": (16, 4), "k8v4": (8, 4), "tiered4": (4, 4)}[mode]
    old = tokens  # the primary representation includes the small recent window
    recent = tokens - max(0, (tokens - recent_tokens) // 256 * 256) if mode == "tiered4" else 0
    channels = FULL_LAYERS * 2 * 256
    packed = channels * (old * (kb + vb) + recent * 16) // 8
    quantized_rows = old * (int(kb < 16) + int(vb < 16)) + recent * 2
    metadata = channels * quantized_rows * 4 // group_size
    indexer = FULL_LAYERS * INDEX_BYTES_PER_TOKEN * tokens
    recurrence = LINEAR_LAYERS * GDN_STATE_BYTES
    return {"attention_data_bytes": packed, "quantization_metadata_bytes": metadata,
            "attention_total_bytes": packed + metadata, "qsa_indexer_bytes_upper": indexer,
            "gdn_state_bytes_estimate": recurrence,
            "live_state_bytes_estimate": packed + metadata + indexer + recurrence}
