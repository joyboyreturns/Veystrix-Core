"""One-layer synthetic Metal probe at checkpoint geometry; never loads weights.

Records capacity and allocator peaks separately from the 12-layer projection.
Rates are packed-cache append and QSA tile operations, not model token rates.
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

STAGES = (4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bf16", "affine4", "affine8", "k16v4", "k8v4", "tiered4"), default="tiered4")
    parser.add_argument("--group", type=int, choices=(32, 64), default=64)
    parser.add_argument("--recent", type=int, default=8192)
    parser.add_argument("--max-stage", type=int, choices=STAGES, default=STAGES[-1])
    args = parser.parse_args(argv)

    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache
    from models.flashnext import kv_telemetry as T, qsa_chunk as Q
    from models.flashnext.mixed_kv import MixedPrecisionQSAKVCache
    from models.flashnext.quantized_kv import AffineQuantizedQSAKVCache
    from models.flashnext.request_admission import strict_read

    baseline = strict_read()

    def check():
        r = strict_read()
        if r.available < 3 * 2**30 or r.pressure != 1 or r.swapouts > baseline.swapouts or r.swap_used > baseline.swap_used:
            raise SystemExit("synthetic probe stopped at a stage boundary: host headroom, pressure or swap changed")
        return r

    if args.mode == "bf16":
        cache = QSAKVCache()
    elif args.mode in ("affine4", "affine8"):
        cache = AffineQuantizedQSAKVCache(args.group, int(args.mode[-1]))
    else:
        cache = MixedPrecisionQSAKVCache(args.mode, args.group, args.recent)
    mx.set_cache_limit(128 * 2**20)
    mx.random.seed(42)
    spec = SimpleNamespace(block_topk=512, compress_ratio=4, head_dim=128)
    for end in (n for n in STAGES if n <= args.max_stage):
        check()
        mx.reset_peak_memory()
        start = cache.offset
        began = time.perf_counter()
        keys = values = None
        while cache.offset < end:
            n = min(2048, end - cache.offset)
            k, v = (mx.random.normal((1, 2, n, 256)).astype(mx.bfloat16) for _ in range(2))
            raw = mx.random.normal((1, n, 128)).astype(mx.bfloat16)
            positions = mx.broadcast_to(mx.arange(cache.offset, cache.offset + n)[None, None, :], (3, 1, n))
            cache.update_indexer(raw, positions)
            keys, values = cache.update_and_fetch(k, v)
            mx.eval(cache.state)
        append_seconds = time.perf_counter() - began
        append_peak = mx.get_peak_memory()
        # Pool representative indexer activations once, then exercise exactly
        # the production selection, gather and GQA shapes for eight queries.
        blocks = end // 4
        pooled = mx.mean(cache.index_keys.reshape(1, 1, blocks, 4, 128).astype(mx.float32), axis=3).astype(mx.bfloat16)
        query = mx.random.normal((1, 4, 8, 128)).astype(mx.bfloat16)
        q = mx.random.normal((8, 24, 1, 256)).astype(mx.bfloat16)
        mx.eval(pooled, query, q)
        mx.clear_cache()
        mx.reset_peak_memory()
        attention_base = mx.get_active_memory()
        began = time.perf_counter()
        for _ in range(5):
            indices, valid = Q._selected_indices(spec, query, pooled, end, end - 8, blocks, 0, 8)
            gk, gv = Q._gather_kv(keys, values, cache, indices)
            out = mx.fast.scaled_dot_product_attention(q, gk, gv, scale=1/16, mask=valid.reshape(8, 1, 1, -1))
            mx.eval(out)
            del gk, gv, indices, valid, out
        attention_seconds = (time.perf_counter() - began) / 5
        reading = check()
        print(json.dumps({"mode": args.mode, "group": args.group, "tokens": end,
                          "one_layer": T.cache_breakdown([cache]),
                          "projected_model": T.projected_cache(end, args.mode, args.group, args.recent),
                          "one_layer_append_tokens_s": (end - start) / append_seconds,
                          "append_peak_active_bytes": append_peak,
                          "eight_query_seconds_mean": attention_seconds,
                          "attention_start_active_bytes": attention_base,
                          "attention_peak_active_bytes": mx.get_peak_memory(),
                          "mlx_cached_bytes": mx.get_cache_memory(),
                          "swap_growth_bytes": reading.swap_used - baseline.swap_used}), flush=True)
        del keys, values, pooled, query, q
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
