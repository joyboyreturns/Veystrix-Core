"""Restore overhead of the prefix cache on synthetic state of the real shapes (no model, no checkpoint).

Builds the cache the real model would hold after N tokens (36 GatedDeltaNet layers with a fp32
recurrent state, 12 QSA layers with KV and indexer keys) from random values, snapshots it at 2,048-token
chunk edges, and times capture, SSD write, an SSD restore after a "restart" and a RAM restore. These
numbers are restoration cost, NOT model prefill speed.

    python -m models.flashnext.tests.bench.bench_prefix_cache_synthetic [--tokens 10000] [--dir DIR]
"""
from __future__ import annotations

import argparse
import gc
import shutil
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import QSAKVCache

from models.flashnext import prefix_cache as P

LAYERS, FULL_EVERY, CHUNK = 48, 4, 2048
PLE_LAYER = 2


def build_cache(tokens: int):
    """The cache after ``tokens`` tokens, with the real classes, shapes and dtypes."""
    caches = []
    for layer in range(LAYERS):
        if (layer + 1) % FULL_EVERY:
            cache = ArraysCache(size=4 if layer == PLE_LAYER else 2)
            cache.cache[0] = mx.random.normal((1, 3, 10240)).astype(mx.bfloat16)
            cache.cache[1] = mx.random.normal((1, 48, 128, 128)).astype(mx.float32)
            if layer == PLE_LAYER:
                cache.cache[2] = mx.random.normal((1, 3, 2560)).astype(mx.bfloat16)
                cache.cache[3] = mx.random.normal((1, 3, 2560)).astype(mx.bfloat16)
        else:
            cache = QSAKVCache()
            keys = mx.random.normal((1, 2, tokens, 256)).astype(mx.bfloat16)
            values = mx.random.normal((1, 2, tokens, 256)).astype(mx.bfloat16)
            index_keys = mx.random.normal((1, tokens, 128)).astype(mx.bfloat16)
            positions = mx.broadcast_to(mx.arange(tokens, dtype=mx.int32)[None, None], (3, 1, tokens))
            cache.state = (keys, values, index_keys, mx.array(positions))
        caches.append(cache)
    mx.eval([P._state_arrays(c, P.kind_of(c)) for c in caches])
    return caches


def timed(fn):
    began = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - began


def run(tokens: int, directory: Path) -> dict:
    fp = P.fingerprint(model="synthetic")
    ids = list(range(tokens + 8))
    caches = build_cache(tokens)
    edges = [e for e in range(CHUNK, tokens, CHUNK)] + [tokens]
    out = {"tokens": tokens, "edges": len(edges)}
    store = P.PrefixCache(fp, 1 << 40, directory, 1 << 42)
    # Chained puts need the cache at each edge; reuse the final cache's prefix by slicing (sizes and
    # timings are what matter here, so a per-edge cache is built from slices of the full one).
    capture_s = write_s = 0.0
    last = None
    for edge in edges:
        prefix = _prefix(caches, edge)
        (entry, took) = timed(lambda: P.capture(prefix, ids[:edge], fp, last))
        entry._tokens = tuple(ids[:edge])
        capture_s += took
        store._tokens[entry.key] = entry._tokens
        store._lengths.add(entry.n_tokens)
        store._admit(entry)
        _, took = timed(lambda: store._persist(entry))
        write_s += took
        last = entry
    out.update(capture_s=capture_s, ssd_write_s=write_s, ssd_bytes=store.stats.ssd_bytes_written,
               last_entry_mb=last.nbytes() / 1e6)
    # RAM restore of the whole chain.
    (hit_caches), ram_s = timed(lambda: store.lookup(ids, lambda: build_empty())[1])
    out["ram_restore_s"] = ram_s
    del hit_caches, store
    gc.collect()
    mx.clear_cache()
    for verify in (True, False):
        (opened, scan_s) = timed(lambda: P.PrefixCache(fp, 1 << 40, directory, 1 << 42, verify_hash=verify))
        (result, ssd_s) = timed(lambda: opened.lookup(ids, lambda: build_empty()))
        hit, restored = result
        out[f"restart_scan_s"] = scan_s
        out[f"ssd_restore_s_verify_{verify}"] = ssd_s
        out["ssd_bytes_read"] = hit.bytes_loaded
        out["hit_tokens"] = hit.n_tokens
        del restored, opened
        gc.collect()
        mx.clear_cache()
    return out


def build_empty():
    caches = []
    for layer in range(LAYERS):
        if (layer + 1) % FULL_EVERY:
            caches.append(ArraysCache(size=4 if layer == PLE_LAYER else 2))
        else:
            caches.append(QSAKVCache())
    return caches


def _prefix(caches, edge):
    """The same cache truncated to ``edge`` tokens (new objects, so capture sees a consistent cache)."""
    out = []
    for cache in caches:
        if P.kind_of(cache) == "arrays":
            clone = ArraysCache(size=len(cache.cache))
            clone.cache = list(cache.cache)
            out.append(clone)
        else:
            clone = QSAKVCache()
            clone.state = (cache.keys[..., :edge, :], cache.values[..., :edge, :], cache.index_keys[:, :edge],
                           cache.index_position_ids[..., :edge])
            out.append(clone)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 4096, 10_000])
    parser.add_argument("--dir", default=None, help="directory on the SSD under test (default: a temp dir)")
    args = parser.parse_args()
    root = Path(args.dir or tempfile.mkdtemp(prefix="prefix-cache-bench-"))
    print(f"{'tokens':>7} {'edges':>5} {'capture':>8} {'ssd write':>10} {'ssd MB':>8} {'ram restore':>12} "
          f"{'scan':>6} {'ssd restore (sha256)':>21} {'ssd restore (no hash)':>22}")
    for tokens in args.tokens:
        directory = root / f"t{tokens}"
        shutil.rmtree(directory, ignore_errors=True)
        r = run(tokens, directory)
        print(f"{r['tokens']:>7} {r['edges']:>5} {r['capture_s']:7.3f}s {r['ssd_write_s']:9.3f}s "
              f"{r['ssd_bytes'] / 1e6:8.1f} {r['ram_restore_s']:11.3f}s {r['restart_scan_s']:5.3f}s "
              f"{r['ssd_restore_s_verify_True']:20.3f}s {r['ssd_restore_s_verify_False']:21.3f}s")
        shutil.rmtree(directory, ignore_errors=True)
    if not args.dir:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
