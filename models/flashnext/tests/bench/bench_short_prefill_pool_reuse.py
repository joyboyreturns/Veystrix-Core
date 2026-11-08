"""Microbenchmark: Short-Prefill Resident Decode Pool Reuse vs SSD Fallback (Goal 1.6).

Compares prefill latency, SSD expert reads, pool hits, and verifies exact numerical equality
between the baseline SSD read path and the opt-in resident-pool reuse path.
"""
from __future__ import annotations

import os
import tempfile
import time
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_cache, expert_io_stats, expert_pool
from models.flashnext.store import SafeTensorStore
from models.flashnext.tests.unit.test_prefill_ring import (
    EXPERTS,
    HIDDEN,
    LAYERS,
    build_checkpoint,
    prefix_of,
    silu_glu,
)


def run_benchmark(num_trials: int = 50, tokens: int = 21):
    with tempfile.TemporaryDirectory() as d:
        build_checkpoint(d)
        store = SafeTensorStore(d)

        # Build 2 layers
        layers = [
            expert_cache.StreamingSwitchGLU(
                store, prefix_of(l), 32, 4, "affine", silu_glu, layer_id=l
            )
            for l in LAYERS
        ]

        # Allocate 64 MB decode pool
        pool = expert_pool.ExpertPool(store, 64 * 1024 * 1024, 32)
        store._flashnext_expert_pool = pool

        # Pre-warm pool with experts 0..5 on each layer (simulating warm decode pool after initial prompt)
        for l in LAYERS:
            slots, misses = pool.assign(l, [0, 1, 2, 3, 4, 5])
            futures = pool.fill(prefix_of(l), misses, expert_cache._submit_read, 2)
            for f in futures:
                f.result()

        # Fixed 21-token input with top-2 routing (mostly in warm pool, some misses)
        mx.random.seed(2026)
        x = mx.random.normal((1, tokens, HIDDEN)).astype(mx.bfloat16)
        # Routes: tokens choose among [0, 1, 2, 3, 4, 5] (hits) plus [6, 7] (misses)
        indices = mx.random.randint(0, 8, (1, tokens, 2)).astype(mx.uint32)

        # 1. Baseline Run (Reuse OFF)
        expert_io_stats.reset_stats()
        t0 = time.perf_counter()
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "0",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            for _ in range(num_trials):
                for glu in layers:
                    y_base = glu(x, indices)
                    mx.eval(y_base)
        t_baseline = time.perf_counter() - t0
        stats_base = expert_io_stats.get_stats()

        # 2. Optimized Run (Reuse ON)
        expert_io_stats.reset_stats()
        t0 = time.perf_counter()
        with mock.patch.dict(os.environ, {expert_io_stats.FLAG_POOL_REUSE: "1",
                                          expert_io_stats.FLAG_DIAGNOSTICS: "1"}):
            for _ in range(num_trials):
                for glu in layers:
                    y_opt = glu(x, indices)
                    mx.eval(y_opt)
        t_opt = time.perf_counter() - t0
        stats_opt = expert_io_stats.get_stats()

        # 3. Assert bit-identical output
        assert mx.array_equal(y_base.view(mx.uint16), y_opt.view(mx.uint16)), "Outputs diverged!"

        return {
            "trials": num_trials,
            "tokens": tokens,
            "baseline_s": t_baseline,
            "optimized_s": t_opt,
            "speedup": t_baseline / t_opt if t_opt > 0 else 1.0,
            "baseline_ssd_reads": stats_base.ssd_reads,
            "optimized_ssd_reads": stats_opt.ssd_reads,
            "reads_avoided": stats_base.ssd_reads - stats_opt.ssd_reads,
            "baseline_ssd_bytes": stats_base.ssd_bytes_read,
            "optimized_ssd_bytes": stats_opt.ssd_bytes_read,
            "bytes_saved": stats_opt.pool_bytes_saved,
            "pool_hits": stats_opt.pool_hits,
            "pool_misses": stats_opt.pool_misses,
            "pool_hit_rate": stats_opt.hit_rate(),
        }


def main():
    print("=== Running Short-Prefill Resident Decode Pool Reuse Benchmark ===")
    results = run_benchmark(num_trials=50, tokens=21)
    print(f"Trials:                 {results['trials']} requests of {results['tokens']} tokens")
    print(f"Baseline Time:          {results['baseline_s']:.4f}s")
    print(f"Optimized Time:         {results['optimized_s']:.4f}s")
    print(f"Speedup:                {results['speedup']:.2f}x")
    print(f"Baseline SSD Bytes:     {results['baseline_ssd_bytes'] / (1024*1024):.2f} MB")
    print(f"Optimized SSD Bytes:    {results['optimized_ssd_bytes'] / (1024*1024):.2f} MB")
    print(f"SSD Bytes Avoided:      {results['bytes_saved'] / (1024*1024):.2f} MB")
    print(f"Pool Hits:              {results['pool_hits']} ({results['pool_hit_rate']*100:.1f}% hit rate)")
    print(f"Pool Misses:            {results['pool_misses']}")
    print("Verification:           Bit-identical numerical equality PASSED")


if __name__ == "__main__":
    main()
