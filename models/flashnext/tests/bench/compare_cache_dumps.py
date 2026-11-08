"""Compare two cache dumps written by ``bench_prefill_memory --dump-dir`` (offline, no model).

Reports, per tensor, whether the two are bit-identical and, if not, how far apart they are. Bit-identical
means identical bytes. Anything else is a numerical difference and is reported as one, separately from
whether generated tokens agree.

    python -m models.flashnext.tests.bench.compare_cache_dumps A.safetensors B.safetensors
"""
from __future__ import annotations

import argparse
import json
import sys

import mlx.core as mx


def _bytes_equal(a, b) -> bool:
    return bool(mx.array_equal(a.view(mx.uint8), b.view(mx.uint8)).item()) if a.dtype == b.dtype else False


def compare(path_a: str, path_b: str) -> dict:
    a, b = mx.load(path_a), mx.load(path_b)
    rows, missing = [], sorted(set(a) ^ set(b))
    for name in sorted(set(a) & set(b)):
        x, y = a[name], b[name]
        row = {"tensor": name, "shape": list(x.shape), "dtype": str(x.dtype)}
        if x.shape != y.shape or x.dtype != y.dtype:
            row.update(identical=False, reason=f"shape or dtype differs: {x.shape} {x.dtype} vs {y.shape} {y.dtype}")
            rows.append(row)
            continue
        same = _bytes_equal(x, y)
        row["identical"] = same
        if not same:
            wide_x, wide_y = x.astype(mx.float32), y.astype(mx.float32)
            diff = mx.abs(wide_x - wide_y)
            scale = mx.maximum(mx.abs(wide_x), mx.abs(wide_y))
            denominator = mx.maximum(scale, mx.array(1e-30))
            row.update(
                max_abs=float(mx.max(diff).item()),
                max_rel=float(mx.max(diff / denominator).item()),
                differing_elements=int(mx.sum(diff > 0).item()), elements=int(x.size),
                max_abs_value=float(mx.max(scale).item()))
        rows.append(row)
    different = [r for r in rows if not r["identical"]]
    return {
        "tensors": len(rows), "identical_tensors": len(rows) - len(different),
        "only_in_one": missing, "all_identical": not different and not missing,
        "worst_abs": max((r.get("max_abs", 0.0) for r in different), default=0.0),
        "worst_rel": max((r.get("max_rel", 0.0) for r in different), default=0.0),
        "different": sorted(different, key=lambda r: -r.get("max_abs", 0.0)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--top", type=int, default=8, help="how many differing tensors to list")
    args = parser.parse_args(argv)
    result = compare(args.first, args.second)
    summary = {k: v for k, v in result.items() if k != "different"}
    summary["different"] = result["different"][:args.top]
    print(json.dumps(summary, indent=1))
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
