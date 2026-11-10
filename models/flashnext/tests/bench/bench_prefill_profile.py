"""One supervised prefill arm with the per-layer trace. Diagnostic only; never promotes a speed result.

Runs ``bench_layer_gate.run_arm`` (supervised ``serve.sh`` server, private port and prefix cache, independent watchdog,
swap/power/thermal stops) for a single arm and prints where the prefill time goes: per window, expert read seconds against
everything else, and attention against linear-attention (GatedDeltaNet) layers. Layer-major arms carry ``layer_seconds``
in their execution segments; chunk-major arms carry per-chunk seconds only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

from veystrix.measurement import MeasurementRun
from veystrix.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G


def layer_split(segment: dict) -> dict | None:
    seconds = segment.get("layer_seconds")
    if not seconds:
        return None
    attention = set(segment.get("attention_layers") or [])
    attn = [s for i, s in enumerate(seconds) if i in attention]
    linear = [s for i, s in enumerate(seconds) if i not in attention]
    return {
        "window": [segment["start"], segment["end"]], "seconds": segment["seconds"], "read_seconds": segment.get("read_seconds"),
        "attention_layers": len(attn), "attention_total_s": round(sum(attn), 2),
        "attention_median_s": round(st.median(attn), 3) if attn else None,
        "linear_layers": len(linear), "linear_total_s": round(sum(linear), 2),
        "linear_median_s": round(st.median(linear), 3) if linear else None,
        "layer_seconds": seconds,
        "layer_memory_mb": segment.get("layer_memory_mb"),
        "layer_reads": segment.get("layer_reads"), "chunk_seconds": segment.get("chunk_seconds"),
        "chunk_memory_mb": segment.get("chunk_memory_mb"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--source-tokens", type=int, default=6000)
    parser.add_argument("--pool-gb", default="2")
    parser.add_argument("--candidate", choices=tuple(G.ORDERS), default="layer-major")
    parser.add_argument("--condition", choices=("layer", "chunk"), default="layer")
    parser.add_argument("--power-rules", choices=("battery-ok", "ac"), default="battery-ok")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    initial, postponed, purge = G.prepare_headroom(candidate=args.condition == "layer", memory_rules="swap-only",
                                                   power_rules=args.power_rules)
    if postponed:
        print(f"Profile postponed (nothing started): {postponed}", flush=True)
        return 2
    payload = G.make_prompt(checkpoint, args.source_tokens)
    payload["max_tokens"] = 32
    path = output_path("flashnext", "prefill-profile", "prefill-profile.jsonl")
    run = MeasurementRun(path, runtime="flashnext", experiment="prefill-profile", metadata={
        "checkpoint": str(checkpoint), "source_tokens": args.source_tokens, "condition": args.condition,
        "candidate": args.candidate, "pool_gb": args.pool_gb, "memory_rules": "swap-only", "initial_machine": initial,
        "purpose": "per-layer prefill time split; one arm, not a speed result"})
    run.start()
    try:
        command, rows, machine = G.run_arm(args.condition, "p1", checkpoint, payload, path.parent, candidate=args.candidate,
                                           uncached=True, memory_rules="swap-only", base_pool_gb=args.pool_gb, reads="prefill",
                                           power_rules=args.power_rules)
        run.arm(arm_id="p1", condition=args.condition, round_index=1, command=command, metrics=rows, **machine)
        prefilled = rows["prefilled"]
        report = {"prefill_seconds": prefilled["prefill_seconds"], "tokens": prefilled["state"]["token_count"],
                  "physical_gb": prefilled["physical_bytes"] / 1e9, "mlx_peak_gb": prefilled["peak_bytes"] / 1e9,
                  "windows": [layer_split(s) or s for s in prefilled["prefix"].get("execution_segments", [])],
                  "chunks": prefilled["prefix"].get("chunk_trace", [])}
        run.finish("completed", report=report)
        printable = json.loads(json.dumps(report))
        for window in printable["windows"]:
            window.pop("layer_seconds", None)
            window.pop("layer_memory_mb", None)
            window.pop("layer_reads", None)
            window.pop("chunk_seconds", None)
            window.pop("chunk_memory_mb", None)
        print(json.dumps(printable, indent=1), flush=True)
        return 0
    except Exception as error:
        run.failure(str(error), error_type="incomplete_or_failed_profile")
        run.finish("stopped", reason=str(error))
        print(f"Profile stopped: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
