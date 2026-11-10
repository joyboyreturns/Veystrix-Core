"""Supervised single-arm cold prefill on the chunk-major reference server at full routing. Diagnostic, not a paired result.

One fresh supervised process (``bench_layer_gate.run_arm``) under the strict gate (swap stop 2304 MiB, pressure level 1,
3 GiB available, 1 s sampling, independent watchdog). ``--power-rules battery-ok`` records battery operation; the run refuses
below ``--min-battery``. It never relaxes a guard and exits on the first anomaly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from macqwen.measurement import MeasurementRun
from macqwen.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--source-tokens", type=int, default=6000, help="6000 gives about 8K prompt tokens")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--pool-gb", default="2")
    parser.add_argument("--power-rules", choices=("ac", "battery-ok"), default="ac")
    parser.add_argument("--min-battery", type=int, default=40)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    initial, postponed, purge = G.prepare_headroom(candidate=False, memory_rules="strict", power_rules=args.power_rules)
    if not postponed and args.power_rules == "battery-ok" and initial.get("power_source") == "battery" \
            and (initial.get("battery_percent") or 0) < args.min_battery:
        postponed = f"battery below this run's {args.min_battery}% floor"
    if postponed:
        print(f"Prefill postponed (nothing started): {postponed}", flush=True)
        return 2
    payload = G.make_prompt(checkpoint, args.source_tokens)
    payload["max_tokens"] = args.tokens
    path = output_path("flashnext", "reference-prefill", "reference-prefill.jsonl")
    git = lambda *cmd: subprocess.run(["git", *cmd], cwd=G.ROOT, capture_output=True, text=True, check=True).stdout
    run = MeasurementRun(path, runtime="flashnext", experiment="reference-prefill", metadata={
        "checkpoint": str(checkpoint), "commit": git("rev-parse", "HEAD").strip(), "worktree": git("status", "--short"),
        "source_tokens": args.source_tokens, "decode_tokens": args.tokens, "pool_gb": args.pool_gb, "routing_threshold": 1.0,
        "memory_rules": "strict", "power_rules": args.power_rules, "initial_machine": initial, "initial_purge": purge,
        "purpose": "single-arm cold prefill speed on the reference server; diagnostic, no resolution band"})
    run.start()
    try:
        command, rows, machine = G.run_arm("chunk", "ref-prefill", checkpoint, payload, path.parent, uncached=True,
                                           memory_rules="strict", base_pool_gb=args.pool_gb, reads="prefill",
                                           power_rules=args.power_rules)
        run.arm(arm_id="ref-prefill", condition="reference", round_index=1, command=command, metrics=rows, **machine)
        done = rows["prefilled"]
        print({"prompt_tokens": done.get("prompt_tokens"), "prefill_seconds": done["prefill_seconds"],
               "physical_bytes": done.get("physical_bytes")}, flush=True)
        run.finish("completed")
        return 0
    except Exception as exc:
        run.failure(str(exc), error_type="safety_stop_or_incomplete")
        run.finish("stopped", reason=str(exc))
        print(f"Stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
