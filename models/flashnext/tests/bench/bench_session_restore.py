"""Cross-session prefix restoration timing on the real checkpoint (reference chunk-major server, full routing). Diagnostic.

Arm A: cold ~2.7K-token coding prompt in a fresh process, writing prefix snapshots to a shared directory.
Arm B: a NEW process on the same directory with the same long prefix and a different final user line (what a new client
session looks like). Reports prefill seconds, prompt tokens and the cached-token count for both. Swap-only (3 GiB) rule.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from veystrix.measurement import MeasurementRun
from veystrix.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--source-tokens", type=int, default=600)
    parser.add_argument("--pool-gb", default="2")
    parser.add_argument("--power-rules", choices=("ac", "battery-ok"), default="ac")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    initial, postponed, purge = G.prepare_headroom(candidate=False, memory_rules="swap-only", power_rules=args.power_rules)
    if postponed:
        print(f"Postponed (nothing started): {postponed}", flush=True)
        return 2
    first = G.make_prompt(checkpoint, args.source_tokens)
    first["max_tokens"] = 8
    second = copy.deepcopy(first)
    second["messages"][-1]["content"] += "\n\nAlso name the single riskiest function in the code above."
    path = output_path("flashnext", "session-restore", "session-restore.jsonl")
    git = lambda *cmd: subprocess.run(["git", *cmd], cwd=G.ROOT, capture_output=True, text=True, check=True).stdout
    run = MeasurementRun(path, runtime="flashnext", experiment="session-restore", metadata={
        "checkpoint": str(checkpoint), "commit": git("rev-parse", "HEAD").strip(), "source_tokens": args.source_tokens,
        "routing_threshold": 1.0, "memory_rules": "swap-only", "power_rules": args.power_rules, "initial_machine": initial,
        "purpose": "cold vs new-process restored prefill; diagnostic, one pair, no resolution band"})
    run.start()
    out = {}
    try:
        with tempfile.TemporaryDirectory(dir=str(G.ROOT / "results")) as shared:
            prefix = Path(shared) / "prefix"
            for name, payload in (("cold", first), ("restored", second)):
                command, rows, machine = G.run_arm("chunk", name, checkpoint, payload, path.parent, uncached=True,
                                                   memory_rules="swap-only", base_pool_gb=args.pool_gb, reads="prefill",
                                                   power_rules=args.power_rules, prefix_dir=prefix)
                run.arm(arm_id=name, condition=name, round_index=1, command=command, metrics=rows, **machine)
                done = rows["prefilled"]
                out[name] = {k: done.get(k) for k in ("prefill_seconds", "prompt_tokens", "cached_tokens", "physical_bytes")}
                print(json.dumps({name: out[name], "prefilled": {k: v for k, v in done.items() if k != "logits"}}, default=str)[:900], flush=True)
        run.finish("completed", summary=out)
        return 0
    except Exception as exc:
        run.failure(str(exc), error_type="incomplete_or_failed_restore")
        run.finish("stopped", reason=str(exc), partial=out)
        print(f"Stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
