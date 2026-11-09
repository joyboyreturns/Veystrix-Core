"""Supervised restore check for layer-major inner snapshots on the real checkpoint. Correctness diagnostic only.

Three fresh supervised AGENTIC servers (``bench_layer_gate.run_arm``: independent watchdog, swap/power/thermal stops):

* A  layer-major (+prefetch), private persistent prefix directory P, prompt A (system prompt + repository excerpt 1). Cold; its
     prefill stores the system-prompt snapshot from the layers' frozen caches (an inner snapshot inside one window).
* B  a NEW process on the same P, layer-major, prompt B (same system prompt, different repository excerpt). Must restore the system
     snapshot from the SSD tier and prefill only the rest.
* C  chunk-major reference, empty prefix directory, prompt B, cold.

B must equal C on prompt identity, first token, 32 generated token IDs, final-row logits and every attention / recurrent cache
array. Only exactness and the restore itself are judged here; the speeds are not a result.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from macqwen.measurement import MeasurementRun
from macqwen.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G


def validate_restore(reference, restored, system_boundary_min=1024):
    """None when ``restored`` (a prefix hit, then layer-major) equals ``reference`` (cold chunk-major); else the reason."""
    for row in (reference, restored):
        if set(row) != {"logits", "prefilled", "finished"}:
            return "missing logits, prefill or completion probe"
        if len(row["finished"].get("token_ids") or []) != 32:
            return "generated tokens missing"
        if not G.array_evidence_valid(row["logits"].get("value")):
            return "final-row logits missing"
    a, b = reference["prefilled"], restored["prefilled"]
    if (a.get("prefix") or {}).get("hit_tokens") != 0:
        return "the reference was not cold"
    prefix = b.get("prefix") or {}
    hit = prefix.get("hit_tokens") or 0
    if hit < system_boundary_min:
        return f"the restored arm did not restore a system-prompt snapshot (hit_tokens={hit})"
    if prefix.get("prefilled_tokens") != a["state"]["token_count"] - hit:
        return "the restored arm prefilled the wrong number of tokens"
    for key in ("prompt_sha256", "first_token"):
        if a.get(key) != b.get(key):
            return f"{key} differs"
    if reference["logits"] != restored["logits"]:
        return "final-row logits differ"
    if reference["finished"]["token_ids"] != restored["finished"]["token_ids"]:
        return "generated token IDs differ"
    if a["state"].get("layers") != b["state"].get("layers"):
        return "cache arrays differ"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--source-tokens", type=int, default=600)
    parser.add_argument("--pool-gb", default="2")
    parser.add_argument("--settle", type=int, default=30)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    initial, postponed, purge = G.prepare_headroom(candidate=True, memory_rules="swap-only", power_rules="ac")
    if postponed:
        print(f"Restore check postponed (nothing started): {postponed}", flush=True)
        return 2
    payload_a = G.make_prompt(checkpoint, args.source_tokens)
    payload_b = G.make_prompt(checkpoint, args.source_tokens, skip_tokens=args.source_tokens)
    assert payload_a["messages"][0] == payload_b["messages"][0] and payload_a["messages"][1] != payload_b["messages"][1]
    path = output_path("flashnext", "restore-check", "restore-check.jsonl")
    run = MeasurementRun(path, runtime="flashnext", experiment="restore-check", metadata={
        "checkpoint": str(checkpoint), "source_tokens": args.source_tokens, "pool_gb": args.pool_gb,
        "memory_rules": "swap-only", "initial_machine": initial,
        "purpose": "restore-from-inner-snapshot exactness; speeds are not a result"})
    run.start()
    with tempfile.TemporaryDirectory(prefix="veystrix-restore-check-") as shared:
        prefix_dir = Path(shared) / "prefix"
        try:
            arms = {}
            plan = (("a", "layer", payload_a, prefix_dir), ("b", "layer", payload_b, prefix_dir), ("c", "chunk", payload_b, None))
            for label, name, payload, directory in plan:
                print(f"Starting {label}-{name}", flush=True)
                command, rows, machine = G.run_arm(name, label, checkpoint, payload, path.parent, candidate="layer-major",
                                                   uncached=True, memory_rules="swap-only", prefetch=True, base_pool_gb=args.pool_gb,
                                                   reads="prefill", power_rules="ac", prefix_dir=directory)
                run.arm(arm_id=label, condition=name, round_index=1, command=command, metrics=rows, **machine)
                arms[label] = rows
                time.sleep(args.settle)
            reason = validate_restore(arms["c"], arms["b"])
            run.validation("restore", "fail" if reason else "pass", reason=reason)
            if reason:
                raise RuntimeError("restore check: " + reason)
            b = arms["b"]["prefilled"]
            report = {"restored_tokens": b["prefix"]["hit_tokens"], "tier": b["prefix"].get("tier"),
                      "prefilled_tokens": b["prefix"]["prefilled_tokens"], "outcome": b["prefix"].get("outcome"),
                      "seconds": {k: round(v["prefilled"]["prefill_seconds"], 2) for k, v in arms.items()}}
            print(json.dumps(report, indent=1), flush=True)
            run.finish("completed", exactness="passed", report=report)
            return 0
        except Exception as error:
            run.failure(str(error), error_type="incomplete_or_failed_restore_check")
            run.finish("stopped", reason=str(error))
            print(f"Restore check stopped: {error}", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
