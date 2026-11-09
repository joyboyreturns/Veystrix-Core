"""Supervised decode profile on the real checkpoint. Diagnostic only; never promotes a speed result.

Runs ``bench_layer_gate.run_arm`` (the supervised ``serve.sh`` launcher, private port and prefix cache, independent
watchdog, swap/power/thermal stops) with the chunk-major reference configuration and an optional decode profile:

* ``plain``   no profiling: the decode rate to compare the others against (profiling overhead),
* ``io``      FLASHNEXT_PROFILE_IO=1 plus the decode profile (timers, per-token and per-layer wall time, routes),
* ``host``    ``io`` plus FLASHNEXT_HOST_WINDOW=1 (host time while both devices are idle),
* ``plain2``  a second unprofiled arm after the others (drift control).

Profiled arms add synchronisation and timers, so their decode rate is not a speed result; their value is the split and the
route trace (``<arm>-profile/decode-*.json``). The timers can overlap (reads run on worker threads), so they are reported
beside wall time and never summed into a total.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

from macqwen.measurement import MeasurementRun
from macqwen.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G

ARMS = {
    "plain": {},
    "io": {"FLASHNEXT_PROFILE_IO": "1"},
    "host": {"FLASHNEXT_PROFILE_IO": "1", "FLASHNEXT_HOST_WINDOW": "1"},
    "plain2": {},
}
TIMER_KEYS = ("io_wait", "router_sync", "to_mx", "moe_issue", "score_sync", "topk_python", "shared_expert", "ngram_wait")


def split(profile: dict) -> dict:
    """Per-token milliseconds of every timer and host window, plus the per-token wall time distribution."""
    tokens = profile["tokens"]
    count = max(len(tokens), 1)
    # The first token's interval starts at the first streamed layer, so it includes prefill->decode hand-off; keep it
    # in the record but exclude it from the median.
    walls = [t["ms"] for t in tokens]
    steady = walls[1:] if len(walls) > 2 else walls
    timers = profile.get("timers", {})
    out = {
        "tokens_profiled": len(tokens),
        "token_ms_median": st.median(steady) if steady else None,
        "token_ms_mean": st.fmean(steady) if steady else None,
        "token_ms_min": min(steady) if steady else None,
        "token_ms_max": max(steady) if steady else None,
        "timers_ms_per_token": {k: timers.get(k, 0.0) * 1000.0 / count for k in TIMER_KEYS},
        "io_calls_per_token": timers.get("io_calls", 0) / count,
        "pread_bytes_per_token": timers.get("pread_bytes", 0) / count,
        "host_windows_ms_per_token": {
            name: {"exclusive": slot[0] * 1000.0 / count, "shared": slot[1] * 1000.0 / count, "calls": slot[2] / count}
            for name, slot in profile.get("host_windows", {}).items()
        },
        "expert_io": profile.get("expert_io", {}),
    }
    # Layer gaps: time from one streamed layer's route to the next layer's route (includes that layer's expert read,
    # the previous layer's compute and the next router eval). Averaged by layer across the profiled tokens.
    by_layer: dict[int, list[float]] = {}
    for token in tokens[1:]:
        for layer, gap in enumerate(token["layer_gap_ms"], start=1):
            by_layer.setdefault(layer, []).append(gap)
    out["layer_gap_ms_mean"] = {str(k): st.fmean(v) for k, v in sorted(by_layer.items())}
    return out


def decode_rate(rows) -> float | None:
    finished = rows["finished"]
    seconds = finished.get("tail_seconds") or 0
    return finished["tail_tokens"] / seconds if seconds else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--source-tokens", type=int, default=600, help="600 gives a 2.7K prompt, 6000 an 8K prompt")
    parser.add_argument("--tokens", type=int, default=96, help="decode tokens per arm")
    parser.add_argument("--pool-gb", default="2")
    parser.add_argument("--arms", default="plain,io,host,plain2")
    parser.add_argument("--settle", type=int, default=30)
    parser.add_argument("--power-rules", choices=("battery-ok", "ac"), default="battery-ok")
    args = parser.parse_args()
    names = args.arms.split(",")
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    if args.settle < 30:
        parser.error("at least 30 seconds between arms are required")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if shutil.disk_usage(G.ROOT).free < 50 * 10**9:
        raise SystemExit("at least 50 GB of free storage is required")
    initial, postponed, purge = G.prepare_headroom(candidate=False, memory_rules="swap-only", power_rules=args.power_rules)
    if postponed:
        print(f"Profile postponed (nothing started): {postponed}", flush=True)
        return 2
    payload = G.make_prompt(checkpoint, args.source_tokens)
    payload["max_tokens"] = args.tokens
    path = output_path("flashnext", "decode-profile", "decode-profile.jsonl")
    git = lambda *cmd: subprocess.run(["git", *cmd], cwd=G.ROOT, capture_output=True, text=True, check=True).stdout
    run = MeasurementRun(path, runtime="flashnext", experiment="decode-profile", metadata={
        "checkpoint": str(checkpoint), "commit": git("rev-parse", "HEAD").strip(), "worktree": git("status", "--short"),
        "source_tokens": args.source_tokens, "decode_tokens": args.tokens, "pool_gb": args.pool_gb, "arms": names,
        "memory_rules": "swap-only", "power_rules": args.power_rules, "initial_machine": initial, "initial_purge": purge,
        "reads": "prefill-only F_NOCACHE, chunk-major reference",
        "purpose": "decode time distribution and current-checkpoint route trace; profiled arms are not speed results"})
    run.start()
    summary = {}
    try:
        for number, name in enumerate(names):
            arm_id = f"p{number + 1}-{name}"
            profile_dir = path.parent / f"{arm_id}-profile"
            extra = dict(ARMS[name])
            if name != "plain" and name != "plain2":
                extra["FLASHNEXT_DECODE_PROFILE_DIR"] = str(profile_dir)
            print(f"Starting {arm_id}", flush=True)
            try:
                command, rows, machine = G.run_arm("chunk", arm_id, checkpoint, payload, path.parent, uncached=True,
                                                   memory_rules="swap-only", base_pool_gb=args.pool_gb, reads="prefill",
                                                   power_rules=args.power_rules, extra_env=extra)
            except Exception as error:
                run.arm(arm_id=arm_id, condition=name, round_index=number + 1, command=[], status="failed",
                        metrics={"error": str(error)})
                raise
            run.arm(arm_id=arm_id, condition=name, round_index=number + 1, command=command, metrics=rows, **machine)
            row = {"decode_tok_s": decode_rate(rows), "prefill_seconds": rows["prefilled"]["prefill_seconds"],
                   "decode_physical_bytes": rows["finished"]["physical_bytes_total"] - rows["prefilled"]["physical_bytes"]}
            profiles = sorted(profile_dir.glob("decode-*.json")) if profile_dir.exists() else []
            if profiles:
                row["profile"] = split(json.loads(profiles[-1].read_text()))
                row["profile_file"] = str(profiles[-1].relative_to(path.parent))
            elif name not in ("plain", "plain2"):
                raise RuntimeError(f"{arm_id}: profiled arm wrote no decode profile")
            summary[arm_id] = row
            print(json.dumps({arm_id: {k: v for k, v in row.items() if k != "profile"}}), flush=True)
            if number + 1 < len(names):
                time.sleep(args.settle)
        run.finish("completed", summary=summary)
        print(json.dumps(summary, indent=1)[:6000], flush=True)
        return 0
    except Exception as exc:
        run.failure(str(exc), error_type="incomplete_or_failed_profile")
        run.finish("stopped", reason=str(exc), partial=summary)
        print(f"Profile stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
