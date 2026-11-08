"""Extract per-run memory evidence from the preserved real-checkpoint traces and compare it with the admission formula.

Reads results/flashnext/<run>/guard-trace.jsonl (or the chat guard's veystrix-chat-*.jsonl), reduces each run to the numbers
admission cares about, and writes them to ``fixtures/admission_traces.json`` so offline tests can replay them without the
results directory. Nothing here loads a model or touches memory settings.

    python -m models.flashnext.tests.bench.admission_audit            # print the table
    python -m models.flashnext.tests.bench.admission_audit --write    # refresh the fixture
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results" / "flashnext"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "admission_traces.json"
GIB = 1024.0

# name -> (trace file, prompt tokens prefilled cold, outcome, what was in force). Outcomes come from the run summaries and
# the docs/flashnext reports, not from the traces.
RUNS = {
    "chat-success": ("20261108-chat-pool0-success/veystrix-chat-87188.jsonl", 60, "success",
                     "five CHAT answers, pool 0, no stop"),
    "chat-first-stop": ("20261108-first-chat-guard-stop/veystrix-chat-85653.jsonl", 0, "guard-stop",
                        "stopped at 2.94 GiB available by the 3 GiB floor of that day (floor is 2 GiB now)"),
    "agentic-first-stop": ("20261108-first-agentic-smoke-guard-stop/veystrix-chat-7444.jsonl", 898, "guard-stop",
                           "compressor guard stop in the first cold prefill, pressure level 2"),
    "agentic-gpu-hang": ("20261108-agentic-smoke-retry-gpu-hang/veystrix-chat-13285.jsonl", 902, "gpu-hang",
                         "GPU/GIL deadlock (repaired); memory was not the cause"),
    "agentic-1k": ("20261108-agentic-smoke-retry-4-max-tokens-1/guard-trace.jsonl", 902, "success",
                   "compact tier, one decoded token"),
    "agentic-4k": ("20261108-224316-gate-a-attempt2/guard-trace.jsonl", 3935, "success",
                   "standard tier, 3,935-token cold prefill, then two warm turns"),
    "agentic-8k": ("20261108-232323-gate-8k-attempt1/guard-trace.jsonl", 8200, "incomplete",
                   "owner-waived start (8.68 GiB); prefill did not finish in 900 s; pressure 2 transient, no guard stop"),
}


def rows_of(path: Path) -> list:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "available_mb" in row and "mlx_peak" in row and "phase" in row:
            rows.append(row)
    return rows


def reduce_run(name: str) -> dict:
    trace, tokens, outcome, note = RUNS[name]
    rows = rows_of(RESULTS / trace)
    first = rows[0]
    start = first["available_mb"] / GIB
    by_phase = {}
    for row in rows:
        by_phase.setdefault(row["phase"], []).append(row)

    def drop(phase_rows):
        return round(start - min(r["available_mb"] for r in phase_rows) / GIB, 3)

    lowest = min(rows, key=lambda r: r["available_mb"])
    return {
        "name": name, "outcome": outcome, "note": note, "prefill_tokens": tokens, "samples": len(rows),
        "start": {"available_gib": round(start, 3), "anonymous_gib": round(first["anonymous_mb"] / GIB, 3),
                  "compressor_gib": round(first["compressor_mb"] / GIB, 3),
                  "file_backed_gib": round(first["file_backed_mb"] / GIB, 3),
                  "swap_gib": round(first["swap_used_mb"] / GIB, 3), "pressure": first["pressure_level"]},
        "drop_gib": {phase: drop(rs) for phase, rs in by_phase.items() if phase in ("init", "prefill", "decode")},
        "drop_total_gib": round(start - lowest["available_mb"] / GIB, 3),
        "min_available_gib": round(lowest["available_mb"] / GIB, 3),
        "max_compressor_gib": round(max(r["compressor_mb"] for r in rows) / GIB, 3),
        "max_swap_gib": round(max(r["swap_used_mb"] for r in rows) / GIB, 3),
        "max_pressure": max(r["pressure_level"] for r in rows),
        "mlx_peak_gib": round(max(r["mlx_peak"] for r in rows) / 2 ** 30, 3),
        "mlx_active_max_gib": round(max(r["mlx_active"] for r in rows) / 2 ** 30, 3),
        "rss_max_gib": round(max(r["rss"] for r in rows) / 2 ** 30, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    runs = [reduce_run(name) for name in RUNS]
    print(f"{'run':20s} {'tok':>5s} {'start':>6s} {'anon':>5s} {'comp':>5s} | {'load':>5s} {'pref':>5s} {'dec':>5s} {'total':>6s} | "
          f"{'minav':>5s} {'pk':>4s} {'mlx':>5s} {'press':>5s}  outcome")
    for r in runs:
        d = r["drop_gib"]
        print(f"{r['name']:20s} {r['prefill_tokens']:5d} {r['start']['available_gib']:6.2f} {r['start']['anonymous_gib']:5.2f} "
              f"{r['start']['compressor_gib']:5.2f} | {d.get('init', 0):5.2f} {d.get('prefill', 0):5.2f} {d.get('decode', 0):5.2f} "
              f"{r['drop_total_gib']:6.2f} | {r['min_available_gib']:5.2f} {r['max_compressor_gib']:4.1f} {r['mlx_peak_gib']:5.2f} "
              f"{r['max_pressure']:5d}  {r['outcome']}")
    if args.write:
        FIXTURE.write_text(json.dumps({"unit": "GiB", "runs": runs}, indent=1) + "\n")
        print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
