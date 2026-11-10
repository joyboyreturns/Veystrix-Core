"""Supervised sustained-decode comparison on the real checkpoint. Not an exactness gate (the 32-token A/B is) and not a prefill result.

Each arm is a fresh supervised server (``bench_layer_gate.run_arm``, with the swap guards, auto-purge ON, the absolute swap guard)
driven through three consecutive streamed turns in ONE process (a ~2.5K-token coding prompt, then two follow-ups of up to 96 tokens each).
Decode speed per turn is measured at the client from the first to the last streamed token, so the page cache that buffered pool
fills depend on has time to warm. Reference: the default server (uncached prefill reads only). Candidate: the same plus
``--enable pool-buffered-fill``. Reversed pairs; both arms greedy, so the generated text is compared as well.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from veystrix.measurement import MeasurementRun
from veystrix.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G

FOLLOW_UPS = ("Now explain in detail how the prefix cache decides whether a snapshot can be restored.",
              "Next, list the failure modes you would test for in that code and why.")


def turn(url, messages, max_tokens=96):
    payload = {"model": "flashnext", "messages": messages, "max_tokens": max_tokens, "stream": True}
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": "Bearer local"})
    began, first, last, tokens, text = time.monotonic(), None, None, 0, []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                delta = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                now = time.monotonic()
                first = first or now
                last = now
                tokens += 1
                text.append(delta)
    decode = (tokens - 1) / (last - first) if tokens > 1 and last > first else None
    return {"tokens": tokens, "ttft_s": round(first - began, 2) if first else None,
            "decode_tok_s": round(decode, 3) if decode else None, "text": "".join(text)}


def make_driver(payload, report):
    def driver(url):
        ledger = (G.http_json(url + "/health", timeout=10) or {}).get("memory_policy") or {}
        if ledger.get("swap_ceiling_mib") != 3072.0 or ledger.get("swap_ceiling_margin_mib") != 768.0:
            raise RuntimeError("absolute swap guard is not active in the server")
        messages = list(payload["messages"])
        turns = []
        for index in range(3):
            result = turn(url, messages)
            turns.append(result)
            messages += [{"role": "assistant", "content": result["text"]}]
            if index < len(FOLLOW_UPS):
                messages += [{"role": "user", "content": FOLLOW_UPS[index]}]
        report["turns"] = turns
        return report
    return driver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--settle", type=int, default=30)
    parser.add_argument("--pool-gb", default="2")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    initial, postponed, purge = G.prepare_headroom(candidate=False, memory_rules="swap-only", power_rules="ac")
    if postponed:
        print(f"Sustained-decode run postponed (nothing started): {postponed}", flush=True)
        return 2
    payload = G.make_prompt(checkpoint, 600)
    path = output_path("flashnext", "sustained-decode", "sustained-decode.jsonl")
    run = MeasurementRun(path, runtime="flashnext", experiment="sustained-decode", metadata={
        "checkpoint": str(checkpoint), "pool_gb": args.pool_gb, "rounds": args.rounds, "memory_rules": "swap-only",
        "initial_machine": initial, "purpose": "three-turn sustained decode, default vs pool-buffered-fill; not an exactness gate"})
    run.start()
    results = {"base": [], "buffered": []}
    try:
        for number in range(args.rounds):
            for index, name in enumerate(("base", "buffered") if number % 2 == 0 else ("buffered", "base")):
                arm = f"r{number + 1}-{name}"
                print(f"Starting {arm}", flush=True)
                report = {}
                command, evidence, machine = G.run_arm(
                    "chunk", arm, checkpoint, None, path.parent, uncached=True, memory_rules="swap-only", base_pool_gb=args.pool_gb,
                    reads="prefill", power_rules="ac", driver=make_driver(payload, report), auto_purge=True,
                    extra_env={"FLASHNEXT_SWAP_CEILING_MIB": "3072", "FLASHNEXT_SWAP_CEILING_MARGIN_MIB": "768"},
                    extra_enables=("pool-buffered-fill",) if name == "buffered" else ())
                run.arm(arm_id=arm, condition=name, round_index=number + 1, command=command, metrics=evidence, **machine)
                results[name].append(evidence["turns"])
                if index == 0:
                    time.sleep(args.settle)
        summary = {}
        for name, rounds in results.items():
            summary[name] = {"decode_tok_s_per_turn_per_round": [[t["decode_tok_s"] for t in r] for r in rounds],
                             "tokens": [[t["tokens"] for t in r] for r in rounds],
                             "median_by_turn": [round(statistics.median(r[i]["decode_tok_s"] for r in rounds), 3) for i in range(3)]}
        same = all(a[i]["text"] == b[i]["text"] for a in results["base"] for b in results["buffered"] for i in range(3))
        summary["generated_text_identical_across_arms"] = same
        print(json.dumps(summary, indent=1), flush=True)
        run.finish("completed", summary=summary)
        return 0
    except Exception as error:
        run.failure(str(error), error_type="incomplete_or_failed_sustained_decode")
        run.finish("stopped", reason=str(error))
        print(f"Sustained-decode run stopped: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
