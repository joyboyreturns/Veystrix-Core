"""Bounded, supervised 8K-after-turns safety experiment; not a speed benchmark.

This harness has not been exercised with the checkpoint. It requires strict
headroom, a separately spawned watchdog, the candidate's own request
admission, and an additional request-boundary headroom estimate. Power defaults
to AC; ``--power-rules battery-ok`` (owner permission 2026-11-10) allows battery
power above a charge floor (default 40%, never below the 25% in-flight floor)
and records the rule, floor and every sample's source, charge and thermal state.
It exits on the first pressure anomaly or refusal. Polling and these estimates cannot
guarantee the absolute swap ceiling, especially with an unmeasured full-routing
working set; do not interpret a pass as proof for other workloads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from macqwen.measurement import MeasurementRun
from macqwen.results import output_path
from models.flashnext import request_admission
from models.flashnext.tests.bench import bench_layer_gate as G


IN_FLIGHT_BATTERY_FLOOR = 25
DEFAULT_MIN_BATTERY = 40


def stop_reason(sample, baseline=None, power_rules="ac", min_battery=DEFAULT_MIN_BATTERY):
    """First reason to stop: the shared strict gate plus this harness's charge floor on battery."""
    reason = G.safety_reason(sample, baseline, memory_rules="swap-only", power_rules=power_rules)
    if reason or power_rules != "battery-ok":
        return reason
    if sample.get("power_source") == "battery" and (sample.get("battery_percent") or 0) < min_battery:
        return f"battery charge is below this run's {min_battery}% floor"
    return None


def bounded_prompt(checkpoint):
    """Construct the coding request and a conservative token budget, offline."""
    from transformers import AutoTokenizer

    payload = G.make_prompt(checkpoint, 6000)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=True)
    # Avoid literal control tokens inside the source excerpt: the server's
    # untrusted-content encoder otherwise changes its tokenization.
    for message in payload["messages"]:
        for token in tokenizer.all_special_tokens:
            if token:
                message["content"] = message["content"].replace(token, "[control token]")
    payload.update(max_tokens=8, stream=False, enable_thinking=False)
    rendered = tokenizer.apply_chat_template(payload["messages"], tools=None, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False, reasoning_effort="medium")
    count = len(tokenizer.encode(rendered, add_special_tokens=False))
    if not 7500 <= count <= 9500:
        raise ValueError(f"prompt renders to {count} tokens, outside the bounded 8K workload")
    # Extra budget for differences in server rendering; actual usage is checked
    # after the call. This is not a formal bound on every tokenizer behavior.
    return payload, count, count + 1024


def boundary_headroom(reading, tokens_upper, max_output=8):
    """Owner rule 2026-11-10: swap at the 3 GiB ceiling is the only memory condition; sizes are recorded, not gated."""
    ceiling = G.SWAP_CEILING_MB * 2**20
    admitted = reading.swap_used < ceiling
    return {"admitted": admitted, "reason": None if admitted else "swap is at the 3 GiB ceiling",
            "available_bytes": reading.available, "token_budget": tokens_upper, "swap_bytes": reading.swap_used,
            "pressure": reading.pressure}


def valid_chat(response):
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("chat response has no single valid choice")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
        raise RuntimeError("chat response contains no text")
    if choices[0].get("finish_reason") not in ("stop", "length"):
        raise RuntimeError("chat response did not complete normally")
    return message["content"]


def make_driver(payload, count, upper, report, power_rules="ac", min_battery=DEFAULT_MIN_BATTERY):
    def driver(url):
        health = G.http_json(url + "/health", timeout=10)
        policy = health.get("memory_policy") or {}
        if (policy.get("swap_ceiling_mib"), policy.get("swap_ceiling_margin_mib")) != (3072.0, 0.0):
            raise RuntimeError("candidate absolute guard is absent or differs from 3072/0 MiB")
        report["guard"] = {"swap_ceiling_mib": 3072, "margin_mib": 0}
        initial = G.machine_sample()
        report["initial_server_sample"] = initial
        reason = stop_reason(initial, None, power_rules, min_battery)
        if reason:
            raise RuntimeError("post-start safety: " + reason)
        messages = [{"role": "system", "content": "Answer briefly."},
                    {"role": "user", "content": "Reply with the single word ready."}]
        for index in range(2):
            if index:
                messages.append({"role": "user", "content": "Reply with the single word next."})
            response = G.http_json(url + "/v1/chat/completions", {"model": "flashnext", "messages": messages,
                                   "max_tokens": 8, "stream": False, "enable_thinking": False}, timeout=180)
            text = valid_chat(response)
            messages.append({"role": "assistant", "content": text})
            report.setdefault("initial_turns", []).append({"finish_reason": response["choices"][0]["finish_reason"],
                                                           "usage": response.get("usage")})
            current = G.machine_sample()
            report["after_turn_" + str(index + 1)] = current
            reason = stop_reason(current, initial, power_rules, min_battery)
            if reason:
                raise RuntimeError("after-turn safety: " + reason)
        try:
            gate = boundary_headroom(request_admission.strict_read(), upper, payload["max_tokens"])
        except Exception as exc:
            raise RuntimeError(f"independent admission probe unavailable: {exc}") from exc
        report["pre_8k_admission"] = gate
        if not gate["admitted"]:
            raise RuntimeError("8K request postponed: " + str(gate["reason"]))
        current = G.machine_sample()
        report["before_8k_request"] = current
        reason = stop_reason(current, initial, power_rules, min_battery)
        if reason:
            raise RuntimeError("pre-8K safety: " + reason)
        response = G.http_json(url + "/v1/chat/completions", payload, timeout=600)
        valid_chat(response)
        usage = response.get("usage") or {}
        actual = usage.get("prompt_tokens")
        if not isinstance(actual, int) or not 7500 <= actual <= min(9500, upper):
            raise RuntimeError(f"actual prompt token count {actual!r} outside the admitted 8K range")
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict) or details.get("cached_tokens") != 0:
            raise RuntimeError("8K request was not demonstrably cold (cached-token count missing or nonzero)")
        report["request_8k"] = {"estimated_prompt_tokens": count, "admitted_upper_tokens": upper,
                                "usage": usage, "finish_reason": response["choices"][0]["finish_reason"]}
        return report
    return driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--pool-gb", choices=("0", "1", "2"), default="2")
    parser.add_argument("--power-rules", choices=("ac", "battery-ok"), default="ac",
                        help="battery-ok allows discharging above --min-battery; the choice is recorded")
    parser.add_argument("--min-battery", type=int, default=DEFAULT_MIN_BATTERY,
                        help="charge floor for battery-ok (at least 25)")
    args = parser.parse_args()
    if not IN_FLIGHT_BATTERY_FLOOR <= args.min_battery <= 100:
        parser.error("--min-battery must be between 25 and 100")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    path = output_path("flashnext", "phase2-after-turns", "phase2-after-turns.jsonl")
    run = MeasurementRun(path, runtime="flashnext", experiment="phase2-after-turns", metadata={
        "checkpoint": str(checkpoint), "pool_gb": args.pool_gb, "routing_threshold": 1.0,
        "memory_rules": "strict", "power_rules": args.power_rules,
        "min_battery_percent": args.min_battery if args.power_rules == "battery-ok" else None, "purpose": "bounded safety diagnostic; not a performance gate"})
    run.start()
    report = {}
    try:
        initial, reason, purge = G.prepare_headroom(candidate=True, memory_rules="swap-only", power_rules=args.power_rules)
        report.update(initial_machine=initial, initial_purge=purge, power_rules=args.power_rules)
        reason = reason or stop_reason(initial, None, args.power_rules, args.min_battery)
        if reason:
            raise RuntimeError("preflight postponed: " + reason)
        payload, count, upper = bounded_prompt(checkpoint)
        command, evidence, machine = G.run_arm("layer", "after-turns", checkpoint, None, path.parent,
                                               candidate="layer-major", uncached=True, memory_rules="swap-only",
                                               prefetch=True, base_pool_gb=args.pool_gb, reads="prefill",
                                               power_rules=args.power_rules,
                                               driver=make_driver(payload, count, upper, report, args.power_rules, args.min_battery),
                                               auto_purge=True,
                                               extra_env={"FLASHNEXT_SWAP_CEILING_MIB": "3072",
                                                          "FLASHNEXT_SWAP_CEILING_MARGIN_MIB": "0"})
        run.arm(arm_id="after-turns", condition="layer", round_index=1, command=command, metrics=evidence, **machine)
        run.finish("completed", report=report)
        print("Bounded 8K-after-turns request completed; inspect recorded safety evidence.")
        return 0
    except Exception as exc:
        run.failure(str(exc), error_type="safety_stop_or_incomplete")
        run.finish("stopped", reason=str(exc), partial_report=report)
        print(f"Stopped: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
