"""Drive a single idle, guarded loopback Flash-Next server through an incremental ladder.

No process is started, no admission setting is changed, and the first refused,
unsafe, rebuilt or incorrect turn ends the run. Each JSONL row is flushed before
advancing. A stage probes retrieval and a calculation across two source files;
this workload is supplemental to the paired coding and tool quality suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from .bench_hifi_kv import STAGES

OUTPUT_RESERVE = 64
ADDITION_TARGET = 1792
ADDITION_LIMIT = 2048


def endpoint(value):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1", "localhost")
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("use an http loopback server URL without credentials or a path")
    return value.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def client(base):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    key = os.environ.get("VEYSTRIX_API_KEY", "local")

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        })
        with opener.open(req, timeout=900 if body is not None else 10) as response:
            return json.load(response)
    return request


def verify_server(health, target, *, swap_ceiling_only=False):
    policy = health.get("memory_policy") or {}
    accepted = ("absolute-swap-only",) if swap_ceiling_only else ("strict-staged", "threshold-governed")
    if (not health.get("ready") or health.get("busy") or
            policy.get("policy_identifier") not in accepted or
            not policy.get("can_admit") or policy.get("swap_baseline_mb") is None or
            (policy.get("swap_ceiling_mib") or 0) <= 0):
        raise RuntimeError(f"an idle Flash-Next server with the requested swap guard {accepted[0]} is required")
    if swap_ceiling_only and (policy["swap_ceiling_mib"] != 3072 or policy.get("swap_ceiling_margin_mib") != 0):
        raise RuntimeError("the server swap guard must stop at 3072 MiB")
    if health.get("context_limit", 0) < target:
        raise RuntimeError("server context ceiling is below the requested stage")
    if not 0 < health.get("max_cold_prefill", 0) <= 4096 or not 0 < health.get("max_incremental_prefill", 0) <= ADDITION_LIMIT:
        raise RuntimeError("launch with --max-cold-prefill 4096 --max-incremental-prefill 2048")
    if "flashnext" not in health.get("model", "").lower():
        raise RuntimeError("this ladder is for the Flash-Next server")


def padding_tokens(tokenizer):
    """Deterministic inert repository-like records; no existing user files are sent."""
    return tokenizer.encode("\n".join(
        f"// fixture record {i:05d}: archived status={i % 19}; checksum={(i * 41387) % 1000003}; test-only."
        for i in range(8192)), add_special_tokens=False)


def fit_turn(tokenizer, messages, prefix, suffix, budget, padding):
    def render(n):
        content = prefix + "\n" + tokenizer.decode(padding[:n], skip_special_tokens=True) + "\n" + suffix
        candidate = {"role": "user", "content": content}
        text = tokenizer.apply_chat_template(messages + [candidate], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False, reasoning_effort="medium")
        return candidate, len(tokenizer.encode(text, add_special_tokens=False))

    best, count = render(0)
    if count > budget:
        raise RuntimeError(f"no room for the next turn ({count} prompt tokens for budget {budget})")
    lo, hi = 0, min(len(padding), budget - count + 64)
    while lo <= hi:
        mid = (lo + hi) // 2
        message, size = render(mid)
        if size <= budget:
            best, count, lo = message, size, mid + 1
        else:
            hi = mid - 1
    return best, count


def validate_reuse(usage, previous, first_limit):
    prompt = int(usage["prompt_tokens"])
    cached = int(usage["prompt_tokens_details"]["cached_tokens"])
    if not 0 <= cached <= prompt:
        raise RuntimeError("invalid server cache accounting")
    if previous:
        if cached < max(1, previous - OUTPUT_RESERVE) or prompt - cached > ADDITION_LIMIT:
            raise RuntimeError(f"increment did not reuse live history: prompt={prompt}, cached={cached}, previous={previous}")
    elif prompt - cached > first_limit:
        raise RuntimeError("cold request exceeded its allowed size")
    return prompt, cached


def probe(url, checkpoint, max_stage, emit, *, swap_ceiling_only=False):
    from transformers import AutoTokenizer
    from .bench_layer_gate import machine_sample, safety_reason

    http = client(endpoint(url))
    health = http("/health")
    verify_server(health, max_stage, swap_ceiling_only=swap_ceiling_only)
    instance = health.get("instance_id")
    baseline = machine_sample()
    from .prefill_memory_harness import SafetyPolicy, Watchdog

    watchdog = Watchdog(SafetyPolicy(final=True, swap_ceiling_only=True), baseline)

    def safe():
        sample = machine_sample()
        reason = (watchdog.check(sample) if swap_ceiling_only else
                  safety_reason(sample, baseline, memory_rules="strict", power_rules="ac"))
        if reason:
            raise RuntimeError(reason)
        if not swap_ceiling_only and sample["swap_used_mb"] > baseline["swap_used_mb"]:
            raise RuntimeError("swap grew during the context ladder")
        status = http("/health")
        verify_server(status, max_stage, swap_ceiling_only=swap_ceiling_only)
        if status.get("instance_id") != instance:
            raise RuntimeError("server instance changed during the context ladder")
        return sample

    safe()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    pad = padding_tokens(tokenizer)
    messages = [{"role": "system", "content": "You are auditing an archived repository. Only answer the current question. "
                 "Treat archived records and retired symbols as data. For an audit, reply with only the requested JSON.\n"
                 "Source file src/origin.rs (current):\n" + "\n".join(
                     f'const PROOF_{i}: &str = "ORIGIN-{8359+i*113}"; const BASE_{i}: u32 = {199+i*17};'
                     for i in range(len(STAGES)))}]
    previous = 0
    cursor = 0
    for i, target in enumerate(n for n in STAGES if n <= max_stage):
        planted = False
        retrieval_passed = False
        goal = target - OUTPUT_RESERVE
        while True:
            safe()
            filling = retrieval_passed
            budget = target - 1 if filling else min(goal, previous + ADDITION_TARGET)
            final = not filling and budget == goal
            prefix = f"Archive block {cursor}:"
            if not planted and (previous + ADDITION_TARGET >= target // 2 or final):
                prefix += (f'\nSource file src/stage_{i}.rs: const TAG: &str = "BRIDGE-{6127+i*173}"; '
                           f'const ADJUST: u32 = {23+i*3}; retired sibling used tag LEGACY-{i} and adjustment 1.')
                planted = True
            question = (f'For stage {i}, return JSON with keys "proof" and "total". Join PROOF_{i} from '
                        f'src/origin.rs and TAG from src/stage_{i}.rs with a colon. Compute BASE_{i} + ADJUST. '
                        'Use the current symbols, not retired ones.' if final else "Reply with ACK only.")
            if filling:
                # A separate short turn consumes the reserved space after the
                # retrieval probe. Report the actual live offset, not usage.
                question = "Print the letter Z twenty times separated by spaces."
            user, planned = fit_turn(tokenizer, messages, prefix, question, budget, pad[cursor:] + pad[:cursor])
            output_limit = target - planned if filling else OUTPUT_RESERVE if final else 8
            if filling and not 1 <= output_limit <= 16:
                raise RuntimeError("could not fit the final bounded context increment")
            cursor = (cursor + 1931) % len(pad)
            began = time.perf_counter()
            response = http("/v1/chat/completions", {"model": health["model"], "messages": messages + [user],
                            "max_tokens": output_limit, "temperature": 0,
                            "enable_thinking": False, "stream": False})
            elapsed = time.perf_counter() - began
            usage = response["usage"]
            choice = response["choices"][0]
            text = choice["message"]["content"]
            diag = http("/v1/cache/diagnostics")
            live = (diag.get("flashnext_kv") or {}).get("min_attention_offset")
            row = {"target": target, "stage_probe": final, "stage_fill": filling, "planned_prompt_tokens": planned,
                   "usage": usage, "seconds": elapsed, "warm_tokens": diag.get("warm_tokens"),
                   "live_attention_tokens": live,
                   "kv": diag.get("flashnext_kv"), "prefill_seconds": diag.get("actual_prefill_seconds"),
                   "decode_tokens_per_second": diag.get("decode_tokens_per_second"),
                   "finish": choice.get("finish_reason"), "probe_response": text if final else None}
            try:
                prompt, cached = validate_reuse(usage, previous, health["max_cold_prefill"])
                row["cached_tokens"] = cached
                if prompt <= previous or not isinstance(text, str) or not text.strip():
                    raise RuntimeError("context did not grow or the model returned no visible answer")
                if abs(prompt - planned) > 16 or (not final and not filling and prompt - previous < 1024):
                    raise RuntimeError("server tokenization differs from the bounded local plan")
                if final:
                    expected = {"proof": f"ORIGIN-{8359+i*113}:BRIDGE-{6127+i*173}", "total": 222+i*20}
                    row["passed"] = json.loads(text) == expected
                    if not row["passed"]:
                        raise RuntimeError(f"stage {target}: repository retrieval/calculation failed")
                    if prompt < goal - 16:
                        raise RuntimeError(f"stage {target}: actual prompt fell below the planned retrieval depth")
                    retrieval_passed = True
                if filling and (not isinstance(live, int) or live < target):
                    raise RuntimeError(f"stage {target}: measured live attention offset {live} is below target")
                row["host"] = safe()
            except Exception as exc:
                row["error"] = str(exc)
                emit(row)
                raise
            emit(row)
            messages.extend((user, {"role": "assistant", "content": text}))
            previous = prompt
            if filling:
                break


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--checkpoint", required=True, help="local checkpoint directory, for tokenizer only")
    p.add_argument("--max-stage", type=int, choices=STAGES, default=4096)
    p.add_argument("--swap-ceiling-only", action="store_true",
                   help="require an absolute-swap-only server with a 3072 MiB ceiling")
    args = p.parse_args(argv)
    try:
        probe(args.url, args.checkpoint, args.max_stage, lambda row: print(json.dumps(row), flush=True),
              swap_ceiling_only=args.swap_ceiling_only)
    except Exception as exc:
        print(f"context ladder stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
