"""Supervised protocol/integration smoke on the real checkpoint with the layer-major opt-ins. Not a speed or exactness result.

One supervised AGENTIC server (``bench_layer_gate.run_arm``: independent watchdog, swap/power/thermal stops) started with
``--enable layer-major --enable layer-prefetch`` and driven through, in order:

1. a tool-enabled chat request (non-streaming): does the model return a tool call, and is the HTTP envelope sound;
2. the follow-up turn with the tool result, streaming (SSE): chunks, finish reason, ``[DONE]``, and the prefix-cache counters
   before and after (a follow-up should reuse the first turn's prefix);
3. a CANCEL in the middle of an ~8K layer-major prefill (the client drops the connection): the server must stay alive, return to
   ready, and answer the next request;
4. Anthropic ``/v1/messages`` and OpenAI ``/v1/responses`` requests.

This exercises the opt-in path through the normal server, not the real Qwen Code or DeepSeek Harness clients.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from macqwen.measurement import MeasurementRun
from macqwen.results import output_path
from models.flashnext.tests.bench import bench_layer_gate as G

TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a repository file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}


def post(url, path, payload, timeout=600):
    request = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": "Bearer local", "x-api-key": "local",
        "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, {"error": error.read().decode(errors="replace")[:500]}


def stream(url, path, payload, timeout=600):
    request = urllib.request.Request(url + path, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": "Bearer local"})
    events, done, text = 0, False, []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                done = True
                continue
            events += 1
            try:
                delta = json.loads(body)["choices"][0].get("delta", {})
                text.append(delta.get("content") or "")
            except (ValueError, KeyError, IndexError):
                pass
    return {"status": status, "events": events, "done": done, "text": "".join(text)[:300]}


def get(url, path):
    try:
        return G.http_json(url + path, timeout=10)
    except Exception as error:                       # recorded as data
        return {"error": repr(error)}


def counters(url):
    diagnostics = get(url, "/v1/cache/diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def wait_ready(url, limit=180):
    began = time.monotonic()
    while time.monotonic() - began < limit:
        health = get(url, "/health")
        if isinstance(health, dict) and health.get("ready") and not health.get("busy"):
            return round(time.monotonic() - began, 1), health
        time.sleep(2)
    return None, get(url, "/health")


def make_driver(checkpoint, report):
    base = G.make_prompt(checkpoint, 100)
    system = base["messages"][0]
    big = G.make_prompt(checkpoint, 6000)

    def driver(url):
        # 1. tool-enabled turn
        user = {"role": "user", "content": "Use the read_file tool to read models/flashnext/layer_major.py, then wait for the result."}
        first = {"model": "flashnext", "messages": [system, user], "tools": [TOOL], "tool_choice": "auto", "max_tokens": 96,
                 "stream": False}
        status, body = post(url, "/v1/chat/completions", first)
        choice = ((body.get("choices") or [{}])[0]) if isinstance(body, dict) else {}
        calls = (choice.get("message") or {}).get("tool_calls") or []
        report["tool_turn"] = {"status": status, "finish_reason": choice.get("finish_reason"), "tool_calls": len(calls),
                               "first_call": calls[0]["function"] if calls else None,
                               "content": ((choice.get("message") or {}).get("content") or "")[:200]}
        # 2. follow-up with the tool result, streaming
        assistant = choice.get("message") or {"role": "assistant", "content": ""}
        if calls:
            tool_msg = {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": "def window_tokens(): return 8192"}
            followup = [system, user, {"role": "assistant", "content": assistant.get("content") or "", "tool_calls": calls}, tool_msg]
        else:
            followup = [system, user, {"role": "assistant", "content": assistant.get("content") or "ok"},
                        {"role": "user", "content": "Summarise in one sentence what a layer-major window is."}]
        before = counters(url)
        report["follow_up"] = stream(url, "/v1/chat/completions", {"model": "flashnext", "messages": followup,
                                     "tools": [TOOL], "max_tokens": 64, "stream": True})
        report["cache_counters"] = {"before_follow_up": before, "after_follow_up": counters(url)}
        # 3. cancel in the middle of an 8K layer-major prefill
        host, port = url.split("//")[1].split(":")
        body = json.dumps({**big, "stream": True}).encode()
        sock = socket.create_connection((host, int(port)), timeout=30)
        sock.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
                      f"Authorization: Bearer local\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
        time.sleep(20)                                # well inside the ~90 s prefill
        sock.close()
        report["cancel"] = {"dropped_after_s": 20}
        ready_after, health = wait_ready(url)
        report["cancel"].update(ready_after_s=ready_after, ready=bool(ready_after is not None))
        after_cancel = post(url, "/v1/chat/completions", {"model": "flashnext", "messages": [
            {"role": "user", "content": "Reply with the single word: pong"}], "max_tokens": 16, "stream": False})
        report["after_cancel"] = {"status": after_cancel[0], "text": str(((after_cancel[1].get("choices") or [{}])[0]
                                  .get("message") or {}).get("content"))[:80]}
        # 4. other protocols
        status, body = post(url, "/v1/messages", {"model": "flashnext", "max_tokens": 16, "messages": [
            {"role": "user", "content": "Reply with the single word: pong"}]})
        report["anthropic"] = {"status": status, "text": json.dumps(body.get("content"))[:120] if isinstance(body, dict) else None}
        status, body = post(url, "/v1/responses", {"model": "flashnext", "input": "Reply with the single word: pong",
                                                   "max_output_tokens": 16})
        report["responses"] = {"status": status, "keys": sorted(body)[:8] if isinstance(body, dict) else None}
        report["final_health"] = get(url, "/health")
        return report

    return driver


def verdict(report):
    problems = []
    if report.get("tool_turn", {}).get("status") != 200:
        problems.append("tool turn failed")
    follow = report.get("follow_up") or {}
    if follow.get("status") != 200 or not follow.get("done") or not follow.get("events"):
        problems.append("streamed follow-up incomplete")
    if not report.get("cancel", {}).get("ready"):
        problems.append("server did not return to ready after the cancel")
    if report.get("after_cancel", {}).get("status") != 200:
        problems.append("request after the cancel failed")
    for key in ("anthropic", "responses"):
        if report.get(key, {}).get("status") != 200:
            problems.append(f"{key} request failed")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=G.CHECKPOINT)
    parser.add_argument("--pool-gb", default="2")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    initial, postponed, purge = G.prepare_headroom(candidate=True, memory_rules="swap-only", power_rules="ac")
    if postponed:
        print(f"Smoke postponed (nothing started): {postponed}", flush=True)
        return 2
    path = output_path("flashnext", "integration-smoke", "integration-smoke.jsonl")
    run = MeasurementRun(path, runtime="flashnext", experiment="integration-smoke", metadata={
        "checkpoint": str(checkpoint), "pool_gb": args.pool_gb, "memory_rules": "swap-only", "initial_machine": initial,
        "purpose": "protocol and cancel smoke with layer-major + layer-prefetch; not a speed or exactness result"})
    run.start()
    report = {}
    try:
        command, evidence, machine = G.run_arm("layer", "smoke", checkpoint, None, path.parent, candidate="layer-major",
                                               uncached=True, memory_rules="swap-only", prefetch=True, base_pool_gb=args.pool_gb,
                                               reads="prefill", power_rules="ac", driver=make_driver(checkpoint, report))
        run.arm(arm_id="smoke", condition="layer", round_index=1, command=command, metrics=evidence, **machine)
        problems = verdict(report)
        run.validation("integration-smoke", "fail" if problems else "pass", reason="; ".join(problems) or None)
        print(json.dumps(report, indent=1, default=str), flush=True)
        run.finish("completed" if not problems else "stopped", report=report, problems=problems)
        print("PROBLEMS: " + "; ".join(problems) if problems else "All protocol and cancel checks passed", flush=True)
        return 1 if problems else 0
    except Exception as error:
        run.failure(str(error), error_type="incomplete_or_failed_smoke")
        run.finish("stopped", reason=str(error), partial_report=report)
        print(json.dumps(report, indent=1, default=str), flush=True)
        print(f"Smoke stopped: {error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
