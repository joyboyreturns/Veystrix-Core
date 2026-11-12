"""Measure pooled cold prefill and identical-prompt replay on the signed server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading

from models.flashnext.memory_policy import FLAG_SWAP_CEILING_MIB, absolute_swap_ceiling_mib
from models.flashnext.overnight import swap_used_mb
from veystrix import effort_slot
from veystrix.conversation import reasoning_system_text, split_content_slots
from veystrix.measurement import MeasurementRun, source_hash
from veystrix.model_registry import installation_id
from veystrix.results import ROOT, output_path
from veystrix.server import ModelService, build_user_encoder

from .bench_server_short_paired import counters, get, listen_port, request, wait_ready

OUTPUT_TOKENS = 32
TARGETS = (8192, 32768)
FILES = (
    "docs/chat/research.md", "veystrix/server.py", "veystrix/conversation.py",
    "veystrix/model_router.py", "models/flashnext/prefill.py", "models/flashnext/expert_cache.py",
    "models/flashnext/prefix_cache.py", "veystrix/backends/nemotron35.py",
    "models/nemotron35/streamed_experts.py", "models/nemotron35/backend_core.py",
)


def prompts(checkpoint):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    safe_content = build_user_encoder(tokenizer)[1]
    corpus = "\n\n".join(f"# {name}\n" + (ROOT / name).read_text() for name in FILES)
    prefix = ("Use this repository excerpt as reference. Explain the code paths and their performance implications "
              "in complete paragraphs.\n\n")
    question = "Explain how the server manages prompt caching and streams responses. Give concrete examples."

    def build(length):
        messages = [{"role": "system", "content": prefix + corpus[:length]},
                    {"role": "user", "content": question}]
        if effort_slot.placement() == "late":
            system, effort = messages[0]["content"], "medium"
        else:
            system, effort = reasoning_system_text(messages[0]["content"], "none")
        framed, slots = ModelService._substitute_slots([dict(messages[0], content=system), messages[1]], tokenizer)
        rendered = tokenizer.apply_chat_template(
            framed, add_generation_prompt=True,
            tokenize=False, enable_thinking=False, reasoning_effort=effort,
        )
        if not slots:
            return len(tokenizer.encode(rendered, add_special_tokens=False)), messages
        parts = split_content_slots(rendered, len(slots))
        if parts is None:
            raise RuntimeError("the tokenizer's template changed a content placeholder")
        structural = sum(len(tokenizer.encode(part, add_special_tokens=False)) for part in parts)
        return structural + sum(len(safe_content(slot)) for slot in slots), messages

    result = {}
    for target in TARGETS:
        low, high = 0, len(corpus)
        best = build(low)
        while low <= high:
            middle = (low + high) // 2
            size, messages = build(middle)
            if size <= target:
                if size > best[0]:
                    best = (size, messages)
                low = middle + 1
            else:
                high = middle - 1
        if target - best[0] > 16:
            raise RuntimeError(f"could not size prompt for {target}: {best[0]}")
        result[target] = best
    return result


def stop_process(process):
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            continue


def last_json(path):
    return json.loads(path.read_text().splitlines()[-1])


def run_size(run, result, engine, family, checkpoint, model_id, target, estimated, messages, swap_limit):
    interpreter = str(engine / "python/bin/python3")
    source = engine / "lib"
    port = listen_port()
    usage_path = result.parent / f"{target}-usage.jsonl"
    flight_path = result.parent / f"{target}-flight.jsonl"
    log_path = result.parent / f"{target}-server.log"
    swap = {"peak_mib": swap_used_mb(), "crossed_limit": False}
    stop = threading.Event()
    ok = True
    with tempfile.TemporaryDirectory(prefix="veystrix-context-cache-") as scratch, log_path.open("x") as log:
        private = Path(scratch)
        prefs = private / "preferences.json"
        prefs.write_text('{"temperature": 0, "thinking_enabled": false}')
        env = os.environ | {
            "VEYSTRIX_PYTHON": interpreter, "PYTHONPATH": str(source),
            "VEYSTRIX_USAGE_LOG": str(usage_path), "FLASHNEXT_FLIGHT_LOG": str(flight_path),
            "FLASHNEXT_PIN_CACHE": str(private / "pins.json"), "FLASHNEXT_WATCHDOG_DIR": str(private),
            "FLASHNEXT_MEMORY_POLICY": "absolute-swap-only", FLAG_SWAP_CEILING_MIB: str(swap_limit),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "VEYSTRIX_NEMOTRON_KV": "bf16",
            "VEYSTRIX_NEMOTRON_AUTHORIZED": "1", "VEYSTRIX_SERVER_API_KEY": "",
        }
        command = [interpreter, "-m", "veystrix.cli", "--model", family, "--checkpoint", checkpoint,
                   "--server", "--host", "127.0.0.1", "--port", str(port), "--no-auto-purge",
                   "--pool-gb", "2", "--context-limit", "65536",
                   "--preferences-file", str(prefs), "--api-keys-file", str(private / "keys.json"),
                   "--prefix-dir", str(private / "prefix")]
        if family == "flashnext":
            command.extend(("--enable", "short-throughput", "--max-cold-prefill", "32768"))
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)

        def watch_swap():
            while not stop.is_set():
                used = swap_used_mb()
                swap["peak_mib"] = max(used, swap["peak_mib"])
                if used >= swap_limit:
                    swap["crossed_limit"] = True
                    print(f"{target}: stop at {used:.2f} MiB total swap", flush=True)
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    return
                stop.wait(1)

        watcher = threading.Thread(target=watch_swap, daemon=True)
        watcher.start()
        cold_digest = None
        condition = "cold"
        try:
            ready = wait_ready(process, port)
            pid = json.loads(flight_path.read_text().splitlines()[0])["pid"]
            for condition in ("cold", "warm"):
                arm = f"{target}-{condition}"
                before, swap_before = counters(pid), swap_used_mb()
                print(f"{target} {condition}: request started (estimated {estimated} prompt tokens, "
                      f"swap {swap_before:.2f} MiB)", flush=True)
                metrics = request(port, model_id, messages=messages, max_tokens=OUTPUT_TOKENS, timeout=1800)
                after = counters(pid)
                _, health = get(port, "/health")
                _, diagnostics = get(port, "/v1/cache/diagnostics")
                usage = last_json(usage_path)
                flight = [json.loads(line) for line in flight_path.read_text().splitlines()]
                decode = next(event for event in reversed(flight) if event["event"] == "generate_end")
                pool = health.get("engine", {}).get("pool") or [
                    line for line in log_path.read_text().splitlines() if "FlashNext expert pool:" in line
                ]
                metrics.update({
                    "runtime_usage": usage, "decode_tokens_per_second": decode["decode_tokens_per_second"],
                    "uncached_prefill_tokens_per_second": usage["uncached"] / usage["prefill_s"] if usage["prefill_s"] else None,
                    "physical_read_bytes": after["physical_read_bytes"] - before["physical_read_bytes"] if before and after else None,
                    "pool": pool, "swap_mib_before": swap_before, "swap_mib_after": swap_used_mb(),
                    "size_peak_swap_mib": swap["peak_mib"], "ready": ready, "health": health, "cache": diagnostics,
                })
                digest = metrics["output_text_sha256"]
                run.arm(arm_id=arm, condition=condition, round_index=1, command=command, metrics=metrics)
                if condition == "cold":
                    cold_digest = digest
                if abs(usage["prompt"] - target) > 16 or (condition == "cold" and usage["cached"]):
                    run.failure("unexpected prompt size or cold-cache state", arm_id=arm)
                    ok = False
                if condition == "warm" and (not usage["cached"] or cold_digest != digest):
                    run.failure("warm replay missed prefix cache or changed greedy text", arm_id=arm)
                    ok = False
                print(f"{target} {condition}: {usage['prompt']} prompt, {usage['cached']} cached, "
                      f"prefill {metrics['uncached_prefill_tokens_per_second']} tok/s, "
                      f"decode {metrics['decode_tokens_per_second']} tok/s, TTFT {metrics['wire_ttft_seconds']:.2f}s",
                      flush=True)
        except Exception as exc:
            run.failure(str(exc), arm_id=f"{target}-{condition}", swap=swap.copy())
            print(f"{target} {condition} failed: {exc}", flush=True)
            ok = False
        finally:
            stop.set()
            watcher.join()
            stop_process(process)
    return ok, swap["crossed_limit"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("flashnext", "nemotron35"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", type=int, choices=TARGETS, default=os.environ.get("VEYSTRIX_BENCH_CONTEXT_SIZE"))
    args = parser.parse_args()
    swap_limit = absolute_swap_ceiling_mib()
    engine = Path(os.environ["VEYSTRIX_BENCH_PACKAGED_ENGINE"]).resolve()
    family = "flashnext" if args.runtime == "flashnext" else "nemotron35-lightning"
    model_id = installation_id(family, args.checkpoint)
    inputs = prompts(args.checkpoint)
    sizes = (args.size,) if args.size else TARGETS
    result = output_path(args.runtime, "server-context-cache", "context-arms.jsonl")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run = MeasurementRun(result, runtime=args.runtime, experiment="server-context-cache", metadata={
        "diagnostic": True, "revision": revision, "checkpoint": args.checkpoint, "installed_model_id": model_id,
        "engine_manifest_sha256": source_hash(engine / "BUNDLE-MANIFEST.txt"),
        "harness_sha256": source_hash(Path(__file__)),
        "prompt_sha256": {str(target): hashlib.sha256(json.dumps(inputs[target][1]).encode()).hexdigest()
                          for target in sizes},
        "cold": "fresh process and isolated prefix directory; OS file cache retained",
        "warm": "identical prompt in the same process after cold request",
        "output_cap": OUTPUT_TOKENS, "temperature": 0, "kv": "bf16",
        "requested_pool_gb": 2, "swap_limit_mib": swap_limit,
    })
    run.start()
    outcomes = {}
    for target in sizes:
        if swap_used_mb() >= swap_limit:
            run.failure("absolute swap ceiling reached before startup", arm_id=str(target))
            break
        estimated, messages = inputs[target]
        outcomes[str(target)], stopped = run_size(run, result, engine, family, args.checkpoint,
                                                  model_id, target, estimated, messages, swap_limit)
        if stopped:
            break
    complete = len(outcomes) == len(sizes) and all(outcomes.values())
    run.finish("diagnostic" if complete else "incomplete", sizes=outcomes)
    return int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
