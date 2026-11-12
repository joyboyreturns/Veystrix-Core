"""Short normal-server comparison. This case measures wire text parity, not token IDs."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
from statistics import median
import subprocess
import sys
import tempfile
import time

from models.flashnext.overnight import swap_used_mb
from models.flashnext.diskio import vm_counters
from veystrix.measurement import MeasurementRun, source_hash
from veystrix.results import ROOT, output_path
from veystrix.model_registry import discover, installation_id

MESSAGES = [{"role": "system", "content": "Give a clear, concise factual answer."},
            {"role": "user", "content": "Explain why the sky appears blue during a clear day."}]
OUTPUT_TOKENS = 8


def counters(pid):
    buffer = (ctypes.c_uint8 * 512)()
    lib = ctypes.CDLL("/usr/lib/libSystem.dylib")
    if lib.proc_pid_rusage(pid, 4, ctypes.byref(buffer)):
        return {}
    return {key: int.from_bytes(bytes(buffer[16 + offset * 8:24 + offset * 8]), "little")
            for key, offset in (("footprint_bytes", 7), ("physical_read_bytes", 16))}


def get(port, path, key=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    try:
        conn.request("GET", path, headers={"Authorization": f"Bearer {key}"} if key else {})
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def listen_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(process, port, key=None):
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode} before readiness")
        try:
            status, health = get(port, "/ready", key)
            if status == 200 and health.get("ready"):
                return health
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError("server readiness timed out")


def request(port, model_id, key=None, *, messages=MESSAGES, max_tokens=OUTPUT_TOKENS, timeout=125):
    payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens,
               "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    started = time.monotonic()
    times, chunks, usage, done = [], [], None, False
    try:
        headers = {"Content-Type": "application/json", "X-Session-Id": "paired-short"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode()}")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    times.append(time.monotonic() - started)
                    chunks.append(delta)
    finally:
        connection.close()
    if not times or not usage or not done:
        raise RuntimeError("stream did not finish with text, usage and terminal marker")
    return {"wire_ttft_seconds": times[0], "wire_delta_seconds": times, "wall_seconds": time.monotonic() - started,
            "usage": usage, "output_text_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest()}


def execute(source, runtime, checkpoint, model_id, arm, result, *, interpreter=sys.executable):
    family = "flashnext" if runtime == "flashnext" else "nemotron35-lightning"
    port = listen_port()
    log_path = result.parent / f"{arm}-server.log"
    flight_path = result.parent / f"{arm}-flight.jsonl"
    usage_path = result.parent / f"{arm}-usage.jsonl"
    with tempfile.TemporaryDirectory(prefix="veystrix-server-measure-") as scratch, log_path.open("x") as log:
        private = Path(scratch)
        prefs = private / "preferences.json"
        prefs.write_text('{"temperature": 0, "thinking_enabled": false}')
        env = os.environ | {
            "VEYSTRIX_PYTHON": interpreter, "PYTHONPATH": str(source),
            "VEYSTRIX_USAGE_LOG": str(usage_path), "FLASHNEXT_FLIGHT_LOG": str(flight_path),
            "FLASHNEXT_PIN_CACHE": str(private / "pins.json"), "FLASHNEXT_WATCHDOG_DIR": str(private),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VEYSTRIX_NEMOTRON_KV": "bf16", "VEYSTRIX_NEMOTRON_AUTHORIZED": "1",
            "VEYSTRIX_SERVER_API_KEY": "",
        }
        command = [interpreter, "-m", "veystrix.cli", "--model", family, "--checkpoint", checkpoint,
                   "--server", "--host", "127.0.0.1", "--port", str(port), "--no-auto-purge",
                   "--preferences-file", str(prefs), "--api-keys-file", str(private / "keys.json"),
                   "--prefix-dir", str(private / "prefix")]
        if runtime == "flashnext":
            command.extend(("--enable", "short-throughput"))
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            ready = wait_ready(process, port)
            events = [json.loads(line) for line in flight_path.read_text().splitlines()]
            worker_pid = events[-1]["pid"]
            before, vm_before, swap_before = counters(worker_pid), vm_counters(), swap_used_mb()
            metrics = request(port, model_id)
            after = counters(worker_pid)
            _, health = get(port, "/health")
            diagnostics_status, diagnostics = get(port, "/v1/cache/diagnostics")
            usage = json.loads(usage_path.read_text().splitlines()[-1])
            events = [json.loads(line) for line in flight_path.read_text().splitlines()]
            decode = next((event for event in events if event["event"] == "generate_end"), {})
            read = after.get("physical_read_bytes", -1) - before.get("physical_read_bytes", -1)
            metrics.update({"ready": ready, "final_health": health, "runtime_usage": usage,
                            "diagnostics_http_status": diagnostics_status, "cache_diagnostics": diagnostics,
                            "runtime_decode_tokens_per_second": decode.get("decode_tokens_per_second"),
                            "uncached_prefill_tokens_per_second": usage["uncached"] / usage["prefill_s"] if usage["prefill_s"] else None,
                            "physical_read_bytes": read if before and after else None,
                            "process_footprint_bytes": after.get("footprint_bytes"),
                            "swap_mib_before": swap_before, "swap_mib_after": swap_used_mb(),
                            "vm_counters_before": vm_before, "vm_counters_after": vm_counters()})
            return command, metrics
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("flashnext", "nemotron35"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--packaged", action="store_true")
    parser.add_argument("--cross-model", action="store_true")
    args = parser.parse_args()
    provenance = Path(os.environ["VEYSTRIX_BENCH_PROVENANCE"]).resolve()
    family = "flashnext" if args.runtime == "flashnext" else "nemotron35-lightning"
    model_id = installation_id(family, args.checkpoint)
    if args.cross_model:
        return cross_model_smoke(args, provenance, model_id)
    if args.packaged:
        return packaged_smoke(args, provenance, model_id)
    base = Path(os.environ["VEYSTRIX_BENCH_BASELINE_SOURCE"]).resolve()
    sources = {"baseline": base, "candidate": ROOT}
    result = output_path(args.runtime, "server-short-paired", "server-arms.jsonl")
    manifests = {name: source_hash(provenance / f"{name}-files.json") for name in sources}
    run = MeasurementRun(result, runtime=args.runtime, experiment="server-short-paired",
                         metadata={"diagnostic": True, "checkpoint": args.checkpoint, "installed_model_id": model_id,
                                   "source_manifest_sha256": manifests, "revision": os.environ["VEYSTRIX_BENCH_REVISION"],
                                   "messages": MESSAGES, "temperature": 0, "max_tokens": OUTPUT_TOKENS,
                                   "reasoning": "none", "kv": "bf16", "cache": "fresh application cache; OS file cache retained"})
    run.start()
    metrics = {name: [] for name in sources}
    for round_index, order in enumerate((tuple(sources), tuple(reversed(sources)), tuple(sources)), start=1):
        for name in order:
            arm = f"{round_index}-{name}"
            try:
                command, measured = execute(sources[name], args.runtime, args.checkpoint, model_id, arm, result)
            except Exception as exc:
                run.arm(arm_id=arm, condition=name, round_index=round_index, command=[], status="failed")
                run.failure(str(exc), arm_id=arm)
                print(f"{arm} failed: {exc}", flush=True)
                return 1
            run.arm(arm_id=arm, condition=name, round_index=round_index, command=command, metrics=measured,
                    source_root=str(sources[name]), source_manifest_sha256=manifests[name])
            metrics[name].append(measured)
            print(f"{arm} TTFT {measured['wire_ttft_seconds']:.2f}s, "
                  f"runtime decode {measured['runtime_decode_tokens_per_second']} tok/s", flush=True)
    digests = {row["output_text_sha256"] for rows in metrics.values() for row in rows}
    keys = ("wire_ttft_seconds", "uncached_prefill_tokens_per_second", "runtime_decode_tokens_per_second", "physical_read_bytes")
    summary = {name: {key: median(values) for key in keys
                      if (values := [row[key] for row in rows if row[key] is not None])}
               for name, rows in metrics.items()}
    run.finish("diagnostic" if len(digests) == 1 else "failed", text_parity=len(digests) == 1, medians=summary)
    return int(len(digests) != 1)


def packaged_smoke(args, provenance, model_id):
    engine = Path(os.environ["VEYSTRIX_BENCH_PACKAGED_ENGINE"]).resolve()
    result = output_path(args.runtime, "packaged-server-smoke", "packaged-arm.jsonl")
    run = MeasurementRun(result, runtime=args.runtime, experiment="packaged-server-smoke",
                         metadata={"diagnostic": True, "checkpoint": args.checkpoint,
                                   "installed_model_id": model_id, "engine": str(engine),
                                   "revision": os.environ["VEYSTRIX_BENCH_REVISION"],
                                   "candidate_manifest_sha256": source_hash(provenance / "final-files.json"),
                                   "engine_manifest_sha256": source_hash(engine / "BUNDLE-MANIFEST.txt")})
    run.start()
    try:
        command, metrics = execute(engine / "lib", args.runtime, args.checkpoint, model_id, "packaged", result,
                                   interpreter=str(engine / "python/bin/python3"))
    except Exception as exc:
        run.failure(str(exc), arm_id="packaged")
        print(f"Packaged server failed: {exc}", flush=True)
        return 1
    run.arm(arm_id="packaged", condition="packaged", round_index=1, command=command, metrics=metrics)
    passed = metrics["diagnostics_http_status"] == 200 and metrics["final_health"].get("ready") is True
    run.finish("diagnostic" if passed else "failed", passed=passed)
    print(f"Packaged server TTFT {metrics['wire_ttft_seconds']:.2f}s, diagnostics HTTP {metrics['diagnostics_http_status']}", flush=True)
    return int(not passed)


def cross_model_smoke(args, provenance, initial_id):
    import secrets

    engine = Path(os.environ["VEYSTRIX_BENCH_PACKAGED_ENGINE"]).resolve()
    models = discover()
    other, = [model for model in models if model.family == "nemotron35-lightning"]
    result = output_path(args.runtime, "cross-model-input-smoke", "switch-arms.jsonl")
    run = MeasurementRun(result, runtime=args.runtime, experiment="cross-model-input-smoke",
                         metadata={"diagnostic": True, "models": [initial_id, other.id],
                                   "engine_manifest_sha256": source_hash(engine / "BUNDLE-MANIFEST.txt"),
                                   "final_manifest_sha256": source_hash(provenance / "final-files.json")})
    run.start()
    interpreter = str(engine / "python/bin/python3")
    source = engine / "lib"
    port, key = listen_port(), secrets.token_hex(24)
    with tempfile.TemporaryDirectory(prefix="veystrix-cross-model-") as scratch, (
        result.parent / "router.log"
    ).open("x") as log:
        private = Path(scratch)
        command = [interpreter, "-m", "veystrix.model_router", "--port", str(port),
                   "--checkpoint", args.checkpoint, "--prefix-dir", str(private / "prefix")]
        env = os.environ | {
            "VEYSTRIX_PYTHON": interpreter, "PYTHONPATH": str(source),
            "VEYSTRIX_SERVER_API_KEY": key, "VEYSTRIX_USAGE_LOG": "0",
            "FLASHNEXT_FLIGHT_LOG": str(result.parent / "router-flight.jsonl"),
            "FLASHNEXT_PIN_CACHE": str(private / "pins.json"), "FLASHNEXT_WATCHDOG_DIR": str(private),
            "FLASHNEXT_AUTO_PURGE": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VEYSTRIX_NEMOTRON_KV": "bf16", "VEYSTRIX_NEMOTRON_AUTHORIZED": "1",
        }
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            wait_ready(process, port, key)
            for index, model_id in enumerate((initial_id, other.id, initial_id)):
                metrics = request(port, model_id, key)
                _, health = get(port, "/health", key)
                metrics["input_reuse"] = health.get("input_reuse")
                metrics["installed_model_id"] = health.get("installed_model_id")
                run.arm(arm_id=str(index), condition=model_id, round_index=index, command=command, metrics=metrics)
                cached = metrics["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
                if (metrics["installed_model_id"] != model_id
                        or metrics["input_reuse"] != {"cross_model_hits": index} or cached != 0):
                    raise RuntimeError("unexpected model identity, input reuse count, or restored token count")
                print(f"switch {index} input hits {index}, model tokens restored {cached}", flush=True)
            run.finish("diagnostic", verified_directions=2)
            return 0
        except Exception as exc:
            run.failure(str(exc))
            print(f"Router smoke failed: {exc}", flush=True)
            return 1
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
