"""Supervised, exact layer-order gate. Diagnostic only; never promotes a speed result.

Each arm starts ./serve.sh on a private port with a private prefix cache and
the AGENTIC guard and independent stall watchdog. Three reversed pairs use the
same checkpoint and coding prompt. Stops immediately on any safety, startup,
missing-evidence or exactness failure. Does not purge the macOS file cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

from macqwen.measurement import MeasurementRun, source_hash
from macqwen.results import output_path
from models.flashnext.tests.bench import prefill_memory_harness as H
from models.flashnext.tests.bench.bench_prefill_memory import _memory_reading, _vm_counters

ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
SOURCES = ("macqwen/backends/flashnext.py", "macqwen/server.py", "models/flashnext/prefill.py",
           "models/flashnext/layer_major.py", "models/flashnext/expert_cache.py",
           "models/flashnext/prefix_session.py", "models/flashnext/tests/bench/layer_gate_probe.py",
           "models/flashnext/tests/bench/bench_layer_gate.py", "models/flashnext/overnight.py",
           "models/flashnext/settings/modes.py", "serve.sh", "chat.sh")


def machine_sample():
    sample = H.Sampler(vm=_vm_counters, memory=_memory_reading).sample()
    # pmset -g omits this field on some macOS versions; read the profile of the ACTIVE power source (AC or Battery).
    custom = H._run_text(["pmset", "-g", "custom"])
    heading = "Battery Power:" if sample.get("power_source") == "battery" else "AC Power:"
    other = "AC Power:" if heading == "Battery Power:" else "Battery Power:"
    active = custom.split(heading, 1)[1].split(other, 1)[0] if heading in custom else ""
    match = re.search(r"^\s*lowpowermode\s+(\d+)", active, re.MULTILINE)
    sample["low_power"] = int(match.group(1)) != 0 if match else None
    thermal = H._run_text(["pmset", "-g", "therm"])
    sample["thermal_readable"] = ("Error:" not in thermal and
                                  ("No thermal warning level has been recorded" in thermal or
                                   "CPU_Speed_Limit" in thermal))
    # OS thermal pressure (0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping); readable without root, None when unavailable.
    # Recorded in every sample so a slowdown that tracks it can be told apart from a memory effect.
    level = H._run_text(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"])
    match = re.search(r"thermalpressurelevel\s+(\d+)\s*$", level.strip())
    sample["thermal_pressure"] = int(match.group(1)) if match else None
    return sample


SWAP_CEILING_MB = 3 * 1024      # owner rule (2026-11-09): swap in use must not reach 3 GB


def safety_reason(sample, baseline=None, memory_rules="strict", power_rules="ac"):
    """Why a run must stop. ``strict`` is the original gate (pressure level 1, 3 GiB available floor, compressor growth).

    ``power_rules`` is ``ac`` (AC and 25% required) or ``battery-ok`` (owner rule: battery power is allowed until 25%).
    ``swap-only`` is the owner's rule for the performance runs: the only in-flight memory limit is swap, which must stay below
    3 GiB (``SWAP_CEILING_MB``), plus a stop at critical pressure (level 4, where macOS may kill processes). Power, battery,
    Low Power Mode and thermal checks apply in both; so does the server's own watchdog.
    """
    source, percent = sample.get("power_source"), sample.get("battery_percent") or 0
    if power_rules == "battery-ok":
        # Owner rule 2026-11-09: run on battery until the charge reaches 25%. Below 25% stops, on either source.
        if source not in ("ac", "battery") or percent < 25:
            return "battery charge is below 25% or the power source is unreadable"
    elif source != "ac" or percent < 25:
        return "AC power and a battery charge of at least 25% are required"
    if source == "ac" and sample.get("battery_state") == "discharging":
        return "battery is discharging on reported AC"
    if sample.get("low_power") is not False:
        return "Low Power Mode is enabled or unreadable"
    if not sample.get("thermal_readable") or sample.get("thermal_warning"):
        return "thermal status is unhealthy or unreadable"
    if sample.get("thermal_speed_limit") is not None and sample["thermal_speed_limit"] < 100:
        return "CPU thermal speed limit is below 100%"
    if sample.get("thermal_pressure") is not None and sample["thermal_pressure"] >= 3:
        return "OS thermal pressure is trapping or worse"
    if memory_rules == "swap-only":
        swap = sample.get("swap_used_mb")
        if swap is None or sample.get("pressure_level") is None:
            return "memory pressure or swap telemetry is unreadable"
        if swap >= SWAP_CEILING_MB:
            return f"swap in use reached {swap:.0f} MiB (owner ceiling {SWAP_CEILING_MB} MiB)"
        if sample["pressure_level"] >= 4:
            return "macOS memory pressure is critical"
        return None
    if sample.get("pressure_level") != 1 or sample.get("swap_used_mb") is None:
        return "memory pressure or swap telemetry is unsafe or unreadable"
    available = sample.get("available_mb")
    if available is None or available < 3 * 1024:
        return "available memory is below 3 GiB or unreadable"
    compressor = sample.get("compressor_mb")
    initial = (baseline or {}).get("compressor_mb")
    if compressor is None:
        return "compressor telemetry is unreadable"
    if initial is not None and compressor - initial > 2 * 1024 and available < 4 * 1024:
        return "compressor grew over 2 GiB while available memory fell below 4 GiB"
    return None


# Headroom budget: a start condition, in addition to (never instead of) the in-flight floors above.
# The stopped 2026-11-09 reference arm began at 8,540.2 MiB available and its minimum was 2,693.5 MiB (a 5,846.7 MiB drop;
# results/flashnext/20261109-185459-layer-major-gate/reference-safety-trace.jsonl). A start that leaves less than the
# floor plus that footprint plus a margin will trip the 3 GiB floor again, so the arm is postponed before any process
# starts. The owner-run gate of the same day (results/flashnext/20261109-194853-layer-gate-manual/) dropped from 11,426.0 to
# 4,300.8 MiB (7,125 MiB), so the footprint was raised to 7,300 MiB. It is a measurement of two runs, not a bound: it is
# rounded up, a later run that drops further must raise it, and nothing may lower it to admit a run. It does NOT guarantee
# macOS pressure level 1: that run stopped at level 2 with 4.5 GiB still available, which this budget cannot predict.
FLOOR_MB = 3 * 1024
OBSERVED_FOOTPRINT_MB = 7300
HEADROOM_MARGIN_MB = 512
# The layer-major candidate has NOT been measured. It holds one layer's 512 experts (512 x 3,072,000 B = 1,500 MiB) as the
# expert operand where the reference holds about 400 (about 1,170 MiB), plus per-window hidden states (about 80 MiB at 4,096
# tokens, several live copies). That is roughly +0.4 GiB; 1,024 MiB is allowed so an unmeasured arm is not budgeted at the
# reference's footprint. Replace it with a measured value once a candidate arm has run.
CANDIDATE_EXTRA_MB = 1024
REQUIRED_START_MB = FLOOR_MB + OBSERVED_FOOTPRINT_MB + HEADROOM_MARGIN_MB
REQUIRED_CANDIDATE_START_MB = REQUIRED_START_MB + CANDIDATE_EXTRA_MB


def headroom_reason(sample, candidate=False):
    """Why an arm must not start yet, or None when the start leaves room for the observed footprint above the floor."""
    available = (sample or {}).get("available_mb")
    if available is None:
        return "available memory is unreadable, so the headroom budget cannot be checked"
    required = REQUIRED_CANDIDATE_START_MB if candidate else REQUIRED_START_MB
    if available < required:
        return (f"headroom budget: {available:.0f} MiB available at start; the footprint of one "
                f"{'candidate' if candidate else 'reference'} arm plus the {FLOOR_MB} MiB floor and a "
                f"{HEADROOM_MARGIN_MB} MiB margin needs {required} MiB. "
                "Postponed: free memory or reduce the footprint, never lower the floor")
    return None


PURGE_COMMAND = ("sudo", "-n", "/usr/sbin/purge")      # the passwordless rule the server's own auto-purge uses
PURGE_SETTLE_SECONDS = 3.0


def _run_purge():
    return subprocess.run(PURGE_COMMAND, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)


def prepare_headroom(sampler=None, candidate=False, runner=None, sleep=time.sleep, memory_rules="strict", power_rules="ac"):
    """Sample memory; if the headroom budget is short and macOS is not critical, purge the file cache once and re-sample.

    The owner authorized purging whenever memory is needed. A purge only drops reclaimable file cache (measured +2.8 GiB on
    2026-11-09, no pressure or swap change); it runs here, before any server exists, so it never overlaps inference or a timed
    region, and it is never retried in a loop. Under ``strict`` rules a short budget still postpones the run (the budget
    protects the 3 GiB available floor). Under ``swap-only`` (the owner's rule) there is no available-memory floor, so a short
    budget is recorded (``headroom_short``) but never blocks: the only memory stop is swap, checked live. Returns
    ``(sample, reason, record)``.
    """
    sampler = sampler or machine_sample
    swap_only = memory_rules == "swap-only"

    def judge(row):
        why = safety_reason(row, memory_rules=memory_rules, power_rules=power_rules)
        return why if why or swap_only else headroom_reason(row, candidate=candidate)

    sample = sampler()
    reason = judge(sample)
    short = headroom_reason(sample, candidate=candidate) is not None
    record = {"purged": False, "available_before_mb": sample.get("available_mb"), "headroom_short": short}
    unsafe = safety_reason(sample, memory_rules=memory_rules, power_rules=power_rules) is not None
    if not short or unsafe or sample.get("pressure_level") not in (1, 2):
        return sample, reason, record
    try:
        result = (runner or _run_purge)()
        record["returncode"] = result.returncode
        record["purged"] = result.returncode == 0
        if not record["purged"]:
            record["error"] = (result.stderr or "")[:200]
    except (OSError, subprocess.SubprocessError) as error:
        record["error"] = repr(error)
    if not record["purged"]:
        return sample, reason, record
    sleep(PURGE_SETTLE_SECONDS)
    sample = sampler()
    record["available_after_mb"] = sample.get("available_mb")
    record["headroom_short"] = headroom_reason(sample, candidate=candidate) is not None
    return sample, judge(sample), record


def array_evidence_valid(value):
    return (isinstance(value, dict) and isinstance(value.get("shape"), list)
            and all(isinstance(n, int) and n >= 0 for n in value["shape"])
            and isinstance(value.get("dtype"), str) and bool(value["dtype"])
            and isinstance(value.get("sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value["sha256"])))


def validate_pair(left, right, expected_order="layer-major-v1/w4096", token_range=(2048, 4096)):
    """No missing fields, matching token and cache identities, matching chunk cuts."""
    for row in (left, right):
        if set(row) != {"logits", "prefilled", "finished"}:
            return "missing logits, prefill or completion probe"
        p, f = row["prefilled"], row["finished"]
        prefix = p.get("prefix") or {}
        state = p.get("state") or {}
        if not (token_range[0] <= state.get("token_count", 0) <= token_range[1]):
            return f"prompt is outside the {token_range[0]}-{token_range[1]} token range of this run"
        if prefix.get("hit_tokens") != 0 or prefix.get("prefilled_tokens") != state["token_count"]:
            return "arm did not prefill a cold, complete prompt"
        if not prefix.get("executed_edges"):
            return "the prompt did not cross a chunk boundary"
        layers = state.get("layers") or []
        if any(not isinstance(layer, dict) for layer in layers):
            return "malformed cache state"
        kinds = [layer.get("kind") for layer in layers]
        if (len(layers) != 48 or kinds.count("arrays") != 36 or kinds.count("qsa_kv") != 12
                or not state.get("language") or len(f.get("token_ids") or []) != 32):
            return "state or generated tokens missing"
        if (not isinstance(state["language"], dict) or
                any(not isinstance(layer.get("states"), list) for layer in layers if layer["kind"] == "arrays") or
                any(not isinstance(layer.get("kv"), dict) for layer in layers if layer["kind"] == "qsa_kv")):
            return "malformed cache or position state"
        if any(not any(layer.get("states") or []) for layer in layers if layer["kind"] == "arrays"):
            return "recurrent state is empty"
        if any(not {"keys", "values"} <= set(layer.get("kv") or {}) for layer in layers if layer["kind"] == "qsa_kv"):
            return "attention state is incomplete"
        if any(not all(array_evidence_valid(layer["kv"][key]) for key in ("keys", "values"))
               for layer in layers if layer["kind"] == "qsa_kv"):
            return "attention key or value evidence is malformed"
        arrays = [value for layer in layers for value in
                  (layer.get("states", []) if layer["kind"] == "arrays" else (layer.get("kv") or {}).values())
                  if value is not None]
        language = state["language"]
        if (set(language) != {"position_ids", "rope_deltas"} or
                not all(array_evidence_valid(value) for value in [*arrays, *language.values()])):
            return "state array or position metadata is missing or malformed"
        if p.get("physical_bytes") is None or p.get("peak_bytes") is None:
            return "physical I/O or peak-memory telemetry unavailable"
        if not array_evidence_valid(row["logits"].get("value")):
            return "final-row logits missing"
        if f["token_ids"][0] != p.get("first_token"):
            return "first generated token does not match prefill selection"
    a, b = left["prefilled"], right["prefilled"]
    if a["prefix"].get("execution_order"):
        return "reference did not run chunk-major"
    for key in ("prompt_sha256", "first_token", "state"):
        if a.get(key) != b.get(key):
            return f"{key} differs"
    if left["logits"] != right["logits"]:
        return "final-row logits differ"
    if a["prefix"]["executed_edges"] != b["prefix"]["executed_edges"]:
        return "chunk boundaries differ"
    candidate = b["prefix"]
    segments = candidate.get("execution_segments") or []
    edges = candidate["executed_edges"]
    if candidate.get("execution_order") != expected_order or not any(
            segment.get("execution") == "layer" and segment["start"] < edge < segment["end"]
            for segment in segments for edge in edges):
        return "candidate did not execute a multi-chunk layer-major window"
    if left["finished"]["token_ids"] != right["finished"]["token_ids"]:
        return "generated token IDs differ"
    return None


def validate_decode_pair(left, right, token_range=(2048, 4096)):
    """Exactness for the read-mode decode A/B: both arms chunk-major, identical prompt, state, logits and generated tokens."""
    for row in (left, right):
        if set(row) != {"logits", "prefilled", "finished"}:
            return "missing logits, prefill or completion probe"
        p, f = row["prefilled"], row["finished"]
        if not (token_range[0] <= (p.get("state") or {}).get("token_count", 0) <= token_range[1]):
            return "prompt is outside the token range of this run"
        if (p.get("prefix") or {}).get("execution_order"):
            return "an arm did not run chunk-major"
        if len(f.get("token_ids") or []) != 32 or f["token_ids"][0] != p.get("first_token"):
            return "generated tokens missing or inconsistent with the prefill token"
        if not array_evidence_valid(row["logits"].get("value")):
            return "final-row logits missing"
        if f.get("generation_seconds") is None or f.get("physical_bytes_total") is None or p.get("physical_bytes") is None:
            return "decode timing or physical-byte telemetry unavailable"
    a, b = left["prefilled"], right["prefilled"]
    for key in ("prompt_sha256", "first_token", "state"):
        if a.get(key) != b.get(key):
            return f"{key} differs"
    if left["logits"] != right["logits"]:
        return "final-row logits differ"
    if left["finished"]["token_ids"] != right["finished"]["token_ids"]:
        return "generated token IDs differ"
    return None


def summarize_decode(arms):
    """Raw decode numbers per condition from ``{arm_id: metrics}``: whole 32-token horizon and the 23-token tail."""
    import statistics
    out = {}
    for condition in ("chunk", "layer"):
        rows = [m for aid, m in sorted(arms.items()) if aid.endswith(condition)]
        if not rows:
            continue
        horizon = [31 / m["finished"]["generation_seconds"] for m in rows]
        tail = [m["finished"]["tail_tokens"] / m["finished"]["tail_seconds"] for m in rows
                if m["finished"].get("tail_seconds")]
        phys = [(m["finished"]["physical_bytes_total"] - m["prefilled"]["physical_bytes"]) / 31 / 1e9 for m in rows]
        prefill = [m["prefilled"]["prefill_seconds"] for m in rows]
        tokens = rows[0]["prefilled"]["state"]["token_count"]
        out[condition] = {"arms": len(rows), "prompt_tokens": tokens, "prefill_seconds": [round(x, 2) for x in prefill],
                          "prefill_tok_s": [round(tokens / x, 2) for x in prefill],
                          "median_prefill_tok_s": round(tokens / statistics.median(prefill), 2),
                          "mlx_peak_gb": [round(m["prefilled"]["peak_bytes"] / 1e9, 2) for m in rows],
                          "horizon_tok_s": [round(x, 3) for x in horizon],
                          "tail_tok_s": [round(x, 3) for x in tail], "median_horizon_tok_s": round(statistics.median(horizon), 3),
                          "median_tail_tok_s": round(statistics.median(tail), 3) if tail else None,
                          "spread_pct": round(100 * (max(horizon) - min(horizon)) / statistics.median(horizon), 1),
                          "decode_physical_gb_per_token": [round(x, 3) for x in phys]}
    if len(out) == 2:
        out["candidate_over_reference_median_horizon"] = round(
            out["layer"]["median_horizon_tok_s"] / out["chunk"]["median_horizon_tok_s"], 3)
        out["candidate_over_reference_median_prefill"] = round(
            out["layer"]["median_prefill_tok_s"] / out["chunk"]["median_prefill_tok_s"], 3)
    return out


def make_prompt(checkpoint, source_tokens=600, skip_tokens=0):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=True)
    tool_schema = {"type": "function", "function": {"name": "read_file", "description": "Read a repository file",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
    # Keep the established 32-token control. The protocol refuses tool-enabled
    # requests below 64 tokens, so include the schema as prompt context here;
    # actual tool calls require a separate protocol/integration test.
    system = H.AGENT_PROMPT_FILE.read_text().strip() + "\nAvailable tool schema for context: " + json.dumps(tool_schema)
    # Real repository source, in a fixed order, cut at a token count (600 reproduces the original 2.7K prompt).
    files = ("models/flashnext/prefill.py", "models/flashnext/expert_cache.py", "models/flashnext/prefix_session.py",
             "models/flashnext/layer_major.py", "models/flashnext/store.py", "macqwen/server.py")
    source = "\n\n".join(f"# {name}\n" + (ROOT / name).read_text() for name in files)
    ids = tokenizer.encode(source, add_special_tokens=False)
    if len(ids) < skip_tokens + source_tokens:
        raise ValueError("the repository excerpt is shorter than the requested prompt")
    excerpt = tokenizer.decode(ids[skip_tokens:skip_tokens + source_tokens])   # skip_tokens > 0: a different user turn, same system prompt
    messages = [{"role": "system", "content": system}, {"role": "user", "content":
        "Review this repository excerpt for concrete correctness risks. Explain the relevant execution path and evidence in detail.\n\n" + excerpt}]
    return {"model": "flashnext", "messages": messages, "max_tokens": 32, "stream": False}


def http_json(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def port_number():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def watchdog_pid(log_dir):
    for path in log_dir.glob("server-launch-*.log"):
        match = re.search(r"watchdog\s+ON\s+\(separate process (\d+)", path.read_text(errors="replace"))
        if match:
            return int(match.group(1))
    return None


def server_pid(log_dir):
    path = log_dir / "supervisor.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return next(int(row["pid"]) for row in rows if row.get("event") == "server_started")
    except (OSError, ValueError, KeyError, StopIteration):
        return None


def parent_pid(pid):
    try:
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5, check=True).stdout
        return int(out.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def probe_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 3 or [r.get("event") for r in rows] != ["logits", "prefilled", "finished"]:
        raise ValueError("probe did not produce exactly one logits, prefill and completion record")
    return {row["event"]: row for row in rows}


def checkpoint_fingerprint(checkpoint):
    """Identity metadata for every checkpoint file, plus hashes of small manifests."""
    files = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            stat = path.stat()
            info = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
            if path.suffix == ".json":
                info["sha256"] = source_hash(path)
            files[str(path.relative_to(checkpoint))] = info
    if not files:
        raise ValueError("checkpoint has no files")
    return files


def runtime_fingerprint():
    digest = hashlib.sha256()
    for directory in (ROOT / "macqwen", ROOT / "models/flashnext"):
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
            digest.update(bytes.fromhex(source_hash(path)))
    for name in ("chat.sh", "serve.sh"):
        digest.update(name.encode() + b"\0" + bytes.fromhex(source_hash(ROOT / name)))
    return digest.hexdigest()


def run_arm(name, index, checkpoint, payload, result_dir, sampler=machine_sample, candidate="layer-major-4k", uncached=True, memory_rules="strict", prefetch=False, decode_ab=False, pool_gb="0", pool_ab=False, base_pool_gb="0", reads="all", power_rules="ac", extra_env=None, candidate_enable=None, prefix_dir=None, driver=None, auto_purge=False):
    from models.flashnext.progress import process_identity

    probe = result_dir / f"{index}-{name}-probe.jsonl"
    log_dir = result_dir / f"{index}-{name}-supervisor"
    log_dir.mkdir()
    sample_path = result_dir / f"{index}-{name}-safety.jsonl"
    baseline = None
    def sample(stage):
        row = sampler()
        with sample_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "reading": row, "reason": safety_reason(row, baseline, memory_rules, power_rules)}) + "\n")
            handle.flush()
        return row
    port = port_number()
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="veystrix-layer-gate-") as temporary:
        preferences = Path(temporary) / "preferences.json"
        preferences.write_text(json.dumps({"temperature": 0.0, "thinking_enabled": False}))
        command = [str(ROOT / "serve.sh"), "--port", str(port), "--pool-gb", pool_gb if (name == "layer" and pool_ab) else base_pool_gb, "--checkpoint", str(checkpoint),
                   "--prefix-dir", str(prefix_dir if prefix_dir is not None else Path(temporary) / "prefix"),
                   "--preferences-file", str(preferences)] + ([] if auto_purge else ["--no-auto-purge"])
        # Both arms read experts with F_NOCACHE (same bytes, same order, so exactness is unaffected): three owner/agent runs
        # stopped on compressor growth while the file cache filled with streamed experts. Symmetric, recorded, opt-in.
        if decode_ab:
            # Read-mode decode A/B: both arms are the chunk-major reference; only when F_NOCACHE applies differs.
            command += ["--enable", "uncached-prefill-reads" if (name == "layer" or pool_ab) else "uncached-reads"]
        elif uncached:
            command += ["--enable", "uncached-prefill-reads" if reads == "prefill" else "uncached-reads"]
        if name == "layer" and decode_ab and candidate_enable:
            command += ["--enable", candidate_enable]      # decode A/B: the candidate arm adds one opt-in, the reference arm none
        if name == "layer" and not decode_ab:
            command += ["--enable", candidate]
            if prefetch:
                command += ["--enable", "layer-prefetch"]
        # The state probe records exactly one request and refuses a second, so a multi-request driver runs without it.
        env = dict(os.environ, VEYSTRIX_SUPERVISOR_LOG_DIR=str(log_dir),
                   VEYSTRIX_WARM_DIR=str(Path(temporary) / "warm"), VEYSTRIX_API_KEY="local",
                   FLASHNEXT_SWAP_GROWTH_STOP_GIB="2.5")
        # The launcher's mode owns the reference/candidate settings; inherited
        # diagnostics must not silently change the comparison.
        for key in list(env):
            if key.startswith("FLASHNEXT_") and key != "FLASHNEXT_SWAP_GROWTH_STOP_GIB":
                del env[key]
        # Purge (if short) before the server exists; arms themselves keep --no-auto-purge so no purge lands inside a run.
        # Diagnostic observer only (one JSON line per finished chunk: MLX active/peak, available, pressure); no behavior change.
        env["FLASHNEXT_CHUNK_LOG"] = str(result_dir / f"{index}-{name}-chunks.jsonl")
        if driver is None:
            env["VEYSTRIX_LAYER_GATE_PROBE"] = str(probe)
        # Diagnostic-only additions chosen by the caller (the decode-profile harness); nothing by default.
        env.update(extra_env or {})
        before, reason, purge = prepare_headroom(sampler, candidate=name == "layer", memory_rules=memory_rules, power_rules=power_rules)
        with sample_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "before", "reading": before, "reason": reason, "purge": purge}) + "\n")
        if reason:
            raise RuntimeError("preflight: " + reason)
        baseline = before
        # The server's own growth stop is the live swap guard. Under the owner's 3 GiB rule it is lowered so that it fires
        # before swap in use reaches the ceiling (never above the original 2.5 GiB).
        growth_gib = 2.5
        if memory_rules == "swap-only":
            growth_gib = round(min(2.5, (SWAP_CEILING_MB - before["swap_used_mb"]) / 1024 - 0.05), 2)
            if growth_gib < 0.5:
                raise RuntimeError(f"preflight: swap in use is already {before['swap_used_mb']:.0f} MiB, too close to the "
                                   f"{SWAP_CEILING_MB} MiB ceiling to run")
        env["FLASHNEXT_SWAP_GROWTH_STOP_GIB"] = f"{growth_gib:g}"
        start = time.monotonic()
        proc = None
        owned_child = owned_identity = None
        with (log_dir / "launcher.log").open("w") as log:
            try:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                ready = None
                while time.monotonic() - start < 180:
                    if proc.poll() is not None:
                        raise RuntimeError(f"supervisor exited during startup: {proc.returncode}")
                    reason = safety_reason(sample("startup"), baseline, memory_rules, power_rules)
                    if reason:
                        raise RuntimeError("startup safety: " + reason)
                    try:
                        ready = http_json(url + "/health", timeout=2)
                    except (OSError, ValueError):
                        ready = None
                    pid = watchdog_pid(log_dir)
                    child_pid = server_pid(log_dir)
                    if child_pid and parent_pid(child_pid) == proc.pid:
                        owned_child, owned_identity = child_pid, process_identity(child_pid)
                    if ready and ready.get("ready") and alive(pid) and parent_pid(pid) == child_pid:
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError("server or independent watchdog did not become ready")
                policy = ready.get("memory_policy") or {}
                if (policy.get("threshold_gib") is None or abs(policy["threshold_gib"] - growth_gib) > 0.01
                        or policy.get("swap_baseline_mb") is None):
                    raise RuntimeError(f"live +{growth_gib:g} GiB swap guard could not be confirmed")
                result = {}
                def request():
                    try:
                        # driver (integration smoke): any sequence of requests against this one supervised server.
                        result["response"] = driver(url) if driver is not None else http_json(url + "/v1/chat/completions", payload, timeout=900)
                    except urllib.error.HTTPError as exc:
                        result["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
                    except Exception as exc:
                        result["error"] = repr(exc)
                worker = threading.Thread(target=request, daemon=True)
                worker.start()
                while worker.is_alive():
                    worker.join(5)
                    if proc.poll() is not None or not alive(pid) or parent_pid(pid) != child_pid:
                        raise RuntimeError("server or independent watchdog exited during request")
                    reading = sample("request")
                    reason = safety_reason(reading, baseline, memory_rules, power_rules)
                    if reason:
                        raise RuntimeError("request safety: " + reason)
                    if reading["swap_used_mb"] - before["swap_used_mb"] > growth_gib * 1024:
                        raise RuntimeError(f"swap growth exceeded +{growth_gib:g} GiB")
                    if time.monotonic() - start > 900:
                        raise RuntimeError("arm exceeded 900-second limit")
                if "error" in result:
                    raise RuntimeError("request failed: " + result["error"])
                rows = probe_rows(probe) if driver is None else result["response"]
                return command, rows, {"before": before, "after": sample("after"), "health": ready, "preflight_purge": purge,
                                       "watchdog_pid": pid, "response_usage": result["response"].get("usage") if driver is None else None}
            finally:
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        # Only the supervisor's own process group; its server has a
                        # separate group and normally exits via supervisor cleanup.
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                # If the supervisor itself crashed, clean up only its verified
                # child process group, never a recycled or unrelated PID.
                if owned_child and owned_identity and process_identity(owned_child) == owned_identity:
                    try:
                        os.killpg(owned_child, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 10
                    while process_identity(owned_child) == owned_identity and time.monotonic() < deadline:
                        time.sleep(0.25)
                    if process_identity(owned_child) == owned_identity:
                        try:
                            os.killpg(owned_child, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


def summarize_arms(arms):
    """Raw per-arm numbers and spread bands from recorded evidence: ``{arm_id: metrics}``. No arm is dropped."""
    import statistics
    out = {}
    for condition in ("chunk", "layer"):
        rows = [m for aid, m in sorted(arms.items()) if aid.endswith(condition)]
        if not rows:
            continue
        seconds = [m["prefilled"]["prefill_seconds"] for m in rows]
        tokens = rows[0]["prefilled"]["state"]["token_count"]
        median = statistics.median(seconds)
        out[condition] = {
            "arms": len(rows), "tokens": tokens, "prefill_seconds": [round(x, 2) for x in seconds],
            "tok_s": [round(tokens / x, 2) for x in seconds], "median_tok_s": round(tokens / median, 2),
            "spread_pct": round(100 * (max(seconds) - min(seconds)) / median, 1),
            "physical_gb": [round(m["prefilled"]["physical_bytes"] / 1e9, 1) for m in rows],
            "mlx_peak_gb": [round(m["prefilled"]["peak_bytes"] / 1e9, 2) for m in rows]}
    if len(out) == 2:
        out["layer_over_chunk_median_tok_s"] = round(out["layer"]["median_tok_s"] / out["chunk"]["median_tok_s"], 3)
    return out


ORDERS = {"layer-major": "layer-major-v1/w8192", "layer-major-4k": "layer-major-v1/w4096",
          "layer-major-hybrid-4k": "layer-major-v1/w4096/hybrid-small-v1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--settle", type=int, default=30)
    parser.add_argument("--buffered-reads", action="store_true",
                        help="use buffered expert reads in both arms (the old gate configuration; filled the file cache and stopped on memory)")
    parser.add_argument("--memory-rules", choices=("swap-only", "strict"), default="swap-only",
                        help="swap-only (default, owner rule 2026-11-09): swap in use must stay below 3 GiB, critical pressure stops; "
                             "strict: also pressure level 1, 3 GiB available and compressor-growth stops (the original gate)")
    parser.add_argument("--decode-ab", action="store_true",
                        help="decode read-mode A/B: reference = F_NOCACHE in all phases, candidate = F_NOCACHE during prefill only; "
                             "both chunk-major; exactness enforced, decode rate and decode physical bytes compared")
    parser.add_argument("--pool-gb", default="0",
                        help="expert pool size in GB for BOTH arms (default 0; the production default is a pool); "
                             "--candidate-pool-gb overrides the candidate arm only")
    parser.add_argument("--reads", choices=("all", "prefill"), default="all",
                        help="F_NOCACHE in all phases (the original gate) or prefill only (decode keeps the page cache)")
    parser.add_argument("--candidate-pool-gb", default=None,
                        help="with --decode-ab: both arms read prefill-only uncached; the candidate arm also runs this expert pool "
                             "size in GB (reference pool 0), so the pair measures the pool's decode effect")
    parser.add_argument("--candidate-enable", default=None,
                        help="with --decode-ab: the candidate arm also runs --enable NAME (e.g. pool-buffered-fill); the reference arm does not")
    parser.add_argument("--prefetch", action="store_true",
                        help="candidate arm also runs --enable layer-prefetch (reads overlap compute; same bytes, same results)")
    parser.add_argument("--power-rules", choices=("battery-ok", "ac"), default="battery-ok",
                        help="battery-ok (default, owner rule 2026-11-09): run on battery until the charge reaches 25%; ac: AC power required. "
                             "Runs that started or ran on battery are preliminary per docs/measurement-standard.md")
    parser.add_argument("--candidate", choices=tuple(ORDERS), default="layer-major-4k")
    parser.add_argument("--source-tokens", type=int, default=600,
                        help="repository tokens in the prompt: 600 gives the 2.7K exactness gate, about 6000 an 8K paired prefill comparison")
    args = parser.parse_args()
    if not 0 <= float(args.pool_gb) <= 2:
        parser.error("--pool-gb must be between 0 and 2")
    if args.candidate_pool_gb and not args.decode_ab:
        parser.error("--candidate-pool-gb needs --decode-ab")
    if args.candidate_enable and not args.decode_ab:
        parser.error("--candidate-enable needs --decode-ab")
    if args.candidate_pool_gb and not 0 < float(args.candidate_pool_gb) <= 2:
        parser.error("--candidate-pool-gb must be above 0 and at most 2")
    if args.rounds < 3 or args.settle < 30:
        parser.error("at least three reversed pairs and 30 seconds between arms are required")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if shutil.disk_usage(ROOT).free < 50 * 10**9:
        raise SystemExit("at least 50 GB of free storage is required")
    initial, postponed, initial_purge = prepare_headroom(candidate=True, memory_rules=args.memory_rules, power_rules=args.power_rules)   # the larger arm decides
    if postponed:
        # Before any result directory, tokenizer load or process exists: a postponed gate leaves no record that could
        # be mistaken for a run.
        print(f"Gate postponed (nothing started): {postponed}", flush=True)
        return 2
    payload = make_prompt(checkpoint, args.source_tokens)
    token_range = (2048, 4096) if args.source_tokens <= 600 else (6144, 12288) if args.source_tokens <= 9000 else (12288, 20480)
    path = output_path("flashnext", "layer-gate", "layer-gate.jsonl")
    git = lambda *cmd: subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    checkpoint_id = checkpoint_fingerprint(checkpoint)
    source_id = runtime_fingerprint()
    metadata = {"checkpoint": str(checkpoint), "checkpoint_files": checkpoint_id, "commit": git("rev-parse", "HEAD").strip(),
                "worktree": git("status", "--short"), "sources": {p: source_hash(ROOT / p) for p in SOURCES},
                "runtime_fingerprint": source_id,
                "prompt_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                "rounds": args.rounds, "candidate": args.candidate, "sampler": "greedy", "memory_rules": args.memory_rules, "layer_prefetch": args.prefetch, "decode_ab": args.decode_ab, "candidate_pool_gb": args.candidate_pool_gb, "candidate_enable": args.candidate_enable, "pool_gb_both_arms": args.pool_gb, "power_rules": args.power_rules, "reads": args.reads, "expert_reads": ("reference F_NOCACHE all phases vs candidate F_NOCACHE prefill only" if args.decode_ab else "buffered" if args.buffered_reads else "F_NOCACHE (both arms)"),
                "cache": "isolated prefix cache per arm; OS file cache uncontrolled",
                "purpose": ("exactness diagnostic, not cold-throughput promotion" if args.source_tokens <= 600 else "paired prefill comparison with the exactness gate enforced on every pair; probe overhead and uncontrolled file cache apply"), "source_tokens": args.source_tokens, "initial_machine": initial, "initial_purge": initial_purge,
                "headroom_budget_mb": {"floor": FLOOR_MB, "observed_footprint": OBSERVED_FOOTPRINT_MB,
                                       "margin": HEADROOM_MARGIN_MB, "candidate_extra_unmeasured": CANDIDATE_EXTRA_MB,
                                       "required_at_start": REQUIRED_START_MB,
                                       "required_at_start_candidate": REQUIRED_CANDIDATE_START_MB}}
    run = MeasurementRun(path, runtime="flashnext", experiment="layer-gate", metadata=metadata)
    run.start()
    all_arms = {}
    try:
        for number in range(args.rounds):
            rows = {}
            for index, name in enumerate(("chunk", "layer") if number % 2 == 0 else ("layer", "chunk")):
                if checkpoint_fingerprint(checkpoint) != checkpoint_id or runtime_fingerprint() != source_id:
                    raise RuntimeError("checkpoint or source changed during the comparison")
                arm_id = f"r{number + 1}-{name}"
                print(f"Starting {arm_id}", flush=True)
                try:
                    command, evidence, machine = run_arm(name, arm_id, checkpoint, payload, path.parent, candidate=args.candidate,
                                                      uncached=not args.buffered_reads, memory_rules=args.memory_rules, prefetch=args.prefetch, decode_ab=args.decode_ab,
                                                      pool_gb=args.candidate_pool_gb or "0", pool_ab=bool(args.candidate_pool_gb), candidate_enable=args.candidate_enable,
                                                      base_pool_gb=args.pool_gb, reads=args.reads, power_rules=args.power_rules)
                except Exception as error:
                    run.arm(arm_id=arm_id, condition=name, round_index=number + 1, command=[], status="failed",
                            metrics={"error": str(error), "probe_file": f"{arm_id}-{name}-probe.jsonl",
                                     "supervisor_directory": f"{arm_id}-{name}-supervisor"})
                    raise
                run.arm(arm_id=arm_id, condition=name, round_index=number + 1, command=command,
                        metrics=evidence, **machine)
                rows[name] = evidence
                all_arms[arm_id] = evidence
                if index == 0:
                    time.sleep(args.settle)
            reason = (validate_decode_pair(rows["chunk"], rows["layer"], token_range) if args.decode_ab else
                      validate_pair(rows["chunk"], rows["layer"], expected_order=ORDERS[args.candidate], token_range=token_range))
            run.validation(f"pair-{number + 1}", "fail" if reason else "pass", reason=reason)
            if reason:
                raise RuntimeError("exactness gate: " + reason)
            if number + 1 < args.rounds:
                time.sleep(args.settle)
        summary = (summarize_decode(all_arms) if args.decode_ab else summarize_arms(all_arms))
        print(json.dumps(summary, indent=2), flush=True)
        run.finish("completed", pairs=args.rounds, exactness="passed", prefill_summary=summary)
        return 0
    except Exception as exc:
        run.failure(str(exc), error_type="incomplete_or_failed_gate")
        run.finish("stopped", reason=str(exc))
        print(f"Gate stopped: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
