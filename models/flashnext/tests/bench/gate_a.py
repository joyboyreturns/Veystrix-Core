#!/usr/bin/env python3
"""Gate A: ONE real 4K DeepSeek Harness coding task (glob, grep, final answer, follow-up) on the real checkpoint.

    .venv/bin/python -m models.flashnext.tests.bench.gate_a            # real checkpoint, one attempt
    .venv/bin/python -m models.flashnext.tests.bench.gate_a --stub     # offline mechanics test, never the checkpoint
    .venv/bin/python -m models.flashnext.tests.bench.gate_a --gate-only
    .venv/bin/python -m models.flashnext.tests.bench.gate_a --analyze <evidence dir>

One launch, one task, one follow-up. No retry inside this script: a retry is a new invocation after a diagnosed fix.
The product's own admission, the ChatGuard and the out-of-process watchdog are the safety system; this script only
adds pre-checks (it never lowers a product threshold) and writes the evidence. Battery runs are explicit: below
BATTERY_FLOOR the gate refuses, and the run aborts (clean shutdown) at BATTERY_ABORT, above the guard's own 25% stop.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DSH = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
CHECKPOINT = Path.home() / "models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
PATCH = REPO / "docs/flashnext/dsh/veystrix-agentic.patch.yml"
PORT = 8080
FIXTURES = REPO / "models/flashnext/tests/bench/fixtures"
# One orchestrator, two sizes. "4k" is Gate A (attempt history in docs); "8k" is the authorized direct 8K AGENTIC test.
PROFILES = {
    "4k": {"label": "gate-a", "size": "4K", "task": FIXTURES / "gate_a_task_4k.txt", "patch": PATCH, "context": 8192,
           "band": (3500, 4800), "ok_tiers": ("long", "8k", "standard"), "decoy": True, "strict_order": True},
    "8k": {"label": "gate-8k", "size": "8K", "task": FIXTURES / "gate_8k_task.txt",
           "patch": REPO / "docs/flashnext/dsh/veystrix-agentic-8k.patch.yml", "context": 12288,
           "band": (7600, 9000), "ok_tiers": ("long", "8k"), "decoy": False, "strict_order": False},
}
PROFILE = dict(PROFILES["4k"])


def use_profile(name: str) -> dict:
    PROFILE.clear()
    PROFILE.update(PROFILES[name], name=name)
    return PROFILE


def server_args() -> list:
    return ["--mode", "agentic", "--pool-gb", "0", "--auto-purge", "--enable", "long-context",
            "--context-limit", str(PROFILE["context"])]


SERVER_ARGS = server_args()          # the 4K arguments, kept for the existing tests
FIRST_REQUEST_BAND = PROFILES["4k"]["band"]
# Battery. Measured drain under real CHAT load: 3% in 12.6 min (0.24%/min). The wall cap is 35 min (about 8%), the
# product guard stops at 25%. FLOOR = guard 25 + 8 measured + margin; ABORT keeps a clean-shutdown margin above 25.
BATTERY_FLOOR = 35
BATTERY_ABORT = 28
READY_CAP, TASK_CAP, FOLLOW_CAP, EXIT_WAIT = 600, 900, 600, 90
HEAVY = re.compile(r"(Safari|Google Chrome|Chromium|Firefox|Arc\b|Brave|Xcode|Docker|Slack|zoom\.us|Teams|Photoshop|"
                   r"Final Cut|Premiere|Microsoft (Word|Excel|PowerPoint)|Simulator|Android Studio|IntelliJ|PyCharm|"
                   r"Code Helper|Figma|Notion|Spotify|Discord)", re.I)


def sh(*cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as error:
        return f"<{type(error).__name__}: {error}>"


def vm_stat():
    out = sh("vm_stat")
    page = int((re.search(r"page size of (\d+) bytes", out) or [0, 16384])[1])
    values = {m[1].strip(): int(m[2]) for m in (re.match(r"(.+?):\s+(\d+)\.", l) for l in out.splitlines()) if m}
    gib = lambda key: round(values.get(key, 0) * page / 2 ** 30, 3)
    return {"free_gib": gib("Pages free"), "inactive_gib": gib("Pages inactive"), "wired_gib": gib("Pages wired down"),
            "compressor_gib": gib("Pages occupied by compressor"), "swapouts": values.get("Swapouts", 0)}


def swap_mb():
    m = re.search(r"used = ([\d.]+)M", sh("sysctl", "vm.swapusage"))
    return float(m[1]) if m else None


def pressure():
    out = sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()
    return int(out) if out.isdigit() else None


def power():
    batt, full, therm = sh("pmset", "-g", "batt"), sh("pmset", "-g"), sh("pmset", "-g", "therm")
    pct = re.search(r"(\d+)%", batt)
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    return {"source": "ac" if "AC Power" in batt else "battery", "percent": int(pct[1]) if pct else None,
            "discharging": "discharging" in batt, "low_power_mode": bool(re.search(r"lowpowermode\s+1", full)),
            "thermal_warning": "No thermal warning level has been recorded" not in therm,
            "cpu_speed_limit": int(limit[1]) if limit else None}


TASK_FILE = PROFILES["4k"]["task"]


def load_task() -> str:
    """The precomputed 4K task. This process must stay small: its memory counts as 'other anonymous memory' in the
    product's admission arithmetic (the first real attempt was admitted only at the minimal tier because this script
    had loaded a tokenizer, 0.52 GiB). So no tokenizer, no model library, no fixture build happens here."""
    return Path(PROFILE["task"]).read_text()


def admission_precheck(reading=None, total=None, processes=()):
    """Dry run of the product's own tiered admission (no checkpoint, no thresholds changed): would a launch be admitted at one
    of the tiers this profile accepts? The measured gate (``prefill_readiness.required_gate_bytes``) is compared with the
    available memory PLUS what a purge can be counted on to return (the smallest gain on record, never more than the file
    cache that is really there), because the product's memory preparation purges before it decides. Returns
    ``(ok, detail, values)``."""
    from models.flashnext import admission_model as M, prefill_readiness as R, purge_recovery as P

    reading = reading if reading is not None else P.read_system()
    saved = {k: os.environ.get(k) for k in ("FLASHNEXT_PREFILL_CHUNK_POLICY", "VEYSTRIX_CONTEXT_REQUIRED",
                                            "VEYSTRIX_CONTEXT_LIMIT")}
    os.environ["FLASHNEXT_PREFILL_CHUNK_POLICY"] = "budgeted"        # what --enable long-context sets for the server
    os.environ["VEYSTRIX_CONTEXT_REQUIRED"] = "1"                    # an explicit --context-limit is a requirement
    os.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(PROFILE["context"])
    try:
        tiers = R.active_tiers()
        counted = M.expected_purge_gain_bytes(P.accounting(reading)["purge_gain_upper"])
        effective = reading.available + counted
        rows, admitted = [], None
        for tier in tiers:
            required = R.required_gate_bytes(reading, tier.max_cold_prefill)
            ok_tier = effective >= required and reading.pressure <= R.PRESSURE_OK_LEVEL
            rows.append({"tier": tier.name, "required_gib": round(required / R.GIB, 2), "ok": ok_tier,
                         "effective_gib": round(effective / R.GIB, 2)})
            if ok_tier and admitted is None:
                admitted = tier.name
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    anon = round(reading.anonymous * reading.page / R.GIB, 2)
    ok = admitted in PROFILE["ok_tiers"]
    detail = (f"would be admitted at tier {admitted}" if admitted else "no tier admitted") + \
        f"; available {reading.available / R.GIB:.2f} GiB + {counted / R.GIB:.2f} counted from a purge; other anonymous {anon} GiB; " + \
        ", ".join(f"{r['tier']} needs {r['required_gib']} GiB" for r in rows) + \
        f"; the {PROFILE['size']} task needs one of {PROFILE['ok_tiers']} (context {PROFILE['context']})"
    return ok, detail, {"tiers": rows, "admitted": admitted, "anonymous_gib": anon}


WAIVER = {"on": False}


def gate():
    checks = []

    def check(name, ok, detail, hard=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "hard": hard})

    p = power()
    if p["source"] == "ac":
        check("power", True, f"AC power ({p['percent']}%)")
    else:
        check(f"battery >= {BATTERY_FLOOR}% (run is on BATTERY, explicitly authorized; figures are battery-mode)",
              p["percent"] is not None and p["percent"] >= BATTERY_FLOOR, f"{p['percent']}% discharging={p['discharging']}")
    check("Low Power Mode off", not p["low_power_mode"], f"low_power_mode={p['low_power_mode']}")
    check("no thermal warning, no CPU speed limit", not p["thermal_warning"] and p["cpu_speed_limit"] in (None, 100), str(p))
    check("memory pressure level 1", pressure() == 1, f"level={pressure()}")
    ok_admit, detail, _ = admission_precheck()
    if WAIVER["on"]:
        check("dry admission (ADVISORY: the owner waived admission, FLASHNEXT_MEMORY_WAIVER=swap-only)", True,
              ("would pass" if ok_admit else "WOULD BE REFUSED") + ": " + detail, hard=False)
    else:
        check(f"dry admission (product rule, unchanged): one of {PROFILE['ok_tiers']}", ok_admit, detail)
    before, swap0 = vm_stat(), swap_mb()
    time.sleep(8)
    after, swap1 = vm_stat(), swap_mb()
    check("no swap-out in the last 8 s", after["swapouts"] == before["swapouts"] and (swap1 or 0) <= (swap0 or 0) + 1,
          f"swapouts {before['swapouts']}->{after['swapouts']}, swap {swap0}->{swap1} MB")
    heavy = []
    for line in sh("ps", "-axo", "rss=,comm=").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and HEAVY.search(parts[1]) and int(parts[0]) / 1024 >= 200:
            heavy.append(f"{os.path.basename(parts[1])} {int(parts[0]) / 1024:.0f} MB")
    check("heavy applications (advisory: the product's own admission decides; nothing is closed)", True,
          ", ".join(heavy) or "none", hard=False)
    stray = [l.strip()[:100] for l in sh("ps", "-axo", "pid=,command=").splitlines()
             if re.search(r"(veystrix/cli\.py|models\.flashnext\.watchdog|unittest)", l) and "gate_a" not in l and "grep" not in l]
    check("no stray Veystrix/test processes", not stray, "; ".join(stray) or "none")
    check(f"port {PORT} free", not sh("lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN").strip(), f"port {PORT}")
    check("free SSD >= 60 GB", shutil.disk_usage(Path.home()).free / 1e9 >= 60, f"{shutil.disk_usage(Path.home()).free / 1e9:.0f} GB")
    check("checkpoint present", CHECKPOINT.is_dir(), str(CHECKPOINT))
    check("harness present", Path(DSH).exists(), DSH)
    check("AGENTIC patch present", Path(PROFILE["patch"]).exists(), str(PROFILE["patch"]))
    check("precomputed task present", Path(PROFILE["task"]).exists(), str(PROFILE["task"]))
    return all(c["ok"] for c in checks if c["hard"]), checks


def http(path, key, timeout=5):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read())
        except Exception:
            return error.code, None
    except Exception as error:
        return None, str(error)


def listener_pid():
    out = sh("lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t").split()
    return int(out[0]) if out else None


class Sampler(threading.Thread):
    """Every 3 s: host memory, swap, pressure, server RSS, battery, power mode, thermal. Appends to samples.jsonl."""

    def __init__(self, path, key, stop):
        super().__init__(daemon=True)
        self.path, self.key, self.stop_event = path, key, stop
        self.last_power = {}
        self.lowest_battery = None

    def run(self):
        tick = 0
        with open(self.path, "a") as out:
            while not self.stop_event.is_set():
                row = {"t": time.time(), **vm_stat(), "swap_mb": swap_mb(), "pressure": pressure()}
                if tick % 2 == 0:
                    self.last_power = power()
                row["power"] = self.last_power
                if self.last_power.get("percent") is not None:
                    self.lowest_battery = min(self.lowest_battery or 100, self.last_power["percent"])
                pid = listener_pid()
                if pid:
                    rss = sh("ps", "-o", "rss=", "-p", str(pid)).strip()
                    row["server_rss_gib"] = round(int(rss) / 1048576, 2) if rss.isdigit() else None
                out.write(json.dumps(row) + "\n")
                out.flush()
                tick += 1
                self.stop_event.wait(3)


class StackProbe(threading.Thread):
    """Native stack samples of the server while a request is long (macOS ``sample``, 2 s each): the first after 360 s, then
    every 240 s. Small, read-only, and the only way to see where a silent prefill is (the 8K gate's two no-disk-read gaps
    left nothing to read). Writes ``stack-<n>.txt`` next to the evidence."""

    def __init__(self, pid, out, stop):
        super().__init__(daemon=True)
        self.pid, self.out, self.stop_event = pid, out, stop

    def run(self):
        began, number = time.time(), 0
        due = 360
        while not self.stop_event.wait(5):
            if time.time() - began >= due and self.pid:
                number += 1
                due += 240
                subprocess.run(["sample", str(self.pid), "2", "-file", str(self.out / f"stack-{number}.txt")],
                               capture_output=True, timeout=60)


def run_harness(env, work, patch, text, out_file, err_file, cap, session=None, stop_check=None):
    cmd = [DSH, "headless", "--patch", str(patch)] + (["--session-id", session] if session else []) + ["--json", text]
    began = time.time()
    proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=open(out_file, "wb"), stderr=open(err_file, "wb"),
                            start_new_session=True)
    ended_by = None
    while proc.poll() is None:
        if time.time() - began > cap:
            ended_by = "wall cap"
        elif stop_check and stop_check():
            ended_by = stop_check.reason
        if ended_by:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(3)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(1)
    return {"exit": proc.poll(), "seconds": round(time.time() - began, 1), "ended_by": ended_by}


def events(path):
    rows = []
    try:
        for line in open(path, errors="replace"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return rows


def _cache_state(reading) -> dict:
    """The ``cache`` block of a /health reading; {} when the reading is an error string (server already gone)."""
    cache = reading.get("cache") if isinstance(reading, dict) else None
    return cache if isinstance(cache, dict) else {}


def _glob_then_grep(calls, results, strict: bool) -> bool:
    """Real model-generated calls, executed by the harness. ``strict`` (Gate A): the first two calls are glob, grep.
    Otherwise (8K gate): a glob call comes before the first grep call and both completed, whatever else the model also
    called (it may issue several globs in one step)."""
    done = {r.get("callId"): r.get("status") == "completed" for r in results}
    if strict:
        return [c.get("tool") for c in calls[:2]] == ["glob", "grep"] and all(r.get("status") == "completed" for r in results[:2])
    tools = [c.get("tool") for c in calls]
    if "glob" not in tools or "grep" not in tools:
        return False
    first_glob, first_grep = tools.index("glob"), tools.index("grep")
    return first_glob < first_grep and done.get(calls[first_glob].get("callId")) is True \
        and done.get(calls[first_grep].get("callId")) is True


def phase_churn(mem) -> dict:
    """Compress/decompress/page-in volume (GiB) per phase from the 0.25 s trace: the churn that RSS and 'available' hide."""
    out, page = {}, 16384 / 2 ** 30
    segments = []
    for row in mem:
        if segments and segments[-1][0] == row.get("phase"):
            segments[-1][1].append(row)
        else:
            segments.append((row.get("phase"), [row]))
    for phase, rows in segments:
        a, b = rows[0], rows[-1]
        if b["t"] - a["t"] < 1 or any(k not in a or a[k] is None or b.get(k) is None for k in ("compress", "decompress", "pagein")):
            continue
        item = out.setdefault(phase, {"seconds": 0.0, "compress_gib": 0.0, "decompress_gib": 0.0, "pagein_gib": 0.0,
                                      "min_available_gib": 1e9, "max_compressor_gib": 0.0})
        item["seconds"] += b["t"] - a["t"]
        for key, name in (("compress", "compress_gib"), ("decompress", "decompress_gib"), ("pagein", "pagein_gib")):
            item[name] += (b[key] - a[key]) * page
        item["min_available_gib"] = min(item["min_available_gib"], min(r["available_mb"] for r in rows) / 1024)
        item["max_compressor_gib"] = max(item["max_compressor_gib"], max(r["compressor_mb"] for r in rows) / 1024)
    return {k: {n: round(v, 2) for n, v in item.items()} for k, item in out.items()}


def analyze(out: Path) -> dict:
    """Judge a finished attempt from its evidence. Every criterion is mechanical and listed with its measured value."""
    from models.flashnext.tests.bench.fixtures import agentic_fixture as F

    s = json.loads((out / "summary.json").read_text())
    task, follow = events(out / "task.jsonl"), events(out / "follow.jsonl")
    calls = [e for e in task if e.get("type") == "tool_call"]
    results = [e for e in task if e.get("type") == "tool_result"]
    final = next((e for e in reversed(task) if e.get("type") == "final"), {})
    answer = final.get("text") or ""
    follow_final = next((e for e in reversed(follow) if e.get("type") == "final"), {})
    follow_text = follow_final.get("text") or ""
    requests = [json.loads(l) for l in open(out / "requests.jsonl")] if (out / "requests.jsonl").exists() else []
    health = _cache_state(s.get("health_after_follow"))
    verdict = F.check_answer(answer)
    trace = [json.loads(l) for l in open(out / "guard-trace.jsonl") if l.strip().startswith("{")] \
        if (out / "guard-trace.jsonl").exists() else []
    mem = [r for r in trace if "mlx_peak" in r]
    first_tokens = (requests[0].get("prompt_tokens") if requests else None)
    criteria = {
        "server ready": bool(s.get("stage", {}).get("ready", {}).get("ok")),
        f"first request about {PROFILE['size']} real tokens":
            first_tokens is not None and PROFILE["band"][0] <= first_tokens <= PROFILE["band"][1],
        "harness task exited 0": (s.get("task") or {}).get("exit") == 0,
        "glob then grep generated and executed": _glob_then_grep(calls, results, PROFILE["strict_order"]),
        "tool arguments valid (non-empty objects)": bool(calls) and all(isinstance(c.get("input"), dict) and c["input"] for c in calls[:2]),
        "final answer has batch size and token expiry, no decoy claimed":
            verdict["batch_size"] and verdict["token_expiry"] and not verdict["decoy_claimed"],
        "follow-up exited 0 with an answer": (s.get("follow_up") or {}).get("exit") == 0 and bool(follow_text.strip()),
        "follow-up uses retained context": bool(re.search(r"expir|batch|5400|1536", follow_text, re.I)),
        "conversation state preserved (rebuilt once, reused >= 3, no eviction)":
            health.get("rebuilt") == 1 and (health.get("reused") or 0) >= 3 and not health.get("evictions"),
        "server shut down cleanly": bool((s.get("stage", {}).get("shutdown") or {}).get("ok")),
        "no safety guard stop": "safety guard stopped" not in (out / "server.log").read_text(errors="replace")
            if (out / "server.log").exists() else False,
        "no leftover processes": not s.get("leftover_processes"),
    }
    measured = {
        "first_request_tokens": first_tokens,
        "requests": [{k: r.get(k) for k in ("prompt_tokens", "prefill_seconds", "prefill_tokens_per_second", "ttft_seconds",
                                            "completion_tokens", "decode_seconds", "decode_tokens_per_second", "finish",
                                            "tool_calls", "wall_seconds")} for r in requests],
        "chunk_trace": (requests[0].get("prefix") or {}).get("chunk_trace") if requests else None,
        "plan": {k: (requests[0].get("prefix") or {}).get(k) for k in ("plan_edges", "executed_edges", "plan_deviated", "stepdowns",
                                                                       "hit_tokens", "snapshots", "restore_seconds")} if requests else None,
        "mlx_peak_gib": round(max((r["mlx_peak"] for r in mem), default=0) / 2 ** 30, 2),
        "mlx_active_max_gib": round(max((r["mlx_active"] for r in mem), default=0) / 2 ** 30, 2),
        "min_available_gib": round(min((r["available_mb"] for r in mem), default=0) / 1024, 2) if mem else None,
        "compressor_growth_max_mb": max((r.get("compressor_growth_mb") or 0 for r in mem), default=None),
        "phase_churn": phase_churn(mem),
        "memory_states": [{k: r.get(k) for k in ("t", "state", "previous", "reasons", "pressure", "available_mb")}
                          for r in trace if r.get("event") == "memory_state"],
        "memory_ledger": next((r.get("memory_ledger") for r in reversed(trace) if r.get("event") == "shutdown"), None),
        "turn_memory": [r.get("turn_memory") for r in requests],
        "swapout_pages_delta": (mem[-1]["swapout"] - mem[0]["swapout"]) if len(mem) > 1 and "swapout" in mem[0] else None,
        "max_pressure": max((r.get("pressure_level") or 1 for r in mem), default=None),
        "battery": s.get("battery"), "cache_state": health, "answer": answer, "follow_up_answer": follow_text,
        "tool_calls": [{"tool": c.get("tool"), "input": c.get("input")} for c in calls],
    }
    return {"passed": all(criteria.values()), "criteria": criteria, "measured": measured}


# ---- offline mechanics test ------------------------------------------------------------------------------------------


def serve_stub(port: int, key: str) -> None:
    """A scripted real-tokenizer server with the same HTTP surface. Never loads the checkpoint."""
    from transformers import AutoTokenizer
    from veystrix import preferences, server as S
    from veystrix.scheduler import Scheduler
    from veystrix.tests.test_agentic_long_fixtures import RealTokenizerBackend
    from veystrix.tests.test_server import FakeSession
    from models.flashnext.tests.bench.fixtures import agentic_fixture as F

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    replies = [F.tool_call_text(F.GLOB_CALL), F.tool_call_text(F.GREP_CALL), F.final_answer(),
               "Lower the token expiry first: it controls how often every client re-authenticates."]
    session = FakeSession([])
    session.backend = RealTokenizerBackend(tokenizer, replies)
    session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False)
    service = S.ModelService(session, context_limit=PROFILE["context"])
    server = S.VeystrixHTTPServer(("127.0.0.1", port), service, api_key=key, scheduler=Scheduler())
    signal.signal(signal.SIGINT, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f"server ready at http://127.0.0.1:{port}", flush=True)
    server.serve_forever(poll_interval=0.05)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--analyze", metavar="DIR")
    parser.add_argument("--serve-stub", nargs=2, metavar=("PORT", "KEY"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--waive-memory", action="store_true",
                        help="owner's waiver: launch despite an admission refusal; only +2.5 GiB swap growth (and the "
                             "non-memory protections) stops the run. See models/flashnext/memory_waiver.py")
    parser.add_argument("--size", choices=sorted(PROFILES), default="4k",
                        help="4k: Gate A; 8k: the authorized direct 8K AGENTIC test (long-context, context 12288)")
    parser.add_argument("--wait-minutes", type=float, default=0.0,
                        help="poll the dry admission (cheap, loads nothing) until it passes, up to this long")
    args = parser.parse_args()
    use_profile(args.size)
    WAIVER["on"] = bool(args.waive_memory)
    if args.serve_stub:
        serve_stub(int(args.serve_stub[0]), args.serve_stub[1])
        return 0
    if args.analyze:
        print(json.dumps(analyze(Path(args.analyze)), indent=2, default=str))
        return 0
    from models.flashnext.tests.bench import dsh_wire_probe as W      # stdlib only at import time
    from models.flashnext.tests.bench.fixtures import agentic_fixture as F    # pure Python

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = REPO / "results/flashnext" / f"{stamp}-{PROFILE['label']}-{'stub' if args.stub else 'attempt'}{args.attempt}"
    out.mkdir(parents=True)
    summary = {"started": stamp, "attempt": args.attempt, "stub": args.stub, "outcome": None,
               "profile": PROFILE["name"], "memory_waiver": WAIVER["on"], "server_args": server_args(), "battery_run": power()["source"] != "ac"}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"evidence directory: {out}")
    if args.wait_minutes > 0 and not args.stub:
        end, last = time.time() + args.wait_minutes * 60, None
        while time.time() < end:
            ok_now, detail, _ = admission_precheck()
            percent = power().get("percent")
            line = f"waiting for admission: battery {percent}%, {detail[:150]}"
            if line != last:
                print(time.strftime("%H:%M:%S ") + line, flush=True)
                last = line
            if ok_now or (percent is not None and percent < BATTERY_FLOOR and power()["source"] != "ac"):
                break
            time.sleep(20)
    ok, checks = gate()
    summary["gate"] = checks
    for item in checks:
        print(("PASS " if item["ok"] else ("FAIL " if item["hard"] else "NOTE ")) + f"{item['check']}: {item['detail']}")
    if args.gate_only or (not ok and not args.stub):
        summary["outcome"] = "GATE ONLY" if ok else "GATE FAILED: checkpoint not touched"
        save()
        print("\n" + summary["outcome"])
        return 0 if ok else 70
    # fixture workspace, isolated harness home, the reviewed patch
    meta = json.loads(Path(PROFILE["task"]).with_suffix(".json").read_text())
    fixture = {"tokens": meta.get("fixture_tokens", meta.get("expected_first_request"))}
    overhead = meta["harness_overhead_tokens"]
    task_text = load_task()
    work, home = out / "ws", out / "dsh_home"
    work.mkdir(); home.mkdir()
    W.make_project(work, decoy=PROFILE["decoy"])
    (out / "task.txt").write_text(task_text)
    key = secrets.token_urlsafe(24)
    summary["env"] = {"power": power(), "vm": vm_stat(), "swap_mb": swap_mb(), "pressure": pressure(),
                      "git_head": sh("git", "-C", str(REPO), "rev-parse", "HEAD").strip(),
                      "git_status_lines": len(sh("git", "-C", str(REPO), "status", "--short").splitlines()),
                      "fixture_tokens": fixture["tokens"], "harness_overhead_tokens": overhead,
                      "checkpoint": str(CHECKPOINT)}
    env = {**os.environ, "VEYSTRIX_SERVER_API_KEY": key, "VEYSTRIX_REQUEST_LOG": str(out / "requests.jsonl")}
    env["FLASHNEXT_CHUNK_LOG"] = str(out / "chunks.jsonl")        # one line per finished prefill chunk, written live
    if WAIVER["on"]:
        env["FLASHNEXT_MEMORY_WAIVER"] = "swap-only"
    cmd = ([sys.executable, "-m", "models.flashnext.tests.bench.gate_a", "--size", args.size, "--serve-stub", str(PORT), key] if args.stub
           else [str(REPO / "chat.sh"), *server_args()])
    summary["server_cmd"] = [c if c != key else "<key>" for c in cmd]
    launched = time.time()
    log = open(out / "server.log", "wb")
    server = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              stdin=subprocess.DEVNULL)
    stop = threading.Event()
    sampler = Sampler(out / "samples.jsonl", key, stop)
    sampler.start()
    summary["stage"] = {}

    class LowBattery:
        reason = None

        def __call__(self):
            p = sampler.last_power
            if p.get("source") == "battery" and p.get("percent") is not None and p["percent"] <= BATTERY_ABORT:
                self.reason = f"battery {p['percent']}% reached the abort level {BATTERY_ABORT}%"
            elif p.get("low_power_mode"):
                self.reason = "Low Power Mode turned on"
            elif p.get("thermal_warning"):
                self.reason = "thermal warning"
            elif server.poll() is not None:
                self.reason = "server exited"
            return self.reason is not None

    low = LowBattery()
    worker = None
    try:
        ready = False
        while time.time() - launched < READY_CAP:
            if server.poll() is not None:
                summary["outcome"] = f"server exited before ready, code {server.returncode} (69 = product admission refused)"
                break
            status, body = http("/ready", key, 3)
            if status == 200 and body.get("ready"):
                ready = True
                break
            time.sleep(1)
        summary["stage"]["ready"] = {"ok": ready, "seconds": round(time.time() - launched, 1)}
        if ready:
            worker = listener_pid()
            summary["worker_pid"] = worker
            summary["health_before"] = http("/health", key)[1]
            patch = out / "patch.yml"
            shutil.copy(PROFILE["patch"], patch)
            probe = StackProbe(worker, out, stop)
            probe.start()
            henv = {**os.environ, "DSH_HOME": str(home), "VEYSTRIX_API_KEY": key}
            summary["task"] = run_harness(henv, work, patch, task_text, out / "task.jsonl", out / "task.err", TASK_CAP, stop_check=low)
            summary["health_after_task"] = http("/health", key)[1]
            session_id = next((e.get("sessionId") for e in events(out / "task.jsonl") if e.get("type") == "session"), None)
            summary["session_id"] = session_id
            if summary["task"]["exit"] == 0 and session_id and server.poll() is None:
                summary["follow_up"] = run_harness(henv, work, patch, F.FOLLOW_UP, out / "follow.jsonl", out / "follow.err",
                                                   FOLLOW_CAP, session=session_id, stop_check=low)
            summary["health_after_follow"] = http("/health", key)[1]
        shutdown = {"ok": False}
        target = worker or listener_pid()
        if server.poll() is None:
            if target:
                os.kill(target, signal.SIGINT)
            began = time.time()
            while time.time() - began < EXIT_WAIT and server.poll() is None:
                time.sleep(0.5)
            shutdown["seconds"] = round(time.time() - began, 1)
            if server.poll() is None:
                shutdown["hung"] = True
                if target:
                    subprocess.run(["sample", str(target), "3", "-file", str(out / "exit-hang-sample.txt")],
                                   capture_output=True, timeout=30)
                    os.kill(target, signal.SIGTERM)
                    time.sleep(5)
                    if server.poll() is None:
                        os.kill(target, signal.SIGKILL)
            else:
                shutdown["ok"] = server.returncode in (0, None, -signal.SIGINT)
        shutdown["returncode"] = server.poll()
        summary["stage"]["shutdown"] = shutdown
    finally:
        stop.set()
        sampler.join(5)
        log.close()
        if server.poll() is None:
            server.terminate()
    summary["wall_seconds"] = round(time.time() - launched, 1)
    summary["battery"] = {"start": summary["env"]["power"].get("percent"), "end": power().get("percent"),
                          "lowest": sampler.lowest_battery, "source": summary["env"]["power"].get("source"),
                          "note": "performance figures were measured on battery" if summary["battery_run"] else "AC"}
    text = (out / "server.log").read_text(errors="replace")
    trace = re.search(r"FlashNext safety trace: (\S+)", text)
    if trace and Path(trace[1]).exists():
        shutil.copy(trace[1], out / "guard-trace.jsonl")
    wd = Path.home() / ".flashnext/watchdog"
    if wd.is_dir():
        for item in wd.glob("*"):
            if item.stat().st_mtime >= launched - 5:
                shutil.copy(item, out / f"watchdog-{item.name}")
    for name in ("prep-log.jsonl", "admission-log.jsonl", "purge-log.jsonl"):
        source = Path.home() / ".flashnext" / name
        if source.exists():
            keep = []
            for line in source.read_text().splitlines()[-60:]:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if float(row.get("wall", row.get("t", row.get("time", 0))) or 0) >= launched - 30:
                    keep.append(line)
            (out / name).write_text("\n".join(keep) + ("\n" if keep else ""))
    summary["leftover_processes"] = [l for l in sh("ps", "-axo", "pid=,command=").splitlines()
                                     if "models.flashnext.watchdog" in l or "veystrix/cli.py" in l or "--serve-stub" in l]
    summary["outcome"] = summary["outcome"] or "completed"
    save()
    result = analyze(out)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"criteria": result["criteria"], "passed": result["passed"], "wall_seconds": summary["wall_seconds"],
                      "battery": summary["battery"]}, indent=2))
    print(f"\nevidence: {out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
