#!/usr/bin/env python3
"""ONE guarded AGENTIC smoke run: gate -> launch -> one DeepSeek Harness task -> clean shutdown -> evidence.

Real mode (no flags):  python smoke_run.py
Stub mode (offline mechanics test only, never loads the checkpoint):  python smoke_run.py --stub

Exactly one launch and one task. No retry. Any gate failure aborts BEFORE the checkpoint is touched (exit 70).
"""
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

REPO = Path("/Users/vincent/Documents/Veystrix")
SMOKE = Path("/private/tmp/claude-501/-Users-vincent-Documents-Veystrix/567b946c-8253-4400-8f35-732265a800bf/scratchpad/smoke")
DSH = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
CHECKPOINT = Path.home() / "models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
TASK = ("List the files in the current directory using the available file tool, then report their names. "
        "Do not modify any files.")
PORT = 8080
WALL_CAP = 20 * 60            # seconds for the whole attempt, from launch
EXIT_WAIT = 90                # seconds the server gets to exit cleanly after SIGINT
HEAVY_NAMES = re.compile(r"(Safari|Google Chrome|Chromium|Firefox|Arc\b|Brave|Xcode|Docker|Slack|zoom\.us|Teams|"
                         r"Photoshop|Final Cut|Premiere|Microsoft (Word|Excel|PowerPoint)|Simulator|Android Studio|"
                         r"IntelliJ|PyCharm|Code Helper|Figma|Notion|Spotify|Discord)", re.I)
ALLOWED_BIG = re.compile(r"(claude|Terminal|WindowServer|Finder|kernel_task|corespotlightd|mds)", re.I)


def sh(*cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as error:
        return f"<{type(error).__name__}: {error}>"


# -- readings --------------------------------------------------------------------------------------------------------


def vm_stat():
    out = sh("vm_stat")
    page = int((re.search(r"page size of (\d+) bytes", out) or [0, 16384])[1])
    values = {}
    for line in out.splitlines():
        match = re.match(r"(.+?):\s+(\d+)\.", line)
        if match:
            values[match[1].strip()] = int(match[2])
    gib = lambda key: round(values.get(key, 0) * page / 2 ** 30, 2)
    return {"free_gib": gib("Pages free"), "inactive_gib": gib("Pages inactive"), "active_gib": gib("Pages active"),
            "wired_gib": gib("Pages wired down"), "compressor_gib": gib("Pages occupied by compressor"),
            "swapouts": values.get("Swapouts", 0), "swapins": values.get("Swapins", 0)}


def swap_used_mb():
    match = re.search(r"used = ([\d.]+)M", sh("sysctl", "vm.swapusage"))
    return float(match[1]) if match else None


def pressure_level():
    out = sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()
    return int(out) if out.isdigit() else None


def power():
    out = sh("pmset", "-g", "batt")
    full = sh("pmset", "-g")
    percent = re.search(r"(\d+)%", out)
    return {"source": "ac" if "AC Power" in out else "battery", "percent": int(percent[1]) if percent else None,
            "low_power_mode": bool(re.search(r"lowpowermode\s+1", full)), "raw": out.splitlines()[0] if out else ""}


def free_percent():
    match = re.search(r"free percentage:\s+(\d+)%", sh("memory_pressure"))
    return int(match[1]) if match else None


# -- the gate (hard: any failure means the checkpoint is never touched) ---------------------------------------------


def gate(allow_battery=False):
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    p = power()
    if allow_battery and p["source"] != "ac":
        check("AC power (WAIVED by the owner for this run; result is preliminary)", p["percent"] is not None and p["percent"] >= 30,
              p["raw"] + f" ({p['percent']}%), battery run accepted by the owner, needs >= 30%")
    else:
        check("AC power", p["source"] == "ac", p["raw"] + f" ({p['percent']}%)")
    check("Low Power Mode off", not p["low_power_mode"], f"low_power_mode={p['low_power_mode']}")
    check("memory pressure normal (level 1)", pressure_level() == 1, f"level={pressure_level()}")
    free = free_percent()
    check("system free memory >= 40%", free is not None and free >= 40, f"free={free}%")
    before, swap_before = vm_stat(), swap_used_mb()
    time.sleep(8)
    after, swap_after = vm_stat(), swap_used_mb()
    check("no swap-out in the last 8 s", after["swapouts"] == before["swapouts"] and swap_after <= (swap_before or 0) + 1,
          f"swapouts {before['swapouts']}->{after['swapouts']}, swap {swap_before}->{swap_after} MB")
    heavy = []
    for line in sh("ps", "-axo", "rss=,comm=").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        mb, name = int(parts[0]) / 1024, parts[1]
        if (HEAVY_NAMES.search(name) and mb >= 200) or (mb > 1024 and not ALLOWED_BIG.search(name)):
            heavy.append(f"{os.path.basename(name)} {mb:.0f} MB")
    check("no memory-heavy applications", not heavy, ", ".join(heavy) or "none")
    stray = [l for l in sh("ps", "-axo", "pid=,command=").splitlines()
             if re.search(r"(macqwen/cli\.py|models\.flashnext\.watchdog|unittest|veystrix)", l) and "smoke_run.py" not in l
             and " ps " not in l and "grep" not in l]
    check("no stray Veystrix/test processes", not stray, "; ".join(s.strip()[:90] for s in stray) or "none")
    check(f"port {PORT} free", not sh("lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN").strip(), f"port {PORT}")
    free_gb = shutil.disk_usage(Path.home()).free / 1e9
    check("free SSD space >= 60 GB", free_gb >= 60, f"{free_gb:.0f} GB")
    check("checkpoint present", CHECKPOINT.is_dir(), str(CHECKPOINT))
    check("harness present", Path(DSH).exists(), DSH)
    ok = all(c["ok"] for c in checks)
    return ok, checks


# -- run -------------------------------------------------------------------------------------------------------------


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
    def __init__(self, path, key, stop):
        super().__init__(daemon=True)
        self.path, self.key, self.stop_event = path, key, stop
        self.peak = {"compressor_gib": 0, "pressure_level": 1, "swap_mb": 0, "server_rss_gb": 0, "swapouts": None}

    def run(self):
        with open(self.path, "a") as out:
            while not self.stop_event.is_set():
                row = {"t": time.time(), **vm_stat(), "swap_mb": swap_used_mb(), "pressure_level": pressure_level()}
                pid = listener_pid()
                if pid:
                    rss = sh("ps", "-o", "rss=", "-p", str(pid)).strip()
                    row["server_rss_gb"] = round(int(rss) / 1048576, 2) if rss.isdigit() else None
                    status, health = http("/health", self.key, 2)
                    if status == 200:
                        row["busy"], row["queue"] = health.get("busy"), health["queue"]
                for key, value in (("compressor_gib", row["compressor_gib"]), ("pressure_level", row["pressure_level"]),
                                   ("swap_mb", row["swap_mb"]), ("server_rss_gb", row.get("server_rss_gb"))):
                    if value is not None:
                        self.peak[key] = max(self.peak[key], value)
                if self.peak["swapouts"] is None:
                    self.peak["swapouts"] = row["swapouts"]
                row["swapouts_since_start"] = row["swapouts"] - self.peak["swapouts"]
                out.write(json.dumps(row) + "\n")
                out.flush()
                self.stop_event.wait(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-battery", action="store_true", help="owner-authorized: waive ONLY the AC check (needs >= 30%%)")
    parser.add_argument("--stub", action="store_true", help="offline mechanics test with a fake model; never the checkpoint")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = SMOKE / (f"stub-{stamp}" if args.stub else f"run-{stamp}")
    out.mkdir(parents=True)
    summary = {"stub": args.stub, "started": stamp, "outcome": None}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"evidence directory: {out}")
    ok, checks = gate(args.allow_battery)
    summary["preliminary_battery_run"] = bool(args.allow_battery and power()["source"] != "ac")
    summary["gate"] = checks
    for item in checks:
        print(("PASS " if item["ok"] else "FAIL ") + f"{item['check']}: {item['detail']}")
    if not ok and not args.stub:
        summary["outcome"] = "GATE FAILED: checkpoint not touched"
        save()
        print("\nGATE FAILED. The checkpoint was not loaded. Fix the failing checks and run again.")
        return 70
    key = secrets.token_urlsafe(24)
    (out / "env.json").write_text(json.dumps({
        "power": power(), "vm_before": vm_stat(), "swap_mb": swap_used_mb(), "pressure_level": pressure_level(),
        "git_head": sh("git", "-C", str(REPO), "rev-parse", "HEAD").strip(),
        "git_status_lines": len(sh("git", "-C", str(REPO), "status", "--short").splitlines()),
        "python": sys.version}, indent=2))
    env = {**os.environ, "MACQWEN_SERVER_API_KEY": key, "MACQWEN_REQUEST_LOG": str(out / "requests.jsonl")}
    if args.stub:
        cmd = [sys.executable, str(SMOKE / "stub_server.py"), str(PORT), key]
    else:
        cmd = [str(REPO / "chat.sh"), "--mode", "agentic", "--pool-gb", "0"]
    launched = time.time()
    log = open(out / "server.log", "wb")
    server = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              stdin=subprocess.DEVNULL)
    stop = threading.Event()
    sampler = Sampler(out / "samples.jsonl", key, stop)
    sampler.start()
    summary["stage"] = {}
    verdict = "FAIL"
    task_result = None
    worker = None
    try:
        # stage: ready (admission + load)
        ready = False
        while time.time() - launched < WALL_CAP:
            if server.poll() is not None:
                summary["outcome"] = f"server exited early with code {server.returncode} (admission refusal or crash)"
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
            began = time.time()
            task = subprocess.Popen(
                ([sys.executable, str(SMOKE / "stub_task.py"), str(PORT), key] if args.stub else
                 [DSH, "headless", "--patch", str(SMOKE / "patch.yml"), "--json", TASK]),
                cwd=SMOKE / "ws", env={**os.environ, "DSH_HOME": str(SMOKE / "dsh_home"), "VEYSTRIX_API_KEY": key},
                stdout=open(out / "task.jsonl", "wb"), stderr=open(out / "task.err", "wb"), start_new_session=True)
            while task.poll() is None and time.time() - launched < WALL_CAP:
                if server.poll() is not None:
                    break
                time.sleep(1)
            if task.poll() is None:
                os.killpg(task.pid, signal.SIGTERM)     # only the harness's own process group
                time.sleep(3)
                if task.poll() is None:
                    os.killpg(task.pid, signal.SIGKILL)
                summary["task_ended_by"] = "wall-clock cap or server exit"
            task_result = {"exit": task.poll(), "seconds": round(time.time() - began, 1)}
            summary["task"] = task_result
            summary["health_after"] = http("/health", key)[1]
        # stage: clean shutdown
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
                shutdown["ok"] = server.returncode in (0, None) or server.returncode == -signal.SIGINT
        shutdown["returncode"] = server.poll()
        summary["stage"]["shutdown"] = shutdown
    finally:
        stop.set()
        sampler.join(5)
        log.close()
        if server.poll() is None:
            server.terminate()
    summary["peaks"] = sampler.peak
    summary["swapouts_during_run"] = None
    try:
        rows = [json.loads(l) for l in open(out / "samples.jsonl")]
        summary["swapouts_during_run"] = rows[-1]["swapouts_since_start"] if rows else None
    except Exception:
        pass
    summary["wall_seconds"] = round(time.time() - launched, 1)
    # evidence copies
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
            rows = [l for l in source.read_text().splitlines() if l.strip()]
            recent = []
            for line in rows[-40:]:
                try:
                    if json.loads(line).get("wall", json.loads(line).get("t", 0)) >= time.time() - (time.time() - launched) - 30:
                        recent.append(line)
                except ValueError:
                    pass
            (out / name).write_text("\n".join(recent) + ("\n" if recent else ""))
    if (out / "requests.jsonl").exists():
        summary["requests"] = [json.loads(l) for l in open(out / "requests.jsonl")]
    leftovers = [l for l in sh("ps", "-axo", "pid=,command=").splitlines()
                 if "models.flashnext.watchdog" in l or "macqwen/cli.py" in l]
    summary["leftover_processes"] = leftovers
    summary["outcome"] = summary["outcome"] or "completed (judge the stages from the evidence, not this field)"
    save()
    print(json.dumps({k: summary.get(k) for k in ("stage", "task", "peaks", "swapouts_during_run", "wall_seconds",
                                                  "leftover_processes", "outcome")}, indent=2, default=str))
    print(f"\nevidence: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
