"""Owner-run launcher and unattended supervisor for the AGENTIC server (2026-11-09).

``serve.sh`` runs this module. It starts ``./chat.sh --mode agentic ...`` (the only launch path that enables the swap
guard, chunked prefill, the prefix cache and the watchdog), waits for ``/health``, and, in ``--overnight`` mode, keeps a
server available for hours without anyone watching.

What it does NOT do, on purpose:
* It never retries a request. A restart brings up a fresh, empty server; the client decides what to send next.
* It never deletes swapfiles, purges caches, closes other apps or edits the cache directory.
* It never loosens the server's own protections (swap guard +2.5 GiB, thermal, battery, Low Power Mode, watchdog).
  A guard stop is respected: the supervisor waits for the machine to settle and restarts only inside a bounded budget.

Restart policy (every number is an option):
* exit 0 / SIGTERM from the owner      -> stop, no restart
* exit 70 (launch configuration error) -> stop, restarting cannot help
* exit 71 (port already served)         -> stop, never start a second server next to a live one
* exit 69 (admission refused)          -> wait ``--settle-seconds`` for memory to recover, then retry, counted as a stop
* exit 75 (guard stop) or anything else -> cool down, wait for swap to stop moving, restart
* more than ``--max-restarts`` restarts inside ``--window-minutes`` -> the supervisor stops and says why (crash loop).
Restarts after a *guard* stop have their own, smaller budget (``--max-guard-stops``) because each one means the machine
was under real memory pressure.

Everything is logged to ``~/.flashnext/logs/supervisor.jsonl`` and summarised in ``supervisor-status.json``
(``./serve.sh --status``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.path.expanduser(os.environ.get("VEYSTRIX_SUPERVISOR_LOG_DIR", "~/.flashnext/logs")))
DEFAULT_CHECKPOINT = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
GUARD_EXIT = 75
ADMISSION_EXIT = 69
CONFIG_EXIT = 70
PORT_BUSY_EXIT = 71
from models.flashnext.power_gate import POWER_EXIT_CODE as POWER_EXIT   # 72: battery below the floor


def bundled() -> bool:
    """True inside the packaged app: there is no chat.sh or source tree, only compiled modules and a bundled Python."""
    return Path(sys.executable).resolve().is_relative_to((REPO.parent / "python").resolve())


def server_command(args) -> list:
    """The one supported launch line (see models/flashnext/supervision.py LAUNCH_COMMAND)."""
    launcher = [sys.executable, "-m", "veystrix.cli"] if bundled() else [str(REPO / "chat.sh")]
    command = [*launcher, "--mode", "agentic",
               "--pool-gb", str(args.pool_gb),
               "--host", args.host, "--port", str(args.port),
               "--context-limit", "262144",
               "--max-cold-prefill", "32768", "--max-incremental-prefill", "32768",
               "--model", "flashnext", "--checkpoint", args.checkpoint]
    for name in getattr(args, "enable", None) or ():
        command += ["--enable", name]          # an explicit experimental option, e.g. layer-major
    if getattr(args, "prefix_dir", None):
        command += ["--prefix-dir", args.prefix_dir]
    if getattr(args, "preferences_file", None):
        command += ["--preferences-file", args.preferences_file]
    if getattr(args, "no_auto_purge", False):
        command.append("--no-auto-purge")
    if getattr(args, "threshold", None) is not None:
        command += ["--threshold", str(args.threshold)]
    return command


def swap_used_mb() -> float:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else 0.0


def power_state() -> dict:
    try:
        out = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {"source": "unknown"}
    percent = re.search(r"(\d+)%", out)
    return {"source": "ac" if "AC Power" in out else "battery", "percent": int(percent.group(1)) if percent else None}


def port_in_use(host: str, port: int) -> bool:
    """True when something already accepts connections on host:port (another server, maybe an older supervisor's)."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def health(host: str, port: int, timeout: float = 5.0):
    """Parsed /health, or None when the server cannot be reached."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


class Policy:
    """Pure restart decisions (no clock, no process): the part worth unit-testing."""

    def __init__(self, *, max_restarts=8, window_s=1800.0, max_guard_stops=4, guard_window_s=21600.0,
                 base_cooldown_s=20.0, guard_cooldown_s=180.0, max_cooldown_s=600.0):
        self.max_restarts, self.window_s = max_restarts, window_s
        self.max_guard_stops, self.guard_window_s = max_guard_stops, guard_window_s
        self.base_cooldown_s, self.guard_cooldown_s, self.max_cooldown_s = base_cooldown_s, guard_cooldown_s, max_cooldown_s
        self.restarts = deque()
        self.guard_stops = deque()
        self.consecutive_failures = 0
        self.last_failure_time = None

    @staticmethod
    def _prune(queue, now, window):
        while queue and now - queue[0] > window:
            queue.popleft()

    def decide(self, exit_code: int, now: float, *, owner_stop: bool = False) -> tuple:
        """Return ``(action, cooldown_seconds, reason)`` with action in {"stop", "restart"}."""
        if owner_stop or exit_code == 0:
            self.consecutive_failures = 0
            self.last_failure_time = None
            return "stop", 0.0, "stopped by the owner" if owner_stop else "server exited cleanly"
        if exit_code == PORT_BUSY_EXIT:
            return "stop", 0.0, "the port is already served by another process: stop it first (./serve.sh --status shows this supervisor)"
        if exit_code == POWER_EXIT:
            return "stop", 0.0, "the battery is below the minimum (exit 72): plug in or charge, then start again; restarting cannot fix it"
        if exit_code == CONFIG_EXIT:
            return "stop", 0.0, "launch configuration refused (exit 70): restarting cannot fix it"

        # If previous crash was longer ago than the window, reset consecutive failure count
        if self.last_failure_time is not None and now - self.last_failure_time > self.window_s:
            self.consecutive_failures = 0
        self.last_failure_time = now
        self.consecutive_failures += 1

        self._prune(self.restarts, now, self.window_s)
        self._prune(self.guard_stops, now, self.guard_window_s)
        if exit_code in (GUARD_EXIT, ADMISSION_EXIT):
            self.guard_stops.append(now)
            if len(self.guard_stops) > self.max_guard_stops:
                return "stop", 0.0, (f"{len(self.guard_stops)} memory stops within {self.guard_window_s / 3600:g} h "
                                     f"(limit {self.max_guard_stops}): the machine cannot sustain this load, so the supervisor stops")
            self.restarts.append(now)
            return "restart", self.guard_cooldown_s, f"memory guard stop (exit {exit_code}); restarting after cooldown"
        self.restarts.append(now)
        failures = max(self.consecutive_failures, len(self.restarts))
        if failures > self.max_restarts:
            return "stop", 0.0, f"{failures} restarts within {self.window_s / 60:g} min (limit {self.max_restarts}): crash loop"
        cooldown = min(self.max_cooldown_s, self.base_cooldown_s * (2 ** (self.consecutive_failures - 1)))
        return "restart", cooldown, f"server exited with code {exit_code}; restarting after cooldown"


class AutoWarmer:
    """While the owner is away, keep every remembered client's prompt prefix cached for every reasoning effort.

    Runs only when the server has been idle ``idle_seconds`` on AC power, one prefill-only request at a time, through the normal
    API (swap guard active). A real request that queues behind it drops the warm-up first. A safety stop ends auto-warming until
    the server is restarted; nothing is retried. Each success also refreshes the entry's last-use time, which is what keeps these
    shared prefixes from being the oldest thing the SSD budget evicts.
    """

    def __init__(self, args, log, *, power_fn, clock, warm_module=None, thread_factory=None):
        from veystrix import warm as warm_module_default
        self.w = warm_module or warm_module_default
        self.args, self.log, self.power_fn, self.clock = args, log, power_fn, clock
        self.idle_seconds = float(getattr(args, "warm_idle_minutes", 15.0)) * 60.0
        self.enabled = bool(getattr(args, "auto_warm", False)) and self.idle_seconds > 0
        self.refresh_seconds = float(getattr(args, "warm_refresh_hours", 6.0)) * 3600.0
        self.directory = None
        self.stopped_until_restart = False
        self.thread = None
        self.abort = None
        self._thread_factory = thread_factory or (lambda target: __import__("threading").Thread(target=target, daemon=True))

    def reset_for_new_server(self):
        self.stopped_until_restart = False

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def cancel(self):
        if self.abort is not None:
            self.abort.set()

    def _quiet(self, report) -> bool:
        worker = report.get("worker") or {}
        queue = report.get("queue") or {}
        return (report.get("status") == "ok" and report.get("ready") and worker.get("stage") == "idle" and not worker.get("busy")
                and not queue.get("waiting") and float(worker.get("seconds_in_stage") or 0.0) >= self.idle_seconds)

    def tick(self, report) -> None:
        if not self.enabled or self.stopped_until_restart or report is None:
            return
        if self.running:
            queue = report.get("queue") or {}
            if queue.get("waiting"):                      # a real request is waiting behind the warm-up: it goes first
                self.log("auto_warm_yield", waiting=queue.get("waiting"))
                self.cancel()
            return
        if not self._quiet(report):
            return
        power = self.power_fn()
        if power.get("source") != "ac":
            return
        entries = self.w.load(self.directory)
        if not entries:
            return
        state = self.w.load_state(self.directory)
        jobs = self.w.due_jobs(entries, self.w.DEFAULT_EFFORTS, state, self.clock(), self.refresh_seconds)
        if not jobs:
            return
        entry, effort = jobs[0]
        self.abort = __import__("threading").Event()
        self.thread = self._thread_factory(lambda: self._run(entry, effort, self.abort))
        self.thread.start()

    def _run(self, entry, effort, abort) -> None:
        base = f"http://{self.args.host}:{self.args.port}"
        key = os.environ.get("VEYSTRIX_API_KEY", "local")
        began = self.clock()
        try:
            result = self.w.warm_one(base, entry, effort, key=key, timeout=3600.0, abort=abort)
        except self.w.WarmAborted:
            self.log("auto_warm_aborted", client=entry["family"], effort=effort)
            return
        except self.w.WarmStop as exc:
            self.stopped_until_restart = True
            self.log("auto_warm_stopped", client=entry["family"], effort=effort, reason=str(exc)[:200],
                     note="auto-warming is off until the server restarts; nothing is retried")
            return
        except Exception as exc:
            self.stopped_until_restart = True
            self.log("auto_warm_failed", client=entry["family"], effort=effort, error=f"{type(exc).__name__}: {exc}"[:200])
            return
        state = self.w.load_state(self.directory)
        state[self.w.job_key(entry, effort)] = self.clock()
        self.w.save_state(state, self.directory)
        self.log("auto_warm_done", client=entry["family"], effort=effort, seconds=round(self.clock() - began, 1), **{
            k: result.get(k) for k in ("prompt_tokens", "hit_tokens", "prefilled_tokens", "outcome")})


class Supervisor:
    def __init__(self, args, *, policy=None, popen=subprocess.Popen, sleep=time.sleep, clock=time.time,
                 health_fn=health, swap_fn=swap_used_mb, power_fn=power_state, log_dir=LOG_DIR,
                 port_fn=port_in_use, warmer=None):
        self.args = args
        self.policy = policy or Policy(max_restarts=args.max_restarts, window_s=args.window_minutes * 60.0,
                                       max_guard_stops=args.max_guard_stops, guard_cooldown_s=args.guard_cooldown)
        self.popen, self.sleep, self.clock = popen, sleep, clock
        self.health_fn, self.swap_fn, self.power_fn, self.port_fn = health_fn, swap_fn, power_fn, port_fn
        self.warmer = warmer if warmer is not None else AutoWarmer(args, self.log, power_fn=power_fn, clock=clock)
        self.log_dir = Path(log_dir)
        self.child = None
        self.owner_stop = False
        self.starts = 0
        self.history = []
        self.started_at = clock()

    # ---- logging ----
    def log(self, event: str, **fields) -> None:
        row = {"t": round(self.clock(), 3), "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(self.log_dir / "supervisor.jsonl", "a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass
        print(f"[supervisor {row['iso']}] {event} " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)

    def write_status(self, state: str, **extra) -> None:
        status = {"state": state, "pid": os.getpid(), "server_pid": getattr(self.child, "pid", None), "starts": self.starts,
                  "since": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_at)),
                  "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "recent": self.history[-10:], **extra}
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            tmp = self.log_dir / "supervisor-status.json.tmp"
            tmp.write_text(json.dumps(status, indent=1, default=str))
            os.replace(tmp, self.log_dir / "supervisor-status.json")
        except OSError:
            pass

    def prune_launch_logs(self, keep: int = 12) -> None:
        try:
            files = sorted(self.log_dir.glob("server-launch-*.log"), key=lambda p: p.stat().st_mtime)
            for stale in files[:-keep]:
                stale.unlink()
        except OSError:
            pass

    # ---- steps ----
    def wait_until_settled(self) -> None:
        """Before a restart: swap must stop moving (a fresh server measures its own +2.5 GiB from the launch baseline)."""
        deadline = self.clock() + self.args.settle_seconds
        last = self.swap_fn()
        while self.clock() < deadline and not self.owner_stop:
            self.sleep(15)
            now = self.swap_fn()
            if abs(now - last) < 64:
                self.log("settled", swap_mb=round(now))
                return
            last = now
        self.log("settle_timeout", swap_mb=round(last))

    def launch(self):
        self.starts += 1
        self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        logfile = self.log_dir / f"server-launch-{time.strftime('%Y%m%d-%H%M%S')}-{self.starts}.log"
        handle = open(logfile, "ab")
        env = dict(os.environ)
        env.setdefault("VEYSTRIX_API_KEY", "local")
        if bundled():
            env.setdefault("VEYSTRIX_PYTHON", sys.executable)      # the bundled interpreter already has the pinned runtime
            env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
        # An unattended goal must survive one malformed model tool call: the client returns its own tool error and the
        # model retries. Set VEYSTRIX_INVALID_TOOL_CALLS=error to get the strict 400 instead.
        env.setdefault("VEYSTRIX_INVALID_TOOL_CALLS", "pass")
        command = server_command(self.args)
        try:
            self.child = self.popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(REPO), start_new_session=True)
        finally:
            handle.close()   # the child owns its inherited copy
        self.log("server_started", pid=self.child.pid, start=self.starts, logfile=str(logfile), command=" ".join(command),
                 power=self.power_fn(), swap_mb=round(self.swap_fn()))
        return self.child

    def monitor(self) -> int:
        """Block until the child exits (return its code). Kill and report -9 if /health is gone for too long."""
        ready_logged = False
        unreachable_since = None
        next_status = 0.0
        while True:
            code = self.child.poll()
            if code is not None:
                self.warmer.cancel()
                return code
            if self.owner_stop:
                self.terminate_child()
                return 0
            report = self.health_fn(self.args.host, self.args.port)
            now = self.clock()
            if report is None:
                unreachable_since = unreachable_since or now
                if ready_logged and now - unreachable_since > self.args.unreachable_seconds:
                    self.log("health_lost", seconds=round(now - unreachable_since))
                    self.terminate_child()
                    return -9
            else:
                unreachable_since = None
                if report.get("status") == "stopped":
                    self.log("server_reports_stop", report={k: report.get(k) for k in ("status", "ready", "stopped")})
                if report.get("ready") and not ready_logged and self.child.poll() is None:
                    ready_logged = True
                    self.warmer.reset_for_new_server()
                    self.log("server_ready", pid=self.child.pid)
                if ready_logged:
                    self.warmer.tick(report)
            if now >= next_status:
                next_status = now + 60
                self.write_status("running", ready=ready_logged, health=(report or {}).get("status"),
                                  swap_mb=round(self.swap_fn()), power=self.power_fn())
            self.sleep(self.args.poll_seconds)

    def terminate_child(self, grace: float = 45.0) -> None:
        self.warmer.cancel()
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            child.terminate()
        deadline = self.clock() + grace
        while child.poll() is None and self.clock() < deadline:
            self.sleep(0.5)
        if child.poll() is None:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                child.kill()

    def request_stop(self, *_):
        self.owner_stop = True

    def run(self) -> int:
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, self.request_stop)
            except ValueError:   # not the main thread
                pass
        try:
            return self._run()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _run(self) -> int:
        self.log("supervisor_started", overnight=self.args.overnight, port=self.args.port, pool_gb=self.args.pool_gb,
                 power=self.power_fn())
        while True:
            if self.port_fn(self.args.host, self.args.port):
                # Never start a second server next to a live one: its /health would pass for ours. On the first start this
                # is fatal; after a restart the old child may still be releasing the port, so give it a minute.
                waited = 0.0
                while self.port_fn(self.args.host, self.args.port) and waited < (60.0 if self.starts else 0.0) and not self.owner_stop:
                    self.sleep(5.0)
                    waited += 5.0
                if self.port_fn(self.args.host, self.args.port):
                    self.log("port_busy", host=self.args.host, port=self.args.port)
                    self.history.append({"at": time.strftime("%H:%M:%S"), "exit": PORT_BUSY_EXIT, "action": "stop", "reason": "port busy"})
                    self.write_status("stopped", reason="port already in use", exit=PORT_BUSY_EXIT)
                    return PORT_BUSY_EXIT
            try:
                self.launch()
            except OSError as exc:
                reason = f"Unable to start engine launcher: {exc}. Check the bundled runtime and permissions."
                self.log("launch_failed", reason=reason)
                self.write_status("stopped", reason=reason, exit=CONFIG_EXIT)
                return CONFIG_EXIT
            code = self.monitor()
            action, cooldown, reason = self.policy.decide(code, self.clock(), owner_stop=self.owner_stop)
            if action == "restart" and not self.args.overnight:
                action, reason = "stop", reason.split(";")[0] + " (single run: not restarting; use --overnight)"
            self.history.append({"at": time.strftime("%H:%M:%S"), "exit": code, "action": action, "reason": reason})
            self.log("server_exited", exit=code, action=action, reason=reason, cooldown=cooldown)
            if action == "stop" or not self.args.overnight:
                self.write_status("stopped", reason=reason, exit=code)
                return 0 if code in (0, -15) or self.owner_stop else (code if code > 0 else 1)
            self.write_status("cooling_down", reason=reason, cooldown=cooldown)
            waited = 0.0
            while waited < cooldown and not self.owner_stop:
                self.sleep(min(5.0, cooldown - waited))
                waited += 5.0
            if self.owner_stop:
                self.write_status("stopped", reason="stopped by the owner during cooldown")
                return 0
            self.wait_until_settled()


def build_parser():
    p = argparse.ArgumentParser(description="Launch the Veystrix AGENTIC server yourself; --overnight keeps it up unattended.")
    p.add_argument("--overnight", action="store_true", help="restart the server after a crash or guard stop (bounded, logged)")
    p.add_argument("--status", action="store_true", help="print the supervisor status file and the last events, then exit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("VEYSTRIX_PORT", "8080")))
    p.add_argument("--pool-gb", default=os.environ.get("VEYSTRIX_POOL_GB", "0"))
    p.add_argument("--enable", action="append", default=[], metavar="NAME",
                   help="pass an experimental option to chat.sh (see its --help), e.g. --enable layer-major")
    p.add_argument("--checkpoint", default=os.environ.get("VEYSTRIX_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--prefix-dir", help="separate prefix-cache directory (default: normal AGENTIC cache)")
    p.add_argument("--preferences-file", help="preferences file for the child server")
    p.add_argument("--no-auto-purge", action="store_true", help="disable automatic file-cache purge")
    p.add_argument("--threshold", type=float, default=None,
                   help="expert mass retained; values below 1.0 enable approximate legacy routing")
    p.add_argument("--max-restarts", type=int, default=8)
    p.add_argument("--window-minutes", type=float, default=30.0)
    p.add_argument("--max-guard-stops", type=int, default=4)
    p.add_argument("--guard-cooldown", type=float, default=180.0, help="seconds to wait after a memory guard stop")
    p.add_argument("--settle-seconds", type=float, default=600.0, help="longest wait for swap to stop moving before a restart")
    p.add_argument("--unreachable-seconds", type=float, default=300.0, help="kill and restart when /health is gone this long after ready")
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--auto-warm", dest="auto_warm", action="store_true", default=None,
                   help="while idle on AC power, keep every remembered client's prompt prefix cached for every effort (default with --overnight)")
    p.add_argument("--no-auto-warm", dest="auto_warm", action="store_false")
    p.add_argument("--warm-idle-minutes", type=float, default=15.0, help="idle time before a warm-up may start")
    p.add_argument("--warm-refresh-hours", type=float, default=6.0, help="a variant warmed within this long is left alone")
    return p


def print_status(log_dir=LOG_DIR) -> int:
    status = Path(log_dir) / "supervisor-status.json"
    print(status.read_text() if status.exists() else "no supervisor status yet")
    log = Path(log_dir) / "supervisor.jsonl"
    if log.exists():
        print("--- last supervisor events ---")
        for line in log.read_text().splitlines()[-15:]:
            row = json.loads(line)
            print(row.get("iso"), row.get("event"), {k: v for k, v in row.items() if k not in ("t", "iso", "event", "command")})
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.auto_warm is None:
        args.auto_warm = bool(args.overnight)
    if args.status:
        return print_status()
    return Supervisor(args).run()


if __name__ == "__main__":
    sys.exit(main())
