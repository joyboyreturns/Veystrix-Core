"""Out-of-process watchdog for the inference worker (``python -m models.flashnext.watchdog``).

Why it exists: on 2026-11-08 the first full-checkpoint AGENTIC run deadlocked (GIL held across a Metal wait). Every safety
mechanism that was a Python thread in the same interpreter, the memory/thermal guard included, was frozen with it and the
agent client waited forever. This process shares no interpreter, no GIL and no import of MLX with the worker.

What it watches is *work*, not a timer: the worker advances a counter in a shared record only when a streamed MoE layer
finishes in prefill or a token is decoded (see ``progress.py``), and names the stage it is in. A stage is stalled when

* the counter has not moved for that stage's deadline, AND the process has shown no activity (CPU above 10% of a core, or
  8 MiB of disk reads per poll) for the activity window: SSD-bound prefill keeps the disk busy and is slow, not stalled; a
  worker deadlocked on a GPU wait uses neither, OR
* the counter has not moved for the stage's hard limit, activity or not.
Idle is never stalled. Every stage including loading and stopping has limits, so a hang at shutdown is bounded too.

On a stall, in order: write a report; ask the worker for Python stacks (SIGUSR1 -> faulthandler, which works with the GIL
held); take a native ``sample``; tell the client whose request is in flight, over a duplicate of its socket the worker handed
over at request start; SIGTERM; wait out a grace period; SIGKILL only if it is still alive. Before every signal the process
identity (pid plus start time) is re-checked, so a recycled pid is never signalled, and only the one worker is ever signalled.
The optional independent swap backstop samples macOS outside the worker's GIL. It immediately kills the identity-checked
worker on an unsafe active-work reading or a failed probe. Sampling and process termination take time, so this cannot
guarantee a hard swap ceiling; admission and the worker's other guards remain necessary.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

from models.flashnext import progress as P

EXIT_STALL_HANDLED = 3
EXIT_MEMORY_HANDLED = 4

DEFAULT_DEADLINES = {"starting": 300, "loading": 900, "request": 60, "prefill": 180, "decode": 60, "sync": 60,
                     "cleanup": 60, "stopping": 60}
DEFAULT_HARD = {"starting": 900, "loading": 3600, "request": 300, "prefill": 1800, "decode": 600, "sync": 300,
                "cleanup": 300, "stopping": 180}


def _table(text, defaults) -> dict:
    table = dict(defaults)
    for part in (text or "").split(","):
        name, _, value = part.partition("=")
        if name.strip() in defaults and value.strip():
            table[name.strip()] = float(value)
    return table


@dataclass
class Config:
    deadlines: dict = field(default_factory=lambda: dict(DEFAULT_DEADLINES))
    hard: dict = field(default_factory=lambda: dict(DEFAULT_HARD))
    activity_window: float = 30.0
    poll: float = 1.0
    grace: float = 10.0
    kill_wait: float = 5.0
    cpu_fraction: float = 0.10
    io_bytes: int = 8 * 2 ** 20
    sample: bool = True            # native stack sample (3 s) before termination
    memory_watch: bool = False
    memory_poll: float = 0.25
    memory_stop_mib: float = 0.0

    @classmethod
    def from_environment(cls, environ=None):
        environ = os.environ if environ is None else environ
        number = lambda name, default: float(environ.get(name) or default)       # noqa: E731
        from models.flashnext.memory_controller import Policy
        policy = Policy.from_environment(environ)
        enabled = environ.get("FLASHNEXT_WATCHDOG_MEMORY", "1" if environ.get("FLASHNEXT_PREFILL_ORDER", "").strip().lower() == "layer" else "0") == "1"
        interval = number("FLASHNEXT_WATCHDOG_MEMORY_POLL", 0.25)
        if not 0.05 <= interval <= 1.0:
            raise ValueError("FLASHNEXT_WATCHDOG_MEMORY_POLL must be between 0.05 and 1 second")
        if enabled and (policy.swap_ceiling_mib <= 0 or policy.swap_ceiling_margin_mib >= policy.swap_ceiling_mib):
            raise ValueError("independent memory watchdog requires a positive ceiling and a smaller margin")
        return cls(deadlines=_table(environ.get("FLASHNEXT_WATCHDOG_DEADLINES"), DEFAULT_DEADLINES),
                   hard=_table(environ.get("FLASHNEXT_WATCHDOG_HARD"), DEFAULT_HARD),
                   activity_window=number("FLASHNEXT_WATCHDOG_ACTIVITY_WINDOW", 30.0),
                   poll=number("FLASHNEXT_WATCHDOG_POLL", 1.0), grace=number("FLASHNEXT_WATCHDOG_GRACE", 10.0),
                   kill_wait=number("FLASHNEXT_WATCHDOG_KILL_WAIT", 5.0),
                   sample=environ.get("FLASHNEXT_WATCHDOG_SAMPLE", "1") != "0",
                   memory_watch=enabled, memory_poll=interval,
                   memory_stop_mib=policy.swap_ceiling_mib - policy.swap_ceiling_margin_mib)


# -- what the process is doing, from outside ----------------------------------------------------------------------


class _RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
        "diskio_bytesread", "diskio_byteswritten")]


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class _SwapUsage(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("available", ctypes.c_uint64), ("used", ctypes.c_uint64),
                ("pagesize", ctypes.c_uint32), ("encrypted", ctypes.c_int32)]


def swap_used_mib() -> float:
    """Read the kernel's swap counter directly, without launching a subprocess."""
    usage = _SwapUsage()
    size = ctypes.c_size_t(ctypes.sizeof(usage))
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.sysctlbyname
    call.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    if call(b"vm.swapusage", ctypes.byref(usage), ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctlbyname vm.swapusage failed")
    if size.value != ctypes.sizeof(usage) or usage.used > usage.total or not usage.pagesize:
        raise ValueError("invalid vm.swapusage reading")
    return usage.used / 2**20


def usage(pid: int):
    """``(cpu_ns, disk_read_bytes)`` for a process, or None. CPU is user plus system time."""
    info = _RusageV2()
    if P._libproc().proc_pid_rusage(int(pid), 2, ctypes.byref(info)) != 0:
        return None
    base = _Timebase()
    ctypes.CDLL(None).mach_timebase_info(ctypes.byref(base))
    ticks = info.user_time + info.system_time
    return (ticks * base.numer // max(base.denom, 1), int(info.diskio_bytesread))


# -- the decision (pure; the clock and the readings are arguments) -------------------------------------------------


@dataclass
class Verdict:
    state: str                  # "ok", "slow" or "stalled"
    reason: str
    stage: str = ""
    progress_age: float = 0.0


class Judge:
    def __init__(self, config: Config):
        self.config = config
        self.last_activity = None
        self.previous = None            # (wall, cpu_ns, io_bytes)
        self.cpu_fraction = 0.0

    def observe(self, row: dict, reading, now: float) -> Verdict:
        config, stage = self.config, row["stage"]
        if reading is not None:
            if self.previous is not None:
                wall = max(now - self.previous[0], 1e-6)
                self.cpu_fraction = (reading[0] - self.previous[1]) / 1e9 / wall
                moved = reading[1] - self.previous[2]
                if self.cpu_fraction >= config.cpu_fraction or moved >= config.io_bytes * max(wall / config.poll, 1.0):
                    self.last_activity = now
            self.previous = (now, reading[0], reading[1])
        if self.last_activity is None:
            self.last_activity = now
        if stage == "idle":
            return Verdict("ok", "idle", stage)
        since = max(row["progress_ns"], row["entered_ns"]) / 1e9
        age = max(0.0, now - since)
        deadline, hard = config.deadlines.get(stage), config.hard.get(stage)
        quiet = now - self.last_activity
        detail = (f"stage {stage}{' (' + row['detail'] + ')' if row['detail'] else ''}: no progress for {age:.0f}s "
                  f"(deadline {deadline:.0f}s, hard {hard:.0f}s), process quiet for {quiet:.0f}s "
                  f"(cpu {self.cpu_fraction * 100:.0f}% of a core)")
        if hard is not None and age >= hard:
            return Verdict("stalled", "hard limit reached; " + detail, stage, age)
        if deadline is not None and age >= deadline:
            if quiet >= config.activity_window:
                return Verdict("stalled", "no progress and no activity; " + detail, stage, age)
            return Verdict("slow", "no progress but still busy; " + detail, stage, age)
        return Verdict("ok", detail, stage, age)


# -- acting on a stall ---------------------------------------------------------------------------------------------


def client_error_bytes(path: str, started: bool, message: str, code: str = "worker_stalled") -> bytes:
    """The failure as the client's own protocol expects it, so the agent sees a clean error instead of a dead socket."""
    error = {"message": message, "type": "server_error", "code": code}
    if not started:
        body = json.dumps({"error": error}).encode()
        return (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 30\r\n"
                b"Connection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    if path.startswith("/v1/messages"):
        data = (b"\n\nevent: error\ndata: " + json.dumps({"type": "error", "error": {"type": "api_error", "message": message}}).encode()
                + b"\n\nevent: message_stop\ndata: " + json.dumps({"type": "message_stop"}).encode() + b"\n\n")
    elif path.startswith("/v1/responses"):
        data = (b"\n\nevent: response.failed\ndata: " + json.dumps(
            {"type": "response.failed", "response": {"status": "failed", "error": error}}).encode() + b"\n\n")
    else:
        data = b"\n\ndata: " + json.dumps({"error": error}).encode() + b"\n\ndata: [DONE]\n\n"
    return data


class Responder:
    def __init__(self, pid, identity, config, directory, trace=None, log=lambda text: None, sleep=time.sleep, clock=time.time):
        self.pid, self.identity, self.config, self.directory = pid, identity, config, directory
        self.trace, self.log, self.sleep, self.clock = trace, log, sleep, clock
        self.clients: dict = {}                  # request seq -> (socket, path)

    # identity-checked signalling: only the exact worker, never a recycled pid
    def same_process(self) -> bool:
        return P.process_identity(self.pid) == self.identity

    def signal(self, sig) -> bool:
        if not self.same_process():
            return False
        try:
            os.kill(self.pid, sig)
            return True
        except ProcessLookupError:
            return False

    def accept_clients(self, channel) -> None:
        while True:
            try:
                data, fds, _flags, _addr = socket.recv_fds(channel, 4096, 8, socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            if not data:
                return
            try:
                message = json.loads(data)
            except ValueError:
                for fd in fds:
                    os.close(fd)
                continue
            if message.get("op") == "begin" and fds:
                self.close_client(message["seq"])
                self.clients[message["seq"]] = (socket.socket(fileno=fds[0]), message.get("path", ""))
                for extra in fds[1:]:
                    os.close(extra)
            elif message.get("op") == "end":
                self.close_client(message["seq"])

    def close_client(self, seq) -> None:
        item = self.clients.pop(seq, None)
        if item is not None:
            try:
                item[0].close()
            except OSError:
                pass

    def notify(self, row: dict, message: str, code: str = "worker_stalled") -> bool:
        item = self.clients.get(row["request_seq"])
        if item is None:
            return False
        sock, path = item
        try:
            sock.setblocking(False)
            sock.send(client_error_bytes(path, bool(row["response_started"]), message, code))
            sock.shutdown(socket.SHUT_WR)
            return True
        except OSError:
            return False

    def diagnostics(self, stamp: str) -> dict:
        out = {}
        frames = os.path.join(self.directory, f"pyframes-{self.pid}.txt")
        before = os.path.getsize(frames) if os.path.exists(frames) else 0
        if self.signal(signal.SIGUSR1):
            self.sleep(1.0)
            try:
                with open(frames, "rb") as handle:
                    handle.seek(before)
                    out["python_stacks"] = handle.read(60000).decode("utf-8", "replace")
            except OSError:
                pass
        sample = os.path.join(self.directory, f"sample-{self.pid}-{stamp}.txt")
        if not self.config.sample:
            return out
        try:
            subprocess.run(["/usr/bin/sample", str(self.pid), "3", "-file", sample], capture_output=True, timeout=25,
                           stdin=subprocess.DEVNULL)
            if os.path.exists(sample):
                out["native_sample"] = sample
        except Exception as error:
            out["native_sample_error"] = f"{type(error).__name__}: {error}"
        return out

    def terminate(self) -> dict:
        outcome = {"sigterm": self.signal(signal.SIGTERM), "sigkill": False}
        deadline = self.clock() + self.config.grace
        while self.same_process() and self.clock() < deadline:
            self.sleep(0.25)
        if self.same_process():
            outcome["sigkill"] = self.signal(signal.SIGKILL)
            deadline = self.clock() + self.config.kill_wait
            while self.same_process() and self.clock() < deadline:
                self.sleep(0.25)
        outcome["gone"] = not self.same_process()
        return outcome

    def handle_stall(self, row: dict, verdict: Verdict, reading) -> dict:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        report = {"time": stamp, "pid": self.pid, "identity": list(self.identity), "verdict": verdict.reason,
                  "stage": verdict.stage, "record": row, "cpu_ns": None if reading is None else reading[0],
                  "disk_read_bytes": None if reading is None else reading[1], "guard_trace": self.trace}
        message = (f"The Veystrix inference worker stalled in '{verdict.stage}' and is being restarted. "
                   f"Nothing was generated for this request. {verdict.reason}")
        report.update(self.diagnostics(stamp))
        report["client_notified"] = self.notify(row, message)
        report["termination"] = self.terminate()
        path = os.path.join(self.directory, f"stall-{self.pid}-{stamp}.json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(report, handle, indent=2)
        report["report_path"] = path
        self.log(f"stall handled: {path}")
        return report

    def handle_memory(self, row: dict, reason: str) -> dict:
        """Stop immediately; diagnostic sampling and graceful wait would add unsafe latency."""
        killed = self.signal(signal.SIGKILL)
        message = f"Veystrix stopped its inference worker for memory safety: {reason}. Retry only after memory pressure falls."
        notified = self.notify(row, message, "memory_safety_stop") if killed else False
        report = {"pid": self.pid, "identity": list(self.identity), "reason": reason,
                  "stage": row["stage"], "sigkill": killed, "client_notified": notified, "guard_trace": self.trace}
        path = os.path.join(self.directory, f"memory-{self.pid}-{time.time_ns()}.json")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(report, handle, indent=2)
            self.log(f"memory stop: {path}")
        except OSError as error:
            self.log(f"memory stop report failed: {error}; reason: {reason}")
        return report


# -- the loop ------------------------------------------------------------------------------------------------------


def run(pid, identity, record_path, channel, directory, config=None, trace=None, log=lambda text: None,
        sleep=time.sleep, clock=time.time, usage_fn=usage, swap_fn=swap_used_mib) -> int:
    config = config or Config.from_environment()
    record = P.Record(record_path)
    judge, responder = Judge(config), Responder(pid, identity, config, directory, trace, log, sleep, clock)
    last_judge = None
    heartbeat_fd = None
    heartbeat_at = None
    try:
        while True:
            if P.process_identity(pid) != identity:
                log("worker exited")
                return 0
            if channel is not None:
                responder.accept_clients(channel)
            row = record.read()
            if row is None or row["pid"] != pid or (row["start_sec"], row["start_usec"]) != identity:
                sleep(min(config.poll, config.memory_poll) if config.memory_watch else config.poll)
                continue
            if config.memory_watch:
                try:
                    used = float(swap_fn())
                    if not (0 <= used < float("inf")):
                        raise ValueError("invalid swap counter")
                    reason = (f"swap {used:.1f} MiB reached safety stop {config.memory_stop_mib:.1f} MiB"
                              if used >= config.memory_stop_mib and row["stage"] != "idle" else None)
                    if reason is None and (heartbeat_at is None or clock() - heartbeat_at >= 1.0):
                        if heartbeat_fd is None:
                            heartbeat_fd = os.open(record_path + ".memory", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        os.utime(heartbeat_fd, None)
                        heartbeat_at = clock()
                except Exception as error:
                    reason = f"independent swap probe failed: {type(error).__name__}: {error}"
                if reason:
                    responder.handle_memory(row, reason)
                    return EXIT_MEMORY_HANDLED
            now = clock()
            if last_judge is not None and now - last_judge < config.poll:
                sleep(config.memory_poll)
                continue
            last_judge = now
            verdict = judge.observe(row, usage_fn(pid), now)
            if verdict.state == "stalled":
                responder.handle_stall(row, verdict, judge.previous and (judge.previous[1], judge.previous[2]))
                return EXIT_STALL_HANDLED
            if verdict.state == "slow":
                log(verdict.reason)
            sleep(min(config.poll, config.memory_poll) if config.memory_watch else config.poll)
    finally:
        if heartbeat_fd is not None:
            os.close(heartbeat_fd)
        for seq in list(responder.clients):
            responder.close_client(seq)
        record.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Out-of-process watchdog for the Veystrix inference worker.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start", required=True, help="start time of the worker as <sec>.<usec> (its identity)")
    parser.add_argument("--record", required=True)
    parser.add_argument("--channel-fd", type=int)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--trace")
    args = parser.parse_args(argv)
    sec, _, usec = args.start.partition(".")
    channel = socket.socket(fileno=args.channel_fd) if args.channel_fd is not None else None
    if channel is not None:
        channel.setblocking(False)
    log = lambda text: print(time.strftime("%H:%M:%S"), text, file=sys.stderr, flush=True)      # noqa: E731
    return run(args.pid, (int(sec), int(usec)), args.record, channel, args.dir, trace=args.trace, log=log)


if __name__ == "__main__":
    sys.exit(main())
