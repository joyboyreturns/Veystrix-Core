"""Worker-side forward-progress record for the out-of-process watchdog (``watchdog.py``).

The inference process writes where it is (stage) and that it is advancing (a counter moved by real work: a completed
streamed MoE layer in prefill, a decoded token) into a small memory-mapped file. A separate process reads it. Nothing here
is a heartbeat from a timer thread: a thread that keeps ticking while the inference thread is deadlocked would hide
exactly the failure this exists to catch, so only the work itself moves the counter.

Pure standard library (no MLX), so the watchdog can import the record layout without touching the GPU stack.
With no record configured every call is a no-op and costs one attribute test.
"""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import mmap
import os
import socket
import struct
import subprocess
import sys
import threading
import time

MAGIC = b"VSTRXPRG"
VERSION = 1
STAGES = ("starting", "loading", "idle", "request", "prefill", "decode", "sync", "cleanup", "stopping")
STAGE_ID = {name: index for index, name in enumerate(STAGES)}
PHASE_STAGE = {"init": "idle", "idle": "idle", "prefill": "prefill", "decode": "decode"}      # trace_phase names

# magic, version, lock(seqlock), pid, start_sec, start_usec, stage, request_seq, counter, entered_ns, progress_ns,
# tokens, response_started, detail
LAYOUT = struct.Struct("<8sIIIQQIQQQQQI96s")
SIZE = 4096

FLAG = "FLASHNEXT_WATCHDOG"
DIR_FLAG = "FLASHNEXT_WATCHDOG_DIR"


# -- process identity (never act on a reused pid) ------------------------------------------------------------------


class _BsdInfo(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("status", ctypes.c_uint32), ("xstatus", ctypes.c_uint32),
                ("pid", ctypes.c_uint32), ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32),
                ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32), ("rgid", ctypes.c_uint32),
                ("svuid", ctypes.c_uint32), ("svgid", ctypes.c_uint32), ("rfu", ctypes.c_uint32),
                ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32), ("nfiles", ctypes.c_uint32),
                ("pgid", ctypes.c_uint32), ("pjobc", ctypes.c_uint32), ("tdev", ctypes.c_uint32),
                ("tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32), ("start_sec", ctypes.c_uint64),
                ("start_usec", ctypes.c_uint64)]


_LIBPROC = None


def _libproc():
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
    return _LIBPROC


def process_identity(pid: int):
    """``(start_sec, start_usec)`` of a live process, or None when no such process exists. A recycled pid differs."""
    info = _BsdInfo()
    got = _libproc().proc_pidinfo(int(pid), 3, 0, ctypes.byref(info), ctypes.sizeof(info))   # PROC_PIDTBSDINFO
    if got != ctypes.sizeof(info) or info.pid != int(pid):
        return None
    return (int(info.start_sec), int(info.start_usec))


# -- the record ---------------------------------------------------------------------------------------------------


class Record:
    """Single-writer seqlock record in a private file. Readers retry while the lock is odd or changes."""

    def __init__(self, path: str, create: bool = False):
        self.path = path
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self._fd = os.open(path, flags, 0o600)
        if create:
            os.ftruncate(self._fd, SIZE)
        self._mm = mmap.mmap(self._fd, SIZE)
        self._lock = threading.Lock()
        self._seq = 0

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            os.close(self._fd)

    def write(self, **values) -> None:
        with self._lock:
            row = self.read(wait=False) or self.blank()
            row.update(values)
            detail = str(row["detail"]).encode("utf-8", "replace")[:95]
            self._seq += 1
            struct.pack_into("<I", self._mm, 12, self._seq)                          # odd: write in progress
            LAYOUT.pack_into(self._mm, 0, MAGIC, VERSION, self._seq, row["pid"], row["start_sec"], row["start_usec"],
                             STAGE_ID[row["stage"]], row["request_seq"], row["counter"], row["entered_ns"],
                             row["progress_ns"], row["tokens"], row["response_started"], detail)
            self._seq += 1
            struct.pack_into("<I", self._mm, 12, self._seq)                          # even: consistent

    @staticmethod
    def blank() -> dict:
        now = time.time_ns()
        return {"pid": 0, "start_sec": 0, "start_usec": 0, "stage": "starting", "request_seq": 0, "counter": 0,
                "entered_ns": now, "progress_ns": now, "tokens": 0, "response_started": 0, "detail": ""}

    def read(self, wait: bool = True):
        for _ in range(200 if wait else 1):
            magic, version, first, pid, sec, usec, stage, req, counter, entered, progress, tokens, started, detail = \
                LAYOUT.unpack_from(self._mm, 0)
            if magic != MAGIC or version != VERSION:
                return None
            if first % 2 == 0 and struct.unpack_from("<I", self._mm, 12)[0] == first:
                return {"pid": pid, "start_sec": sec, "start_usec": usec,
                        "stage": STAGES[stage] if stage < len(STAGES) else "starting", "request_seq": req,
                        "counter": counter, "entered_ns": entered, "progress_ns": progress, "tokens": tokens,
                        "response_started": started, "detail": detail.rstrip(b"\0").decode("utf-8", "replace")}
            time.sleep(0.0005)
        return None


# -- the worker's handle ------------------------------------------------------------------------------------------


class Progress:
    def __init__(self):
        self.record: Record | None = None
        self.channel: socket.socket | None = None
        self._stage = "starting"
        self._detail = ""
        self._request = 0
        self._process = None
        self._frames = None

    # configuration
    def attach(self, record: Record, channel=None) -> None:
        self.record, self.channel = record, channel
        identity = process_identity(os.getpid()) or (0, 0)
        record.write(pid=os.getpid(), start_sec=identity[0], start_usec=identity[1], stage="starting",
                     counter=0, entered_ns=time.time_ns(), progress_ns=time.time_ns())

    def detach(self) -> None:
        record, self.record = self.record, None
        if record is not None:
            with contextlib.suppress(Exception):
                record.write(stage="stopping")
            record.close()

    @property
    def active(self) -> bool:
        return self.record is not None

    # stages and progress
    def stage(self, name: str, detail: str = "") -> None:
        self._stage, self._detail = name, detail
        record = self.record
        if record is not None:
            now = time.time_ns()
            record.write(stage=name, detail=detail, entered_ns=now, progress_ns=now)

    def tick(self, tokens: int | None = None) -> None:
        record = self.record
        if record is not None:
            row = record.read(wait=False)
            if row is not None:
                values = {"counter": row["counter"] + 1, "progress_ns": time.time_ns()}
                if tokens is not None:
                    values["tokens"] = tokens
                record.write(**values)

    @contextlib.contextmanager
    def waiting(self, name: str, detail: str = ""):
        """A bounded native wait (for example the GPU barrier): its own stage and deadline, then back to where we were."""
        before = (self._stage, self._detail)
        self.stage(name, detail)
        try:
            yield
        finally:
            self.stage(*before)

    # requests and clients
    def request_begin(self, connection=None, path: str = "") -> int:
        self._request += 1
        record = self.record
        if record is not None:
            now = time.time_ns()
            record.write(stage="request", request_seq=self._request, response_started=0, tokens=0,
                         entered_ns=now, progress_ns=now, detail="")
            self._stage, self._detail = "request", ""
            if connection is not None and self.channel is not None:
                with contextlib.suppress(Exception):
                    socket.send_fds(self.channel, [json.dumps({"op": "begin", "seq": self._request, "path": path}).encode()],
                                    [connection.fileno()])
        return self._request

    def request_end(self) -> None:
        if self.record is not None:
            if self.channel is not None:
                with contextlib.suppress(Exception):
                    self.channel.send(json.dumps({"op": "end", "seq": self._request}).encode())
            self.stage("idle")

    def response_started(self) -> None:
        if self.record is not None:
            self.record.write(response_started=1)


PROGRESS = Progress()


def enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(FLAG, "0") == "1"


def status_dir(environ=None) -> str:
    environ = os.environ if environ is None else environ
    return os.path.expanduser(environ.get(DIR_FLAG) or "~/.flashnext/watchdog")


def start_watchdog(trace_path: str | None = None, environ=None, python: str | None = None):
    """Create the record and spawn the watchdog in its own session. Returns the Popen, or None when not enabled."""
    if not enabled(environ):
        return None
    directory = status_dir(environ)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    record_path = os.path.join(directory, f"progress-{os.getpid()}.bin")
    record = Record(record_path, create=True)
    parent_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    PROGRESS.attach(record, parent_end)
    import faulthandler
    import signal

    frames = open(os.path.join(directory, f"pyframes-{os.getpid()}.txt"), "ab", buffering=0)
    faulthandler.register(signal.SIGUSR1, file=frames, all_threads=True)     # C-level: works with the GIL held
    PROGRESS._frames = frames
    identity = process_identity(os.getpid())
    command = [python or sys.executable, "-m", "models.flashnext.watchdog", "--pid", str(os.getpid()),
               "--start", f"{identity[0]}.{identity[1]}", "--record", record_path, "--channel-fd", str(child_end.fileno()),
               "--dir", directory]
    if trace_path:
        command += ["--trace", trace_path]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ if environ is None else environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(command, pass_fds=[child_end.fileno()], start_new_session=True, cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=open(os.path.join(directory, f"watchdog-{os.getpid()}.log"), "ab"))
    child_end.close()
    PROGRESS._process = process
    return process
