"""Conditional, automatic macOS file-cache recovery (opt-in: ``FLASHNEXT_AUTO_PURGE=1``).

The goal is a model that can start and keep running, not a file cache forced to zero. Five seconds
(``FLASHNEXT_PURGE_CHECK_SECONDS``) is how often the machine is *looked at*; a purge only happens when every
condition below holds, and every outcome is recorded so an ineffective purge teaches the next decision.

A purge runs only if ALL of these are true:

* Nothing of ours is active: no model load, no request (prefill, decode, GPU work and expert reads all happen
  inside a request), no other purge. This is a hard gate (``Activity``), not a heuristic; a request that arrives
  while a purge runs waits for it, so a purge never overlaps inference.
* Available memory (``free + inactive + speculative``, the same quantity ``ChatGuard`` gates on) is below what the
  caller needs: the 9 GiB start gate at startup, ``FLASHNEXT_PURGE_IDLE_HEADROOM_GB`` (default 5.5 GB, the memory
  manager's constrained threshold) while idle.
* A purge could close the gap: the file-backed pages not already counted as available are at least the shortfall
  and at least 256 MiB (upper bound, see ``accounting``).
* No swap-out is in progress, pressure is not critical, and ``sudo -n`` is permitted.
* Veystrix-owned memory was reclaimed first (allocator cache; RAM prefix snapshots only when pressure demands it).
* The backoff allows it: after an ineffective purge the wait doubles (30 s up to 30 min); even after an effective one
  at least 60 s pass, so a refill-then-purge loop cannot run tight. There is no daily count limit.

Only ``sudo -n /usr/sbin/purge`` is ever executed, with no terminal input, no shell and a timeout. If sudo would need
a password the feature records that and backs off; it never asks for credentials and never edits sudoers.

Accounting. ``vm_stat`` identities on this machine: ``file_backed + anonymous == active + inactive + speculative``
(exact), so the gate's ``free + inactive + speculative`` counts nothing twice. It is a queue measure, not a cache
measure: it excludes *active* file-backed pages (recent cache, such as the checkpoint's own shard reads) and includes
*inactive anonymous* pages (application memory that must be compressed before reuse). Activity Monitor's "Cached
Files" is ``file_backed + purgeable``, so it can read several GB while the gate reads low. A purge frees the file
cache, which raises the gate figure by the active file-backed pages only. ``accounting`` reports both bounds.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

FLAG = "FLASHNEXT_AUTO_PURGE"
FLAG_INTERVAL = "FLASHNEXT_PURGE_CHECK_SECONDS"
FLAG_STARTUP_WAIT = "FLASHNEXT_PURGE_STARTUP_WAIT_SECONDS"
FLAG_IDLE_HEADROOM = "FLASHNEXT_PURGE_IDLE_HEADROOM_GB"
FLAG_LOG = "FLASHNEXT_PURGE_LOG"

GIB = 2 ** 30
PURGE_COMMAND = ("sudo", "-n", "/usr/sbin/purge")
CAPABILITY_COMMAND = ("sudo", "-n", "-l", "/usr/sbin/purge")   # lists the permission, runs nothing


def enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(FLAG, "0") == "1"


def _number(name, default, environ=None):
    raw = (os.environ if environ is None else environ).get(name)
    try:
        value = float(raw) if raw not in (None, "") else float(default)
    except ValueError:
        return float(default)
    return value if value > 0 else float(default)


@dataclass(frozen=True)
class Config:
    interval_s: float = 5.0
    start_gate_bytes: int = 9 * GIB
    start_margin_bytes: int = GIB // 8          # aim above the gate so the guard's own read does not race it
    idle_headroom_bytes: int = int(5.5e9)
    min_useful_bytes: int = 256 * 2 ** 20
    min_interval_effective_s: float = 60.0
    backoff_base_s: float = 30.0
    backoff_max_s: float = 1800.0
    settle_s: float = 1.0
    purge_timeout_s: float = 30.0
    permission_recheck_s: float = 600.0
    startup_wait_s: float = 90.0
    pressure_skip_level: int = 4

    @classmethod
    def from_environment(cls, environ=None):
        return cls(interval_s=_number(FLAG_INTERVAL, 5.0, environ),
                   idle_headroom_bytes=int(_number(FLAG_IDLE_HEADROOM, 5.5, environ) * 1e9),
                   startup_wait_s=_number(FLAG_STARTUP_WAIT, 90.0, environ))


# -- machine reading and accounting ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    page: int = 16384
    free: int = 0
    active: int = 0
    inactive: int = 0
    speculative: int = 0
    wired: int = 0
    compressor: int = 0
    file_backed: int = 0
    anonymous: int = 0
    purgeable: int = 0
    swapouts: int = 0          # cumulative pages
    swapins: int = 0
    pressure: int = 1
    swap_used: int = 0

    @property
    def available(self) -> int:
        """Bytes ``ChatGuard`` calls available: free + inactive + speculative."""
        return (self.free + self.inactive + self.speculative) * self.page


_LABELS = {"Pages free": "free", "Pages active": "active", "Pages inactive": "inactive",
           "Pages speculative": "speculative", "Pages wired down": "wired",
           "Pages occupied by compressor": "compressor", "File-backed pages": "file_backed",
           "Anonymous pages": "anonymous", "Pages purgeable": "purgeable", "Swapouts": "swapouts",
           "Swapins": "swapins"}


def parse_vm_stat(text: str, pressure: int = 1, swap_used: int = 0) -> Reading:
    page = re.search(r"page size of (\d+) bytes", text)
    values = {}
    for line in text.splitlines():
        label, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        key = _LABELS.get(label.strip().strip('"'))
        if key and value.isdigit():
            values[key] = int(value)
    return Reading(page=int(page.group(1)) if page else 16384, pressure=pressure, swap_used=swap_used, **values)


def read_system() -> Reading:
    def run(command):
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL).stdout
        except Exception:
            return ""
    sysctls = run(["sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level"])
    swap = re.search(r"used = ([\d.]+)M", sysctls)
    pressure = re.search(r"kern\.memorystatus_vm_pressure_level:\s*(\d+)", sysctls)
    return parse_vm_stat(run(["vm_stat"]), int(pressure.group(1)) if pressure else 1,
                         int(float(swap.group(1)) * 2 ** 20) if swap else 0)


def accounting(reading: Reading) -> dict:
    """Validate the accounting and bound what a purge can recover (all values in bytes)."""
    page = reading.page
    queues = reading.active + reading.inactive + reading.speculative
    cached = reading.file_backed - reading.speculative          # file-backed pages not already in the gate figure
    return {
        "gate_available": reading.available,
        "file_plus_anonymous_equals_queues": reading.file_backed + reading.anonymous == queues,
        "activity_monitor_cached_files": (reading.file_backed + reading.purgeable) * page,
        # Upper bound: no inactive page is file-backed. Lower bound: every inactive page is.
        "purge_gain_upper": max(0, cached) * page,
        "purge_gain_lower": max(0, cached - reading.inactive) * page,
        "inactive_anonymous_upper": min(reading.inactive, reading.anonymous) * page,
        "compressor": reading.compressor * page,
    }


# -- the activity gate -------------------------------------------------------------------------------------------


class Activity:
    """Process-wide record of what the engine is doing. A purge may start only when it is idle."""

    def __init__(self):
        self._cv = threading.Condition()
        self.requests = 0
        self.loading = 0
        self.purging = False

    @contextmanager
    def request(self, wait_s: float = 120.0):
        """Wrap a whole generate call (prefill, decode, GPU work, expert reads, cleanup). Waits out a running purge."""
        with self._cv:
            self._cv.wait_for(lambda: not self.purging, timeout=wait_s)
            self.requests += 1
        try:
            yield
        finally:
            with self._cv:
                self.requests -= 1
                self._cv.notify_all()

    @contextmanager
    def loading_model(self, wait_s: float = 120.0):
        with self._cv:
            self._cv.wait_for(lambda: not self.purging, timeout=wait_s)
            self.loading += 1
        try:
            yield
        finally:
            with self._cv:
                self.loading -= 1
                self._cv.notify_all()

    def busy_reason(self):
        with self._cv:
            return self._reason()

    def _reason(self):
        if self.loading:
            return "model is loading"
        if self.requests:
            return "a request is active"
        if self.purging:
            return "a purge is running"
        return None

    def try_begin_purge(self):
        """Atomically claim the purge slot. Returns None on success, else why not."""
        with self._cv:
            reason = self._reason()
            if reason is None:
                self.purging = True
            return reason

    def end_purge(self):
        with self._cv:
            self.purging = False
            self._cv.notify_all()


ACTIVITY = Activity()


# -- the recovery policy -----------------------------------------------------------------------------------------


@dataclass
class PurgeRecord:
    t: float
    trigger: str
    available_before: int
    available_after: int
    recovered: int
    duration_s: float
    effective: bool
    file_backed_before: int
    file_backed_after: int
    owned_reclaimed: int = 0
    returncode: int | None = None
    error: str = ""
    refill_file_bytes: int = 0
    ssd_read_bytes: int = 0


@dataclass
class Outcome:
    action: str                # "ok", "purged", "skipped"
    reason: str
    record: PurgeRecord | None = None


def default_runner(command, timeout):
    """Run exactly the given argv: no shell, no terminal input, bounded time."""
    return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


class PurgeRecovery:
    def __init__(self, cfg: Config = Config(), *, read=read_system, runner=default_runner, clock=time.monotonic,
                 sleep=time.sleep, activity: Activity = ACTIVITY, disk_read=None, log_path=None, reclaim=(), blocker=None):
        self.cfg, self._read, self._runner, self._clock, self._sleep = cfg, read, runner, clock, sleep
        # blocker(reading) -> reason or None: a startup condition that file cache cannot fix (AGENTIC prefill readiness)
        self._blocker = blocker
        self.activity = activity
        self._disk_read = disk_read or _process_disk_read
        self._reclaim = list(reclaim)
        self.log_path = log_path
        self._lock = threading.Lock()
        self.records: list[PurgeRecord] = []
        self.skips: dict[str, int] = {}
        self.consecutive_ineffective = 0
        self.not_before = 0.0
        self.permission: bool | None = None
        self._permission_checked = -1e18
        self._last_swapouts: int | None = None
        self._disk_at_purge = 0
        self.closed = False

    # -- helpers
    def add_reclaim(self, hook) -> None:
        self._reclaim.append(hook)

    def _skip(self, reason) -> Outcome:
        key = re.sub(r"[\d.]+", "#", reason)
        self.skips[key] = self.skips.get(key, 0) + 1
        return Outcome("skipped", reason)

    def check_permission(self, now=None) -> bool:
        """Whether ``sudo -n`` may run purge. Lists the permission; runs no purge and never prompts."""
        now = self._clock() if now is None else now
        if self.permission is None or (not self.permission and now - self._permission_checked >= self.cfg.permission_recheck_s):
            self._permission_checked = now
            try:
                self.permission = self._runner(CAPABILITY_COMMAND, 10).returncode == 0
            except Exception:
                self.permission = False
        return bool(self.permission)

    def backoff_remaining(self, now) -> float:
        return max(0.0, self.not_before - now)

    def _owned_memory(self, reading) -> int:
        freed = 0
        for hook in self._reclaim:
            try:
                freed += int(hook(reading) or 0)
            except Exception:
                pass
        return freed

    def track_refill(self) -> None:
        """Update the last purge's refill cost: file cache regrown and SSD bytes read since."""
        if not self.records:
            return
        record = self.records[-1]
        reading = self._read()
        record.refill_file_bytes = max(record.refill_file_bytes,
                                       reading.file_backed * reading.page - record.file_backed_after)
        record.ssd_read_bytes = max(0, self._disk_read() - self._disk_at_purge)

    # -- decision
    def decide(self, required_bytes: int, trigger: str, reading=None, now=None) -> Outcome:
        now = self._clock() if now is None else now
        reading = reading or self._read()
        swapped = self._last_swapouts is not None and reading.swapouts > self._last_swapouts
        self._last_swapouts = reading.swapouts
        if reading.available >= required_bytes:
            return Outcome("ok", f"available {reading.available / GIB:.2f} GiB meets {required_bytes / GIB:.2f} GiB")
        if reading.pressure >= self.cfg.pressure_skip_level:
            return self._skip(f"memory pressure level {reading.pressure} is critical; not adding purge work")
        if swapped:
            return self._skip("swap-out in progress; a purge cannot help and adds I/O")
        wait = self.backoff_remaining(now)
        if wait > 0:
            return self._skip(f"backoff: {wait:.0f}s remaining after {self.consecutive_ineffective} ineffective purge(s)")
        if self._blocker is not None and trigger == "startup":
            reason = self._blocker(reading)
            if reason:
                # The launch would be refused whatever the file cache holds, so flushing it only costs SSD reads later.
                return self._skip(f"a purge cannot admit this launch: {reason}")
        shortfall = required_bytes - reading.available
        gain = accounting(reading)["purge_gain_upper"]
        if gain < max(shortfall, self.cfg.min_useful_bytes):
            return self._skip(f"a purge cannot close the gap: shortfall {shortfall / 1e6:.0f} MB, at most "
                              f"{gain / 1e6:.0f} MB of file cache is recoverable")
        if not self.check_permission(now):
            return self._skip("sudo -n /usr/sbin/purge is not permitted (no password prompt is ever shown)")
        return Outcome("purge", f"short by {shortfall / 1e6:.0f} MB, up to {gain / 1e6:.0f} MB recoverable")

    # -- action
    def attempt(self, required_bytes: int, trigger: str) -> Outcome:
        """One guarded recovery attempt. Never overlaps another purge or any inference."""
        if self.closed:
            return self._skip("recovery is closed")
        if not self._lock.acquire(blocking=False):
            return self._skip("a purge attempt is already in progress")
        try:
            busy = self.activity.try_begin_purge()
            if busy:
                return self._skip(busy)
            try:
                return self._attempt_locked(required_bytes, trigger)
            finally:
                self.activity.end_purge()
        finally:
            self._lock.release()

    def _attempt_locked(self, required_bytes, trigger) -> Outcome:
        reading = self._read()
        first = self.decide(required_bytes, trigger, reading)
        if first.action != "purge":
            return first
        reclaimed = self._owned_memory(reading)       # Veystrix-owned temporary memory first
        if reclaimed:
            reading = self._read()
            second = self.decide(required_bytes, trigger, reading)
            if second.action != "purge":
                return second
        started = self._clock()
        returncode, error = None, ""
        try:
            result = self._runner(PURGE_COMMAND, self.cfg.purge_timeout_s)
            returncode = result.returncode
            if returncode != 0:
                error = ((result.stderr or "").strip().splitlines() or [f"exit {returncode}"])[-1][:200]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
        duration = self._clock() - started
        if returncode != 0:
            # Permission or execution failure: do not retry in a loop, and look at the permission again later.
            self.permission, self._permission_checked = False, self._clock()
            record = PurgeRecord(self._clock(), trigger, reading.available, reading.available, 0, duration, False,
                                 reading.file_backed * reading.page, reading.file_backed * reading.page,
                                 reclaimed, returncode, error)
            self._finish(record, failed=True)
            return Outcome("skipped", f"purge failed: {error}", record)
        self._sleep(self.cfg.settle_s)
        after = self._read()
        recovered = max(0, after.available - reading.available)
        record = PurgeRecord(self._clock(), trigger, reading.available, after.available, recovered, duration,
                             recovered >= self.cfg.min_useful_bytes, reading.file_backed * reading.page,
                             after.file_backed * after.page, reclaimed, 0, "")
        self._disk_at_purge = self._disk_read()
        self._finish(record)
        return Outcome("purged", f"recovered {recovered / 1e6:.0f} MB in {duration:.1f}s", record)

    def _finish(self, record, failed=False) -> None:
        now = self._clock()
        if self.records:
            self.track_refill()
            self._log({"event": "refill", **_refill(self.records[-1], len(self.records) - 1)})
        self.records.append(record)
        if failed or not record.effective:
            self.consecutive_ineffective += 1
            delay = min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2 ** (self.consecutive_ineffective - 1))
        else:
            self.consecutive_ineffective = 0
            delay = self.cfg.min_interval_effective_s
        self.not_before = now + delay
        self._log({"event": "purge", "index": len(self.records) - 1, "next_allowed_in_s": delay, **asdict(record)})

    def _log(self, row) -> None:
        if not self.log_path:
            return
        try:
            path = os.path.expanduser(self.log_path)
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as handle:
                handle.write(json.dumps({"wall": time.time(), **row}) + "\n")
        except OSError:
            pass

    def summary(self) -> dict:
        if self.records:
            self.track_refill()
        done = [r for r in self.records if r.returncode == 0]
        return {"purges": len(done), "effective": sum(r.effective for r in done),
                "failed": len(self.records) - len(done),
                "recovered_bytes": sum(r.recovered for r in done),
                "seconds": round(sum(r.duration_s for r in done), 2),
                "refill_file_bytes": [r.refill_file_bytes for r in done],
                "ssd_read_bytes_after_last": self.records[-1].ssd_read_bytes if self.records else 0,
                "consecutive_ineffective": self.consecutive_ineffective, "skips": dict(self.skips),
                "permission": self.permission}

    def close(self) -> None:
        self.closed = True
        if self.records:
            self.track_refill()
            self._log({"event": "refill", **_refill(self.records[-1], len(self.records) - 1)})
        self._log({"event": "summary", **self.summary()})


def _refill(record, index) -> dict:
    return {"index": index, "refill_file_bytes": record.refill_file_bytes, "ssd_read_bytes": record.ssd_read_bytes}


def _process_disk_read() -> int:
    try:
        from .diskio import disk_bytes_read
        return max(0, disk_bytes_read())
    except Exception:
        return 0


# -- startup and idle lifecycles ---------------------------------------------------------------------------------


@dataclass
class StartupResult:
    ready: bool
    waited_s: float
    purges: int
    reason: str
    available: int = 0


def ensure_startup(recovery: PurgeRecovery, out=None) -> StartupResult:
    """Block (bounded) until the start gate plus a small margin is met, purging only when justified.

    Returning not-ready is not an error: the guard that follows applies its own, unchanged, 9 GiB rule."""
    cfg = recovery.cfg
    required = cfg.start_gate_bytes + cfg.start_margin_bytes
    began = recovery._clock()
    purges = 0
    while True:
        outcome = recovery.attempt(required, "startup")
        if outcome.action == "purged":
            purges += 1
        reading = recovery._read()
        waited = recovery._clock() - began
        if reading.available >= required:
            return StartupResult(True, waited, purges, outcome.reason, reading.available)
        if out:
            out(f"auto-purge: {reading.available / GIB:.2f} GiB available, need {required / GIB:.2f} GiB: {outcome.reason}")
        if outcome.action == "skipped" and outcome.reason.startswith("a purge cannot admit this launch"):
            return StartupResult(False, waited, purges, outcome.reason, reading.available)   # waiting cannot fix it
        if waited >= cfg.startup_wait_s or recovery.closed:
            return StartupResult(False, waited, purges, outcome.reason, reading.available)
        recovery._sleep(cfg.interval_s)


class Monitor:
    """Looks at the machine every ``interval_s`` while the engine is idle and recovers only when justified."""

    def __init__(self, recovery: PurgeRecovery):
        self.recovery = recovery
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="veystrix-purge-monitor", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def tick(self) -> Outcome | None:
        recovery = self.recovery
        if recovery.activity.busy_reason():
            return None                                  # never even look while loading or serving
        recovery.track_refill() if recovery.records else None
        return recovery.attempt(recovery.cfg.idle_headroom_bytes, "idle")

    def _run(self):
        while not self._stop.wait(self.recovery.cfg.interval_s):
            try:
                self.tick()
            except Exception:
                pass

    def stop(self, timeout: float = 45.0):
        """Stop watching; an in-flight purge is allowed to finish (it is one short, bounded process)."""
        self._stop.set()
        self.recovery.closed = True
        if self._thread.is_alive():
            self._thread.join(timeout)
        self.recovery.close()


# -- recovery at the request boundary ----------------------------------------------------------------------------

_REGISTERED: list = [None]


def register(recovery: PurgeRecovery | None) -> None:
    """The process's recovery policy, used by ``prepare_for_request`` (set when the idle monitor starts)."""
    _REGISTERED[0] = recovery


def request_required_bytes(cfg: Config, prefill_tokens: int) -> int:
    """Available memory a request should start with: the idle headroom plus what its largest prefill chunk adds
    (ESTIMATE from the budget model; the unchanged runtime guards still decide during the request)."""
    from .prefill_budget import chunk_increment_bytes

    chunk = min(max(int(prefill_tokens), 0), 1024)
    return int(cfg.idle_headroom_bytes + (chunk_increment_bytes(chunk, 0) if chunk else 0))


def prepare_for_request(prefill_tokens: int) -> Outcome | None:
    """Recover memory immediately BEFORE a request starts, when RAM is short for it.

    This is a safe boundary by construction: it runs before the caller enters ``ACTIVITY.request()``, and ``attempt``
    claims the purge slot only when no request, load or other purge is active, so a purge never overlaps inference, prefill,
    decode, GPU work or the cleanup after them. It adds no new policy: the same decision rule applies (skip when file cache
    cannot close the gap, when swap-out is in progress, under critical pressure, or during the backoff after an ineffective
    purge), so it cannot loop. Returns None when no recovery is registered. Never raises."""
    recovery = _REGISTERED[0]
    if recovery is None or recovery.closed:
        return None
    try:
        return recovery.attempt(request_required_bytes(recovery.cfg, prefill_tokens), "pre-request")
    except Exception as error:                      # recovery must never stop a request from running
        return Outcome("skipped", f"recovery failed: {type(error).__name__}: {error}")


def from_environment(environ=None, **overrides) -> PurgeRecovery | None:
    environ = os.environ if environ is None else environ
    if not enabled(environ):
        return None
    log = environ.get(FLAG_LOG) or "~/.flashnext/purge-log.jsonl"
    return PurgeRecovery(Config.from_environment(environ), log_path=log, **overrides)


def main(argv=None) -> int:
    """``python -m models.flashnext.purge_recovery [--recover]``: show the accounting and the decision.

    Without ``--recover`` nothing is purged. With it, the startup recovery runs once (purging only if justified)."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args(argv)
    recovery = PurgeRecovery(Config.from_environment())
    reading = recovery._read()
    info = accounting(reading)
    gb = lambda value: f"{value / 2 ** 30:6.2f} GiB ({value:,} bytes)"          # noqa: E731
    required = recovery.cfg.start_gate_bytes + recovery.cfg.start_margin_bytes
    print(f"available (gate quantity)   {gb(info['gate_available'])}   need {gb(required)}")
    print(f"Activity Monitor cached     {gb(info['activity_monitor_cached_files'])}")
    print(f"purge could add             between {gb(info['purge_gain_lower'])} and {gb(info['purge_gain_upper'])}")
    print(f"file + anonymous == queues  {info['file_plus_anonymous_equals_queues']}   pressure {reading.pressure}   "
          f"swap used {reading.swap_used / 2 ** 20:.0f} MiB")
    print(f"sudo -n purge permitted     {recovery.check_permission()}   auto-purge flag {'ON' if enabled() else 'off'}")
    decision = recovery.decide(required, "status", reading)
    print(f"decision                    {decision.action}: {decision.reason}")
    if args.recover:
        result = ensure_startup(recovery, out=print)
        print(f"recover: ready={result.ready} purges={result.purges} waited={result.waited_s:.0f}s "
              f"available={gb(result.available)}")
        print(json.dumps(recovery.summary(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
