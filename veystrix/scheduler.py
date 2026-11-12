"""One active inference at a time, a short bounded queue behind it, and a clear busy answer when that is full.

The model holds one live conversation and one set of GPU buffers, so two requests must never run together. HTTP
connections are handled by threads (so a health probe answers while a request generates, and a second client gets an
honest "busy" instead of a silent wait in the TCP backlog), but every inference job runs on the thread that called
``serve_jobs`` (the main thread in production). That keeps the inference thread, and everything bound to it (the guards,
the GPU stream, the watchdog's record), exactly where it was before. When no job loop is running (unit tests), the job runs
inline in the handler thread under the same one-at-a-time gate.

A waiting request is dropped, never run, when its client disconnects or its wait expires. A job that already started is
never interrupted here; cancelling a running generation is the caller's business (it can raise from its own token callback).
"""
from __future__ import annotations

import queue
import threading
import time

POLL_SECONDS = 0.25
DEFAULT_MAX_WAIT = 20.0
DEFAULT_MAX_WAITING = 4
PRIORITY_AGING_SECONDS = 1.0


class Busy(Exception):
    """No slot became available in time (or the queue is full). Maps to HTTP 503 with Retry-After."""

    def __init__(self, message: str, retry_after: int = 5):
        super().__init__(message)
        self.retry_after = retry_after


class ClientGone(Exception):
    """The client closed its connection before (or while) its request ran."""


class _Ticket:
    def __init__(self, fn, path):
        self.fn, self.path = fn, path
        self.started = threading.Event()
        self.done = threading.Event()
        self.cancelled = False
        self.result = None
        self.error: BaseException | None = None


class Scheduler:
    def __init__(self, max_wait: float = DEFAULT_MAX_WAIT, max_waiting: int = DEFAULT_MAX_WAITING, clock=time.monotonic):
        self.max_wait, self.max_waiting, self._clock = float(max_wait), int(max_waiting), clock
        self._lock = threading.Lock()
        self._gate = threading.Lock()              # inline mode: the single inference slot
        self._jobs: queue.PriorityQueue[tuple[float, int, _Ticket]] = queue.PriorityQueue()
        self._sequence = 0
        self.loop_running = False
        self.active: dict | None = None
        self.waiting = 0
        self.completed = self.rejected = self.dropped = 0

    # -- state for /health ------------------------------------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            active = dict(self.active) if self.active else None
            if active:
                active["seconds"] = round(self._clock() - active.pop("since"), 1)
            return {"active": active, "waiting": self.waiting, "max_waiting": self.max_waiting,
                    "max_wait_seconds": self.max_wait, "completed": self.completed, "rejected": self.rejected,
                    "dropped": self.dropped}

    @property
    def busy(self) -> bool:
        with self._lock:
            return self.active is not None

    # -- handler side -----------------------------------------------------------------------------------------------

    def run(self, fn, gone=lambda: False, path: str = "", *, wait=None, priority: int = 0):
        """Run ``fn`` once a slot is free. Queued callers may choose an integer priority from 0 to 10."""
        if not isinstance(priority, int) or not 0 <= priority <= 10:
            raise ValueError("scheduler priority must be an integer from 0 to 10")
        ticket = _Ticket(fn, path)
        with self._lock:
            enqueued = self._clock()
            deadline = enqueued + self.max_wait
            if self.waiting >= self.max_waiting:
                self.rejected += 1
                raise Busy(f"the server is handling another request and {self.waiting} are already waiting; retry shortly")
            self.waiting += 1
            queued = self.loop_running
            if queued:
                self._sequence += 1
                # All waiting jobs age at the same rate, so their relative ranks are fixed.
                self._jobs.put((enqueued - priority * PRIORITY_AGING_SECONDS, self._sequence, ticket))
        if queued:
            while not ticket.started.wait(POLL_SECONDS):
                problem = self._expired(ticket, gone, deadline)
                if problem is not None:
                    raise problem
            if wait is not None:
                wait(ticket.done)
            ticket.done.wait()
            if ticket.error is not None:
                raise ticket.error
            return ticket.result
        while not self._gate.acquire(timeout=POLL_SECONDS):
            problem = self._expired(ticket, gone, deadline)
            if problem is not None:
                raise problem
        try:
            if not self._claim(ticket):
                raise ClientGone("cancelled while waiting")
            try:
                return fn()
            finally:
                self._finish()
        finally:
            self._gate.release()
            if wait is not None:
                done = threading.Event()
                done.set()
                wait(done)

    def _expired(self, ticket, gone, deadline):
        """Called while waiting. Returns the exception to raise if the wait must end, else None."""
        if gone():
            reason = ClientGone("client disconnected while waiting")
        elif self._clock() >= deadline:
            reason = Busy(f"waited {self.max_wait:g}s for the active request to finish; retry shortly")
        else:
            return None
        with self._lock:
            if ticket.started.is_set() or ticket.cancelled:
                return None                            # claimed (or failed for shutdown) just now: the executor owns it
            ticket.cancelled = True
            self.waiting -= 1
            self.dropped += isinstance(reason, ClientGone)
            self.rejected += isinstance(reason, Busy)
        return reason

    # -- executor side ----------------------------------------------------------------------------------------------

    def _claim(self, ticket) -> bool:
        with self._lock:
            if ticket.cancelled:
                return False
            ticket.started.set()
            self.waiting -= 1
            self.active = {"path": ticket.path, "since": self._clock()}
            return True

    def _finish(self) -> None:
        with self._lock:
            self.active = None
            self.completed += 1

    def serve_jobs(self, stop: threading.Event) -> None:
        """Run queued jobs on the calling thread until ``stop`` is set. Exceptions of a job go back to its handler."""
        with self._lock:
            self.loop_running = True
        try:
            while not stop.is_set():
                try:
                    _, _, ticket = self._jobs.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                if not self._claim(ticket):
                    continue
                try:
                    ticket.result = ticket.fn()
                except KeyboardInterrupt:
                    ticket.error = Busy("the server is stopping", retry_after=30)
                    raise
                except BaseException as error:         # handed to the handler thread, which owns the response
                    ticket.error = error
                finally:
                    self._finish()
                    ticket.done.set()
        finally:
            with self._lock:
                self.loop_running = False
            self._fail_pending()

    def _fail_pending(self) -> None:
        while True:
            try:
                _, _, ticket = self._jobs.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                if ticket.cancelled:
                    continue                           # its handler already gave up and fixed the count
                ticket.cancelled = True
                self.waiting -= 1
            ticket.error = Busy("the server is stopping", retry_after=30)
            ticket.started.set()
            ticket.done.set()
