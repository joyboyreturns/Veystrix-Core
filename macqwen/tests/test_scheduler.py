"""The inference scheduler: one active job, a bounded FIFO queue, bounded waits, cancellation while waiting."""
import threading
import time
import unittest

from macqwen.scheduler import Busy, ClientGone, Scheduler


def start(target):
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def wait_until(check, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return False


class Harness:
    """Runs submissions on threads and records outcomes. ``loop`` selects job-loop mode (as in production)."""

    def __init__(self, loop, **options):
        self.scheduler = Scheduler(**options)
        self.stop = threading.Event()
        self.outcomes = {}
        self.order = []
        self.threads = {}
        self.executor = None
        if loop:
            self.executor = start(lambda: self.scheduler.serve_jobs(self.stop))
            assert wait_until(lambda: self.scheduler.loop_running)

    def submit(self, name, work=None, gone=lambda: False):
        def job():
            self.order.append(name)
            self.threads[name] = threading.current_thread().name
            if work is not None:
                work()
            return name

        def body():
            try:
                self.outcomes[name] = ("ok", self.scheduler.run(job, gone=gone, path="/x"))
            except BaseException as error:
                self.outcomes[name] = (type(error).__name__, error)

        return start(body)

    def close(self):
        self.stop.set()
        if self.executor is not None:
            self.executor.join(2)


class SchedulerTests(unittest.TestCase):
    def check_modes(self, check):
        for loop in (True, False):
            with self.subTest(loop=loop):
                harness = Harness(loop, max_wait=2.0, max_waiting=2)
                self.addCleanup(harness.close)
                check(harness)

    def test_only_one_job_runs_at_a_time_and_the_rest_run_in_arrival_order(self):
        def check(h):
            release, running = threading.Event(), []
            overlap = []

            def work():
                running.append(1)
                overlap.append(len(running))
                release.wait(5)
                running.pop()

            h.submit("a", work)
            self.assertTrue(wait_until(lambda: h.scheduler.busy))
            h.submit("b", work)
            self.assertTrue(wait_until(lambda: h.scheduler.snapshot()["waiting"] == 1))
            h.submit("c", work)
            self.assertTrue(wait_until(lambda: h.scheduler.snapshot()["waiting"] == 2))
            release.set()
            self.assertTrue(wait_until(lambda: len(h.outcomes) == 3))
            self.assertEqual(max(overlap), 1)
            if h.scheduler.loop_running:                 # inline mode is a plain lock: no arrival-order promise
                self.assertEqual(h.order, ["a", "b", "c"])
            self.assertEqual(sorted(h.order), ["a", "b", "c"])
            self.assertTrue(all(kind == "ok" for kind, _ in h.outcomes.values()))
            self.assertEqual(h.scheduler.snapshot()["waiting"], 0)

        self.check_modes(check)

    def test_a_full_queue_is_refused_at_once(self):
        def check(h):
            release = threading.Event()
            h.submit("a", lambda: release.wait(5))
            self.assertTrue(wait_until(lambda: h.scheduler.busy))
            h.submit("b")
            h.submit("c")
            self.assertTrue(wait_until(lambda: h.scheduler.snapshot()["waiting"] == 2))
            began = time.monotonic()
            h.submit("d")
            self.assertTrue(wait_until(lambda: "d" in h.outcomes))
            self.assertLess(time.monotonic() - began, 1.0)
            self.assertEqual(h.outcomes["d"][0], "Busy")
            self.assertEqual(h.scheduler.snapshot()["rejected"], 1)
            release.set()
            self.assertTrue(wait_until(lambda: len(h.outcomes) == 4))
            self.assertNotIn("d", h.order)

        self.check_modes(check)

    def test_a_wait_that_runs_out_is_busy_and_the_job_never_runs(self):
        for loop in (True, False):
            with self.subTest(loop=loop):
                h = Harness(loop, max_wait=0.4, max_waiting=2)
                self.addCleanup(h.close)
                release = threading.Event()
                h.submit("a", lambda: release.wait(5))
                self.assertTrue(wait_until(lambda: h.scheduler.busy))
                h.submit("b")
                self.assertTrue(wait_until(lambda: "b" in h.outcomes, 3))
                self.assertEqual(h.outcomes["b"][0], "Busy")
                self.assertGreaterEqual(h.outcomes["b"][1].retry_after, 1)
                release.set()
                self.assertTrue(wait_until(lambda: "a" in h.outcomes))
                time.sleep(0.5)                          # a cancelled ticket must not run later
                self.assertEqual(h.order, ["a"])
                self.assertEqual(h.scheduler.snapshot()["waiting"], 0)

    def test_a_client_that_leaves_while_waiting_is_dropped_without_running(self):
        def check(h):
            release, gone = threading.Event(), threading.Event()
            h.submit("a", lambda: release.wait(5))
            self.assertTrue(wait_until(lambda: h.scheduler.busy))
            h.submit("b", gone=gone.is_set)
            self.assertTrue(wait_until(lambda: h.scheduler.snapshot()["waiting"] == 1))
            gone.set()
            self.assertTrue(wait_until(lambda: "b" in h.outcomes, 3))
            self.assertEqual(h.outcomes["b"][0], "ClientGone")
            release.set()
            self.assertTrue(wait_until(lambda: "a" in h.outcomes))
            time.sleep(0.4)
            self.assertEqual(h.order, ["a"])
            snap = h.scheduler.snapshot()
            self.assertEqual((snap["dropped"], snap["waiting"]), (1, 0))

        self.check_modes(check)

    def test_jobs_run_on_the_executor_thread_not_the_handler_thread(self):
        h = Harness(True, max_wait=2.0)
        self.addCleanup(h.close)
        h.submit("a")
        self.assertTrue(wait_until(lambda: "a" in h.outcomes))
        self.assertNotEqual(h.threads["a"], threading.current_thread().name)
        self.assertTrue(h.threads["a"].startswith("Thread-"))        # the thread running serve_jobs, not a handler
        self.assertEqual(h.threads["a"], h.executor.name)

    def test_an_exception_in_a_job_reaches_its_caller_and_frees_the_slot(self):
        def check(h):
            def boom():
                raise ValueError("model failed")

            h.submit("a", boom)
            self.assertTrue(wait_until(lambda: "a" in h.outcomes))
            self.assertEqual(h.outcomes["a"][0], "ValueError")
            h.submit("b")
            self.assertTrue(wait_until(lambda: "b" in h.outcomes))
            self.assertEqual(h.outcomes["b"][0], "ok")
            self.assertFalse(h.scheduler.busy)

        self.check_modes(check)

    def test_stopping_the_executor_fails_waiting_requests_with_busy(self):
        h = Harness(True, max_wait=5.0, max_waiting=3)
        release = threading.Event()
        h.submit("a", lambda: release.wait(5))
        self.assertTrue(wait_until(lambda: h.scheduler.busy))
        h.submit("b")
        self.assertTrue(wait_until(lambda: h.scheduler.snapshot()["waiting"] == 1))
        release.set()
        h.stop.set()
        h.executor.join(3)
        self.assertTrue(wait_until(lambda: "b" in h.outcomes, 3))
        self.assertIn(h.outcomes["b"][0], ("Busy", "ok"))            # ran before the stop, or told to retry
        self.assertEqual(h.scheduler.snapshot()["waiting"], 0)

    def test_snapshot_reports_the_active_request(self):
        h = Harness(True, max_wait=2.0)
        self.addCleanup(h.close)
        release = threading.Event()
        h.submit("a", lambda: release.wait(5))
        self.assertTrue(wait_until(lambda: h.scheduler.busy))
        snap = h.scheduler.snapshot()
        self.assertEqual(snap["active"]["path"], "/x")
        self.assertGreaterEqual(snap["active"]["seconds"], 0)
        release.set()
        self.assertTrue(wait_until(lambda: not h.scheduler.busy))
        self.assertEqual(h.scheduler.snapshot()["completed"], 1)


if __name__ == "__main__":
    unittest.main()
