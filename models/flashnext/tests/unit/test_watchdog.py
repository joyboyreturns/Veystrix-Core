"""Out-of-process watchdog: the decision rule, process identity, client notice, and real hung child processes.

Every subprocess scenario has its own timeout, so a watchdog failure cannot freeze the suite.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock
from pathlib import Path

from models.flashnext import progress as P
from models.flashnext import watchdog as W

ROOT = Path(__file__).resolve().parents[4]
NS = 10 ** 9


def row(stage="decode", age=0.0, entered_age=None, now=1000.0, detail="", counter=1, **extra):
    base = {"pid": 1, "start_sec": 1, "start_usec": 1, "stage": stage, "request_seq": 1, "counter": counter,
            "entered_ns": int((now - (age if entered_age is None else entered_age)) * NS),
            "progress_ns": int((now - age) * NS), "tokens": 0, "response_started": 0, "detail": detail}
    base.update(extra)
    return base


class RecordTests(unittest.TestCase):
    def test_round_trip_and_stage_names(self):
        with tempfile.TemporaryDirectory() as directory:
            record = P.Record(os.path.join(directory, "r.bin"), create=True)
            record.write(pid=7, start_sec=11, start_usec=12, stage="prefill", detail="layer 3", counter=5, tokens=9,
                         request_seq=2, response_started=1)
            got = record.read()
            self.assertEqual((got["pid"], got["start_sec"], got["stage"], got["detail"], got["counter"], got["tokens"],
                              got["request_seq"], got["response_started"]),
                             (7, 11, "prefill", "layer 3", 5, 9, 2, 1))
            self.assertEqual(oct(os.stat(record.path).st_mode & 0o777), "0o600")
            record.write(detail="x" * 500)
            self.assertEqual(len(record.read()["detail"]), 95)
            record.close()

    def test_a_reader_never_returns_a_torn_row(self):
        with tempfile.TemporaryDirectory() as directory:
            record = P.Record(os.path.join(directory, "r.bin"), create=True)
            record.write(pid=1, stage="decode")
            reader = P.Record(record.path)
            import threading

            stop = threading.Event()

            def writer():
                number = 0
                while not stop.is_set():
                    number += 1
                    record.write(counter=number, tokens=number)

            thread = threading.Thread(target=writer)
            thread.start()
            try:
                for _ in range(3000):
                    got = reader.read()
                    self.assertIsNotNone(got)
                    self.assertEqual(got["counter"], got["tokens"])          # written together
            finally:
                stop.set()
                thread.join()

    def test_a_foreign_file_is_not_a_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "x.bin")
            Path(path).write_bytes(b"\0" * P.SIZE)
            self.assertIsNone(P.Record(path).read())

    def test_with_no_record_every_call_is_a_no_op(self):
        progress = P.Progress()
        progress.stage("decode")
        progress.tick(3)
        with progress.waiting("sync"):
            pass
        self.assertEqual(progress.request_begin(None), 1)
        progress.request_end()
        progress.response_started()
        self.assertFalse(progress.active)


class IdentityTests(unittest.TestCase):
    def test_a_live_process_has_a_stable_identity_and_a_missing_one_has_none(self):
        mine = P.process_identity(os.getpid())
        self.assertIsNotNone(mine)
        self.assertEqual(mine, P.process_identity(os.getpid()))
        self.assertIsNone(P.process_identity(2 ** 22 - 3))

    def test_a_signal_never_reaches_a_process_with_a_different_identity(self):
        bystander = subprocess.Popen(["/bin/sleep", "30"])
        try:
            responder = W.Responder(bystander.pid, (1, 1), W.Config(), tempfile.gettempdir())
            self.assertFalse(responder.same_process())
            self.assertFalse(responder.signal(signal.SIGKILL))
            time.sleep(0.2)
            self.assertIsNone(bystander.poll(), "an unrelated process was signalled")
            real = W.Responder(bystander.pid, P.process_identity(bystander.pid), W.Config(), tempfile.gettempdir())
            self.assertTrue(real.same_process())
        finally:
            bystander.kill()
            bystander.wait()

    def test_the_loop_never_acts_on_a_record_that_belongs_to_another_process(self):
        bystander = subprocess.Popen(["/bin/sleep", "30"])
        try:
            identity = P.process_identity(bystander.pid)
            with tempfile.TemporaryDirectory() as directory:
                record = P.Record(os.path.join(directory, "r.bin"), create=True)
                stale = int((time.time() - 9999) * NS)
                record.write(pid=bystander.pid + 1, start_sec=identity[0], start_usec=identity[1], stage="decode",
                             entered_ns=stale, progress_ns=stale)
                calls = []

                def sleeper(seconds):
                    calls.append(seconds)
                    if len(calls) > 5:
                        raise KeyboardInterrupt

                with self.assertRaises(KeyboardInterrupt):
                    W.run(bystander.pid, identity, record.path, None, directory, W.Config(poll=0.01), sleep=sleeper)
            self.assertIsNone(bystander.poll())
        finally:
            bystander.kill()
            bystander.wait()


class JudgeTests(unittest.TestCase):
    def judge(self, **config):
        base = dict(poll=1.0, activity_window=30.0)
        base.update(config)
        return W.Judge(W.Config(**base))

    def feed(self, judge, rows):
        """rows: (now, row, cpu_ns, io_bytes). Returns the last verdict."""
        verdict = None
        for now, record, cpu, io in rows:
            verdict = judge.observe(record, (cpu, io), now)
        return verdict

    def test_idle_is_never_stalled(self):
        judge = self.judge()
        verdict = self.feed(judge, [(t, row("idle", age=9999, now=t), 0, 0) for t in range(1000, 1100, 5)])
        self.assertEqual(verdict.state, "ok")

    def test_a_moving_stage_is_fine(self):
        judge = self.judge()
        verdict = self.feed(judge, [(t, row("decode", age=0.4, now=t), 0, 0) for t in range(1000, 1200, 5)])
        self.assertEqual(verdict.state, "ok")

    def test_a_deadlocked_stage_with_no_cpu_and_no_io_is_stalled_after_deadline_and_quiet_window(self):
        judge = self.judge()
        states = []
        for t in range(1000, 1200):
            states.append((t, judge.observe(row("decode", age=t - 1000, now=t), (5 * NS, 100), float(t))))
        first = next(t for t, v in states if v.state == "stalled")
        self.assertGreaterEqual(first, 1060)                    # not before the 60 s decode deadline
        self.assertLessEqual(first, 1062)                       # 30 s of quiet had long passed by then
        self.assertIn("no progress and no activity", states[first - 1000][1].reason)

    def test_slow_ssd_bound_work_is_slow_not_stalled(self):
        judge = self.judge()
        cpu = io = 0
        verdicts = []
        for t in range(1000, 1500):
            io += 200 * 2 ** 20                                 # the disk keeps reading
            verdicts.append(judge.observe(row("prefill", age=t - 1000, now=t), (cpu, io), float(t)))
        self.assertTrue(all(v.state != "stalled" for v in verdicts[:700]))
        self.assertEqual(verdicts[300].state, "slow")           # past the 180 s deadline but busy

    def test_busy_cpu_alone_also_counts_as_activity(self):
        judge = self.judge()
        cpu = 0
        verdicts = []
        for t in range(1000, 1200):
            cpu += int(0.9 * NS)
            verdicts.append(judge.observe(row("decode", age=t - 1000, now=t), (cpu, 0), float(t)))
        self.assertTrue(all(v.state != "stalled" for v in verdicts))

    def test_the_hard_limit_applies_even_while_busy(self):
        judge = self.judge()
        io = 0
        last = None
        for t in range(1000, 1700):
            io += 500 * 2 ** 20
            last = judge.observe(row("decode", age=t - 1000, now=t), (0, io), float(t))
            if last.state == "stalled":
                break
        self.assertEqual(last.state, "stalled")
        self.assertIn("hard limit", last.reason)

    def test_stage_specific_deadlines(self):
        quick = W.Config(deadlines={**W.DEFAULT_DEADLINES, "sync": 5}, activity_window=2.0)
        judge = W.Judge(quick)
        verdict = None
        for t in range(1000, 1010):
            verdict = judge.observe(row("sync", age=t - 1000, now=t), (0, 0), float(t))
        self.assertEqual(verdict.state, "stalled")
        calm = W.Judge(quick)
        verdict = None
        for t in range(1000, 1010):                              # prefill's own deadline is 180 s
            verdict = calm.observe(row("prefill", age=t - 1000, now=t), (0, 0), float(t))
        self.assertEqual(verdict.state, "ok")

    def test_loading_and_stopping_are_bounded_too(self):
        for stage in ("loading", "stopping", "starting", "request", "cleanup"):
            self.assertIn(stage, W.DEFAULT_DEADLINES)
            self.assertIn(stage, W.DEFAULT_HARD)
            self.assertGreater(W.DEFAULT_HARD[stage], W.DEFAULT_DEADLINES[stage])

    def test_entering_a_new_stage_restarts_the_clock(self):
        judge = self.judge()
        verdict = judge.observe(row("decode", age=500.0, entered_age=1.0, now=1000.0), (0, 0), 1000.0)
        self.assertEqual(verdict.state, "ok")
        self.assertLess(verdict.progress_age, 2.0)

    def test_environment_overrides(self):
        config = W.Config.from_environment({"FLASHNEXT_WATCHDOG_DEADLINES": "decode=7,prefill=9",
                                            "FLASHNEXT_WATCHDOG_HARD": "decode=70", "FLASHNEXT_WATCHDOG_POLL": "0.5",
                                            "FLASHNEXT_WATCHDOG_SAMPLE": "0"})
        self.assertEqual((config.deadlines["decode"], config.deadlines["prefill"], config.deadlines["sync"]), (7, 9, 60))
        self.assertEqual((config.hard["decode"], config.poll, config.sample), (70, 0.5, False))


class MemoryWatchTests(unittest.TestCase):
    def test_layer_major_defaults_and_explicit_rollback(self):
        config = W.Config.from_environment({"FLASHNEXT_PREFILL_ORDER": "layer"})
        self.assertTrue(config.memory_watch)
        self.assertEqual(config.memory_stop_mib, 3072)  # owner rule 2026-11-10: stop exactly at 3 GiB
        self.assertEqual(config.memory_poll, 0.25)
        self.assertFalse(W.Config.from_environment({"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_WATCHDOG_MEMORY": "0"}).memory_watch)
        self.assertFalse(W.Config.from_environment({}).memory_watch)

    def test_invalid_configuration_fails_closed(self):
        for values in ({"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_SWAP_CEILING_MIB": "0"},
                       {"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_WATCHDOG_MEMORY_POLL": "0"}):
            with self.assertRaises(ValueError):
                W.Config.from_environment(values)

    def test_native_swap_read(self):
        self.assertGreaterEqual(W.swap_used_mib(), 0)

    def test_readiness_requires_live_child_and_fresh_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"FLASHNEXT_WATCHDOG": "1", "FLASHNEXT_WATCHDOG_DIR": directory}
            child = mock.Mock()
            child.poll.return_value = None
            path = Path(directory) / f"progress-{os.getpid()}.bin.memory"
            with mock.patch.object(P.PROGRESS, "_process", child):
                self.assertFalse(P.memory_watch_ready(env))
                path.touch()
                self.assertTrue(P.memory_watch_ready(env))
                os.utime(path, (time.time() - 10, time.time() - 10))
                self.assertFalse(P.memory_watch_ready(env))
                path.touch()
                child.poll.return_value = 4
                self.assertFalse(P.memory_watch_ready(env))

    def _run(self, readings, stage="prefill"):
        with tempfile.TemporaryDirectory() as directory:
            record = P.Record(os.path.join(directory, "r.bin"), create=True)
            identity = P.process_identity(os.getpid())
            record.write(pid=os.getpid(), start_sec=identity[0], start_usec=identity[1], stage=stage)
            config = W.Config(memory_watch=True, memory_stop_mib=2304, memory_poll=0.25)
            responder = mock.Mock()
            responder.clients = {}
            sleeper = mock.Mock(side_effect=KeyboardInterrupt)
            try:
                with mock.patch.object(W, "Responder", return_value=responder):
                    result = W.run(os.getpid(), identity, record.path, None, directory, config,
                                   sleep=sleeper, swap_fn=mock.Mock(side_effect=readings))
                return result, responder
            finally:
                record.close()

    def test_active_high_swap_stops(self):
        result, responder = self._run([2304])
        self.assertEqual(result, W.EXIT_MEMORY_HANDLED)
        self.assertIn("swap 2304.0", responder.handle_memory.call_args.args[1])

    def test_probe_failure_stops(self):
        result, responder = self._run([OSError("probe")])
        self.assertEqual(result, W.EXIT_MEMORY_HANDLED)
        self.assertIn("probe failed", responder.handle_memory.call_args.args[1])

    def test_invalid_progress_record_does_not_hide_high_swap(self):
        with mock.patch.object(P.Record, "read", return_value=None):
            result, responder = self._run([2304])
        self.assertEqual(result, W.EXIT_MEMORY_HANDLED)
        self.assertIn("swap 2304.0", responder.handle_memory.call_args.args[1])

    def test_persistently_invalid_progress_record_stops_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            record = P.Record(os.path.join(directory, "r.bin"), create=True)
            identity = P.process_identity(os.getpid())
            record.write(pid=os.getpid(), start_sec=identity[0], start_usec=identity[1], stage="prefill")
            responder = mock.Mock()
            responder.clients = {}
            try:
                with mock.patch.object(W, "Responder", return_value=responder), \
                     mock.patch.object(P.Record, "read", return_value=None):
                    result = W.run(os.getpid(), identity, record.path, None, directory,
                                   W.Config(memory_watch=True, memory_stop_mib=2304, memory_poll=0.25),
                                   swap_fn=lambda: 0)
                self.assertEqual(result, W.EXIT_MEMORY_HANDLED)
                self.assertIn("record unavailable", responder.handle_memory.call_args.args[1])
            finally:
                record.close()

    def test_stalled_worker_stops_without_suspending_swap_monitor_for_diagnostics(self):
        verdict = W.Verdict("stalled", "test stall", "prefill")
        with mock.patch.object(W.Judge, "observe", return_value=verdict):
            result, responder = self._run([0])
        self.assertEqual(result, W.EXIT_MEMORY_HANDLED)
        responder.handle_stall.assert_not_called()
        self.assertIn("worker stalled", responder.handle_memory.call_args.args[1])

    def test_monitor_exception_fails_closed(self):
        with mock.patch.object(W.Judge, "observe", side_effect=OSError("usage probe")):
            result, responder = self._run([0])
        self.assertEqual(result, W.EXIT_MEMORY_HANDLED)
        self.assertIn("watchdog failed", responder.handle_memory.call_args.args[1])

    def test_idle_high_swap_does_not_kill_worker(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run([2304], stage="idle")

    def test_memory_stop_checks_identity_before_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            responder = W.Responder(os.getpid(), (1, 1), W.Config(), directory)
            with mock.patch.object(W.os, "kill") as kill:
                report = responder.handle_memory(row(), "test")
            self.assertFalse(report["sigkill"])
            kill.assert_not_called()

    def test_external_watchdog_stops_only_its_controlled_worker(self):
        worker = subprocess.Popen(["/bin/sleep", "30"])
        try:
            identity = P.process_identity(worker.pid)
            with tempfile.TemporaryDirectory() as directory:
                record = P.Record(os.path.join(directory, "r.bin"), create=True)
                try:
                    record.write(pid=worker.pid, start_sec=identity[0], start_usec=identity[1], stage="prefill")
                    code = ("import sys; from models.flashnext import watchdog as w; "
                            "cfg=w.Config(memory_watch=True, memory_stop_mib=0, memory_poll=.25); "
                            "sys.exit(w.run(int(sys.argv[1]), (int(sys.argv[2]), int(sys.argv[3])), "
                            "sys.argv[4], None, sys.argv[5], cfg))")
                    result = subprocess.run([sys.executable, "-c", code, str(worker.pid), str(identity[0]),
                                             str(identity[1]), record.path, directory], timeout=10)
                    self.assertEqual(result.returncode, W.EXIT_MEMORY_HANDLED)
                    self.assertEqual(worker.wait(timeout=5), -signal.SIGKILL)
                    self.assertEqual(len(list(Path(directory).glob("memory-*.json"))), 1)
                finally:
                    record.close()
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


class ClientNoticeTests(unittest.TestCase):
    def test_before_any_response_the_client_gets_a_503_json_error(self):
        data = W.client_error_bytes("/v1/chat/completions", False, "stalled")
        head, _, body = data.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 503"))
        self.assertEqual(json.loads(body)["error"]["code"], "worker_stalled")
        self.assertIn(b"Content-Length: " + str(len(body)).encode(), head)

    def test_mid_stream_each_protocol_gets_its_own_error_event(self):
        chat = W.client_error_bytes("/v1/chat/completions", True, "m")
        self.assertIn(b'data: {"error"', chat)
        self.assertTrue(chat.rstrip().endswith(b"data: [DONE]"))
        anthropic = W.client_error_bytes("/v1/messages", True, "m")
        self.assertIn(b"event: error", anthropic)
        self.assertIn(b"event: message_stop", anthropic)
        responses = W.client_error_bytes("/v1/responses", True, "m")
        self.assertIn(b"event: response.failed", responses)


# -- real processes -----------------------------------------------------------------------------------------------

WORKER = """
import ctypes, os, signal, socket, sys, time
sys.path.insert(0, %(root)r)
from models.flashnext import progress
port = int(sys.argv[1]); mode = sys.argv[2]
process = progress.start_watchdog(trace_path="/tmp/example-trace.jsonl")
print("WATCHDOG", process.pid, flush=True)
client = socket.create_connection(("127.0.0.1", port))            # stands in for the accepted client connection
progress.PROGRESS.request_begin(client, "/v1/chat/completions")
if os.environ.get("TEST_STARTED", "1") == "1":
    progress.PROGRESS.response_started()                           # the SSE headers are already out
progress.PROGRESS.stage("decode", "token loop")
for token in range(1, 4):
    progress.PROGRESS.tick(token); time.sleep(0.1)
time.sleep(0.5)                                                    # let the watchdog take the connection
if mode == "ignore_sigterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("HANGING", flush=True)
if mode in ("gil_sleep", "ignore_sigterm"):
    libc = ctypes.PyDLL(None)                                      # PyDLL keeps the GIL during the call
    while True:
        libc.sleep(2)                                              # native wait; a signal ends one sleep, not the freeze
elif mode == "busy":
    end = time.time() + 6
    while time.time() < end:
        pass                                                       # busy, no progress ticks
    progress.PROGRESS.stage("idle")
elif mode == "exit":
    progress.PROGRESS.stage("idle")
elif mode == "idle_forever":
    progress.PROGRESS.stage("idle"); time.sleep(6)
elif mode == "real_deadlock":
    import numpy as np
    import mlx.core as mx
    host = np.random.default_rng(0).standard_normal((2048, 2048), dtype=np.float32)
    x = mx.from_dlpack(host, copy=False)
    w = mx.random.normal((2048, 2048)); y = x
    for _ in range(40): y = mx.tanh(y @ w)
    mx.async_eval(y.sum()); del x, y, host
    mx.synchronize()
print("WORKER FINISHED", flush=True)
"""


def metal_available():
    probe = subprocess.run([sys.executable, "-c", "import mlx.core as mx; raise SystemExit(0 if mx.metal.is_available() else 3)"],
                           capture_output=True, timeout=60, cwd=ROOT)
    return probe.returncode == 0


class RealProcessTests(unittest.TestCase):
    ENV = {"FLASHNEXT_WATCHDOG": "1", "FLASHNEXT_WATCHDOG_DEADLINES": "decode=3,prefill=3,sync=3",
           "FLASHNEXT_WATCHDOG_HARD": "decode=40,prefill=40", "FLASHNEXT_WATCHDOG_ACTIVITY_WINDOW": "2",
           "FLASHNEXT_WATCHDOG_POLL": "0.2", "FLASHNEXT_WATCHDOG_GRACE": "2", "FLASHNEXT_WATCHDOG_KILL_WAIT": "3",
           "FLASHNEXT_WATCHDOG_SAMPLE": "0"}

    def run_worker(self, mode, wait_for_report=True, timeout=40, extra_env=None):
        directory = tempfile.mkdtemp(prefix="wd-test-")
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(15)
        env = {**os.environ, **self.ENV, "FLASHNEXT_WATCHDOG_DIR": directory, **(extra_env or {})}
        code = WORKER % {"root": str(ROOT)}
        worker = subprocess.Popen([sys.executable, "-c", code, str(server.getsockname()[1]), mode], env=env, cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        began = time.time()
        client = None
        try:
            client, _ = server.accept()
            client.settimeout(timeout)
            received = b""
            try:
                while time.time() - began < timeout:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    received += chunk
            except (socket.timeout, ConnectionResetError):
                pass
            try:
                worker.wait(timeout=max(1, timeout - (time.time() - began)))
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
                self.fail(f"the worker was never stopped in {timeout}s ({mode})")
            out = worker.stdout.read()
            watchdog_pid = int(out.split("WATCHDOG")[1].split()[0])
            for _ in range(100):                                   # the watchdog finishes its report after the kill
                reports = list(Path(directory).glob("stall-*.json"))
                alive = P.process_identity(watchdog_pid) is not None
                if reports and not alive:
                    break
                if not wait_for_report and not alive:
                    break
                time.sleep(0.1)
            return {"returncode": worker.returncode, "received": received, "stdout": out, "dir": Path(directory),
                    "elapsed": time.time() - began, "reports": list(Path(directory).glob("stall-*.json")),
                    "watchdog_pid": watchdog_pid, "stderr": worker.stderr.read()}
        finally:
            if worker.poll() is None:
                worker.kill()
            if client is not None:
                client.close()
            server.close()

    def test_a_worker_frozen_with_the_gil_held_is_stopped_with_diagnostics_and_the_client_is_told(self):
        result = self.run_worker("gil_sleep")
        self.assertEqual(result["returncode"], -signal.SIGTERM)
        self.assertEqual(len(result["reports"]), 1, result["stderr"])
        report = json.loads(result["reports"][0].read_text())
        self.assertEqual(report["stage"], "decode")
        self.assertIn("no progress and no activity", report["verdict"])
        self.assertTrue(report["client_notified"])
        self.assertTrue(report["termination"]["sigterm"])
        self.assertFalse(report["termination"]["sigkill"])
        self.assertTrue(report["termination"]["gone"])
        self.assertEqual(report["guard_trace"], "/tmp/example-trace.jsonl")
        self.assertIn("in <module>", report["python_stacks"])        # Python stacks despite the held GIL
        text = result["received"].decode(errors="replace")
        self.assertIn("worker_stalled", text)
        self.assertIn("data: [DONE]", text)                          # mid-stream: an SSE error event, then DONE
        self.assertLess(result["elapsed"], 30)
        self.assertEqual(oct(os.stat(result["reports"][0]).st_mode & 0o777), "0o600")

    def test_a_client_still_waiting_for_headers_gets_a_clean_503(self):
        result = self.run_worker("gil_sleep", extra_env={"TEST_STARTED": "0"})
        text = result["received"].decode(errors="replace")
        self.assertTrue(text.startswith("HTTP/1.1 503"), text[:200])
        self.assertEqual(json.loads(text.split("\r\n\r\n", 1)[1])["error"]["code"], "worker_stalled")

    def test_a_worker_that_ignores_sigterm_is_killed_only_after_the_grace_period(self):
        result = self.run_worker("ignore_sigterm")
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        report = json.loads(result["reports"][0].read_text())
        self.assertTrue(report["termination"]["sigterm"] and report["termination"]["sigkill"])
        self.assertTrue(report["termination"]["gone"])

    def test_a_busy_worker_without_progress_is_left_alone_until_it_finishes(self):
        result = self.run_worker("busy", wait_for_report=False)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["reports"], [])
        self.assertIn("WORKER FINISHED", result["stdout"])

    def test_an_idle_worker_is_never_stopped(self):
        result = self.run_worker("idle_forever", wait_for_report=False)
        self.assertEqual((result["returncode"], result["reports"]), (0, []))

    def test_the_watchdog_exits_by_itself_when_the_worker_exits(self):
        result = self.run_worker("exit", wait_for_report=False)
        self.assertEqual(result["returncode"], 0)
        time.sleep(1.0)
        self.assertIsNone(P.process_identity(result["watchdog_pid"]), "the watchdog outlived its worker")

    @unittest.skipUnless(metal_available(), "needs the Metal build of MLX")
    def test_the_known_gpu_gil_deadlock_is_bounded_by_the_watchdog(self):
        result = self.run_worker("real_deadlock", timeout=60)
        self.assertIn(result["returncode"], (-signal.SIGTERM, -signal.SIGKILL), result["stderr"][-800:])
        self.assertEqual(len(result["reports"]), 1)
        report = json.loads(result["reports"][0].read_text())
        self.assertIn("in <module>", report["python_stacks"])
        self.assertTrue(report["termination"]["gone"])
        self.assertLess(result["elapsed"], 45)


class WatchdogWiringTests(unittest.TestCase):
    def test_agentic_enables_the_watchdog_and_chat_does_not(self):
        from models.flashnext.settings import modes

        self.assertEqual(modes.plan_mode(modes.AGENTIC, pool_gb="0").env["FLASHNEXT_WATCHDOG"], "1")
        self.assertNotIn("FLASHNEXT_WATCHDOG", modes.plan_mode(modes.CHAT, pool_gb="0").env)
        self.assertIn("FLASHNEXT_WATCHDOG", modes.OWNED)

    def test_the_watchdog_module_never_imports_mlx_or_the_inference_stack(self):
        code = ("import sys; sys.path.insert(0, %r); import models.flashnext.watchdog; "
                "bad = [m for m in sys.modules if m.split('.')[0] in ('mlx', 'mlx_vlm', 'transformers', 'numpy')]; "
                "print('BAD', bad)" % str(ROOT))
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=ROOT)
        self.assertIn("BAD []", done.stdout, done.stderr)

    def test_the_server_marks_requests_and_responses_for_the_watchdog(self):
        source = (ROOT / "veystrix/server.py").read_text()
        self.assertIn("progress.PROGRESS.request_begin", source)
        self.assertIn("progress.PROGRESS.request_end()", source)
        self.assertIn("progress.PROGRESS.response_started()", source)
        backend = (ROOT / "veystrix/backends/flashnext.py").read_text()
        self.assertIn("progress.PROGRESS.tick()", backend)
        self.assertIn("progress.PROGRESS.tick(len(produced) + 1)", backend)


if __name__ == "__main__":
    unittest.main()
