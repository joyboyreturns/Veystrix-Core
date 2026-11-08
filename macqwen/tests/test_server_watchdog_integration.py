"""The production shape end to end, offline: the real HTTP server and scheduler in a worker process, the real out-of-process
watchdog, a fake model that streams a few tokens and then freezes with the GIL held. No checkpoint, no Metal.

Asserts that the watchdog notices, tells the HTTP client in its own protocol, writes its report, and ends ONLY the worker:
an unrelated process started beforehand is still alive afterwards.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from models.flashnext import progress as P

ROOT = Path(__file__).resolve().parents[2]

WORKER = r'''
import ctypes, os, sys, threading, time
sys.path.insert(0, %(root)r)
from models.flashnext import progress
from macqwen import server as S
from macqwen.scheduler import Scheduler
from macqwen.tests.test_server import FakeBackend, FakeSession

watchdog = progress.start_watchdog(trace_path="/tmp/integration-trace.jsonl")
progress.PROGRESS.stage("idle")


class Freezing(FakeBackend):
    def generate(self, max_tokens, out=None, **kwargs):
        progress.PROGRESS.stage("decode", "token loop")        # the real backend does this through trace_phase
        for token in (1, 2, 3):
            if out:
                out("word ")
            progress.PROGRESS.tick(token)
            time.sleep(0.15)
        libc = ctypes.PyDLL(None)                              # a native wait that keeps the GIL: the real failure shape
        while True:
            libc.sleep(2)


session = FakeSession(["x"])
session.backend = Freezing(["x"])
server = S.MacqwenHTTPServer(("127.0.0.1", 0), S.ModelService(session), scheduler=Scheduler())
threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
print("WATCHDOG", watchdog.pid, "PORT", server.server_port, flush=True)
server.scheduler.serve_jobs(threading.Event())           # inference on the main thread, like serve()
'''


class ServerWatchdogIntegrationTests(unittest.TestCase):
    ENV = {"FLASHNEXT_WATCHDOG": "1", "FLASHNEXT_WATCHDOG_DEADLINES": "decode=3,prefill=3,sync=3,request=3",
           "FLASHNEXT_WATCHDOG_HARD": "decode=40,prefill=40,request=40", "FLASHNEXT_WATCHDOG_ACTIVITY_WINDOW": "2",
           "FLASHNEXT_WATCHDOG_POLL": "0.2", "FLASHNEXT_WATCHDOG_GRACE": "2", "FLASHNEXT_WATCHDOG_KILL_WAIT": "3",
           "FLASHNEXT_WATCHDOG_SAMPLE": "0"}

    def test_a_frozen_worker_behind_the_real_server_is_detected_reported_and_the_only_process_ended(self):
        directory = tempfile.mkdtemp(prefix="wd-int-")
        bystander = subprocess.Popen(["sleep", "120"])
        self.addCleanup(lambda: (bystander.kill(), bystander.wait()))
        env = {**os.environ, **self.ENV, "FLASHNEXT_WATCHDOG_DIR": directory}
        worker = subprocess.Popen([sys.executable, "-c", WORKER % {"root": str(ROOT)}], env=env, cwd=ROOT,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: worker.poll() is None and (worker.kill(), worker.wait()))
        first = worker.stdout.readline()
        self.assertIn("PORT", first, worker.stderr.read() if worker.poll() is not None else first)
        watchdog_pid, port = int(first.split()[1]), int(first.split()[3])
        began = time.time()

        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 16}).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=40)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        received = b""
        try:
            while time.time() - began < 40:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                received += chunk
        except (socket.timeout, ConnectionResetError):
            pass
        finally:
            sock.close()
        try:
            worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.fail("the frozen worker was never stopped")

        # the worker, and only the worker, was ended: by SIGTERM, within the configured bounds
        self.assertEqual(worker.returncode, -signal.SIGTERM)
        self.assertLess(time.time() - began, 30)
        self.assertIsNone(bystander.poll(), "the watchdog ended a process it was not started for")
        for _ in range(100):                                       # the watchdog finishes its report, then exits itself
            if P.process_identity(watchdog_pid) is None:
                break
            time.sleep(0.1)
        self.assertIsNone(P.process_identity(watchdog_pid), "the watchdog should exit once its worker is gone")

        # the client saw a stream that had started and was then ended in the protocol it speaks
        text = received.decode(errors="replace")
        self.assertIn("200 OK", text)
        self.assertIn("chat.completion.chunk", text)               # tokens were already flowing
        self.assertIn("worker_stalled", text)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"), text[-300:])

        reports = list(Path(directory).glob("stall-*.json"))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["stage"], "decode")
        self.assertIn("no progress and no activity", report["verdict"])
        self.assertTrue(report["client_notified"])
        self.assertTrue(report["termination"]["sigterm"])
        self.assertTrue(report["termination"]["gone"])
        self.assertIn("in generate", report["python_stacks"])      # Python stacks despite the GIL being held


if __name__ == "__main__":
    unittest.main()
