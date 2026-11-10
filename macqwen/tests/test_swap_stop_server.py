"""Server behaviour under a memory safety stop, a stalled worker, slow prefill and the 32K admission ceilings.

A real in-process HTTP server (real scheduler, real handler) with a scripted backend whose "prefill" is a slow loop of
streamed-layer ticks. Nothing here loads a model or touches swap: readings are synthetic (see docs/flashnext/
deepseek-swap-incident-2026-11-09.md for what each test stands for).
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace

from macqwen import server as S
from macqwen.backends.base import GenerationCancelled
from macqwen.server import (MacqwenHTTPServer, ModelService, OversizedPrefillError, RecoveryRequiredError,
                            ContextLimitError)
from macqwen.tests.test_server import FakeBackend, FakeSession
from models.flashnext import memory_controller as MC, progress

STOP_REASON = "swap grew 2.50 GiB over the session baseline (hard limit 2.5 GiB)"


class SlowPrefillBackend(FakeBackend):
    """generate() = N streamed-layer steps. Cancellation is honoured at each step like the real layer_completed callback,
    unless ``deaf`` is set (a blocked native call that never returns to Python until released)."""

    def __init__(self):
        super().__init__(["answer"])
        self.layers = 400
        self.step = 0.01
        self.deaf = False
        self.release = threading.Event()
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.done_layers = 0

    def generate(self, max_tokens, out=None, should_cancel=None, on_prefill_progress=None):
        self.entered.set()
        self.tape.extend(self.pending)
        self.pending = []
        if self.deaf:
            self.release.wait(15)
        for layer in range(self.layers):
            time.sleep(self.step)
            self.done_layers = layer + 1
            progress.PROGRESS.tick()
            if on_prefill_progress is not None:
                on_prefill_progress(layer + 1, self.layers)
            if should_cancel is not None and should_cancel():
                self.cancelled.set()
                raise GenerationCancelled
        if out:
            out("answer")
        return "answer", SimpleNamespace(finish="stop", tokens=1, prompt_tokens=4, prefill_seconds=0.1, rate=10.0)


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.backend = SlowPrefillBackend()
        session = FakeSession(["x"])
        session.backend = self.backend
        self.service = ModelService(session)
        self.server = MacqwenHTTPServer(("127.0.0.1", 0), self.service, None, (), None)
        self.stop = threading.Event()
        self.threads = [threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True),
                        threading.Thread(target=self.server.scheduler.serve_jobs, args=(self.stop,), daemon=True)]
        for thread in self.threads:
            thread.start()
        self.port = self.server.server_port
        self.controller = MC.Controller(MC.Policy(swap_only=True))
        MC.install(self.controller)
        MC.register_worker_probe(lambda: self.server.scheduler.busy)
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.backend.release.set()
        MC.install(None)
        MC.register_worker_probe(None)
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()

    def get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        response = conn.getresponse()
        body = json.loads(response.read() or b"{}")
        conn.close()
        return response.status, body

    def post_stream(self, extra=None):
        body = {"model": "macqwen-flashnext", "stream": True, "messages": [{"role": "user", "content": "hello"}]}
        body.update(extra or {})
        raw = json.dumps(body).encode()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw)
        return sock

    @staticmethod
    def read_until(sock, marker: bytes, limit=10.0):
        data, deadline = b"", time.time() + limit
        while marker not in data and time.time() < deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
        return data

    def wait_idle(self, seconds=5.0):
        end = time.time() + seconds
        while self.server.scheduler.busy and time.time() < end:
            time.sleep(0.02)
        return not self.server.scheduler.busy


class SafetyStopServerTests(ServerCase):
    def test_a_stop_cancels_the_prefill_and_the_client_gets_an_actionable_error_not_a_hang(self):
        sock = self.post_stream()
        self.assertTrue(self.backend.entered.wait(3))
        time.sleep(0.1)
        started = time.time()
        self.controller._stop(STOP_REASON)               # what the guard's controller does at +2.5 GiB
        data = self.read_until(sock, b"[DONE]", 5)
        sock.close()
        self.assertLess(time.time() - started, 2.0, "cancellation must be prompt (next layer boundary), not a hang")
        self.assertTrue(self.backend.cancelled.is_set())
        self.assertLess(self.backend.done_layers, 400, "the prefill must have stopped early")
        text = data.decode()
        self.assertIn("memory_critical_stop", text)
        self.assertIn("Do NOT retry automatically", text)
        self.assertIn("swap grew 2.50 GiB", text)
        self.assertTrue(self.wait_idle(), "the inference worker must be free again")
        self.assertEqual(self.backend.tape, [], "the half-built state must not survive")

    def test_new_requests_are_refused_after_a_stop_and_health_says_so(self):
        self.controller._stop(STOP_REASON)
        status, health = self.get("/health")
        self.assertEqual((health["status"], health["ready"]), ("stopped", False))
        self.assertEqual(health["memory_policy"]["stop_reason"], STOP_REASON)
        self.assertEqual(self.get("/ready")[0], 503)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read())
        self.assertEqual(response.status, 503)
        self.assertEqual(payload["error"]["code"], "memory_critical_stop")
        self.assertEqual(response.getheader("x-should-retry"), "false", "SDK clients must not auto-retry a safety stop")
        self.assertFalse(self.backend.entered.is_set(), "no inference may start after the stop")

    def test_non_streaming_requests_also_get_the_503_with_the_reason(self):
        self.backend.layers = 400
        result = {}

        def call():
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            conn.request("POST", "/v1/chat/completions",
                         json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            result["status"], result["body"] = response.status, json.loads(response.read())

        thread = threading.Thread(target=call)
        thread.start()
        self.assertTrue(self.backend.entered.wait(3))
        time.sleep(0.05)
        self.controller._stop(STOP_REASON)
        thread.join(5)
        self.assertEqual(result["status"], 503)
        self.assertEqual(result["body"]["error"]["code"], "memory_critical_stop")
        self.assertIn("Do NOT retry", result["body"]["error"]["message"])

    def test_a_client_that_disconnects_cancels_the_prefill(self):
        sock = self.post_stream()
        self.assertTrue(self.backend.entered.wait(3))
        time.sleep(0.1)
        sock.close()
        self.assertTrue(self.backend.cancelled.wait(5), "a closed client must end the work it can no longer see")
        self.assertTrue(self.wait_idle())


class StalledWorkerTests(ServerCase):
    def test_health_never_claims_ok_for_a_permanently_stalled_worker_and_stays_responsive(self):
        self.backend.deaf = True                          # a blocked native call: no ticks, no cancellation
        original = S.STALL_SECONDS
        S.STALL_SECONDS = 0.4
        self.addCleanup(setattr, S, "STALL_SECONDS", original)
        sock = self.post_stream()
        self.assertTrue(self.backend.entered.wait(3))
        status, early = self.get("/health")
        self.assertEqual(early["status"], "ok")
        time.sleep(0.8)
        started = time.time()
        status, health = self.get("/health")             # answered by an HTTP thread while the worker is blocked
        self.assertLess(time.time() - started, 1.0)
        self.assertEqual(health["status"], "stalled")
        self.assertFalse(health["ready"])
        self.assertTrue(health["worker"]["stalled"])
        self.assertGreater(health["worker"]["seconds_since_progress"], 0.4)
        self.assertEqual(self.get("/ready")[0], 503)
        self.backend.release.set()
        self.read_until(sock, b"[DONE]", 10)
        sock.close()
        self.assertTrue(self.wait_idle(10))

    def test_sse_heartbeats_keep_flowing_while_the_worker_is_blocked(self):
        self.backend.deaf = True
        original = S.SSE_KEEPALIVE_SECONDS
        S.SSE_KEEPALIVE_SECONDS = 0.1
        self.addCleanup(setattr, S, "SSE_KEEPALIVE_SECONDS", original)
        sock = self.post_stream()
        self.assertTrue(self.backend.entered.wait(3))
        time.sleep(0.8)
        sock.settimeout(0.5)
        data = self.read_until(sock, b"never-present", 0.9)
        self.assertGreaterEqual(data.count(b": generating"), 3, "heartbeats must flow while the worker is stuck")
        self.assertNotIn(b'"content": "answer"', data)
        self.backend.release.set()
        sock.settimeout(10)
        rest = self.read_until(sock, b"[DONE]", 10)
        sock.close()
        self.assertIn(b"answer", data + rest)

    def test_progress_diagnostics_show_stage_and_forward_progress(self):
        sock = self.post_stream()
        self.assertTrue(self.backend.entered.wait(3))
        time.sleep(0.15)
        _, health = self.get("/health")
        worker = health["worker"]
        self.assertTrue(worker["busy"] and worker["in_request"])
        self.assertGreater(worker["progress_ticks"], 0)
        self.assertLess(worker["seconds_since_progress"], 1.0)
        self.assertFalse(worker["stalled"])
        sock.close()
        self.assertTrue(self.wait_idle())


class PrefillCeilingTests(unittest.TestCase):
    """32,768-token ceilings (cold and uncached increment); the 262,144 window is separate and filled incrementally."""

    def service(self, **kw):
        session = FakeSession(["ok"])
        return session.backend, ModelService(session, **kw)

    def adopt(self, service, text):
        service._adopt(text)

    def test_the_defaults_are_32768_and_the_native_window_is_262144(self):
        self.assertEqual((S.MAX_COLD_PREFILL_DEFAULT, S.MAX_INCREMENTAL_PREFILL_DEFAULT), (32768, 32768))
        backend, service = self.service()
        report = S._context_report(service, None, False)
        self.assertEqual((report["max_cold_prefill"], report["max_incremental_prefill"]), (32768, 32768))
        self.assertEqual(report["architectural_context_maximum"], 262144)
        self.assertEqual(report["preferred_internal_chunk_size"], 2048)
        self.assertEqual(report["validated_prefill_tokens_on_real_checkpoint"], 3974)

    def test_incremental_and_cold_limits_are_independent(self):
        _, service = self.service(max_cold_prefill=100, max_incremental_prefill=128)
        service._check_prefill(120, total_tokens=184, matched_prefix=64, incremental=True)
        with self.assertRaises(S.ContextLimitError):
            service._check_prefill(120, total_tokens=120, matched_prefix=0, incremental=False)

    def test_a_cold_prompt_of_exactly_32768_tokens_is_admitted_and_one_more_is_refused(self):
        backend, service = self.service(context_limit=262144, max_cold_prefill=32768, max_incremental_prefill=32768)
        self.adopt(service, "a" * 32768)
        self.assertEqual(len(backend.pending), 32768)
        backend.reset()
        with self.assertRaises(OversizedPrefillError) as caught:
            self.adopt(service, "a" * 32769)
        self.assertEqual(caught.exception.code, "oversized_prefill_rejected")
        self.assertIn("32768", str(caught.exception))
        self.assertEqual(backend.pending, [], "a refusal must run nothing")

    def test_a_262k_conversation_with_a_small_suffix_prefills_only_the_suffix(self):
        backend, service = self.service(context_limit=262144, max_cold_prefill=32768, max_incremental_prefill=32768)
        backend.tape = [97] * 250000
        self.adopt(service, "a" * 250100)
        self.assertEqual(len(backend.pending), 100)
        self.assertEqual(service.reused, 1)

    def test_the_incremental_ceiling_is_enforced_on_its_own(self):
        backend, service = self.service(context_limit=262144, max_cold_prefill=32768, max_incremental_prefill=4096)
        backend.tape = [97] * 100000
        with self.assertRaises(OversizedPrefillError) as caught:
            self.adopt(service, "a" * 105000)
        self.assertIn("incremental prefill limit of 4096", str(caught.exception))
        self.assertEqual(backend.pending, [])
        self.adopt(service, "a" * 103000)                 # 3,000 new tokens: fine
        self.assertEqual(len(backend.pending), 3000)

    def test_a_giant_cold_prompt_is_refused_not_split_across_chunks(self):
        backend, service = self.service(context_limit=262144, max_cold_prefill=32768, max_incremental_prefill=32768)
        with self.assertRaises(OversizedPrefillError):
            self.adopt(service, "a" * 200000)
        self.assertEqual((backend.pending, backend.tape), ([], []))

    def test_over_the_window_is_a_context_error(self):
        backend, service = self.service(context_limit=262144, max_cold_prefill=32768, max_incremental_prefill=32768)
        backend.tape = [97] * 262000
        with self.assertRaises(ContextLimitError):
            self.adopt(service, "a" * 262144)             # no room for the minimum output reserve

    def test_health_reports_the_ceilings(self):
        backend, service = self.service(max_cold_prefill=32768, max_incremental_prefill=32768)
        report = S.health(service, SimpleNamespace(busy=False, snapshot=lambda: {}))
        self.assertEqual((report["max_cold_prefill"], report["max_incremental_prefill"]), (32768, 32768))
        self.assertEqual(report["status"], "ok")


class DiagnosticsSwapTests(unittest.TestCase):
    def test_cache_diagnostics_reports_real_swap_growth_from_the_controller(self):
        controller = MC.Controller(MC.Policy(swap_only=True))
        MC.install(controller)
        self.addCleanup(MC.install, None)
        reading = dict(pressure=1, swapout_pages=0, swapin_pages=0, available_mb=5000, compressor_mb=500)
        controller.observe(100.0, swap_used_mb=1000.0, **reading)
        controller.observe(101.0, swap_used_mb=1800.0, **reading)
        controller.observe(102.0, swap_used_mb=1200.0, **reading)
        service = ModelService(FakeSession(["x"]))
        report = service.cache_diagnostics()
        self.assertEqual(report["swap_baseline_mb"], 1000.0)
        self.assertEqual(report["current_swap_growth_mb"], 200.0)
        self.assertEqual(report["peak_swap_growth_mb"], 800.0)


if __name__ == "__main__":
    unittest.main()
