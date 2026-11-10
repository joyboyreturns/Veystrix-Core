"""The shared HTTP server driving the real NemotronBackend on a tiny synthetic checkpoint (no real weights)."""
import http.client
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from models.nemotron35.tiny_checkpoint import permissive_config_check, write_checkpoint
from models.nemotron35.tests.unit.test_backend import StubTokenizer, machine
from veystrix import preferences, server as S
from veystrix.backends.nemotron35 import NemotronBackend
from veystrix.scheduler import Scheduler
from veystrix.session import Session


def call(port, method, path, body=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        raw = response.read()
        return response.status, raw, dict(response.getheaders())
    finally:
        conn.close()


class NemotronServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="nemotron-server-")
        write_checkpoint(Path(cls.dir) / "ck")
        cls.backend = NemotronBackend(
            Path(cls.dir) / "ck", authorized=True, config_check=permissive_config_check, machine=machine(),
            tokenizer=StubTokenizer(), pool_bytes=64 * 1024, chunk_tokens=4, session_dir=str(Path(cls.dir) / "s"))
        cls.backend._current_machine = lambda: machine()
        prefs = preferences.load(str(Path(cls.dir) / "prefs.json"))
        prefs["model"] = "nemotron35-lightning"
        session = Session(cls.backend, "plain", prefs, str(Path(cls.dir) / "prefs.json"),
                          str(Path(cls.dir) / "keys.json"), migrate_system_prompt=False)
        cls.service = S.ModelService(session, context_limit=cls.backend.context_limit)
        cls.server = S.VeystrixHTTPServer(("127.0.0.1", 0), cls.service, api_key=None, scheduler=Scheduler())
        cls.port = cls.server.server_port
        threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.backend.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_models_endpoint_names_nemotron_and_honest_limits(self):
        status, raw, _ = call(self.port, "GET", "/v1/models")
        self.assertEqual(status, 200)
        text = raw.decode()
        self.assertIn("veystrix-nemotron35-lightning", text)
        self.assertIn("Nemotron", text)
        self.assertNotIn("Flash-Next", text)

    def test_health_reports_ready_model(self):
        status, raw, _ = call(self.port, "GET", "/health")
        body = json.loads(raw)
        self.assertEqual((status, body["model"]), (200, "veystrix-nemotron35-lightning"))

    def test_non_streaming_chat_reports_usage(self):
        status, raw, _ = call(self.port, "POST", "/v1/chat/completions", {
            "model": "veystrix-nemotron35-lightning", "max_tokens": 5,
            "messages": [{"role": "user", "content": "hello there"}]})
        self.assertEqual(status, 200, raw)
        body = json.loads(raw)
        self.assertGreater(body["usage"]["prompt_tokens"], 0)
        self.assertLessEqual(body["usage"]["completion_tokens"], 5)
        self.assertTrue(self.backend.check_invariant())

    def test_streaming_chat_emits_sse_and_done(self):
        status, raw, headers = call(self.port, "POST", "/v1/chat/completions", {
            "model": "veystrix-nemotron35-lightning", "max_tokens": 4, "stream": True,
            "messages": [{"role": "user", "content": "stream please"}]})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))
        self.assertIn(b"data: [DONE]", raw)

    def test_second_request_reuses_or_rebuilds_without_corruption(self):
        for text in ("first", "first and second", "unrelated"):
            status, raw, _ = call(self.port, "POST", "/v1/chat/completions", {
                "model": "veystrix-nemotron35-lightning", "max_tokens": 3,
                "messages": [{"role": "user", "content": text}]})
            self.assertEqual(status, 200, raw)
            self.assertTrue(self.backend.check_invariant())

    def test_oversized_prompt_is_refused_not_truncated(self):
        status, raw, _ = call(self.port, "POST", "/v1/chat/completions", {
            "model": "veystrix-nemotron35-lightning", "max_tokens": 2,
            "messages": [{"role": "user", "content": "x" * (self.backend.context_limit + 50)}]})
        self.assertEqual(status, 400, raw)


if __name__ == "__main__":
    unittest.main()
