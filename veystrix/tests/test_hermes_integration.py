"""Hermes Agent <-> Veystrix integration tests against a SYNTHETIC server. No model is loaded.

Always run (fast, offline, no Hermes needed):
  * the profile configurator: merge, backup, no secrets, loopback only, cloud routes closed;
  * the mock server's wire behaviour (stream, reasoning, tool calls, errors, auth, Origin refusal, cancellation).

Run only when Hermes Agent is installed and VEYSTRIX_HERMES_E2E=1 (about a minute; the first call prepares Hermes' runtime):
  * real `hermes -z` against the mock: streaming, reasoning content, tool call + tool-result continuation,
    multi-turn persistence, engine-style errors, cancellation and reconnection, and zero egress to the Internet.

    VEYSTRIX_HERMES_E2E=1 .venv/bin/python -m unittest veystrix.tests.test_hermes_integration -v
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations" / "hermes"))

import configure_profile as cp                      # noqa: E402
from mock_veystrix_server import MockVeystrixServer  # noqa: E402

HERMES = Path(os.environ.get("HERMES_BIN", "~/.local/bin/hermes")).expanduser()
E2E = os.environ.get("VEYSTRIX_HERMES_E2E") == "1" and HERMES.exists()


def _request(mock, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", mock.port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    return response.status, response.getheaders(), raw


def _sse(raw: bytes) -> list:
    events = []
    for block in raw.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(line[6:])
    return events


class ConfigureProfileTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="veystrix-hermes-cfg-"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_desired_config_is_loopback_only_with_no_cloud_routes(self):
        import yaml
        cfg = cp.desired_config(8080, "veystrix-flashnext", 65536)
        text = yaml.safe_dump(cfg)
        self.assertEqual(cfg["model"]["base_url"], "http://127.0.0.1:8080/v1")
        self.assertEqual(cfg["fallback_providers"], [])
        self.assertFalse(cfg["auth"]["adopt_external_logins"])
        self.assertFalse(cfg["updates"]["check"])
        self.assertFalse(cfg["telemetry"]["shared_metrics"]["send"])
        for task, section in cfg["auxiliary"].items():                  # every auxiliary task stays on the local endpoint
            self.assertEqual(section["base_url"], "http://127.0.0.1:8080/v1", task)
            self.assertEqual(section["provider"], "custom", task)
        for task in cp.AUX_TASKS_OFF:
            self.assertFalse(cfg["auxiliary"][task]["enabled"], task)
        for host in ("openrouter", "nousresearch", "anthropic.com", "openai.com", "googleapis"):
            self.assertNotIn(host, text)
        self.assertFalse({"web", "browser", "image_gen", "tts", "vision"} & set(cfg["platform_toolsets"]["cli"]))

    def test_context_length_meets_hermes_minimum(self):
        self.assertGreaterEqual(cp.DEFAULT_CONTEXT, 64000)       # Hermes 0.23.1 refuses smaller windows

    def test_no_secret_is_written_only_the_documented_placeholder(self):
        import yaml
        cp.main(["--hermes-home", str(self.dir), "--port", "8080"])
        self.assertNotIn("sk-", (self.dir / "config.yaml").read_text())
        self.assertEqual(yaml.safe_load((self.dir / "config.yaml").read_text())["providers"]["veystrix"]["key_env"], "VEYSTRIX_API_KEY")
        env = (self.dir / ".env").read_text()
        self.assertIn("VEYSTRIX_API_KEY=local", env)
        self.assertEqual(oct((self.dir / ".env").stat().st_mode & 0o777), "0o600")

    def test_existing_key_in_env_is_not_replaced(self):
        (self.dir / ".env").write_text("VEYSTRIX_API_KEY=my-real-server-key\nOTHER=1\n")
        cp.main(["--hermes-home", str(self.dir)])
        env = (self.dir / ".env").read_text()
        self.assertIn("VEYSTRIX_API_KEY=my-real-server-key", env)
        self.assertNotIn("VEYSTRIX_API_KEY=local", env)
        self.assertIn("OTHER=1", env)

    def test_merge_preserves_unrelated_settings_and_backs_up_first(self):
        import yaml
        (self.dir / "config.yaml").write_text(yaml.safe_dump({
            "providers": {"my-corp-proxy": {"base_url": "https://llm.example.com/v1", "key_env": "CORP_KEY"}},
            "display": {"streaming": True}, "model": {"default": "old/model", "provider": "auto"}}))
        cp.main(["--hermes-home", str(self.dir)])
        merged = yaml.safe_load((self.dir / "config.yaml").read_text())
        self.assertIn("my-corp-proxy", merged["providers"])                 # unrelated provider kept
        self.assertEqual(merged["display"], {"streaming": True})
        self.assertEqual(merged["model"]["provider"], "veystrix")
        backups = list(self.dir.glob("config.yaml.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(yaml.safe_load(backups[0].read_text())["model"]["default"], "old/model")

    def test_idempotent(self):
        import yaml
        cp.main(["--hermes-home", str(self.dir)])
        first = yaml.safe_load((self.dir / "config.yaml").read_text())
        cp.main(["--hermes-home", str(self.dir)])
        self.assertEqual(yaml.safe_load((self.dir / "config.yaml").read_text()), first)

    def test_dry_run_changes_nothing(self):
        cp.main(["--hermes-home", str(self.dir), "--dry-run"])
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_refuses_a_missing_profile_and_bad_ports(self):
        with self.assertRaises(SystemExit):
            cp.main(["--hermes-home", str(self.dir / "nope")])
        with self.assertRaises(SystemExit):
            cp.main(["--hermes-home", str(self.dir), "--port", "80"])

    def test_custom_port_is_used_everywhere(self):
        cp.main(["--hermes-home", str(self.dir), "--port", "9191"])
        text = (self.dir / "config.yaml").read_text()
        self.assertIn("127.0.0.1:9191/v1", text)
        self.assertNotIn("8080", text)


class MockServerProtocolTests(unittest.TestCase):
    def test_models_card_has_the_engine_fields(self):
        with MockVeystrixServer() as mock:
            status, _, raw = _request(mock, "GET", "/v1/models")
            card = json.loads(raw)["data"][0]
            self.assertEqual(status, 200)
            self.assertEqual(card["id"], "veystrix-flashnext")
            self.assertGreaterEqual(card["context_window"], 1)
            self.assertIn("suggested_client_context_window", card)
            self.assertIn("native_positional_capability", card["context"])

    def test_streaming_has_reasoning_content_then_content_then_done(self):
        with MockVeystrixServer() as mock:
            mock.queue({"text": "Final answer.", "reasoning": "Because."})
            status, headers, raw = _request(mock, "POST", "/v1/chat/completions", {"model": "m", "stream": True, "messages": []})
            events = _sse(raw)
            self.assertEqual(events[-1], "[DONE]")
            deltas = [json.loads(e)["choices"][0]["delta"] for e in events[:-1]]
            reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
            content = "".join(d.get("content", "") for d in deltas)
            self.assertEqual((reasoning, content), ("Because.", "Final answer."))
            self.assertLess(next(i for i, d in enumerate(deltas) if "reasoning_content" in d),
                            next(i for i, d in enumerate(deltas) if "content" in d))

    def test_tool_call_stream_reassembles_and_finishes_with_tool_calls(self):
        with MockVeystrixServer() as mock:
            mock.queue({"tool_calls": [{"id": "call_1", "name": "read_file", "arguments": {"path": "/tmp/x", "n": 1}}]})
            _, _, raw = _request(mock, "POST", "/v1/chat/completions", {"stream": True, "messages": []})
            args, finish, name = "", None, None
            for e in _sse(raw)[:-1]:
                choice = json.loads(e)["choices"][0]
                finish = choice["finish_reason"] or finish
                for call in choice["delta"].get("tool_calls", []):
                    name = call.get("function", {}).get("name") or name
                    args += call.get("function", {}).get("arguments", "")
            self.assertEqual((finish, name, json.loads(args)), ("tool_calls", "read_file", {"path": "/tmp/x", "n": 1}))

    def test_non_streaming_message_shape(self):
        with MockVeystrixServer() as mock:
            mock.queue({"text": "hi", "reasoning": "r"})
            _, _, raw = _request(mock, "POST", "/v1/chat/completions", {"stream": False, "messages": []})
            message = json.loads(raw)["choices"][0]["message"]
            self.assertEqual((message["content"], message["reasoning_content"]), ("hi", "r"))

    def test_engine_style_errors(self):
        with MockVeystrixServer() as mock:
            mock.queue({"error": {"status": 400, "type": "context_length_exceeded", "message": "too long"}})
            status, _, raw = _request(mock, "POST", "/v1/chat/completions", {"messages": []})
            self.assertEqual(status, 400)
            self.assertEqual(json.loads(raw)["error"]["type"], "context_length_exceeded")

    def test_api_key_is_enforced_when_configured_and_origin_is_refused(self):
        with MockVeystrixServer(api_key="secret") as mock:
            self.assertEqual(_request(mock, "GET", "/v1/models")[0], 401)
            self.assertEqual(_request(mock, "GET", "/v1/models", headers={"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(_request(mock, "GET", "/v1/models", headers={"Authorization": "Bearer secret", "Origin": "https://evil.example"})[0], 403)

    def test_server_listens_on_loopback_only(self):
        with MockVeystrixServer() as mock:
            self.assertEqual(mock.httpd.server_address[0], "127.0.0.1")

    def test_a_dropped_client_is_recorded_as_cancelled(self):
        with MockVeystrixServer() as mock:
            mock.queue({"text": "x" * 600, "delay": 0.05})
            sock = socket.create_connection(("127.0.0.1", mock.port), timeout=5)
            body = json.dumps({"stream": True, "messages": []}).encode()
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            sock.recv(200)
            sock.close()
            deadline = time.time() + 10
            while time.time() < deadline and not mock.chat_requests()[0]["cancelled"]:
                time.sleep(0.1)
            self.assertTrue(mock.chat_requests()[0]["cancelled"])


class _Tripwire:
    """A fake HTTP(S) proxy that records every connection attempt and refuses it: proof of no Internet egress."""

    def __init__(self):
        self.seen: list[str] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                conn.settimeout(2)
                first = conn.recv(512).split(b"\r\n")[0].decode(errors="replace")
                self.seen.append(first)
                conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self.sock.close()


@unittest.skipUnless(E2E, "set VEYSTRIX_HERMES_E2E=1 and install Hermes Agent to run the real-client tests")
class HermesEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="hermes-e2e-home-")
        cls.work = tempfile.mkdtemp(prefix="hermes-e2e-work-")
        Path(cls.work, "note.txt").write_text("SENTINEL-4242 hello file")
        cls.mock = MockVeystrixServer().start()
        cp.main(["--hermes-home", cls.home, "--port", str(cls.mock.port)])
        # Warm-up without the tripwire: Hermes prepares its runtime for a new data root on first use.
        cls.mock.queue({"text": "warm"})
        cls.run_hermes(["-z", "warm up"], tripwire=False)
        cls.tripwire = _Tripwire()

    @classmethod
    def tearDownClass(cls):
        try:                                           # leave no background gateway behind
            cls.run_hermes(["gateway", "stop", "--yes"], tripwire=False, timeout=60)
        except Exception:
            pass
        cls.mock.stop()
        cls.tripwire.close()
        shutil.rmtree(cls.home, ignore_errors=True)
        shutil.rmtree(cls.work, ignore_errors=True)

    @classmethod
    def env(cls, tripwire=True):
        env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}
        env.update({"HERMES_HOME": cls.home, "VEYSTRIX_API_KEY": "local", "NO_PROXY": "127.0.0.1,localhost"})
        if tripwire:
            proxy = f"http://127.0.0.1:{cls.tripwire.port}"
            env.update({"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "ALL_PROXY": proxy, "http_proxy": proxy, "https_proxy": proxy})
        return env

    @classmethod
    def run_hermes(cls, args, tripwire=True, timeout=240):
        return subprocess.run([str(HERMES), *args], env=cls.env(tripwire), cwd=cls.work, capture_output=True, text=True, timeout=timeout)

    def chats_since(self, n):
        return self.mock.chat_requests()[n:]

    def test_01_streaming_text_with_reasoning_content_and_local_auth(self):
        n = len(self.mock.chat_requests())
        self.mock.queue({"text": "Hello from the mock.", "reasoning": "Let me think."})
        p = self.run_hermes(["-z", "Say hello"])
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        self.assertEqual(p.stdout.strip(), "Hello from the mock.")            # reasoning is not mixed into the answer
        request = self.chats_since(n)[0]
        self.assertTrue(request["body"]["stream"])
        self.assertEqual(request["body"]["model"], "veystrix-flashnext")
        self.assertEqual(request["authorization"], "Bearer local")
        self.assertTrue(request["body"].get("tools"), "tools are offered to the model")

    def test_02_tool_call_and_tool_result_continuation(self):
        n = len(self.mock.chat_requests())
        self.mock.queue({"tool_calls": [{"id": "call_a", "name": "read_file", "arguments": {"path": str(Path(self.work, "note.txt"))}}]},
                        {"text": "The file says SENTINEL-4242."})
        p = self.run_hermes(["-z", "Read note.txt and tell me what it says"])
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        self.assertIn("SENTINEL-4242", p.stdout)
        second = self.chats_since(n)[1]["body"]["messages"]
        self.assertEqual([m["role"] for m in second][-2:], ["assistant", "tool"])
        self.assertIn("SENTINEL-4242", [m for m in second if m["role"] == "tool"][0]["content"])

    def test_03_multi_turn_session_persists_across_invocations(self):
        n = len(self.mock.chat_requests())
        self.mock.queue({"text": "Noted."}, {"text": "PINEAPPLE-77."})
        self.assertEqual(self.run_hermes(["-z", "My code word is PINEAPPLE-77"]).returncode, 0)
        p = self.run_hermes(["--continue", "-z", "What is my code word?"])
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        self.assertIn("PINEAPPLE-77", json.dumps(self.chats_since(n)[1]["body"]["messages"]))

    def test_04_context_length_error_is_surfaced_and_not_retried(self):
        n = len(self.mock.chat_requests())
        self.mock.queue({"error": {"status": 400, "type": "context_length_exceeded", "message": "prompt too long"}})
        p = self.run_hermes(["-z", "trigger overflow"])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("prompt too long", p.stdout + p.stderr)
        self.assertEqual(len(self.chats_since(n)), 1)

    def test_05_admission_refusal_is_retried_at_most_once(self):
        n = len(self.mock.chat_requests())
        self.mock.queue({"error": {"status": 503, "type": "memory_admission_refused", "message": "4.9 GiB available, 6.3 GiB required"}},
                        {"error": {"status": 503, "type": "memory_admission_refused", "message": "4.9 GiB available, 6.3 GiB required"}},
                        {"error": {"status": 503, "type": "memory_admission_refused", "message": "4.9 GiB available, 6.3 GiB required"}})
        self.run_hermes(["-z", "trigger refusal"])
        self.assertLessEqual(len(self.chats_since(n)), 2, "api_max_retries is floored at 1 by Hermes: one retry at most")
        with self.mock.state.lock:                    # the third scripted refusal is deliberately unused: do not leak it
            self.mock.state.script.clear()

    def test_06_cancellation_and_reconnection(self):
        """Hermes 0.23.1 `-z` hands the request to a detached background gateway. Ctrl-C on the CLI detaches
        ("accepted work continues at the gateway") and does NOT cancel the model request, so the engine would keep
        prefilling. `hermes gateway stop --yes` does cancel: the server sees the disconnect."""
        self.run_hermes(["gateway", "stop", "--yes"], timeout=60)                  # start from no gateway: deterministic
        n = len(self.mock.chat_requests())
        self.mock.queue({"text": "x" * 400, "delay": 0.2, "hang": 40})
        proc = subprocess.Popen([str(HERMES), "-z", "long answer please"], env=self.env(), cwd=self.work,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        start = time.time()
        def hanging():
            return [r for r in self.mock.chat_requests()[n:] if (r.get("step") or {}).get("hang")]
        while not hanging() and time.time() - start < 120:
            time.sleep(0.3)
        self.assertTrue(hanging(), "the long request never reached the engine")
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("the Hermes CLI did not return on SIGINT")
        proc.stdout.close(); proc.stderr.close()
        request = hanging()[0]
        time.sleep(3)
        self.assertFalse(request["cancelled"], "documented behaviour: Ctrl-C detaches, it does not cancel the backend request")
        stop = self.run_hermes(["gateway", "stop", "--yes"], timeout=60)          # the real cancellation path
        self.assertEqual(stop.returncode, 0, stop.stdout + stop.stderr)
        deadline = time.time() + 15
        while time.time() < deadline and not request["cancelled"]:
            time.sleep(0.2)
        self.assertTrue(request["cancelled"], "stopping the gateway must close the stream so the engine sees the disconnect: " + (stop.stdout + stop.stderr)[-300:])
        self.mock.queue({"text": "Back again."})
        p = self.run_hermes(["-z", "are you there"])                               # reconnection: a new gateway starts
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "Back again."))

    def test_07_context_metadata_is_discovered_and_no_unexpected_endpoints_are_used(self):
        paths = {r["path"] for r in self.mock.requests}
        self.assertIn("/v1/models", paths)                                   # context window discovery
        allowed = {"/v1/chat/completions", "/v1/models", "/health", "/ready"}
        # Hermes probes other server flavours' metadata routes; they are harmless 404s on a loopback server.
        probes = {"/api/tags", "/api/v1/models", "/props", "/v1/props", "/version"}
        self.assertEqual(paths - allowed - probes, set(), "unexpected routes requested")

    def test_08_no_internet_egress_during_any_run(self):
        self.assertEqual(self.tripwire.seen, [], f"Hermes tried to reach: {self.tripwire.seen}")


if __name__ == "__main__":
    unittest.main()
