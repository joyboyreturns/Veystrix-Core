"""The HTTP server as a client sees it: status routes, standard errors, busy answers, disconnects, warm-conversation
protection. A real server on a loopback port, a fake model, no checkpoint."""
import http.client
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from veystrix import server as S
from veystrix.scheduler import Scheduler
from veystrix.tests.test_server import FakeBackend, FakeSession


class SlowBackend(FakeBackend):
    """generate() blocks until released and reports each piece, like a model producing tokens."""

    def __init__(self, pieces):
        super().__init__(pieces)
        self.release = threading.Event()
        self.entered = threading.Event()
        self.generations = 0

    def generate(self, max_tokens, out=None):
        self.generations += 1
        self.entered.set()
        for _ in range(400):
            if self.release.wait(0.05):
                break
            if out:
                out("w ")
        return super().generate(max_tokens, out)


def request(port, method, path, body=None, headers=None, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = json.dumps(body).encode() if body is not None else None
        merged = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, body=data, headers=merged)
        response = conn.getresponse()
        raw = response.read()
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = raw
        return response.status, parsed, dict(response.getheaders())
    finally:
        conn.close()


CHAT = {"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}


class Running:
    def __init__(self, test, backend=None, context_limit=None, max_cold_prefill=None, api_key=None, **scheduler):
        self.session = FakeSession(["ok"])
        if backend is not None:
            self.session.backend = backend
        self.service = S.ModelService(self.session, context_limit, max_cold_prefill)
        self.scheduler = Scheduler(**scheduler)
        self.server = S.VeystrixHTTPServer(("127.0.0.1", 0), self.service, api_key=api_key, scheduler=self.scheduler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        test.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def wait_until(check, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


class StatusRouteTests(unittest.TestCase):
    def test_health_and_ready_report_a_loaded_idle_model(self):
        run = Running(self)
        status, body, _ = request(run.port, "GET", "/health")
        self.assertEqual((status, body["status"], body["ready"], body["busy"]), (200, "ok", True, False))
        self.assertEqual(body["model"], run.service.model)
        self.assertIn("queue", body)
        self.assertEqual(request(run.port, "GET", "/ready")[0], 200)

    def test_ready_is_503_after_a_gpu_lifecycle_fault(self):
        from models.flashnext import pool_lifecycle

        run = Running(self)
        self.addCleanup(lambda: pool_lifecycle._LIFECYCLE_FAULT.__setitem__(0, None))
        pool_lifecycle._LIFECYCLE_FAULT[0] = "GPU synchronization failed: device lost"
        status, body, _ = request(run.port, "GET", "/ready")
        self.assertEqual((status, body["ready"], body["status"]), (503, False, "unavailable"))
        self.assertIn("device lost", body["fault"])
        self.assertEqual(request(run.port, "GET", "/health")[0], 200)     # liveness stays answerable

    def test_models_lists_one_model_with_the_enforced_window_not_the_theoretical_one(self):
        run = Running(self, context_limit=4096)
        status, body, _ = request(run.port, "GET", "/v1/models")
        self.assertEqual(status, 200)
        entry = body["data"][0]
        self.assertEqual((body["object"], entry["object"], entry["id"]), ("list", "model", run.service.model))
        self.assertEqual(entry["context_window"], 4096)
        self.assertTrue(entry["context_limit_enforced"])
        self.assertEqual(entry["max_output_tokens"], 4096 - S.MIN_OUTPUT_RESERVE)
        self.assertIsInstance(entry["created"], int)
        self.assertEqual(body["models"][0]["context_window"], 4096)       # the established card is unchanged

    def test_models_without_a_limit_does_not_claim_an_output_cap(self):
        entry = request(Running(self).port, "GET", "/v1/models")[1]["data"][0]
        self.assertFalse(entry["context_limit_enforced"])
        self.assertIsNone(entry["max_output_tokens"])

    def test_a_single_model_can_be_retrieved_and_an_unknown_one_is_a_standard_404(self):
        run = Running(self)
        status, body, _ = request(run.port, "GET", f"/v1/models/{run.service.model}")
        self.assertEqual((status, body["id"]), (200, run.service.model))
        status, body, _ = request(run.port, "GET", "/v1/models/gpt-4o")
        self.assertEqual(status, 404)
        self.assertEqual((body["error"]["code"], body["error"]["type"]), ("model_not_found", "invalid_request_error"))

    def test_status_routes_need_the_key_when_one_is_configured(self):
        run = Running(self, api_key="secret")
        for path in ("/health", "/ready", "/v1/models"):
            with self.subTest(path=path):
                self.assertEqual(request(run.port, "GET", path)[0], 401)
                self.assertEqual(request(run.port, "GET", path, headers={"Authorization": "Bearer secret"})[0], 200)


class ErrorShapeTests(unittest.TestCase):
    def assertStandard(self, body, kind, code):
        error = body["error"]
        self.assertEqual(set(error), {"message", "type", "param", "code"})
        self.assertEqual((error["type"], error["code"]), (kind, code))
        self.assertTrue(error["message"])

    def test_wrong_key_is_authentication_error_invalid_api_key(self):
        run = Running(self, api_key="secret")
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", CHAT, {"Authorization": "Bearer nope"})
        self.assertEqual(status, 401)
        self.assertStandard(body, "authentication_error", "invalid_api_key")

    def test_unknown_route_and_unsupported_field_and_bad_json_are_invalid_request_errors(self):
        run = Running(self)
        status, body, _ = request(run.port, "POST", "/v1/nothing", CHAT)
        self.assertEqual(status, 404)
        self.assertStandard(body, "invalid_request_error", "not_found")
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "logit_bias": {"1": 1}})
        self.assertEqual(status, 400)
        self.assertStandard(body, "invalid_request_error", None)
        self.assertIn("logit_bias", body["error"]["message"])
        conn = http.client.HTTPConnection("127.0.0.1", run.port, timeout=5)
        conn.request("POST", "/v1/chat/completions", body=b"{not json", headers={"Content-Length": "9"})
        response = conn.getresponse()
        self.assertEqual(response.status, 400)
        self.assertStandard(json.loads(response.read()), "invalid_request_error", None)
        conn.close()

    def test_context_overflow_is_context_length_exceeded(self):
        run = Running(self, context_limit=300)
        run.session.backend.tokenizer.rendered = "x" * 100                 # 100 + 256 reserve > 300
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", CHAT)
        self.assertEqual(status, 400)
        self.assertStandard(body, "invalid_request_error", "context_length_exceeded")
        self.assertEqual(run.session.backend.tape, [])                        # nothing ran, nothing truncated

    def test_a_malformed_model_tool_call_is_a_model_error_not_a_client_error(self):
        run = Running(self)
        run.session.backend.pieces = ['<tool_call>{"name": "t", "arguments": {}}</tool_call>']
        tools = [{"type": "function", "function": {"name": "t", "parameters": {
            "type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}}]
        status, body, headers = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "tools": tools})
        self.assertEqual(status, 422)                                   # non-retryable on purpose
        self.assertStandard(body, "server_error", "model_tool_call_invalid")
        self.assertIn("required parameter 'x'", body["error"]["message"])
        # the same request with a valid call serializes normally: the API shape is not the problem
        run.session.backend.pieces = ['<tool_call>{"name": "t", "arguments": {"x": "1"}}</tool_call>']
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "tools": tools})
        self.assertEqual(status, 200)
        choice = body["choices"][0]
        self.assertEqual((choice["finish_reason"], choice["message"]["content"]), ("tool_calls", None))
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"x": "1"})

    def test_a_tool_request_with_a_one_token_output_cap_is_refused_up_front_with_the_fix(self):
        # The first real round trip: DeepSeek Harness asked for max_completion_tokens=1 because its context window was
        # configured equal to the server's; the model could only emit "<tool_call>". Now that is a 400 before any prefill.
        run = Running(self)
        tools = [{"type": "function", "function": {"name": "glob", "parameters": {"type": "object", "properties": {}}}}]
        for stream in (False, True):
            status, body, _ = request(run.port, "POST", "/v1/chat/completions",
                                      {**CHAT, "tools": tools, "max_tokens": 1, "stream": stream})
            self.assertEqual(status, 400)
            self.assertStandard(body, "invalid_request_error", "max_tokens_too_small")
            self.assertIn("suggested_client_context_window", body["error"]["message"])
        self.assertEqual(run.session.backend.tape, [])                    # nothing was prefilled
        # without tools a tiny cap is a legitimate request
        status, _, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "max_tokens": 1})
        self.assertEqual(status, 200)

    def test_models_endpoint_publishes_the_client_window_that_leaves_the_clients_own_reserve(self):
        run = Running(self, context_limit=16384)
        entry = request(run.port, "GET", "/v1/models")[1]["data"][0]
        self.assertEqual(entry["context_window"], 16384)
        self.assertEqual(entry["suggested_client_context_window"], 16384 + 4096)

    def test_a_tool_call_cut_off_by_the_output_limit_is_an_explicit_error_not_raw_markup(self):
        # The model started a tool call and a cap that passed the up-front check still ended it: never raw markup.
        run = Running(self)
        tools = [{"type": "function", "function": {"name": "glob", "parameters": {
            "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}]
        run.session.backend.pieces = ["<tool_call>"]
        run.session.backend.generate = lambda max_tokens, out=None: (
            (out("<tool_call>") if out else None) or "<tool_call>",
            SimpleNamespace(finish="length", tokens=1, prompt_tokens=902))
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "tools": tools, "max_tokens": 64})
        self.assertEqual(status, 422)
        self.assertStandard(body, "server_error", "model_tool_call_invalid")
        self.assertIn("output limit", body["error"]["message"])
        self.assertNotIn("choices", body)
        conn = http.client.HTTPConnection("127.0.0.1", run.port, timeout=10)
        conn.request("POST", "/v1/chat/completions", body=json.dumps({**CHAT, "tools": tools, "max_tokens": 64, "stream": True}),
                     headers={"Content-Type": "application/json"})
        raw = conn.getresponse().read().decode()
        conn.close()
        self.assertIn("model_tool_call_invalid", raw)
        self.assertNotIn('"content": "<tool_call>"', raw)
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))

    def test_plain_text_that_hits_the_output_limit_is_still_a_normal_length_finish(self):
        run = Running(self)
        tools = [{"type": "function", "function": {"name": "glob", "parameters": {"type": "object", "properties": {}}}}]
        run.session.backend.generate = lambda max_tokens, out=None: (
            (out("The files are") if out else None) or "The files are",
            SimpleNamespace(finish="length", tokens=3, prompt_tokens=10))
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "tools": tools, "max_tokens": 64})
        self.assertEqual((status, body["choices"][0]["finish_reason"]), (200, "length"))
        self.assertEqual(body["choices"][0]["message"]["content"], "The files are")

    def test_ordinary_text_is_never_turned_into_a_tool_call(self):
        run = Running(self)
        tools = [{"type": "function", "function": {"name": "glob", "parameters": {
            "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}]
        for text in ("I would use the glob tool to list the files.",
                     '{"name": "glob", "arguments": {"pattern": "*"}}',                  # call-shaped JSON, no tool_call tags
                     "glob(pattern='*')",
                     "Here is the call: <tool>glob</tool>"):
            with self.subTest(text=text):
                run.session.backend.pieces = [text]
                for stream in (False, True):
                    payload = {**CHAT, "tools": tools, "stream": stream}
                    if not stream:
                        status, body, _ = request(run.port, "POST", "/v1/chat/completions", payload)
                        choice = body["choices"][0]
                        self.assertEqual((status, choice["finish_reason"]), (200, "stop"))
                        self.assertNotIn("tool_calls", choice["message"])
                        self.assertEqual(choice["message"]["content"], text)
                    else:
                        conn = http.client.HTTPConnection("127.0.0.1", run.port, timeout=10)
                        conn.request("POST", "/v1/chat/completions", body=json.dumps(payload),
                                     headers={"Content-Type": "application/json"})
                        raw = conn.getresponse().read().decode()
                        conn.close()
                        self.assertNotIn("tool_calls", raw)
                        self.assertIn('"finish_reason": "stop"', raw)

    def test_a_model_failure_is_a_server_error(self):
        run = Running(self)
        run.session.backend.generate = mock.Mock(side_effect=RuntimeError("model blew up"))
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", CHAT)
        self.assertEqual(status, 500)
        self.assertStandard(body, "server_error", "internal_error")


class BusyTests(unittest.TestCase):
    def background(self, run, results, name):
        def go():
            results[name] = request(run.port, "POST", "/v1/chat/completions", CHAT, timeout=30)
        thread = threading.Thread(target=go, daemon=True)
        thread.start()
        return thread

    def test_a_second_request_gets_503_server_busy_after_the_bounded_wait_and_health_stays_responsive(self):
        backend = SlowBackend(["done"])
        run = Running(self, backend=backend, max_wait=0.6, max_waiting=2)
        results = {}
        first = self.background(run, results, "first")
        self.assertTrue(backend.entered.wait(5))
        status, health, _ = request(run.port, "GET", "/health", timeout=2)       # answers during generation
        self.assertEqual((status, health["busy"]), (200, True))
        self.assertEqual(health["queue"]["active"]["path"], "/v1/chat/completions")
        began = time.monotonic()
        status, body, headers = request(run.port, "POST", "/v1/chat/completions", CHAT)
        self.assertEqual(status, 503)
        self.assertGreaterEqual(time.monotonic() - began, 0.5)
        self.assertEqual((body["error"]["type"], body["error"]["code"]), ("server_error", "server_busy"))
        self.assertIn("Retry-After", headers)
        self.assertEqual(backend.generations, 1)                                   # the refused one never ran
        backend.release.set()
        first.join(10)
        self.assertEqual(results["first"][0], 200)

    def test_a_full_queue_is_refused_immediately(self):
        backend = SlowBackend(["done"])
        run = Running(self, backend=backend, max_wait=10, max_waiting=1)
        results = {}
        first = self.background(run, results, "first")
        self.assertTrue(backend.entered.wait(5))
        second = self.background(run, results, "second")
        self.assertTrue(wait_until(lambda: run.scheduler.snapshot()["waiting"] == 1))
        began = time.monotonic()
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", CHAT)
        self.assertEqual((status, body["error"]["code"]), (503, "server_busy"))
        self.assertLess(time.monotonic() - began, 1.5)
        backend.release.set()
        first.join(10)
        second.join(10)
        self.assertEqual((results["first"][0], results["second"][0]), (200, 200))
        self.assertEqual(backend.generations, 2)

    def test_a_client_that_hangs_up_while_waiting_never_runs(self):
        backend = SlowBackend(["done"])
        run = Running(self, backend=backend, max_wait=10, max_waiting=2)
        results = {}
        first = self.background(run, results, "first")
        self.assertTrue(backend.entered.wait(5))
        body = json.dumps(CHAT).encode()
        sock = socket.create_connection(("127.0.0.1", run.port), timeout=5)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        self.assertTrue(wait_until(lambda: run.scheduler.snapshot()["waiting"] == 1))
        sock.close()
        self.assertTrue(wait_until(lambda: run.scheduler.snapshot()["dropped"] == 1, 5))
        backend.release.set()
        first.join(10)
        time.sleep(0.3)
        self.assertEqual(backend.generations, 1)
        self.assertEqual(run.scheduler.snapshot()["waiting"], 0)

    def test_a_client_that_hangs_up_mid_generation_ends_it_and_frees_the_server(self):
        backend = SlowBackend(["done"])
        run = Running(self, backend=backend, max_wait=5, max_waiting=2)
        self.addCleanup(backend.release.set)
        body = json.dumps(CHAT).encode()
        with mock.patch.object(S, "GONE_CHECK_SECONDS", 0.05):
            sock = socket.create_connection(("127.0.0.1", run.port), timeout=5)
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                         b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            self.assertTrue(backend.entered.wait(5))
            sock.close()
            self.assertTrue(wait_until(lambda: not run.scheduler.busy, 5), "generation kept running for a gone client")
        self.assertEqual(run.session.backend.tape, [])          # the live state of the abandoned request was dropped
        self.assertGreaterEqual(run.session.backend.reset_count, 1)
        backend.release.set()
        status, _, _ = request(run.port, "POST", "/v1/chat/completions", CHAT)       # and the server still works
        self.assertEqual(status, 200)


TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]


def ask(service, first_user, tools=(), max_tokens=64, hint=None):
    service.session_hint = hint
    return service.complete([{"role": "user", "content": first_user}], list(tools), max_tokens)


class SessionIsolationTests(unittest.TestCase):
    """No request is refused for being short or tool-less; a different conversation replaces the live state and the
    replaced one is rebuilt exactly from its own tokens."""

    LONG = "a" * 600

    def warm(self, **options):
        session = FakeSession(["answer"])
        session.backend.tokenizer.rendered = self.LONG
        service = S.ModelService(session, **options)
        ask(service, "work on A", TOOLS)
        return session, service

    def test_a_short_toolless_title_request_is_served_not_refused(self):
        session, service = self.warm()
        session.backend.tokenizer.rendered = "Generate a title for this chat"
        result = ask(service, "title please", [], 20)
        self.assertEqual(result.text, "answer")
        self.assertEqual((service.sessions.switches, service.evictions), (1, 1))

    def test_the_replaced_conversation_is_rebuilt_exactly_and_never_sees_the_side_request(self):
        session, service = self.warm()
        a_id = service.sessions.live
        session.backend.tokenizer.rendered = "Generate a title for this chat"
        ask(service, "title please", [], 20)
        self.assertNotEqual(service.sessions.live, a_id)
        self.assertTrue(service.sessions.items[a_id]["cold"])
        # conversation A continues: its full prompt, which extends A's earlier tape but not the live (B) one
        a_next = self.LONG + "answer" + "and the next turn"
        session.backend.tokenizer.rendered = a_next
        ask(service, "work on A", TOOLS)
        expected = [ord(c) for c in a_next]
        self.assertEqual(session.backend.tape[:len(expected)], expected, "A was not rebuilt from its own exact tokens")
        title_tokens = [ord(c) for c in "Generate a title for this chat"]
        flat = session.backend.tape
        self.assertFalse(any(flat[i:i + len(title_tokens)] == title_tokens for i in range(len(flat))),
                         "tokens of the side request leaked into the rebuilt conversation")
        entry = service.sessions.items[a_id]
        self.assertEqual((service.sessions.live, entry["cold"], entry["rebuilds"], entry["requests"]), (a_id, False, 2, 2))
        self.assertEqual(service.sessions.switches, 2)

    def test_a_continuation_of_the_live_conversation_is_not_a_switch(self):
        session, service = self.warm()
        session.backend.tokenizer.rendered = self.LONG + "answer" + "next"
        ask(service, "work on A", TOOLS)
        self.assertEqual((service.sessions.switches, service.reused, service.evictions), (0, 1, 0))

    def test_an_explicit_session_id_overrides_the_derived_one(self):
        session = FakeSession(["ok"])
        service = S.ModelService(session)
        ask(service, "same words", [], 8, hint="agent-1")
        ask(service, "same words", [], 8, hint="agent-2")             # same messages, other session: a switch
        self.assertEqual((service.sessions.live, service.sessions.switches), ("id:agent-2", 1))
        session.backend.tokenizer.rendered = "other prompt"
        ask(service, "completely different first message", [], 8, hint="agent-2")   # same session, new text: no switch
        self.assertEqual(service.sessions.switches, 1)

    def test_keys_are_stable_across_turns_and_differ_between_conversations(self):
        from veystrix.sessions import derive_key

        turn1 = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        turn2 = turn1 + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]
        self.assertEqual(derive_key(turn1), derive_key(turn2))
        self.assertNotEqual(derive_key(turn1), derive_key([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]))
        self.assertNotEqual(derive_key(turn1), derive_key([{"role": "system", "content": "t"}, {"role": "user", "content": "u"}]))
        self.assertEqual(derive_key(turn1, "abc"), "id:abc")

    def test_explicit_ids_come_from_the_header_or_the_metadata(self):
        from veystrix.sessions import explicit_id

        self.assertEqual(explicit_id({"X-Veystrix-Session": "h"}, {"metadata": {"session_id": "m"}}), "h")
        self.assertEqual(explicit_id({}, {"metadata": {"session_id": "m"}}), "m")
        self.assertEqual(explicit_id({}, {"metadata": {"conversation_id": "c"}}), "c")
        self.assertIsNone(explicit_id({}, {"metadata": "nope"}))
        self.assertIsNone(explicit_id({"X-Veystrix-Session": "  "}, {}))

    def test_the_registry_is_bounded_and_a_failed_request_marks_the_live_state_cold(self):
        from veystrix.sessions import SessionRegistry

        registry = SessionRegistry(capacity=3)
        for index in range(5):
            registry.activate(f"k{index}", prompt_tokens=1, reused=False, had_state=index > 0)
        self.assertEqual(len(registry.items), 3)
        self.assertEqual(registry.switches, 4)
        registry.dropped()
        self.assertIsNone(registry.live)
        self.assertTrue(registry.items["k4"]["cold"])

    def test_a_failed_generation_drops_the_live_session(self):
        session, service = self.warm()
        session.backend.generate = mock.Mock(side_effect=RuntimeError("model blew up"))
        session.backend.tokenizer.rendered = self.LONG + "answer" + "more"
        with self.assertRaises(RuntimeError):
            ask(service, "work on A", TOOLS)
        self.assertIsNone(service.sessions.live)

    def test_http_requests_are_served_and_the_header_names_the_session(self):
        run = Running(self)
        run.session.backend.tokenizer.rendered = self.LONG
        status, _, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "tools": TOOLS},
                               {"X-Veystrix-Session": "agent-7"})
        self.assertEqual(status, 200)
        health = request(run.port, "GET", "/health")[1]
        self.assertEqual(health["cache"]["sessions"]["live"], "id:agent-7")
        run.session.backend.tokenizer.rendered = "Generate a title"
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "max_tokens": 20})
        self.assertEqual(status, 200, body)                              # short and tool-less: served
        cache = request(run.port, "GET", "/health")[1]["cache"]
        self.assertEqual((cache["sessions"]["switches"], cache["evictions"]), (1, 1))


class ColdPrefillTierTests(unittest.TestCase):
    def test_a_rebuild_above_the_admitted_tier_is_refused_and_the_live_state_is_untouched(self):
        session = FakeSession(["ok"])
        session.backend.tokenizer.rendered = "a" * 100
        service = S.ModelService(session, max_cold_prefill=120)
        ask(service, "first", [], 8)
        tape, resets = list(session.backend.tape), session.backend.reset_count
        session.backend.tokenizer.rendered = "b" * 121                      # a different conversation, 121 > 120
        with self.assertRaises(S.ContextLimitError) as caught:
            ask(service, "second", [], 8)
        self.assertEqual((caught.exception.status, caught.exception.code), (400, "prefill_budget_exceeded"))
        self.assertEqual((session.backend.tape, session.backend.reset_count), (tape, resets))
        self.assertEqual(service.refused_cold_prefill, 1)
        session.backend.tokenizer.rendered = "b" * 120                      # exactly at the tier is allowed
        ask(service, "second", [], 8)

    def test_only_the_new_tokens_of_a_continuation_count_against_the_tier(self):
        session = FakeSession(["ok"])
        session.backend.tokenizer.rendered = "a" * 100
        service = S.ModelService(session, max_cold_prefill=120)
        ask(service, "first", [], 8)
        session.backend.tokenizer.rendered = "a" * 100 + "ok" + "c" * 90     # 192 total, 90 new
        ask(service, "first", [], 8)
        self.assertEqual((service.reused, service.refused_cold_prefill), (1, 0))

    def test_no_tier_means_no_limit_and_the_http_error_names_the_budget(self):
        session = FakeSession(["ok"])
        session.backend.tokenizer.rendered = "a" * 5000
        ask(S.ModelService(session), "x", [], 8)                              # no limit configured
        run = Running(self, max_cold_prefill=50)
        run.session.backend.tokenizer.rendered = "a" * 60
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", CHAT)
        self.assertEqual(status, 400)
        self.assertEqual((body["error"]["code"], body["error"]["type"]), ("prefill_budget_exceeded", "invalid_request_error"))
        self.assertEqual(request(run.port, "GET", "/health")[1]["cache"]["max_cold_prefill"], 50)


if __name__ == "__main__":
    unittest.main()
