#!/usr/bin/env python3
"""A synthetic Veystrix-compatible server for offline integration tests. It loads NO model.

It imitates the engine's HTTP surface closely enough to exercise a client's protocol handling:

  GET  /health, /ready            liveness/readiness JSON (same field names as the engine)
  GET  /v1/models[/<id>]          the engine's model card shape, incl. context_window and suggested_client_context_window
  POST /v1/chat/completions       streaming (SSE) and non-streaming; reasoning_content, tool calls, finish reasons,
                                  usage, ': generating' keep-alive comments, engine-style JSON errors

Behaviour per request is scripted with `queue(...)`; with an empty script it answers a short default text.
Every request is recorded (method, path, selected headers, parsed body) for assertions. A client that disconnects
mid-stream is noticed and recorded as `cancelled`.

Not a model, not the real engine: it proves wire-protocol behaviour only. It says nothing about speed or quality.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_ID = "macqwen-flashnext"
CONTEXT_WINDOW = 32768


class MockState:
    def __init__(self, api_key: str | None, context_window: int, model_id: str):
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.script: list[dict] = []
        self.api_key = api_key
        self.context_window = context_window
        self.model_id = model_id
        self.event = threading.Event()

    def next_step(self) -> dict:
        with self.lock:
            return self.script.pop(0) if self.script else {"text": "OK from the mock Veystrix engine."}


def _model_entry(state: MockState) -> dict:
    return {
        "id": state.model_id, "object": "model", "created": 1, "owned_by": "veystrix",
        "context_window": state.context_window, "context_limit_enforced": True,
        "max_output_tokens": state.context_window - 256,
        "suggested_client_context_window": state.context_window + 4096,
        "context": {"native_positional_capability": 262144, "configured_server_limit": state.context_window,
                    "memory_admitted_max_cold_prefill": state.context_window, "validated_on_this_machine": None},
        "reasoning": {"efforts": ["low", "medium", "high", "xhigh"], "default": "thinking off, effort medium"},
    }


def _error(status: int, message: str, etype: str, code: str | None = None) -> tuple[int, dict]:
    return status, {"error": {"message": message, "type": etype, "code": code or etype}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VeystrixMock/1.0"
    state: MockState

    def log_message(self, *args):          # keep test output quiet
        pass

    # -- plumbing
    def _record(self, body: Any = None) -> dict:
        entry = {"method": self.command, "path": self.path.split("?")[0], "time": time.time(),
                 "authorization": self.headers.get("Authorization"), "origin": self.headers.get("Origin"),
                 "user_agent": self.headers.get("User-Agent"), "body": body, "cancelled": False}
        with self.state.lock:
            self.state.requests.append(entry)
        self.state.event.set()
        return entry

    def _send_json(self, status: int, payload: dict, extra: dict | None = None):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        if self.state.api_key is None:
            return True
        return self.headers.get("Authorization") == f"Bearer {self.state.api_key}"

    def _guard(self) -> bool:
        """Foreign Origin and bad keys are refused exactly as the engine does. True means handled (refused)."""
        if self.headers.get("Origin"):
            self._send_json(*_error(403, "origin not allowed", "forbidden_origin"))
            return True
        if not self._authorized():
            self._send_json(*_error(401, "invalid API key", "invalid_api_key"))
            return True
        return False

    # -- routes
    def do_GET(self):
        self._record()
        if self._guard():
            return
        path = self.path.split("?")[0]
        if path in ("/health", "/ready"):
            self._send_json(200, {"status": "ok", "ready": True, "model": self.state.model_id, "busy": False,
                                  "queue": {"depth": 0}, "context_limit": self.state.context_window,
                                  "max_cold_prefill": self.state.context_window,
                                  "memory_policy": {"can_admit": True, "stop_reason": None}})
        elif path == "/v1/models":
            card = _model_entry(self.state)
            self._send_json(200, {"object": "list", "data": [card], "models": [card]})
        elif path.startswith("/v1/models/"):
            if path.rsplit("/", 1)[1] == self.state.model_id:
                self._send_json(200, _model_entry(self.state))
            else:
                self._send_json(*_error(404, "the model does not exist", "model_not_found"))
        else:
            self._send_json(*_error(404, "route not found", "not_found"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        entry = self._record(body)
        if self._guard():
            return
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._send_json(*_error(404, "route not found", "not_found"))
            return
        if not isinstance(body, dict):
            self._send_json(*_error(400, "invalid JSON body", "invalid_request_error"))
            return
        step = self.state.next_step()
        entry["step"] = {k: v for k, v in step.items() if k != "arguments_blob"}
        if "error" in step:
            e = step["error"]
            self._send_json(*_error(e.get("status", 400), e.get("message", "error"), e.get("type", "invalid_request_error"), e.get("code")))
            return
        if body.get("stream"):
            self._stream(body, step, entry)
        else:
            self._complete(body, step)

    # -- non-streaming
    def _message(self, step: dict) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": step.get("text", "")}
        if step.get("reasoning"):
            message["reasoning_content"] = step["reasoning"]
        calls = self._tool_calls(step)
        if calls:
            message["tool_calls"] = calls
            message["content"] = step.get("text") or None
        return message

    @staticmethod
    def _tool_calls(step: dict) -> list[dict]:
        out = []
        for i, call in enumerate(step.get("tool_calls") or []):
            out.append({"index": i, "id": call.get("id", f"call_{i}"), "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments", {}))}})
        return out

    def _complete(self, body: dict, step: dict):
        finish = step.get("finish") or ("tool_calls" if step.get("tool_calls") else "stop")
        self._send_json(200, {"id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
                              "model": self.state.model_id,
                              "choices": [{"index": 0, "message": self._message(step), "finish_reason": finish}],
                              "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}},
                        {"X-Veystrix-Reasoning": "off"})

    # -- streaming (SSE)
    def _write_event(self, payload: dict | str) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.wfile.write(f"data: {text}\n\n".encode())
        self.wfile.flush()

    def _chunk(self, delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
        chunk = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": self.state.model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            chunk["usage"] = usage
        return chunk

    def _stream(self, body: dict, step: dict, entry: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        delay = float(step.get("delay", 0))
        try:
            self._write_event(self._chunk({"role": "assistant"}))
            if step.get("keepalive"):
                self.wfile.write(b": generating\n\n")
                self.wfile.flush()
            if step.get("hang"):                       # a long prefill: nothing but keep-alives
                end = time.time() + float(step["hang"])
                while time.time() < end:
                    self.wfile.write(b": generating\n\n")
                    self.wfile.flush()
                    time.sleep(0.2)
            for piece in _pieces(step.get("reasoning", "")):
                self._write_event(self._chunk({"reasoning_content": piece}))
                time.sleep(delay)
            for n, piece in enumerate(_pieces(step.get("text", ""))):
                self._write_event(self._chunk({"content": piece}))
                time.sleep(delay)
                if step.get("abort_after") and n + 1 >= step["abort_after"]:
                    return                              # the connection drops mid-answer
            calls = self._tool_calls(step)
            for call in calls:
                blob = call["function"]["arguments"]
                self._write_event(self._chunk({"tool_calls": [{"index": call["index"], "id": call["id"], "type": "function",
                                                               "function": {"name": call["function"]["name"], "arguments": ""}}]}))
                for part in _pieces(blob, 12):
                    self._write_event(self._chunk({"tool_calls": [{"index": call["index"], "function": {"arguments": part}}]}))
            finish = step.get("finish") or ("tool_calls" if calls else "stop")
            usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
            self._write_event(self._chunk({}, finish, usage if (body.get("stream_options") or {}).get("include_usage") else None))
            self._write_event("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            entry["cancelled"] = True                   # the client went away: this is how a cancellation looks server-side


def _pieces(text: str, size: int = 6) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []


class MockVeystrixServer:
    """In-process mock. ``with MockVeystrixServer() as mock:`` then point a client at ``mock.base_url``."""

    def __init__(self, port: int = 0, api_key: str | None = None, context_window: int = CONTEXT_WINDOW, model_id: str = MODEL_ID):
        self.state = MockState(api_key, context_window, model_id)
        handler = type("BoundHandler", (Handler,), {"state": self.state})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)    # loopback only, like the engine
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def requests(self) -> list[dict]:
        with self.state.lock:
            return list(self.state.requests)

    def chat_requests(self) -> list[dict]:
        return [r for r in self.requests if r["path"] == "/v1/chat/completions"]

    def queue(self, *steps: dict) -> None:
        with self.state.lock:
            self.state.script.extend(steps)

    def start(self) -> "MockVeystrixServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--api-key", default=None, help="require this bearer key (default: accept anything, like loopback mode)")
    ap.add_argument("--context-window", type=int, default=CONTEXT_WINDOW)
    args = ap.parse_args()
    server = MockVeystrixServer(args.port, args.api_key, args.context_window).start()
    print(f"mock Veystrix (no model) on {server.base_url}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
