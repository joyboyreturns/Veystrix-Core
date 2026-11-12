"""Full-transcript inference adapters. No model weights are opened here."""
from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import socket
import threading
import time
from typing import Callable, Protocol

MAX_RESPONSE_BYTES = 256_000
RETRY_CODES = frozenset({"busy", "server_busy", "model_busy"})
ENGINE_ERROR_CODES = frozenset({"invalid_api_key", "model_not_found", "model_handoff_required",
                                "model_lifecycle_fault", "engine_unavailable", "generation_cancelled",
                                "context_length_exceeded", "invalid_request", "internal_error"})


class RetryableInference(RuntimeError):
    """The server explicitly declined to start generation."""


class InferenceBlocked(RuntimeError):
    """The backend could not safely accept or finish the request."""


@dataclass(frozen=True)
class Request:
    task_id: str
    agent_id: str
    model: str
    messages: list[dict]
    tools: list[dict]
    max_tokens: int
    workflow_id: str | None = None


@dataclass(frozen=True)
class Response:
    message: dict
    finish: str = "stop"
    usage: dict = field(default_factory=dict)


class Backend(Protocol):
    def complete(self, request: Request, cancelled: Callable[[], bool]) -> Response: ...


class LocalBackend:
    """Use the existing single-slot server through its OpenAI-compatible API."""
    def __init__(self, port: int, api_key: str | None = None, timeout: float = 120):
        if not 1 <= port <= 65535 or not 0 < timeout <= 1800:
            raise ValueError("invalid loopback port or inference timeout")
        if api_key and any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError("engine API key contains invalid characters")
        self.port, self.api_key, self.timeout = port, api_key, timeout

    def complete(self, request: Request, cancelled: Callable[[], bool]) -> Response:
        if cancelled():
            raise InferenceBlocked("generation cancelled before sending")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        headers = {"Content-Type": "application/json", "X-Veystrix-Session": request.agent_id}
        if request.workflow_id:
            headers["X-Veystrix-Workflow"] = request.workflow_id
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {"model": request.model, "messages": request.messages,
                   "max_completion_tokens": request.max_tokens, "stream": False}
        if request.tools:
            payload["tools"] = request.tools
            payload["parallel_tool_calls"] = False
        finished = threading.Event()
        watcher = None
        deadline = time.monotonic() + self.timeout

        def watch(sock):
            while not finished.wait(0.1):
                if cancelled() or time.monotonic() >= deadline:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return

        try:
            conn.connect()
            if cancelled():
                raise InferenceBlocked("generation cancelled before sending")
            watcher = threading.Thread(target=watch, args=(conn.sock,), daemon=True)
            watcher.start()
            conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
            reply = conn.getresponse()
            body = reply.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise InferenceBlocked("loopback request failed or timed out; check server state before retrying") from exc
        finally:
            finished.set()
            conn.close()
            if watcher is not None:
                watcher.join()
        if len(body) > MAX_RESPONSE_BYTES:
            raise InferenceBlocked("backend response exceeded its size limit")
        try:
            data = json.loads(body)
            if reply.status != 200:
                code = data.get("error", {}).get("code", "")
                if isinstance(code, str) and reply.status in (409, 503) and code in RETRY_CODES:
                    raise RetryableInference("engine is busy")
                detail = f", {code}" if isinstance(code, str) and code in ENGINE_ERROR_CODES else ""
                raise InferenceBlocked(f"engine rejected the request (HTTP {reply.status}{detail})")
            choice = data["choices"][0]
            return Response(choice["message"], choice["finish_reason"], data.get("usage", {}))
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise InferenceBlocked(f"invalid engine response (HTTP {reply.status})") from exc


class ScriptedBackend:
    """Deterministic backend for offline demos. Fixtures are keyed by task ID."""
    def __init__(self, scripts: dict):
        if not isinstance(scripts, dict):
            raise ValueError("mock fixture must be a JSON object keyed by task ID")
        self.scripts = scripts

    def complete(self, request: Request, cancelled: Callable[[], bool]) -> Response:
        turn = sum(message["role"] == "assistant" for message in request.messages)
        try:
            script = self.scripts[request.task_id][turn]
            if isinstance(script, str):
                return Response({"content": script})
            delay = script.get("delay_seconds", 0)
            if type(delay) not in (int, float) or not 0 <= delay <= 10:
                raise ValueError("mock delay must be between zero and ten seconds")
            end = time.monotonic() + delay
            while time.monotonic() < end:
                if cancelled():
                    raise InferenceBlocked("mock generation cancelled")
                time.sleep(min(0.05, max(0, end - time.monotonic())))
            return Response(script["message"], script.get("finish", "stop"), script.get("usage", {}))
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise InferenceBlocked(f"missing or invalid mock response for {request.task_id}, turn {turn + 1}") from exc
