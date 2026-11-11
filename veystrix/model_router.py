"""Local model router: one process owns the active model and stable API port.

Experimental: request state is serialized, and cross-session switches need
explicit handoff. Model loading and inference are never retried automatically.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from veystrix import model_registry
from veystrix.model_coordinator import ModelCoordinator, LifecycleError, ModelNotFound, ModelBusy, HandoffRequired
from veystrix.model_process import ProcessDriver
from veystrix.server import VeystrixHandler, RequestError
from veystrix.sessions import derive_key, explicit_id


def session_owner(headers, payload):
    hint = explicit_id(headers, payload)
    if hint:
        return "explicit:" + hint
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        return None
    if not any(m.get("role") == "user" for m in messages):
        return None
    try:
        return derive_key(messages, tools=payload.get("tools") or [])
    except (TypeError, AttributeError):
        return None


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    def __init__(self, addr, coordinator, driver, *, api_key=None, inventory=model_registry.discover):
        super().__init__(addr, RouterHandler)
        self.coordinator = coordinator
        self.driver = driver
        self.api_key = api_key
        self.allowed_origins = ()
        self.inventory = inventory


class RouterHandler(VeystrixHandler):
    protocol_version = "HTTP/1.1"
    def _access(self):
        if self._reject_foreign_origin():
            return False
        if not self._authorized():
            self._error("invalid API key", 401, "invalid_api_key")
            return False
        return True

    def _catalog(self):
        models = self.server.inventory()
        active = self.server.coordinator.snapshot().model_id
        return [{"id": m.id, "object": "model", "created": 0, "owned_by": "veystrix",
                 "name": m.display_name, "family": m.family,
                 "loaded": m.id == active, "verification": m.verification} for m in models]

    def do_GET(self):
        if not self._access(): return
        path = urlsplit(self.path).path
        if path == "/v1/models":
            self._json({"object": "list", "data": self._catalog()})
        elif path == "/v1/installed-models":
            active = self.server.coordinator.snapshot().model_id
            self._json({"object": "list", "data": [dict(m.as_dict(), loaded=m.id == active) for m in self.server.inventory()]})
        elif path.startswith("/v1/models/"):
            entry = next((m for m in self._catalog() if m["id"] == path[len("/v1/models/"):]), None)
            if entry: self._json(entry)
            else: self._error("model not found", 404, "model_not_found")
        elif path in ("/health", "/ready"):
            snap = self.server.coordinator.snapshot()
            health = self.server.driver.health() if snap.state == "ready" else None
            if health and health.get("installed_model_id") == snap.model_id:
                health["router_state"] = snap.state
                health["busy"] = snap.active
                self._json(health, 200 if path == "/health" or health.get("ready") else 503)
            else:
                self._json({"status": snap.state, "ready": False, "model": snap.model_id,
                            "busy": snap.active, "fault": snap.error}, 200 if path == "/health" else 503)
        else:
            self._error("route not found", 404, "not_found")

    def do_POST(self):
        if not self._access(): return
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            self._error("route not found", 404, "not_found"); return
        try:
            payload = self._payload()
            requested = payload.get("model") or self.server.coordinator.snapshot().model_id
            if not isinstance(requested, str) or not requested:
                raise ModelNotFound("choose an installed model")
            # Legacy family aliases remain accepted only when the family has exactly one installation.
            if requested in ("veystrix-flashnext", "veystrix-nemotron35-lightning"):
                matches = [m for m in self.server.inventory() if "veystrix-" + m.family == requested]
                if len(matches) != 1:
                    raise ModelNotFound("model family is absent or ambiguous; select an installation ID")
                requested = matches[0].id
                payload["model"] = requested
            owner = session_owner(self.headers, payload)
            with self.server.coordinator.request(requested, owner=owner, timeout=20, cancelled=self._client_gone):
                self._forward(path, payload)
        except RequestError as exc:
            self._error(exc, exc.status, exc.code)
        except LifecycleError as exc:
            status = 404 if isinstance(exc, ModelNotFound) else 409 if isinstance(exc, HandoffRequired) else 503
            self._error(exc, status, exc.code, headers={"x-should-retry": "false"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, http.client.HTTPException) as exc:
            if not getattr(self, "_response_started", False):
                self._error(f"engine connection failed: {exc}", 503, "engine_unavailable", headers={"x-should-retry": "false"})
            else:
                self.close_connection = True

    def _forward(self, path, payload):
        driver = self.server.driver
        body = json.dumps(payload).encode()
        conn = http.client.HTTPConnection("127.0.0.1", driver.port, timeout=7200)
        try:
            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + driver.api_key}
            for name in ("X-Session-Id", "X-Veystrix-Session", "X-Cache-Id", "anthropic-version"):
                if self.headers.get(name): headers[name] = self.headers[name]
            conn.request("POST", path, body=body, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            self._response_started = True
            for name in ("Content-Type", "Content-Length", "Retry-After", "x-should-retry", "X-Veystrix-Reasoning"):
                value = response.getheader(name)
                if value is not None: self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                chunk = response.read1(65536)
                if not chunk: break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Veystrix model router")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pool-gb", default="0")
    parser.add_argument("--prefix-dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("flashnext", "nemotron35-lightning"), default="flashnext")
    args = parser.parse_args(argv)
    initial_id = model_registry.installation_id(args.model, args.checkpoint)
    # Do not start the listener with an installation that discovery cannot validate.
    if model_registry.resolve(initial_id) is None:
        parser.error("selected model was not found in the configured model root or is incomplete")
    driver = ProcessDriver(pool_gb=args.pool_gb, prefix_dir=args.prefix_dir)
    coordinator = ModelCoordinator(model_registry.resolve, driver)
    server = RouterServer((args.host, args.port), coordinator, driver, api_key=os.environ.get("VEYSTRIX_SERVER_API_KEY"))
    stopping = threading.Event()
    def shutdown(_signum, _frame):
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    previous = {sig: signal.signal(sig, shutdown) for sig in (signal.SIGINT, signal.SIGTERM)}
    def startup():
        try:
            with coordinator.request(initial_id, cancelled=stopping.is_set):
                pass
        except LifecycleError as exc:
            print(f"initial model load failed: {exc}", flush=True)
    thread = threading.Thread(target=startup, daemon=True)
    thread.start()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stopping.set()
        server.server_close()
        thread.join(timeout=5)
        if coordinator._gate.acquire(timeout=90):
            try:
                driver.unload()
            finally:
                coordinator._gate.release()
        for sig, handler in previous.items(): signal.signal(sig, handler)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
