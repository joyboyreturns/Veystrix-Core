"""Offline verification of the workflow runner and its persisted state."""
from __future__ import annotations

import copy
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from veystrix.workflows.__main__ import _report
from veystrix.workflows.backends import InferenceBlocked, LocalBackend, Request, Response, RetryableInference, ScriptedBackend
from veystrix.workflows.engine import Runner
from veystrix.workflows.state import Settings, WorkflowStore
from veystrix.workflows.workspace import WorkflowRepo


def task(name, role="explorer", deps=()):
    return {"id": name, "role": role, "objective": f"Objective for {name}", "depends_on": list(deps)}


def call(name, **args):
    return {"message": {"tool_calls": [{"id": "call-1", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(args)}}]},
            "finish": "tool_calls"}


def by_id(status, task_id):
    return next(task for task in status["tasks"] if task["id"] == task_id)


class Workflows(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.git("init", "-q")
        (self.repo / "sample.py").write_text("VALUE = 0\n")
        (self.root / "outside.txt").write_text("outside secret")
        (self.repo / "outside.txt").symlink_to(self.root / "outside.txt")
        self.git("add", ".")
        self.git("-c", "user.name=Workflow Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial")
        (self.repo / "sample.py").write_text("VALUE = 99\n")
        (self.repo / "untracked.txt").write_text("untracked")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, text=True).stdout

    def create(self, plan=None, **options):
        settings = Settings(mode="swarm", **options) if "mode" not in options else Settings(**options)
        return WorkflowStore.create(self.root / "goal", self.repo, "Inspect the fixture repository", settings, plan)

    def test_swarm_delegates_with_isolated_contexts_and_serial_inference(self):
        store = self.create()
        fixture = ScriptedBackend({
            "plan": [json.dumps([task("left"), task("right"), task("followup", deps=("left",))])],
            "left": [call("read_file", path="sample.py"), "left public result"],
            "right": [call("list_dir"), "right public result"],
            "followup": ["followup result"],
            "summary": ["coordinator result"],
        })
        records = []
        active = peak = 0
        barrier = threading.Barrier(2)

        class Backend:
            def complete(self, request, cancelled):
                nonlocal active, peak
                active += 1
                peak = max(active, peak)
                try:
                    records.append(copy.deepcopy(request))
                    time.sleep(0.01)
                    return fixture.complete(request, cancelled)
                finally:
                    active -= 1

        original = WorkflowRepo.call

        def rendezvous(repo, name, args):
            barrier.wait(timeout=4)
            return original(repo, name, args)

        with patch.object(WorkflowRepo, "call", rendezvous):
            status = Runner(store, Backend()).run()
        self.assertEqual(status["state"], "completed", status)
        self.assertEqual(peak, 1)
        self.assertEqual(len({request.agent_id for request in records}), 5)
        left = [request for request in records if request.task_id == "left"]
        right = [request for request in records if request.task_id == "right"]
        self.assertEqual(len(left[0].messages), 2)
        self.assertEqual(len(right[0].messages), 2)
        self.assertIn("VALUE = 0", left[1].messages[-1]["content"])
        self.assertNotIn("untracked.txt", right[1].messages[-1]["content"])
        followup = next(request for request in records if request.task_id == "followup")
        self.assertIn("left public result", followup.messages[1]["content"])
        self.assertNotIn("right public result", followup.messages[1]["content"])
        summary = records[-1]
        self.assertEqual(summary.task_id, "summary")
        self.assertIn("left public result", summary.messages[1]["content"])
        self.assertIn("right public result", summary.messages[1]["content"])
        self.assertNotIn("VALUE = 0", summary.messages[1]["content"])

    def test_explicit_overnight_keeps_proposals_isolated_and_reports_overlap(self):
        store = self.create([task("first", "coder"), task("second", "coder"),
                             task("review", "reviewer", ("first", "second"))],
                            mode="overnight", allow_edits=True)
        fixture = ScriptedBackend({
            "first": [call("replace_text", path="sample.py", old_text="0", new_text="1"), "first report"],
            "second": [call("replace_text", path="sample.py", old_text="0", new_text="2"), "second report"],
            "review": [call("write_file", path="forbidden.txt", content="denied"), "review report"],
            "summary": ["summary report"],
        })
        status = Runner(store, fixture).run()
        self.assertEqual(status["state"], "completed", status)
        for index, name in enumerate(("first", "second"), 1):
            result = by_id(status, name)["result"]
            workspace = store.directory / result["workspace"]
            self.assertEqual((workspace / "sample.py").read_text(), f"VALUE = {index}\n")
            self.assertEqual(result["syntax_checked"], ["sample.py"])
            self.assertTrue(result["requires_manual_review"])
        self.assertEqual((self.repo / "sample.py").read_text(), "VALUE = 99\n")
        self.assertEqual((store.directory / "base" / "sample.py").read_text(), "VALUE = 0\n")
        self.assertFalse((store.directory / "base" / "forbidden.txt").exists())
        self.assertEqual(by_id(status, "review")["result"]["tool_errors"], 1)
        self.assertIn("sample.py` from first, second", _report(status))

    def test_dual_model_batches_roles_and_keeps_agent_sessions_distinct(self):
        store = self.create([task("architecture", "architect"), task("first", "coder", ("architecture",)),
                             task("review", "reviewer", ("first",)), task("second", "tester", ("architecture",))],
                            model="qwen", executor_model="nemotron")
        requests = []

        class Backend:
            def complete(self, request, cancelled):
                requests.append(request)
                return Response({"content": "Report from " + request.task_id})

        status = Runner(store, Backend()).run()
        self.assertEqual(status["state"], "completed", status)
        self.assertEqual([item.model for item in requests], ["qwen", "nemotron", "nemotron", "qwen", "qwen"])
        self.assertEqual({item.task_id for item in requests[1:3]}, {"first", "second"})
        self.assertEqual(len({item.agent_id for item in requests}), 5)
        self.assertEqual({item.workflow_id for item in requests}, {status["settings"]["workflow_id"]})
        self.assertIn("Report from architecture", requests[1].messages[1]["content"])
        self.assertIn("Executor model `nemotron`", _report(status))

    def test_permission_failures_and_symlink_do_not_expose_other_files(self):
        store = self.create(mode="single", allow_edits=True)
        fixture = ScriptedBackend({"main": [
            call("read_file", path="outside.txt"),
            call("read_file", path="../outside.txt"),
            call("write_file", path=".git/hooks/pre-commit", content="denied"),
            call("run_command", command="false"),
            call("web_search", query="external"),
            call("write_file", path="invalid.py", content="="),
            "reported permission errors",
        ]})
        status = Runner(store, fixture).run()
        result = by_id(status, "main")["result"]
        self.assertEqual(result["tool_errors"], 6)
        self.assertEqual(result["files"], [])
        self.assertNotIn("outside secret", json.dumps(store.events(100)))
        self.assertFalse((store.directory / "base" / "outside.txt").exists())

    def test_failed_prerequisite_skips_dependents_but_keeps_independent_results(self):
        store = self.create([task("deep", deps=("middle",)), task("middle", deps=("fail",)),
                             task("fail"), task("other")])
        status = Runner(store, ScriptedBackend({"other": ["independent work"]})).run()
        self.assertEqual(status["state"], "blocked")
        self.assertEqual(by_id(status, "deep")["state"], "skipped")
        self.assertEqual(by_id(status, "middle")["state"], "skipped")
        self.assertEqual(by_id(status, "other")["state"], "completed")

    def test_invalid_plan_can_be_repaired_and_invalid_explicit_plan_is_atomic(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.create([task("a", deps=("b",)), task("b", deps=("a",))])
        self.assertFalse((self.root / "goal").exists())
        store = self.create()
        status = Runner(store, ScriptedBackend({
            "plan": ["not JSON", json.dumps([task("a")])], "a": ["ok"], "summary": ["done"],
        })).run()
        self.assertEqual(status["state"], "completed")
        self.assertIn("plan_rejected", [entry["kind"] for entry in store.events()])

    def test_bounded_retry_only_for_explicit_transient_errors(self):
        store = self.create([task("try"), task("persistent")], attempts=2)
        calls = {}

        class Backend:
            def complete(self, request, cancelled):
                n = calls[request.task_id] = calls.get(request.task_id, 0) + 1
                if n == 1 or request.task_id == "persistent":
                    raise RetryableInference("busy")
                return Response({"content": "recovered"})

        status = Runner(store, Backend()).run()
        self.assertEqual(status["state"], "blocked")
        self.assertEqual(by_id(status, "try")["state"], "completed")
        self.assertEqual(by_id(status, "persistent")["attempt"], 2)
        self.assertEqual(calls, {"try": 2, "persistent": 2})

    def test_pause_and_resume_keep_completed_tasks(self):
        store = self.create([task("first"), task("wait", deps=("first",))], mode="overnight")
        started = threading.Event()

        class Backend:
            def complete(self, request, cancelled):
                if request.task_id == "first":
                    return Response({"content": "first done"})
                started.set()
                deadline = time.monotonic() + 4
                while not cancelled() and time.monotonic() < deadline:
                    time.sleep(0.01)
                return Response({"content": "interrupted"})

        thread = threading.Thread(target=Runner(store, Backend()).run)
        thread.start()
        try:
            self.assertTrue(started.wait(4))
            with self.assertRaisesRegex(ValueError, "active runner"):
                Runner(store, Backend()).run()
            store.control("pause")
        finally:
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(store.status()["state"], "paused")
        status = Runner(store, ScriptedBackend({"wait": ["resumed"], "summary": ["done"]})).run(resume=True)
        self.assertEqual(status["state"], "completed")
        self.assertEqual(by_id(status, "first")["attempt"], 1)
        self.assertEqual(by_id(status, "wait")["attempt"], 2)

    def test_crashed_writer_is_blocked_and_completed_work_survives(self):
        store = self.create([task("first"), task("edit", "coder", ("first",))],
                            mode="overnight", allow_edits=True)
        code = r"""
import json, os, sys
from pathlib import Path
from veystrix.workflows.backends import Response
from veystrix.workflows.engine import Runner
from veystrix.workflows.state import WorkflowStore

class Backend:
    def complete(self, request, cancelled):
        if request.task_id == "first":
            return Response({"content": "finished"})
        if len(request.messages) > 2:
            os._exit(23)
        return Response({"tool_calls": [{"id": "edit", "function": {
            "name": "write_file", "arguments": json.dumps({"path": "proposal.txt", "content": "partial"})}}]}, "tool_calls")

Runner(WorkflowStore(Path(sys.argv[1])), Backend()).run()
"""
        crash = subprocess.run([sys.executable, "-c", code, str(store.directory)], timeout=10)
        self.assertEqual(crash.returncode, 23)
        previous = store.status()
        self.assertEqual(by_id(previous, "first")["state"], "completed")
        self.assertEqual(by_id(previous, "edit")["state"], "running")
        status = Runner(store, ScriptedBackend({})).run(resume=True)
        edit = by_id(status, "edit")
        self.assertEqual(status["state"], "blocked")
        self.assertEqual(edit["state"], "blocked")
        self.assertEqual(edit["attempt"], 1)
        self.assertTrue(edit["result"]["requires_manual_review"])
        self.assertEqual((store.directory / edit["result"]["workspace"] / "proposal.txt").read_text(), "partial")

    def test_resource_refusal_and_cancel_before_run_do_not_call_backend(self):
        store = self.create(mode="single")
        status = Runner(store, ScriptedBackend({}), admit=lambda: False).run()
        self.assertEqual(status["state"], "blocked")
        self.assertIn("resource admission", by_id(status, "main")["result"]["reason"])
        other = WorkflowStore.create(self.root / "other", self.repo, "Cancel", Settings())
        other.control("cancel")
        status = Runner(other, ScriptedBackend({})).run()
        self.assertEqual(status["state"], "cancelled")
        self.assertEqual(by_id(status, "main")["attempt"], 0)

    def test_active_cancel_and_time_budget_release_workers(self):
        for control in ("cancel", "deadline"):
            with self.subTest(control=control):
                store = WorkflowStore.create(self.root / control, self.repo, "Interrupt", Settings(seconds=1))
                started = threading.Event()

                class Backend:
                    def complete(self, request, cancelled):
                        started.set()
                        limit = time.monotonic() + 4
                        while not cancelled() and time.monotonic() < limit:
                            time.sleep(0.01)
                        return Response({"content": "interrupted"})

                worker = threading.Thread(target=Runner(store, Backend()).run)
                worker.start()
                try:
                    self.assertTrue(started.wait(3))
                    if control == "cancel":
                        store.control("cancel")
                finally:
                    worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(store.status()["state"], "cancelled" if control == "cancel" else "blocked")

    def test_detached_cli_runs_offline_and_persists_its_report(self):
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps({"plan": [json.dumps([task("inspect")])],
                                       "inspect": ["snapshot inspected"], "summary": ["done"]}))
        directory = self.root / "detached"
        command = [sys.executable, "-m", "veystrix.cli", "workflow"]
        completed = subprocess.run(
            [*command, "start", str(directory), "--repo", str(self.repo), "--goal", "Inspect",
             "--mode", "overnight", "--mock", str(fixture), "--detach"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        self.assertIn("runner_pid", json.loads(completed.stdout))
        fixture.unlink()
        store = WorkflowStore(directory)
        deadline = time.monotonic() + 10
        while store.status()["state"] != "completed" and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(store.status()["state"], "completed", (directory / "runner.log").read_text())
        report = subprocess.run([*command, "report", str(directory)], capture_output=True,
                                text=True, check=True, timeout=10).stdout
        self.assertIn("snapshot inspected", report)


class LocalBackendTests(unittest.TestCase):
    def test_loopback_backend_sends_session_and_handles_busy(self):
        requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                requests.append((self.path, dict(self.headers.items()),
                                 json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                if len(requests) == 1:
                    payload = {"choices": [{"message": {"content": "mock"}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 3}}
                    status = 200
                elif len(requests) == 2:
                    payload, status = {"error": {"code": "server_busy"}}, 503
                else:
                    payload, status = {"error": {"code": "model_not_found"}}, 404
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                backend = LocalBackend(server.server_port, "test-key", timeout=2)
                request = Request("task", "agent-1", "installed-model", [{"role": "user", "content": "test"}],
                                  [], 123, "a" * 32)
                response = backend.complete(request, lambda: False)
                self.assertEqual(response.usage["completion_tokens"], 3)
                with self.assertRaises(RetryableInference):
                    backend.complete(request, lambda: False)
                with self.assertRaisesRegex(InferenceBlocked, "model_not_found"):
                    backend.complete(request, lambda: False)
                with self.assertRaises(InferenceBlocked):
                    backend.complete(request, lambda: True)
            finally:
                server.shutdown()
                worker.join()
        self.assertEqual(len(requests), 3)
        path, headers, payload = requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["X-Veystrix-Session"], "agent-1")
        self.assertEqual(headers["X-Veystrix-Workflow"], "a" * 32)
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(payload["model"], "installed-model")
        self.assertEqual(payload["max_completion_tokens"], 123)

    def test_loopback_cancellation_closes_in_flight_connection(self):
        cancel = threading.Event()
        disconnected = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.connection.settimeout(3)
                self.rfile.read(int(self.headers["Content-Length"]))
                cancel.set()
                if self.rfile.read(1) == b"":
                    disconnected.set()

        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                backend = LocalBackend(server.server_port, timeout=3)
                request = Request("task", "agent-1", "installed-model", [], [], 123)
                with self.assertRaises(InferenceBlocked):
                    backend.complete(request, cancel.is_set)
                self.assertTrue(disconnected.wait(1))
            finally:
                server.shutdown()
                worker.join()
