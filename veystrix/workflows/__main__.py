"""Command-line controls for native workflows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading

from .backends import LocalBackend, ScriptedBackend
from .engine import Runner
from .state import Settings, WorkflowStore
from .workspace import path_key


def _demo_fixture():
    return {
        "plan": [json.dumps([
            {"id": "inspect", "role": "explorer", "objective": "List the committed project files."},
            {"id": "review", "role": "reviewer", "objective": "Review the inspection report.", "depends_on": ["inspect"]},
        ])],
        "inspect": [{"message": {"tool_calls": [{"id": "list", "function": {"name": "list_dir", "arguments": "{}"}}]},
                     "finish": "tool_calls", "delay_seconds": 2},
                    "The mock agent listed the committed repository root."],
        "review": [{"message": {"content": "The inspection report is available for review. No edits were proposed."},
                    "delay_seconds": 2}],
        "summary": ["The offline demonstration completed."],
    }


def _execution_args(parser):
    parser.add_argument("--mock", type=Path, metavar="FIXTURE", help="scripted JSON responses without a running engine")
    parser.add_argument("--detach", action="store_true", help="continue after the terminal closes")


def _fixture(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError("mock fixture exceeds 2 MB")
    return json.loads(path.read_text())


def _execute(store, args, *, resume=False):
    settings = Settings(**store.status()["settings"])
    saved_fixture = store.directory / "mock.json"
    fixture = args.mock.resolve(strict=True) if args.mock else saved_fixture if saved_fixture.is_file() else None
    if (settings.model == "mock") != bool(fixture):
        raise ValueError("mock workflows require --mock FIXTURE; live workflows require an explicit model ID")
    backend = (ScriptedBackend(_fixture(fixture)) if fixture else
               LocalBackend(settings.port, os.environ.get("VEYSTRIX_SERVER_API_KEY") or os.environ.get("VEYSTRIX_API_KEY"),
                            timeout=1800 if settings.executor_model else 120))
    if args.detach:
        command = [sys.executable, "-m", "veystrix.workflows", "resume" if resume else "run",
                   str(store.directory)]
        if fixture:
            command.extend(["--mock", str(fixture)])
        with (store.directory / "runner.log").open("a", encoding="utf-8") as log:
            worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"workflow": str(store.directory), "runner_pid": worker.pid}))
        return 0

    pause = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: pause.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = Runner(store, backend, pause_requested=pause.is_set).run(resume=resume)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(_report(result))
    return 0 if result["state"] in {"completed", "paused"} else 1


def _report(status):
    lines = [f"# Workflow report ({status['state']})", "", status["objective"], "",
             f"Commit `{status['commit_id']}`", f"Model `{status['settings']['model']}`", ""]
    if status["settings"].get("executor_model"):
        lines += [f"Executor model `{status['settings']['executor_model']}`", ""]
    proposed = {}
    for task in status["tasks"]:
        result = task["result"] or {}
        lines += [f"## {task['id']} · {task['state']}", "",
                  f"Role **{task['role']}** · Attempts **{task['attempt']}**", ""]
        if result.get("summary"):
            lines += ["Agent report", "", result["summary"], ""]
        if result.get("summary_truncated"):
            lines += ["The report preview was truncated. See the assistant message in the workflow journal.", ""]
        if result.get("reason"):
            lines += [result["reason"], ""]
        if result.get("unavailable_dependencies"):
            lines += ["Unavailable dependencies " + ", ".join(result["unavailable_dependencies"]), ""]
        if result.get("workspace"):
            lines += [f"Proposal workspace `{result['workspace']}`", ""]
        if result.get("requires_manual_review"):
            lines += ["Manual review required before integration.", ""]
        if result.get("syntax_checked"):
            lines += ["Syntax checked " + ", ".join(result["syntax_checked"]), ""]
        if result.get("tool_errors"):
            lines += [f"Tool calls that returned an error **{result['tool_errors']}**", ""]
        if result.get("usage"):
            usage = ", ".join(f"{key.replace('_', ' ')} {value}" for key, value in result["usage"].items())
            lines += [f"Backend token usage {usage}", ""]
        for file in result.get("files", []):
            name, tasks = proposed.setdefault(path_key(file), (file, set()))
            tasks.add(task["id"])
    overlaps = {name: sorted(tasks) for name, tasks in proposed.values() if len(tasks) > 1}
    if overlaps:
        lines += ["## Overlapping proposals", ""]
        lines.extend(f"- `{name}` from {', '.join(tasks)}" for name, tasks in sorted(overlaps.items()))
    return "\n".join(lines).rstrip()


def _history(root):
    entries = []
    if not root.is_dir():
        return entries
    directories = [path for path in root.iterdir() if path.is_dir() and not path.is_symlink()]
    for path in sorted(directories, key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
        store = WorkflowStore(path)
        try:
            with store.connect() as db:
                goal = db.execute("SELECT objective, state FROM goal").fetchone()
                if goal:
                    entries.append({"id": path.name, **dict(goal)})
        except (OSError, ValueError, sqlite3.Error):
            continue
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(prog="veystrix workflow", description="Run local, durable agent workflows")
    actions = parser.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start", help="snapshot the repository and run a new workflow")
    start.add_argument("state", type=Path, help="new workflow directory with an existing parent")
    start.add_argument("--repo", type=Path, required=True)
    start.add_argument("--goal", required=True)
    start.add_argument("--mode", choices=("single", "swarm", "overnight"), default="single")
    start.add_argument("--model", default="mock")
    start.add_argument("--executor-model", help="optional separate model for coder and tester tasks")
    start.add_argument("--demo", action="store_true", help="run a built-in read-only mock swarm")
    start.add_argument("--port", type=int, default=8080, help="existing Veystrix loopback engine port")
    start.add_argument("--plan", type=Path, help="optional explicit JSON task array")
    start.add_argument("--agents", type=int, default=3)
    start.add_argument("--tools", type=int, default=2)
    start.add_argument("--turns", type=int, default=8)
    start.add_argument("--output-tokens", type=int, default=2048)
    start.add_argument("--seconds", type=int, default=3600)
    start.add_argument("--attempts", type=int, default=2)
    start.add_argument("--allow-edits", action="store_true", help="permit coder proposals in isolated copies")
    _execution_args(start)
    history = actions.add_parser("list", help="list workflow journals in a directory")
    history.add_argument("state", type=Path)
    for command in ("run", "resume", "status", "pause", "cancel", "events", "report"):
        action = actions.add_parser(command)
        action.add_argument("state", type=Path)
        if command in {"run", "resume"}:
            _execution_args(action)
        if command == "events":
            action.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            settings = Settings(mode=args.mode, model=args.model, agents=args.agents, tools=args.tools,
                                turns=args.turns, output_tokens=args.output_tokens, seconds=args.seconds,
                                attempts=args.attempts, allow_edits=args.allow_edits, port=args.port,
                                executor_model=args.executor_model)
            if args.demo and (args.mock or args.plan or args.allow_edits or args.mode == "single" or args.executor_model):
                raise ValueError("the built-in demo requires a read-only swarm or overnight goal without a custom plan")
            if (settings.model == "mock") != bool(args.mock or args.demo):
                raise ValueError("provide --demo or --mock FIXTURE, or select a live backend with --model MODEL")
            if args.plan and args.plan.stat().st_size > 128_000:
                raise ValueError("plan exceeds 128 KB")
            plan = json.loads(args.plan.read_text()) if args.plan else None
            if args.plan and plan is None:
                raise ValueError("plan must be a JSON task array")
            fixture = _demo_fixture() if args.demo else _fixture(args.mock) if args.mock else None
            if args.mock or args.demo:
                ScriptedBackend(fixture)  # validate before creating the workflow
            store = WorkflowStore.create(args.state, args.repo, args.goal, settings, plan)
            if args.mock or args.demo:
                (store.directory / "mock.json").write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")))
                args.mock = None
            return _execute(store, args)
        if args.action == "list":
            print(json.dumps(_history(args.state)))
            return 0
        store = WorkflowStore(args.state)
        if args.action in {"run", "resume"}:
            return _execute(store, args, resume=args.action == "resume")
        if args.action in {"pause", "cancel"}:
            store.control(args.action)
            # Reconcile a journal left in "running" by a crashed runner. The
            # persisted control closes admission before this runner can claim work.
            if store.status()["state"] == "running" and not store.runner_active():
                try:
                    Runner(store, ScriptedBackend({})).run()
                except ValueError:
                    if not store.runner_active():
                        raise
        if args.action == "events":
            print(json.dumps(store.events(args.limit), indent=2))
        elif args.action == "report":
            print(_report(store.status()))
        else:
            print(json.dumps({**store.status(), "runner_active": store.runner_active()}, indent=2))
        return 0
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f"workflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
