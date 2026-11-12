"""SQLite journal for a single goal. Only the lock owner may execute tasks."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import fcntl
from graphlib import CycleError, TopologicalSorter
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
from uuid import uuid4

from .workspace import snapshot

ROLES = frozenset({"coordinator", "explorer", "architect", "coder", "tester",
                   "reviewer", "researcher", "integrator"})
MAX_TASKS = 16
FINAL_STATES = frozenset({"completed", "blocked", "failed", "cancelled"})
TASK_TERMINAL = FINAL_STATES | {"skipped"}


@dataclass(frozen=True)
class Settings:
    mode: str = "single"
    model: str = "mock"
    agents: int = 3
    tools: int = 2
    turns: int = 8
    output_tokens: int = 2048
    seconds: int = 3600
    attempts: int = 2
    allow_edits: bool = False
    port: int = 8080
    executor_model: str | None = None
    workflow_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        if self.mode not in {"single", "swarm", "overnight"}:
            raise ValueError("unknown workflow mode")
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 256:
            raise ValueError("provide a model ID of at most 256 characters")
        if self.executor_model is not None:
            if not isinstance(self.executor_model, str) or not self.executor_model.strip() or len(self.executor_model) > 256:
                raise ValueError("provide an executor model ID of at most 256 characters")
            if self.executor_model == self.model or self.mode == "single":
                raise ValueError("an executor model must differ from the primary model and requires a swarm or overnight goal")
        if not isinstance(self.workflow_id, str) or not re.fullmatch(r"[0-9a-f]{32}", self.workflow_id):
            raise ValueError("invalid workflow identity")
        bounds = {"agents": 4, "tools": 4, "turns": 24, "output_tokens": 8192,
                  "seconds": 86400, "attempts": 3, "port": 65535}
        for key, maximum in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{key} must be between 1 and {maximum}")
        if type(self.allow_edits) is not bool:
            raise ValueError("allow_edits must be a boolean")


@dataclass(frozen=True)
class Task:
    id: str
    role: str
    objective: str
    depends_on: tuple[str, ...] = ()


def validate_plan(items: list[dict]) -> list[Task]:
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_TASKS:
        raise ValueError(f"a plan needs 1 to {MAX_TASKS} tasks")
    tasks = []
    for item in items:
        if not isinstance(item, dict) or set(item) - {"id", "role", "objective", "depends_on"}:
            raise ValueError("unexpected task fields")
        name, role, objective = (item.get(key) for key in ("id", "role", "objective"))
        deps = item.get("depends_on", [])
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", name) or name in {"plan", "summary"}:
            raise ValueError("task IDs must be unique lowercase identifiers; plan and summary are reserved")
        if not isinstance(role, str) or role not in ROLES:
            raise ValueError(f"unknown role for {name}")
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 4000:
            raise ValueError(f"{name} needs an objective of at most 4000 characters")
        if (not isinstance(deps, list) or not all(isinstance(dep, str) for dep in deps)
                or len(set(deps)) != len(deps)):
            raise ValueError(f"invalid dependencies for {name}")
        tasks.append(Task(name, role, objective, tuple(deps)))
    ids = {task.id for task in tasks}
    if len(ids) != len(tasks) or any(set(task.depends_on) - ids for task in tasks):
        raise ValueError("duplicate task ID or unknown dependency")
    try:
        tuple(TopologicalSorter({task.id: task.depends_on for task in tasks}).static_order())
    except CycleError as exc:
        raise ValueError("task dependencies contain a cycle") from exc
    return tasks


def plan_tasks(items: list[dict], objective: str) -> list[Task]:
    tasks = validate_plan(items)
    review = Task("summary", "coordinator",
                  "Summarize the completed task reports for this goal. Identify remaining work and conflicting proposals. "
                  "Only describe checks recorded in the task reports. Goal: " + objective,
                  tuple(task.id for task in tasks))
    return [*tasks, review]


class WorkflowStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser().resolve()
        self.db = self.directory / "workflow.sqlite3"

    @classmethod
    def create(cls, directory: Path, repository: Path, objective: str, settings: Settings,
               items: list[dict] | None = None) -> WorkflowStore:
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 4000:
            raise ValueError("provide an objective of at most 4000 characters")
        if settings.mode == "single" and items is not None:
            raise ValueError("explicit plans require swarm or overnight mode")
        tasks = (plan_tasks(items, objective) if items is not None else
                 [Task("main", "coder", objective)] if settings.mode == "single" else
                 [Task("plan", "coordinator", objective)])
        store = cls(directory)
        store.directory.mkdir(mode=0o700, parents=False)
        try:
            repository = Path(repository).expanduser().resolve(strict=True)
            commit = snapshot(repository, store.directory)
            with store.connect(create=True) as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript("""
                    CREATE TABLE goal (
                        objective TEXT NOT NULL, repository TEXT NOT NULL, commit_id TEXT NOT NULL,
                        settings TEXT NOT NULL, state TEXT NOT NULL, control TEXT NOT NULL,
                        deadline REAL
                    );
                    CREATE TABLE task (
                        id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0, agent_id TEXT, result TEXT,
                        ready_after REAL NOT NULL DEFAULT 0
                    );
                    CREATE TABLE event (
                        sequence INTEGER PRIMARY KEY, time REAL NOT NULL, task_id TEXT,
                        attempt INTEGER, kind TEXT NOT NULL, detail TEXT NOT NULL
                    );
                """)
                db.execute("INSERT INTO goal VALUES (?, ?, ?, ?, 'ready', 'run', NULL)",
                           (objective, str(repository), commit, json.dumps(asdict(settings))))
                store.insert_tasks(db, tasks)
                store.event(db, "created", {"commit": commit})
            store.db.chmod(0o600)
        except BaseException:
            shutil.rmtree(store.directory)
            raise
        return store

    @contextmanager
    def connect(self, create=False):
        if not create and not self.db.is_file():
            raise ValueError("workflow database does not exist")
        mode = "rwc" if create else "rw"
        db = sqlite3.connect(f"{self.db.as_uri()}?mode={mode}", uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def runner_lock(self):
        if not self.db.is_file():
            raise ValueError("workflow database does not exist")
        with (self.directory / "runner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("workflow already has an active runner") from exc
            yield

    def runner_active(self) -> bool:
        try:
            lock = (self.directory / "runner.lock").open("rb")
        except FileNotFoundError:
            return False
        with lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False

    @staticmethod
    def insert_tasks(db, tasks: list[Task]):
        db.executemany("INSERT INTO task(id, spec, state) VALUES (?, ?, 'pending')",
                       [(task.id, json.dumps(asdict(task))) for task in tasks])

    @staticmethod
    def event(db, kind: str, detail: dict, task_id: str | None = None, attempt: int | None = None):
        db.execute("INSERT INTO event(time, task_id, attempt, kind, detail) VALUES (?, ?, ?, ?, ?)",
                   (time.time(), task_id, attempt, kind, json.dumps(detail)))

    def status(self) -> dict:
        with self.connect() as db:
            db.execute("BEGIN")
            goal = dict(db.execute("SELECT * FROM goal").fetchone())
            goal["settings"] = json.loads(goal["settings"])
            tasks = []
            for row in db.execute("SELECT * FROM task ORDER BY rowid"):
                task = dict(row)
                task.update(json.loads(task.pop("spec")))
                task["result"] = json.loads(task["result"]) if task["result"] else None
                tasks.append(task)
        return {**goal, "tasks": tasks}

    def events(self, limit=50) -> list[dict]:
        if not 1 <= limit <= 1000:
            raise ValueError("event limit must be between 1 and 1000")
        with self.connect() as db:
            return [{**dict(row), "detail": json.loads(row["detail"])}
                    for row in reversed(db.execute("SELECT * FROM event ORDER BY sequence DESC LIMIT ?", (limit,)).fetchall())]

    def control(self, action: str):
        if action not in {"pause", "cancel"}:
            raise ValueError("expected pause or cancel")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            goal = db.execute("SELECT state, control FROM goal").fetchone()
            if goal["state"] in FINAL_STATES or goal["control"] in {action, "cancel"}:
                return
            db.execute("UPDATE goal SET control = ?", (action,))
            if goal["state"] in {"ready", "paused"}:
                state = "paused" if action == "pause" else "cancelled"
                db.execute("UPDATE goal SET state = ?", (state,))
                if action == "cancel":
                    db.execute("UPDATE task SET state = 'cancelled' WHERE state = 'pending'")
            self.event(db, "control_requested", {"action": action})
