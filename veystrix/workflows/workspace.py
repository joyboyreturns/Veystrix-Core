"""Committed repository snapshots and restricted tools for unattended workflows."""
from __future__ import annotations

import ast
import difflib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tomllib
import unicodedata

from veystrix.tools import TOOLS
from veystrix.tools.repo import Repo

READ_TOOLS = frozenset({"list_dir", "read_file", "find_files", "search"})
EDIT_TOOLS = frozenset({"write_file", "replace_text"})
MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
MAX_TOOL_CHARS = 12_000
MAX_DIFF_CHARS = 4_000


def path_key(path: str) -> str:
    return unicodedata.normalize("NFD", path).casefold()


def snapshot(repository: Path, directory: Path) -> str:
    env = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1",
           "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repository), *args], env=env, text=True,
                                capture_output=True, timeout=30, check=False)
        if result.returncode:
            raise ValueError(result.stderr.strip()[:500] or "Git snapshot failed")
        return result.stdout.strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != repository:
        raise ValueError("select the root of a Git checkout")
    commit = git("rev-parse", "--verify", "HEAD^{commit}")
    archive = directory / "snapshot.tar"
    git("archive", "--format=tar", f"--output={archive}", commit)
    if archive.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError("committed snapshot exceeds 128 MiB")
    base = directory / "base"
    base.mkdir()
    seen = set()
    with tarfile.open(archive) as files:
        for member in files:
            path = PurePosixPath(member.name)
            if path.is_absolute() or any(part.casefold() in {".git", ".."} for part in path.parts):
                raise ValueError("unsafe path in committed snapshot")
            if member.isfile() or member.isdir():
                folded = path_key(str(path))
                if folded in seen:
                    raise ValueError("committed snapshot contains paths that collide on macOS")
                seen.add(folded)
                files.extract(member, base, filter="data")
    archive.unlink()
    return commit


def tool_schemas(writable: bool) -> list[dict]:
    allowed = READ_TOOLS | EDIT_TOOLS if writable else READ_TOOLS
    return [tool for tool in TOOLS if tool["function"]["name"] in allowed]


class WorkflowRepo(Repo):
    def __init__(self, base: Path, root: Path, writable: bool):
        if writable:
            shutil.copytree(base, root)
        super().__init__(root if writable else base)
        self.base = base
        self.writable = writable
        self.changes: set[str] = set()
        self.verified: set[str] = set()

    def safe(self, rel="."):
        relative = Path(rel or ".")
        if relative.is_absolute() or any(part == ".." or part.casefold() == ".git" for part in relative.parts):
            raise ValueError("path must be relative to the workflow workspace")
        return super().safe(relative)

    def _api_check(self, path, content):
        # Only in-process syntax checks. Repo's normal checker may spawn external
        # processes, which this unattended permission profile does not grant.
        check = {".py": ast.parse, ".json": json.loads, ".toml": tomllib.loads}.get(path.suffix)
        if check is not None:
            check(content)
        return []

    def call(self, name, args):
        if name not in READ_TOOLS and (not self.writable or name not in EDIT_TOOLS):
            raise ValueError("tool is outside this agent's permissions")
        result = super().call(name, args.copy())
        if name in EDIT_TOOLS:
            self.changes.add(result["path"])
            if Path(result["path"]).suffix in {".py", ".json", ".toml"}:
                self.verified.add(result["path"])
        return result

    def artifacts(self) -> dict:
        diff = []
        for name in sorted(self.changes):
            original = self.base / name
            before = original.read_text().splitlines(keepends=True) if original.is_file() else []
            after = (self.root / name).read_text().splitlines(keepends=True)
            diff.extend(difflib.unified_diff(before, after, fromfile=f"a/{name}", tofile=f"b/{name}"))
            if sum(map(len, diff)) > MAX_DIFF_CHARS:
                break
        preview = "".join(diff)
        return {"files": sorted(self.changes), "syntax_checked": sorted(self.verified),
                "diff_preview": preview[:MAX_DIFF_CHARS], "diff_truncated": len(preview) > MAX_DIFF_CHARS}
