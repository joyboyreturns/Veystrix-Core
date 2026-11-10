#!/usr/bin/env python3
"""Scan a Git repository (working tree, or all history with --history) for credential-shaped strings.

Heuristic and pattern based: a clean result is evidence, not proof. Exit status 1 when anything matches.
Usage: secret_scan.py [--repo DIR] [--history]
"""
from __future__ import annotations
import argparse, re, subprocess, sys
from pathlib import Path

PATTERN = re.compile(rb'(ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}'
                     rb'|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,}'
                     rb'|hf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.)')
SKIP_SUFFIX = ("uv.lock",)
# Deliberately fake credentials that prove the diagnostics redactor works (each is an obvious alphabet run).
FIXTURES = ("apps/macos/Veystrix/Tests/VeystrixKitTests/KitTests.swift",)
MAX_BYTES = 3_000_000


def git(repo: str, *args: str, **kw):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True, **kw).stdout


def scan_blob(name: str, data: bytes, label: str) -> int:
    hits = 0
    if name in FIXTURES:
        return 0
    for m in PATTERN.finditer(data):
        hits += 1
        print(f"HIT {label}{name}: {m.group(0)[:10].decode(errors='replace')}…")
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--history", action="store_true", help="scan every blob reachable from any ref, not just the working tree")
    args = ap.parse_args(argv)
    hits = scanned = 0
    if args.history:
        listing = git(args.repo, "rev-list", "--objects", "--all").decode().splitlines()
        names = {}
        for line in listing:
            sha, _, name = line.partition(" ")
            names[sha] = name
        info = subprocess.run(["git", "-C", args.repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                              input="\n".join(names).encode(), capture_output=True, check=True).stdout.decode().splitlines()
        for sha, kind, size in (l.split() for l in info):
            if kind != "blob" or int(size) > MAX_BYTES or names[sha].endswith(SKIP_SUFFIX):
                continue
            scanned += 1
            hits += scan_blob(names[sha], git(args.repo, "cat-file", "blob", sha), f"{sha[:10]} ")
    else:
        for rel in git(args.repo, "ls-files", "-z").decode().split("\0"):
            path = Path(args.repo) / rel
            if not rel or rel.endswith(SKIP_SUFFIX) or not path.is_file() or path.stat().st_size > MAX_BYTES:
                continue
            scanned += 1
            hits += scan_blob(rel, path.read_bytes(), "")
    print(f"blobs/files scanned: {scanned}; hits: {hits}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
