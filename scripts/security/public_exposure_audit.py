#!/usr/bin/env python3
"""Prove that a public-repo checkout exposes nothing from the private engine repo.

Checks (each must pass; exit 1 otherwise):
  1. no commit in the public repo's history exists in the private repo (independent history)
  2. no blob in the public repo's history has the same hash as any private-repo blob, except an allowlist of
     files that are meant to be public (legal texts, notices)
  3. no engine source files (*.py outside tools/, *.metal, *.swift) and no checkpoint/weight/cache formats
  4. no developer-machine paths or private-repo names in any text file
  5. no credential-shaped strings (scripts/security/secret_scan.py patterns), in the tree and in history
Usage: public_exposure_audit.py --public DIR --core DIR
"""
from __future__ import annotations
import argparse, re, subprocess, sys
from pathlib import Path

ALLOWED_SHARED = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "LICENSES/MIT-1architect.txt", "LICENSES/Sparkle-2.10.1.txt"}
FORBIDDEN_SUFFIX = (".py", ".metal", ".swift", ".safetensors", ".gguf", ".npz", ".pt", ".bin", ".pkl", ".p12", ".p8", ".pem")
FORBIDDEN_TEXT = (re.compile(rb"/Users/[A-Za-z0-9._-]+"), re.compile(rb"Veystrix-Core"), re.compile(rb"vkv2576"))
HERE = Path(__file__).resolve().parent


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout.decode(errors="replace")


def blobs(repo) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in git(repo, "rev-list", "--objects", "--all").splitlines():
        sha, _, name = line.partition(" ")
        if name:
            out.setdefault(sha, set()).add(name)
    kinds = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                           input="\n".join(out).encode(), capture_output=True, check=True).stdout.decode().splitlines()
    return {s: out[s] for s, k in (l.split() for l in kinds) if k == "blob"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", required=True)
    ap.add_argument("--core", required=True)
    args = ap.parse_args(argv)
    failures: list[str] = []

    pub_commits = set(git(args.public, "rev-list", "--all").split())
    core_commits = set(git(args.core, "rev-list", "--all").split())
    if pub_commits & core_commits:
        failures.append(f"shared commits: {sorted(pub_commits & core_commits)[:3]}")
    print(f"1. commits: public {len(pub_commits)}, private {len(core_commits)}, shared {len(pub_commits & core_commits)}")

    pub_blobs, core_blobs = blobs(args.public), blobs(args.core)
    shared = {s: pub_blobs[s] for s in pub_blobs.keys() & core_blobs.keys()}
    bad = {s: n for s, n in shared.items() if not n <= ALLOWED_SHARED}
    if bad:
        failures.append(f"blobs identical to private-repo content: {sorted(n for v in bad.values() for n in v)[:10]}")
    print(f"2. identical blobs: {len(shared)} (allowlisted legal/notice files only: {not bad})")

    names = {n for v in pub_blobs.values() for n in v}
    src = sorted(n for n in names if n.endswith(FORBIDDEN_SUFFIX) and not n.startswith("tools/"))
    if src:
        failures.append(f"engine-source or weight-like files in public history: {src[:10]}")
    print(f"3. forbidden file types in history: {len(src)}")

    leaks = []
    for sha, ns in pub_blobs.items():
        data = subprocess.run(["git", "-C", args.public, "cat-file", "blob", sha], capture_output=True, check=True).stdout
        if b"\0" in data[:2000]:
            continue
        for pat in FORBIDDEN_TEXT:
            if pat.search(data):
                leaks.append((sorted(ns)[0], pat.pattern.decode()))
    if leaks:
        failures.append(f"private paths/names in public text: {leaks[:10]}")
    print(f"4. private path/name leaks: {len(leaks)}")

    for extra in ([], ["--history"]):
        r = subprocess.run([sys.executable, str(HERE / "secret_scan.py"), "--repo", args.public, *extra], capture_output=True, text=True)
        print(f"5. secret scan {'history' if extra else 'tree'}: {r.stdout.strip().splitlines()[-1]}")
        if r.returncode:
            failures.append(f"secret scan ({'history' if extra else 'tree'}) hit:\n{r.stdout}")
    if failures:
        print("\nEXPOSURE AUDIT FAILED")
        for f in failures:
            print(" -", f)
        return 1
    print("\nEXPOSURE AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
