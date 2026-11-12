"""Inventory and retire legacy Veystrix prefix snapshot pairs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from .universal_cache import _lock_file, _read_json, _regular

QWEN = Path("~/.flashnext/prefix-cache").expanduser()
NEMOTRON = Path("~/.cache/veystrix/nemotron35/prefix").expanduser()


def _pair(path, model):
    key = path.stem
    data = path.with_suffix(".safetensors")
    meta_info, data_info = _regular(path), _regular(data)
    if not meta_info or not data_info:
        return None
    if model == "flashnext":
        meta = _read_json(path, {})
        if (not isinstance(meta, dict) or meta.get("key") != key or meta.get("format") not in ("1", "2", "3")
                or meta.get("file_bytes") != data_info.st_size or not meta.get("fingerprint")):
            return None
    else:
        if not 4 < meta_info.st_size <= 4 + 262144 * 4:
            return None
        tape = path.read_bytes()
        if tape[:4] != b"VXT1" or (len(tape) - 4) % 4 or hashlib.sha256(tape[4:]).hexdigest()[:24] != key:
            return None
        with data.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size <= 1024 * 1024:
                return None
            try:
                meta = json.loads(stream.read(size)).get("__metadata__", {})
                if not all(re.fullmatch(r"[0-9a-f]{64}", meta[k]) for k in ("fingerprint", "sha256")):
                    return None
                if not set(json.loads(meta["types"])) <= {"mamba", "attention"}:
                    return None
            except (ValueError, TypeError, KeyError, AttributeError):
                return None
    return (path, data), meta_info.st_size + data_info.st_size


def migrate(root, model, *, apply=False):
    root = root.expanduser().absolute()
    result = {"directory": str(root), "model": model, "verified_snapshots": 0, "verified_bytes": 0,
              "deleted_snapshots": 0, "deleted_bytes": 0, "skipped": []}
    if not root.exists() and not root.is_symlink():
        return result
    if root.resolve() != root or not root.is_dir() or root.stat().st_uid != os.getuid():
        result["skipped"].append("Directory is not a direct user-owned directory.")
        return result

    with ExitStack() as stack:
        if model == "flashnext" and apply:
            fd = _lock_file(root / ".veystrix-owner.lock")
            stack.callback(os.close, fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["skipped"].append("Flash-Next currently holds the ownership lock.")
                return result
        directories = [root]
        if model == "nemotron35":
            directories = [p for p in root.iterdir() if re.fullmatch(r"[0-9a-f]{16}", p.name)
                           and p.is_dir() and not p.is_symlink() and p.stat().st_uid == os.getuid()]
            result["skipped"].extend(p.name for p in root.iterdir() if p not in directories)
        for directory in directories:
            names = set(directory.iterdir())
            pattern = r"[0-9a-f]{64}\.json" if model == "flashnext" else r"[0-9a-f]{24}\.tokens"
            for path in sorted(names):
                if not re.fullmatch(pattern, path.name):
                    continue
                pair = _pair(path, model)
                if not pair:
                    continue
                files, size = pair
                result["verified_snapshots"] += 1
                result["verified_bytes"] += size
                names.difference_update(files)
                if apply:
                    for file in files:
                        file.unlink()
                    result["deleted_snapshots"] += 1
                    result["deleted_bytes"] += size
            result["skipped"].extend(str(p.relative_to(root)) for p in sorted(names))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", action="append", type=Path, default=[])
    parser.add_argument("--nemotron", action="append", type=Path, default=[])
    parser.add_argument("--apply", action="store_true", help="Delete verified pairs. Stop legacy Nemotron writers first.")
    args = parser.parse_args()
    settings = _read_json(Path("~/Library/Application Support/Veystrix/settings.json").expanduser(), {})
    custom = settings.get("cacheDirectory") if isinstance(settings, dict) else None
    qwen = [QWEN, *args.qwen]
    nemotron = [NEMOTRON, *args.nemotron]
    if custom:
        qwen.append(Path(custom))
        nemotron.append(Path(custom))
    for paths, name in ((qwen, "FLASHNEXT_PREFIX_CACHE_DIR"), (nemotron, "VEYSTRIX_NEMOTRON_PREFIX_DIR")):
        if os.environ.get(name):
            paths.append(Path(os.environ[name]))
    print(json.dumps([migrate(root, model, apply=args.apply)
                      for model, roots in (("flashnext", qwen), ("nemotron35", nemotron))
                      for root in dict.fromkeys(path.expanduser().absolute() for path in roots)], indent=2))


if __name__ == "__main__":
    main()
