"""Persistent Cache inspection and maintenance tools for Veystrix / FlashNext.

Provides safe offline inspection, statistics, and LRU/TTL cleanup of persistent
RAM and SSD snapshots without reading or exposing sensitive prompt or reasoning data.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from models.flashnext import prefix_cache as P

DEFAULT_CACHE_DIR = os.path.expanduser(
    os.environ.get("FLASHNEXT_PREFIX_CACHE_DIR") or "~/.flashnext/prefix-cache"
)


def inspect_cache(directory: str | Path | None = None) -> dict:
    """Inspect stored snapshot manifests and disk files without exposing prompt texts or internal reasoning."""
    target_dir = Path(directory or DEFAULT_CACHE_DIR).expanduser()
    if not target_dir.exists():
        return {
            "status": "not_found",
            "directory": str(target_dir),
            "total_entries": 0,
            "total_bytes": 0,
            "total_tokens_cached": 0,
            "entries": [],
        }

    total_bytes = 0
    total_tokens = 0
    entries = []
    now = time.time()

    for manifest_path in sorted(target_dir.glob("*.json")):
        key = manifest_path.stem
        if manifest_path.name.startswith("."):
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            data_file = target_dir / f"{key}.safetensors"
            file_size = data_file.stat().st_size if data_file.exists() else manifest.get("file_bytes", 0)
            json_size = manifest_path.stat().st_size
            entry_bytes = file_size + json_size
            total_bytes += entry_bytes
            n_tokens = manifest.get("n_tokens", 0)
            total_tokens = max(total_tokens, n_tokens)

            entries.append({
                "key": key,
                "n_tokens": n_tokens,
                "parent": manifest.get("parent"),
                "parent_tokens": manifest.get("parent_tokens", 0),
                "format": manifest.get("format"),
                "fingerprint": manifest.get("fingerprint", "")[:16] + "...",
                "bytes": entry_bytes,
                "created": manifest.get("created", 0.0),
                "age_seconds": round(now - manifest.get("created", now), 1),
                "layers": len(manifest.get("layers", [])),
            })
        except Exception:
            continue

    try:
        usage = shutil.disk_usage(target_dir)
        free_bytes = usage.free
        total_disk_bytes = usage.total
    except OSError:
        free_bytes = None
        total_disk_bytes = None

    return {
        "status": "ok",
        "directory": str(target_dir),
        "total_entries": len(entries),
        "total_bytes": total_bytes,
        "total_bytes_mb": round(total_bytes / (1024 * 1024), 2),
        "total_bytes_gb": round(total_bytes / (1024 * 1024 * 1024), 3),
        "disk_free_bytes": free_bytes,
        "disk_free_gb": round(free_bytes / (1024 * 1024 * 1024), 2) if free_bytes is not None else None,
        "entries": entries,
    }


def clean_cache(
    directory: str | Path | None = None,
    *,
    max_age_days: float | None = None,
    target_gb: float | None = None,
    dry_run: bool = False,
) -> dict:
    """Safely drop unleased leaf snapshots without corrupting chained dependencies."""
    target_dir = Path(directory or DEFAULT_CACHE_DIR).expanduser()
    if not target_dir.exists():
        return {"status": "not_found", "directory": str(target_dir), "removed_entries": 0, "freed_bytes": 0}

    # Discover manifests
    manifests: dict[str, dict] = {}
    for manifest_path in sorted(target_dir.glob("*.json")):
        if manifest_path.name.startswith("."):
            continue
        try:
            m = json.loads(manifest_path.read_text())
            manifests[manifest_path.stem] = m
        except Exception:
            continue

    parents = {m.get("parent") for m in manifests.values() if m.get("parent")}
    now = time.time()
    max_age_seconds = max_age_days * 86400 if max_age_days is not None else None

    # Only leaf entries (nothing chains to them) may be safely evicted
    leaves = [k for k in manifests if k not in parents]
    leaves.sort(key=lambda k: manifests[k].get("created", 0.0))  # oldest first

    removed = 0
    freed_bytes = 0

    target_bytes = int(target_gb * 1e9) if target_gb is not None else None
    current_total = sum(m.get("file_bytes", 0) for m in manifests.values())

    for key in leaves:
        m = manifests[key]
        created = m.get("created", now)
        is_expired = max_age_seconds is not None and (now - created > max_age_seconds)
        is_over_budget = target_bytes is not None and (current_total - freed_bytes > target_bytes)

        if is_expired or is_over_budget:
            f_bytes = m.get("file_bytes", 0)
            if not dry_run:
                (target_dir / f"{key}.safetensors").unlink(missing_ok=True)
                (target_dir / f"{key}.json").unlink(missing_ok=True)
            removed += 1
            freed_bytes += f_bytes

    return {
        "status": "ok",
        "dry_run": dry_run,
        "directory": str(target_dir),
        "removed_entries": removed,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Veystrix Persistent SSD Cache Tools")
    parser.add_argument("--directory", "-d", default=None, help="Cache directory (default: ~/.flashnext/prefix-cache)")
    parser.add_argument("--inspect", action="store_true", help="Inspect stored snapshots and exit")
    parser.add_argument("--clean", action="store_true", help="Perform safe cache cleanup")
    parser.add_argument("--max-age-days", type=float, default=None, help="Maximum snapshot age in days to retain")
    parser.add_argument("--target-gb", type=float, default=None, help="Target disk budget in GB")
    parser.add_argument("--dry-run", action="store_true", help="Simulate cleanup without deleting files")
    args = parser.parse_args()

    if args.clean:
        res = clean_cache(args.directory, max_age_days=args.max_age_days, target_gb=args.target_gb, dry_run=args.dry_run)
        print(json.dumps(res, indent=2))
    else:
        res = inspect_cache(args.directory)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
