"""Model-free registry and disk policy for persistent inference state."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time

GB = 10**9
DEFAULT_QUOTA_GB = 80
DEFAULT_ROOT = "~/.cache/veystrix/prefix-cache-v1"
RESERVE_BYTES = 50 * GB
NAMESPACE_MARKER = ".namespace-v1"
FORMATS = {"flashnext": (64, ".json"), "nemotron35": (24, ".tokens")}
FINGERPRINT = re.compile(r"[0-9a-f]{64}")


def _app_cache_settings():
    settings = _read_json(Path("~/Library/Application Support/Veystrix/settings.json").expanduser(), {})
    return settings if isinstance(settings, dict) else {}


def quota_gb(environ=None):
    env = os.environ if environ is None else environ
    fallback = DEFAULT_QUOTA_GB
    if environ is None and "VEYSTRIX_CACHE_GB" not in env:
        settings = _app_cache_settings()
        saved = settings.get("cacheQuotaGB")
        if saved is not None and (settings.get("schemaVersion") == 2 or saved != 50):
            fallback = saved
    try:
        value = float(env.get("VEYSTRIX_CACHE_GB", fallback))
    except (TypeError, ValueError):
        return DEFAULT_QUOTA_GB
    return value if math.isfinite(value) and 0 < value <= 2000 else DEFAULT_QUOTA_GB


def cache_root(directory=None):
    override = (directory or os.environ.get("VEYSTRIX_CACHE_ROOT")
                or os.environ.get("FLASHNEXT_PREFIX_CACHE_DIR")
                or os.environ.get("VEYSTRIX_NEMOTRON_PREFIX_DIR")
                or _app_cache_settings().get("cacheDirectory"))
    default = Path(DEFAULT_ROOT).expanduser()
    path = Path(override).expanduser() if override else default
    return path if path == default else path / "veystrix-cache-v1"


def _directory(path):
    if path.resolve() != path.absolute():
        raise ValueError(f"Symlink or traversal in cache path: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError(f"Cache directory must be private and owned by this user: {path}")
    return path


def _regular(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return info if stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid() else None


def _read_json(path, default):
    if not _regular(path):
        return default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return default


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path, value):
    fd, temp = tempfile.mkstemp(prefix=".registry-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _sync_directory(path.parent)
    finally:
        Path(temp).unlink(missing_ok=True)


def _lock_file(path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
        os.close(fd)
        raise ValueError(f"Invalid cache lock: {path}")
    return fd


@dataclass(frozen=True)
class Snapshot:
    model: str
    directory: Path
    key: str
    files: tuple[Path, ...]
    size: int
    tokens: int
    mtime: float
    parent: str | None = None
    plan: tuple | None = None

    @property
    def identity(self):
        return f"{self.model}/{self.directory.name}/{self.key}"


class CacheManager:
    def __init__(self, directory=None, *, quota_bytes=None, reserve_bytes=RESERVE_BYTES):
        self.root = cache_root(directory).absolute()
        # macOS publishes these system-owned aliases, including TMPDIR under /var.
        if self.root.parts[:2] in (("/", "var"), ("/", "tmp")) and Path("/private").is_dir():
            self.root = Path("/private") / self.root.relative_to("/")
        self.default_quota = int(quota_gb() * GB) if quota_bytes is None else int(quota_bytes)
        self.reserve = reserve_bytes

    @contextmanager
    def _locked(self):
        root = _directory(self.root)
        fd = _lock_file(root / ".manager.lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    @contextmanager
    def transaction(self):
        with self._locked():
            tx = Transaction(self)
            if "quota_bytes" not in tx.metadata:
                tx.metadata["quota_bytes"] = self.default_quota
                tx._save()
            yield tx

    def namespace(self, model, fingerprint):
        if model not in FORMATS or not FINGERPRINT.fullmatch(fingerprint):
            raise ValueError("Invalid cache namespace")
        with self._locked():
            directory = _directory(_directory(self.root / model) / fingerprint)
            marker = directory / NAMESPACE_MARKER
            identity = {"model": model, "fingerprint": fingerprint}
            if _read_json(marker, None) != identity:
                if any(directory.iterdir()):
                    raise ValueError(f"Cache namespace has unrecognized contents: {directory}")
                _write_json(marker, identity)
            return directory


class Transaction:
    def __init__(self, manager):
        self.manager = manager
        self.root = manager.root
        self.registry = self.root / "registry.json"
        self.metadata = _read_json(self.registry, {})
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        for temp in self.root.glob(".registry-*"):
            if _regular(temp):
                temp.unlink()
        self.entries = self._scan()

    @property
    def quota(self):
        stored = self.metadata.get("quota_bytes")
        return stored if isinstance(stored, int) and 0 < stored <= 2000 * GB else self.manager.default_quota

    @property
    def used(self):
        return sum(entry.size for entry in self.entries.values())

    def _scan(self):
        """Read committed pairs under the global lock. Clean up interrupted writes."""
        entries = {}
        for model, (digits, suffix) in FORMATS.items():
            family = self.root / model
            if not family.is_dir() or family.is_symlink():
                continue
            for directory in family.iterdir():
                if not FINGERPRINT.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
                    continue
                if _read_json(directory / NAMESPACE_MARKER, None) != {"model": model, "fingerprint": directory.name}:
                    continue
                for path in directory.iterdir():
                    if not _regular(path):
                        continue
                    key = path.stem
                    if not re.fullmatch(rf"[0-9a-f]{{{digits}}}", key):
                        temp = (r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.(?:tmp\.safetensors|json\.tmp)"
                                if model == "flashnext" else r"(?:\.nemotron-[\w-]+\.safetensors|[0-9a-f]{24}\.tokens\.tmp)")
                        if re.fullmatch(temp, path.name):
                            path.unlink()
                        continue
                    if path.suffix not in (suffix, ".safetensors"):
                        continue
                    sidecar = directory / f"{key}{suffix}"
                    data = directory / f"{key}.safetensors"
                    side_info, data_info = _regular(sidecar), _regular(data)
                    if not side_info or not data_info:
                        path.unlink(missing_ok=True)
                        continue
                    if path != sidecar:
                        continue
                    parent, plan = None, None
                    if model == "flashnext":
                        manifest = _read_json(path, {})
                        try:
                            tokens, parent = manifest["n_tokens"], manifest["parent"]
                            edges = manifest.get("plan_edges")
                            plan = tuple(edges) if edges is not None else None
                            if (manifest.get("key") != key or manifest.get("fingerprint") != directory.name
                                    or data_info.st_size != manifest["file_bytes"]
                                    or not isinstance(tokens, int) or tokens <= 0
                                    or parent is not None and not FINGERPRINT.fullmatch(parent)
                                    or plan is not None and any(not isinstance(edge, int) for edge in plan)):
                                raise ValueError("invalid manifest")
                        except (KeyError, TypeError, ValueError, AttributeError):
                            tokens, parent, plan = 0, None, None
                    else:
                        with path.open("rb") as stream:
                            magic = stream.read(4)
                        tokens = (side_info.st_size - 4) // 4
                        if magic != b"VXT1" or (side_info.st_size - 4) % 4 or not 0 < tokens <= 1_048_576:
                            tokens = 0
                    entry = Snapshot(model, directory, key, (sidecar, data),
                                     side_info.st_size + data_info.st_size, tokens, side_info.st_mtime, parent, plan)
                    entries[entry.identity] = entry
        # A valid delta must point to a shorter committed snapshot in its own namespace.
        for identity, entry in sorted(entries.items(), key=lambda item: item[1].tokens):
            parent_id = self._parent(entry)
            parent = entries.get(parent_id)
            if parent_id is not None and (parent is None or not 0 < parent.tokens < entry.tokens):
                entries[identity] = Snapshot(entry.model, entry.directory, entry.key, entry.files, entry.size, 0, entry.mtime)
        return entries

    def _parent(self, entry):
        return f"{entry.model}/{entry.directory.name}/{entry.parent}" if entry.parent else None

    def _lease_path(self, entry):
        return entry.directory / f".{entry.key}.lease"

    def _leased(self, entry):
        path = self._lease_path(entry)
        if not path.exists() and not path.is_symlink():
            return False
        fd = _lock_file(path)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            path.unlink()
            return False
        finally:
            os.close(fd)

    def lease(self, directory, key):
        fd = _lock_file(directory / f".{key}.lease")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def set_pin(self, directory, key, pinned):
        identity = f"flashnext/{directory.name}/{key}"
        pins = self.metadata.setdefault("pins", [])
        if identity not in self.entries:
            return False
        if pinned:
            if identity in pins:
                return True
            keep = self._ancestors([*pins, identity])
            if sum(self.entries[i].size for i in keep) > self.quota // 2:
                return False
            pins.append(identity)
        elif identity in pins:
            pins.remove(identity)
        self._save()
        return True

    def _ancestors(self, identities):
        keep = set()
        for identity in identities:
            while identity in self.entries and identity not in keep:
                keep.add(identity)
                identity = self._parent(self.entries[identity])
        return keep

    def _rank(self, entry, now):
        record = self.metadata.get("reuse", {}).get(entry.identity, {})
        age_hours = max(0, now - record.get("last", entry.mtime)) / 3600
        hits = record.get("hits", 0)
        cost = entry.tokens ** 1.25
        return cost * (1 + math.log1p(hits)) / (max(entry.size, 1) * (1 + age_hours / 72))

    def reclaim(self, amount, *, apply=False, protect=(), clear=False, older_than=None):
        requested = max(0, amount)
        pending = self.entries.copy()
        now = time.time()
        protected = set(protect) | set(self.metadata.get("pins", ()))
        protected.update(i for i, entry in pending.items() if self._leased(entry))
        children = {}
        for entry in pending.values():
            parent = self._parent(entry)
            if parent is not None:
                children[parent] = children.get(parent, 0) + 1

        def eligible(identity):
            if identity in protected or children.get(identity, 0):
                return False
            if older_than is None:
                return True
            used = self.metadata.get("reuse", {}).get(identity, {}).get("last", pending[identity].mtime)
            return used < older_than

        queue = [(self._rank(entry, now), i) for i, entry in pending.items() if eligible(i)]
        heapq.heapify(queue)
        freed = count = 0
        while freed < requested and queue:
            _, identity = heapq.heappop(queue)
            victim = pending.pop(identity)
            if apply:
                # Commit marker first. An interruption leaves an orphan for the next scan.
                for path in victim.files:
                    path.unlink()
                self.entries.pop(victim.identity)
                self.metadata.get("reuse", {}).pop(victim.identity, None)
            parent = self._parent(victim)
            if parent in pending:
                children[parent] -= 1
                if eligible(parent):
                    heapq.heappush(queue, (self._rank(pending[parent], now), parent))
            freed += victim.size
            count += 1
        result = {"requested_bytes": requested, "reclaimed_bytes": freed, "snapshots": count,
                  "shortfall_bytes": 0 if clear and not pending else max(0, requested - freed)}
        if apply and requested:
            self.metadata["last_cleanup"] = {**result, "timestamp": now}
            self._save()
        return result

    def make_room(self, size, *, protect=()):
        if size > self.quota:
            return False
        free = shutil.disk_usage(self.root).free
        needed = max(0, self.used + size - self.quota, self.manager.reserve + size - free)
        if needed:
            # First check that the protected entries leave enough eligible bytes.
            if self.reclaim(needed, protect=protect)["shortfall_bytes"]:
                return False
            self.reclaim(needed, apply=True, protect=protect)
        return self.used + size <= self.quota and shutil.disk_usage(self.root).free >= self.manager.reserve + size

    def record(self, model, directory, key=None, tokens=0):
        stats = self.metadata.setdefault("statistics", {}).setdefault(model, {"hits": 0, "misses": 0, "restored_tokens": 0})
        stats["hits" if tokens else "misses"] += 1
        stats["restored_tokens"] += tokens
        identity = f"{model}/{directory.name}/{key}"
        if identity in self.entries:
            record = self.metadata.setdefault("reuse", {}).setdefault(identity, {"hits": 0})
            record["hits"] += 1
            record["last"] = time.time()
        try:
            self._save()
        except OSError:
            # Reuse telemetry must not turn a successful restore into a failure.
            pass

    def _save(self):
        _write_json(self.registry, self.metadata)

    def status(self):
        models = {}
        for model in FORMATS:
            entries = [entry for entry in self.entries.values() if entry.model == model]
            models[model] = {"bytes": sum(entry.size for entry in entries), "snapshots": len(entries),
                             **self.metadata.get("statistics", {}).get(model, {})}
        free = shutil.disk_usage(self.root).free
        return {"root": str(self.root), "used_bytes": self.used, "quota_bytes": self.quota,
                "free_bytes": free, "reserve_bytes": self.manager.reserve,
                "snapshots": len(self.entries), "models": models, "last_cleanup": self.metadata.get("last_cleanup"),
                "healthy": (self.used <= self.quota and free >= self.manager.reserve
                            and all(entry.tokens > 0 for entry in self.entries.values()))}

    def set_quota(self, quota):
        deficit = self.used - quota
        preview = self.reclaim(deficit)
        if preview["shortfall_bytes"]:
            return {**preview, "reclaimed_bytes": 0, "snapshots": 0}
        result = self.reclaim(deficit, apply=True)
        self.metadata["quota_bytes"] = quota
        self._save()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory")
    parser.add_argument("action", choices=("status", "preview", "reclaim", "clear", "quota"))
    parser.add_argument("gb", nargs="?", type=float)
    args = parser.parse_args()
    if args.action in ("preview", "reclaim", "quota") and not (args.gb and 0 < args.gb <= 2000):
        parser.error("amount must be greater than 0 and at most 2000 GB")
    with CacheManager(args.directory).transaction() as tx:
        if args.action == "status":
            result = tx.status()
        elif args.action == "quota":
            result = tx.set_quota(int(args.gb * GB))
        else:
            clear = args.action == "clear"
            result = tx.reclaim(tx.used if clear else int(args.gb * GB),
                                apply=args.action != "preview", clear=clear)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
