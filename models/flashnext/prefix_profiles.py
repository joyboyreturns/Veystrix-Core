"""Persistent prefix profiles and a bounded, self-maintaining SSD cache (opt-in: ``FLASHNEXT_PREFIX_PROFILES=1``).

This sits on top of ``prefix_cache.PrefixCache`` and changes nothing about what a snapshot is or how it is restored:
state is stored exactly, matched exactly (SHA-256 chain over the engine fingerprint and every token id), and used only
when its chunk-plan history fits. What it adds is bookkeeping and housekeeping around those exact snapshots.

Profile. Every stored snapshot has a record (the *profile*) next to its immutable manifest:

* identity: chain key (token-prefix digest under the engine fingerprint), token count, the fingerprint itself (in the
  manifest), chunk-plan provenance (``plan_edges`` in the manifest and inside the checksummed data file);
* state location and size: the ``<key>.safetensors`` file (KV deltas, GatedDeltaNet recurrent and convolution state,
  position state) and its bytes;
* use: hit count, last-used time, estimated recomputation cost (tokens over the learned prefill rate), time saved;
* ``kind`` ("system" for the end of a system message), a pin ("manual" or "auto"), and the execution profile that
  produced it (``config``: pool, ring, chunk and memory settings).

The records live in one sidecar, ``index.profiles`` (atomic, checksummed, with a ``.bak``). The sidecar is only
metadata: losing it loses frequencies and pins, never state, and the manager rebuilds from the manifests.

Recognising a repeated system prompt needs no matching heuristics. The system boundary is an anchor of every chunk
plan, so every request that shares the same system prompt reaches the same exact key; ``observe`` counts those
sightings and a snapshot of the system message that is seen or hit ``pin_after`` times is pinned automatically.

Storage policy.

* Limits: a byte budget for the directory (default 10 GB) and a free-space reserve for the volume (default 50 GB).
  Before a write, room is made by evicting; if the limits cannot be met without touching protected entries, or the
  reserve cannot be met by evicting everything evictable, the write is skipped (the snapshot stays in RAM) and
  ``writes_skipped`` counts it. Nothing is evicted in a futile attempt to recover free space this cache does not hold.
* Eviction order: snapshots of another fingerprint or format first (they cannot hit under this configuration), then the
  lowest *value* = (1 + hits) x recompute seconds x 0.5^(age / half-life) / bytes, oldest first on a tie.
* Never evicted: pinned entries, leased entries (a hit's chain while it is restored, the snapshots a running prefill
  extends), and every ancestor of those. Only leaves are dropped, so a chain is never left without its parent.
* Atomic writes, checksums, crash recovery: ``write_entry`` writes data then manifest through temporary names with
  fsync and rename. At startup, temporary files of an interrupted write and data files that never got a manifest are
  removed (after a grace period, in case another process is writing), manifests without data are dropped, chains with a
  missing ancestor are dropped, and the budget is enforced again (so a lowered limit takes effect across restarts).
* Scope of deletion: only files in the cache directory whose names are exactly ``<64 hex>.json``/``.safetensors``, the
  matching temporary names, and ``index.profiles`` (+ ``.bak``). No symlink is followed or removed. A directory that
  holds anything else and no marker file is refused.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from . import prefix_cache as P

FLAG = "FLASHNEXT_PREFIX_PROFILES"
FLAG_MAX_GB = "FLASHNEXT_PREFIX_PROFILES_MAX_GB"            # directory budget, default 10
FLAG_RESERVE_GB = "FLASHNEXT_PREFIX_PROFILES_FREE_RESERVE_GB"   # volume free-space reserve, default 50
FLAG_PIN_AFTER = "FLASHNEXT_PREFIX_PROFILES_PIN_AFTER"      # auto-pin a system profile after this many uses, 0 = never
FLAG_HALF_LIFE_H = "FLASHNEXT_PREFIX_PROFILES_HALF_LIFE_H"  # recency half-life in hours, default 72
FLAG_FIRST_USE = "FLASHNEXT_PREFIX_FIRST_USE"               # snapshot & pin high-value system prompts on first use
INDEX = "index.profiles"
INDEX_LOCK = ".index.lock"
MARKER = ".veystrix-prefix-cache"
INDEX_VERSION = 1
DEFAULT_PREFILL_TOK_S = 60.0          # measured 2K cold prefill on the target machine: about 60 tokens/s
DISK_SLACK = 64 * 1024                # file header, token tape and manifest on top of the tensor bytes
SEEN_LIMIT = 256

_KEY = r"[0-9a-f]{64}"
FILE_RE = re.compile(rf"^({_KEY})\.(json|safetensors)$")
TMP_RE = re.compile(rf"^\.({_KEY})\.[0-9a-f]{{32}}\.(tmp\.safetensors|json\.tmp)$")
INDEX_TMP_RE = re.compile(r"^\.index\.[0-9a-f]{32}\.tmp$")
# Settings that do not change numerics; a stored profile whose value differs still restores exactly, but is reported.
EXECUTION_ONLY = ("expert_pool_gb", "memory_manager", "decode_pool_gb", "release_pool", "short_uncached")


class UnsafeDirectory(ValueError):
    """The cache directory holds files this cache did not write; it will not delete in it."""


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


def _number(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return float(default)
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def settings_from_env() -> dict:
    return {"disk_budget": int(_number(FLAG_MAX_GB, 10) * 1e9),
            "free_reserve": int(_number(FLAG_RESERVE_GB, 50) * 1e9),
            "pin_after": int(_number(FLAG_PIN_AFTER, 3)),
            "half_life_s": _number(FLAG_HALF_LIFE_H, 72) * 3600.0}


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ProfiledPrefixCache(P.PrefixCache):
    """``PrefixCache`` with profiles, frequency-aware eviction, pins, a size budget and a free-space reserve."""

    def __init__(self, fp: str, ram_budget: int, directory, disk_budget: int = int(10e9), *,
                 free_reserve: int = int(50e9), free_bytes=None, pin_after: int = 3,
                 half_life_s: float = 72 * 3600.0, clock=time.time, config: dict | None = None,
                 index_flush_s: float = 5.0, orphan_grace_s: float = 60.0, verify_hash: bool = True):
        if directory is None:
            raise ValueError("profiles need a cache directory")
        self.free_reserve = int(free_reserve)
        self._free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
        self.pin_after = int(pin_after)
        self.half_life_s = max(1.0, float(half_life_s))
        self._clock = clock
        self.config = dict(config or {})
        self.index_flush_s = float(index_flush_s)
        self.orphan_grace_s = float(orphan_grace_s)
        self.high_value_min_tokens = int(os.environ.get("FLASHNEXT_HIGH_VALUE_MIN_TOKENS", "512"))
        self._high_value_keys: set[str] = set()
        self.eviction_events: list[dict] = []
        self._meta: dict[str, dict] = {}              # key -> profile record (valid entries)
        self._foreign: dict[str, dict] = {}           # key -> {"bytes", "atime", "parent"} (other fingerprint/format)
        self._seen: dict[str, dict] = {}              # key -> {"count", "last", "n"} (exact system-prefix sightings)
        self._rate = {"tokens": 0.0, "seconds": 0.0}  # learned prefill speed
        self._dirty = False
        self._flushed_at = 0.0
        self.index_state = "absent"
        self.extra = {"config_drift": 0, "auto_pins": 0, "foreign_evicted": 0, "scrubbed": 0}
        super().__init__(fp, ram_budget, directory, disk_budget, verify_hash=verify_hash, write_through=True)

    # ---- safety of deletion ----
    def _check_directory(self) -> None:
        marker = self.directory / MARKER
        if marker.exists():
            return
        strangers = []
        for item in os.scandir(self.directory):
            name = item.name
            if name == MARKER or FILE_RE.match(name) or TMP_RE.match(name) or INDEX_TMP_RE.match(name) \
                    or name in (INDEX, INDEX + ".bak", INDEX_LOCK, P.OWNER_LOCK):
                continue
            strangers.append(name)
        if strangers:
            raise UnsafeDirectory(f"{self.directory} holds files that are not prefix-cache files "
                                  f"({sorted(strangers)[:3]}); refusing to manage it")
        marker.write_text("Veystrix prefix cache: only files named <64 hex>.json/.safetensors and index.profiles "
                          "are created or deleted here.\n")
        os.chmod(marker, 0o600)

    def _owned(self, name: str) -> bool:
        return bool(FILE_RE.match(name) or TMP_RE.match(name) or INDEX_TMP_RE.match(name)
                    or name in (INDEX, INDEX + ".bak", INDEX_LOCK))

    def _index_lock(self):
        """Cross-process file lock on .index.lock."""
        class _LockCtx:
            def __init__(self, lock_path):
                self.path = lock_path
                self.fd = None

            def __enter__(self):
                try:
                    self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(self.fd, fcntl.LOCK_EX)
                except OSError:
                    if self.fd is not None:
                        try:
                            os.close(self.fd)
                        except OSError:
                            pass
                        self.fd = None
                    raise
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.fd is not None:
                    try:
                        fcntl.flock(self.fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    try:
                        os.close(self.fd)
                    except OSError:
                        pass
                    self.fd = None
        return _LockCtx(self.directory / INDEX_LOCK)

    def _unlink(self, name: str) -> bool:
        """Remove one of our own files in this directory; never a symlink, never any other name."""
        if not self._owned(name):
            return False
        path = self.directory / name
        try:
            if path.is_symlink() or not path.is_file():
                return False
            path.unlink()
            return True
        except OSError:
            return False

    def _drop_files(self, key) -> None:
        for suffix in (".json", ".safetensors"):            # manifest first: a crash leaves an orphan data file
            self._unlink(f"{key}{suffix}")

    # ---- startup: scan, recover, enforce ----
    def _scan(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self._check_directory()
        now = self._clock()
        names = {item.name: item for item in os.scandir(self.directory)}
        # 1. leftovers of an interrupted write (temporary files, data without a manifest)
        for name, item in names.items():
            if not (TMP_RE.match(name) or INDEX_TMP_RE.match(name)):
                continue
            if now - item.stat(follow_symlinks=False).st_mtime >= self.orphan_grace_s and self._unlink(name):
                self.stats.orphans_removed += 1
        manifests = {m.group(1) for n in names if (m := FILE_RE.match(n)) and m.group(2) == "json"}
        for name, item in names.items():
            m = FILE_RE.match(name)
            if m and m.group(2) == "safetensors" and m.group(1) not in manifests:
                if now - item.stat(follow_symlinks=False).st_mtime >= self.orphan_grace_s and self._unlink(name):
                    self.stats.orphans_removed += 1
        # 2. manifests: valid, foreign or broken
        created = {}
        for key in sorted(manifests):
            path = self.directory / f"{key}.json"
            try:
                manifest = P.read_manifest(self.directory, key)
                data_size = (self.directory / f"{key}.safetensors").stat().st_size
                if data_size != manifest["file_bytes"]:
                    raise P.CorruptEntry("size")
                atime = path.stat().st_mtime
                if manifest.get("fingerprint") != self.fp or manifest.get("format") != P.FORMAT_VERSION:
                    self._foreign[key] = {"bytes": manifest["file_bytes"], "atime": atime,
                                          "parent": manifest.get("parent")}
                    continue
                self._disk[key] = {"n": manifest["n_tokens"], "parent": manifest["parent"],
                                   "bytes": manifest["file_bytes"], "atime": atime}
                edges = manifest.get("plan_edges")
                self._plans[key] = None if edges is None else tuple(int(e) for e in edges)
                created[key] = manifest.get("created", atime)
            except (P.CorruptEntry, OSError, KeyError, TypeError):
                self._drop_files(key)
                self.stats.invalidated += 1
        # 3. a chain is usable only with every ancestor
        for key in list(self._disk):
            if key in self._disk and not self._ancestors_present(key):
                self._drop_files(key)
                self._disk.pop(key, None)
                self._plans.pop(key, None)
                self.stats.invalidated += 1
        # 4. profile records (sidecar); anything without a record gets defaults from the manifest
        saved = self._load_index()
        self._seen = {k: v for k, v in saved.get("seen", {}).items() if isinstance(v, dict)}
        self._rate = {**self._rate, **{k: float(v) for k, v in saved.get("rate", {}).items() if k in self._rate}}
        records = saved.get("entries", {})
        for key, meta in self._disk.items():
            record = records.get(key, {}) if isinstance(records.get(key, {}), dict) else {}
            self._meta[key] = {
                "kind": record.get("kind"), "hits": int(record.get("hits", 0)),
                "last_used": float(record.get("last_used", meta["atime"])),
                "created": float(record.get("created", created.get(key, meta["atime"]))),
                "pinned": record.get("pinned") if record.get("pinned") in ("manual", "auto") else None,
                "saved_s": float(record.get("saved_s", 0.0)), "config": record.get("config") or {},
                "n": meta["n"]}
            meta["atime"] = self._meta[key]["last_used"]
        self._lengths = {meta["n"] for meta in self._disk.values()}
        self._enforce_budget()
        self._flush_index(force=True)

    # ---- sidecar ----
    def _load_index(self) -> dict:
        for label, name in (("ok", INDEX), ("backup", INDEX + ".bak")):
            path = self.directory / name
            try:
                wrapper = json.loads(path.read_text())
                payload = wrapper["payload"]
                if wrapper["checksum"] != hashlib.sha256(_canonical(payload)).hexdigest():
                    continue
                if payload.get("version") != INDEX_VERSION:
                    continue
                self.index_state = label
                return payload
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        self.index_state = "rebuilt" if self._disk else "absent"
        return {}

    def _flush_index(self, force: bool = False) -> bool:
        now = self._clock()
        if not force and (not self._dirty or now - self._flushed_at < self.index_flush_s):
            return False
        payload = {"version": INDEX_VERSION, "fingerprint": self.fp, "written": now,
                   "entries": {k: dict(v) for k, v in self._meta.items() if k in self._disk},
                   "seen": self._seen, "rate": self._rate}
        wrapper = {"checksum": hashlib.sha256(_canonical(payload)).hexdigest(), "payload": payload}
        tmp = self.directory / f".index.{uuid.uuid4().hex}.tmp"
        with self._index_lock():
            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as handle:
                    handle.write(json.dumps(wrapper, sort_keys=True))
                    handle.flush()
                    os.fsync(handle.fileno())
                current = self.directory / INDEX
                if current.exists():
                    os.replace(current, self.directory / (INDEX + ".bak"))
                os.replace(tmp, current)
                P._fsync_dir(self.directory)
            except OSError:
                self._unlink(tmp.name)
                return False
        self._dirty, self._flushed_at = False, now
        return True

    def flush(self) -> bool:
        with self._lock:
            return self._flush_index(force=True)

    # ---- value and protection ----
    def prefill_rate(self) -> float:
        """Learned prefill tokens per second (an estimate for recomputation cost), or the default."""
        if self._rate["seconds"] > 0 and self._rate["tokens"] > 0:
            return self._rate["tokens"] / self._rate["seconds"]
        return DEFAULT_PREFILL_TOK_S

    def recompute_seconds(self, n_tokens: int) -> float:
        return n_tokens / self.prefill_rate()

    def record_prefill(self, tokens: int, seconds: float) -> None:
        """Feed one measured prefill (tokens computed, seconds taken) into the recompute-cost estimate."""
        if tokens < 256 or seconds <= 0:                  # short runs are mostly fixed overhead
            return
        with self._lock:
            decay = 0.8 if self._rate["seconds"] > 0 else 0.0
            self._rate["tokens"] = self._rate["tokens"] * decay + tokens
            self._rate["seconds"] = self._rate["seconds"] * decay + seconds
            self._dirty = True

    def _value(self, key) -> float:
        meta = self._meta.get(key)
        if meta is None:
            return 0.0
        if meta["pinned"]:
            return float("inf")
        age = max(0.0, self._clock() - meta["last_used"])
        size = self._disk[key]["bytes"] if key in self._disk else (self._ram[key].nbytes() if key in self._ram else 1)
        # Goal 15.2: Use measured recomputation and restore times (saved_s / hits) when available,
        # with conservative fallback to theoretical recompute seconds.
        if meta["hits"] > 0 and meta.get("saved_s", 0.0) > 0.0:
            unit_time = meta["saved_s"] / meta["hits"]
        else:
            unit_time = self.recompute_seconds(meta["n"])
        return (1 + meta["hits"]) * unit_time * 0.5 ** (age / self.half_life_s) / max(1, size)

    def _protected(self) -> set:
        """Pinned and leased entries, and every ancestor of those: nothing here may be evicted."""
        roots = {k for k, m in self._meta.items() if m["pinned"]} | set(self._leases)
        protected = set()
        for key in roots:
            protected.update(self._chain_keys(key))
        return protected

    def _disk_total(self) -> int:
        return sum(m["bytes"] for m in self._disk.values()) + sum(m["bytes"] for m in self._foreign.values())

    def _foreign_victim(self):
        parents = {m["parent"] for m in self._foreign.values()}
        leaves = [k for k in self._foreign if k not in parents]
        return min(leaves, key=lambda k: self._foreign[k]["atime"]) if leaves else None

    def _disk_victim(self):
        protected = self._protected()
        parents = {m["parent"] for m in self._disk.values()}
        leaves = [k for k in self._disk if k not in parents and k not in protected]
        if not leaves:
            return None
        return min(leaves, key=lambda k: (self._value(k), self._disk[k]["atime"]))

    def _evict_one(self) -> int | None:
        """Drop a victim: snapshots of another fingerprint or format first (oldest leaf first), then local entries by value."""
        victim = self._foreign_victim()
        if victim is not None:
            victim_bytes = self._foreign[victim]["bytes"] if victim in self._foreign else 0
            self._drop_files(victim)
            self._foreign.pop(victim, None)
            self.extra["foreign_evicted"] += 1
            self.stats.evicted_ssd += 1
            self.stats.evicted_ssd_bytes += victim_bytes
            self._dirty = True
            self.eviction_events.append({
                "key": victim, "reason": "foreign", "bytes": victim_bytes,
                "remaining_disk": self._disk_total(), "timestamp": self._clock(),
            })
            return victim_bytes
        victim = self._disk_victim()
        if victim is None:
            return None
        victim_bytes = self._disk[victim]["bytes"] if victim in self._disk else 0
        freed = self._drop_disk_entry(victim)
        self._meta.pop(victim, None)
        self._dirty = True
        self.eviction_events.append({
            "key": victim, "reason": "least_valuable", "bytes": freed or victim_bytes,
            "remaining_disk": self._disk_total(), "timestamp": self._clock(),
        })
        return freed

    def _evictable_bytes(self) -> int:
        protected = self._protected()
        return sum(m["bytes"] for k, m in self._disk.items() if k not in protected)

    def _refresh_lengths(self) -> None:
        self._lengths = {m["n"] for m in self._disk.values()} | {e.n_tokens for e in self._ram.values()}

    def _enforce_budget(self) -> None:
        if self.disk_budget:
            while self._disk_total() > self.disk_budget and self._evict_one() is not None:
                pass
        self._refresh_lengths()

    def _evict_disk(self) -> None:                     # called by the base after every write
        self._enforce_budget()
        self._flush_index(force=True)

    def _make_room(self, need: int, protect=()) -> bool:
        """Make space for ``need`` bytes inside the budget and the reserve; False (nothing evicted for the reserve)
        when that cannot be done without breaking protection."""
        self._acquire(protect)
        try:
            if self.disk_budget and need > self.disk_budget:
                return False
            if self.disk_budget:
                while self._disk_total() + need > self.disk_budget:
                    if self._evict_one() is None:
                        return False
            if self.free_reserve:
                deficit = self.free_reserve - (self._free_bytes(self.directory) - need)
                if deficit > 0:
                    if deficit > self._evictable_bytes():
                        return False                    # evicting everything evictable would still not be enough
                    freed = 0
                    while freed < deficit:
                        step = self._evict_one()
                        if step is None:
                            return False
                        freed += step
            return True
        finally:
            self._release(protect)
            self._refresh_lengths()

    def _persist(self, entry: P.Entry) -> None:
        if entry.key in self._disk:
            return
        if not self._make_room(entry.nbytes() + DISK_SLACK, protect=self._chain_keys(entry.parent) if entry.parent else ()):
            self.stats.writes_skipped += 1
            return
        super()._persist(entry)

    # ---- recording ----
    def register_high_value(self, tokens_or_key) -> str:
        """Register a high-value system prompt key to snapshot/pin on first use."""
        with self._lock:
            if isinstance(tokens_or_key, (list, tuple)):
                key = P.chain_key(self.fp, tuple(int(t) for t in tokens_or_key))
            else:
                key = str(tokens_or_key)
            self._high_value_keys.add(key)
            if key in self._meta:
                self._meta[key]["high_value"] = True
                self._maybe_auto_pin(key)
            return key

    def is_high_value(self, tokens, kind=None) -> bool:
        if kind != "system" and kind is not None:
            return False
        if len(tokens) >= self.high_value_min_tokens:
            return True
        key = P.chain_key(self.fp, tuple(int(t) for t in tokens))
        return key in self._high_value_keys

    def get_eviction_events(self) -> list[dict]:
        with self._lock:
            return list(self.eviction_events)

    def put(self, tokens, cache_list, parent=None, language=None, plan=None, kind=None, high_value=None):
        with self._lock:
            entry = super().put(tokens, cache_list, parent, language, plan, kind)
            meta = self._meta.get(entry.key)
            is_hv = high_value if high_value is not None else self.is_high_value(tokens, kind)
            if meta is None:
                now = self._clock()
                meta = self._meta[entry.key] = {
                    "kind": kind, "hits": 0, "last_used": now, "created": now, "pinned": None, "saved_s": 0.0,
                    "config": dict(self.config), "n": entry.n_tokens, "high_value": is_hv}
                self._dirty = True
            else:
                if kind and not meta["kind"]:
                    meta["kind"], self._dirty = kind, True
                if is_hv and not meta.get("high_value"):
                    meta["high_value"], self._dirty = True, True
            self._maybe_auto_pin(entry.key)
            self._flush_index(force=entry.key in self._disk and self._dirty)   # a new snapshot's record is saved at once
            return entry

    def observe(self, tokens) -> int:
        """Count one request that starts with exactly these tokens (the system message). Returns the sightings before this
        one, so 0 means a new prompt and 1 or more a recognised one."""
        with self._lock:
            tokens = tuple(int(t) for t in tokens)
            key = P.chain_key(self.fp, tokens)
            row = self._seen.get(key)
            before = row["count"] if row else 0
            self._seen[key] = {"count": before + 1, "last": self._clock(), "n": len(tokens)}
            if len(self._seen) > SEEN_LIMIT:
                for victim in sorted(self._seen, key=lambda k: (self._seen[k]["count"], self._seen[k]["last"]))[
                        : len(self._seen) - SEEN_LIMIT]:
                    del self._seen[victim]
            self._dirty = True
            if key in self._meta:
                self._maybe_auto_pin(key)
            return before

    def _maybe_auto_pin(self, key) -> None:
        meta = self._meta.get(key)
        if not self.pin_after or meta is None or meta["pinned"] or meta["kind"] != "system":
            return
        is_hv = bool(meta.get("high_value") or key in self._high_value_keys)
        threshold = 1 if is_hv else self.pin_after
        sightings = self._seen.get(key, {}).get("count", 0)
        hits = meta.get("hits", 0)
        if is_hv:
            effective_count = max(hits + 1, sightings, 1)
        else:
            effective_count = max(hits, sightings - 1)
        if effective_count < threshold:
            return
        pinned_bytes = sum(self._disk[k]["bytes"] for k in self._protected_by_pins() if k in self._disk)
        new_bytes = self._disk.get(key, {"bytes": 0})["bytes"]
        if self.disk_budget and pinned_bytes + new_bytes > self.disk_budget // 2:
            # Stale auto-pins (older than half_life_s) yield their pin status so active high-value prompts can be protected
            now = self._clock()
            stale_candidates = [
                k for k, m in self._meta.items()
                if m.get("pinned") == "auto" and k != key and (now - m.get("last_used", now)) >= self.half_life_s
            ]
            stale_candidates.sort(key=lambda k: self._meta[k].get("last_used", 0.0))
            for stale_k in stale_candidates:
                self._meta[stale_k]["pinned"] = None
                self._dirty = True
                pinned_bytes = sum(self._disk[k]["bytes"] for k in self._protected_by_pins() if k in self._disk)
                if pinned_bytes + new_bytes <= self.disk_budget // 2:
                    break
            else:
                return                                                # pins may not take more than half the budget
        meta["pinned"] = "auto"
        self.extra["auto_pins"] += 1
        self._dirty = True

    def _protected_by_pins(self) -> set:
        keys = set()
        for key, meta in self._meta.items():
            if meta["pinned"]:
                keys.update(self._chain_keys(key))
        return keys

    def _on_hit(self, key, n_tokens, tier, seconds) -> None:
        now = self._clock()
        meta = self._meta.get(key)
        if meta is None:
            return
        meta["hits"] += 1
        meta["last_used"] = now
        saved = max(0.0, self.recompute_seconds(n_tokens) - seconds)
        meta["saved_s"] += saved
        self.stats.time_saved_seconds += saved
        for ancestor in self._chain_keys(key):                        # the whole chain was used
            if ancestor in self._meta:
                self._meta[ancestor]["last_used"] = now
            if ancestor in self._disk:
                self._disk[ancestor]["atime"] = now
        drift = {k: (meta["config"].get(k), v) for k, v in self.config.items()
                 if k in EXECUTION_ONLY and k in meta["config"] and meta["config"][k] != v}
        if drift:
            self.extra["config_drift"] += 1
        self._maybe_auto_pin(key)
        self._dirty = True
        self._flush_index()

    def _touch(self, key) -> None:
        super()._touch(key)
        meta = self._meta.get(key)
        if meta is not None:
            meta["last_used"] = self._clock()
            self._dirty = True

    def _ram_order(self) -> list:
        order = list(self._ram)                                       # least recently used first
        return sorted(order, key=self._value)                         # stable: ties stay in LRU order

    def _invalidate(self, key) -> None:
        before = set(self._disk)
        super()._invalidate(key)
        for gone in before - set(self._disk):
            self._meta.pop(gone, None)
        self._dirty = True

    # ---- operator API ----
    def _key_of(self, ref) -> str | None:
        if isinstance(ref, str):
            return ref if ref in self._disk else None
        key = P.chain_key(self.fp, tuple(int(t) for t in ref))
        return key if key in self._disk else None

    def pin(self, ref, label: str = "manual") -> bool:
        """Protect a stored snapshot (by key or by its exact token ids) and its whole chain from eviction."""
        with self._lock:
            key = self._key_of(ref)
            if key is None:
                return False
            self._meta[key]["pinned"] = "manual" if label != "auto" else "auto"
            self._dirty = True
            self._flush_index(force=True)
            return True

    def unpin(self, ref) -> bool:
        with self._lock:
            key = self._key_of(ref)
            if key is None or not self._meta[key]["pinned"]:
                return False
            self._meta[key]["pinned"] = None
            self._dirty = True
            self._flush_index(force=True)
            return True

    def profiles(self) -> list:
        """One dict per stored snapshot, most valuable first: identity, size, use, cost, pin and execution profile."""
        with self._lock:
            rows = []
            for key, disk in self._disk.items():
                meta = self._meta.get(key, {})
                rows.append({
                    "key": key, "n_tokens": disk["n"], "parent": disk["parent"], "bytes": disk["bytes"],
                    "tier": "ram+ssd" if key in self._ram else "ssd", "plan_edges": self._plans.get(key),
                    "kind": meta.get("kind"), "hits": meta.get("hits", 0), "last_used": meta.get("last_used"),
                    "seen": self._seen.get(key, {}).get("count", 0), "pinned": meta.get("pinned"),
                    "recompute_seconds": self.recompute_seconds(disk["n"]), "saved_seconds": meta.get("saved_s", 0.0),
                    "config": meta.get("config", {}), "leased": key in self._leases})
            return sorted(rows, key=lambda r: (r["pinned"] is None, -self._value(r["key"])))

    def usage(self) -> dict:
        with self._lock:
            out = super().usage()
            out.update({
                "foreign_entries": len(self._foreign), "foreign_bytes": sum(m["bytes"] for m in self._foreign.values()),
                "pinned": sum(1 for m in self._meta.values() if m["pinned"]),
                "disk_total_bytes": self._disk_total(), "free_reserve": self.free_reserve,
                "index_state": self.index_state, "prefill_tokens_per_second": self.prefill_rate()})
            return out

    def report(self) -> dict:
        """Counters for the benchmark and server logs: hits, misses, restored tokens, SSD use, evictions, time saved."""
        with self._lock:
            s = self.stats
            return {"lookups": s.lookups, "hits": s.hits, "misses": s.lookups - s.hits, "restored_tokens": s.reused_tokens,
                    "ram_hits": s.ram_hits, "ssd_hits": s.ssd_hits, "evicted_ram": s.evicted_ram,
                    "evicted_ssd": s.evicted_ssd, "evicted_ssd_bytes": s.evicted_ssd_bytes,
                    "writes_skipped": s.writes_skipped, "orphans_removed": s.orphans_removed,
                    "invalidated": s.invalidated, "time_saved_seconds": round(s.time_saved_seconds, 3),
                    "ssd_bytes_written": s.ssd_bytes_written, "ssd_bytes_read": s.ssd_bytes_read,
                    **self.extra, **self.usage()}

    def scrub(self, delete: bool = True) -> list:
        """Verify every stored entry against its manifest and checksum. Returns the keys that failed (removed with
        ``delete``, descendants included)."""
        with self._lock:
            bad = []
            for key in list(self._disk):
                if key not in self._disk:
                    continue
                try:
                    P.read_entry(self.directory, key, self.fp, verify_hash=True)
                except P.CorruptEntry:
                    bad.append(key)
                    if delete:
                        self._invalidate(key)
                        self.extra["scrubbed"] += 1
            return bad

    def warm_ram(self, limit_bytes: int | None = None) -> int:
        """After a restart: load the most valuable snapshots (pinned first) into RAM, up to the RAM budget. Returns the
        number loaded. Entries load exactly as a lookup would load them."""
        loaded = 0
        with self._lock:
            budget = self.ram_budget if limit_bytes is None else min(self.ram_budget, int(limit_bytes))
            for row in self.profiles():
                key = row["key"]
                if key in self._ram:
                    continue
                if self._ram_bytes + row["bytes"] > budget:
                    continue
                try:
                    self._acquire((key,))
                    try:
                        self._chain_loading(key)
                    finally:
                        self._release((key,))
                    loaded += 1
                except P.CorruptEntry:
                    self._invalidate(key)
            return loaded

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._meta.clear()
            self._dirty = True
            self._flush_index(force=True)
