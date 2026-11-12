"""Persistent prefix cache: exact Mamba + KV state snapshots on SSD, keyed by the tokens they represent.

A hybrid model's recurrent state cannot be rewound, so a snapshot is only usable when **all** of its tokens are a
prefix of the new prompt. Entries are therefore (tokens, state) pairs; ``best`` returns the longest entry whose
whole token list is a prefix of the prompt, strictly shorter than it (one token must still be fed for logits).

Snapshots are written at the end of prefill and at checkpoints during a long prefill, so a cancelled or restarted
request resumes from the last checkpoint instead of from zero. Each file is fingerprinted by the model identity and
its tokens and carries a tensor hash (see ``state.py``); anything that fails validation is deleted and counted.
The cache is namespaced by model identity, so another checkpoint, tokenizer or runtime version never reads it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
from contextlib import nullcontext

import numpy as np

from . import state

MAX_TOKENS = 1_048_576
# A coding harness with one tool can have a useful system prefix below 512
# tokens. Keep that boundary so canonicalized tool history can reuse it.
MIN_TOKENS = 256
_MAGIC = b'VXT1'


@dataclass
class Stats:
    reused_tokens: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0
    cold_lookups: int = 0
    restore_failures: int = 0
    prefix_mismatches: int = 0
    state_invalids: int = 0
    evicted_hits: int = 0
    evicted_ram: int = 0
    evicted_ssd: int = 0
    ssd_writes: int = 0
    ssd_bytes_written: int = 0
    ssd_bytes_read: int = 0


def _managed(method):
    @wraps(method)
    def operation(self, *args, **kwargs):
        with self._lock:
            context = self.manager.transaction() if self.manager else nullcontext(None)
            with context as tx:
                self._sync_disk(tx)
                return method(self, *args, _tx=tx, **kwargs)
    return operation


class PrefixCache:
    """Doubles as the ``store()`` object the shared server reads for /health (``_disk``, ``stats`` ...)."""

    def __init__(self, directory, identity: dict, *, quota_bytes: int, min_tokens: int = MIN_TOKENS, manager=None, compatible_positions=None):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.identity = identity
        self.manager = manager
        self.directory = manager.namespace("nemotron35", digest) if manager else Path(directory).expanduser() / digest[:16]
        self.disk_budget = int(quota_bytes)
        self.ram_budget = 0
        self._ram, self._ram_bytes = {}, 0
        self.min_tokens = min_tokens
        self.compatible_positions = compatible_positions
        self.stats = Stats()
        self._lock = threading.RLock()
        self._recovery_lease = None
        self._disk: dict[str, dict] = {}
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if manager is None:
            self._scan()

    # -- the object the server reads ---------------------------------------------------------------------------
    def store(self):
        return self

    # -- index -------------------------------------------------------------------------------------------------
    def _paths(self, key: str):
        return self.directory / f'{key}.safetensors', self.directory / f'{key}.tokens'

    @staticmethod
    def _key(tokens) -> str:
        return hashlib.sha256(np.asarray(tokens, dtype='<u4').tobytes()).hexdigest()[:24]

    def _read_tokens(self, path: Path):
        try:
            if path.stat().st_size > 4 + MAX_TOKENS * 4:
                return None
            raw = path.read_bytes()
        except OSError:
            return None
        if len(raw) < 8 or raw[:4] != _MAGIC or (len(raw) - 4) % 4:
            return None
        tokens = np.frombuffer(raw, dtype='<u4', offset=4)
        return tokens if 0 < len(tokens) <= MAX_TOKENS else None

    def _sync_disk(self, tx):
        if tx is not None:
            self.disk_budget = tx.quota
            self._disk = {entry.key: {"bytes": entry.size, "tokens": entry.tokens, "mtime": entry.mtime}
                          for entry in tx.entries.values()
                          if entry.directory == self.directory and entry.tokens > 0}

    def _scan(self) -> None:
        for tokens_path in self.directory.glob('*.tokens'):
            key = tokens_path.stem
            state_path = self.directory / f'{key}.safetensors'
            tokens = self._read_tokens(tokens_path)
            if tokens is None or not state_path.is_file() or self._key(tokens) != key:
                for stale in (tokens_path, state_path):
                    stale.unlink(missing_ok=True)
                continue
            self._disk[key] = {'bytes': state_path.stat().st_size, 'tokens': len(tokens), 'mtime': state_path.stat().st_mtime}
        for orphan in self.directory.glob('*.safetensors'):
            if orphan.stem not in self._disk:
                orphan.unlink(missing_ok=True)

    # -- lookup ------------------------------------------------------------------------------------------------
    def _best_key(self, ids: np.ndarray, excluded=()):
        accepts = self.compatible_positions(ids) if self.compatible_positions else lambda n: True
        lengths = sorted({entry["tokens"] for entry in self._disk.values()
                          if self.min_tokens <= entry["tokens"] < len(ids)})
        if not lengths:
            return None, 0
        digest = hashlib.sha256()
        packed = memoryview(np.ascontiguousarray(ids[:lengths[-1]])).cast("B")
        candidates, previous = [], 0
        for n in lengths:
            digest.update(packed[4 * previous:4 * n])
            previous = n
            key = digest.hexdigest()[:24]
            if key in self._disk and key not in excluded:
                candidates.append((key, n))
        for key, n in reversed(candidates):
            if accepts(n):
                tokens = self._read_tokens(self._paths(key)[1])
                if tokens is not None and np.array_equal(tokens, ids[:n]):
                    return key, n
        return None, 0

    def _lookup_transaction(self):
        return self.manager.transaction(namespace=self.directory) if self.manager else nullcontext(None)

    def reusable_tokens(self, ids) -> int:
        """How many leading tokens of ``ids`` a stored snapshot could provide (0 when none)."""
        with self._lock, self._lookup_transaction() as tx:
            self._sync_disk(tx)
            return self._best_key(np.asarray(ids, dtype='<u4'))[1]

    def restore(self, ids):
        """(cache objects, token count) for the longest usable snapshot, or (None, 0)."""
        ids = np.asarray(ids, dtype='<u4')
        with self._lock:
            return self._restore(ids)

    def _restore(self, ids):
        excluded = set()
        while True:
            with self._lookup_transaction() as tx:
                self._sync_disk(tx)
                key, n = self._best_key(ids, excluded)
                if key is None:
                    self.stats.cold_lookups += 1
                    if tx is not None:
                        tx.record("nemotron35", self.directory)
                    return None, 0
                lease = tx.lease(self.directory, key) if tx is not None else None
            state_path, _ = self._paths(key)
            try:
                cache = state.restore(state_path, self.identity, ids[:n].tolist())
                if lease is not None:
                    with self._lookup_transaction() as tx:
                        tx.record("nemotron35", self.directory, key, n)
                break
            except state.SnapshotError:
                self.stats.restore_failures += 1
                excluded.add(key)
                if lease is None:
                    self._forget(key)
                else:
                    with self._lookup_transaction() as tx:
                        os.close(lease)
                        lease = None
                        entry = tx.entries.get(f"nemotron35/{self.directory.name}/{key}")
                        if entry is not None and tx.remove(entry):
                            self._disk.pop(key, None)
            finally:
                if lease is not None:
                    os.close(lease)
        self.stats.ssd_hits += 1
        self.stats.reused_tokens += n
        self.stats.ssd_bytes_read += self._disk[key]['bytes']
        if self.manager is None:
            try:
                os.utime(state_path)
            except OSError:
                pass
        self._disk[key]['mtime'] = time.time()
        return cache, n

    # -- store -------------------------------------------------------------------------------------------------
    @_managed
    def save(self, cache, tokens, *, hold=False, _tx) -> bool:
        """Snapshot ``cache`` (which must hold exactly ``tokens``). Returns False when skipped."""
        tokens = np.asarray(tokens, dtype='<u4')
        if not self.min_tokens <= len(tokens) <= MAX_TOKENS:
            return False
        key = self._key(tokens)
        state_path, tokens_path = self._paths(key)
        if _tx is not None:
            entry = _tx.entries.get(f"nemotron35/{self.directory.name}/{key}")
            if entry is not None:
                stored = self._read_tokens(tokens_path) if entry.tokens == len(tokens) else None
                if stored is not None and np.array_equal(stored, tokens):
                    if hold:
                        self._hold(_tx, key)
                    return False
                if not _tx.remove(entry):
                    return False
            estimate = sum(a.nbytes for entry in cache for a in entry.state if a is not None)
            if not _tx.make_room(estimate + tokens.nbytes + 1024 * 1024):
                return False
        elif key in self._disk:
            return False
        tmp = tokens_path.with_suffix('.tokens.tmp')
        try:
            state.save(state_path, cache, self.identity, tokens.tolist())
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(_MAGIC + tokens.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            size = state_path.stat().st_size + tmp.stat().st_size
            if _tx is not None and (_tx.used + size > _tx.quota or
                                    shutil.disk_usage(self.directory).free < _tx.manager.reserve):
                state_path.unlink()
                return False
            os.replace(tmp, tokens_path)
            from veystrix.universal_cache import _sync_directory
            _sync_directory(self.directory)
        finally:
            tmp.unlink(missing_ok=True)
        self._disk[key] = {'bytes': size, 'tokens': len(tokens), 'mtime': time.time()}
        if hold and _tx is not None:
            self._hold(_tx, key)
        self.stats.ssd_writes += 1
        self.stats.ssd_bytes_written += size
        if _tx is None:
            self._enforce_quota(keep=key)
        return True

    def _hold(self, tx, key):
        fd = tx.lease(self.directory, key)
        self.release_recovery()
        self._recovery_lease = fd

    def release_recovery(self):
        with self._lock:
            if self._recovery_lease is not None:
                os.close(self._recovery_lease)
                self._recovery_lease = None

    @_managed
    def prune(self, keep_tokens, run_lengths, keep_multiple: int = 8192, *, _tx) -> None:
        """After a run finishes, drop its intermediate checkpoints that are strict prefixes of ``keep_tokens``
        (except every ``keep_multiple`` tokens, which later conversations sharing a system prompt can use)."""
        keep_tokens = np.asarray(keep_tokens, dtype='<u4')
        for n in run_lengths:
            if n >= len(keep_tokens) or n % keep_multiple == 0:
                continue
            key = self._key(keep_tokens[:n])
            if _tx is None:
                self._forget(key)

    def _forget(self, key: str) -> None:
        with self._lock:
            self._disk.pop(key, None)
        for path in self._paths(key):
            path.unlink(missing_ok=True)

    def _enforce_quota(self, keep: str) -> None:
        while True:
            with self._lock:
                total = sum(m['bytes'] for m in self._disk.values())
                if total <= self.disk_budget:
                    return
                victims = sorted((m['mtime'], k) for k, m in self._disk.items() if k != keep)
            if not victims:
                return
            self._forget(victims[0][1])
            self.stats.evicted_ssd += 1

    @_managed
    def clear(self, *, _tx) -> None:
        if _tx is None:
            for key in list(self._disk):
                self._forget(key)
        else:
            for entry in list(_tx.entries.values()):
                if entry.directory == self.directory and _tx.remove(entry):
                    self._disk.pop(entry.key, None)
