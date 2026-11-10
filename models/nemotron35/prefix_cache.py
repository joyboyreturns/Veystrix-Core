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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import state

MAX_TOKENS = 262144
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


class PrefixCache:
    """Doubles as the ``store()`` object the shared server reads for /health (``_disk``, ``stats`` ...)."""

    def __init__(self, directory, identity: dict, *, quota_bytes: int, min_tokens: int = 512):
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.identity = identity
        self.directory = Path(directory).expanduser() / digest
        self.disk_budget = int(quota_bytes)
        self.ram_budget = 0
        self._ram, self._ram_bytes = {}, 0
        self.min_tokens = min_tokens
        self.stats = Stats()
        self._lock = threading.Lock()
        self._tokens: dict[str, np.ndarray] = {}
        self._disk: dict[str, dict] = {}
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
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
            raw = path.read_bytes()
        except OSError:
            return None
        if len(raw) < 8 or raw[:4] != _MAGIC or (len(raw) - 4) % 4:
            return None
        tokens = np.frombuffer(raw, dtype='<u4', offset=4)
        return tokens if 0 < len(tokens) <= MAX_TOKENS else None

    def _scan(self) -> None:
        for tokens_path in self.directory.glob('*.tokens'):
            key = tokens_path.stem
            state_path = self.directory / f'{key}.safetensors'
            tokens = self._read_tokens(tokens_path)
            if tokens is None or not state_path.is_file() or self._key(tokens) != key:
                for stale in (tokens_path, state_path):
                    stale.unlink(missing_ok=True)
                continue
            self._tokens[key] = tokens
            self._disk[key] = {'bytes': state_path.stat().st_size, 'tokens': len(tokens), 'mtime': state_path.stat().st_mtime}
        for orphan in self.directory.glob('*.safetensors'):
            if orphan.stem not in self._tokens:
                orphan.unlink(missing_ok=True)

    # -- lookup ------------------------------------------------------------------------------------------------
    def _best_key(self, ids: np.ndarray):
        best, best_len = None, 0
        with self._lock:
            for key, tokens in self._tokens.items():
                n = len(tokens)
                if best_len < n < len(ids) and n >= self.min_tokens and np.array_equal(tokens, ids[:n]):
                    best, best_len = key, n
        return best, best_len

    def reusable_tokens(self, ids) -> int:
        """How many leading tokens of ``ids`` a stored snapshot could provide (0 when none)."""
        return self._best_key(np.asarray(ids, dtype='<u4'))[1]

    def restore(self, ids):
        """(cache objects, token count) for the longest usable snapshot, or (None, 0)."""
        ids = np.asarray(ids, dtype='<u4')
        key, n = self._best_key(ids)
        if key is None:
            self.stats.cold_lookups += 1
            return None, 0
        state_path, tokens_path = self._paths(key)
        try:
            cache = state.restore(state_path, self.identity, [int(t) for t in ids[:n]])
        except state.SnapshotError:
            self.stats.restore_failures += 1
            self._forget(key)
            return None, 0
        self.stats.ssd_hits += 1
        self.stats.reused_tokens += n
        self.stats.ssd_bytes_read += self._disk[key]['bytes']
        try:
            os.utime(state_path)
        except OSError:
            pass
        self._disk[key]['mtime'] = time.time()
        return cache, n

    # -- store -------------------------------------------------------------------------------------------------
    def save(self, cache, tokens) -> bool:
        """Snapshot ``cache`` (which must hold exactly ``tokens``). Returns False when skipped."""
        tokens = np.asarray(tokens, dtype='<u4')
        if not self.min_tokens <= len(tokens) <= MAX_TOKENS:
            return False
        key = self._key(tokens)
        with self._lock:
            if key in self._tokens:
                return False
        state_path, tokens_path = self._paths(key)
        state.save(state_path, cache, self.identity, [int(t) for t in tokens])
        tmp = tokens_path.with_suffix('.tokens.tmp')
        tmp.write_bytes(_MAGIC + tokens.tobytes())
        os.chmod(tmp, 0o600)
        os.replace(tmp, tokens_path)
        size = state_path.stat().st_size
        with self._lock:
            self._tokens[key] = tokens
            self._disk[key] = {'bytes': size, 'tokens': len(tokens), 'mtime': time.time()}
        self.stats.ssd_writes += 1
        self.stats.ssd_bytes_written += size
        self._enforce_quota(keep=key)
        return True

    def prune(self, keep_tokens, run_lengths, keep_multiple: int = 8192) -> None:
        """After a run finishes, drop its intermediate checkpoints that are strict prefixes of ``keep_tokens``
        (except every ``keep_multiple`` tokens, which later conversations sharing a system prompt can use)."""
        keep_tokens = np.asarray(keep_tokens, dtype='<u4')
        for n in run_lengths:
            if n >= len(keep_tokens) or n % keep_multiple == 0:
                continue
            self._forget(self._key(keep_tokens[:n]))

    def _forget(self, key: str) -> None:
        with self._lock:
            self._tokens.pop(key, None)
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

    def clear(self) -> None:
        for key in list(self._tokens):
            self._forget(key)
