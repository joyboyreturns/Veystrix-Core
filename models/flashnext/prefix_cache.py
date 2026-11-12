"""Exact prefix-state cache for Flash-Next: hot RAM tier, cold SSD tier (opt-in prototype).

What a reusable prefix is for this model. Attention KV alone is not a snapshot. The state after N
tokens is:

* 12 full-attention layers (``QSAKVCache``): ``keys``, ``values`` (``[1, 2, N, 256]`` bf16), the QSA
  indexer's raw ``index_keys`` (``[1, N, 128]``) and ``index_position_ids``. These are sliceable along
  the token axis, so they are stored as **deltas** and chained: a snapshot at 10K that extends one
  at 8K stores only tokens 8K to 10K of KV.
* 36 GatedDeltaNet layers (``ArraysCache``): a fp32 recurrent state and a bf16 convolution state (one
  extra pair on the PLE layer). These **cannot be sliced or rewound**; the state after N tokens only
  exists at N. Each snapshot therefore carries a full copy (about 113 MB for the 36 layers).
* The token tape, which is the key.

So a prefix can be reused only at a token count where a snapshot was taken. Snapshots are taken at
boundaries the prefill is stopped at (end of the system prompt, chunk edges), never inferred later.

Values are stored exactly as they are in the live cache (no lossy codec), so a restored cache is
bit-identical to the one that was captured. Arrays handed back by ``restore`` are fresh Python
objects: continuing generation cannot mutate a stored snapshot (copy-on-write at the object level;
MLX buffers are only reused in place when unshared).

Safety: every entry is keyed by a SHA-256 chain over the engine fingerprint and the exact token ids,
and the token ids are compared again on load. A fingerprint mismatch, a corrupt or truncated file, or
an unknown cache type is a miss (and the entry is removed), never an approximation. Files are written
to a temporary name, fsynced and renamed, the manifest last, with mode 0600 in a 0700 directory.

Nothing here is wired into inference unless ``PrefixCache`` is constructed and called.
"""
from __future__ import annotations

import copy

import hashlib
import fcntl
import json
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import mlx.core as mx

FORMAT_VERSION = "3"      # 3: bind reconstruction metadata to the checksummed tensor payload
OWNER_LOCK = ".veystrix-owner.lock"
_OWNER_MUTEX = threading.Lock()
_OWNERS = {}
_OWNER_PID = os.getpid()


class CacheOwnershipError(RuntimeError):
    """Another process owns the writable prefix-cache directory."""


def _claim_directory(directory: Path):
    """Lock order: process registry, directory ownership, instance lock, profile index lock.

    The lock file is never unlinked; flock is released automatically on process
    exit. Instances within one process share the owner descriptor.
    """
    global _OWNER_PID
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = str(directory.resolve())
    pid = os.getpid()
    with _OWNER_MUTEX:
        if _OWNER_PID != pid:
            for fd, _ in _OWNERS.values():
                os.close(fd)
            _OWNERS.clear()
            _OWNER_PID = pid
        if path in _OWNERS:
            fd, refs = _OWNERS[path]
            _OWNERS[path] = (fd, refs + 1)
            return path, pid
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(Path(path) / OWNER_LOCK), flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(fd)
            raise CacheOwnershipError(f"prefix cache is owned by another process: {path}") from error
        _OWNERS[path] = (fd, 1)
        return path, pid


def _release_directory(path, pid):
    with _OWNER_MUTEX:
        if pid != os.getpid() or pid != _OWNER_PID or path not in _OWNERS:
            return
        fd, refs = _OWNERS[path]
        if refs > 1:
            _OWNERS[path] = (fd, refs - 1)
        else:
            del _OWNERS[path]
            os.close(fd)


def _with_directory_owner(function):
    """Standalone file helpers obey the same ownership rule as cache instances."""
    @wraps(function)
    def wrapped(directory, *args, **kwargs):
        owner = _claim_directory(Path(directory))
        try:
            return function(directory, *args, **kwargs)
        finally:
            _release_directory(*owner)
    return wrapped
KV_FIELDS = ("keys", "values", "index_keys", "index_position_ids")
QUANTIZED_KV_FIELDS = (
    "keys_0", "keys_1", "keys_2",
    "values_0", "values_1", "values_2",
    "index_keys", "index_position_ids",
)
# Axis of the token dimension for each sliceable field (position ids are sliced on the last axis).
KV_AXIS = {
    "keys": 2, "values": 2, "index_keys": 1, "index_position_ids": -1,
    "keys_0": 2, "keys_1": 2, "keys_2": 2,
    "values_0": 2, "values_1": 2, "values_2": 2,
}


class UnsupportedCache(RuntimeError):
    """A cache object this module cannot snapshot exactly (fail closed)."""


class CorruptEntry(RuntimeError):
    pass


def fingerprint(**parts) -> str:
    """Digest of everything that must match for a stored prefix to be valid.

    Pass the checkpoint identity, tokenizer and chat-template digests, engine and settings that change
    numerics (quantization, QSA flags, ring or chunk settings). Order does not matter.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _chain_key_bytes(fp: str, length: int, token_bytes) -> str:
    digest = hashlib.sha256(fp.encode())
    digest.update(length.to_bytes(8, "little"))
    digest.update(token_bytes)
    return digest.hexdigest()


def chain_key(fp: str, tokens) -> str:
    return _chain_key_bytes(fp, len(tokens), _token_bytes(tokens))


def _token_bytes(tokens) -> bytes:
    if hasattr(tokens, "__len__"):
        try:
            # Keep existing cache keys and pack ordinary token sequences in one call.
            return struct.pack(f"<{len(tokens)}i", *tokens)
        except struct.error:
            pass
    return b"".join(int(t).to_bytes(4, "little", signed=True) for t in tokens)


def _own(array):
    """A new Python object for the same values (so no later in-place update can reach it)."""
    return None if array is None else array[...]


# -- capture and restore ---------------------------------------------------------------------


@dataclass
class Layer:
    kind: str                                   # "arrays" or "qsa_kv"
    states: list = field(default_factory=list)  # arrays: every slot, in order (None allowed)
    kv: dict = field(default_factory=dict)      # qsa_kv: delta fields for tokens [parent_n, n)

    def nbytes(self) -> int:
        total = sum(a.nbytes for a in self.states if a is not None)
        return total + sum(a.nbytes for a in self.kv.values() if a is not None)


@dataclass
class Entry:
    key: str
    parent: str | None
    n_tokens: int
    parent_tokens: int
    token_delta: tuple                          # ids [parent_tokens, n_tokens)
    layers: list
    created: float = field(default_factory=time.time)
    # Language-model-level position state (``_position_ids``, ``_rope_deltas``). A fresh engine has
    # neither, and without them the first call on a restored cache numbers its tokens from 0.
    lang: dict = field(default_factory=dict)
    # Chunk-plan provenance: the internal chunk edges (strictly between 0 and n_tokens) of the run that
    # produced this state. A state is only a valid continuation point for a prompt whose own cold plan
    # makes exactly these cuts before n_tokens and also cuts at n_tokens, because chunk shapes change the
    # floating-point result. None means the history is unknown, which a prefix session refuses.
    plan: tuple | None = None

    def nbytes(self) -> int:
        return sum(layer.nbytes() for layer in self.layers) + sum(
            a.nbytes for a in self.lang.values() if a is not None)


LANG_FIELDS = (("position_ids", "_position_ids"), ("rope_deltas", "_rope_deltas"))


def capture_language(language) -> dict:
    return {name: _own(getattr(language, attr, None)) for name, attr in LANG_FIELDS
            if getattr(language, attr, None) is not None}


def apply_language(language, lang: dict) -> None:
    """Put the position state back on the language model (None for anything not stored)."""
    for name, attr in LANG_FIELDS:
        setattr(language, attr, _own(lang.get(name)))


def _slice(array, field_name, start, stop):
    axis = KV_AXIS[field_name]
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, stop)
    return array[tuple(index)]


def kind_of(cache) -> str:
    name = type(cache).__name__
    if name == "ArraysCache":
        if getattr(cache, "lengths", None) is not None or getattr(cache, "left_padding", None) is not None:
            raise UnsupportedCache("ArraysCache with batch padding state")
        return "arrays"
    if name == "QSAKVCache":
        return "qsa_kv"
    if name in ("QSAQuantizedKVCache", "AffineQuantizedQSAKVCache"):
        return "qsa_quantized_kv"
    if name == "MixedPrecisionQSAKVCache":
        return "qsa_mixed_v1"
    raise UnsupportedCache(f"cannot snapshot a {name}")


def _checked_plan(plan, n_tokens):
    if plan is None:
        return None
    edges = tuple(int(e) for e in plan)
    if any(not 0 < e < n_tokens for e in edges) or list(edges) != sorted(set(edges)):
        raise ValueError("plan edges must be increasing and strictly inside the prefix")
    return edges


def capture(cache_list, tokens, fp: str, parent: Entry | None = None, language=None, plan=None) -> Entry:
    """Snapshot the live cache after ``tokens``. ``parent`` must be the entry for a token prefix.

    The cache is evaluated first so the stored arrays are materialised values.
    """
    tokens = tuple(int(t) for t in tokens)
    plan = _checked_plan(plan, len(tokens))
    parent_n = 0 if parent is None else parent.n_tokens
    if parent is not None and tokens[:parent_n] != parent_prefix(parent):
        raise ValueError("parent entry is not a prefix of these tokens")
    kinds = [kind_of(c) for c in cache_list]
    mx.eval([_state_arrays(c, k) for c, k in zip(cache_list, kinds)])
    layers = []
    for cache, kind in zip(cache_list, kinds):
        if kind == "arrays":
            layers.append(Layer("arrays", states=[_own(a) for a in cache.cache]))
            continue
        offset = int(cache.offset)
        if offset != len(tokens):
            raise ValueError(f"cache holds {offset} tokens but {len(tokens)} were given")
        if kind == "qsa_mixed_v1":
            if parent is not None and parent.layers[len(layers)].kind != kind:
                raise ValueError("mixed KV parent has a different cache kind")
            layers.append(Layer(kind, kv=cache.snapshot(parent_n, parent_n)))
            continue
        kv = {}
        if kind == "qsa_quantized_kv":
            # Quantized keys & values (data, scales, biases)
            for i, part in enumerate(cache.keys):
                kv[f"keys_{i}"] = _own(_slice(part, f"keys_{i}", parent_n, offset))
            for i, part in enumerate(cache.values):
                kv[f"values_{i}"] = _own(_slice(part, f"values_{i}", parent_n, offset))
            if cache.index_keys is not None:
                kv["index_keys"] = _own(_slice(cache.index_keys, "index_keys", parent_n, offset))
            if cache.index_position_ids is not None:
                kv["index_position_ids"] = _own(_slice(cache.index_position_ids, "index_position_ids", parent_n, offset))
            layers.append(Layer("qsa_quantized_kv", kv=kv))
            continue
        for name in KV_FIELDS:
            value = getattr(cache, name)
            if value is None:
                continue
            if value.shape[KV_AXIS[name] % value.ndim] != offset and name != "keys" and name != "values":
                raise ValueError(f"{name} holds {value.shape[KV_AXIS[name] % value.ndim]} tokens, cache offset is {offset}")
            kv[name] = _own(_slice(value, name, parent_n, offset))
        layers.append(Layer("qsa_kv", kv=kv))
    lang = capture_language(language) if language is not None else {}
    entry = Entry(chain_key(fp, tokens), None if parent is None else parent.key, len(tokens),
                  parent_n, tokens[parent_n:], layers, lang=lang, plan=plan)
    mx.eval([a for layer in layers for a in (*layer.states, *layer.kv.values()) if a is not None]
            + list(lang.values()))
    return entry


def freeze_cache(cache):
    """A stand-in for ``cache`` as it is right now, for a snapshot taken after the live cache has moved on.

    Layer-major prefill reaches a chunk edge in the middle of a window layer by layer; the snapshot for that edge is assembled
    after the window from these per-layer stand-ins. Same class (so ``kind_of`` and ``capture`` treat it as the real thing), array
    attributes replaced by new objects over the same values cut to the cache's own offset (the arrays are immutable values, so no
    later in-place update reaches them). The caller must have evaluated the cache already."""
    kind = kind_of(cache)
    frozen = copy.copy(cache)
    if kind == "arrays":
        frozen.cache = [_own(a) for a in cache.cache]
        return frozen
    offset = int(cache.offset)
    if kind == "qsa_mixed_v1":
        for name in ("keys", "values", "recent_keys", "recent_values", "index_keys", "index_position_ids"):
            value = getattr(cache, name)
            setattr(frozen, name, tuple(_own(a) for a in value) if isinstance(value, tuple) else _own(value))
        return frozen
    if kind == "qsa_quantized_kv":
        for attr in ("keys", "values"):
            parts = getattr(cache, attr)
            if parts is not None:
                setattr(frozen, attr, tuple(_own(_slice(part, f"{attr}_{i}", 0, offset)) for i, part in enumerate(parts)))
        for attr in ("index_keys", "index_position_ids"):
            value = getattr(cache, attr)
            if value is not None:
                setattr(frozen, attr, _own(_slice(value, attr, 0, offset)))
        return frozen
    for name in KV_FIELDS:
        value = getattr(cache, name)
        if value is not None:
            setattr(frozen, name, _own(_slice(value, name, 0, offset)))
    return frozen


def parent_prefix(entry: Entry) -> tuple:
    return getattr(entry, "_tokens", None) or ()


def _state_arrays(cache, kind):
    if kind == "arrays":
        return [a for a in cache.cache if a is not None]
    if kind == "qsa_mixed_v1":
        return cache.state
    if kind == "qsa_quantized_kv":
        arrays = []
        for tup in (cache.keys, cache.values):
            if tup:
                arrays.extend([a for a in tup if a is not None])
        if cache.index_keys is not None:
            arrays.append(cache.index_keys)
        if cache.index_position_ids is not None:
            arrays.append(cache.index_position_ids)
        return arrays
    return [a for a in (cache.keys, cache.values, cache.index_keys, cache.index_position_ids) if a is not None]


def restore(chain, make_cache):
    """Build a fresh cache list from ``chain`` (root first, leaf last)."""
    caches = make_cache()
    leaf = chain[-1]
    if len(caches) != len(leaf.layers):
        raise CorruptEntry("layer count differs from the model's cache")
    for index, cache in enumerate(caches):
        kind = kind_of(cache)
        layer = leaf.layers[index]
        if layer.kind != kind:
            raise CorruptEntry(f"layer {index}: stored {layer.kind}, model has {kind}")
        if kind == "arrays":
            if len(layer.states) != len(cache.cache):
                raise CorruptEntry(f"layer {index}: {len(layer.states)} state slots, model has {len(cache.cache)}")
            for slot, value in enumerate(layer.states):
                cache.cache[slot] = _own(value)
            continue
        if kind == "qsa_mixed_v1":
            try:
                cache.restore_snapshot(_join_mixed(chain, index), leaf.n_tokens)
            except (ValueError, TypeError, IndexError, KeyError) as error:
                raise CorruptEntry(f"layer {index}: {error}") from error
            continue
        fields = QUANTIZED_KV_FIELDS if kind == "qsa_quantized_kv" else KV_FIELDS
        parts = {name: [] for name in fields}
        for entry in chain:
            for name, value in entry.layers[index].kv.items():
                if name in parts:
                    parts[name].append(value)
        joined = {}
        for name, pieces in parts.items():
            if not pieces:
                joined[name] = None
            elif len(pieces) == 1:
                joined[name] = _own(pieces[0])
            else:
                axis = KV_AXIS[name] % pieces[0].ndim
                joined[name] = mx.concatenate(pieces, axis=axis)
        if kind == "qsa_quantized_kv":
            keys = (joined["keys_0"], joined["keys_1"], joined["keys_2"])
            values = (joined["values_0"], joined["values_1"], joined["values_2"])
            cache.state = (keys, values, joined["index_keys"], joined["index_position_ids"])
        else:
            cache.state = (joined["keys"], joined["values"], joined["index_keys"], joined["index_position_ids"])
    mx.eval([_state_arrays(c, kind_of(c)) for c in caches])
    return caches


def _join_mixed(chain, index):
    """Reconstruct packed history/index deltas; only the leaf's recent tier matters."""
    parts = {}
    history = 0
    identity = None
    for entry in chain:
        layer = entry.layers[index]
        if layer.kind != "qsa_mixed_v1":
            raise ValueError("mixed KV chain contains a different cache kind")
        fields = layer.kv
        config = fields["config"]
        if config.shape != (8,) or config.dtype != mx.int64:
            raise ValueError("invalid mixed KV chain configuration")
        values = config.tolist()
        settings = values[:5] + values[7:]
        if identity is None:
            identity = settings
        offset, end = values[5:7]
        if identity != settings or offset != entry.n_tokens or not history <= end <= offset:
            raise ValueError("incompatible mixed KV chain offsets or settings")
        for name, value in fields.items():
            if name.startswith(("keys_", "values_")):
                axis, length = -2, entry.n_tokens - entry.parent_tokens
            elif name in ("index_keys", "index_position_ids"):
                axis = 1 if name == "index_keys" else -1
                length = entry.n_tokens - entry.parent_tokens
            else:
                continue
            if value.shape[axis] != length:
                raise ValueError(f"wrong mixed KV delta length: {name}")
            parts.setdefault(name, []).append(value)
        history = end
    result = {name: value for name, value in chain[-1].layers[index].kv.items()
              if name == "config" or name.startswith("recent_")}
    for name, values in parts.items():
        axis = 1 if name == "index_keys" else -1 if name == "index_position_ids" else -2
        if name == "index_position_ids" and any(a.ndim == 3 for a in values):
            channels = next(a.shape[0] for a in values if a.ndim == 3)
            values = [mx.broadcast_to(a[None], (channels, *a.shape)) if a.ndim == 2 else a for a in values]
        result[name] = values[0] if len(values) == 1 else mx.concatenate(values, axis=axis)
    return result


# -- disk format -----------------------------------------------------------------------------


def _tensors_of(entry: Entry) -> dict:
    tensors = {"tokens": mx.array(list(entry.token_delta), dtype=mx.int32)}
    for index, layer in enumerate(entry.layers):
        for slot, value in enumerate(layer.states):
            if value is not None:
                tensors[f"l{index}.s{slot}"] = value
        for name, value in layer.kv.items():
            tensors[f"l{index}.{name}"] = value
    for name, value in entry.lang.items():
        tensors[f"lang.{name}"] = value
    return tensors


def _layer_meta(entry: Entry) -> list:
    return [{"kind": layer.kind, "slots": len(layer.states),
             "none": [i for i, a in enumerate(layer.states) if a is None]} for layer in entry.layers]


def _tensor_spec(tensors: dict) -> dict:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(tensors.items())}


def _binding(entry: Entry, fp: str, tensors: dict) -> dict:
    return {"format": FORMAT_VERSION, "key": entry.key, "parent": entry.parent, "fingerprint": fp,
            "n_tokens": entry.n_tokens, "parent_tokens": entry.parent_tokens,
            "tokens_sha256": hashlib.sha256(_token_bytes(entry.token_delta)).hexdigest(),
            "layers": _layer_meta(entry), "lang": sorted(entry.lang),
            "plan_edges": None if entry.plan is None else list(entry.plan),
            "tensor_spec": _tensor_spec(tensors)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@_with_directory_owner
def write_entry(directory: Path, entry: Entry, fp: str) -> int:
    """Write one entry atomically. Returns bytes written. The manifest is committed last."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    data = directory / f"{entry.key}.safetensors"
    tmp = directory / f".{entry.key}.{uuid.uuid4().hex}.tmp.safetensors"  # MLX appends the suffix if missing
    tensors = _tensors_of(entry)
    binding = _binding(entry, fp, tensors)
    mx.save_safetensors(str(tmp), tensors,
                        metadata={"format": FORMAT_VERSION, "key": entry.key,
                                  "plan": json.dumps(binding["plan_edges"]),
                                  "binding": json.dumps(binding, sort_keys=True, separators=(",", ":"))})
    os.chmod(tmp, 0o600)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, data)
    manifest = {**binding, "file_bytes": data.stat().st_size,
                "file_sha256": _sha256_file(data), "created": entry.created}
    mtmp = directory / f".{entry.key}.{uuid.uuid4().hex}.json.tmp"
    fd = os.open(mtmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(manifest, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(mtmp, directory / f"{entry.key}.json")
    _fsync_dir(directory)
    return manifest["file_bytes"] + (directory / f"{entry.key}.json").stat().st_size


@_with_directory_owner
def read_manifest(directory: Path, key: str) -> dict:
    try:
        return json.loads((directory / f"{key}.json").read_text())
    except (OSError, ValueError) as error:
        raise CorruptEntry(f"manifest unreadable: {error}") from error


@_with_directory_owner
def read_entry(directory: Path, key: str, fp: str, verify_hash: bool = True, *, managed: bool = False) -> Entry:
    manifest = read_manifest(directory, key)
    if manifest.get("format") != FORMAT_VERSION or manifest.get("key") != key:
        raise CorruptEntry("format or key mismatch")
    if manifest.get("fingerprint") != fp:
        raise CorruptEntry("fingerprint mismatch")
    data = directory / f"{key}.safetensors"
    try:
        # Managed snapshots are immutable under the global transaction. Hash and load
        # the same open file. Standalone readers still load a private verified copy.
        from contextlib import nullcontext
        fd = os.open(data, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source, (nullcontext(None) if managed else tempfile.TemporaryFile(mode="w+b")) as copy:
            before = os.fstat(source.fileno())
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                size += len(block)
                digest.update(block)
                if copy is not None:
                    copy.write(block)
            if size != manifest["file_bytes"]:
                raise CorruptEntry(f"size {size} differs from manifest {manifest['file_bytes']}")
            if digest.hexdigest() != manifest["file_sha256"]:
                raise CorruptEntry("checksum mismatch")
            verified = source if copy is None else copy
            if copy is not None:
                copy.flush()
            verified.seek(0)
            tensors, metadata = mx.load(verified, format="safetensors", return_metadata=True)
            mx.eval(*tensors.values())
            after = os.fstat(source.fileno())
            if managed and (before.st_ctime_ns, before.st_size) != (after.st_ctime_ns, after.st_size):
                raise CorruptEntry("snapshot changed during restore")
    except CorruptEntry:
        raise
    except Exception as error:
        raise CorruptEntry(f"unreadable safetensors: {error}") from error
    if metadata.get("key") != key:
        raise CorruptEntry("embedded key mismatch")
    try:
        binding = json.loads(metadata["binding"])
        expected = {name: manifest[name] for name in binding}
        if (not isinstance(binding, dict) or set(binding) != {
                "format", "key", "parent", "fingerprint", "n_tokens", "parent_tokens",
                "tokens_sha256", "layers", "lang", "plan_edges", "tensor_spec"}
                or binding != expected or binding["tensor_spec"] != _tensor_spec(tensors)):
            raise CorruptEntry("snapshot binding or tensor metadata mismatch")
    except (KeyError, TypeError, ValueError) as error:
        raise CorruptEntry(f"snapshot binding unreadable: {error}") from error
    # The plan is in the checksummed file as well as the manifest; the two must agree.
    try:
        embedded_plan = json.loads(metadata.get("plan", "null"))
    except ValueError as error:
        raise CorruptEntry("embedded plan unreadable") from error
    if embedded_plan != manifest.get("plan_edges"):
        raise CorruptEntry("plan provenance mismatch between manifest and data file")
    tokens = tuple(int(t) for t in tensors["tokens"].tolist())
    if hashlib.sha256(_token_bytes(tokens)).hexdigest() != manifest["tokens_sha256"]:
        raise CorruptEntry("token digest mismatch")
    if len(tokens) != manifest["n_tokens"] - manifest["parent_tokens"]:
        raise CorruptEntry("token count mismatch")
    layers = []
    for index, meta in enumerate(manifest["layers"]):
        if meta["kind"] == "arrays":
            states = [None if slot in meta["none"] else tensors.get(f"l{index}.s{slot}") for slot in range(meta["slots"])]
            if any(s is None for slot, s in enumerate(states) if slot not in meta["none"]):
                raise CorruptEntry(f"layer {index}: missing state tensor")
            layers.append(Layer("arrays", states=states))
        elif meta["kind"] == "qsa_mixed_v1":
            prefix = f"l{index}."
            kv = {name[len(prefix):]: a for name, a in tensors.items() if name.startswith(prefix)}
            if "config" not in kv:
                raise CorruptEntry(f"layer {index}: missing mixed KV configuration")
            layers.append(Layer("qsa_mixed_v1", kv=kv))
        elif meta["kind"] == "qsa_quantized_kv":
            kv = {name: tensors[f"l{index}.{name}"] for name in QUANTIZED_KV_FIELDS if f"l{index}.{name}" in tensors}
            if not all(k in kv for k in ("keys_0", "keys_1", "keys_2", "values_0", "values_1", "values_2")):
                raise CorruptEntry(f"layer {index}: missing quantized KV tensors")
            layers.append(Layer("qsa_quantized_kv", kv=kv))
        else:
            kv = {name: tensors[f"l{index}.{name}"] for name in KV_FIELDS if f"l{index}.{name}" in tensors}
            if "keys" not in kv or "values" not in kv:
                raise CorruptEntry(f"layer {index}: missing KV tensors")
            layers.append(Layer("qsa_kv", kv=kv))
    lang = {}
    for name in manifest.get("lang", []):
        if f"lang.{name}" not in tensors:
            raise CorruptEntry(f"missing language state {name}")
        lang[name] = tensors[f"lang.{name}"]
    return Entry(key, manifest["parent"], manifest["n_tokens"], manifest["parent_tokens"], tokens, layers,
                 created=manifest.get("created", 0.0), lang=lang,
                 plan=None if embedded_plan is None else tuple(int(e) for e in embedded_plan))


# -- the two-tier cache ----------------------------------------------------------------------


OUTCOME_RAM_HIT = "RAM_HIT"
OUTCOME_SSD_HIT = "SSD_HIT"
OUTCOME_PREFIX_MISMATCH = "PREFIX_MISMATCH"
OUTCOME_STATE_INVALID = "STATE_INVALID"
OUTCOME_EVICTED = "EVICTED"
OUTCOME_RESTORE_FAILED = "RESTORE_FAILED"
OUTCOME_COLD = "COLD"
ALL_OUTCOMES = (
    OUTCOME_RAM_HIT,
    OUTCOME_SSD_HIT,
    OUTCOME_PREFIX_MISMATCH,
    OUTCOME_STATE_INVALID,
    OUTCOME_EVICTED,
    OUTCOME_RESTORE_FAILED,
    OUTCOME_COLD,
)


class LookupResult(tuple):
    """2-tuple (hit, caches) exposing .outcome, .hit, .caches for backward compatibility."""

    def __new__(cls, hit, caches, outcome: str):
        obj = super().__new__(cls, (hit, caches))
        obj.hit = hit
        obj.caches = caches
        obj.outcome = outcome
        return obj


@dataclass
class Hit:
    n_tokens: int                 # prompt tokens whose state was restored
    tier: str                     # "ram" or "ssd"
    seconds: float                # restoration time (not prefill speed)
    bytes_loaded: int
    chain: tuple                  # entry keys, root first
    outcome: str = ""             # "RAM_HIT" or "SSD_HIT"


@dataclass
class Stats:
    lookups: int = 0
    hits: int = 0
    reused_tokens: int = 0
    missed_tokens: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0
    prefix_mismatches: int = 0
    state_invalids: int = 0
    evicted_hits: int = 0
    restore_failures: int = 0
    cold_lookups: int = 0
    puts: int = 0
    deduped_puts: int = 0
    ssd_writes: int = 0
    ssd_bytes_written: int = 0
    ssd_bytes_read: int = 0
    evicted_ram: int = 0
    pressure_invalidated: int = 0
    pressure_invalidated_bytes: int = 0
    evicted_ssd: int = 0
    invalidated: int = 0
    restore_seconds: float = 0.0
    rejected_plan: int = 0        # candidates skipped because their chunk-plan history did not fit
    plan_conflicts: int = 0       # puts at known tokens whose plan differed (the stored entry is kept)
    evicted_ssd_bytes: int = 0
    writes_skipped: int = 0       # SSD writes not made because no room could be found inside the limits
    orphans_removed: int = 0      # files of an interrupted write removed at startup
    time_saved_seconds: float = 0.0   # estimated prefill seconds avoided by hits, minus restore time


class _Lease:
    def __init__(self, cache, keys):
        self._cache, self._keys = cache, keys

    def __enter__(self):
        self._cache._acquire(self._keys)
        return self

    def __exit__(self, *exc):
        self._cache._release(self._keys)
        return False


def _managed_disk(method):
    """Refresh the SSD view and serialize disk operations across local engines."""
    @wraps(method)
    def operation(self, *args, **kwargs):
        if self.manager is None:
            return method(self, *args, **kwargs)
        with self._lock:
            if self._transaction is not None:
                return method(self, *args, **kwargs)
            with self.manager.transaction() as tx:
                self._transaction = tx
                try:
                    self._sync_managed_disk()
                    return method(self, *args, **kwargs)
                finally:
                    self._transaction = None
    return operation


class PrefixCache:
    """Longest-exact-prefix lookup over chained snapshots.

    ``ram_budget`` and ``disk_budget`` are byte limits. ``directory=None`` disables the SSD tier.
    Entries are never modified after creation. Thread-safe for the calls below.
    """

    def __init__(self, fp: str, ram_budget: int, directory=None, disk_budget: int = 0,
                 verify_hash: bool = True, write_through: bool = True,
                 min_disk_free_bytes: int | None = None, manager=None):
        self.fp = fp
        self.manager = manager
        self._transaction = None
        self._lease_fds = {}
        if manager is not None:
            directory = manager.namespace("flashnext", fp)
        self.ram_budget = int(ram_budget)
        self.disk_budget = int(disk_budget)
        self.directory = Path(directory) if directory else None
        self.verify_hash = verify_hash
        self.write_through = write_through
        self.min_disk_free_bytes = (
            int(min_disk_free_bytes) if min_disk_free_bytes is not None
            else int(float(os.environ.get("FLASHNEXT_DISK_RESERVE_GB", "50.0")) * 1e9)
        )
        self.stats = Stats()
        self._lock = threading.RLock()
        self._ram: OrderedDict[str, Entry] = OrderedDict()
        self._ram_bytes = 0
        self._disk: dict[str, dict] = {}              # key -> {"n", "parent", "bytes", "atime"}
        self._foreign: dict[str, dict] = {}           # key -> {"bytes", "atime", "parent"}
        self._tokens: dict[str, tuple] = {}           # key -> full prefix tokens (for exact checks)
        self._rolling: dict[bytes, str] = {}          # token-only SHA-256 -> verified entry key
        self._rolling_keys: dict[str, bytes] = {}
        self._plans: dict[str, tuple | None] = {}     # key -> chunk-plan provenance of the stored state
        self._lengths: set[int] = set()
        self._leases: dict[str, int] = {}             # key -> holders; a leased entry is never evicted or dropped
        self._evicted_keys: OrderedDict[str, float] = OrderedDict()
        if manager is not None:
            self._initialize_managed_disk()
        elif self.directory is not None:
            owner = _claim_directory(self.directory)
            self._owner_finalizer = weakref.finalize(self, _release_directory, *owner)
            try:
                self._scan()
            except BaseException:
                self._owner_finalizer()
                raise

    @_managed_disk
    def _initialize_managed_disk(self):
        pass

    def _sync_managed_disk(self):
        tx = self._transaction
        self.disk_budget = tx.quota
        local = {entry.key: entry for entry in tx.entries.values()
                 if entry.directory == self.directory and entry.tokens > 0}
        removed = self._disk.keys() - local.keys()
        self.stats.evicted_ssd += len(removed)
        self.stats.evicted_ssd_bytes += sum(self._disk[key]["bytes"] for key in removed)
        # A dropped disk entry can also invalidate a RAM-only descendant.
        for key, entry in sorted(self._ram.items(), key=lambda item: item[1].n_tokens):
            if entry.parent in removed:
                removed.add(key)
        for key in removed:
            entry = self._ram.pop(key, None)
            if entry is not None:
                self._ram_bytes -= entry.nbytes()
            self._forget_tokens(key)
            self._plans.pop(key, None)
        self._disk = {key: {"n": entry.tokens, "parent": entry.parent,
                            "bytes": entry.size, "atime": entry.mtime}
                      for key, entry in local.items()}
        self._plans.update({key: entry.plan for key, entry in local.items()})
        self._lengths = {e.n_tokens for e in self._ram.values()} | {e.tokens for e in local.values()}

    # ---- index ----
    def _scan(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        for manifest_path in sorted(self.directory.glob("*.json")):
            key = manifest_path.stem
            if not re.fullmatch(r"[0-9a-f]{64}", key):
                continue
            try:
                manifest = read_manifest(self.directory, key)
                data = self.directory / f"{key}.safetensors"
                if manifest.get("fingerprint") != self.fp or manifest.get("format") != FORMAT_VERSION:
                    # Count actual bytes on disk, including the manifest, as
                    # write_entry does for newly persisted snapshots.
                    size = (data.stat().st_size if data.exists() else 0) + manifest_path.stat().st_size
                    self._foreign[key] = {
                        "bytes": size,
                        "atime": manifest_path.stat().st_mtime,
                        "parent": manifest.get("parent"),
                    }
                    continue
                if data.stat().st_size != manifest["file_bytes"]:
                    raise CorruptEntry("size")
                self._disk[key] = {"n": manifest["n_tokens"], "parent": manifest["parent"],
                                   "bytes": manifest["file_bytes"] + manifest_path.stat().st_size,
                                   "atime": manifest_path.stat().st_mtime}
                edges = manifest.get("plan_edges")
                self._plans[key] = None if edges is None else tuple(int(e) for e in edges)
            except (CorruptEntry, OSError, KeyError):
                self._drop_files(key)
                self.stats.invalidated += 1
        for tmp in self.directory.glob(".*.tmp*"):
            if re.fullmatch(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.(?:tmp\.safetensors|json\.tmp)", tmp.name):
                tmp.unlink(missing_ok=True)
        # A chain is only usable if every ancestor is present.
        for key in list(self._disk):
            if not self._ancestors_present(key):
                self._drop_files(key)
                self._disk.pop(key, None)
                self.stats.invalidated += 1
        self._lengths = {meta["n"] for meta in self._disk.values()}

    def _ancestors_present(self, key) -> bool:
        parent = self._disk[key]["parent"]
        while parent is not None:
            if parent not in self._disk:
                return False
            parent = self._disk[parent]["parent"]
        return True

    def _drop_files(self, key) -> None:
        if self.directory is None:
            return
        for suffix in (".json", ".safetensors"):
            (self.directory / f"{key}{suffix}").unlink(missing_ok=True)
        if self._transaction is not None:
            self._transaction.entries.pop(f"flashnext/{self.fp}/{key}", None)

    # ---- store ----
    @_managed_disk
    def put(self, tokens, cache_list, parent: Hit | Entry | None = None, language=None, plan=None,
            kind: str | None = None, *, auto_parent: bool = True) -> Entry:
        """Capture the live cache at ``tokens`` and store it. ``parent`` is the Hit or Entry it extends.

        ``plan`` is the chunk-plan provenance (internal edges before ``len(tokens)``). An entry stored
        without one can be restored by a bare ``lookup`` but never by a prefix session. If the tokens are
        already stored with a different plan, the stored entry is kept (children are chained to it, so it
        is never replaced) and ``plan_conflicts`` counts the miss. ``kind`` labels the snapshot ("system" for the
        end of the system message); this class ignores it, ``prefix_profiles`` records it."""
        with self._lock:
            tokens = tuple(int(t) for t in tokens)
            key = chain_key(self.fp, tokens)
            self.stats.puts += 1
            if key in self._ram or key in self._disk:
                self.stats.deduped_puts += 1
                if plan is not None and self._plans.get(key) != _checked_plan(plan, len(tokens)):
                    self.stats.plan_conflicts += 1
                self._touch(key)
                return self._ram.get(key) or self._load_entry(key)[0]
            parent_entry = self._as_entry(parent, tokens) if parent is not None or auto_parent else None
            entry = capture(cache_list, tokens, self.fp, parent_entry, language, plan)
            self._plans[key] = entry.plan
            entry._tokens = tokens
            self._remember_tokens(key, tokens)
            self._lengths.add(entry.n_tokens)
            self._admit(entry)
            if self.directory is not None and self.write_through:
                self._persist(entry)
            return entry

    def _as_entry(self, parent, tokens):
        if parent is None:
            # Auto-extend the longest stored prefix of these tokens, if any.
            best = self._longest_prefix(tokens, strictly_shorter=True)
            if best is None:
                return None
            return self._chain(best[0])[-1]
        if isinstance(parent, Hit):
            return self._chain(parent.chain[-1])[-1]
        return parent

    def _admit(self, entry: Entry) -> None:
        self._ram[entry.key] = entry
        self._ram.move_to_end(entry.key)
        self._ram_bytes += entry.nbytes()
        self._evict_ram()

    def _ram_order(self) -> list:
        """RAM keys, first = best to drop. Least recently used here; a subclass may rank by value."""
        return list(self._ram)

    def _evict_ram(self) -> None:
        # Restoring a child needs its whole chain in RAM or on disk, so without an SSD tier a parent stays while a
        # child is resident; with one, a parent is written out first and may leave RAM. A lease protects the
        # snapshot from loss, but its RAM copy can be released when its complete chain is durable on SSD.
        spill = self.directory is not None
        newest = next(reversed(self._ram), None)
        while self._ram_bytes > self.ram_budget and len(self._ram) > 1:
            victim = next((k for k in self._ram_order()
                           if k != newest and (k not in self._leases or
                                               (k in self._disk and self._ancestors_present(k)))
                           and (spill or not self._has_resident_child(k))),
                          None)
            if victim is None:
                break
            entry = self._ram[victim]
            if self.directory is not None and victim not in self._disk:
                self._persist(entry)
            if victim not in self._disk:
                # A failed spill is not durable, even for a leaf. Keep it and try
                # other candidates once; never silently discard the only copy.
                candidates = [k for k in self._ram_order()
                              if k != newest and k != victim and (k not in self._leases or
                                                                  (k in self._disk and self._ancestors_present(k)))]
                victim = None
                for key in candidates:
                    candidate = self._ram.get(key)
                    if candidate is None:
                        continue
                    if self.directory is not None and key not in self._disk:
                        self._persist(candidate)
                    if key in self._disk:
                        victim, entry = key, candidate
                        break
                if victim is None:
                    if self.manager is not None:
                        self.invalidate_unbacked_under_pressure(self.ram_budget, exclude=(newest,))
                    break
            if self._ram.pop(victim, None) is not None:
                self._ram_bytes -= entry.nbytes()
            self.stats.evicted_ram += 1

    # ---- leases ----
    def lease(self, keys):
        """Protect entries from loss while in use; durable RAM copies may be released and restored from SSD."""
        return _Lease(self, tuple(keys))

    @_managed_disk
    def _acquire(self, keys) -> None:
        with self._lock:
            for key in keys:
                if self._transaction is not None and key not in self._leases:
                    self._lease_fds[key] = self._transaction.lease(self.directory, key)
                self._leases[key] = self._leases.get(key, 0) + 1

    def _release(self, keys) -> None:
        with self._lock:
            for key in keys:
                left = self._leases.get(key, 0) - 1
                if left > 0:
                    self._leases[key] = left
                else:
                    self._leases.pop(key, None)
                    fd = self._lease_fds.pop(key, None)
                    if fd is not None:
                        os.close(fd)

    @_managed_disk
    def shrink_ram(self, target_bytes: int) -> int:
        """Drop least-recently-used RAM entries until at most ``target_bytes`` stay. Returns the bytes freed.

        Only with an SSD tier: an entry is written out first when it is not there yet, so nothing is lost and the
        next lookup restores it from SSD (bit-identical, as for any SSD hit). Without a directory nothing is
        dropped, because that would discard a snapshot.
        """
        if self.directory is None:
            return 0
        with self._lock:
            before = self._ram_bytes
            for key in self._ram_order():
                if self._ram_bytes <= max(0, int(target_bytes)):
                    break
                if key in self._leases:
                    continue
                entry = self._ram[key]
                if key not in self._disk:
                    self._persist(entry)
                if key not in self._disk:
                    # Could not persist to SSD (e.g. disk reserve limit or write error).
                    # Do not discard snapshot or break chain for resident children.
                    continue
                self._ram.pop(key)
                self._ram_bytes -= entry.nbytes()
                self.stats.evicted_ram += 1
            return before - self._ram_bytes

    @_managed_disk
    def invalidate_unbacked_under_pressure(self, target_bytes: int = 0, *, exclude=()) -> dict:
        """Explicitly discard unleased, undurable leaf snapshots under pressure.

        Persisted copies and their ancestry are never removed here. A discarded
        snapshot becomes a cache miss and must be recomputed; a lease always wins.
        """
        with self._lock:
            freed = count = 0
            target = max(0, int(target_bytes))
            while self._ram_bytes > target:
                parents = {entry.parent for entry in self._ram.values()}
                parents.update(meta["parent"] for meta in self._disk.values())
                key = next((k for k in self._ram_order()
                            if k not in self._disk and k not in self._leases and k not in parents and k not in exclude), None)
                if key is None:
                    break
                entry = self._ram.pop(key)
                size = entry.nbytes()
                self._ram_bytes -= size
                self._forget_tokens(key)
                self._plans.pop(key, None)
                self._evicted_keys[key] = time.time()
                if len(self._evicted_keys) > 2048:
                    self._evicted_keys.popitem(last=False)
                count += 1
                freed += size
            if count:
                self.stats.invalidated += count
                self.stats.pressure_invalidated += count
                self.stats.pressure_invalidated_bytes += freed
                self._lengths = {m["n"] for m in self._disk.values()} | {e.n_tokens for e in self._ram.values()}
            return {"invalidated_entries": count, "invalidated_bytes": freed,
                    "remaining_ram_bytes": self._ram_bytes, "recompute_required": bool(count)}

    def _has_resident_child(self, key) -> bool:
        return any(e.parent == key for e in self._ram.values())

    def _persist(self, entry: Entry) -> None:
        if entry.key in self._disk:
            return
        # A delta on SSD is durable only if its complete parent chain is also
        # durable. In particular, a failed parent spill must not let a child
        # masquerade as a recoverable disk copy.
        if entry.parent is not None and entry.parent not in self._disk:
            parent = self._ram.get(entry.parent)
            if parent is None:
                self.stats.writes_skipped += 1
                return
            self._persist(parent)
            if entry.parent not in self._disk:
                self.stats.writes_skipped += 1
                return
        tx = self._transaction
        if tx is not None:
            ancestors = self._chain_keys(entry.parent) if entry.parent else ()
            protect = [f"flashnext/{self.fp}/{key}" for key in ancestors]
            if not tx.make_room(entry.nbytes() + 1024 * 1024, protect=protect):
                self.stats.writes_skipped += 1
                return
            self._sync_managed_disk()
        elif self.directory is not None:
            try:
                usage = shutil.disk_usage(self.directory)
                if usage.free < self.min_disk_free_bytes:
                    self.stats.writes_skipped += 1
                    return
            except OSError:
                pass
        try:
            written = write_entry(self.directory, entry, self.fp)
            if tx is not None:
                tx.entries = tx._scan()
                if tx.used > tx.quota or shutil.disk_usage(self.directory).free < tx.manager.reserve:
                    self._drop_files(entry.key)
                    tx.entries = tx._scan()
                    self.stats.writes_skipped += 1
                    return
            self._disk[entry.key] = {"n": entry.n_tokens, "parent": entry.parent, "bytes": written,
                                     "atime": time.time()}
            self.stats.ssd_writes += 1
            self.stats.ssd_bytes_written += written
            # Keep the ancestor until quota enforcement has seen the child.
            if entry.parent is not None:
                self._acquire((entry.parent,))
            try:
                if tx is None:
                    self._evict_disk()
            finally:
                if entry.parent is not None:
                    self._release((entry.parent,))
        except OSError:
            self.stats.writes_skipped += 1

    def _disk_total(self) -> int:
        return sum(m["bytes"] for m in self._disk.values()) + sum(m["bytes"] for m in self._foreign.values())

    def _foreign_victim(self):
        """The foreign SSD entry to drop next, or None. Only a leaf (nothing chains to it); oldest first."""
        parents = {m["parent"] for m in self._foreign.values()}
        leaves = [k for k in self._foreign if k not in parents]
        return min(leaves, key=lambda k: self._foreign[k]["atime"]) if leaves else None

    def _disk_victim(self):
        """The SSD entry to drop next, or None. Only a leaf (nothing chains to it) that is not leased; oldest first here."""
        parents = {m["parent"] for m in self._disk.values()}
        leaves = [k for k in self._disk if k not in parents and k not in self._leases]
        return min(leaves, key=lambda k: self._disk[k]["atime"]) if leaves else None

    def _drop_disk_entry(self, victim, reason: str = "budget") -> int:
        meta = self._disk.pop(victim)
        self._drop_files(victim)
        resident = self._ram.pop(victim, None)
        if resident is not None:
            self._ram_bytes -= resident.nbytes()
        self._forget_tokens(victim)
        self._plans.pop(victim, None)
        self._evicted_keys[victim] = time.time()
        if len(self._evicted_keys) > 2048:
            self._evicted_keys.popitem(last=False)
        self.stats.evicted_ssd += 1
        self.stats.evicted_ssd_bytes += meta["bytes"]
        return meta["bytes"]

    def _evict_disk(self) -> None:
        if not self.disk_budget:
            return
        while self._disk_total() > self.disk_budget and (self._foreign or len(self._disk) > 1):
            victim = self._foreign_victim()
            if victim is not None:
                meta = self._foreign.pop(victim)
                self._drop_files(victim)
                self.stats.evicted_ssd += 1
                self.stats.evicted_ssd_bytes += meta["bytes"]
                continue
            victim = self._disk_victim()
            if victim is None:
                break
            self._drop_disk_entry(victim)
        self._lengths = {m["n"] for m in self._disk.values()} | {e.n_tokens for e in self._ram.values()}

    def _touch(self, key) -> None:
        if key in self._ram:
            self._ram.move_to_end(key)
        if key in self._disk:
            self._disk[key]["atime"] = time.time()

    # ---- lookup ----
    def _remember_tokens(self, key, tokens):
        self._tokens[key] = tokens
        if key not in self._rolling_keys:
            digest = hashlib.sha256(_token_bytes(tokens)).digest()
            self._rolling_keys[key] = digest
            self._rolling[digest] = key

    def _forget_tokens(self, key):
        self._tokens.pop(key, None)
        digest = self._rolling_keys.pop(key, None)
        if self._rolling.get(digest) == key:
            del self._rolling[digest]

    def _longest_prefix_with_reason(self, tokens, strictly_shorter=False, accept=None):
        limit = len(tokens) - (1 if strictly_shorter else 0)
        saw_plan_mismatch = False
        saw_evicted = False
        candidates = sorted((n for n in self._lengths if n <= limit), reverse=True)
        if not candidates:
            return None, "cold"
        packed = memoryview(_token_bytes(tokens[:candidates[0]]))
        rolling_matches, unindexed = {}, None
        for index, n in enumerate(candidates):
            # Common longest-prefix hits need only one legacy hash. Use the
            # rolling index if we have to search deeper through the chain.
            if index == 2 and self._rolling and len(candidates) >= 6:
                digest = hashlib.sha256()
                previous = 0
                for length in reversed(candidates[index:]):
                    digest.update(packed[4 * previous:4 * length])
                    previous = length
                    match = self._rolling.get(digest.digest())
                    if match is not None and len(self._tokens[match]) == length:
                        rolling_matches[length] = match
                unindexed = {meta["n"] for key, meta in self._disk.items() if key not in self._rolling_keys}
                unindexed.update(entry.n_tokens for key, entry in self._ram.items() if key not in self._rolling_keys)
            key = rolling_matches.get(n)
            if key is None:
                if unindexed is not None and n not in unindexed and not self._evicted_keys:
                    continue
                key = _chain_key_bytes(self.fp, n, packed[:4 * n])
            if key in self._ram or key in self._disk:
                stored = self._tokens.get(key)
                if stored is None or stored == tuple(tokens[:n]):
                    if accept is not None:
                        plan = self._plans.get(key)
                        if plan is None and key in self._ram:
                            plan = self._ram[key].plan
                        if not accept(n, plan):
                            self.stats.rejected_plan += 1
                            saw_plan_mismatch = True
                            continue
                    return (key, n), "hit"
            elif key in self._evicted_keys:
                saw_evicted = True
        if saw_plan_mismatch:
            return None, "plan_mismatch"
        if saw_evicted:
            return None, "evicted"
        return None, "cold"

    def _longest_prefix(self, tokens, strictly_shorter=False, accept=None):
        """Longest stored prefix of ``tokens``. ``accept(n, plan)`` may veto a candidate (its plan
        provenance does not fit the caller's continuation); a vetoed entry stays stored for other plans."""
        found, _ = self._longest_prefix_with_reason(tokens, strictly_shorter=strictly_shorter, accept=accept)
        return found

    @_managed_disk
    def peek(self, ids, accept=None) -> int:
        """How many leading tokens of ``ids`` a lookup would reuse, without loading, restoring or counting anything.

        An entry that later fails to load is invalidated by ``lookup`` and the prompt is prefilled whole, so this is
        an estimate for planning memory, never a promise.
        """
        with self._lock:
            ids = tuple(int(t) for t in ids)
            vetoed = self.stats.rejected_plan
            found = self._longest_prefix(ids, strictly_shorter=True, accept=accept)
            self.stats.rejected_plan = vetoed
            return 0 if found is None else found[1]

    @_managed_disk
    def lookup(self, ids, make_cache, language=None, accept=None) -> tuple:
        """Return ``LookupResult(hit, caches, outcome)``. ``hit`` is None on a miss (then ``caches`` is None).

        ``accept(n_tokens, plan)`` vetoes candidates whose chunk-plan provenance does not fit (see ``put``).

        The restored prefix is always strictly shorter than ``ids``: at least one token must still be
        prefilled to produce the first output token.
        """
        started = time.perf_counter()
        with self._lock:
            ids = tuple(int(t) for t in ids)
            self.stats.lookups += 1
            saw_invalid = False
            saw_restore_fail = False
            while True:
                found, reason = self._longest_prefix_with_reason(ids, strictly_shorter=True, accept=accept)
                if found is None:
                    self.stats.missed_tokens += len(ids)
                    if saw_invalid:
                        outcome = OUTCOME_STATE_INVALID
                        self.stats.state_invalids += 1
                    elif saw_restore_fail:
                        outcome = OUTCOME_RESTORE_FAILED
                        self.stats.restore_failures += 1
                    elif reason == "plan_mismatch":
                        outcome = OUTCOME_PREFIX_MISMATCH
                        self.stats.prefix_mismatches += 1
                    elif reason == "evicted":
                        outcome = OUTCOME_EVICTED
                        self.stats.evicted_hits += 1
                    else:
                        outcome = OUTCOME_COLD
                        self.stats.cold_lookups += 1
                    if self._transaction is not None:
                        self._transaction.record("flashnext", self.directory)
                    return LookupResult(None, None, outcome)
                key, n = found
                holding = self._chain_keys(key)
                self._acquire(holding)                  # nothing in this chain may be evicted while it is restored
                try:
                    try:
                        chain, loaded = self._chain_loading(key, for_restore=True)
                    except CorruptEntry:
                        self._release(holding)
                        holding = ()
                        self._invalidate(key)
                        saw_invalid = True
                        continue
                    try:
                        caches = restore(chain, make_cache)
                    except (CorruptEntry, UnsupportedCache):
                        self._release(holding)
                        holding = ()
                        self._invalidate(key)
                        saw_restore_fail = True
                        continue
                finally:
                    if holding:
                        self._release(holding)
                if language is not None:
                    apply_language(language, chain[-1].lang)
                tier = "ssd" if loaded else "ram"
                outcome = OUTCOME_SSD_HIT if loaded else OUTCOME_RAM_HIT
                self._touch(key)
                seconds = time.perf_counter() - started
                self.stats.hits += 1
                self.stats.reused_tokens += n
                self.stats.missed_tokens += len(ids) - n
                self.stats.restore_seconds += seconds
                self.stats.ram_hits += tier == "ram"
                self.stats.ssd_hits += tier == "ssd"
                self.stats.ssd_bytes_read += loaded
                self._on_hit(key, n, tier, seconds)
                if self._transaction is not None:
                    self._transaction.record("flashnext", self.directory, key, n)
                hit = Hit(n, tier, seconds, loaded, tuple(e.key for e in chain), outcome=outcome)
                return LookupResult(hit, caches, outcome)

    def _chain_keys(self, key) -> tuple:
        """Keys of ``key`` and its ancestors (from the in-memory index only; nothing is loaded)."""
        keys, current = [], key
        while current is not None and current not in keys:
            keys.append(current)
            entry = self._ram.get(current)
            current = entry.parent if entry is not None else self._disk.get(current, {}).get("parent")
        return tuple(keys)

    def observe(self, tokens) -> int:
        """Count a request that starts with exactly these tokens; returns earlier sightings. No-op here (see prefix_profiles)."""
        return 0

    def record_prefill(self, tokens: int, seconds: float) -> None:
        """Feed one measured prefill into a recompute-cost estimate. No-op here (see prefix_profiles)."""

    def _on_hit(self, key, n_tokens, tier, seconds) -> None:
        """Called once per successful restore, under the lock. ``prefix_profiles`` records frequency and time saved."""

    def _chain(self, key) -> list:
        return self._chain_loading(key)[0]

    def _chain_loading(self, key, *, for_restore=False):
        chain, loaded, current = [], 0, key
        while current is not None:
            entry = self._ram.get(current)
            if entry is None:
                entry, size = self._load_entry(current)
                loaded += size
            if chain and for_restore:
                # Ancestors contribute only history/index deltas. Holding their
                # recurrent states and recent tiers throughout a long restore
                # can dwarf the compressed live state.
                slim = copy.copy(entry)
                slim.layers = [
                    Layer("arrays") if layer.kind == "arrays" else
                    Layer(layer.kind, kv={k: a for k, a in layer.kv.items() if not k.startswith("recent_")})
                    if layer.kind == "qsa_mixed_v1" else layer
                    for layer in entry.layers
                ]
                slim.lang = {}
                chain.append(slim)
            else:
                chain.append(entry)
            current = entry.parent
        chain.reverse()
        return chain, loaded

    def _load_entry(self, key):
        if self.directory is None or key not in self._disk:
            raise CorruptEntry("entry is not in RAM or on disk")
        # Disk files can be replaced or modified by another process at any time.
        # A previous verification of this key does not verify the current bytes.
        entry = read_entry(self.directory, key, self.fp, self.verify_hash, managed=self._transaction is not None)
        if entry.parent is not None:
            parent_tokens = self._tokens.get(entry.parent)
            if parent_tokens is None:
                parent_tokens = self._rebuild_tokens(entry.parent)
            entry._tokens = parent_tokens + entry.token_delta
        else:
            entry._tokens = entry.token_delta
        if chain_key(self.fp, entry._tokens) != key:
            raise CorruptEntry("token chain does not hash to the key")
        self._remember_tokens(key, entry._tokens)
        self._plans[key] = entry.plan
        self._admit(entry)
        return entry, self._disk[key]["bytes"]

    def _rebuild_tokens(self, key) -> tuple:
        deltas, current = [], key
        while current is not None:
            if current in self._tokens:
                deltas.append(self._tokens[current])
                break
            entry = self._ram.get(current) or read_entry(self.directory, current, self.fp, self.verify_hash, managed=self._transaction is not None)
            deltas.append(entry.token_delta)
            current = entry.parent
        return tuple(t for delta in reversed(deltas) for t in delta)

    def _invalidate(self, key) -> None:
        """Remove an entry and every descendant (they cannot be restored without it)."""
        doomed = {key}
        changed = True
        known = {**{k: m["parent"] for k, m in self._disk.items()},
                 **{k: e.parent for k, e in self._ram.items()}}
        while changed:
            changed = False
            for child, parent in known.items():
                if parent in doomed and child not in doomed:
                    doomed.add(child)
                    changed = True
        for k in doomed:
            entry = self._ram.pop(k, None)
            if entry is not None:
                self._ram_bytes -= entry.nbytes()
            self._disk.pop(k, None)
            self._forget_tokens(k)
            self._drop_files(k)
            self.stats.invalidated += 1
        self._lengths = {m["n"] for m in self._disk.values()} | {e.n_tokens for e in self._ram.values()}

    # ---- reporting ----
    @_managed_disk
    def usage(self) -> dict:
        with self._lock:
            return {
                "ram_entries": len(self._ram), "ram_bytes": self._ram_bytes, "ram_budget": self.ram_budget,
                "disk_entries": len(self._disk), "disk_bytes": sum(m["bytes"] for m in self._disk.values()),
                "disk_budget": self.disk_budget,
            }

    @_managed_disk
    def clear(self) -> None:
        with self._lock:
            for key in list(self._disk):
                self._drop_files(key)
            self._ram.clear()
            self._disk.clear()
            self._tokens.clear()
            self._rolling.clear()
            self._rolling_keys.clear()
            self._lengths.clear()
            self._ram_bytes = 0

    def inspect(self) -> dict:
        """Safe inspection of cache contents without exposing raw prompt texts or internal reasoning."""
        with self._lock:
            disk_bytes = sum(m["bytes"] for m in self._disk.values())
            try:
                free_bytes = shutil.disk_usage(self.directory).free if self.directory else None
            except OSError:
                free_bytes = None
            return {
                "fingerprint": self.fp,
                "ram_entries": len(self._ram),
                "ram_bytes": self._ram_bytes,
                "ram_budget_bytes": self.ram_budget,
                "ssd_entries": len(self._disk),
                "ssd_bytes": disk_bytes,
                "ssd_budget_bytes": self.disk_budget,
                "ssd_directory": str(self.directory) if self.directory else None,
                "disk_free_bytes": free_bytes,
                "min_disk_free_bytes": self.min_disk_free_bytes,
                "leased_entries": len(self._leases),
                "lengths": sorted(self._lengths),
                "stats": {
                    "lookups": self.stats.lookups,
                    "hits": self.stats.hits,
                    "ram_hits": self.stats.ram_hits,
                    "ssd_hits": self.stats.ssd_hits,
                    "prefix_mismatches": self.stats.prefix_mismatches,
                    "state_invalids": self.stats.state_invalids,
                    "evicted_hits": self.stats.evicted_hits,
                    "restore_failures": self.stats.restore_failures,
                    "cold_lookups": self.stats.cold_lookups,
                    "reused_tokens": self.stats.reused_tokens,
                    "ssd_writes": self.stats.ssd_writes,
                    "ssd_bytes_written": self.stats.ssd_bytes_written,
                    "ssd_bytes_read": self.stats.ssd_bytes_read,
                    "writes_skipped": self.stats.writes_skipped,
                    "pressure_invalidated": self.stats.pressure_invalidated,
                    "pressure_invalidated_bytes": self.stats.pressure_invalidated_bytes,
                    "evicted_ram": self.stats.evicted_ram,
                    "evicted_ssd": self.stats.evicted_ssd,
                },
            }

    def cleanup(self, max_age_seconds: float | None = None, target_disk_bytes: int | None = None) -> dict:
        """Safe LRU / age-based reclamation of unleased leaf snapshots."""
        with self._lock:
            freed_bytes = 0
            removed_entries = 0
            now = time.time()
            if max_age_seconds is not None:
                parents = {m["parent"] for m in self._disk.values()}
                for key in list(self._disk):
                    if key not in parents and key not in self._leases:
                        if now - self._disk[key]["atime"] > max_age_seconds:
                            freed_bytes += self._drop_disk_entry(key, reason="ttl")
                            removed_entries += 1
            if target_disk_bytes is not None:
                while sum(m["bytes"] for m in self._disk.values()) > max(0, target_disk_bytes) and len(self._disk) > 1:
                    victim = self._disk_victim()
                    if victim is None:
                        break
                    freed_bytes += self._drop_disk_entry(victim, reason="cleanup_target")
                    removed_entries += 1
            self._lengths = {m["n"] for m in self._disk.values()} | {e.n_tokens for e in self._ram.values()}
            return {"removed_entries": removed_entries, "freed_bytes": freed_bytes}
