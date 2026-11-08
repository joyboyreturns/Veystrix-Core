"""Per-request decode-pool release (opt-in: FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=1, off by default).

What the 2K traces show (results/flashnext/20261108-003402-prefill-memory-manual, request A then B in one process):

* After request A's decode, MLX allocator cache is only 0.08 GB. The 1.6 to 2.3 GB of "cache" seen in request B's
  prefill is the per-chunk working set that request A's prefill had too, not memory kept across requests.
* What is kept is the 2 GB decode expert pool (MLX active 2.98 GB before the pool is created, 5.13 GB after). It lives on
  ``store._flashnext_expert_pool`` for the life of the process.
* While that pool exists, ``prefill_ring.acquire`` takes its "later turn" branch: no scratch pool, the ring lives in the
  decode pool. Request B's prefill then ran at a 6.97 GB MLX peak (request A: 5.82 GB) with the 2 GB pool resident, the
  compressor grew by 6 GiB and the memory guard stopped it.

The smallest correct change is to give request B the path request A took: when a fresh conversation is about to run a
ring-eligible prefill and a decode pool exists, drop the pool first. The ring then takes the validated first-turn path
(scratch pool, disposed after the prefill), and decode creates a new pool lazily at its first token, exactly as for A.
``expert_pool.for_store`` already creates the pool on demand, so nothing else holds it.

Between turns of one conversation the pool is kept (its warm slots are worth something, and the next prefill is short).
Prefix-cache snapshots live in ``PrefixCache`` (RAM and SSD), not here, and are untouched.

Order of operations (same as ``prefill_ring.release``): refuse while a prefill ring is leased; fence the GPU; detach the
pool from the store; drop every reference (file descriptors, MLX view, memoryview); collect; clear the allocator cache.
Memory goes back when the last MLX array on the buffer dies, so an array still viewing the pool keeps it valid.
"""
from __future__ import annotations

import contextlib
import gc
import os
import threading
import time

FLAG = "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL"

_LOCK = threading.Lock()
_IN_FLIGHT = [0]                     # generate() calls running in this process (counted only while the flag is on)
_REQUEST_LOCK = threading.Lock()
_REQUEST_LOCAL = threading.local()
_LIFECYCLE_FAULT = [None]


def mark_lifecycle_fault(reason: str) -> None:
    """Poison this process after an unsafe cleanup; shared pools span backends."""
    _LIFECYCLE_FAULT[0] = str(reason)


def lifecycle_fault() -> str | None:
    return _LIFECYCLE_FAULT[0]


def _gpu_synchronize():
    """The GIL-safe barrier (never ``mx.synchronize``; see gpu_sync). Imported late: gpu_sync imports this module."""
    from .gpu_sync import synchronize

    return synchronize


def synchronize_or_fault(synchronize) -> None:
    if synchronize is None:
        return
    try:
        synchronize()
    except BaseException as error:
        mark_lifecycle_fault(f"GPU synchronization failed: {type(error).__name__}: {error}")
        raise


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


@contextlib.contextmanager
def exclusive_request():
    """Serialize requests while an opt-in policy can change shared GPU ownership.

    Keep the lock through cleanup so another request cannot start between the
    idle check and resource release. Nested generation would deadlock and is
    rejected before touching inference state.
    """
    active = (enabled() or os.environ.get("FLASHNEXT_MEMORY_MANAGER", "0") == "1"
              or os.environ.get("FLASHNEXT_AUTO_MEMORY_RECLAIM", "0") == "1")
    if not active:
        yield
        return
    if getattr(_REQUEST_LOCAL, "active", False):
        raise RuntimeError("nested generation is unsafe while memory lifecycle management is enabled")
    with _REQUEST_LOCK:
        _REQUEST_LOCAL.active = True
        try:
            yield
        finally:
            _REQUEST_LOCAL.active = False


def should_release(tokens: int, *, has_pool: bool, ring_experts: int, ring_min_tokens: int,
                   scratch: bool) -> bool:
    """Release only when the flag is on, a pool exists, and the prefill will use the ring in a scratch pool if it can.

    A prompt below the ring minimum never uses the ring, so dropping a warm pool would cost decode speed for nothing.
    Without the scratch option the ring is designed to live in the decode pool, so that configuration keeps it.
    """
    return bool(enabled() and has_pool and ring_experts > 0 and tokens >= ring_min_tokens and scratch)


@contextlib.contextmanager
def in_flight():
    """Count this generate() call while the flag is on, so a concurrent request can see it is not alone.

    Off (the default) it does nothing. The decode pool is shared by every request in the process; ``make_room``
    never drops it while another request is between its prefill and the end of its decode.
    """
    if not (enabled() or os.environ.get("FLASHNEXT_MEMORY_MANAGER", "0") == "1"
            or os.environ.get("FLASHNEXT_AUTO_MEMORY_RECLAIM", "0") == "1"):
        yield
        return
    with _LOCK:
        _IN_FLIGHT[0] += 1
    try:
        yield
    finally:
        with _LOCK:
            _IN_FLIGHT[0] -= 1


def others_in_flight() -> int:
    with _LOCK:
        return max(0, _IN_FLIGHT[0] - 1)


def pool_bytes(pool) -> int:
    mm = getattr(pool, "_mm", None)
    try:
        return len(mm) if mm is not None else 0
    except (ValueError, TypeError):
        return 0


def release_decode_pool(store, *, synchronize=None, clear_cache=None, ring_active=None, others_active=None) -> int:
    """Drop the store's decode pool. Returns the bytes released, 0 when there is nothing to do or it is unsafe.

    ``synchronize`` is called first (gpu_sync.synchronize in production, never mx.synchronize); if it raises, the pool is kept. ``ring_active``
    returns True while a prefill ring is leased (the ring may read the pool) and ``others_active`` returns True while
    another request is in flight; in either case nothing is released.
    """
    pool = getattr(store, "_flashnext_expert_pool", None)
    if pool is None:
        return 0
    if (others_in_flight if others_active is None else others_active)():
        return 0
    if ring_active is None:
        from . import expert_cache

        ring_active = lambda: expert_cache._PREFILL_RING[0] is not None
    if ring_active():
        return 0
    synchronize_or_fault(synchronize)       # failure leaves the pool attached and poisons the process
    size = pool_bytes(pool)
    store._flashnext_expert_pool = None
    from .prefill_ring import _release_pool

    _release_pool(pool)
    del pool
    gc.collect()
    if clear_cache is not None:
        clear_cache()
    return size


def store_of(language):
    """The expert store behind the first streamed layer (the same lookup ``prefill_ring.acquire`` uses)."""
    from . import expert_cache

    body = getattr(language, "model", language)
    for layer in getattr(body, "layers", ()):
        glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if isinstance(glu, expert_cache.StreamingSwitchGLU):
            return glu.gate_proj.cache.store
    return None


def make_room(language, tokens: int, now=time.perf_counter) -> dict | None:
    """Before a fresh ring-eligible prefill: drop the decode pool if the policy says so.

    Returns None when the flag is off, else a report: ``released_bytes``, ``reason`` (why it did or did not release),
    ``tokens``, ``seconds`` and, when it released, ``active_before``/``active_after`` (MLX active bytes) and
    ``cache_after``. Never raises: a failure leaves the pool attached and is reported.
    """
    if not enabled():
        return None
    report = {"released_bytes": 0, "reason": None, "tokens": int(tokens), "seconds": 0.0}
    began = now()
    try:
        import mlx.core as mx

        from . import expert_cache, expert_pool, prefill_ring

        store = store_of(language)
        if store is None:
            report["reason"] = "no streamed store"
        elif getattr(store, "_flashnext_expert_pool", None) is None:
            report["reason"] = "no decode pool"
        elif not should_release(
                tokens, has_pool=True,
                ring_experts=(prefill_ring.configured_experts()
                              if prefill_ring.enabled() and expert_pool.enabled() else 0),
                ring_min_tokens=prefill_ring.min_tokens(), scratch=prefill_ring.scratch_enabled()):
            report["reason"] = "prefill is not eligible for the scratch ring"
        elif others_in_flight():
            report["reason"] = "another request is in flight"
        elif expert_cache._PREFILL_RING[0] is not None:
            report["reason"] = "a prefill ring is leased"
        else:
            before = mx.get_active_memory()
            report["released_bytes"] = release_decode_pool(
                store, synchronize=_gpu_synchronize(), clear_cache=mx.clear_cache)
            report["active_before"] = int(before)
            report["active_after"] = int(mx.get_active_memory())
            report["cache_after"] = int(mx.get_cache_memory())
            report["reason"] = "released"
    except Exception as error:
        if lifecycle_fault():
            raise RuntimeError(lifecycle_fault()) from error
        report["reason"] = f"error: {type(error).__name__}: {error}"
    report["seconds"] = now() - began
    return report
