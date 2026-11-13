"""Stage 1 of the in-pool prefill read path (opt-in, off by default).

Today a prefill MoE layer reads its routed experts into freshly allocated NumPy
arrays through the page cache, one buffered ``pread`` and copy per row. Stage 1
changes only where those bytes land and how they are read:

* The destination is one half of a part-major ring leased from the 6 GB expert
  pool (``pool_ring.PoolRing``), so no per-layer read buffer is allocated.
* Rows are read with ``preadv`` on ``F_NOCACHE`` descriptors, straight into the
  ring, so the page cache is not filled.
* The nine part arrays handed to ``mx.gather_qmm`` are zero-copy contiguous
  slices of the pool's Metal-wrapped buffer.

It reads exactly the experts the router selected, in first-seen order, so the
``gather_qmm`` operands and the ``local`` index map are identical to the
current path. There is no prediction and no read concurrent with GPU work: a
layer's reads still start only after the previous layer's compute is done, which
is why one ring half is enough.

Ownership and safety
    ``session`` leases the ring when a prefill starts and returns the slots when
    it ends, on every exit path. The half used by a layer stays COMPUTING until
    the next layer's reads begin (the router sync that precedes them proves the
    previous layer's GPU work finished) or the prefill ends. Any failure,
    capacity miss, busy half or working-set pressure makes ``stage`` return
    None and the caller uses the existing read path for that layer.

Memory
    The ring adds nothing beyond the pool. ``acquire`` refuses unless MLX's
    active memory (which already includes the wrapped pool) plus a transient
    estimate fit under Metal's recommended working set with a margin, and ``stage`` re-checks the live active memory
    every layer and disables the ring for the rest of the prefill if it does
    not.

Everything is off unless ``FLASHNEXT_PREFILL_RING_EXPERTS`` is a positive
number and the expert pool is enabled.

First-turn scratch pool (``FLASHNEXT_PREFILL_RING_SCRATCH=1``, off by default)
    The 6 GB pool is created at the first decode, so a first-turn prefill has
    none, and MLX counts a wrapped pool at its full size. Putting the ring in
    the pool would therefore add the whole pool to the first prefill. With the
    scratch option, a first-turn prefill (no pool yet) uses a temporary pool
    sized for the ring alone (the half plus 16 unused slots, about 1.6 GB for
    512 experts). It is private, never registered with the store, and freed
    when the session ends: the ring is closed, the GPU fenced, every reference
    dropped. The normal pool is then created at first decode as before. When
    the pool already exists (later turns) the ring lives in it as before.

Accounting
    Memory is counted once: MLX's active memory already contains every wrapped
    pool, the allocator cache is a separate counter, and a pool that does not
    exist yet is added once, before it is created. A request that would not fit
    is refused before anything is allocated.
"""
from __future__ import annotations

import contextlib
import gc
import mmap
import os
import time
from collections import namedtuple
from concurrent.futures import wait
from itertools import groupby

from .pool_ring import (
    MIN_DECODE_SLOTS, HazardError, PoolRing, RingError, fits_working_set,
    plan_ring, slots_per_half,
)
from .store import expert_read_batch_rows

FLAG_EXPERTS = "FLASHNEXT_PREFILL_RING_EXPERTS"
FLAG_SCRATCH = "FLASHNEXT_PREFILL_RING_SCRATCH"
FLAG_MIN_TOKENS = "FLASHNEXT_PREFILL_RING_MIN_TOKENS"
FLAG_MARGIN_GB = "FLASHNEXT_PREFILL_RING_MARGIN_GB"
FLAG_TRANSIENT_GB = "FLASHNEXT_PREFILL_RING_TRANSIENT_GB"

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "scales", "biases")
CLOSE_TIMEOUT = 30.0
INT32_MAX = 2**31 - 1

PartPlan = namedtuple("PartPlan", "name dtype tail row_bytes")

# Why the last acquire declined, for diagnostics and tests.
LAST_REASON = [None]

# Sessions whose ring would not close in time. Their temporary pool is kept
# alive on purpose: a writer or a GPU job may still be using it, and freeing
# the memory under them is worse than holding it.
_RETAINED = []


class RingUnavailable(RuntimeError):
    """The checkpoint layout cannot use the ring."""


def _number(name: str, default: str) -> float:
    raw = os.environ.get(name, default) or default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def configured_experts() -> int:
    """Experts per ring half requested by the environment (0 disables)."""
    return max(0, int(_number(FLAG_EXPERTS, "0")))


def enabled() -> bool:
    return configured_experts() > 0


def ring_halves() -> int:
    """Ring halves for this prefill: 2 when layer-major prefetch asks for the overlapped ring, else 1.

    Two halves let the next layer's rows be read while the current layer computes; the scratch pool grows by one half
    (about 1.5 GB at 512 experts). Off unless layer-major and layer-prefetch are both enabled."""
    from . import layer_major

    return 2 if layer_major.enabled() and layer_major.prefetch_enabled() else 1


def scratch_enabled() -> bool:
    """Whether a first-turn prefill may use a temporary right-sized pool."""
    raw = os.environ.get(FLAG_SCRATCH, "0") or "0"
    if raw not in ("0", "1"):
        raise ValueError(f"{FLAG_SCRATCH} must be 0 or 1")
    return raw == "1"


def min_tokens() -> int:
    return max(0, int(_number(FLAG_MIN_TOKENS, "512")))


def margin_bytes() -> int:
    return int(max(0.0, _number(FLAG_MARGIN_GB, "0.5")) * 1e9)


def transient_bytes() -> int:
    return int(max(0.0, _number(FLAG_TRANSIENT_GB, "2.0")) * 1e9)


def plan_parts(store, switch_prefix: str):
    """The nine parts of one layer's experts, in ring order."""
    plans = []
    for projection in PROJECTIONS:
        for kind in KINDS:
            ref = store.refs.get(f"{switch_prefix}.{projection}.{kind}")
            if ref is None:
                raise RingUnavailable(f"missing tensor {projection}.{kind}")
            if ref.dtype not in ("U32", "BF16"):
                raise RingUnavailable(f"unsupported dtype {ref.dtype}")
            plans.append(PartPlan(
                f"{projection}.{kind}", ref.dtype, tuple(ref.shape[1:]), ref.row_bytes,
            ))
    return tuple(plans)


def fence() -> None:
    """Wait until every GPU job queued before this call has finished (GIL released; see ``gpu_sync``).

    ``mx.synchronize`` is never used: it keeps the GIL while it waits, and MLX's completion handler can need the
    GIL to free a numpy-backed operand, which deadlocks (seen in the first full-checkpoint AGENTIC run).
    """
    from .gpu_sync import synchronize

    synchronize()


def working_set_bytes(active: int, pool_bytes: int) -> int:
    """Bytes MLX has resident, counting the wrapped expert pool exactly once.

    The pool is wrapped zero-copy as an MLX array (``ExpertPool.buffer_mx``),
    and MLX counts it in ``get_active_memory`` and ``get_peak_memory``:
    creating a 6 GB pool raised active memory by exactly 6.000 GB. Adding the
    pool again double counts it. If active memory ever reads lower than the
    pool, something is off, so the pool is added to stay on the safe side.
    """
    active, pool_bytes = int(active), int(pool_bytes)
    return active if active >= pool_bytes else active + pool_bytes


def pool_length(size_bytes: int, stride: int) -> int:
    """Bytes ``ExpertPool(store, size_bytes)`` maps: header plus whole records,
    rounded up to a page. Mirrors ``expert_pool.ExpertPool.__init__``."""
    from . import expert_pool

    slots = (int(size_bytes) - expert_pool.HEADER) // int(stride)
    length = expert_pool.HEADER + slots * int(stride)
    return -(-length // mmap.PAGESIZE) * mmap.PAGESIZE


def _release_pool(pool) -> None:
    """Drop every reference this process holds to a temporary pool.

    Call only when the GPU is idle and the ring is closed. The memory goes back
    when the last MLX array on the buffer dies; arrays that still view it keep
    it valid until then, so nothing can read freed memory.
    """
    try:
        pool.close()                    # the file descriptors
    except Exception:
        pass
    pool.buffer_mx = None
    pool.array = None
    view, pool._view = getattr(pool, "_view", None), None
    if view is not None:
        try:
            view.release()
        except (BufferError, ValueError):
            pass                        # a leftover slice: garbage collection frees it


def metal_working_set_limit():
    """Metal's recommended working set in bytes, or None when unknown."""
    import mlx.core as mx

    reader = getattr(mx, "device_info", None)
    if reader is None:
        reader = getattr(getattr(mx, "metal", None), "device_info", None)
    if reader is None:
        return None
    try:
        return int(reader().get("max_recommended_working_set_size", 0)) or None
    except Exception:
        return None


def _read_rows(lease, view, fd, base, file_start, row_bytes, rows) -> None:
    """Read rows of one part into their ring slots. Runs on a worker thread."""
    with lease.writing():
        for _, adjacent in groupby(enumerate(rows), key=lambda entry: entry[1][1] - entry[0]):
            run = [row for _, row in adjacent]
            dest = [view[base + slot * row_bytes:base + (slot + 1) * row_bytes] for slot, _ in run]
            try:
                got = os.preadv(fd, dest, file_start + run[0][1] * row_bytes)
            finally:
                for part in dest:
                    part.release()
            expected = len(run) * row_bytes
            if got != expected:
                raise OSError(f"short expert read: {got} of {expected} bytes")


class Staging:
    """A begun, not yet consumed staging: the leased half and the queued reads."""

    __slots__ = ("lease", "futures", "count")

    def __init__(self, lease, futures, count):
        self.lease = lease
        self.futures = futures
        self.count = count


class PrefillRing:
    """Reads one layer's routed experts into the ring and hands back views."""

    def __init__(
        self, pool, ring, plans, submit, resolve, keepwarm=None,
        active_memory=None, limit_bytes=None, margin=0, pool_bytes=0,
        synchronize=None, cache_memory=None, owns_pool=False,
    ):
        self.pool = pool
        self.ring = ring
        self.plans = tuple(plans)
        self._submit = submit
        self._resolve = resolve
        self._keepwarm = keepwarm
        self._active_memory = active_memory
        self._limit = limit_bytes
        self._margin = int(margin)
        self._pool_bytes = int(pool_bytes)
        self._synchronize = synchronize
        self._cache_memory = cache_memory
        self.owns_pool = bool(owns_pool)    # a temporary pool this session frees
        self.disposed = False
        self._pool_view = pool._view
        self._buf32 = None
        self.pending = None
        self._seeded_layers = set()
        self.disabled = None
        self.last_error = None
        self._stats = {
            "staged_layers": 0, "staged_experts": 0, "staged_bytes": 0,
            "wait_seconds": 0.0, "fallback_capacity": 0, "fallback_busy": 0,
            "fallback_error": 0, "fallback_disabled": 0, "fallback_memory": 0,
        }

    def stats(self) -> dict:
        snapshot = dict(self._stats)
        snapshot["disabled"] = self.disabled
        snapshot["half_experts"] = self.ring.half_experts
        snapshot["ring_bytes"] = self.ring.ring_bytes
        snapshot["last_error"] = self.last_error
        snapshot["scratch_pool"] = self.owns_pool
        snapshot["disposed"] = self.disposed
        return snapshot

    # -- one layer ----------------------------------------------------------

    def stage(self, store, switch_prefix: str, wanted):
        """Zero-copy ``[(weight, scales, biases)] x 3`` for ``wanted``, or None.

        None always means "use the existing read path for this layer".
        """
        self._retire()
        if self.disabled:
            self._stats["fallback_disabled"] += 1
            return None
        count = len(wanted)
        if count == 0 or count > self.ring.half_experts:
            self._stats["fallback_capacity"] += 1
            return None
        if self._over_limit():
            self.disabled = "working set"
            self._stats["fallback_memory"] += 1
            return None
        lease = self.ring.begin_fill()
        if lease is None:
            self._stats["fallback_busy"] += 1
            return None
        futures = []
        began = time.perf_counter()
        try:
            futures = self._queue_reads(lease, store, switch_prefix, wanted)
            if self._keepwarm is not None:
                self._keepwarm(futures)
            for future in futures:
                self._resolve(future, None)
            lease.mark_filled(timeout=CLOSE_TIMEOUT)
            lease.begin_compute()
            weights = self._views(lease, count)
        except BaseException as error:
            if futures:
                wait(futures)
            lease.release()
            if isinstance(error, Exception):
                self._stats["fallback_error"] += 1
                self.last_error = repr(error)
                return None
            raise
        self._stats["wait_seconds"] += time.perf_counter() - began
        self._stats["staged_layers"] += 1
        self._stats["staged_experts"] += count
        self._stats["staged_bytes"] += count * self.ring.expert_bytes
        self.pending = lease
        return weights

    # -- overlapped staging (two-half ring, layer-major prefetch) -----------

    @property
    def overlap_capable(self) -> bool:
        return self.ring.halves >= 2 and not self.disabled

    def stage_begin(self, store, switch_prefix: str, wanted):
        """Lease a FREE half and queue its reads without waiting. Returns a ``Staging`` or None.

        None means "no overlap for this layer": the caller stages it with ``stage`` when its turn comes. The half is in the
        FILLING state, so nothing computes on it; ``stage_finish`` (or ``cancel_stage``) must follow.
        """
        if not self.overlap_capable:
            return None
        count = len(wanted)
        if count == 0 or count > self.ring.half_experts:
            self._stats["fallback_capacity"] += 1
            return None
        if self._over_limit():
            self.disabled = "working set"
            self._stats["fallback_memory"] += 1
            return None
        lease = self.ring.begin_fill()
        if lease is None:
            self._stats["fallback_busy"] += 1
            return None
        futures = []
        try:
            futures = self._queue_reads(lease, store, switch_prefix, wanted)
        except BaseException as error:
            if futures:
                wait(futures)
            lease.release()
            if isinstance(error, Exception):
                self._stats["fallback_error"] += 1
                self.last_error = repr(error)
                return None
            raise
        return Staging(lease, futures, count)

    def stage_finish(self, staging):
        """Wait for a begun staging and return its zero-copy views, or None (the half is released; use the read path)."""
        self._retire()                          # the previous layer's half: its GPU work has finished
        lease, futures, count = staging.lease, staging.futures, staging.count
        began = time.perf_counter()
        try:
            if self._keepwarm is not None:
                self._keepwarm(futures)
            for future in futures:
                self._resolve(future, None)
            lease.mark_filled(timeout=CLOSE_TIMEOUT)
            lease.begin_compute()
            weights = self._views(lease, count)
        except BaseException as error:
            wait(futures)
            lease.release()
            if isinstance(error, Exception):
                self._stats["fallback_error"] += 1
                self.last_error = repr(error)
                return None
            raise
        self._stats["wait_seconds"] += time.perf_counter() - began
        self._stats["staged_layers"] += 1
        self._stats["staged_experts"] += count
        self._stats["staged_bytes"] += count * self.ring.expert_bytes
        self.pending = lease
        return weights

    def cancel_stage(self, staging) -> None:
        """Abandon a begun staging: no worker may still write when the half is released. Never raises."""
        try:
            wait(staging.futures)
        finally:
            try:
                staging.lease.release()
            except Exception:
                pass

    def _over_limit(self) -> bool:
        if self._limit is None or self._active_memory is None:
            return False
        used = working_set_bytes(self._active_memory(), self._pool_bytes)
        if self._cache_memory is not None:
            used += self._cache_memory()    # freed-but-kept buffers are not "active"
        return used + self._margin > self._limit

    def _queue_reads(self, lease, store, switch_prefix, wanted):
        # Order by shard position so preadv can combine adjacent file rows
        # while writing each expert to its original routing slot.
        rows = sorted(enumerate(wanted), key=lambda item: item[1])
        base_chunk = max(1, int(getattr(store, "_pread_chunk", 2)))
        futures = []
        for plan in self.plans:
            ref = store.refs[f"{switch_prefix}.{plan.name}"]
            fd = self.pool._fd(ref.shard)               # F_NOCACHE, read-ahead off
            base, _ = lease.row_range(plan.name, 0, 1)
            chunk = expert_read_batch_rows(plan.row_bytes, base_chunk)
            for start in range(0, len(rows), chunk):
                futures.append(self._submit(
                    _read_rows, lease, self._pool_view, fd, base, ref.start,
                    plan.row_bytes, rows[start:start + chunk],
                ))
        return futures

    def _views(self, lease, count: int):
        """Nine zero-copy contiguous slices of the pool's Metal buffer.

        MLX slice indices are 32-bit. The pool is wrapped as uint32, whose
        element index stays below 2**31 for pools up to 8 GiB, so every part
        is sliced as uint32 and bf16 parts are reinterpreted afterwards. A
        bf16 element index would pass 2**31 once the ring sits above 4 GiB in
        a 6 GB pool, which an earlier version hit.
        """
        import mlx.core as mx

        if self._buf32 is None:
            self._buf32 = self.pool.buffer_mx
        arrays = {}
        for plan in self.plans:
            offset, _ = lease.row_range(plan.name, 0, 1)
            elements = count
            for dimension in plan.tail:
                elements *= dimension
            nbytes = elements * (4 if plan.dtype == "U32" else 2)
            if offset % 4 or nbytes % 4:
                raise RingUnavailable(f"{plan.name} is not 4-byte aligned in the ring")
            first, words = offset // 4, nbytes // 4
            if first + words > INT32_MAX:
                raise RingUnavailable("ring lies beyond the 32-bit slice range")
            part = self._buf32[first:first + words]
            if plan.dtype != "U32":
                part = part.view(mx.bfloat16)
            arrays[plan.name] = part.reshape((count, *plan.tail))
        return [
            tuple(arrays[f"{projection}.{kind}"] for kind in KINDS)
            for projection in PROJECTIONS
        ]

    def seed_decode(self, layer, positions, routed):
        """Retain one frequently routed expert per layer from its completed ring reads."""
        if self.owns_pool or self.pending is None or not self.pool._free or layer < 0:
            return
        if layer in self._seeded_layers:
            return
        from collections import Counter

        for expert, _ in Counter(routed).most_common():
            if (layer, expert) in self.pool._slot_of:
                continue
            rows = {}
            try:
                for plan in self.plans:
                    start, size = self.pending.row_range(plan.name, positions[expert], 1)
                    rows[tuple(plan.name.split("."))] = self._pool_view[start:start + size]
                size = self.pool.seed_from_ring(layer, expert, rows)
            finally:
                for row in rows.values():
                    row.release()
            if size:
                self._seeded_layers.add(layer)
                self._stats["seeded_bytes"] = self._stats.get("seeded_bytes", 0) + size
            return

    # -- lifetime -----------------------------------------------------------

    def _retire(self) -> None:
        """Release the previous layer's half. Call only after its GPU work
        is known to have finished."""
        lease, self.pending = self.pending, None
        if lease is None:
            return
        try:
            lease.end_compute()
        except HazardError:
            pass
        lease.release()

    def finish(self) -> None:
        """Normal end: wait for the GPU, then release the last half."""
        from .pool_lifecycle import synchronize_or_fault
        synchronize_or_fault(self._synchronize)
        self._retire()

    def abort(self) -> None:
        """Failure path: wait for the GPU, then release."""
        from .pool_lifecycle import synchronize_or_fault
        synchronize_or_fault(self._synchronize)
        self._retire()

    def close(self) -> bool:
        return self.ring.close(timeout=CLOSE_TIMEOUT)

    def dispose(self) -> bool:
        """Free a temporary pool. Returns True when it was freed.

        Call only after ``close()`` returned True. The GPU is fenced first, so
        no kernel is still reading the buffer when the last reference goes,
        and nothing here frees memory a lazily evaluated array still views.
        """
        if not self.owns_pool or self.disposed:
            return False
        from .pool_lifecycle import synchronize_or_fault
        synchronize_or_fault(self._synchronize)
        pool = self.pool
        self._buf32 = None
        self.pending = None
        self.ring.pool = None
        self.pool = None
        self._pool_view = None
        _release_pool(pool)
        self.disposed = True
        del pool
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
        return True


def acquire(language, tokens: int):
    """Lease the ring for one prefill, or None (the reason is in LAST_REASON)."""
    LAST_REASON[0] = None
    requested = configured_experts()
    if requested <= 0:
        LAST_REASON[0] = "disabled"
        return None
    if tokens < min_tokens():
        LAST_REASON[0] = "short prompt"
        return None
    import mlx.core as mx

    from . import expert_cache, expert_pool

    # A temporary scratch ring has its own allocation and lifetime; it does
    # not require the persistent decode pool to be enabled.
    if not expert_pool.enabled() and not scratch_enabled():
        LAST_REASON[0] = "expert pool and scratch ring are off"
        return None
    layers = []
    body = getattr(language, "model", language)
    for layer in getattr(body, "layers", ()):
        glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if isinstance(glu, expert_cache.StreamingSwitchGLU):
            layers.append(glu)
    if not layers:
        LAST_REASON[0] = "no streamed layers"
        return None
    first = layers[0]
    store = first.gate_proj.cache.store

    def prefix_of(glu):
        return glu.gate_proj.cache.prefix.rsplit(".", 1)[0]

    try:
        plans = plan_parts(store, prefix_of(first))
        for glu in layers[1:]:
            if plan_parts(store, prefix_of(glu)) != plans:
                raise RingUnavailable("layers differ in expert layout")
    except RingUnavailable as error:
        LAST_REASON[0] = str(error)
        return None
    group_size = first.gate_proj.group_size
    expert_bytes = sum(plan.row_bytes for plan in plans)
    wanted_experts = min(requested, first.gate_proj.num_experts)
    halves = ring_halves()
    stride = expert_pool.get_slab_layout(group_size).record_stride
    limit = metal_working_set_limit()
    if limit is None:
        LAST_REASON[0] = "working-set limit unknown"
        return None

    existing = getattr(store, "_flashnext_expert_pool", None)
    scratch = existing is None and scratch_enabled()
    active = mx.get_active_memory()
    cache = mx.get_cache_memory()
    if existing is not None:
        # Later turn: the ring lives in the pool, which active memory contains.
        pool_bytes = len(existing._mm)
        used = working_set_bytes(active, pool_bytes) + cache
        pool_note = f"includes the {pool_bytes / 1e9:.2f} GB pool"
        new_bytes, scratch_size = 0, 0
    else:
        # First turn: the pool does not exist yet, so add it once, before it
        # is created. Refusing here leaves nothing allocated.
        if scratch:
            scratch_size = expert_pool.HEADER + stride * (
                slots_per_half(wanted_experts, expert_bytes, stride) * halves + MIN_DECODE_SLOTS
            )
            new_bytes = pool_length(scratch_size, stride)
            kind = "scratch pool"
        else:
            scratch_size = 0
            new_bytes = pool_length(int(expert_pool.configured_gb() * 1e9), stride)
            kind = "pool created for the ring"
        used = active + cache + new_bytes
        pool_note = f"no pool yet; {kind} {new_bytes / 1e9:.2f} GB added"
    if not fits_working_set(0, used + transient_bytes(), limit, margin_bytes()):
        LAST_REASON[0] = (
            f"working set: MLX active {active / 1e9:.2f} GB ({pool_note}) + cache "
            f"{cache / 1e9:.2f} GB + transient {transient_bytes() / 1e9:.2f} GB + "
            f"margin {margin_bytes() / 1e9:.2f} GB exceeds limit {limit / 1e9:.2f} GB"
        )
        return None

    if existing is not None:
        pool = existing
    elif scratch:
        pool = expert_pool.ExpertPool(store, scratch_size, group_size)   # temporary
    else:
        pool = expert_pool.for_store(store, group_size)
    if pool is None:
        LAST_REASON[0] = "no pool"
        return None
    half_experts = plan_ring(
        pool.slots, pool.stride, expert_bytes, wanted_experts,
        max_pool_fraction=1.0 if scratch else 0.5, halves=halves,
    )
    ring = None
    if half_experts <= 0:
        LAST_REASON[0] = "pool too small"
    else:
        try:
            ring = PoolRing(
                pool, [(plan.name, plan.row_bytes) for plan in plans], half_experts,
                header=expert_pool.HEADER, halves=halves,
            )
        except RingError as error:
            LAST_REASON[0] = f"lease refused: {error}"
    if ring is None:
        if scratch:
            _release_pool(pool)
            del pool
            gc.collect()
        return None
    keepwarm = expert_cache._keep_gpu_warm_until if expert_cache._KEEPWARM[0] else None
    state = PrefillRing(
        pool, ring, plans, expert_cache._submit_read, expert_cache._resolve_future,
        keepwarm=keepwarm, active_memory=mx.get_active_memory,
        limit_bytes=limit, margin=margin_bytes(), pool_bytes=len(pool._mm),
        synchronize=fence, cache_memory=mx.get_cache_memory, owns_pool=scratch,
    )
    expert_cache.set_prefill_ring(state)
    return state


def release(state, failed: bool = False) -> None:
    """Return the ring. Safe on every exit path, including cancellation.

    Order: stop routing layers to the ring, wait for the GPU (``abort`` or
    ``finish`` fence), close the ring, and only then free a temporary pool. If
    the ring does not close in time a writer may still be active, so the
    temporary pool is kept alive rather than freed under it.
    """
    if state is None:
        return
    from . import expert_cache

    expert_cache.set_prefill_ring(None)
    closed = False
    fenced = False
    try:
        if failed:
            state.abort()
        else:
            state.finish()
        fenced = True
    finally:
        try:
            closed = state.close()
            if closed and fenced:
                state.dispose()
        finally:
            if state.owns_pool and not state.disposed:
                if state not in _RETAINED:
                    _RETAINED.append(state)
                LAST_REASON[0] = "ring could not be safely fenced or closed; temporary pool retained"


@contextlib.contextmanager
def session(language, tokens: int):
    """Hold the ring for one prefill. Yields the ring or None."""
    state = acquire(language, tokens) if enabled() else None
    from models.flashnext import flight_recorder
    flight_recorder.record("prefill_ring_acquire", enabled=state is not None,
                           reason=LAST_REASON[0] if state is None else None,
                           scratch=bool(state is not None and state.owns_pool),
                           ring_view_bytes=state.ring.ring_bytes if state is not None else 0,
                           scratch_mapped_bytes=state._pool_bytes if state is not None and state.owns_pool else 0)
    failed = False
    try:
        yield state
    except BaseException:
        failed = True
        raise
    finally:
        release(state, failed)
