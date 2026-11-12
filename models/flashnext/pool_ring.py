"""Part-major prefill ring leased from the expert pool (opt-in, not wired in).

The expert pool is one anonymous region of fixed-stride expert records. A
synthetic MLX test (docs/flashnext/prefetch-integration.md, section 9) showed
that strided views over those records make ``gather_qmm`` materialize a copy
(about 1.6 GB per layer) and run 40 to 90% slower, while nine *contiguous*
part arrays are consumed in place, bit-identical, at compact-array speed.

So a prefill ring takes a contiguous range of slots out of the pool and
re-lays it out part-major: for each half, nine contiguous arrays of
``half_experts`` rows, in the checkpoint tensors' own layout. While leased
those slots are not decode records, so the layout is free. Nothing is
allocated beyond the pool: the ring is the 2 x half bytes already inside it.

Ownership
    ``PoolRing`` leases the highest slots of the pool. It evicts any cached
    (layer, expert) entries in that range and removes the slots from the free
    list, so decode assignment cannot hand them out. ``close`` gives them
    back. The pool object itself is not modified by this module.

Hazards
    A half is FREE, FILLING, READY or COMPUTING. Writes are accepted only in
    FILLING and only from the lease generation that owns the half, so a late
    read task can never write into a half that has been recycled, and no
    write can reach a half while the GPU may read it (READY or COMPUTING).
    The caller states the GPU contract explicitly: ``begin_compute`` before
    the graph that reads the half is evaluated, ``end_compute`` only after
    ``mx.eval`` has returned. Release during COMPUTING is deferred to
    ``end_compute``.

Pure Python: no MLX, no NumPy. The pool is any object with ``slots``,
``stride``, ``_view`` (a writable memoryview of header plus records),
``_slot_of`` (ordered mapping of key to slot) and ``_free`` (a list of free
slots), which is what ``expert_pool.ExpertPool`` has.
"""
from __future__ import annotations

import threading
from collections import namedtuple

POOL_HEADER = 4096          # expert_pool.HEADER
MIN_DECODE_SLOTS = 16       # expert_pool refuses a pool smaller than this

FREE, FILLING, READY, COMPUTING = "free", "filling", "ready", "computing"

PartSpec = namedtuple("PartSpec", "name row_bytes")


class RingError(RuntimeError):
    """The ring cannot be leased or used as asked."""


class HazardError(RuntimeError):
    """An operation would let a write overlap a GPU read, or use a dead lease."""


def slots_per_half(half_experts: int, expert_bytes: int, stride: int) -> int:
    return -(-int(half_experts) * int(expert_bytes) // int(stride))


def plan_ring(
    pool_slots: int,
    stride: int,
    expert_bytes: int,
    requested_experts: int,
    budget_bytes=None,
    max_pool_fraction: float = 0.5,
    keep_slots: int = MIN_DECODE_SLOTS,
    halves: int = 2,
) -> int:
    """Largest experts-per-half that fits every limit, at most the request.

    Limits: the ring (``halves`` halves) may not exceed ``max_pool_fraction``
    of the pool, must leave ``keep_slots`` for decode, and may not exceed
    ``budget_bytes`` when given. Returns 0 when not even one expert fits.
    """
    limit_slots = min(
        int(pool_slots * max_pool_fraction), int(pool_slots) - int(keep_slots)
    )
    if budget_bytes is not None:
        limit_slots = min(limit_slots, int(budget_bytes) // int(stride))
    half_slots = limit_slots // max(1, int(halves))
    if half_slots <= 0:
        return 0
    fit = (half_slots * int(stride)) // int(expert_bytes)
    return max(0, min(int(requested_experts), fit))


def fits_working_set(
    pool_bytes: int, mlx_peak_bytes: int, limit_bytes: int, margin_bytes: int = 0
) -> bool:
    """Whether the pool plus MLX's own peak stays under the Metal working set.

    The ring adds nothing beyond the pool, so this is the whole check; callers
    read ``max_recommended_working_set_size`` from the device.
    """
    return int(pool_bytes) + int(mlx_peak_bytes) + int(margin_bytes) <= int(limit_bytes)


def lease_range(pool, count: int) -> int:
    """Take the highest ``count`` slots out of the pool; returns the first."""
    count = int(count)
    first = int(pool.slots) - count
    if count <= 0 or first < MIN_DECODE_SLOTS:
        raise RingError("a ring of that size would leave decode too few slots")
    discard = getattr(pool, "discard_slot", pool._slot_of.pop)
    for key in [k for k, slot in pool._slot_of.items() if slot >= first]:
        discard(key)
    pool._free[:] = [slot for slot in pool._free if slot < first]
    return first


def return_range(pool, first: int, count: int) -> None:
    """Give a leased range back to the pool's free list."""
    taken = set(pool._free)
    taken.update(pool._slot_of.values())
    slots = range(int(first) + int(count) - 1, int(first) - 1, -1)
    if any(slot in taken for slot in slots):
        raise RingError("a leased slot is already in use by the pool")
    pool._free.extend(slots)


class _Half:
    __slots__ = ("state", "generation", "writers", "retired", "views")

    def __init__(self):
        self.state = FREE
        self.generation = 0
        self.writers = 0
        self.retired = False
        self.views = []


class HalfLease:
    """One lease of one ring half. Dead once the half is recycled."""

    def __init__(self, ring, index, generation):
        self._ring = ring
        self.index = index
        self.generation = generation

    @property
    def state(self) -> str:
        return self._ring._state(self)

    def part_view(self, part_name: str):
        """Writable memoryview of one part: ``half_experts`` rows."""
        return self._ring._part_view(self, part_name)

    def row_range(self, part_name: str, row: int, rows: int = 1):
        """(absolute byte offset in the pool view, byte length) of rows."""
        return self._ring._row_range(self, part_name, row, rows)

    def writing(self):
        """Context manager around one write; refused unless FILLING."""
        return _Writing(self)

    def mark_filled(self, timeout: float = 0.0) -> None:
        """FILLING to READY once no write is active (waits up to ``timeout``)."""
        self._ring._mark_filled(self, timeout)

    def begin_compute(self) -> None:
        self._ring._begin_compute(self)

    def end_compute(self) -> None:
        self._ring._end_compute(self)

    def release(self) -> None:
        self._ring._release(self)


class _Writing:
    def __init__(self, lease):
        self._lease = lease

    def __enter__(self):
        self._lease._ring._enter_write(self._lease)
        return self._lease

    def __exit__(self, *_exc):
        self._lease._ring._exit_write(self._lease)


class PoolRing:
    """One or two part-major halves inside a leased range of the expert pool."""

    def __init__(
        self,
        pool,
        parts,
        half_experts: int,
        header: int = POOL_HEADER,
        lease=lease_range,
        give_back=return_range,
        halves: int = 2,
    ):
        self.parts = tuple(PartSpec(name, int(size)) for name, size in parts)
        if not self.parts or any(part.row_bytes <= 0 for part in self.parts):
            raise ValueError("a ring needs parts with positive row sizes")
        if half_experts < 1:
            raise ValueError("half_experts must be at least 1")
        if halves not in (1, 2):
            raise ValueError("halves must be 1 or 2")
        self.halves = int(halves)
        self.pool = pool
        self.half_experts = int(half_experts)
        self.expert_bytes = sum(part.row_bytes for part in self.parts)
        self.stride = int(pool.stride)
        self.slots_per_half = slots_per_half(
            self.half_experts, self.expert_bytes, self.stride
        )
        self.count = self.halves * self.slots_per_half
        self._give_back = give_back
        self._header = int(header)
        self._cond = threading.Condition()
        self._halves = tuple(_Half() for _ in range(self.halves))
        self._closed = False
        self._released = False
        self._stats = {"fills": 0, "computes": 0, "refused_writes": 0, "evicted": 0}
        before = len(pool._slot_of)
        self.first_slot = lease(pool, self.count)
        self._stats["evicted"] = before - len(pool._slot_of)
        try:
            self._lay_out()
        except BaseException:
            give_back(pool, self.first_slot, self.count)
            raise
        self._row_bytes = {part.name: part.row_bytes for part in self.parts}

    def _lay_out(self) -> None:
        self._offsets = []
        for half in range(self.halves):
            base = (
                self._header
                + (self.first_slot + half * self.slots_per_half) * self.stride
            )
            table, cursor = {}, base
            for part in self.parts:
                table[part.name] = cursor
                cursor += part.row_bytes * self.half_experts
            if cursor - base > self.slots_per_half * self.stride:
                raise RingError("half layout exceeds its slots")
            self._offsets.append(table)

    # -- sizes --------------------------------------------------------------

    @property
    def ring_bytes(self) -> int:
        return self.count * self.stride

    def stats(self) -> dict:
        with self._cond:
            snapshot = dict(self._stats)
            snapshot["states"] = [half.state for half in self._halves]
            snapshot["ring_bytes"] = self.ring_bytes
            snapshot["first_slot"] = self.first_slot
            snapshot["closed"] = self._closed
            snapshot["released"] = self._released
            return snapshot

    # -- leasing a half -----------------------------------------------------

    def begin_fill(self):
        """Lease a FREE half for filling, or None when none is free."""
        with self._cond:
            if self._closed:
                return None
            for index, half in enumerate(self._halves):
                if half.state == FREE:
                    half.generation += 1
                    half.state = FILLING
                    half.retired = False
                    half.writers = 0
                    self._stats["fills"] += 1
                    return HalfLease(self, index, half.generation)
        return None

    def _half(self, lease):
        half = self._halves[lease.index]
        if half.generation != lease.generation:
            raise HazardError("stale lease: the half has been recycled")
        return half

    def _state(self, lease) -> str:
        half = self._halves[lease.index]
        with self._cond:
            return half.state if half.generation == lease.generation else "stale"

    def _part_view(self, lease, part_name):
        with self._cond:
            half = self._half(lease)
            if self._released:
                raise HazardError("ring is released")
            if half.state == FREE:
                raise HazardError("half is free")
            offset = self._offsets[lease.index].get(part_name)
            if offset is None:
                raise KeyError(part_name)
            size = self._row_bytes[part_name] * self.half_experts
            view = self.pool._view[offset: offset + size]
            half.views.append(view)
            return view

    def _row_range(self, lease, part_name, row, rows):
        with self._cond:
            self._half(lease)
            offset = self._offsets[lease.index].get(part_name)
            if offset is None:
                raise KeyError(part_name)
            size = self._row_bytes[part_name]
            if row < 0 or rows < 1 or row + rows > self.half_experts:
                raise IndexError("rows outside the half")
            return offset + row * size, rows * size

    # -- writes -------------------------------------------------------------

    def _enter_write(self, lease) -> None:
        with self._cond:
            half = self._half(lease)
            if half.state != FILLING or half.retired:
                self._stats["refused_writes"] += 1
                raise HazardError(f"write refused: half is {half.state}"
                                  + (" and retired" if half.retired else ""))
            half.writers += 1

    def _exit_write(self, lease) -> None:
        with self._cond:
            half = self._halves[lease.index]
            if half.generation != lease.generation:
                return
            half.writers -= 1
            self._settle(half)
            self._cond.notify_all()

    # -- state transitions --------------------------------------------------

    def _settle(self, half) -> None:
        """Free a retired half once nothing can write or read it."""
        if (
            half.retired and half.state in (FILLING, READY)
            and half.writers == 0
        ):
            self._free_half(half)

    def _free_half(self, half) -> None:
        for view in half.views:
            try:
                view.release()
            except (ValueError, BufferError):
                pass
        half.views = []
        half.state = FREE
        half.retired = False
        half.generation += 1
        half.writers = 0
        self._cond.notify_all()

    def _mark_filled(self, lease, timeout) -> None:
        with self._cond:
            half = self._half(lease)
            if half.state != FILLING:
                raise HazardError(f"cannot mark a {half.state} half filled")
            if half.writers and timeout:
                self._cond.wait_for(
                    lambda: half.writers == 0 or half.generation != lease.generation,
                    timeout,
                )
                half = self._half(lease)
            if half.writers:
                raise HazardError("writers are still active")
            half.state = READY

    def _begin_compute(self, lease) -> None:
        with self._cond:
            half = self._half(lease)
            if half.state != READY:
                raise HazardError(f"cannot compute on a {half.state} half")
            half.state = COMPUTING
            self._stats["computes"] += 1

    def _end_compute(self, lease) -> None:
        with self._cond:
            half = self._half(lease)
            if half.state != COMPUTING:
                raise HazardError("end_compute without begin_compute")
            if half.retired:
                self._free_half(half)
            else:
                half.state = READY

    def _release(self, lease) -> None:
        with self._cond:
            half = self._halves[lease.index]
            if half.generation != lease.generation or half.state == FREE:
                return
            half.retired = True
            if half.state in (FILLING, READY):
                self._settle(half)

    # -- shutdown -----------------------------------------------------------

    def close(self, timeout=None) -> bool:
        """Stop leasing and return the slots once no half is busy.

        Idle halves are released. A half with active writers or a pending
        compute blocks until it finishes or ``timeout`` passes, and then
        ``close`` returns False and can be called again.
        """
        with self._cond:
            if self._released:
                return True
            self._closed = True
            for half in self._halves:
                if half.state != FREE:
                    half.retired = True
                    self._settle(half)
            idle = self._cond.wait_for(
                lambda: all(h.state == FREE for h in self._halves), timeout
            )
            if not idle:
                return False
            self._give_back(self.pool, self.first_slot, self.count)
            self._released = True
            return True

    # -- adapter for ExpertPrefetcher ---------------------------------------

    def arena(self, lease):
        return RingArena(self, lease)


class RingArena:
    """allocator, free and reader wrapper that stage a ticket in a half.

    ``ExpertPrefetcher`` allocates one buffer per layout part in order, so the
    nth allocation maps to the nth part of the half. Every write goes through
    the lease's write guard. The half is released when the ticket frees its
    last buffer, which the scheduler does only after the claim is released.
    """

    def __init__(self, ring, lease):
        self._ring = ring
        self._lease = lease
        self._order = [part.name for part in ring.parts]
        self._next = 0
        self._freed = 0

    def allocator(self, nbytes: int):
        if self._next >= len(self._order):
            raise RingError("every part of the half is already allocated")
        name = self._order[self._next]
        self._next += 1
        view = self._lease.part_view(name)
        if nbytes > len(view):
            raise RingError("request exceeds the half")
        return view[:nbytes]

    def free(self, buffer) -> bool:
        try:
            buffer.release()
        except BufferError:
            return False
        except ValueError:
            pass
        self._freed += 1
        if self._freed >= self._next:
            self._lease.release()
        return True

    def wrap_reader(self, reader):
        lease = self._lease

        def guarded(part, experts, out):
            with lease.writing():
                reader(part, experts, out)

        return guarded
