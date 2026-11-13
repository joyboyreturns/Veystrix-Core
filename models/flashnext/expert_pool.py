"""Application-owned LRU pool of expert records (research, off by default).

Every cached expert row the production path reads is copied out of the file
cache, about 880 MB/token and about 37 ms/token in the 2026-09-23 read replay.
This pool keeps whole expert records in one anonymous memory region laid out
like the slab pack (a 4 KiB header, then fixed-stride records), wrapped once
as a Metal buffer. A decode layer addresses every routed expert in place
through the packed-slab kernels (bit 31 plus the slot), so a hit costs no read
and no copy. A miss reads its nine rows into a free or least recently used
slot through ``F_NOCACHE`` descriptors, so the file cache does not hold a
second copy.

A fixed pool cannot follow free memory the way the file cache does. In the
replay a 5 GB pool tied production and a 6 GB pool read 15 to 24 ms/token
faster; 3 and 4 GB lost. ``FLASHNEXT_EXPERT_POOL_GB`` sets the size (0, the
default, disables it). ``FLASHNEXT_EXPERT_POOL_LOCK=1`` mlocks the region so
macOS cannot compress it.

A slot is only overwritten after its last reader finished: layer L fills its
misses after layer L's router sync, which depends on layer L-1's MoE output,
and every slot this layer routes to is moved to the most recent end before a
victim is chosen.
"""
from __future__ import annotations

import fcntl
import mmap
import os
import threading
from collections import OrderedDict

import numpy as np

from .slab_pack import get_slab_layout, libc_mlock

SIZE_FLAG = "FLASHNEXT_EXPERT_POOL_GB"
# Opt-in: decode-phase pool fills read through the file cache (no F_NOCACHE), so experts the pool evicted can still be
# served from free file-cache memory (a second-level victim cache) at no reserved memory. The bytes and their order are
# unchanged, so results are identical; the prefill ring keeps its own uncached descriptors. Unset/0 = the existing behaviour.
BUFFERED_FILL_FLAG = "FLASHNEXT_EXPERT_POOL_BUFFERED_FILL"
LOCK_FLAG = "FLASHNEXT_EXPERT_POOL_LOCK"
POLICY_FLAG = "FLASHNEXT_EXPERT_POOL_POLICY"
HEADER = 4096
SLOT_BIT = 0x80000000
_PARTS = (
    ("gate_proj", "weight"), ("gate_proj", "scales"), ("gate_proj", "biases"),
    ("up_proj", "weight"), ("up_proj", "scales"), ("up_proj", "biases"),
    ("down_proj", "weight"), ("down_proj", "scales"), ("down_proj", "biases"),
)


# Set only by the opt-in memory manager (memory_manager.py): the size the NEXT pool will be created with. A pool that
# already exists keeps its size; resizing means releasing it first. None means "use FLASHNEXT_EXPERT_POOL_GB".
_BUDGET_GB = [None]


def set_budget_gb(value) -> None:
    _BUDGET_GB[0] = None if value is None else max(0.0, float(value))


def buffered_fill_enabled() -> bool:
    return os.environ.get(BUFFERED_FILL_FLAG, "0") == "1"


def configured_gb() -> float:
    if _BUDGET_GB[0] is not None:
        return _BUDGET_GB[0]
    try:
        return max(0.0, float(os.environ.get(SIZE_FLAG, "0") or 0))
    except ValueError as error:
        raise ValueError(f"{SIZE_FLAG} must be a number of GB") from error


def enabled() -> bool:
    return configured_gb() > 0


class ExpertPool:
    def __init__(self, store, size_bytes: int, group_size: int = 32,
                 lock: bool = False):
        policy = os.environ.get(POLICY_FLAG, "lru").strip().lower() or "lru"
        if policy not in ("lru", "s3fifo"):
            raise ValueError(f"{POLICY_FLAG} must be 'lru' or 's3fifo'")
        layout = get_slab_layout(group_size)
        self.store = store
        self.stride = layout.record_stride
        self.offsets = {key: layout.offset(*key) for key in _PARTS}
        self._row_sizes = {
            key: layout.shape(*key)[0] * layout.shape(*key)[1] * (4 if key[1] == "weight" else 2)
            for key in _PARTS
        }
        self.slots = (int(size_bytes) - HEADER) // self.stride
        if self.slots < 16:
            raise ValueError("expert pool is too small for one decode layer")
        length = -(-(HEADER + self.slots * self.stride) // mmap.PAGESIZE) * mmap.PAGESIZE
        self._mm = mmap.mmap(-1, length)
        self.array = np.frombuffer(self._mm, dtype=np.uint8)
        self.locked = bool(lock) and libc_mlock(
            self.array.__array_interface__["data"][0], length
        )
        import mlx.core as mx

        # MLX shape dimensions are 32-bit, so a byte view of more than 2 GiB
        # cannot be wrapped. The kernels address the pool through a char
        # pointer, so a uint32 view of the same memory serves (up to 8 GiB).
        self.buffer_mx = mx.from_dlpack(
            np.frombuffer(self._mm, dtype=np.uint32), copy=False
        )
        self._view = memoryview(self._mm)
        self._slot_of: "OrderedDict[tuple, int]" = OrderedDict()
        self._s3 = _S3FIFO(self.slots) if policy == "s3fifo" else None
        self._free = list(range(self.slots - 1, -1, -1))
        self._fds: dict = {}
        self._fill_fds: dict = {}
        self._fd_lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def usable_slots(self) -> int:
        return len(self._free) + len(self._slot_of)

    def assign(self, layer: int, experts):
        """Assign a distinct expert set that fits in the pool, or fail before mutation."""
        if isinstance(experts, int):
            experts = [experts]
        distinct = list(dict.fromkeys(experts))
        if len(distinct) > self.usable_slots:
            raise ValueError("requested experts exceed expert pool capacity")
        slot_of = self._slot_of
        policy = self._s3
        protected = {(layer, expert) for expert in distinct} if policy else None
        slots, misses = {}, []
        for expert in distinct:
            key = (layer, expert)
            slot = slot_of.get(key)
            if slot is not None:
                if policy is None:
                    slot_of.move_to_end(key)
                else:
                    policy.hit(key)
                slots[expert] = slot
        self.hits += len(slots)
        for expert in distinct:
            if expert in slots:
                continue
            if self._free:
                slot = self._free.pop()
            elif policy is None:
                _old, slot = slot_of.popitem(last=False)
            else:
                slot = slot_of.pop(policy.victim(protected, self.usable_slots))
            slot_of[(layer, expert)] = slot
            if policy is not None:
                policy.insert((layer, expert))
            slots[expert] = slot
            misses.append((expert, slot))
        self.misses += len(misses)
        return slots, misses

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.store.dir, shard), os.O_RDONLY)
                    try:
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                        try:
                            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
                        except OSError:
                            pass
                    except BaseException:
                        os.close(fd)
                        raise
                    self._fds[shard] = fd
        return fd

    def _fill_fd(self, shard: str) -> int:
        """Descriptor for decode fills: the shared uncached one, or (opt-in) a separate buffered one."""
        if not buffered_fill_enabled():
            return self._fd(shard)
        fd = self._fill_fds.get(shard)
        if fd is None:
            with self._fd_lock:
                fd = self._fill_fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.store.dir, shard), os.O_RDONLY)
                    try:
                        try:
                            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)  # read-ahead stays off
                        except OSError:
                            pass
                    except BaseException:
                        os.close(fd)
                        raise
                    self._fill_fds[shard] = fd
        return fd

    def _fill_part(self, name, targets) -> None:
        ref = self.store.refs[name]
        fd = self._fill_fd(ref.shard)
        view = self._view
        size = ref.row_bytes
        start = 0
        while start < len(targets):
            expert = targets[start][0]
            end = start + 1
            # Sixteen vectors fit the POSIX minimum IOV_MAX.
            while end < min(len(targets), start + 16) and targets[end][0] == expert + end - start:
                end += 1
            buffers = [view[base:base + size] for _, base in targets[start:end]]
            try:
                if os.preadv(fd, buffers, ref.start + expert * size) != size * len(buffers):
                    raise OSError(f"short expert pool read for {name} row {expert}")
            finally:
                for buffer in buffers:
                    buffer.release()
            start = end

    def invalidate(self, layer: int, misses) -> None:
        """Discard failed fills after all their outstanding writes have drained.

        An evicted entry cannot be restored: its slot may already be partly
        overwritten. Return these slots to the free list, never as cache hits.
        """
        for expert, slot in misses:
            key = (layer, expert)
            if self._slot_of.get(key) == slot:
                self._free.append(self.discard_slot(key))

    def discard_slot(self, key):
        """Remove a mapping without making its slot available to decode."""
        if self._s3 is not None:
            self._s3.small.pop(key, None)
            self._s3.main.pop(key, None)
        return self._slot_of.pop(key)

    def seed_from_ring(self, layer, expert, rows) -> int:
        """Copy one complete, already-loaded expert into a free decode slot."""
        key = (layer, expert)
        if not self._free or key in self._slot_of or rows.keys() != self.offsets.keys():
            return 0
        if any(len(row) != self._row_sizes[part] for part, row in rows.items()):
            return 0
        slot = self._free[-1]
        base = HEADER + slot * self.stride
        for part, row in rows.items():
            offset = base + self.offsets[part]
            self._view[offset:offset + len(row)] = row
        self._free.pop()
        self._slot_of[key] = slot
        if self._s3 is not None:
            self._s3.insert(key)
        return sum(map(len, rows.values()))

    def fill(self, prefix: str, misses, submit, chunk: int = 2):
        """Queue the nine row reads of every miss; returns the futures."""
        from .store import expert_read_batch_rows

        if chunk < 1:
            raise ValueError("expert pool fill chunk must be positive")
        misses = sorted(misses, key=lambda target: target[0])
        futures = []
        try:
            for projection, part in _PARTS:
                name = f"{prefix}.{projection}.{part}"
                offset = HEADER + self.offsets[(projection, part)]
                ref = self.store.refs.get(name)
                batch = expert_read_batch_rows(ref.row_bytes, chunk) if ref is not None else chunk
                for start in range(0, len(misses), batch):
                    targets = [
                        (expert, offset + slot * self.stride)
                        for expert, slot in misses[start:start + batch]
                    ]
                    futures.append(submit(self._fill_part, name, targets))
        except BaseException:
            # The caller cannot see a partial submission's futures. Wait for
            # every worker before allowing it to invalidate and reuse slots.
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            raise
        return futures

    def close(self) -> None:
        for fd in (*self._fds.values(), *self._fill_fds.values()):
            os.close(fd)
        self._fds.clear()
        self._fill_fds.clear()


class _S3FIFO:
    def __init__(self, capacity):
        self.capacity = capacity
        self.small = OrderedDict()
        self.main = OrderedDict()
        self.ghost = OrderedDict()

    def hit(self, key):
        lane = self.small if key in self.small else self.main
        lane[key] = min(3, lane[key] + 1)

    def insert(self, key):
        if key in self.ghost:
            del self.ghost[key]
            self.main[key] = 0
        else:
            self.small[key] = 0

    def victim(self, protected, capacity):
        # The batch fits, so at least one resident is outside the requested set.
        # A main-queue entry has at most three second chances before eviction.
        while True:
            prefer_small = len(self.small) >= max(1, capacity // 10)
            lanes = (self.small, self.main) if prefer_small else (self.main, self.small)
            for lane in lanes:
                key = next((candidate for candidate in lane if candidate not in protected), None)
                if key is not None:
                    break
            else:
                raise RuntimeError("expert pool has no unprotected S3-FIFO victim")
            count = lane.pop(key)
            if lane is self.small:
                if count:
                    self.main[key] = 0
                    continue
                self.ghost[key] = None
                if len(self.ghost) > self.capacity:
                    self.ghost.popitem(last=False)
            elif count:
                self.main[key] = count - 1
                continue
            return key


AUTO_FLAG = "FLASHNEXT_EXPERT_POOL_AUTO"
GIB = 2 ** 30
# The guard stops when compressor growth passes 2 GiB while available memory is under 3 GiB, so a pool is only
# allowed if memory still stays above that line AFTER the pool is resident, plus a margin. The pool is filled by
# the first decode tokens (first real chat 2026-11-08: available fell 4.35 to 3.21 GiB within 0.3 s of creation and
# to 2.94 GiB by the first token, with the compressor still 3.4 GiB above its start), so the reserve is its full size.
POOL_FLOOR_BYTES = 3 * GIB
POOL_MARGIN_BYTES = GIB // 2
_HEADROOM_READER = [None]      # tests inject a reader returning (available_bytes, pressure_level)


def auto_enabled() -> bool:
    return os.environ.get(AUTO_FLAG, "0") == "1"


def headroom_decision(available_bytes: int, pool_bytes: int, pressure: int = 1,
                      floor: int = POOL_FLOOR_BYTES, margin: int = POOL_MARGIN_BYTES):
    """Return ``(allowed, reason)``. The pool is optional, so any doubt means no pool."""
    from . import memory_controller

    if memory_controller.forbid_optional():
        return False, f"memory controller is {memory_controller.state().upper()} (no optional allocation)"
    need = int(pool_bytes) + floor + margin
    if pressure >= 2:
        return False, f"memory pressure level {pressure}"
    if available_bytes < need:
        return False, (f"{available_bytes / GIB:.2f} GiB available, a {pool_bytes / GIB:.2f} GiB pool needs "
                       f"{need / GIB:.2f} GiB ({floor / GIB:.0f} GiB guard floor + {margin / GIB:.1f} GiB margin)")
    return True, f"{available_bytes / GIB:.2f} GiB available covers the {need / GIB:.2f} GiB the pool needs"


def _read_headroom():
    if _HEADROOM_READER[0] is not None:
        return _HEADROOM_READER[0]()
    from . import purge_recovery

    reading = purge_recovery.read_system()
    return reading.available, reading.pressure


def _trim_allocator_cache() -> None:
    """Release unused MLX allocator blocks left by the prefill before measuring headroom (idle-safe, no sync)."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def reset_decline(store) -> None:
    """Forget an earlier refusal so the next request looks at memory again (deferred, per-request retry)."""
    if store is not None and getattr(store, "_flashnext_pool_declined", None) is not None:
        store._flashnext_pool_declined = None


def _auto_allows(store, size_bytes: int) -> bool:
    _trim_allocator_cache()
    available, pressure = _read_headroom()
    allowed, reason = headroom_decision(available, size_bytes, pressure)
    store._flashnext_pool_decision = {"allowed": allowed, "reason": reason, "available": int(available)}
    if not allowed:
        store._flashnext_pool_declined = reason
        if os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
            print(f"FlashNext expert pool: not allocated ({reason}); decoding from SSD without a pool "
                  f"(slower, memory-safe)", flush=True)
    return allowed


def for_store(store, group_size: int = 32):
    """The process's pool for this store, created on first use (``None``: run without a pool)."""
    pool = getattr(store, "_flashnext_expert_pool", None)
    if pool is None and enabled():
        if getattr(store, "_flashnext_pool_declined", None) is not None:
            return None                       # refused for this request; reset_decline() lets the next one retry
        target_gb = configured_gb()
        if auto_enabled() and not _auto_allows(store, int(target_gb * 1e9)):
            return None
        lock = os.environ.get(LOCK_FLAG, "0") == "1"
        try:
            pool = ExpertPool(
                store, int(target_gb * 1e9), group_size,
                lock=lock,
            )
        except (MemoryError, RuntimeError, OSError):
            # Guarded interactive chat must report a failed allocation rather
            # than silently running with a different pool or masking a Metal error.
            if os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
                raise
            # Recovery from allocation failure: fall back to 1 GB if larger
            if target_gb > 1.0:
                try:
                    pool = ExpertPool(
                        store, int(1.0 * 1e9), group_size,
                        lock=lock,
                    )
                    set_budget_gb(1.0)
                except (MemoryError, RuntimeError, OSError):
                    pool = None
            else:
                pool = None
        if pool is None:
            store._flashnext_pool_declined = "allocation_failed"
        store._flashnext_expert_pool = pool
        if pool is not None and os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
            print(f"FlashNext expert pool: {len(pool._mm)} bytes "
                  f"({pool.slots} slots; requested {target_gb:g} GB)", flush=True)
    return pool
