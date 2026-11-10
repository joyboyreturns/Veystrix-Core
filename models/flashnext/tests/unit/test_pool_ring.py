"""Part-major pool ring: lease, layout, hazards, cancellation, lifetime, limits.

Pure Python against a fake pool, plus one class against the real ExpertPool
when MLX and NumPy are importable. No checkpoint, drive or GPU.
"""
from __future__ import annotations

import threading
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from models.flashnext.pool_ring import (
    COMPUTING, FILLING, FREE, READY,
    HazardError, PoolRing, RingError,
    fits_working_set, lease_range, plan_ring, return_range, slots_per_half,
)
from models.flashnext.prefetch import ExpertPrefetcher, PrefetchLayout

PARTS = [("w", 8), ("s", 4)]             # 12 bytes per expert
STRIDE = 12
HEADER = 16
WAIT = 5.0


class FakePool:
    def __init__(self, slots=40, stride=STRIDE):
        self.slots = slots
        self.stride = stride
        self._buf = bytearray(HEADER + slots * stride)
        self._view = memoryview(self._buf)
        self._slot_of = OrderedDict()
        self._free = list(range(slots - 1, -1, -1))

    @property
    def usable_slots(self):
        return len(self._free) + len(self._slot_of)

    def assign(self, layer, expert):
        key = (layer, expert)
        if key in self._slot_of:
            return self._slot_of[key]
        slot = self._free.pop() if self._free else self._slot_of.popitem(last=False)[1]
        self._slot_of[key] = slot
        return slot


def make_ring(pool=None, half_experts=4, **options):
    pool = pool or FakePool()
    return pool, PoolRing(pool, PARTS, half_experts, header=HEADER, **options)


def pattern(part, expert, size):
    return bytes((ord(part[0]) + expert + i) % 251 for i in range(size))


class PlanTests(unittest.TestCase):
    def test_ring_is_capped_by_fraction_budget_and_decode_floor(self):
        # 1,953 slots, 3,072,000-byte records, one expert per slot.
        self.assertEqual(plan_ring(1953, 3_072_000, 3_072_000, 512), 488)
        self.assertEqual(plan_ring(1953, 3_072_000, 3_072_000, 100), 100)
        self.assertEqual(
            plan_ring(1953, 3_072_000, 3_072_000, 512,
                      budget_bytes=2 * 300 * 3_072_000), 300)
        self.assertEqual(plan_ring(20, 12, 12, 512, max_pool_fraction=1.0), 2)
        self.assertEqual(plan_ring(16, 12, 12, 8), 0)

    def test_working_set_check_includes_the_pool_once(self):
        limit = 12_713_115_648
        self.assertTrue(fits_working_set(6_000_000_000, 5_000_000_000, limit, 500_000_000))
        self.assertFalse(fits_working_set(6_000_000_000, 6_300_000_000, limit, 500_000_000))

    def test_slots_per_half_rounds_up(self):
        self.assertEqual(slots_per_half(4, 12, 12), 4)
        self.assertEqual(slots_per_half(5, 12, 24), 3)


class LeaseTests(unittest.TestCase):
    def test_lease_evicts_cached_entries_and_removes_free_slots(self):
        pool = FakePool(slots=40)
        for expert in range(30):
            pool.assign(1, expert)               # slots 0..29 cached, 30..39 free
        before = dict(pool._slot_of)
        pool_, ring = make_ring(pool, half_experts=5)        # leases 10 slots: 30..39
        self.assertEqual(ring.first_slot, 30)
        self.assertEqual(pool._slot_of, before)              # 30..39 held none
        self.assertTrue(all(slot < 30 for slot in pool._free))
        ring.close()

        pool = FakePool(slots=40)
        for expert in range(40):
            pool.assign(1, expert)               # every slot cached
        _p, ring = make_ring(pool, half_experts=5)
        self.assertEqual(ring.stats()["evicted"], 10)
        self.assertTrue(all(slot < 30 for slot in pool._slot_of.values()))
        self.assertEqual(pool._free, [])
        for expert in range(100, 130):           # decode keeps working beside the lease
            self.assertLess(pool.assign(2, expert), 30)
        ring.close()

    def test_return_restores_the_slots(self):
        pool = FakePool(slots=40)
        _p, ring = make_ring(pool, half_experts=5)
        self.assertEqual(len(pool._free), 30)
        self.assertTrue(ring.close())
        self.assertEqual(len(pool._free), 40)
        self.assertEqual(sorted(pool._free), list(range(40)))
        slots = {pool.assign(3, e) for e in range(40)}
        self.assertEqual(slots, set(range(40)))

    def test_lease_that_would_starve_decode_is_refused_and_leaves_no_trace(self):
        pool = FakePool(slots=20)
        with self.assertRaises(RingError):
            PoolRing(pool, PARTS, 3, header=HEADER)          # 6 slots, 14 left < 16
        self.assertEqual(len(pool._free), 20)

    def test_return_refuses_slots_the_pool_already_uses(self):
        pool = FakePool(slots=40)
        first = lease_range(pool, 4)
        pool._free.append(first)                             # corrupt on purpose
        with self.assertRaises(RingError):
            return_range(pool, first, 4)

    def test_failed_layout_gives_the_lease_back(self):
        pool = FakePool(slots=40)
        calls = []

        class Bad(PoolRing):
            def _lay_out(self):
                raise RingError("boom")

        with self.assertRaises(RingError):
            Bad(pool, PARTS, 4, header=HEADER, give_back=lambda *a: calls.append(a))
        self.assertEqual(len(calls), 1)


class LayoutTests(unittest.TestCase):
    def test_parts_are_contiguous_non_overlapping_and_inside_the_half(self):
        pool, ring = make_ring(half_experts=4)
        lease = ring.begin_fill()
        spans = []
        for name, row in PARTS:
            offset, length = lease.row_range(name, 0, 4)
            self.assertEqual(length, 4 * row)
            spans.append((offset, offset + length))
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertEqual(a1, b0)                         # contiguous, no gap, no overlap
        base = HEADER + ring.first_slot * STRIDE
        self.assertEqual(spans[0][0], base)
        self.assertEqual(spans[-1][1] - base, 4 * 12)
        other = ring.begin_fill()
        self.assertEqual(other.row_range("w", 0)[0], base + 4 * STRIDE)
        lease.release()
        other.release()
        ring.close()

    def test_bytes_written_through_views_land_at_the_reported_offsets(self):
        pool, ring = make_ring(half_experts=4)
        lease = ring.begin_fill()
        with lease.writing():
            view = lease.part_view("w")
            for row in range(4):
                view[row * 8:(row + 1) * 8] = pattern("w", row, 8)
            del view
        offset, length = lease.row_range("w", 2)
        self.assertEqual(bytes(pool._view[offset:offset + length]), pattern("w", 2, 8))
        with self.assertRaises(IndexError):
            lease.row_range("w", 3, 2)
        with self.assertRaises(KeyError):
            lease.part_view("nope")
        lease.release()
        ring.close()

    def test_ring_stays_inside_the_pool(self):
        pool, ring = make_ring(half_experts=4)
        self.assertEqual(ring.ring_bytes, 8 * STRIDE)
        self.assertLessEqual(HEADER + (ring.first_slot + ring.count) * STRIDE, len(pool._view))
        ring.close()


class HazardTests(unittest.TestCase):
    def test_state_machine_and_illegal_transitions(self):
        _pool, ring = make_ring()
        lease = ring.begin_fill()
        self.assertEqual(lease.state, FILLING)
        with self.assertRaises(HazardError):
            lease.begin_compute()                            # not filled yet
        with self.assertRaises(HazardError):
            lease.end_compute()
        lease.mark_filled()
        self.assertEqual(lease.state, READY)
        with self.assertRaises(HazardError):
            lease.mark_filled()
        lease.begin_compute()
        self.assertEqual(lease.state, COMPUTING)
        with self.assertRaises(HazardError):
            lease.begin_compute()
        lease.end_compute()
        self.assertEqual(lease.state, READY)
        lease.release()
        self.assertEqual(lease.state, "stale")
        ring.close()

    def test_no_write_can_reach_a_half_the_gpu_may_read(self):
        _pool, ring = make_ring()
        lease = ring.begin_fill()
        with lease.writing():
            pass
        lease.mark_filled()
        for state_action in (None, "compute"):
            if state_action:
                lease.begin_compute()
            with self.assertRaises(HazardError):
                with lease.writing():
                    self.fail("write entered a READY or COMPUTING half")
        self.assertEqual(ring.stats()["refused_writes"], 2)
        lease.end_compute()
        lease.release()
        ring.close()

    def test_release_while_computing_waits_for_end_compute(self):
        _pool, ring = make_ring()
        lease = ring.begin_fill()
        lease.mark_filled()
        lease.begin_compute()
        lease.release()
        self.assertEqual(lease.state, COMPUTING)
        self.assertEqual(ring.stats()["states"].count(FREE), 1)   # the other half only
        taken = ring.begin_fill()
        self.assertIsNotNone(taken)
        self.assertIsNone(ring.begin_fill())                 # the computing half is not offered
        lease.end_compute()
        self.assertEqual(lease.state, "stale")
        self.assertIsNotNone(ring.begin_fill())              # recycled after the GPU finished
        ring.close(timeout=0.1)

    def test_a_recycled_half_refuses_late_writers(self):
        _pool, ring = make_ring()
        old = ring.begin_fill()
        old.release()                                        # FREE again, generation bumped
        new = ring.begin_fill()
        self.assertEqual(new.index, old.index)
        with self.assertRaises(HazardError):
            with old.writing():
                self.fail("a stale lease wrote into a recycled half")
        with self.assertRaises(HazardError):
            old.part_view("w")
        with new.writing():
            pass
        new.release()
        ring.close()

    def test_release_during_an_active_write_defers_until_the_writer_exits(self):
        _pool, ring = make_ring()
        lease = ring.begin_fill()
        entered, leave = threading.Event(), threading.Event()

        def writer():
            with lease.writing():
                entered.set()
                leave.wait(WAIT)

        thread = threading.Thread(target=writer)
        thread.start()
        self.assertTrue(entered.wait(WAIT))
        lease.release()
        self.assertEqual(lease.state, FILLING)               # still held by the writer
        with self.assertRaises(HazardError):
            with lease.writing():                            # no new writers once retired
                pass
        with self.assertRaises(HazardError):
            lease.mark_filled()
        leave.set()
        thread.join(WAIT)
        self.assertEqual(lease.state, "stale")
        ring.close()

    def test_mark_filled_can_wait_for_a_writer(self):
        _pool, ring = make_ring()
        lease = ring.begin_fill()
        entered, leave = threading.Event(), threading.Event()

        def writer():
            with lease.writing():
                entered.set()
                leave.wait(WAIT)

        thread = threading.Thread(target=writer)
        thread.start()
        entered.wait(WAIT)
        with self.assertRaises(HazardError):
            lease.mark_filled()
        threading.Timer(0.05, leave.set).start()
        lease.mark_filled(timeout=WAIT)
        self.assertEqual(lease.state, READY)
        thread.join(WAIT)
        lease.release()
        ring.close()

    def test_two_halves_ping_pong_and_a_third_lease_is_refused(self):
        _pool, ring = make_ring()
        a, b = ring.begin_fill(), ring.begin_fill()
        self.assertNotEqual(a.index, b.index)
        self.assertIsNone(ring.begin_fill())
        a.release()
        c = ring.begin_fill()
        self.assertEqual(c.index, a.index)
        b.release()
        c.release()
        ring.close()


class CloseTests(unittest.TestCase):
    def test_close_waits_for_busy_halves_and_is_idempotent(self):
        pool, ring = make_ring()
        lease = ring.begin_fill()
        lease.mark_filled()
        lease.begin_compute()
        self.assertFalse(ring.close(timeout=0.05))           # GPU still reading
        self.assertEqual(len(pool._free), 40 - ring.count)   # nothing returned yet
        self.assertIsNone(ring.begin_fill())                 # but closed to new fills
        lease.end_compute()
        self.assertTrue(ring.close(timeout=WAIT))
        self.assertTrue(ring.close())
        self.assertEqual(len(pool._free), 40)
        with self.assertRaises(HazardError):
            lease.part_view("w")

    def test_close_releases_idle_filling_and_ready_halves(self):
        pool, ring = make_ring()
        a, b = ring.begin_fill(), ring.begin_fill()
        b.mark_filled()
        self.assertTrue(ring.close(timeout=WAIT))
        self.assertEqual(a.state, "stale")
        self.assertEqual(len(pool._free), 40)


class SchedulerIntegrationTests(unittest.TestCase):
    LAYOUT = PrefetchLayout(PARTS)

    def reader(self, calls=None):
        sizes = dict(PARTS)

        def read(part, experts, out):
            if calls is not None:
                calls.append((part, tuple(experts)))
            for position, expert in enumerate(experts):
                size = sizes[part]
                out[position * size:(position + 1) * size] = pattern(part, expert, size)
        return read

    def stage(self, ring, lease, reader, **options):
        arena = ring.arena(lease)
        scheduler = ExpertPrefetcher(
            self.LAYOUT, 12 * 8, arena.wrap_reader(reader),
            executor=ThreadPoolExecutor(max_workers=2),
            allocator=arena.allocator, free=arena.free, **options,
        )
        return scheduler

    def test_experts_staged_into_the_ring_hold_their_bytes(self):
        pool, ring = make_ring(half_experts=4)
        lease = ring.begin_fill()
        scheduler = self.stage(ring, lease, self.reader())
        try:
            self.assertIsNotNone(scheduler.submit(1, [7, 3, 9]))
            claim = scheduler.claim(1, [3, 7, 9])
            self.assertEqual(claim.staged, {3: 0, 7: 1, 9: 2})
            lease.mark_filled(timeout=WAIT)
            lease.begin_compute()
            for part, size in PARTS:
                for expert, row in claim.staged.items():
                    offset, length = lease.row_range(part, row)
                    self.assertEqual(bytes(pool._view[offset:offset + length]),
                                     pattern(part, expert, size))
            lease.end_compute()
            claim.release()
            self.assertTrue(scheduler.drain(WAIT))
            self.assertEqual(lease.state, "stale")           # half returned on last free
        finally:
            scheduler.close()
            ring.close(timeout=WAIT)

    def test_cancel_returns_the_half_and_late_reads_cannot_write(self):
        pool, ring = make_ring(half_experts=4)
        lease = ring.begin_fill()
        started, release = threading.Event(), threading.Event()
        written = []

        def slow(part, experts, out):
            started.set()
            release.wait(WAIT)
            out[:] = bytes([0xEE]) * len(out)
            written.append(part)

        scheduler = self.stage(ring, lease, slow, chunk=1)
        try:
            scheduler.submit(1, [0, 1, 2, 3])
            self.assertTrue(started.wait(WAIT))
            scheduler.cancel(1)
            self.assertEqual(lease.state, FILLING)           # a read is still running
            self.assertEqual(ring.stats()["states"].count(FREE), 1)
            release.set()
            self.assertTrue(scheduler.drain(WAIT))
            self.assertEqual(lease.state, "stale")
            self.assertEqual(ring.stats()["states"].count(FREE), 2)
            base = HEADER + ring.first_slot * STRIDE
            data = bytes(pool._view[base:base + ring.ring_bytes])
            self.assertLessEqual(data.count(0xEE), 16)       # at most the two running chunks
        finally:
            release.set()
            scheduler.close()
            self.assertTrue(ring.close(timeout=WAIT))

    def test_gpu_reading_half_a_never_sees_a_write_while_half_b_fills(self):
        pool, ring = make_ring(half_experts=4)
        a = ring.begin_fill()
        with a.writing():
            view = a.part_view("w")
            view[:] = bytes([0xAA]) * len(view)
            del view
        a.mark_filled()
        a.begin_compute()                                    # GPU reads half A
        b = ring.begin_fill()
        scheduler = self.stage(ring, b, self.reader())
        try:
            scheduler.submit(2, [0, 1, 2, 3])
            with scheduler.claim(2, [0, 1, 2, 3]):
                pass
            offset, length = a.row_range("w", 0, 4)
            self.assertEqual(bytes(pool._view[offset:offset + length]), bytes([0xAA]) * length)
            with self.assertRaises(HazardError):
                with a.writing():
                    pass
        finally:
            a.end_compute()
            a.release()
            scheduler.close()
            ring.close(timeout=WAIT)


try:
    from models.flashnext import expert_pool as _expert_pool
    import mlx.core  # noqa: F401
except Exception:                                            # pragma: no cover
    _expert_pool = None


@unittest.skipIf(_expert_pool is None, "needs MLX and NumPy")
class SingleHalfTests(unittest.TestCase):
    """Stage 1 has no overlap, so one half is enough."""

    def test_plan_divides_the_ring_budget_by_the_half_count(self):
        self.assertEqual(plan_ring(1953, 3_072_000, 3_072_000, 512, halves=2), 488)
        self.assertEqual(plan_ring(1953, 3_072_000, 3_072_000, 512, halves=1), 512)
        self.assertEqual(plan_ring(1953, 3_072_000, 3_072_000, 2000, halves=1), 976)
        self.assertEqual(plan_ring(32, 12, 12, 100, halves=1), 16)

    def test_single_half_ring_leases_half_the_slots(self):
        pool, ring = make_ring(half_experts=4, halves=1)
        self.assertEqual(ring.halves, 1)
        self.assertEqual(ring.ring_bytes, 4 * STRIDE)
        self.assertEqual(len(pool._free), 40 - 4)
        first = ring.begin_fill()
        self.assertIsNotNone(first)
        self.assertIsNone(ring.begin_fill())          # no second half exists
        first.release()
        self.assertIsNotNone(ring.begin_fill())
        ring.close(timeout=WAIT)
        self.assertEqual(len(pool._free), 40)

    def test_single_half_walks_the_state_machine(self):
        _pool, ring = make_ring(half_experts=4, halves=1)
        lease = ring.begin_fill()
        with lease.writing():
            lease.part_view("w")[:8] = b"abcdefgh"
        lease.mark_filled(timeout=WAIT)
        lease.begin_compute()
        self.assertEqual(lease.state, COMPUTING)
        lease.end_compute()
        lease.release()
        self.assertEqual(lease.state, "stale")        # the handle is dead...
        self.assertEqual(ring.stats()["states"], [FREE])  # ...the half is free
        again = ring.begin_fill()
        self.assertIsNotNone(again)
        again.release()
        ring.close(timeout=WAIT)

    def test_invalid_half_counts_are_refused(self):
        for halves in (0, 3):
            with self.assertRaises((RingError, ValueError)):
                make_ring(halves=halves)


class RealPoolTests(unittest.TestCase):
    ROWS = [("gate_w", 819_200), ("gate_s", 102_400), ("gate_b", 102_400),
            ("up_w", 819_200), ("up_s", 102_400), ("up_b", 102_400),
            ("down_w", 819_200), ("down_s", 102_400), ("down_b", 102_400)]

    class Store:
        dir = "/nonexistent"
        refs = {}

    def make_pool(self, slots=24):
        from models.flashnext.slab_pack import get_slab_layout

        stride = get_slab_layout(32).record_stride
        return _expert_pool.ExpertPool(
            self.Store(), _expert_pool.HEADER + slots * stride
        )

    def test_real_pool_lease_assign_and_return(self):
        pool = self.make_pool(24)
        slots, _ = pool.assign(1, list(range(20)))           # fills slots 0..19, more evict
        ring = PoolRing(pool, self.ROWS, 2, header=_expert_pool.HEADER)
        self.assertEqual(ring.count, 4)
        self.assertEqual(ring.first_slot, 20)
        for layer in range(2, 6):
            assigned, _ = pool.assign(layer, list(range(10)))
            self.assertTrue(all(slot < 20 for slot in assigned.values()))
        lease = ring.begin_fill()
        with lease.writing():
            view = lease.part_view("gate_w")
            view[:8] = b"abcdefgh"
            del view
        offset, _length = lease.row_range("gate_w", 0)
        self.assertEqual(bytes(pool._view[offset:offset + 8]), b"abcdefgh")
        lease.release()
        self.assertTrue(ring.close(timeout=WAIT))
        assigned, _ = pool.assign(9, list(range(24)))
        self.assertEqual(sorted(assigned.values()), list(range(24)))


if __name__ == "__main__":
    unittest.main()
