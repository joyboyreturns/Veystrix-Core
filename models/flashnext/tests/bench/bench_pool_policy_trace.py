"""Replay a recorded route trace through expert-pool replacement policies (offline; no model, no SSD).

Answers two questions with measured routing instead of an assumed residency:

* decode: what share of routed experts would a pool of N GB already hold, under the production LRU and
  under alternatives, and how far is that from the offline optimum (Belady, an upper bound no online policy
  reaches)?
* short prefill: what share of a request's prefill experts is resident in the pool left behind by the
  previous request's decode (the premise of ``FLASHNEXT_SHORT_PREFILL_POOL_REUSE``)?

Pool semantics follow ``ExpertPool.assign``: one global pool of slots keyed ``(layer, expert)``; per event
the hits are touched first, then misses take a free slot or a victim, and an entry used by the current event
is never a victim. Prefill events only query the pool (the reuse path never fills it).

The trace must be the ``route-trace.json.gz`` format of ``set_route_trace``: ``events`` rows
``[turn, phase, token, layer, [experts]]`` and ``record_bytes``. This is a replay of ONE recorded session; the
results describe that routing, not the current checkpoint's, and say nothing about speed.

    python -m models.flashnext.tests.bench.bench_pool_policy_trace TRACE.json.gz [--gb 1 2 4]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import OrderedDict

POLICIES = ("lru", "arc", "bip8", "belady")


class Lru:
    """Production policy (``ExpertPool.assign``)."""

    def __init__(self, slots, insert_cold_every=0):
        self.slots, self.pool, self.every, self.count = slots, OrderedDict(), insert_cold_every, 0

    def contains(self, key):
        return key in self.pool

    def access(self, layer, experts):
        keys = [(layer, e) for e in experts]
        current = set(keys)
        hits = 0
        for key in keys:
            if key in self.pool:
                self.pool.move_to_end(key)
                hits += 1
        for key in keys:
            if key in self.pool:
                continue
            if len(self.pool) >= self.slots:
                victim = next((k for k in self.pool if k not in current), None)
                if victim is None:
                    continue
                del self.pool[victim]
            self.pool[key] = 1
            if self.every:                       # BIP: most inserts enter at the cold end
                self.count += 1
                if self.count % self.every:
                    self.pool.move_to_end(key, last=False)
        return hits


class Arc:
    def __init__(self, slots):
        self.n, self.p = slots, 0
        self.t1, self.t2, self.b1, self.b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def contains(self, key):
        return key in self.t1 or key in self.t2

    def _replace(self, in_b2, current):
        while len(self.t1) + len(self.t2) >= self.n:
            use_t1 = self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p) or not self.t2)
            for seg, ghost in ((self.t1, self.b1), (self.t2, self.b2)) if use_t1 else ((self.t2, self.b2), (self.t1, self.b1)):
                victim = next((k for k in seg if k not in current), None)
                if victim is not None:
                    del seg[victim]
                    ghost[victim] = 1
                    while len(self.b1) > self.n:
                        self.b1.popitem(last=False)
                    while len(self.b2) > self.n:
                        self.b2.popitem(last=False)
                    break
            else:
                return

    def access(self, layer, experts):
        keys = [(layer, e) for e in experts]
        current = set(keys)
        hits = 0
        for key in keys:
            if key in self.t1:
                del self.t1[key]
                self.t2[key] = 1
                hits += 1
            elif key in self.t2:
                self.t2.move_to_end(key)
                hits += 1
        for key in keys:
            if self.contains(key):
                continue
            if key in self.b1:
                self.p = min(self.n, self.p + max(1, len(self.b2) // max(1, len(self.b1))))
                del self.b1[key]
                self._replace(False, current)
                self.t2[key] = 1
            elif key in self.b2:
                self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
                del self.b2[key]
                self._replace(True, current)
                self.t2[key] = 1
            else:
                self._replace(False, current)
                self.t1[key] = 1
        return hits


class Belady:
    """Offline optimum for this event order: evict the entry used furthest in the future."""

    def __init__(self, slots, events):
        self.slots, self.pool, self.index = slots, {}, 0
        later, self.next_use = {}, [None] * len(events)
        for i in range(len(events) - 1, -1, -1):
            layer, experts = events[i]
            self.next_use[i] = {(layer, e): later.get((layer, e), 1 << 60) for e in experts}
            for e in experts:
                later[(layer, e)] = i

    def contains(self, key):
        return key in self.pool

    def access(self, layer, experts):
        row = self.next_use[self.index]
        self.index += 1
        keys = [(layer, e) for e in experts]
        current = set(keys)
        hits = 0
        for key in keys:
            if key in self.pool:
                self.pool[key] = row[key]
                hits += 1
        for key in keys:
            if key in self.pool:
                continue
            if len(self.pool) >= self.slots:
                victim = max((k for k in self.pool if k not in current), key=self.pool.get, default=None)
                if victim is None:
                    continue
                del self.pool[victim]
            self.pool[key] = row[key]
        return hits


def make_policy(name, slots, decode_events):
    if name == "lru":
        return Lru(slots)
    if name == "bip8":
        return Lru(slots, insert_cold_every=8)
    if name == "arc":
        return Arc(slots)
    if name == "belady":
        return Belady(slots, decode_events)
    raise ValueError(name)


def load_trace(path):
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    return data["events"], int(data["record_bytes"])


def replay(events, slots, policy):
    """Return (decode_hit_rate, prefill_resident_rate) for one policy. Prefill rows only query."""
    decode = [(layer, tuple(experts)) for _, phase, _, layer, experts in events if phase == "decode"]
    pool = make_policy(policy, slots, decode)
    decode_hits = decode_total = prefill_hits = prefill_total = 0
    for _, phase, _, layer, experts in events:
        if phase == "decode":
            decode_hits += pool.access(layer, experts)
            decode_total += len(experts)
        else:
            prefill_hits += sum(pool.contains((layer, e)) for e in experts)
            prefill_total += len(experts)
    return (decode_hits / decode_total if decode_total else 0.0,
            prefill_hits / prefill_total if prefill_total else 0.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace")
    parser.add_argument("--gb", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    args = parser.parse_args(argv)
    events, record = load_trace(args.trace)
    print(f"{len(events)} events, record {record / 1e6:.2f} MB (routing of one recorded session)")
    for gb in args.gb:
        slots = int(gb * 1e9) // record
        row = []
        for name in args.policies:
            decode, prefill = replay(events, slots, name)
            row.append(f"{name} decode {decode:.1%} / prefill-resident {prefill:.1%}")
        print(f"{gb:g} GB = {slots} slots: " + "; ".join(row))


if __name__ == "__main__":
    main()
