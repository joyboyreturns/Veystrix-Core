"""Offline cache-policy comparison with train/holdout turns kept separate.

Input is an existing route trace, not a checkpoint run. Counts are simulated
expert-record misses, not measured physical SSD reads or throughput. Static
pins include an explicit cold-fill cost; no policy changes production routing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from veystrix.results import output_path
from models.flashnext.tests.bench.bench_pool_policy_trace import Arc, Lru, load_trace


class LayerLru:
    """Equal fixed partitions; unused space cannot migrate to another layer."""

    def __init__(self, capacity, layers):
        self.pools = {}
        for index, layer in enumerate(sorted(layers)):
            size = capacity // len(layers) + (index < capacity % len(layers))
            self.pools[layer] = Lru(size) if size else None

    def access(self, layer, experts):
        pool = self.pools.get(layer)
        return pool.access(layer, experts) if pool else 0


class LeastStale:
    """Layer-cycle-aware replacement in the spirit of SpecMD's Least-Stale (an approximation, not the paper's system).

    Decode visits layers in a fixed cycle, so an entry's next use is known from its layer alone: after layer ``c`` ran, an
    entry of layer ``L`` is needed again in ``(L - c - 1) mod n`` layers. The victim is the entry needed furthest ahead
    (never one used by the current call); ties go to the least recently used. No prediction, no training.
    """

    def __init__(self, capacity, layers):
        self.capacity, self.n = capacity, len(layers)
        self.order = {layer: i for i, layer in enumerate(sorted(layers))}
        self.pool = {}                                   # key -> last-use tick
        self.tick = 0

    def access(self, layer, experts):
        current = {(layer, e) for e in experts}
        hits = 0
        for key in current:
            self.tick += 1
            if key in self.pool:
                self.pool[key] = self.tick
                hits += 1
        here = self.order[layer]
        for key in sorted(current):
            if key in self.pool or self.capacity <= 0:
                continue
            if len(self.pool) >= self.capacity:
                candidates = [k for k in self.pool if k not in current]
                if not candidates:
                    continue                             # nothing evictable: served transiently, not cached
                victim = max(candidates, key=lambda k: (
                    (self.order[k[0]] - here - 1) % self.n, -self.pool[k]))
                del self.pool[victim]
            self.tick += 1
            self.pool[key] = self.tick
        return hits


class RecencyFrequency:
    """Lightweight recency+frequency replacement (a FlashMoE-style baseline without a learned predictor).

    Each resident entry carries a use count that is halved for every entry after ``capacity`` accesses, so old popularity
    decays. The victim is the lowest count, then the least recently used; the current call's entries are never evicted.
    """

    def __init__(self, capacity):
        self.capacity, self.pool, self.tick, self.since_decay = capacity, {}, 0, 0

    def _touch(self, key):
        self.tick += 1
        count = self.pool.get(key, (0, 0))[0]
        self.pool[key] = (count + 1, self.tick)
        self.since_decay += 1
        if self.since_decay >= max(1, self.capacity):
            self.since_decay = 0
            self.pool = {k: (c // 2, t) for k, (c, t) in self.pool.items()}

    def access(self, layer, experts):
        current = {(layer, e) for e in experts}
        hits = 0
        for key in sorted(current):
            if key in self.pool:
                self._touch(key)
                hits += 1
        for key in sorted(current):
            if key in self.pool or self.capacity <= 0:
                continue
            if len(self.pool) >= self.capacity:
                candidates = [k for k in self.pool if k not in current]
                if not candidates:
                    continue
                victim = min(candidates, key=lambda k: self.pool[k])
                del self.pool[victim]
            self._touch(key)
        return hits


def evaluate(events, holdout_turn, slots, record_bytes):
    """Train static expert selection on earlier turns; evaluate a fresh holdout."""
    if slots <= 0 or record_bytes <= 0:
        raise ValueError("capacity and record size must be positive")
    train = [r for r in events if r[1] == "decode" and r[0] < holdout_turn]
    test = [r for r in events if r[1] == "decode" and r[0] == holdout_turn]
    if not train or not test:
        raise ValueError("training and holdout must both contain decode events")
    layers = {r[3] for r in test}
    counts = Counter((layer, expert) for _, _, _, layer, experts in train for expert in experts)
    # Deterministic tie-break; never inspect holdout frequencies when choosing pins.
    pins = set(sorted(counts, key=lambda key: (-counts[key], key))[:slots])
    policies = {"global_lru": Lru(slots), "arc": Arc(slots), "per_layer_lru": LayerLru(slots, layers),
                "least_stale": LeastStale(slots, layers), "recency_frequency": RecencyFrequency(slots)}
    hits = {name: 0 for name in (*policies, "static_frequency")}
    total = 0
    for _, _, _, layer, experts in test:
        total += len(experts)
        for name, policy in policies.items():
            hits[name] += policy.access(layer, experts)
        hits["static_frequency"] += sum((layer, expert) in pins for expert in experts)
    out = {}
    for name, count in hits.items():
        misses = total - count
        fill = len(pins) if name == "static_frequency" else 0
        out[name] = {"hits": count, "accesses": total, "hit_rate": count / total,
                     "miss_records": misses, "estimated_miss_bytes": misses * record_bytes,
                     "cold_pin_fill_records": fill,
                     "estimated_bytes_with_cold_fill": (misses + fill) * record_bytes}
    return {"train_turns": sorted({r[0] for r in train}), "holdout_turn": holdout_turn,
            "slots": slots, "record_bytes": record_bytes, "policies": out}


def verification_bounds(events, holdout_turn, slots, record_bytes, widths=(1, 2, 3, 4, 5, 6, 7)):
    """Replay unions by layer under perfect acceptance and zero draft cost.

    Width is target positions in a verification block (native draft depth + 1).
    The same exact recorded routes are used at every width, with a cold empty
    global LRU. It estimates demand records only; no speculative execution,
    cache rollback, GPU costs, actual acceptance or physical I/O is measured.
    """
    if slots < 0 or record_bytes <= 0:
        raise ValueError("capacity must be nonnegative and record size positive")
    rows = [r for r in events if r[0] == holdout_turn and r[1] == "decode"]
    by_token = {}
    for _, _, token, layer, experts in rows:
        layers = by_token.setdefault(token, {})
        if layer in layers:
            raise ValueError("duplicate layer within a token")
        layers[layer] = set(experts)
    tokens = sorted(by_token)
    if not tokens:
        raise ValueError("no holdout decode tokens")
    layers = set(by_token[tokens[0]])
    if not layers or any(set(by_token[token]) != layers for token in tokens):
        raise ValueError("incomplete per-token layer trace")
    result = {}
    for width in widths:
        if width < 1:
            raise ValueError("block width must be positive")
        cache = Lru(slots) if slots else None
        total = misses = blocks = maximum_union = 0
        for start in range(0, len(tokens), width):
            block = tokens[start:start + width]
            blocks += 1
            for layer in sorted(layers):
                union = sorted(set().union(*(by_token[token][layer] for token in block)))
                total += len(union)
                maximum_union = max(maximum_union, len(union))
                misses += len(union) - (cache.access(layer, union) if cache else 0)
        result[str(width)] = {"blocks": blocks, "tokens": len(tokens), "union_records": total,
                              "max_experts_per_layer_block": maximum_union, "miss_records": misses,
                              "miss_bytes_per_output_token": misses * record_bytes / len(tokens)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--holdout-turn", type=int, default=2)
    parser.add_argument("--gb", nargs="+", type=float, default=[1, 2, 4])
    args = parser.parse_args()
    trace = args.trace.resolve(strict=True)
    events, record = load_trace(trace)
    results = [evaluate(events, args.holdout_turn, int(gb * 1e9) // record, record) for gb in args.gb]
    verification = {str(gb): verification_bounds(events, args.holdout_turn, int(gb * 1e9) // record, record)
                    for gb in (0, 2)}
    report = {"trace": str(trace), "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
              "scope": "offline routing replay, one held-out turn; not physical I/O or a target-checkpoint benchmark",
              "results": results, "verification_idealized": verification}
    path = output_path("flashnext", "cache-holdout", "cache-holdout.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(f'{row["slots"]} slots: ' + ", ".join(
            f'{name} {value["hit_rate"]:.1%} ({value["miss_records"]} simulated misses)'
            for name, value in row["policies"].items()))
    for gb, rows in verification.items():
        print(f"idealized verification, {gb} GB pool: " + ", ".join(
            f'{width} positions {row["miss_bytes_per_output_token"] / 1e6:.0f} MB/token'
            for width, row in rows.items()))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
