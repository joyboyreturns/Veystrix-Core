#!/usr/bin/env python3
"""End-to-end prefill memory and speed: baseline, scratch pool, ring, cache cap.

Orchestrator and child in one file. The orchestrator builds a reversed,
fresh-process schedule (``prefill_memory_harness.build_schedule``), runs each
arm as ``--child`` under a safety watchdog, and records the lifecycle with
``MeasurementRun``. The child loads the real checkpoint, runs one scenario at
one size, and writes one JSON file. Plan: docs/flashnext/prefetch-integration.md
section 14.

This needs the real checkpoint and a plugged-in Mac, so it must not be run
without approval. ``--dry-run`` prints the schedule, commands and flag overlays
and touches nothing (no child, no results file, no checkpoint).

    python -m models.flashnext.tests.bench.bench_prefill_memory --dry-run
    python -m models.flashnext.tests.bench.bench_prefill_memory \
        --sizes 1k --conditions F0 F2 --rounds 1        # smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from veystrix.results import output_path  # noqa: E402
from models.flashnext.tests.bench import prefill_memory_harness as H  # noqa: E402

MODULE = "models.flashnext.tests.bench.bench_prefill_memory"
PRODUCTION_RING_MIN_TOKENS = 512   # prefill_ring.min_tokens() with the flag unset (a test keeps this honest)
SOURCE_TEXT = ROOT / "docs" / "flashnext" / "research.md"
FINGERPRINTED = (
    "models/flashnext/qsa_chunk.py", "models/flashnext/quantized_kv.py", "models/flashnext/mixed_kv.py",
    "models/flashnext/prefill.py", "models/flashnext/prefill_ring.py",
    "models/flashnext/prefill_memory.py", "models/flashnext/expert_pool.py",
    "models/flashnext/expert_cache.py", "models/flashnext/pool_ring.py",
    "models/flashnext/tests/bench/bench_prefill_memory.py",
    "models/flashnext/tests/bench/prefill_memory_harness.py",
)
EFFECT_METRICS = ("prefill_tok_s", "decode_tok_s_1_64", "decode_tok_s_65_128",
                  "peak_gb", "ssd_prefill_gb")


# -- child: one scenario in one fresh process ---------------------------------

class RealProbe:
    """Readings the child takes. Real MLX, vm_stat, rusage; faked in tests."""

    def memory(self) -> dict:
        import mlx.core as mx

        return {"active": mx.get_active_memory(), "peak": mx.get_peak_memory(),
                "cache": mx.get_cache_memory()}

    def reset_peak(self) -> None:
        import mlx.core as mx

        mx.reset_peak_memory()

    def metal_limit(self) -> int | None:
        from models.flashnext.prefill_ring import metal_working_set_limit

        return metal_working_set_limit()

    def vm(self) -> dict:
        from models.flashnext.diskio import vm_counters

        return vm_counters()

    def disk_bytes(self) -> int:
        from models.flashnext.diskio import disk_bytes_read

        return disk_bytes_read()

    def free_mb(self) -> float:
        from models.flashnext.diskio import free_memory_mb

        return free_memory_mb()

    def faults(self) -> dict:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"minor": usage.ru_minflt, "major": usage.ru_majflt}

    def rss_bytes(self) -> int | None:
        try:
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                                 capture_output=True, text=True, timeout=5).stdout
            return int(out.strip()) * 1024
        except Exception:
            return None

    def now(self) -> float:
        return time.perf_counter()

    def machine(self) -> dict:
        """One ``vm_stat`` read: available memory, compressor, counters."""
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            return H.parse_vm_stat(out)
        except Exception:
            return {}

    def swap_used_mb(self) -> float | None:
        return H.parse_swap_used_mb(H._run_text(["sysctl", "vm.swapusage"]))

    def pressure_level(self) -> int | None:
        return H.parse_pressure_level(H._run_text(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))


class Phase:
    """The child's current stage: the main thread sets it, the tracer reads it."""

    def __init__(self, name: str = "init"):
        self.name = name

    def set(self, name: str) -> None:
        self.name = name


def _set_phase(instrument, name: str) -> None:
    phase = getattr(instrument, "phase", None)
    if phase is not None:
        phase.set(name)


class MemoryTracer:
    """One row a second of what memory is doing, and the child's own stop rule.

    A row has the phase, MLX active/cache/peak, process RSS, memory macOS could
    reuse, the compressor, swap, the pressure level, the swap and compression
    counters and the expert pool. Every row is flushed to a JSONL file as it is
    written, so the file survives the kill that a pressure stop causes. ``event``
    adds a row between samples (the expert pool being created, for example).

    ``guard`` returns a stop reason from a row (``prefill_memory_harness.guard_reason``).
    On a reason the tracer writes it, then calls ``on_guard`` (the child exits with
    ``GUARD_EXIT_CODE``). ``step`` is one sample, so tests need no thread.
    """

    FAST_PHASES = ("feed", "prefill", "ring-acquire", "ring-release", "post-prefill")

    def __init__(self, probe, path, *, interval: float = 1.0, phase=None, store=None,
                 guard=None, on_guard=None, metal_limit=None, fast_interval: float | None = None,
                 compressor_warn_mib: float | None = None, strict_readings: bool = False,
                 clock=time.monotonic, sleep=time.sleep):
        self.probe, self.path, self.interval = probe, Path(path), interval
        # During prefill the tracer samples every ``fast_interval`` seconds with the cheap readings
        # only (MLX, one vm_stat read: about 2 ms); RSS, swap and pressure keep the ``interval`` cadence.
        self.fast_interval = fast_interval
        self.compressor_warn_mib = compressor_warn_mib
        self.strict_readings = strict_readings
        self._warned = False
        self._last_full: float | None = None
        self.phase = phase or Phase()
        self.store, self.guard, self.metal_limit = store, guard, metal_limit
        self.on_guard = on_guard or (lambda reason: os._exit(H.GUARD_EXIT_CODE))
        self.clock, self.sleep = clock, sleep
        self.rows: list = []
        self.guard_reason: str | None = None
        self._origin = clock()
        self.compressor_start_mb: float | None = None   # the arm's own baseline: its first reading
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._file = None

    def _write(self, row: dict) -> None:
        with self._lock:
            self.rows.append(row)
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("a", buffering=1)
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._file.flush()

    def event(self, name: str, **fields) -> None:
        self._write({"t": round(self.clock() - self._origin, 2), "event": name,
                     "phase": None, "during": self.phase.name, **fields})

    def _read(self, call, default=None):
        try:
            return call()
        except Exception:
            return default

    def step(self, full: bool = True) -> dict:
        now = self.clock()
        if full:
            self._last_full = now
        mem = self._read(self.probe.memory, {})
        machine = self._read(self.probe.machine, {})
        counters = machine.get("counters", {})
        pool = self._read(lambda: pool_state(self.store), {"exists": None})
        compressor = machine.get("compressor_mb")
        if compressor is not None and self.compressor_start_mb is None:
            self.compressor_start_mb = compressor
        growth = (None if compressor is None or self.compressor_start_mb is None
                  else round(compressor - self.compressor_start_mb, 1))
        row = {
            "t": round(self.clock() - self._origin, 2), "phase": self.phase.name,
            "mlx_active": mem.get("active"), "mlx_cache": mem.get("cache"), "mlx_peak": mem.get("peak"),
            "rss": self._read(self.probe.rss_bytes) if full else None,
            "available_mb": machine.get("available_mb"), "compressor_mb": compressor, "compressor_growth_mb": growth,
            "file_backed_mb": machine.get("file_backed_mb"), "anonymous_mb": machine.get("anonymous_mb"),
            "swap_used_mb": self._read(self.probe.swap_used_mb) if full else None,
            "pressure_level": self._read(self.probe.pressure_level) if full else None,
            "swapin": counters.get("swapin"), "swapout": counters.get("swapout"),
            "compress": counters.get("compress"), "decompress": counters.get("decompress"),
            "pagein": counters.get("pagein"),
            "pool": pool,
        }
        if not full:
            row["fast"] = True
        self._write(row)
        if (self.compressor_warn_mib is not None and growth is not None
                and growth > self.compressor_warn_mib and not self._warned):
            self._warned = True        # a warning is recorded, with the phase it happened in
            self.event("compressor_warning", growth_mb=growth, limit_mb=self.compressor_warn_mib,
                       available_mb=machine.get("available_mb"))
        if self.guard is not None:
            reason = self.guard({**row, "mlx_active": row["mlx_active"] if self.strict_readings else row["mlx_active"] or 0,
                                 "mlx_cache": row["mlx_cache"] if self.strict_readings else row["mlx_cache"] or 0,
                                 "metal_limit_bytes": self.metal_limit})
            if reason and self.guard_reason is None:
                self.guard_reason = reason
                self._write({"t": row["t"], "phase": None, "guard": reason, "during": row["phase"]})
                self.on_guard(reason)
        return row

    def _run(self) -> None:
        while not self._stop.is_set():
            fast = (self.fast_interval is not None and self.phase.name in self.FAST_PHASES)
            due = (self._last_full is None or self.clock() - self._last_full >= self.interval)
            self.step(full=(not fast) or due)
            self.sleep(self.fast_interval if fast else self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="memory-tracer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 2 * self.interval))
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def make_reads_uncached(store) -> int:
    """TEST-ONLY control: ``F_NOCACHE`` on every shard descriptor the store reads experts through.

    The production baseline reads experts through buffered descriptors, so every row passes
    through the file cache. This puts the same reads on uncached descriptors (as the expert
    pool and the ring already do) and changes nothing else: the same store, the same read
    mode, no pool, no ring. It patches the instance in this benchmark process only. Returns
    how many descriptors were already open (later ones are set as they are opened).

    Every descriptor it sets is remembered in ``store._uncached_applied``; ``uncached_report``
    compares that with the descriptors the store actually holds, because Python cannot read
    the flag back. ``fcntl`` raises if the OS refuses, so a descriptor in the set was accepted.
    """
    import fcntl

    applied: set = set()

    def mark(fd):
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        applied.add(fd)

    for fd in list(store._fds.values()):
        mark(fd)
    original = store._fd

    def _fd(shard):
        known = shard in store._fds
        fd = original(shard)
        if not known:
            mark(fd)
        return fd

    store._fd = _fd
    store._uncached_applied = applied
    return len(store._fds)


def uncached_report(store) -> dict:
    """Which of the store's shard descriptors had ``F_NOCACHE`` applied (none unprotected = verified)."""
    applied = getattr(store, "_uncached_applied", set())
    held = dict(getattr(store, "_fds", {}))
    return {"descriptors": len(held), "applied": len([fd for fd in held.values() if fd in applied]),
            "unprotected": sorted(shard for shard, fd in held.items() if fd not in applied),
            "shards": sorted(held)}


def find_store(language):
    """The checkpoint store the streamed layers share, or None."""
    body = getattr(language, "model", language)
    for layer in getattr(body, "layers", ()):
        glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        cache = getattr(getattr(glu, "gate_proj", None), "cache", None)
        store = getattr(cache, "store", None)
        if store is not None:
            return store
    return None


def pool_state(store) -> dict:
    pool = getattr(store, "_flashnext_expert_pool", None) if store is not None else None
    if pool is None:
        return {"exists": False}
    return {"exists": True, "slots": pool.slots, "used": len(pool._slot_of),
            "hits": pool.hits, "misses": pool.misses}


class Instrument:
    """Wraps ``prefill_target``, ``acquire`` and ``release`` for the measured turns.

    It only observes: every wrapper calls the original with the same arguments
    and returns its result unchanged. Memory is read after ``release`` returns,
    so scratch cleanup (G4) is judged on the state the decode pool will see.
    """

    def __init__(self, probe, prefill_module, ring_module, language, phase=None, events=None):
        self.probe, self.prefill, self.ring = probe, prefill_module, ring_module
        self.language = language
        self.phase = phase if phase is not None else Phase()
        self.events = events          # ``MemoryTracer.event`` or None
        self._saved = {}
        self._pool_init = None
        self.reset()

    def reset(self) -> None:
        self.calls, self.cache_max = [], 0
        self.acquired, self.reason, self.ring_stats = False, None, None
        self.acquire_s = self.dispose_s = None
        self.acquire_faults = None

    def _seen(self, reading: dict) -> dict:
        self.cache_max = max(self.cache_max, reading["cache"])
        return reading

    def __enter__(self):
        instrument = self
        original = {"target": self.prefill.prefill_target,
                    "acquire": self.ring.acquire, "release": self.ring.release}
        self._saved = dict(original)

        def prefill_target(language, ids, cache, *args, **kwargs):
            store = find_store(language)
            before = instrument._seen(instrument.probe.memory())
            record = {"tokens": int(ids.shape[1]),
                      "before": {**before, "pool_exists": pool_state(store)["exists"]}}
            began = instrument.probe.now()
            _set_phase(instrument, "prefill")
            try:
                result = original["target"](language, ids, cache, *args, **kwargs)
            finally:
                _set_phase(instrument, "post-prefill")
            record["seconds"] = instrument.probe.now() - began
            after = instrument._seen(instrument.probe.memory())
            record["after"] = {**after, "pool_exists": pool_state(store)["exists"],
                               "retained": len(instrument.ring._RETAINED)}
            instrument.calls.append(record)
            return result

        def acquire(language, tokens):
            faults = instrument.probe.faults()
            began = instrument.probe.now()
            before_phase = instrument.phase.name
            _set_phase(instrument, "ring-acquire")
            try:
                state = original["acquire"](language, tokens)
            finally:
                _set_phase(instrument, before_phase)
            instrument.acquire_s = (instrument.acquire_s or 0.0) + instrument.probe.now() - began
            instrument.acquire_faults = {
                key: instrument.probe.faults()[key] - faults[key] for key in faults
            }
            instrument.acquired = state is not None
            instrument.reason = instrument.ring.LAST_REASON[0]
            return state

        def release(state, failed=False):
            if state is not None:
                instrument.ring_stats = dict(state.stats())
            began = instrument.probe.now()
            before_phase = instrument.phase.name
            _set_phase(instrument, "ring-release")
            try:
                original["release"](state, failed)
            finally:
                _set_phase(instrument, before_phase)
            if state is not None:
                instrument.dispose_s = (instrument.dispose_s or 0.0) + instrument.probe.now() - began

        self.prefill.prefill_target = prefill_target
        self.ring.acquire = acquire
        self.ring.release = release
        if self.events is not None:
            # Observe pool creation (the decode pool and the scratch pool both pass here).
            from models.flashnext import expert_pool

            original_init = expert_pool.ExpertPool.__init__
            self._pool_init = original_init

            def pool_init(pool, store, size_bytes, *args, **kwargs):
                instrument.events("expert_pool_creating", size_bytes=int(size_bytes))
                original_init(pool, store, size_bytes, *args, **kwargs)
                instrument.events("expert_pool_created", slots=getattr(pool, "slots", None),
                                  mapped_bytes=len(getattr(pool, "_mm", b"")))

            expert_pool.ExpertPool.__init__ = pool_init
        return self

    def __exit__(self, *exc):
        self.prefill.prefill_target = self._saved["target"]
        self.ring.acquire = self._saved["acquire"]
        self.ring.release = self._saved["release"]
        if self._pool_init is not None:
            from models.flashnext import expert_pool

            expert_pool.ExpertPool.__init__ = self._pool_init
            self._pool_init = None
        return False


def kv_digest(cache) -> str:
    """Hash populated cache state, with integer packed words kept bit-exact."""
    try:
        import mlx.core as mx
        import numpy as np

        digest = hashlib.sha256()

        def add(state):
            if isinstance(state, (tuple, list)):
                for value in state:
                    add(value)
            elif state is not None:
                wide = state if state.dtype in (mx.uint32, mx.int64) else state.astype(mx.float32)
                mx.eval(wide)
                digest.update(np.asarray(wide).tobytes())

        for entry in cache:
            if hasattr(entry, "gather_selected"):
                add(list(entry.snapshot(own=False).values()))
            else:
                add(getattr(entry, "state", None))
        return digest.hexdigest()[:16]
    except Exception:
        return "unavailable"


def run_turn(backend, probe, instrument, store, name: str, feed, decode: int,
             hasher=kv_digest) -> dict:
    """Feed one turn, prefill it, decode, and return every measurement."""
    _set_phase(instrument, "feed")
    feed()
    pending = len(backend.pending)
    instrument.reset()
    probe.reset_peak()
    start_mem = probe.memory()
    start_pool = pool_state(store)
    vm_before, disk_before = probe.vm(), probe.disk_bytes()
    free_before = probe.free_mb()
    marks, stamps = {}, []

    def on_prefilled():
        _set_phase(instrument, "decode")
        marks["time"] = probe.now()
        marks["mem"] = probe.memory()
        marks["vm"], marks["disk"] = probe.vm(), probe.disk_bytes()
        marks["pool"] = pool_state(store)
        marks["rss"] = probe.rss_bytes()
        marks["first_token"] = None
        marks["kv_hash"] = hasher(backend.cache)

    def on_token(value, piece):
        stamps.append(probe.now())
        if marks.get("first_token") is None:
            marks["first_token"] = int(value)

    began = probe.now()
    _text, stats = backend.generate(decode, on_prefilled=on_prefilled, on_decode_token=on_token)
    generate_wall = probe.now() - began
    _set_phase(instrument, "turn-end")
    end_mem = probe.memory()
    ids = list(backend.tape[-stats.tokens:]) if stats.tokens else []
    prefilled = sum(call["tokens"] for call in instrument.calls)
    prefill_s = stats.prefill_seconds
    vm_end, disk_end = probe.vm(), probe.disk_bytes()
    after_pool = pool_state(store)
    return {
        "name": name,
        "pending_tokens": pending,
        "prefilled_tokens": prefilled,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefilled / prefill_s if prefill_s and prefilled else None,
        "prefill_calls": instrument.calls,
        "mem_start": start_mem,
        "mem_end_prefill": marks["mem"],
        "mem_end_decode": end_mem,
        "cache_max": instrument.cache_max,
        "rss_end_prefill": marks.get("rss"),
        "pool_start": start_pool, "pool_end_prefill": marks["pool"], "pool_end": after_pool,
        "pool_hits_decode": (after_pool.get("hits", 0) - marks["pool"].get("hits", 0)),
        "pool_misses_decode": (after_pool.get("misses", 0) - marks["pool"].get("misses", 0)),
        "pool_release": getattr(backend, "last_pool_release", None),
        "memory_report": getattr(backend, "last_memory_report", None),
        "memory_after": getattr(backend, "last_memory_after", None),
        "ring_acquired": instrument.acquired, "ring_reason": instrument.reason,
        "ring": instrument.ring_stats,
        "scratch": {"acquire_s": instrument.acquire_s, "dispose_s": instrument.dispose_s,
                    "acquire_faults": instrument.acquire_faults},
        "generate_wall_s": generate_wall,
        "first_token": marks.get("first_token"),
        "kv_hash": marks.get("kv_hash"),
        "digest": H.digest_ids(ids[:decode]),
        "generated_ids": [int(i) for i in ids[:decode]],
        "generated_tokens": stats.tokens,
        "decode": {
            "tokens": stats.tokens,
            "first_token_s": (stamps[0] - marks["time"]) if stamps and "time" in marks else None,
            "tok_s_1_64": H.window_rate(stamps, 1, 64),
            "tok_s_65_128": H.window_rate(stamps, 65, 128),
            "tok_s_overall": stats.rate,
        },
        "ssd_prefill_bytes": (marks["disk"] - disk_before) if disk_before >= 0 else None,
        "ssd_decode_bytes": (disk_end - marks["disk"]) if disk_end >= 0 else None,
        "vm": H.vm_delta(vm_before, marks["vm"]),
        "vm_decode": H.vm_delta(marks["vm"], vm_end),
        "free_mb_before": free_before,
        "finish": stats.finish,
    }


def dump_cache(caches, language, path: Path) -> None:
    """Every array of the live cache plus the language model's position state, for offline comparison."""
    import mlx.core as mx

    from models.flashnext import prefix_cache as P

    tensors = {}
    for index, cache in enumerate(caches):
        if P.kind_of(cache) == "arrays":
            for slot, value in enumerate(cache.cache):
                if value is not None:
                    tensors[f"l{index}.s{slot}"] = value
        else:
            n = int(cache.offset)
            tensors[f"l{index}.keys"] = cache.keys[..., :n, :]
            tensors[f"l{index}.values"] = cache.values[..., :n, :]
            tensors[f"l{index}.index_keys"] = cache.index_keys
            tensors[f"l{index}.index_position_ids"] = cache.index_position_ids
    for name, attr in P.LANG_FIELDS:
        value = getattr(language, attr, None)
        if value is not None:
            tensors[f"lang.{name}"] = value
    mx.eval(list(tensors.values()))
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), tensors)


def dumping_hasher(backend, directory, label: str):
    """The KV hasher, plus a dump of the cache when ``directory`` is set."""
    def hasher(cache):
        value = kv_digest(cache)
        if directory:
            dump_cache(cache, backend.language, Path(directory) / f"{label}.safetensors")
        return value
    return hasher


def store_checksums(session) -> dict:
    """SHA-256 of every array of every RAM entry in the prefix cache (to prove nothing mutated them)."""
    import mlx.core as mx
    import numpy as np

    out = {}
    for key, entry in list(session.store()._ram.items()):
        digest = hashlib.sha256()
        for layer in entry.layers:
            for value in (*layer.states, *layer.kv.values()):
                if value is not None:
                    digest.update(np.array(value.view(mx.uint8) if value.dtype != mx.uint8 else value).tobytes())
        for name in sorted(entry.lang):
            digest.update(np.array(entry.lang[name].view(mx.uint8)).tobytes())
        out[key] = digest.hexdigest()[:16]
    return out


def request_text(spec: str, prompts: dict, long_prompts: dict | None, agent_prompts: dict | None = None):
    """(system, user) for one request spec."""
    if spec == "ref":
        return prompts["system"], prompts["first"]
    if spec == "later":     # the same system message with a different question
        return prompts["system"], prompts["later"]
    kind, _, name = spec.partition(":")
    chosen = {"long": long_prompts, "agent": agent_prompts}.get(kind)
    if chosen is None or name not in chosen["questions"]:
        raise ValueError(f"unknown request {spec!r}")
    return chosen["system"], chosen["questions"][name]


class ChunkRecorder:
    """Records the real token count of every chunk the prefill runs (what actually executed, not the plan)."""

    def __init__(self):
        self.spans = []
        self.seconds = []

    def __enter__(self):
        from models.flashnext import prefill

        self._prefill = prefill
        self._saved = {name: getattr(prefill, name) for name in ("_prefill_chunk", "_prefill_target")}

        def wrap(name):
            original = self._saved[name]

            def run(language, ids, *args, **kwargs):
                self.spans.append(int(ids.shape[1]))
                started = time.perf_counter()
                try:
                    return original(language, ids, *args, **kwargs)
                finally:
                    self.seconds.append(time.perf_counter() - started)
            return run

        for name in self._saved:
            setattr(prefill, name, wrap(name))
        return self

    def __exit__(self, *exc):
        for name, original in self._saved.items():
            setattr(self._prefill, name, original)
        return False

    def edges(self, offset: int = 0) -> list:
        """Absolute end position of each chunk."""
        total, out = offset, []
        for width in self.spans:
            total += width
            out.append(total)
        return out


def prefix_report(backend) -> dict | None:
    """The backend's last prefix-cache report as a plain dict (the real one is a ``Report`` object, a
    test double may already be a dict), so it can be logged and written to JSON."""
    report = getattr(backend, "last_prefix", None)
    if report is None:
        return None
    return report.as_dict() if hasattr(report, "as_dict") else dict(report)


def run_prefix_requests(backend, probe, instrument, store, prompts, long_prompts, requests, decode,
                        dump_dir, event=lambda *a, **k: None, save=lambda turns: None,
                        agent_prompts=None) -> tuple[list, dict]:
    """Each request is a fresh conversation (reset, then open), as the server does for a request that is
    not an extension. Returns the turns and the cache-integrity evidence."""
    from models.flashnext import prefix_session

    turns, evidence = [], {"checksums": []}
    for index, spec in enumerate(requests):
        if index:
            backend.reset()
        system, user = request_text(spec, prompts, long_prompts, agent_prompts)
        label = f"{index + 1}-{spec.replace(':', '_')}"
        with ChunkRecorder() as chunks:
            turn = run_turn(
                backend, probe, instrument, store, label,
                lambda: backend.open_conversation(system, user, enable_thinking=False), decode,
                dumping_hasher(backend, dump_dir, f"{evidence.get('arm', 'arm')}-{label}"))
        turn["chunk_tokens_run"] = list(chunks.spans)
        turn["chunk_seconds"] = [round(x, 3) for x in chunks.seconds]
        turns.append(turn)
        save(turns)             # the measured turn (output digest, KV hash, speeds) is on disk before any extras
        try:
            prompt_ids = list(backend.tape[:turn["pending_tokens"]])
            prefix = prefix_report(backend)
            turn["chunk_end_positions"] = chunks.edges((prefix or {}).get("hit_tokens", 0))
            session_ = getattr(backend, "_prefix", None)
            if session_ is not None:
                turn["system_boundary_tokens"] = session_.system_boundary(prompt_ids)
            turn.update({"request": spec, "prefix": prefix,
                         "prompt_digest": H.digest_ids(prompt_ids), "prompt_tokens": len(prompt_ids)})
            session = getattr(backend, "_prefix", None)
            if session is not None and session._store is not None:
                evidence["checksums"].append({"after": label, "entries": store_checksums(session)})
            event("prefix_request", request=label, hit=(prefix or {}).get("hit_tokens"))
        except Exception as error:      # reporting must never lose a measured turn
            turn["report_error"] = f"{type(error).__name__}: {error}"
        save(turns)
    sums = evidence["checksums"]
    # An entry present at several checkpoints must hash identically at each (nothing mutated it).
    changed = {}
    for earlier, later in zip(sums, sums[1:]):
        for key, digest in earlier["entries"].items():
            if key in later["entries"] and later["entries"][key] != digest:
                changed[key] = (earlier["after"], later["after"])
    evidence["entries_changed_between_requests"] = changed
    session = getattr(backend, "_prefix", None)
    if session is not None and session._store is not None:
        store_ = session.store()
        evidence["store_usage"] = store_.usage()
        evidence["store_stats"] = dict(store_.stats.__dict__)
    return turns, evidence


def run_quality(backend, suite, event=lambda *a, **k: None, save=lambda results: None) -> dict:
    """The task-level quality suite (quality_suite.py) on the loaded backend: each case a fresh conversation,
    greedy decoding, thinking off. ``suite`` is smoke, full, or comma-separated case ids. The memory guards
    keep running underneath; results are saved after every case so a guard stop keeps what finished."""
    from models.flashnext.tests.bench import quality_suite as Q

    cases = Q.build_cases()
    ids = [i for i in suite.split(",") if i] if suite not in ("smoke", "full") else None
    chosen = Q.select(cases, tier=suite if ids is None else "smoke", ids=ids)
    systems = Q.system_prompts()
    began = time.perf_counter()
    results = Q.run_suite(backend, chosen, systems, event=event, save=lambda r: save(r),
                          prefix_report=prefix_report)
    return {"suite": suite, "systems": Q.describe(systems, getattr(backend, "tokenizer", None)),
            "cases": results, "passed": sum(1 for r in results if r["passed"]), "total": len(results),
            "wall_s": time.perf_counter() - began}


def run_scenario(backend, probe, instrument, store, cond: H.Condition, prompts: dict,
                 decode: int, environ, flags: dict | None = None,
                 hasher=kv_digest, extra: dict | None = None) -> list[dict]:
    """First-turn: apply the condition, measure turn 1. Second-turn: run turn 1
    under the baseline flags as setup, apply the condition, measure turn 2."""
    turns = []

    def overlay(condition):
        # ``extra`` (diagnostic overrides) goes on after the overlay, which unsets every
        # managed flag; otherwise an override of one of them would be wiped out.
        H.apply_overlay(environ, H.condition_overlay(condition, flags))
        H.apply_overlay(environ, extra or {})

    if cond.turn == "second":
        overlay(H.ALL_CONDITIONS["S0"])
        turns.append(run_turn(
            backend, probe, instrument, store, "setup",
            lambda: backend.open_conversation(prompts["system"], prompts["first"],
                                              enable_thinking=False),
            decode, hasher))
        overlay(cond)
        turns.append(run_turn(
            backend, probe, instrument, store, "measured",
            lambda: backend.append_user(prompts["later"], enable_thinking=False),
            decode, hasher))
    else:
        overlay(cond)
        turns.append(run_turn(
            backend, probe, instrument, store, "measured",
            lambda: backend.open_conversation(prompts["system"], prompts["first"],
                                              enable_thinking=False),
            decode, hasher))
    return turns


def child_main(args) -> int:
    cond = H.ALL_CONDITIONS[args.condition]
    H.apply_overlay(os.environ, H.launch_overlay(cond))
    from models.flashnext.settings.launch import apply_chat_environment

    apply_chat_environment(os.environ)
    os.environ["FLASHNEXT_GPU_KEEPWARM"] = "0"
    # Diagnostic overrides, applied on top of the production chat environment and recorded
    # in the arm's evidence. The Q4 weights and the routing are never touched.
    pool_gb = getattr(args, "expert_pool_gb", None)
    ring_min = getattr(args, "ring_min_tokens", None)
    if pool_gb is not None:
        os.environ["FLASHNEXT_EXPERT_POOL_GB"] = f"{pool_gb:g}"
    from models.flashnext import prefill_ring as _ring_flags

    extra = {} if ring_min is None else {_ring_flags.FLAG_MIN_TOKENS: str(ring_min)}
    if cond.prefix_cache:
        from models.flashnext import prefix_session as _prefix

        if not getattr(args, "prefix_dir", None):
            raise SystemExit("F2P needs --prefix-dir")
        extra.update({_prefix.FLAG_DIR: str(args.prefix_dir),
                      _prefix.FLAG_MIN_TOKENS: str(args.prefix_min_tokens),
                      _prefix.FLAG_RAM_GB: f"{args.prefix_ram_gb:g}",
                      _prefix.FLAG_DISK_GB: f"{args.prefix_disk_gb:g}"})
    H.apply_overlay(os.environ, extra)
    target = output_path("flashnext", "prefill-memory", f"arm-{args.condition}.json", args.json)
    pins = target.parent / "pins"
    pins.mkdir(exist_ok=True)
    source = Path(os.path.expanduser("~/.cache/flashnext"))
    for item in list(source.glob("pins.json")) + list(source.glob("slab-frozen-*.json")):
        shutil.copy2(item, pins / item.name)
    os.environ["FLASHNEXT_PIN_CACHE"] = str(pins / "pins.json")

    probe = RealProbe()
    phase = Phase("load")
    trace_path = target.parent / f"{target.stem}.trace.jsonl"
    min_available = getattr(args, "min_available_gb", H.MIN_AVAILABLE_GB)
    compressor_gib = getattr(args, "max_compressor_growth_gib", H.MAX_COMPRESSOR_GROWTH_GIB)
    state = {"evidence": None, "store": None, "uncached": False}

    def leave(reason: str) -> None:
        """The child's own stop: say why in the arm file (the trace already has the row), then
        exit at once. Nothing is left to clean up: the pools are anonymous memory and the
        descriptors are closed by the exit."""
        try:
            last = next((row for row in reversed(tracer.rows) if row.get("phase") is not None), None)
            target.write_text(json.dumps({
                "status": "guard_stop", "reason": reason, "classification": "resource_stop",
                "phase": (last or {}).get("phase"), "last_sample": last,
                "trace_file": trace_path.name, "evidence": state["evidence"],
                "uncached_reads": uncached_report(state["store"]) if state["uncached"] else None},
                indent=1) + "\n")
        except Exception:
            pass
        finally:
            os._exit(H.GUARD_EXIT_CODE)

    fast_interval = getattr(args, "fast_trace_interval", H.FAST_TRACE_SECONDS)
    swap_ceiling_only = getattr(args, "swap_ceiling_only", False)
    stop_available_gib = getattr(args, "compressor_stop_available_gib", H.COMPRESSOR_STOP_AVAILABLE_GIB)
    tracer = MemoryTracer(
        probe, trace_path, interval=getattr(args, "trace_interval", 1.0), phase=phase,
        metal_limit=probe.metal_limit(), on_guard=leave,
        fast_interval=fast_interval or None, compressor_warn_mib=compressor_gib * 1024,
        guard=lambda row: (None if row.get("fast") else H.absolute_swap_stop(row)) if swap_ceiling_only
        else H.guard_reason(row, min_available_gb=min_available,
                                         compressor_limit_mib=compressor_gib * 1024,
                                         compressor_stop_available_mib=stop_available_gib * 1024))
    tracer.start()          # before the model loads, so the load is traced too

    from veystrix.backends.flashnext import FlashNextBackend
    from veystrix.checkpoints import resolve_flashnext
    from models.flashnext import prefill, prefill_ring
    from models.flashnext.checkpoint_policy import resolve_resident_experts
    from models.flashnext.routing import checkpoint_identity_for_store

    checkpoint = str(resolve_flashnext(os.environ.get("VEYSTRIX_FLASHNEXT_MODEL")))
    resident = resolve_resident_experts(None, checkpoint) or 32
    backend = FlashNextBackend(model_path=checkpoint, resident_experts=resident)
    store = find_store(backend.language)
    tracer.store = store
    uncached = make_reads_uncached(store) if cond.uncached_reads and store is not None else None
    state["store"], state["uncached"] = store, uncached is not None
    if uncached is not None:
        tracer.event("uncached_reads_applied", **uncached_report(store))
    phase.set("prompts")
    prompts = H.build_prompts(backend.tokenizer, SOURCE_TEXT.read_text(), H.ALL_SIZES[args.size])
    evidence = {
        "arm_id": args.arm_id or f"{args.size}-{args.condition}", "size": args.size,
        "turn": cond.turn, "condition": args.condition, "round": args.round,
        "reverse": args.reverse, "checkpoint": checkpoint,
        "checkpoint_identity": checkpoint_identity_for_store(store) if store else None,
        "metal_limit_bytes": probe.metal_limit(),
        "diagnostic_overrides": {"expert_pool_gb": pool_gb, "ring_min_tokens": ring_min,
                                 "min_available_gb": min_available,
                                 "max_compressor_growth_gib": compressor_gib,
                                 "compressor_stop_available_gib": stop_available_gib,
                                 "trace_interval_s": tracer.interval,
                                 "fast_trace_interval_s": fast_interval},
        "prompts": {key: prompts[key] for key in
                    ("source_hash", "first_hash", "later_hash", "first_ids", "later_ids")},
        "after_load": {**probe.memory(), "rss": probe.rss_bytes(),
                       "pool": pool_state(store)},
        "environment": {k: os.environ.get(k) for k in sorted(os.environ)
                        if k.startswith("FLASHNEXT_")},
        "trace_file": trace_path.name,
        "uncached_reads": None if uncached is None else {"descriptors_open_at_start": uncached},
        "status": "running", "turns": [],
    }
    state["evidence"] = evidence
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    with Instrument(probe, prefill, prefill_ring, backend.language,
                    phase=phase, events=tracer.event) as instrument:
        if getattr(args, "suite", None):
            H.apply_overlay(os.environ, H.condition_overlay(cond))
            H.apply_overlay(os.environ, extra)
            evidence["quality_suite"] = args.suite
            evidence["prefix_settings"] = {k: os.environ.get(k) for k in sorted(os.environ)
                                           if k.startswith(("FLASHNEXT_PREFIX_CACHE", "FLASHNEXT_PREFILL"))}

            def save_quality(results):
                evidence["quality"] = {"suite": args.suite, "cases": list(results), "partial": True}
                target.write_text(json.dumps(evidence, indent=1, default=str) + "\n")

            evidence["quality"] = run_quality(backend, args.suite, event=tracer.event, save=save_quality)
        elif cond.prefix_cache:
            requests = [r for r in str(getattr(args, "requests", "ref")).split(",") if r]
            long_prompts = None
            if any(r.startswith("long") for r in requests):
                long_prompts = H.build_long_prompts(backend.tokenizer, SOURCE_TEXT.read_text())
                evidence["long_prompts"] = {k: v for k, v in long_prompts.items() if k not in ("system", "questions")}
            agent_prompts = None
            if any(r.startswith("agent") for r in requests):
                agent_prompts = H.build_agent_prompts(backend.tokenizer)
                evidence["agent_prompts"] = {k: v for k, v in agent_prompts.items() if k not in ("system", "questions")}
            H.apply_overlay(os.environ, H.condition_overlay(cond))
            H.apply_overlay(os.environ, extra)
            evidence["requests"] = requests
            evidence["prefix_settings"] = {k: os.environ.get(k) for k in sorted(os.environ)
                                           if k.startswith("FLASHNEXT_PREFIX_CACHE")}
            evidence["arm"] = args.arm_id or f"{args.size}-{args.condition}"
            def save_turns(turns):
                evidence["turns"] = list(turns)
                target.write_text(json.dumps(evidence, indent=1, default=str) + "\n")

            evidence["turns"], extra_evidence = run_prefix_requests(
                backend, probe, instrument, store, prompts, long_prompts, requests, args.decode,
                getattr(args, "dump_dir", None), event=tracer.event, save=save_turns,
                agent_prompts=agent_prompts)
            evidence["prefix_integrity"] = extra_evidence
        else:
            hasher = dumping_hasher(backend, getattr(args, "dump_dir", None),
                                    args.arm_id or f"{args.size}-{args.condition}")
            evidence["turns"] = run_scenario(
                backend, probe, instrument, store, cond, prompts, args.decode, os.environ,
                hasher=hasher, extra=extra)
    phase.set("done")
    if uncached is not None:
        report = uncached_report(store)
        tracer.event("uncached_reads_final", **report)
        evidence["uncached_reads"] = {"descriptors_open_at_start": uncached, **report,
                                      "verified": report["descriptors"] > 0 and not report["unprotected"]}
    tracer.step()
    tracer.stop()
    evidence["trace"] = {"file": trace_path.name, **H.summarize_trace(tracer.rows)}
    evidence["status"] = "completed"
    target.write_text(json.dumps(evidence, indent=1) + "\n")
    print(f"wrote {target}", flush=True)
    return 0


# -- readiness check (no model, no child, no inference) ---------------------------

def backend_contract() -> list[str]:
    """Problems with the production API this harness calls (empty = compatible).

    The child wraps ``prefill.prefill_target`` and ``prefill_ring.acquire/release``,
    reads ``Stats`` and the expert pool's counters, and drives the backend through
    ``open_conversation``, ``append_user`` and ``generate``. A rename in any of them
    would break a real run after the model had already loaded, so check them first.
    """
    import inspect

    from veystrix.backends.flashnext import FlashNextBackend, Stats
    from models.flashnext import expert_pool, prefill, prefill_ring

    problems = []

    def need(owner, name, kind, label):
        if not hasattr(owner, name):
            problems.append(f"{label} is missing")
        elif kind == "callable" and not callable(getattr(owner, name)):
            problems.append(f"{label} is not callable")

    def params(function, names, label):
        have = set(inspect.signature(function).parameters)
        problems.extend(f"{label} lacks parameter {n}" for n in names if n not in have)

    params(FlashNextBackend.__init__, ("model_path", "resident_experts"), "FlashNextBackend()")
    params(FlashNextBackend.generate, ("max_tokens", "on_prefilled", "on_decode_token"),
           "FlashNextBackend.generate")
    params(FlashNextBackend.open_conversation, ("system", "user", "enable_thinking"),
           "FlashNextBackend.open_conversation")
    params(FlashNextBackend.append_user, ("text", "enable_thinking"), "FlashNextBackend.append_user")
    from veystrix.conversation import Conversation

    source = inspect.getsource(FlashNextBackend) + inspect.getsource(Conversation)
    problems.extend(f"backend does not set self.{n}" for n in ("tape", "pending", "cache")
                    if f"self.{n}" not in source)
    stats_fields = set(Stats.__dataclass_fields__)
    problems.extend(f"Stats lacks {n}" for n in ("tokens", "prefill_seconds", "finish", "prompt_tokens")
                    if n not in stats_fields)
    need(Stats, "rate", "any", "Stats.rate")
    for name in ("prefill_target",):
        need(prefill, name, "callable", f"prefill.{name}")
    for name in ("acquire", "release", "metal_working_set_limit"):
        need(prefill_ring, name, "callable", f"prefill_ring.{name}")
    for name in ("LAST_REASON", "_RETAINED", "FLAG_EXPERTS", "FLAG_SCRATCH"):
        need(prefill_ring, name, "any", f"prefill_ring.{name}")
    pool_source = inspect.getsource(expert_pool.ExpertPool)
    for name in ("slots", "hits", "misses", "_slot_of"):
        if f"self.{name}" not in pool_source:
            problems.append(f"ExpertPool does not set self.{name}")
    return problems


def prompt_sizes(tokenizer, sizes: list[str], source: str) -> list[dict]:
    """Real first-turn and second-turn token counts per size, from the tokenizer alone."""
    from veystrix.conversation import Conversation

    rows = []
    for size in sizes:
        target = H.ALL_SIZES[size]
        prompts = H.build_prompts(tokenizer, source, target)
        conversation = Conversation(tokenizer)
        first = conversation.open_conversation(prompts["system"], prompts["first"], enable_thinking=False)
        conversation.tape.extend(conversation.pending)
        conversation.pending = []
        conversation.tape.extend(tokenizer.encode("A short reply."))
        later = conversation.append_user(prompts["later"], enable_thinking=False)
        rows.append({"size": size, "target": target, "first": int(first), "later": int(later)})
    return rows


def check_ready(args, *, sampler=None, preflight_module=None, tokenizer_loader=None,
                contract=backend_contract, environ=None, free_gb=None):
    """Everything the F0-versus-F2 smoke test needs, checked without starting it.

    Returns a ``preflight.Report`` whose verdict is READY or BLOCKED. Nothing here loads
    the model, starts a child process or runs inference.
    """
    from models.flashnext import preflight as pf

    preflight_module = preflight_module or pf
    environ = os.environ if environ is None else environ
    path = pf.resolve_target(getattr(args, "checkpoint", None))
    report = pf.Report(path)
    inner = preflight_module.check_checkpoint(path)
    report.add("checkpoint preflight", {pf.READY: pf.PASS}.get(inner.verdict, pf.FAIL),
               f"{inner.verdict} ({len(inner.by_status(pf.FAIL))} failing, {len(inner.by_status(pf.WARN))} warning checks)"
               + ("" if inner.verdict == pf.READY else "; run: python -m models.flashnext.preflight"))
    # the child resolves the checkpoint exactly like this
    try:
        from veystrix.checkpoints import resolve_flashnext

        resolved = str(resolve_flashnext(getattr(args, "checkpoint", None)
                                         or environ.get("VEYSTRIX_FLASHNEXT_MODEL") or None))
        report.add("child resolution", pf.PASS if Path(resolved) == Path(path).resolve() else pf.WARN,
                   f"resolve_flashnext -> {resolved}")
    except (ValueError, SystemExit) as error:
        report.add("child resolution", pf.FAIL, str(error).splitlines()[0])
    problems = contract()
    report.add("backend API", pf.FAIL if problems else pf.PASS,
               "; ".join(problems) if problems else "the calls the harness makes still exist")
    if inner.verdict == pf.READY or tokenizer_loader is not None:
        try:
            tokenizer = (tokenizer_loader or _load_tokenizer)(path)
            rows = prompt_sizes(tokenizer, args.sizes, SOURCE_TEXT.read_text())
            for row in rows:
                off = max(abs(row["first"] - row["target"]), abs(row["later"] - row["target"])) / row["target"]
                report.add(f"prompt size {row['size']}", pf.PASS if off <= 0.03 else (pf.WARN if off <= 0.10 else pf.FAIL),
                           f"turn 1 prefills {row['first']} tokens, turn 2 {row['later']} (target {row['target']}, off by {off:.1%})",
                           **row)
        except Exception as error:      # a tokenizer problem must be reported, not crash the check
            report.add("prompt sizes", pf.FAIL, f"{type(error).__name__}: {error}")
    else:
        report.add("prompt sizes", pf.SKIP, "needs a READY checkpoint (tokenizer files)")
    sampler = sampler or H.Sampler(vm=_vm_counters, load=lambda: os.getloadavg()[0],
                                   memory=_memory_reading)
    sample = sampler.sample()
    reason = H.Watchdog(H.SafetyPolicy(
        final=bool(getattr(args, "final", False)),
        swap_ceiling_only=getattr(args, "swap_ceiling_only", False),
        min_available_gb=getattr(args, "min_available_gb", H.MIN_AVAILABLE_GB)), sample).check(sample)
    reason = reason or start_gate_reason(sample, getattr(args, "start_available_gb", None))
    avail_mb = sample.get("available_mb")
    avail_str = f"{avail_mb / 1000:.2f} GB available" if avail_mb is not None else ""
    report.add("machine state", pf.FAIL if reason else pf.PASS,
               reason or f"{avail_str + ', ' if avail_str else ''}battery {sample.get('battery_percent')}% on {sample.get('power_source')}, "
                         f"Low Power Mode {sample.get('low_power')}, pressure level {sample.get('pressure_level')}, "
                         f"swap used {sample.get('swap_used_mb')} MB")
    if sample.get("power_source") != "ac":
        report.add("power", pf.WARN, "on battery: every record will be marked preliminary; "
                   "final numbers need AC power and Low Power Mode off (--final enforces it)")
    free = free_gb if free_gb is not None else _free_memory_gb()
    if free is not None:
        report.add("free memory", pf.WARN if free < 3.0 else pf.PASS,
                   f"{free:.1f} GB free or inactive" + ("; close browsers and other apps before a real run" if free < 3.0 else ""))
    leaked = {k: v for k, v in environ.items() if k.startswith("FLASHNEXT_") and k not in H.production_flags().values()}
    report.add("environment", pf.WARN if leaked else pf.PASS,
               f"set in this shell, overriding the production chat defaults: {sorted(leaked)}" if leaked
               else "no FLASHNEXT_* variables set in this shell (the baseline is the production chat environment)")
    overrides = diagnostic_overrides(args)
    start_gb = overrides.get("start_available_gb")
    min_gb = overrides.get("min_available_gb", H.MIN_AVAILABLE_GB)
    thresholds_desc = (
        f"launch gate: {start_gb:g} GiB available memory; child runtime stop floor: {min_gb:g} x 1000 MiB available memory"
        if start_gb is not None else
        f"child runtime stop floor: {min_gb:g} GB available memory"
    )
    active = {k: v for k, v in overrides.items() if k in ("expert_pool_gb", "ring_min_tokens") and v is not None}
    if active:
        report.add("diagnostic overrides", pf.INFO,
                   f"{active}: decode pool and ring threshold only; Q4 weights and routing are unchanged; "
                   f"{thresholds_desc}",
                   **overrides)
    elif start_gb is not None:
        report.add("memory thresholds", pf.INFO,
                   thresholds_desc,
                   **overrides)
    ring_floor = overrides["ring_min_tokens"] if overrides["ring_min_tokens"] is not None else PRODUCTION_RING_MIN_TOKENS
    if any(H.ALL_CONDITIONS[name].ring_on for name in args.conditions):
        short = [size for size in args.sizes if H.ALL_SIZES[size] < ring_floor]
        report.add("ring activation", pf.WARN if short else pf.PASS,
                   (f"prompts of {short} are shorter than the ring minimum ({ring_floor} tokens): the ring will "
                    f"report 'short prompt' and not activate, so an F0-versus-F2 comparison would compare nothing. "
                    f"Pass --ring-min-tokens to lower it for a diagnostic run; the memory guard stays on")
                   if short else f"every size is at least the ring minimum ({ring_floor} tokens)")
    arms = H.build_schedule(args.sizes, args.conditions, args.rounds)
    speed_ok = args.rounds >= H.MIN_ROUNDS_FOR_EFFECT
    report.add("cache warming", pf.INFO if speed_ok else pf.WARN,
               ("3 or more reversed rounds: speed effects can be judged against the resolution band, "
                "with the cache state recorded per arm" if speed_ok else H.CACHE_WARNING))
    report.add("schedule", pf.PASS, f"{len(arms)} fresh-process arms: " + ", ".join(a.arm_id for a in arms[:8])
               + (" ..." if len(arms) > 8 else ""))
    report.verdict = "READY" if not report.by_status(pf.FAIL) else "BLOCKED"
    return report


def _load_tokenizer(path):
    from veystrix.backends.flashnext import _load_transformers_tokenizer

    return _load_transformers_tokenizer().from_pretrained(path)


def _free_memory_gb():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(out.split("page size of")[1].split("bytes")[0])
        pages = {}
        for line in out.splitlines():
            label, _, value = line.partition(":")
            if label.strip() in ("Pages free", "Pages inactive", "Pages speculative"):
                pages[label.strip()] = int(value.strip().rstrip("."))
        return sum(pages.values()) * page / 1e9
    except Exception:
        return None


# -- orchestrator -------------------------------------------------------------

def child_command(arm: H.Arm, decode: int, result: Path, extra: list[str] | None = None) -> list[str]:
    return [sys.executable, "-m", MODULE, "--child", "--size", arm.size,
            "--condition", arm.condition, "--decode", str(decode),
            "--arm-id", arm.arm_id, "--round", str(arm.round + 1),
            "--json", str(result)] + (["--reverse"] if arm.reverse else []) + list(extra or [])


def diagnostic_args(args) -> list[str]:
    """The options the children need beyond the arm itself."""
    out = ["--min-available-gb", f"{getattr(args, 'min_available_gb', H.MIN_AVAILABLE_GB):g}",
           "--max-compressor-growth-gib",
           f"{getattr(args, 'max_compressor_growth_gib', H.MAX_COMPRESSOR_GROWTH_GIB):g}",
           "--compressor-stop-available-gib",
           f"{getattr(args, 'compressor_stop_available_gib', H.COMPRESSOR_STOP_AVAILABLE_GIB):g}",
           "--trace-interval", f"{getattr(args, 'trace_interval', 1.0):g}",
           "--fast-trace-interval", f"{getattr(args, 'fast_trace_interval', H.FAST_TRACE_SECONDS):g}"]
    if getattr(args, "swap_ceiling_only", False):
        out.append("--swap-ceiling-only")
    if getattr(args, "expert_pool_gb", None) is not None:
        out += ["--expert-pool-gb", f"{args.expert_pool_gb:g}"]
    if getattr(args, "ring_min_tokens", None) is not None:
        out += ["--ring-min-tokens", str(args.ring_min_tokens)]
    out += ["--requests", getattr(args, "requests", "ref"),
            "--prefix-min-tokens", str(getattr(args, "prefix_min_tokens", 16)),
            "--prefix-ram-gb", f"{getattr(args, 'prefix_ram_gb', 0.75):g}",
            "--prefix-disk-gb", f"{getattr(args, 'prefix_disk_gb', 5.0):g}"]
    for name in ("prefix_dir", "dump_dir", "suite"):
        if getattr(args, name, None):
            out += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
    return out


def diagnostic_overrides(args) -> dict:
    return {"swap_ceiling_only": getattr(args, "swap_ceiling_only", False),
            "expert_pool_gb": getattr(args, "expert_pool_gb", None),
            "ring_min_tokens": getattr(args, "ring_min_tokens", None),
            "start_available_gb": getattr(args, "start_available_gb", None),
            "min_available_gb": getattr(args, "min_available_gb", H.MIN_AVAILABLE_GB),
            "max_compressor_growth_gib": getattr(args, "max_compressor_growth_gib",
                                                 H.MAX_COMPRESSOR_GROWTH_GIB),
            "compressor_stop_available_gib": getattr(args, "compressor_stop_available_gib",
                                                     H.COMPRESSOR_STOP_AVAILABLE_GIB),
            "trace_interval_s": getattr(args, "trace_interval", 1.0),
            "fast_trace_interval_s": getattr(args, "fast_trace_interval", H.FAST_TRACE_SECONDS),
            "requests": getattr(args, "requests", "ref"),
            "prefix_dir": getattr(args, "prefix_dir", None), "dump_dir": getattr(args, "dump_dir", None),
            "suite": getattr(args, "suite", None),
            "prefix_min_tokens": getattr(args, "prefix_min_tokens", 16),
            "prefix_ram_gb": getattr(args, "prefix_ram_gb", 0.75),
            "prefix_disk_gb": getattr(args, "prefix_disk_gb", 5.0)}


def _available_mb() -> float | None:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        return H.parse_vm_stat(out)["available_mb"]
    except Exception:
        return None


def _memory_reading() -> dict:
    """Available memory, compressor size and page cache from one ``vm_stat`` read."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        return H.parse_vm_stat(out)
    except Exception:
        return {}


def child_environment(base: dict, cond: H.Condition, flags: dict | None = None) -> dict:
    return H.apply_overlay(dict(base), H.launch_overlay(cond, flags))


def describe_schedule(arms: list[H.Arm], decode: int, flags: dict | None = None) -> list[str]:
    lines = []
    for arm in arms:
        cond = H.ALL_CONDITIONS[arm.condition]
        overlay = {k: v for k, v in H.launch_overlay(cond, flags).items() if v is not None}
        later = {k: v for k, v in H.condition_overlay(cond, flags).items() if v is not None}
        lines.append(
            f"{arm.index:3d} {arm.arm_id:14s} {'reversed' if arm.reverse else 'forward ':8s} "
            f"turn={cond.turn:6s} launch_flags={overlay or '{}'}"
            + (f" flags_for_turn_2={later or '{}'}" if cond.turn == "second" else "")
        )
    return lines


def fingerprints() -> dict:
    from veystrix.measurement import source_hash

    return {name: source_hash(ROOT / name) for name in FINGERPRINTED if (ROOT / name).is_file()}


def validate_arms(arms: list[dict], *, swap_ceiling_only=False) -> dict:
    """Gates G1 to G5 for the arms collected so far."""
    by_id = {(a["size"], a["condition"], a["round"]): a for a in arms}
    report = {"exactness": H.exactness(arms), "arms": {}, "decode": H.decode_regression(arms)}
    for arm in arms:
        cond = H.ALL_CONDITIONS[arm["condition"]]
        base = by_id.get((arm["size"], H.BASELINES[cond.turn], arm["round"]))
        report["arms"][arm["arm_id"]] = {
            "G2": H.ring_health(arm, cond),
            "G3": {"status": "n/a"} if swap_ceiling_only else H.memory_gate(arm),
            "G4": H.scratch_cleanup(arm, cond, base),
        }
    return report


def start_gate_reason(sample: dict, start_gb) -> str | None:
    """The launch gate (memory required BEFORE an arm starts). It is not the child's stop floor: loading the weights
    alone takes about 3 GB of available memory, so a floor equal to the launch gate stops every arm at load time
    (that is what happened to the first F2Bm launch, 2026-11-08, with --min-available-gb 9)."""
    available = sample.get("available_mb")
    if start_gb and available is None:
        return "available memory is unavailable; launch gate cannot be checked"
    if start_gb and available < start_gb * 1024:
        return f"available memory {available / 1024:.2f} GiB is below the {start_gb:g} GiB launch gate"
    return None


def orchestrate(args, sampler=None, popen=subprocess.Popen, sleep=time.sleep) -> int:
    from veystrix.measurement import MeasurementRun

    arms = H.build_schedule(args.sizes, args.conditions, args.rounds)
    sampler = sampler or H.Sampler(vm=_vm_counters, load=lambda: os.getloadavg()[0],
                                   memory=_memory_reading)
    policy = H.SafetyPolicy(
        final=args.final, min_available_gb=getattr(args, "min_available_gb", H.MIN_AVAILABLE_GB),
        swap_ceiling_only=getattr(args, "swap_ceiling_only", False),
        compressor_limit_mib=getattr(args, "max_compressor_growth_gib",
                                     H.MAX_COMPRESSOR_GROWTH_GIB) * 1024,
        compressor_stop_available_mib=getattr(args, "compressor_stop_available_gib",
                                              H.COMPRESSOR_STOP_AVAILABLE_GIB) * 1024)
    start = sampler.sample()
    watchdog = H.Watchdog(policy, start)
    refusal = watchdog.check(start) or start_gate_reason(start, getattr(args, "start_available_gb", None))
    records = output_path("flashnext", "prefill-memory", "records.jsonl", args.record)
    run = MeasurementRun(records, runtime="flashnext", experiment="prefill-memory", metadata={
        "plan": "docs/flashnext/prefetch-integration.md section 14",
        "preliminary": not args.final or start.get("power_source") != "ac",
        "environment_at_start": start, "sizes": args.sizes, "conditions": args.conditions,
        "rounds": args.rounds, "decode_tokens": args.decode,
        "diagnostic_overrides": diagnostic_overrides(args),
        "diagnostic": any(getattr(args, name, None) is not None
                          for name in ("expert_pool_gb", "ring_min_tokens")),
        "schedule": [a.arm_id for a in arms], "sources": fingerprints(),
        "sampler": "greedy; closed thinking block",
        "speed_evidence": args.rounds >= H.MIN_ROUNDS_FOR_EFFECT,
        "cache_warning": None if args.rounds >= H.MIN_ROUNDS_FOR_EFFECT else H.CACHE_WARNING,
    })
    run.start()
    if args.rounds < H.MIN_ROUNDS_FOR_EFFECT:
        print("NOTE:", H.CACHE_WARNING, flush=True)
    if refusal:
        run.failure(refusal, error_type="safety_stop", phase="preflight")
        run.finish("stopped", reason=refusal)
        print(f"refusing to start: {refusal}", flush=True)
        return 3
    finished: list[dict] = []
    skipped_sizes: set[str] = set()
    current = None
    for arm in arms:
        if arm.size in skipped_sizes:
            continue
        cond = H.ALL_CONDITIONS[arm.condition]
        if current is not None and arm.size != current:
            worst = H.worst_headroom(finished, current)
            if not policy.swap_ceiling_only and worst is not None and worst < H.NEXT_SIZE_HEADROOM_BYTES:
                reason = (f"{current} left {worst / 1e9:.2f} GB of headroom, "
                          f"need {H.NEXT_SIZE_HEADROOM_BYTES / 1e9:.0f} GB for the next size")
                run.failure(reason, error_type="size_gate", arm_id=arm.arm_id)
                skipped_sizes.update(a.size for a in arms if SIZE_ORDER[a.size] >= SIZE_ORDER[arm.size])
                continue
        current = arm.size
        guarded = (cond.guarded and finished and any(
            v["status"] == "fail" for v in _gate_statuses(validate_arms(finished, swap_ceiling_only=policy.swap_ceiling_only))))
        if guarded:
            run.failure("an earlier arm failed a gate; the guarded condition is skipped",
                        error_type="guard", arm_id=arm.arm_id)
            continue
        pre = sampler.sample()
        reason = watchdog.check(pre)
        if reason:
            run.failure(reason, error_type="safety_stop", arm_id=arm.arm_id, phase="before-arm")
            run.finish("stopped", reason=reason, completed_arms=len(finished))
            print(f"stopped before {arm.arm_id}: {reason}", flush=True)
            return 3
        result_path = output_path("flashnext", "prefill-memory", f"arm-{arm.arm_id}.json")
        command = child_command(arm, args.decode, result_path, diagnostic_args(args))
        outcome = H.run_child(
            command, child_environment(dict(os.environ), cond), sampler=sampler,
            watchdog=watchdog, vm_start=pre.get("vm"), arm_start=pre, interval=args.interval,
            timeout=H.TIME_LIMIT_SECONDS[arm.size], popen=popen, sleep=sleep)
        post = sampler.sample()
        if outcome.status != "ok":
            # A stop on a memory rule is a diagnostic stop, not a correctness failure: no
            # output was compared, so nothing is known about exactness for this arm.
            resource_stop = outcome.status in ("stopped", "memory_guard", "exit_137")
            run.failure(outcome.message, arm_id=arm.arm_id, error_type=outcome.status,
                        classification="resource_stop" if resource_stop else outcome.status,
                        returncode=outcome.returncode, before=pre, after=post,
                        watchdog=H.trace_summary(outcome.trace),
                        child_trace=_child_trace_summary(result_path))
            if outcome.status in ("stopped", "timeout", "exit_137", "memory_guard"):
                run.finish("stopped", reason=outcome.message, completed_arms=len(finished),
                           classification="resource_stop" if resource_stop else outcome.status)
                kind = " (diagnostic stop, not a correctness failure)" if resource_stop else ""
                print(f"stopped at {arm.arm_id}: {outcome.message}{kind}", flush=True)
                return 3
            run.finish("failed", reason=outcome.message, completed_arms=len(finished))
            print(f"failed at {arm.arm_id}: {outcome.message}", flush=True)
            return 1
        data = json.loads(result_path.read_text())
        data.update(arm_id=arm.arm_id, round=arm.round, reverse=arm.reverse)
        finished.append(data)
        run.arm(arm_id=arm.arm_id, condition=arm.condition, round_index=arm.round + 1,
                command=command, metrics=_arm_metrics(data), size=arm.size,
                turn=cond.turn, reverse=arm.reverse, order_index=arm.index,
                environment_before=pre, environment_after=post,
                samples=len(outcome.samples), result_file=str(result_path.name),
                watchdog=H.trace_summary(outcome.trace))
        print(f"{arm.arm_id}: done", flush=True)
        partial_report = validate_arms(finished, swap_ceiling_only=policy.swap_ceiling_only)
        failed_gates = [gate for gate in _gate_statuses(partial_report) if gate.get("status") == "fail"]
        if failed_gates:
            for key, result in partial_report["exactness"].items():
                if result["status"] == "fail":
                    run.validation(f"exactness-{key[0]}-{key[1]}", "fail",
                                   mismatches=[list(m) for m in result["mismatches"]])
            run.failure("validation failed", error_type="validation", arm_id=arm.arm_id,
                        failures=failed_gates)
            run.finish("failed", reason="validation failed", completed_arms=len(finished))
            print(f"stopped after {arm.arm_id}: validation failed", flush=True)
            return 1
        sleep(args.settle)
    report = validate_arms(finished, swap_ceiling_only=policy.swap_ceiling_only)
    for arm in finished:
        gates = report["arms"][arm["arm_id"]]
        ok = all(g["status"] in ("pass", "n/a") for g in gates.values())
        run.validation(arm["arm_id"], "pass" if ok else "fail",
                       **{k: v for k, v in gates.items()})
    for key, result in report["exactness"].items():
        run.validation(f"exactness-{key[0]}-{key[1]}", result["status"],
                       mismatches=[list(m) for m in result["mismatches"]])
    effects = H.effects_by_group(finished, EFFECT_METRICS)
    run.finish("completed", arms=len(finished),
               effects={"/".join(k): {kk: vv for kk, vv in v.items() if kk != "differences"}
                        for k, v in effects.items()},
               decode={"/".join(k): v for k, v in report["decode"].items()})
    return 0


SIZE_ORDER = {name: index for index, name in enumerate(sorted(H.ALL_SIZES, key=H.ALL_SIZES.get))}


def _child_trace_summary(result_path: Path) -> dict | None:
    """Per-phase extremes from the child's own 1 s trace file, if it wrote one."""
    path = result_path.parent / f"{result_path.stem}.trace.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return None
    return {"file": path.name, **H.summarize_trace(rows)}


def _vm_counters() -> dict:
    from models.flashnext.diskio import vm_counters

    return vm_counters()


def _gate_statuses(report: dict):
    for gates in report["arms"].values():
        yield from (g for g in gates.values() if g["status"] != "n/a")
    yield from (v for v in report["exactness"].values())


def _arm_metrics(data: dict) -> dict:
    if not data.get("turns"):
        quality = data.get("quality") or {}
        return {
            "quality_suite": quality.get("suite"),
            "passed": sum(1 for c in quality.get("cases", []) if c.get("passed")),
            "total": len(quality.get("cases", [])),
        }
    turn = H.measured_turn(data)
    return {
        "prefilled_tokens": turn["prefilled_tokens"], "prefill_tok_s": turn["prefill_tok_s"],
        "decode_tok_s_1_64": turn["decode"]["tok_s_1_64"],
        "decode_tok_s_65_128": turn["decode"]["tok_s_65_128"],
        "peak_bytes": turn["mem_end_prefill"]["peak"], "digest": turn["digest"],
        "first_token": turn["first_token"], "kv_hash": turn["kv_hash"],
        "ssd_prefill_bytes": turn["ssd_prefill_bytes"], "vm": turn["vm"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", choices=sorted(H.ALL_SIZES, key=H.ALL_SIZES.get), default=["1k"])
    parser.add_argument("--conditions", nargs="+", choices=sorted(H.ALL_CONDITIONS),
                        default=list(H.CONDITIONS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--decode", type=int, default=H.DECODE_TOKENS)
    parser.add_argument("--interval", type=float, default=5.0, help="watchdog sample seconds")
    parser.add_argument("--settle", type=float, default=30.0, help="pause between arms")
    parser.add_argument("--final", action="store_true",
                        help="require AC power; otherwise records are marked preliminary")
    parser.add_argument("--checkpoint", help="checkpoint directory or alias; sets VEYSTRIX_FLASHNEXT_MODEL "
                        "for the children (default: $VEYSTRIX_FLASHNEXT_MODEL, else the single checkpoint in ~/models)")
    parser.add_argument("--check-ready", action="store_true",
                        help="check the checkpoint, backend API, prompt sizes and machine state; "
                             "starts nothing and writes nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the schedule and flags; start nothing, write nothing")
    parser.add_argument("--expert-pool-gb", type=float, default=None,
                        help="diagnostic: size of the decode expert pool in GB instead of the production "
                             "chat default (6). Q4 weights and routing are unchanged")
    parser.add_argument("--ring-min-tokens", type=int, default=None,
                        help="diagnostic: the shortest prompt the ring may serve (production: 512), "
                             "so a short-prompt F2 arm can activate. The memory guard still applies")
    parser.add_argument("--min-available-gb", type=float, default=H.MIN_AVAILABLE_GB,
                        help="stop when memory macOS could reuse falls below this (default %(default)s)")
    parser.add_argument("--start-available-gb", type=float, default=None,
                        help="launch gate in GiB: refuse to start below this much available memory. "
                             "Unlike --min-available-gb it does not stop a running child")
    parser.add_argument("--swap-ceiling-only", action="store_true",
                        help="use 3072 MiB of total system swap as the sole memory stop")
    parser.add_argument("--max-compressor-growth-gib", type=float, default=H.MAX_COMPRESSOR_GROWTH_GIB,
                        help="stop when the compressor grows by this much within one arm, measured "
                             "from the arm's own start (default %(default)s)")
    parser.add_argument("--compressor-stop-available-gib", type=float, default=H.COMPRESSOR_STOP_AVAILABLE_GIB,
                        help="compressor growth over its limit only stops an arm when available memory is "
                             "below this (default %(default)s); growth alone is a recorded warning")
    parser.add_argument("--fast-trace-interval", type=float, default=H.FAST_TRACE_SECONDS,
                        help="child sampling interval during prefill, cheap readings only; 0 = off "
                             "(default %(default)s)")
    parser.add_argument("--trace-interval", type=float, default=1.0,
                        help="seconds between the child's memory trace rows (default %(default)s)")
    parser.add_argument("--requests", default="ref",
                        help="prefix-cache arms (F2P): comma-separated requests run in one process, each a fresh "
                             "conversation: ref (the 1K reference prompt), long:A or long:B (the same long "
                             "system message with question A or B). Default %(default)s")
    parser.add_argument("--prefix-dir", help="prefix-cache directory (F2P). Keep it outside results/ and the repo; "
                                             "reuse a directory to test restoring after a restart")
    parser.add_argument("--dump-dir", help="write the cache after each prefill here for offline comparison "
                                           "(about 150 MB per request); keep it outside results/")
    parser.add_argument("--suite", help="run the task-level quality suite instead of the usual requests: smoke "
                                        "(7 cases), full (about 23) or comma-separated case ids. Greedy, thinking "
                                        "off, one fresh conversation per case; the guards stay on")
    parser.add_argument("--prefix-min-tokens", type=int, default=16,
                        help="shortest system prompt worth a snapshot (default %(default)s; production 512)")
    parser.add_argument("--prefix-ram-gb", type=float, default=0.75, help="prefix-cache RAM budget")
    parser.add_argument("--prefix-disk-gb", type=float, default=5.0, help="prefix-cache SSD budget")
    parser.add_argument("--record", help="measurement JSONL path (inside results/)")
    parser.add_argument("--json", help="child result path (inside results/)")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", choices=sorted(H.ALL_SIZES), help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=sorted(H.ALL_CONDITIONS), help=argparse.SUPPRESS)
    parser.add_argument("--arm-id", help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--reverse", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.swap_ceiling_only and args.start_available_gb:
        parser.error("--swap-ceiling-only cannot be combined with --start-available-gb")
    if args.child:
        if not (args.size and args.condition):
            parser.error("--child needs --size and --condition")
        args.round -= 1
        return child_main(args)
    if args.checkpoint:
        os.environ["VEYSTRIX_FLASHNEXT_MODEL"] = args.checkpoint
    if args.check_ready:
        report = check_ready(args)
        print(report.text())
        return 0 if report.verdict == "READY" else 1
    if args.dry_run:
        arms = H.build_schedule(args.sizes, args.conditions, args.rounds)
        print(f"{len(arms)} fresh-process arms; sizes {args.sizes}; rounds {args.rounds}")
        print("diagnostic overrides:", diagnostic_overrides(args))
        print("\n".join(describe_schedule(arms, args.decode)))
        print("example command:", " ".join(
            child_command(arms[0], args.decode, Path("<results>/arm.json"), diagnostic_args(args))))
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
