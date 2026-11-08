"""Memory-aware prefill chunk control for Flash-Next (Apple Silicon M4, 16 GB).

Two separate jobs, deliberately kept apart because they have different determinism needs:

* PLAN (deterministic): ``prefill.budgeted_edges`` cuts a prompt from a byte budget and the positions alone, so a cold
  run and a run restored from a prefix snapshot take the same chunks and stay bit-identical. Nothing live goes in.
* GUARD (live, emergency only): between chunks this scheduler reads host headroom and pressure and may step the chunk
  cap DOWN (and back up after sustained safety). Every deviation is recorded in the ``PlanTrace`` and any snapshot
  taken afterwards carries the chunks that really ran, so it can never be mistaken for the cold plan's state.

Memory model. While a prefill runs the weights, scratch ring and allocator are already resident, so host "available"
already excludes them. ``estimate_chunk_increment`` therefore counts only what a chunk ADDS. ``estimate_chunk_memory``
is the pre-load view (a full footprint against free memory) used for admission, not for in-run decisions.

Enabled with ``FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS=1`` and only acts under the "budgeted" chunk policy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from models.flashnext.prefill_budget import (
    ALLOCATOR_CACHE_PEAK,
    HYPER_TRANSIENT,
    MOE_TRANSIENT,
    QSA_QUERY_CHUNK,
    SCRATCH_RING,
    SDPA_PAIR_BYTES,
    WEIGHTS,
    chunk_increment_bytes,
    state_and_kv,
)

FLAG_ADAPTIVE = "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS"
FLAG_MIN_CHUNK = "FLASHNEXT_PREFILL_MIN_CHUNK"
FLAG_MAX_CHUNK = "FLASHNEXT_PREFILL_MAX_CHUNK"

CANDIDATE_CHUNKS = (2048, 1024, 512, 256)
DEFAULT_MIN_CHUNK = 256
DEFAULT_MAX_CHUNK = 2048
GUARD_FLOOR_BYTES = int(3.0 * 1024 ** 3)       # the runtime guard stops near here; never lowered to admit work
COMFORT_MARGIN_BYTES = int(1.0 * 1024 ** 3)    # used by the pre-load estimate only; no longer a runtime step-down trigger
CRITICAL_AVAILABLE_BYTES = int(2.0 * 1024 ** 3)   # runtime step-down: available memory below this
CRITICAL_PRESSURE_LEVEL = 4                       # runtime step-down: macOS critical pressure


def adaptive_enabled() -> bool:
    return os.environ.get(FLAG_ADAPTIVE, "0") == "1"


@dataclass(frozen=True)
class ChunkEstimate:
    chunk_tokens: int
    transient_bytes: int
    attention_score_bytes: int
    scratch_ring_bytes: int
    allocator_cache_bytes: int
    total_peak_bytes: int
    expected_headroom_bytes: int
    fits_comfortably: bool


def estimate_chunk_memory(
    chunk_tokens: int,
    total_tokens: int,
    available_bytes: int,
    ring_active: bool = True,
    cache_cap_bytes: Optional[int] = 512 * 1024 * 1024,
) -> ChunkEstimate:
    """PRE-LOAD view: the whole footprint of a prefill with this chunk size against memory that is still free.

    Not for use while the model is resident: ``available_bytes`` then already excludes the weights and the ring, and
    counting them again would reject every chunk. Use ``estimate_chunk_increment`` in a running process."""
    live = min(chunk_tokens, total_tokens)
    transient = int((HYPER_TRANSIENT + MOE_TRANSIENT) * live)
    scores = int(min(live, QSA_QUERY_CHUNK) * total_tokens * SDPA_PAIR_BYTES)
    scratch = int(SCRATCH_RING) if ring_active else 0
    allocator = int(cache_cap_bytes if cache_cap_bytes is not None else ALLOCATOR_CACHE_PEAK)
    persistent = int(state_and_kv(total_tokens))
    total_peak = int(WEIGHTS + scratch + persistent + transient + scores + allocator)
    headroom = available_bytes - total_peak
    return ChunkEstimate(chunk_tokens, transient, scores, scratch, allocator, total_peak, headroom,
                         headroom >= GUARD_FLOOR_BYTES + COMFORT_MARGIN_BYTES)


def estimate_chunk_increment(chunk_tokens: int, position: int, available_bytes: int) -> tuple[int, int, bool]:
    """IN-RUN view: ``(extra_bytes, headroom_after, fits_comfortably)`` for one chunk at ``position`` cached tokens."""
    extra = chunk_increment_bytes(chunk_tokens, position)
    headroom = int(available_bytes) - extra
    return extra, headroom, headroom >= GUARD_FLOOR_BYTES + COMFORT_MARGIN_BYTES


def plan_worst_case_bytes(edges: Sequence[int], end: int, start: int = 0) -> int:
    """Largest per-chunk increment over a plan's chunks (the number admission compares with live headroom)."""
    points = [int(start)] + [int(e) for e in edges if start < e < end] + [int(end)]
    return max((chunk_increment_bytes(b - a, a) for a, b in zip(points, points[1:]) if b > a), default=0)


class AdaptivePrefillScheduler:
    """Emergency chunk-cap control between chunks, with hysteresis. Holds no GPU state."""

    def __init__(
        self,
        candidate_chunks: Sequence[int] = CANDIDATE_CHUNKS,
        min_chunk: int = DEFAULT_MIN_CHUNK,
        max_chunk: int = DEFAULT_MAX_CHUNK,
        system_reader: Optional[Callable[[], dict]] = None,
    ):
        self.candidate_chunks = tuple(sorted([c for c in candidate_chunks if min_chunk <= c <= max_chunk], reverse=True))
        if not self.candidate_chunks:
            self.candidate_chunks = (min_chunk,)
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self._system_reader = system_reader
        self._consecutive_safe_chunks = 0
        self.current_chunk_size = self.candidate_chunks[0]

    def read_system(self) -> Optional[dict]:
        """``{"available": bytes, "pressure": level}`` or None when the machine cannot be read (then nothing adapts)."""
        try:
            if self._system_reader is not None:
                data = dict(self._system_reader())
                return {"available": int(data["available"]), "pressure": int(data.get("pressure", 1))}
            from models.flashnext import purge_recovery as P
            reading = P.read_system()
            return {"available": int(reading.available), "pressure": int(reading.pressure)}
        except Exception:
            return None

    def start_at(self, cap: int) -> None:
        """Begin from the plan's chunk cap (the nearest candidate not above it)."""
        below = [c for c in self.candidate_chunks if c <= cap]
        self.current_chunk_size = below[0] if below else self.candidate_chunks[-1]
        self._consecutive_safe_chunks = 0

    def select_initial_chunk(self, total_tokens: int, position: int = 0, available_bytes: Optional[int] = None) -> int:
        """Largest chunk whose worst-case increment (the last chunk of the prompt, at its longest context) fits the
        live headroom above the guard floor and comfort margin. Informational: the production plan is deterministic."""
        if available_bytes is None:
            data = self.read_system()
            if data is None:
                self.current_chunk_size = self.min_chunk
                return self.min_chunk
            available_bytes = data["available"]
        for c in self.candidate_chunks:
            worst_position = max(position, total_tokens - c)
            if estimate_chunk_increment(c, worst_position, available_bytes)[2]:
                self.current_chunk_size = c
                return c
        self.current_chunk_size = self.candidate_chunks[-1]
        return self.current_chunk_size

    def observe_and_adapt(self, available_bytes: int, pressure_level: int = 1, elevated: bool = False) -> int:
        """Cap for the next chunk. Steps down ONLY on a critical reading (macOS pressure level 4, or available memory under the
        guard floor); steps back up after one safe chunk.

        Why not on moderate pressure or a low comfort margin (the rule before 2026-11-09): this engine streams every routed
        expert from SSD once per chunk, so a chunk costs about the same wall time and SSD traffic whether it holds 1,024 tokens
        or 256. The real 8K run (results/flashnext/20261108-232323-gate-8k-attempt1) hovered at 3.5-4.5 GiB available, which
        the old 4 GiB comfort rule read as "low" at almost every chunk edge, collapsing the plan to 256-512-token chunks: 13
        chunk passes instead of 9 (replayed offline from the trace), 423 GiB paged in against about 290 GiB for the plan, and
        no relief for the pressure, which comes from the page cache and compressor, not from activation memory (a 1,024-token
        chunk adds 0.5-0.8 GiB). A smaller chunk only helps when the MLX allocation itself is the risk.

        ``elevated`` (the staged memory controller) is accepted and deliberately has no effect on the size, for the same reason."""
        idx = (self.candidate_chunks.index(self.current_chunk_size)
               if self.current_chunk_size in self.candidate_chunks else len(self.candidate_chunks) - 1)
        critical = pressure_level >= CRITICAL_PRESSURE_LEVEL or available_bytes < CRITICAL_AVAILABLE_BYTES
        if critical:
            if idx < len(self.candidate_chunks) - 1:
                self.current_chunk_size = self.candidate_chunks[idx + 1]
            self._consecutive_safe_chunks = 0
            return self.current_chunk_size
        self._consecutive_safe_chunks += 1
        if idx > 0 and self._consecutive_safe_chunks >= 1:
            self.current_chunk_size = self.candidate_chunks[idx - 1]
            self._consecutive_safe_chunks = 0
        return self.current_chunk_size

    def observe(self) -> Optional[int]:
        """Read the machine and adapt; None when it cannot be read."""
        data = self.read_system()
        if data is None:
            return None
        return self.observe_and_adapt(data["available"], data["pressure"])
