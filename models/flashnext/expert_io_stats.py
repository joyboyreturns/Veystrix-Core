"""Opt-in diagnostics and accounting for MoE expert I/O and decode pool reuse.

Tracks expert requests, pool hits, pool misses, SSD reads, bytes read, and latency.
Enabled via FLASHNEXT_EXPERT_IO_DIAGNOSTICS=1 or when profiling.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

FLAG_DIAGNOSTICS = "FLASHNEXT_EXPERT_IO_DIAGNOSTICS"
FLAG_POOL_REUSE = "FLASHNEXT_SHORT_PREFILL_POOL_REUSE"


def diagnostics_enabled() -> bool:
    return os.environ.get(FLAG_DIAGNOSTICS, "0") == "1"


def pool_reuse_enabled() -> bool:
    return os.environ.get(FLAG_POOL_REUSE, "0") == "1"


@dataclass
class ExpertIOStats:
    expert_requests: int = 0
    unique_requested: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    ssd_reads: int = 0
    ssd_bytes_read: int = 0
    pool_bytes_saved: int = 0
    io_wait_seconds: float = 0.0
    pool_copy_seconds: float = 0.0
    layers_evaluated: int = 0

    def hit_rate(self) -> float:
        total = self.pool_hits + self.pool_misses
        return (self.pool_hits / total) if total > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "expert_requests": self.expert_requests,
            "unique_requested": self.unique_requested,
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "ssd_reads": self.ssd_reads,
            "ssd_bytes_read": self.ssd_bytes_read,
            "pool_bytes_saved": self.pool_bytes_saved,
            "io_wait_seconds": round(self.io_wait_seconds, 4),
            "pool_copy_seconds": round(self.pool_copy_seconds, 6),
            "layers_evaluated": self.layers_evaluated,
        }


_STATS = ExpertIOStats()
_LOCK = threading.Lock()


def get_stats() -> ExpertIOStats:
    with _LOCK:
        return ExpertIOStats(**_STATS.as_dict())


def reset_stats() -> None:
    global _STATS
    with _LOCK:
        _STATS = ExpertIOStats()


def record_request(num_experts: int, hits: int, misses: int, ssd_bytes: int = 0,
                   saved_bytes: int = 0, io_s: float = 0.0, copy_s: float = 0.0) -> None:
    with _LOCK:
        _STATS.expert_requests += num_experts
        _STATS.unique_requested += (hits + misses)
        _STATS.pool_hits += hits
        _STATS.pool_misses += misses
        if misses > 0:
            _STATS.ssd_reads += 1
        _STATS.ssd_bytes_read += ssd_bytes
        _STATS.pool_bytes_saved += saved_bytes
        _STATS.io_wait_seconds += io_s
        _STATS.pool_copy_seconds += copy_s
        _STATS.layers_evaluated += 1
