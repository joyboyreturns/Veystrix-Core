"""Per-request memory readiness for the streamed Nemotron backend: look, recover what is safe to recover, look again.

Admission (``admission.py``) decides what the session can ever hold. This module decides, at the start of each request,
whether the machine has room right now, and does the few safe things that can make room, in order:

1. release Veystrix-owned temporary memory (the resident expert pool and the MLX allocator cache; both refill from SSD);
2. a conditional macOS file-cache purge through ``purge_recovery`` and only that: ``sudo -n`` permission required, no
   password prompt ever, never overlapping a request, backs off after an ineffective purge, skipped when the file cache
   cannot close the gap;
3. a settle wait and a fresh reading.

A request proceeds when the machine has the room, or when it is merely short of it without memory pressure (the pool
was already released, and decode runs with a small footprint). It is refused (retriable) only when it is short **and**
macOS reports pressure. Nothing here kills or signals a process, closes an application, deletes a file or weakens a guard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .admission import ACTIVATION_BYTES_PER_CHUNK_TOKEN, AdmissionRefused

GIB = 2 ** 30
HEADROOM_BYTES = GIB                      # always keep this much available beyond what the request itself may grow by
MAX_POOL_GROWTH_BYTES = GIB               # how much of an unfilled pool one request is assumed to fill
SETTLE_SECONDS = 2.0


@dataclass
class Step:
    name: str
    action: str                            # done | nothing | skipped
    detail: str
    available_before: int = 0
    available_after: int = 0

    def as_dict(self) -> dict:
        return {"name": self.name, "action": self.action, "detail": self.detail,
                "available_before_gib": round(self.available_before / GIB, 2),
                "available_after_gib": round(self.available_after / GIB, 2)}


@dataclass
class Report:
    admitted: bool
    required_bytes: int
    available_bytes: int
    pressure: int
    steps: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"admitted": self.admitted, "required_gib": round(self.required_bytes / GIB, 2),
                "available_gib": round(self.available_bytes / GIB, 2), "pressure": self.pressure,
                "steps": [s.as_dict() for s in self.steps]}


def required_bytes(plan, resident_pool_bytes: int) -> int:
    """Room one request may need on top of what is resident: activations, pool growth (bounded) and headroom."""
    unfilled = max(0, plan.pool_bytes - resident_pool_bytes)
    return plan.chunk_tokens * ACTIVATION_BYTES_PER_CHUNK_TOKEN + min(unfilled, MAX_POOL_GROWTH_BYTES) + HEADROOM_BYTES


def ensure_ready(read, *, plan, resident_pool_bytes: int, reclaim=None, purge=None, sleep=time.sleep) -> Report:
    """``read()`` -> admission.Machine. ``reclaim()`` -> bytes freed. ``purge(required)`` -> (action, detail).

    Returns the report; raises ``AdmissionRefused`` (retriable) when short of memory under pressure after recovery."""
    need = required_bytes(plan, resident_pool_bytes)
    machine = read()
    report = Report(True, need, machine.available, machine.pressure)
    if machine.available >= need and machine.pressure < 2:
        return report

    if reclaim is not None:
        before = machine.available
        try:
            freed = int(reclaim() or 0)
        except Exception as error:                       # recovery is best effort; it never fails a request itself
            report.steps.append(Step("owned memory", "skipped", f"{type(error).__name__}: {error}", before, before))
        else:
            machine = read()
            report.steps.append(Step("owned memory", "done" if freed else "nothing",
                                     f"released {freed / GIB:.2f} GiB (expert pool, allocator cache)",
                                     before, machine.available))
    if machine.available < need and purge is not None and machine.pressure < 4:
        before = machine.available
        try:
            action, detail = purge(need)
        except Exception as error:
            action, detail = "skipped", f"{type(error).__name__}: {error}"
        if action == "purged":
            sleep(SETTLE_SECONDS)
        machine = read()
        report.steps.append(Step("file-cache purge", "done" if action == "purged" else
                                 "nothing" if action == "ok" else "skipped", detail, before, machine.available))
    report.available_bytes, report.pressure = machine.available, machine.pressure
    if machine.available < need and machine.pressure >= 2:
        report.admitted = False
        tried = "; ".join(f"{s.name}: {s.detail}" for s in report.steps) or "no recovery was available"
        raise AdmissionRefused(
            f"not enough free memory right now ({machine.available / GIB:.2f} GiB available, about {need / GIB:.2f} GiB "
            f"needed) and macOS reports memory pressure level {machine.pressure}. Tried: {tried}. "
            f"Close large apps or retry shortly.")
    return report
