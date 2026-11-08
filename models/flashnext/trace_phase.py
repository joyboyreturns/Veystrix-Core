"""One shared slot for the safety trace's current phase (prefill, decode, idle).

``ChatGuard`` registers its tracer's ``Phase`` here; the backend sets it around a request. Without a registered
tracer every call is a no-op, so it costs nothing outside a guarded chat.
"""
from __future__ import annotations

PHASE = [None]


def register(phase) -> None:
    PHASE[0] = phase


def set_phase(name: str) -> None:
    phase = PHASE[0]
    if phase is not None:
        phase.set(name)
    from models.flashnext import progress

    if progress.PROGRESS.active and name in progress.PHASE_STAGE:
        progress.PROGRESS.stage(progress.PHASE_STAGE[name])
