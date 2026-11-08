"""The owner's explicit memory waiver (2026-11-08): "launch the run anyway, only stop if it hits 2.5 GB swap".

Opt-in only (``FLASHNEXT_MEMORY_WAIVER=swap-only``), never a default, never set by a mode, printed in the server banner,
recorded in the admission log and the evidence summary. It waives MEMORY THRESHOLDS and nothing else:

waived     the AGENTIC admission refusal (the tiered headroom rule) and the 9 GiB start gate; the available-memory floor;
           compressor growth with low availability; macOS pressure level 4 and persistent level 2; the active swap-out and
           swap-in rate stops. These are still measured, traced and recorded in the controller ledger.
kept       the 2.5 GiB swap-growth hard stop over the session baseline (the owner's limit), the MLX-versus-Metal-limit check,
           a missing or unreadable probe (fail closed), battery floor and Low Power Mode, thermal throttling, the out-of-process
           stall watchdog, and every GPU synchronization rule.
"""
from __future__ import annotations

import os

FLAG = "FLASHNEXT_MEMORY_WAIVER"
SWAP_ONLY = "swap-only"


def active(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get(FLAG, "")
    if value not in ("", "0", SWAP_ONLY):
        raise ValueError(f"{FLAG}={value!r}: the only waiver is {SWAP_ONLY!r}")
    return value == SWAP_ONLY


def banner() -> str:
    return ("MEMORY WAIVER ACTIVE (owner decision): admission and memory-pressure stops are waived; the run stops only on "
            "swap growth of 2.5 GiB over the session baseline, a missing probe, battery/thermal, the stall watchdog or the "
            "Metal limit. Measurements are still recorded.")
