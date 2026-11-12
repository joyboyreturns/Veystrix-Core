"""The memory waiver and policy bridge.

Retained for backward compatibility with existing tests and scripts.
The approved policy is defined in ``models.flashnext.memory_policy``.
"""
from __future__ import annotations

import os
from . import memory_policy

FLAG = "FLASHNEXT_MEMORY_WAIVER"
SWAP_ONLY = "swap-only"


def active(environ=None) -> bool:
    env = os.environ if environ is None else environ
    value = env.get(FLAG, "")
    if value not in ("", "0", SWAP_ONLY):
        raise ValueError(f"{FLAG}={value!r}: the only waiver is {SWAP_ONLY!r}")
    if value == SWAP_ONLY:
        return True
    return memory_policy.is_swap_growth_policy(environ) or memory_policy.is_absolute_swap_policy(environ)


def banner() -> str:
    return memory_policy.banner()
