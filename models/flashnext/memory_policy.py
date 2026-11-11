"""Approved memory policy for Veystrix AGENTIC runtime: swap-growth-only.

Explicit owner decision (2026-11-09):
- The default AGENTIC memory policy is swap-growth-only.
- Startup 9 GiB gate, measured-available-memory admission threshold, and low-available-memory or
  memory-pressure stops are removed (demoted to telemetry-only).
- No memory-waiver environment variable is required for normal startup.
- Telemetry monitors available RAM, compressor, pressure, MLX allocation, and swap.
- The ONLY measurement-based memory stop is a swap increase of 2.5 GiB above the launch baseline.
- Baseline is captured at startup, not against zero system swap.
- High-watermark behavior prevents transient swap decreases from bypassing the threshold.
- Missing or invalid swap probe fails closed with a clear diagnostic.
- Stop crossing halts work admission and initiates safe GPU cancellation without automatic retries.
- All non-memory protections (battery, Low Power Mode, thermal, Metal limits, watchdog) are retained.

DISCLAIMER: The swap-only policy does not guarantee protection against sudden OOM, GPU allocation
failure, compressor growth, or degraded responsiveness.
"""
from __future__ import annotations

import os
from typing import Any

POLICY_SWAP_GROWTH_ONLY = "swap-growth-only"
POLICY_STRICT_STAGED = "strict-staged"
POLICY_ABSOLUTE_SWAP_ONLY = "absolute-swap-only"
ABSOLUTE_SWAP_CEILING_MIB = 3072
FLAG_POLICY = "FLASHNEXT_MEMORY_POLICY"
DEFAULT_SWAP_GROWTH_THRESHOLD_GIB = 2.5

TELEMETRY_ONLY_CHECKS = (
    "startup_9gib_gate",
    "measured_available_memory_admission",
    "available_memory_floor_2gib",
    "compressor_growth_stop",
    "pressure_level_4_stop",
    "persistent_pressure_level_2_stop",
    "active_swap_rate_stop",
)

DISCLAIMER = (
    "The swap-growth-only policy does not guarantee protection against sudden OOM, "
    "GPU allocation failure, compressor growth, or degraded responsiveness."
)


def is_swap_growth_policy(environ: dict[str, str] | None = None) -> bool:
    """True when the current session operates under the approved swap-growth-only policy."""
    env = os.environ if environ is None else environ
    configured = env.get(FLAG_POLICY, "").strip().lower()
    if configured == POLICY_SWAP_GROWTH_ONLY:
        return True
    if configured in (POLICY_STRICT_STAGED, POLICY_ABSOLUTE_SWAP_ONLY):
        return False
    # Backward compatibility with historical waiver flag
    if env.get("FLASHNEXT_MEMORY_WAIVER", "").strip().lower() == "swap-only":
        return True
    # Default for AGENTIC mode: swap-growth-only is the standard policy
    mode = env.get("VEYSTRIX_MODE", "").strip().lower()
    return mode == "agentic"


def is_absolute_swap_policy(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(FLAG_POLICY, "").strip().lower() == POLICY_ABSOLUTE_SWAP_ONLY


def policy_identifier(environ: dict[str, str] | None = None) -> str:
    if is_absolute_swap_policy(environ):
        return POLICY_ABSOLUTE_SWAP_ONLY
    return POLICY_SWAP_GROWTH_ONLY if is_swap_growth_policy(environ) else POLICY_STRICT_STAGED


def telemetry_only_checks(environ: dict[str, str] | None = None) -> list[str]:
    if is_swap_growth_policy(environ) or is_absolute_swap_policy(environ):
        return list(TELEMETRY_ONLY_CHECKS)
    return []


def policy_summary(environ: dict[str, str] | None = None) -> dict[str, Any]:
    absolute = is_absolute_swap_policy(environ)
    active = is_swap_growth_policy(environ) or absolute
    limit = ABSOLUTE_SWAP_CEILING_MIB if absolute else DEFAULT_SWAP_GROWTH_THRESHOLD_GIB * 1024
    return {
        "policy_identifier": policy_identifier(environ),
        "swap_baseline_mb": None,
        "current_swap_growth_mb": 0.0,
        "peak_swap_growth_mb": 0.0,
        "threshold_gib": limit / 1024,
        "threshold_mb": limit,
        "stop_reason": None,
        "can_admit": True,
        "probe_availability": {},
        "telemetry_only_checks": list(TELEMETRY_ONLY_CHECKS) if active else [],
        "disclaimer": DISCLAIMER if active else "",
    }


def banner(environ: dict[str, str] | None = None) -> str:
    """Human-readable startup banner describing the memory policy."""
    if is_absolute_swap_policy(environ):
        return f"MEMORY POLICY: {POLICY_ABSOLUTE_SWAP_ONLY} (stop at 3 GiB total system swap)"
    if is_swap_growth_policy(environ):
        return (
            f"MEMORY POLICY: {POLICY_SWAP_GROWTH_ONLY} (approved AGENTIC policy)\n"
            "  * 9 GiB gate & available-memory admission thresholds are demoted to telemetry-only.\n"
            f"  * Hard stop: swap growth > {DEFAULT_SWAP_GROWTH_THRESHOLD_GIB:g} GiB above launch baseline "
            "(high-watermark enforced; fails closed on missing/invalid probe).\n"
            "  * Safe GPU cancellation on stop; no automatic retry.\n"
            f"  * Notice: {DISCLAIMER}"
        )
    return (
        f"MEMORY POLICY: {POLICY_STRICT_STAGED} (strict available-memory and pressure gates active)"
    )
