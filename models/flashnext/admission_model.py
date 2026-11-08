"""Measured admission model: how far the gate figure falls when Veystrix runs, and what a purge can give back.

Every number here is derived from the preserved real-checkpoint traces (``tests/bench/fixtures/admission_traces.json``,
refreshed by ``tests/bench/admission_audit.py``) and is replayed against them by ``tests/unit/test_admission_model.py``.
Nothing here loads a model, reads macOS or changes a runtime guard.

What the evidence says (seven real runs, 2026-11-08; report in docs/flashnext/admission-audit-2026-11-09.md):

* The gate figure (free + inactive + speculative) falls by 6.1 to 6.65 GiB between launch and its lowest point in EVERY run,
  whether the run was CHAT with a 60-token prompt, an AGENTIC 902-token prefill, a 3,935-token prefill or an 8K prefill.
  The prefill length moves the MLX peak (4.11 GiB at 902 tokens, 5.06 at 3,935, 5.29 at 8,200) but the figure barely follows.
* The two client-free and client-present runs differ by 0.19 GiB, the DeepSeek Harness headless footprint. So the fall is
  ``ENGINE_DROP`` for the engine plus the client that is started after the server.
* Other processes' anonymous memory does NOT predict the compressor: with 2.69 GiB of other anonymous memory the compressor
  grew by 3.15 GiB; with 6.31 GiB it grew by 5.5 GiB. The engine's own churn dominates. The old budget (wired + compressor +
  peak + 1 GiB + 0.75 x anonymous >= 4 GiB headroom) therefore predicts nothing the gate figure does not, and it re-adds the
  runtime floor on top of a gate that already includes it.

The single rule that follows:

    predicted lowest gate figure = start figure - drop(tokens, client)
    admit when predicted lowest >= FLOOR + MARGIN        (and pressure is normal, no swap-out)

``FLOOR`` is the runtime guard's own stop line (ChatGuard stops below 2 GiB available); it is a number the guard enforces,
not a second margin. The required start figure is never below the long-standing 9 GiB gate.
"""
from __future__ import annotations

GIB = 2 ** 30

FLOOR_GIB = 2.0            # ChatGuard stops when available falls below this (chat_guard.py); admission must leave it intact
MARGIN_GIB = 0.30          # what is left above the floor: covers other applications growing during the run
ENGINE_DROP_GIB = 6.46     # CHAT run, no client: 9.70 GiB at start, 3.24 GiB lowest
CLIENT_GIB = 0.25          # measured clients: DeepSeek Harness headless 0.19 GiB, Qwen Code headless CLI 0.21 GiB peak RSS
MANDATORY_GATE_GIB = 9.0   # unchanged long-standing start gate: the required figure is never lower
ANONYMOUS_CAUTION_GIB = 6.0        # the only memory-driven failure on record had 6.31 GiB of other anonymous memory
ANONYMOUS_CAUTION_EXTRA_GIB = 0.5

# (cold-prefill tokens, MLX peak GiB) from the current code's real runs. The peak is what grows with the prefill.
MLX_PEAK_POINTS = ((902, 4.11), (3935, 5.06), (8200, 5.29))
LOAD_DROP_GIB = 3.63       # fall of the gate figure while the checkpoint loads, before any prefill (largest of seven runs)
MLX_REFERENCE_TOKENS = 3935      # the reference run for the drop: 6.65 GiB with the harness client present
BEYOND_MEASURED_GIB_PER_K = 0.12  # above 8,200 tokens, unmeasured: KV 0.03 + attention scratch 0.05 + allocator 0.04

# Real purges (nine of them, results/flashnext/*/purge-log.jsonl) returned 1.4 to 3.1 GiB, median 2.0.
PURGE_GAIN_OBSERVED_MIN_GIB = 1.4
PURGE_GAIN_OBSERVED_MEDIAN_GIB = 2.0


def mlx_peak_gib(tokens: int) -> float:
    """MLX peak for a cold prefill of ``tokens``: interpolated inside the measured range, extrapolated above it."""
    tokens = max(0, int(tokens))
    points = MLX_PEAK_POINTS
    if tokens <= points[0][0]:
        return points[0][1]
    for (t0, m0), (t1, m1) in zip(points, points[1:]):
        if tokens <= t1:
            return m0 + (m1 - m0) * (tokens - t0) / (t1 - t0)
    t_last, m_last = points[-1]
    return m_last + BEYOND_MEASURED_GIB_PER_K * (tokens - t_last) / 1024


def drop_gib(tokens: int, client_gib: float = CLIENT_GIB) -> float:
    """Predicted fall of the gate figure from launch to its lowest point. Never below the reference run's fall."""
    growth = max(0.0, mlx_peak_gib(tokens) - mlx_peak_gib(MLX_REFERENCE_TOKENS))
    return ENGINE_DROP_GIB + client_gib + growth


def required_available_gib(tokens: int, client_gib: float = CLIENT_GIB, other_anonymous_gib: float = 0.0) -> float:
    """The start gate figure a launch needs for a first cold prefill of ``tokens``."""
    need = drop_gib(tokens, client_gib) + FLOOR_GIB + MARGIN_GIB
    if other_anonymous_gib >= ANONYMOUS_CAUTION_GIB:
        need += ANONYMOUS_CAUTION_EXTRA_GIB
    return max(MANDATORY_GATE_GIB, need)


def required_available_bytes(tokens: int, client_gib: float = CLIENT_GIB, other_anonymous_bytes: int = 0) -> int:
    return int(required_available_gib(tokens, client_gib, other_anonymous_bytes / GIB) * GIB)


def predicted_lowest_gib(start_gib: float, tokens: int, client_gib: float = CLIENT_GIB) -> float:
    return start_gib - drop_gib(tokens, client_gib)


def expected_purge_gain_bytes(purge_gain_upper: int) -> int:
    """What admission may count on from a purge: the smallest gain ever observed, never more than the accounting's upper bound.

    ``purge_gain_upper`` is the file cache not already in the gate figure (purge_recovery.accounting). Real gains were 1.4 to
    3.1 GiB, so counting the minimum keeps the plan honest; the recovery still re-reads the machine after every purge."""
    return int(min(int(purge_gain_upper), PURGE_GAIN_OBSERVED_MIN_GIB * GIB))


def stage_lowest_gib(start_gib: float, tokens: int, client_gib: float = CLIENT_GIB) -> dict:
    """Predicted lowest gate figure per stage. Loading takes about 3.6 GiB; the first cold prefill and decoding then take the
    rest, up to the full drop (in the seven runs the prefill phase alone reached 6.1-6.5 GiB of it, decode added at most 0.5)."""
    return {"after load": start_gib - LOAD_DROP_GIB,
            "during first cold prefill and decode": start_gib - drop_gib(tokens, client_gib)}
