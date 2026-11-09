"""Refuse to serve without the safety supervisor.

Root cause of the 2026-11-09 DeepSeek Harness incident (see docs/flashnext/deepseek-swap-incident-2026-11-09.md): the server
was started with ``macqwen/session.py --server`` directly. Everything that makes the AGENTIC server safe (ChatGuard and its
+2.5 GiB swap stop, 2K chunked prefill, the prefix cache, the stall watchdog) is switched on by the launcher
(``./chat.sh --mode agentic``, i.e. ``macqwen/cli.py`` -> ``settings.modes.apply_mode``), not by ``session.py``. A direct launch
therefore ran with no supervisor at all, prefilled an ~8.9K-token DeepSeek request in one call, and nothing could stop it.

``check`` is pure so it can be tested; ``session.py`` calls it before any model loads and exits with ``EXIT_CODE``.
"""
from __future__ import annotations

EXIT_CODE = 70                       # EX_SOFTWARE-adjacent; distinct from the guard (GUARD_EXIT_CODE) and admission codes
REQUIRED = {
    "VEYSTRIX_MODE": "agentic",
    "FLASHNEXT_CHAT_GUARD": "1",
    "FLASHNEXT_PREFILL_CHUNK_TOKENS": None,       # any non-zero value
}
LAUNCH_COMMAND = ("./chat.sh --mode agentic --pool-gb 0 --context-limit 262144 "
                  "--max-cold-prefill 32768 --max-incremental-prefill 32768")


def check(environ: dict, *, model: str, server: bool, benchmarking: bool = False) -> str | None:
    """None when the launch is supervised (or is not a flashnext server); otherwise the refusal message."""
    if model != "flashnext" or not server or benchmarking:
        return None
    problems = []
    if environ.get("VEYSTRIX_MODE") != "agentic":
        problems.append("VEYSTRIX_MODE is not 'agentic' (the mode launcher did not run)")
    if environ.get("FLASHNEXT_CHAT_GUARD") != "1":
        problems.append("FLASHNEXT_CHAT_GUARD is not 1 (the swap-growth supervisor would not run)")
    chunk = environ.get("FLASHNEXT_PREFILL_CHUNK_TOKENS", "0")
    if chunk in ("", "0"):
        problems.append("FLASHNEXT_PREFILL_CHUNK_TOKENS is unset or 0 (a prompt would prefill in ONE call, unbounded memory)")
    if not problems:
        return None
    return ("refusing to start the server unsupervised:\n  - " + "\n  - ".join(problems) +
            "\nStart it through the launcher, which applies the safety configuration:\n  " + LAUNCH_COMMAND +
            "\nNothing was loaded.")
