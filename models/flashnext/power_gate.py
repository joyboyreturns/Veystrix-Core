"""Power start gate: running on battery is allowed, a low battery is not.

Owner decision 2026-11-10: the engine may start when the Mac is not plugged in, but it refuses to start (and the
supervisor refuses to restart it) when the battery is below ``VEYSTRIX_MIN_BATTERY_PERCENT`` (default 25). On AC power
this gate always passes. It reads ``pmset -g batt`` only; it never loads a model and never touches the GPU.

Everything else that protects the machine is unchanged: the in-run watchdog (``chat_guard``) already stops a run
whose battery falls below the same floor, and Low Power Mode, thermal, memory and swap stops are untouched.
Setting ``VEYSTRIX_MIN_BATTERY_PERCENT=0`` removes the floor for a run (the 2026-11-09 owner decision, kept).
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

MIN_BATTERY_ENV = "VEYSTRIX_MIN_BATTERY_PERCENT"
DEFAULT_MIN_BATTERY_PERCENT = 25
POWER_EXIT_CODE = 72          # not retried by the supervisor: restarting cannot fix a flat battery


@dataclass(frozen=True)
class Verdict:
    ok: bool
    message: str
    source: str                # "ac" | "battery" | "unknown"
    percent: int | None
    floor: int


def minimum_percent(environ=None) -> int:
    raw = (os.environ if environ is None else environ).get(MIN_BATTERY_ENV)
    if raw in (None, ""):
        return DEFAULT_MIN_BATTERY_PERCENT
    try:
        return max(0, min(100, int(raw)))
    except ValueError:
        return DEFAULT_MIN_BATTERY_PERCENT


def parse(text: str) -> tuple[str, int | None]:
    """(source, percent) from ``pmset -g batt`` output. A desktop Mac reports AC power and no percentage."""
    if "AC Power" in text:
        source = "ac"
    elif "Battery Power" in text:
        source = "battery"
    else:
        source = "unknown"
    match = re.search(r"(\d+)%", text)
    return source, int(match.group(1)) if match else None


def read_power(runner=None) -> tuple[str, int | None]:
    run = runner or (lambda: subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout)
    try:
        return parse(run())
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown", None


def evaluate(source: str, percent: int | None, floor: int = DEFAULT_MIN_BATTERY_PERCENT) -> Verdict:
    if source == "ac":
        return Verdict(True, "AC power", source, percent, floor)
    if floor <= 0:
        return Verdict(True, f"battery floor disabled ({MIN_BATTERY_ENV}=0)", source, percent, floor)
    if source == "battery":
        if percent is None:
            return Verdict(False, "On battery and the charge level could not be read; refusing to start. Plug in.", source, percent, floor)
        if percent < floor:
            return Verdict(False, f"Battery is {percent}%, below the {floor}% minimum. Plug in or charge before starting the engine.",
                           source, percent, floor)
        return Verdict(True, f"on battery, {percent}% (minimum {floor}%)", source, percent, floor)
    return Verdict(False, "The power source could not be read (pmset unavailable); refusing to start. Plug in.", source, percent, floor)


def check(environ=None, runner=None) -> Verdict:
    source, percent = read_power(runner)
    env = os.environ if environ is None else environ
    if env.get("VEYSTRIX_REQUIRE_AC") == "1" and source != "ac":
        return Verdict(False, "AC power is required by Veystrix Settings. Plug in or allow battery operation.", source, percent, minimum_percent(env))
    return evaluate(source, percent, minimum_percent(env))
