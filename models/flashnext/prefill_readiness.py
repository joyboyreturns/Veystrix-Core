"""AGENTIC cold-prefill readiness: an admission check that goes beyond the 9 GiB start gate.

The start gate (``free + inactive + speculative >= 9 GiB``) is a queue measure. Three real runs started from nearly the same
gate figure (9.15, 9.59 and 9.70 GiB) with 5.6 to 6.4 GiB of anonymous memory, and ended three different ways: five CHAT
answers, a compressor guard stop, and a macOS pressure level 2 stop in the first cold AGENTIC prefill. What the gate cannot
see is how much memory other processes hold that macOS will have to *compress* once the checkpoint's first prefill starts
streaming experts. This module budgets that explicitly.

UPDATE 2026-11-09: the gate is now the measured model of ``admission_model`` (the figure falls 6.1-6.65 GiB in every real run;
require that plus the runtime floor and a margin, never less than 9 GiB) and the budget below is ADVISORY only: it never
refuses and never blocks a purge. See docs/flashnext/admission-audit-2026-11-09.md for the seven-run comparison.

It never touches a runtime guard (compressor, swap, pressure, thermal, GPU). It runs only in AGENTIC mode; CHAT keeps its
verified path.

What macOS exposes (``vm_stat``, ``sysctl``): free, active, inactive, speculative, wired, purgeable, file-backed and
anonymous page counts, the compressor's size, swap usage and the pressure level. It does NOT split active or inactive pages
into file-backed and anonymous. Only bounds follow from ``file_backed + anonymous == active + inactive + speculative``
(speculative pages are file-backed). The bounds are reported; no check below relies on an exact split. Per-process memory
comes from ``top`` (resident plus compressed) and is used for advice only.

Budget (all constants are observations from the real checkpoint, see docs/flashnext/agentic-prefill-readiness.md):

    committed = wired + compressor + cold_prefill_peak + harness + 0.75 * other_anonymous
    headroom  = physical_memory - committed          must be >= 4 GiB

* ``cold_prefill_peak`` 6.0 GiB: MLX peak 4.62 GiB (898 tokens, the failed run) to 5.4 GiB (2,046 tokens) plus about 0.3 GiB of
  process memory, rounded up.
* ``harness`` 1.0 GiB: the DeepSeek Harness client next to the server (headless measured 0.19 GiB; the desktop app is
  unmeasured, so it is rounded up generously).
* ``0.75``: macOS compressed 4.09 GiB of anonymous memory into 2.91 GiB (0.71) when the failed run's prefill began.
* 4 GiB: the 3 GiB guard floor plus 0.66 GiB above the 3.34 GiB available at which pressure level 2 first appeared.

File-backed memory is deliberately not in the budget: a cache purge cannot change this verdict, and the purge logic is told so.

Tiers (added 2026-11-08). 6.0 GiB is the right budget for a cold prefill of up to 2,046 tokens or more, but a launch that only
needs a short cold prefill does not need to reserve it. The MLX peak scales with the tokens prefilled (full-sequence logits alone
are 0.47 MiB per token, the per-layer transient grows with the chunk), and two real peaks bracket it: 4.62 GiB at 898 tokens
(first AGENTIC attempt) and 5.40 GiB at 2,046 tokens. ``cold_prefill_peak_bytes`` interpolates linearly between them (never below
the 898-token value, never above 6.0 GiB), adds the 0.3 GiB of process memory, and rounds UP to 0.25 GiB. Admission tries the
tiers from the largest context down and takes the first whose budget passes UNCHANGED rules (9 GiB gate, pressure, 4 GiB
headroom). The server then enforces that tier: a request whose cold prefill or context exceeds it is refused with a clear error,
so the assumption is never exceeded silently. Evidence is two real runs at 898 tokens and one 2K arm; nothing between or above is
measured, which is why the interpolation rounds up and the standard tier keeps 6.0 GiB.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

from models.flashnext import admission_model as M
from models.flashnext import purge_recovery as P

GIB = 2 ** 30
MANDATORY_START_GATE_BYTES = 9 * GIB            # unchanged; chat_guard enforces the same number
COLD_PREFILL_PEAK_BYTES = int(6.0 * GIB)
HARNESS_ALLOWANCE_BYTES = 1 * GIB
COMPRESSION_RATIO = 0.75
REQUIRED_HEADROOM_BYTES = 4 * GIB
PRESSURE_OK_LEVEL = 1
MEASURED_PEAKS = ((898, 4.62), (2046, 5.40))     # (cold-prefill tokens, MLX peak GiB): real runs, see docs/flashnext/agentic-prefill-readiness.md
PROCESS_OVERHEAD_GIB = 0.3
ROUND_GIB = 0.25
DEFAULT_TOKENS = 4096
ADMISSION_EXIT_CODE = 69                         # EX_UNAVAILABLE: refused before loading, nothing started
LOG_FLAG = "FLASHNEXT_ADMISSION_LOG"


def budgeted_plan_active(environ=None) -> bool:
    """True when the process runs the deterministic "budgeted" chunk plan (the only one that bounds a long prefill)."""
    return (os.environ if environ is None else environ).get("FLASHNEXT_PREFILL_CHUNK_POLICY", "") == "budgeted"


def cold_prefill_peak_bytes(tokens: int, budgeted: bool | None = None) -> int:
    """Budget for the MLX peak plus process memory of a cold prefill of ``tokens`` tokens (interpolated, rounded up).

    Beyond the 2,046-token measurement, the "standard" figure stays at 6.0 GiB (its tiers stop at 4,096 tokens, one chunk plan).
    Under the budgeted plan each chunk's transient is bounded by the chunk budget (about what a 2,048-token chunk added at
    position 0, which the 5.40 GiB measurement already contains), so what grows with the prompt is the persistent KV (bf16
    here, the conservative case) and indexer keys: ESTIMATE = 5.40 GiB + KV growth + 0.3 GiB process, rounded up. Unmeasured
    above 2,046 tokens; the per-chunk trace of the first real 4K run is what confirms or corrects it."""
    import math

    (t0, p0), (t1, p1) = MEASURED_PEAKS
    if budgeted is None:
        budgeted = budgeted_plan_active()
    if tokens >= t1 and budgeted and tokens > t1:
        from models.flashnext.prefill_budget import KV_BYTES_PER_TOKEN_TOTAL

        peak = p1 + (tokens - t1) * KV_BYTES_PER_TOKEN_TOTAL / GIB + PROCESS_OVERHEAD_GIB
        return max(COLD_PREFILL_PEAK_BYTES, int(math.ceil(peak / ROUND_GIB) * ROUND_GIB * GIB))
    if tokens >= t1:
        return COLD_PREFILL_PEAK_BYTES
    peak = p0 + (max(tokens, t0) - t0) * (p1 - p0) / (t1 - t0) + PROCESS_OVERHEAD_GIB
    return min(COLD_PREFILL_PEAK_BYTES, int(math.ceil(peak / ROUND_GIB) * ROUND_GIB * GIB))


@dataclass(frozen=True)
class Tier:
    """An admitted operating size: the context the server accepts and the largest single cold prefill it will run."""
    name: str
    context_limit: int
    max_cold_prefill: int

    @property
    def peak_bytes(self) -> int:
        return cold_prefill_peak_bytes(self.max_cold_prefill, budgeted=True if self.name == "configured" else None)


TIERS = (Tier("standard", 8192, 4096), Tier("compact", 2048, 1536), Tier("minimal", 1536, 1024))
# Only under the budgeted chunk plan (opt-in). 14,336 covers a first request of about 12K tokens with room for its
# system prompt; the context limit leaves room for the answer and tool results of several continuation turns.
LONG_TIER = Tier("long", 16384, 14336)
# The 8K agentic workload (first request about 8-9K tokens, 1,024 reserved for the answer, room for tool results and a
# follow-up): same estimate formula as every other tier, so its budget is what the formula says (6.0 GiB at 10,240
# tokens), not a lowered figure. Real 4K evidence (Gate A attempt 2: process peak 5.43 GiB at 3.7K) is inside it.
MID_TIER = Tier("8k", 12288, 10240)


def required_context(environ=None) -> int | None:
    """The context the launch REQUIRES (an explicit ``--context-limit``), or None when the limit is only a mode default.

    A default may degrade to a smaller admitted tier (the server then refuses what does not fit). A stated requirement may
    not: admitting a tier that cannot serve it loads a 113 GB model only to refuse the first request."""
    environ = os.environ if environ is None else environ
    if environ.get("VEYSTRIX_CONTEXT_REQUIRED") != "1":
        return None
    try:
        value = int(environ.get("VEYSTRIX_CONTEXT_LIMIT") or 0)
    except ValueError:
        return None
    return value if value > 0 else None


def active_tiers(environ=None) -> tuple:
    """The tiers admission may choose from, largest first: TIERS plus the long tier under the budgeted plan, restricted to
    tiers whose context serves a stated requirement (if none does, the largest tier, so the refusal reports the real gap)."""
    env = os.environ if environ is None else environ
    # Agentic requests use bounded chunks. Assess the configured workload as a
    # whole instead of silently substituting an unrelated legacy context tier.
    # The estimate includes KV growth and remains subject to the normal gate.
    try:
        context = int(env.get("VEYSTRIX_CONTEXT_LIMIT", "0"))
        cold = int(env.get("VEYSTRIX_MAX_COLD_PREFILL", "0"))
        chunk = int(env.get("FLASHNEXT_PREFILL_CHUNK_TOKENS", "0"))
    except ValueError:
        context = cold = chunk = 0
    if env.get("VEYSTRIX_MODE") == "agentic" and context > 0 and cold > 0 and 0 < chunk <= 2048:
        return (Tier("configured", context, cold),)
    tiers = (LONG_TIER, MID_TIER, *TIERS) if budgeted_plan_active(environ) else TIERS
    need = required_context(environ)
    if need:
        tiers = tuple(t for t in tiers if t.context_limit >= need) or tiers[:1]
    return tiers


@dataclass
class Check:
    name: str
    ok: bool
    category: str           # the memory category this check is about
    detail: str
    values: dict = field(default_factory=dict)


@dataclass
class Verdict:
    admitted: bool
    checks: list
    categories: dict
    processes: list = field(default_factory=list)
    advice: list = field(default_factory=list)
    tier: Tier | None = None
    tried: list = field(default_factory=list)       # (tier name, admitted) for every tier evaluated

    @property
    def blocking(self) -> list:
        return [check for check in self.checks if not check.ok]

    def as_dict(self) -> dict:
        return {"admitted": self.admitted, "categories": self.categories, "processes": self.processes,
                "advice": self.advice, "tier": self.tier.name if self.tier else None,
                "tier_limits": ({"context_limit": self.tier.context_limit, "max_cold_prefill": self.tier.max_cold_prefill}
                                if self.tier else None), "tried": self.tried,
                "checks": [{"name": c.name, "ok": c.ok, "category": c.category, "detail": c.detail,
                            "values": c.values} for c in self.checks]}


# -- the machine ------------------------------------------------------------------------------------------------


def categories(reading: P.Reading, total: int) -> dict:
    """Every category macOS reports, in bytes, plus bounds for what it does not split."""
    page = reading.page
    b = lambda pages: pages * page                                              # noqa: E731
    file_cache = max(0, reading.file_backed - reading.speculative)              # file-backed pages outside speculative
    inactive_file_max = min(reading.inactive, file_cache)
    inactive_file_min = max(0, min(reading.inactive, file_cache - reading.active))
    inactive_anon_min = max(0, reading.inactive - inactive_file_max)
    inactive_anon_max = min(reading.anonymous, reading.inactive - inactive_file_min)
    return {
        "physical": total,
        "free": b(reading.free),
        "speculative_file": b(reading.speculative),
        "inactive": b(reading.inactive),
        "inactive_file_min": b(inactive_file_min), "inactive_file_max": b(inactive_file_max),
        "inactive_anonymous_min": b(inactive_anon_min), "inactive_anonymous_max": b(inactive_anon_max),
        "active": b(reading.active),
        "wired": b(reading.wired),
        "compressor": b(reading.compressor),
        "file_backed": b(reading.file_backed),
        "anonymous": b(reading.anonymous),
        "purgeable": b(reading.purgeable),
        "swap_used": reading.swap_used,
        "pressure_level": reading.pressure,
        "gate_available": reading.available,
        # memory that is reusable without compressing anything: free, speculative and provably-file inactive pages
        "reusable_floor": b(reading.free + reading.speculative + inactive_file_min),
    }


def _run(command, timeout=8):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL).stdout
    except Exception:
        return ""


def physical_memory() -> int:
    match = re.search(r"\d+", _run(["sysctl", "-n", "hw.memsize"]))
    return int(match.group(0)) if match else 16 * GIB


_SIZE = re.compile(r"^([\d.]+)([BKMGT])[+-]?$")
_UNIT = {"B": 1, "K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30, "T": 2 ** 40}


def parse_top(text: str) -> list:
    """``top -l 1 -o mem -stats pid,command,mem,cmprs`` rows: resident and compressed bytes per process."""
    rows = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\s+(.+?)\s+([\d.]+[BKMGT][+-]?)\s+([\d.]+[BKMGT][+-]?)\s*$", line)
        if not match:
            continue
        sizes = []
        for token in match.group(3, 4):
            unit = _SIZE.match(token)
            sizes.append(int(float(unit.group(1)) * _UNIT[unit.group(2)]) if unit else 0)
        rows.append({"pid": int(match.group(1)), "command": match.group(2), "resident": sizes[0], "compressed": sizes[1]})
    return rows


def read_processes(limit: int = 8) -> list:
    return parse_top(_run(["top", "-l", "1", "-n", str(limit), "-o", "mem", "-stats", "pid,command,mem,cmprs"]))[:limit]


# -- the checks -------------------------------------------------------------------------------------------------


def required_gate_bytes(reading: P.Reading, tokens: int | None = None) -> int:
    """The start figure this launch needs: the measured drop for ``tokens`` plus the runtime floor and margin (admission_model),
    never below the 9 GiB gate. ``tokens=None`` is the unchanged 9 GiB gate."""
    if tokens is None:
        return MANDATORY_START_GATE_BYTES
    return max(MANDATORY_START_GATE_BYTES, M.required_available_bytes(tokens, other_anonymous_bytes=reading.anonymous * reading.page))


def check_gate(reading: P.Reading, tokens: int | None = None) -> Check:
    required = required_gate_bytes(reading, tokens)
    ok = reading.available >= required
    if tokens is None:
        why = "unchanged 9 GiB gate"
    else:
        stages = M.stage_lowest_gib(reading.available / GIB, tokens)
        why = (f"predicted lowest gate figure {stages['after load']:.2f} GiB after load and "
               f"{stages['during first cold prefill and decode']:.2f} GiB through a {tokens}-token cold prefill "
               f"(fall {M.drop_gib(tokens):.2f}); it must stay above the {M.FLOOR_GIB:.1f} GiB runtime floor + "
               f"{M.MARGIN_GIB:.2f} margin")
    return Check("start gate (mandatory, measured)", ok, "available = free + inactive + speculative",
                 f"{reading.available / GIB:.2f} GiB available, needs {required / GIB:.2f} GiB ({why})",
                 {"available": reading.available, "required": required, "tokens": tokens})


def check_pressure(reading: P.Reading, swapout_delta: int = 0) -> Check:
    ok = reading.pressure <= PRESSURE_OK_LEVEL and swapout_delta <= 0
    detail = f"pressure level {reading.pressure}, swap used {reading.swap_used / GIB:.2f} GiB, swap-out pages since last look {swapout_delta}"
    return Check("pressure and swap-out", ok, "pressure / swap", detail,
                 {"pressure": reading.pressure, "swap_used": reading.swap_used, "swapout_delta": swapout_delta})


def check_committed(reading: P.Reading, total: int, peak: int = COLD_PREFILL_PEAK_BYTES) -> Check:
    """The former cold-prefill budget, kept as ADVISORY evidence only (it never refuses a launch since 2026-11-09).

    wired + compressor + peak + 1 GiB harness + 0.75 x other anonymous, against a 4 GiB headroom. The seven real runs show it
    does not predict the lowest gate figure, double-counts the runtime floor already inside the gate and over-reserves the
    client (docs/flashnext/admission-audit-2026-11-09.md). Its figures are still reported so a regression is visible."""
    other = reading.anonymous * reading.page
    squeezed = int(COMPRESSION_RATIO * other)
    wired, compressor = reading.wired * reading.page, reading.compressor * reading.page
    committed = wired + compressor + peak + HARNESS_ALLOWANCE_BYTES + squeezed
    headroom = total - committed
    would_refuse = headroom < REQUIRED_HEADROOM_BYTES
    values = {"physical": total, "wired": wired, "compressor": compressor, "cold_prefill_peak": peak,
              "harness": HARNESS_ALLOWANCE_BYTES, "other_anonymous": other, "other_anonymous_squeezed": squeezed,
              "committed": committed, "headroom": headroom, "required_headroom": REQUIRED_HEADROOM_BYTES,
              "would_refuse_under_legacy_rule": would_refuse, "advisory": True}
    detail = (f"ADVISORY, not a veto: legacy headroom {headroom / GIB:.2f} GiB (was: needs {REQUIRED_HEADROOM_BYTES / GIB:.2f}); "
              f"other anonymous memory {other / GIB:.2f} GiB, wired {wired / GIB:.2f}, compressor {compressor / GIB:.2f}"
              + ("; the legacy rule would have refused" if would_refuse else ""))
    return Check("legacy anonymous-memory budget (advisory)", True, "anonymous memory and compression", detail, values)


def purge_blocker(reading: P.Reading, total: int | None = None) -> str | None:
    """Why a cache purge could not admit this launch (None when a purge could still help).

    Only a pressure or swap-out refusal blocks a purge: a purge adds I/O and cannot fix those. The anonymous-memory budget no
    longer vetoes (it was never measured to predict anything), so a gate shortfall that file cache can close IS recoverable."""
    check = check_pressure(reading)
    return None if check.ok else f"{check.name}: {check.detail}"


def purge_blocker_tiered(reading: P.Reading, total: int | None = None) -> str | None:
    """Like ``purge_blocker`` for the tiered launch: the same pressure rule, plus one the purge cannot get around: the gate
    shortfall against the SMALLEST tier the launch may use must be within what a purge can plausibly return."""
    smallest = active_tiers()[-1]
    check = check_pressure(reading)
    if not check.ok:
        return f"{check.name}: {check.detail}"
    shortfall = required_gate_bytes(reading, smallest.max_cold_prefill) - reading.available
    gain = P.accounting(reading)["purge_gain_upper"]
    if shortfall > 0 and gain < shortfall:
        return (f"the gate is short by {shortfall / GIB:.2f} GiB and at most {gain / GIB:.2f} GiB of file cache can be recovered")
    return None


def evaluate_tiered(reading: P.Reading, total: int, swapout_delta: int = 0, processes=(), tiers=None) -> Verdict:
    """The best tier that passes the unchanged rules, or the verdict for the smallest tier when none does."""
    tiers = active_tiers() if tiers is None else tiers
    tried, last = [], None
    for tier in tiers:
        last = evaluate(reading, total, swapout_delta, processes, peak=tier.peak_bytes, tokens=tier.max_cold_prefill)
        tried.append({"tier": tier.name, "context_limit": tier.context_limit, "max_cold_prefill": tier.max_cold_prefill,
                      "budget_gib": round(tier.peak_bytes / GIB, 2), "admitted": last.admitted})
        if last.admitted:
            last.tier = tier
            break
    last.tried = tried
    return last


def evaluate(reading: P.Reading, total: int, swapout_delta: int = 0, processes=(), peak: int = COLD_PREFILL_PEAK_BYTES,
             tokens: int | None = None) -> Verdict:
    """The verdict for a cold prefill of ``tokens`` (default: the 4,096-token standard tier, the measured reference run)."""
    tokens = DEFAULT_TOKENS if tokens is None else tokens
    checks = [check_gate(reading, tokens), check_pressure(reading, swapout_delta), check_committed(reading, total, peak)]
    cats = categories(reading, total)
    verdict = Verdict(all(check.ok for check in checks), checks, cats, list(processes))
    verdict.advice = advice(verdict, reading)
    return verdict


def advice(verdict: Verdict, reading: P.Reading) -> list:
    cats, lines = verdict.categories, []
    gate, pressure = verdict.checks[0], verdict.checks[1]
    budget = verdict.checks[2] if len(verdict.checks) > 2 else None
    if not gate.ok:
        required = gate.values["required"]
        shortfall = required - reading.available
        accounting = P.accounting(reading)
        upper = accounting["purge_gain_upper"]
        counted = M.expected_purge_gain_bytes(upper)
        if not pressure.ok:
            lines.append(f"The gate is short by {shortfall / GIB:.2f} GiB, but macOS is under pressure, so no purge is attempted.")
        elif counted >= shortfall:
            lines.append(f"The gate is short by {shortfall / GIB:.2f} GiB; recorded purges returned at least "
                         f"{M.PURGE_GAIN_OBSERVED_MIN_GIB:.1f} GiB (median {M.PURGE_GAIN_OBSERVED_MEDIAN_GIB:.1f}) and up to "
                         f"{upper / GIB:.2f} GiB of file cache is recoverable here: FLASHNEXT_AUTO_PURGE=1 (--auto-purge) can "
                         "close this gap before loading, without closing any application.")
        elif upper >= shortfall:
            lines.append(f"The gate is short by {shortfall / GIB:.2f} GiB; a purge might close it (up to {upper / GIB:.2f} GiB of "
                         f"file cache, but the smallest purge on record returned {M.PURGE_GAIN_OBSERVED_MIN_GIB:.1f} GiB), so "
                         "it is attempted and re-measured, not assumed.")
        else:
            lines.append(f"The gate is short by {shortfall / GIB:.2f} GiB and a purge cannot close it (at most {upper / GIB:.2f} GiB "
                         "of file cache); quit applications to free memory.")
    if not pressure.ok:
        lines.append("macOS is already under memory pressure or swapping out. Wait for it to settle or quit applications; "
                     "do not purge, it adds I/O and cannot fix this.")
    if budget is not None and budget.values.get("would_refuse_under_legacy_rule") and gate.ok:
        lines.append("Other applications hold a lot of anonymous memory; the retired budget rule would have refused here. The "
                     "measured gate admits it, and the runtime guards (floor, swap, sustained pressure, thermal) stay on.")
    if verdict.processes and not gate.ok:
        top = ", ".join(f"{p['command']} {(p['resident'] + p['compressed']) / GIB:.2f} GiB" for p in verdict.processes[:5])
        lines.append(f"Largest processes now (resident + compressed, from top): {top}.")
    return lines


# -- reporting --------------------------------------------------------------------------------------------------


def render(verdict: Verdict) -> str:
    gib = lambda value: f"{value / GIB:6.2f} GiB"                               # noqa: E731
    c = verdict.categories
    rows = [
        f"AGENTIC prefill readiness: {'ADMITTED' if verdict.admitted else 'REFUSED'}"
        + (f" at tier {verdict.tier.name} (context {verdict.tier.context_limit}, cold prefill up to "
           f"{verdict.tier.max_cold_prefill} tokens)" if verdict.tier else ""),
        f"  free {gib(c['free'])}   speculative(file) {gib(c['speculative_file'])}   purgeable {gib(c['purgeable'])}",
        f"  inactive {gib(c['inactive'])}  = file {gib(c['inactive_file_min'])} to {gib(c['inactive_file_max'])}  +  anonymous "
        f"{gib(c['inactive_anonymous_min'])} to {gib(c['inactive_anonymous_max'])}   (macOS does not report the split)",
        f"  active {gib(c['active'])}   wired {gib(c['wired'])}   compressor {gib(c['compressor'])}",
        f"  file-backed {gib(c['file_backed'])}   anonymous {gib(c['anonymous'])}   reusable without compressing >= {gib(c['reusable_floor'])}",
        f"  swap used {gib(c['swap_used'])}   pressure level {c['pressure_level']}   gate figure {gib(c['gate_available'])}",
    ]
    for entry in verdict.tried:
        rows.append(f"  tier {entry['tier']:9s} context {entry['context_limit']:5d}  cold prefill <= {entry['max_cold_prefill']:5d}  "
                    f"budget {entry['budget_gib']:.2f} GiB  {'admitted' if entry['admitted'] else 'not admitted'}")
    for check in verdict.checks:
        rows.append(f"  [{'ok' if check.ok else 'NO'}] {check.name}: {check.detail}")
    for line in verdict.advice:
        rows.append(f"  -> {line}")
    return "\n".join(rows)


def log(verdict: Verdict, environ=None) -> None:
    environ = os.environ if environ is None else environ
    path = os.path.expanduser(environ.get(LOG_FLAG) or "~/.flashnext/admission-log.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as handle:
            handle.write(json.dumps({"wall": time.time(), **verdict.as_dict()}) + "\n")
    except OSError:
        pass


def live(read=P.read_system, total=None, processes=read_processes, pause=1.0, sleep=time.sleep) -> Verdict:
    """Two readings a second apart (the second one shows swap-out in progress), then the verdict."""
    first = read()
    sleep(pause)
    reading = read()
    delta = max(0, reading.swapouts - first.swapouts)
    return evaluate(reading, total or physical_memory(), delta, processes())


def live_tiered(read=P.read_system, total=None, processes=read_processes, pause=1.0, sleep=time.sleep) -> Verdict:
    """``live`` with tier selection: what the machine admits right now, and at which size."""
    first = read()
    sleep(pause)
    reading = read()
    delta = max(0, reading.swapouts - first.swapouts)
    return evaluate_tiered(reading, total or physical_memory(), delta, processes())


def main(argv=None) -> int:
    """``python -m models.flashnext.prefill_readiness [--json]``: the verdict on this machine, nothing is started or purged."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    verdict = live()
    print(json.dumps(verdict.as_dict(), indent=2) if args.json else render(verdict))
    return 0 if verdict.admitted else 1


if __name__ == "__main__":
    sys.exit(main())
