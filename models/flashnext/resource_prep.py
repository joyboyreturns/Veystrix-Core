"""Autonomous memory preparation before a model load (CHAT and AGENTIC).

One bounded procedure replaces "free a few hundred MB by hand and start again". It looks at the machine, does only what is
safe, waits for macOS accounting to settle, looks again, and either admits the launch or stops with ONE actionable report.

Per round, in this order, each step recorded with its effect:

1. Veystrix-owned temporary memory (allocator caches, RAM prefix snapshots, expert staging). At startup nothing is loaded, so
   this is a recorded no-op; the same hooks serve a manager that runs while an engine is idle.
2. A conditional macOS file-cache purge, through ``purge_recovery`` and ONLY that: it never overlaps a load or a request,
   needs ``sudo -n`` permission, backs off after an ineffective purge, and is skipped when file cache cannot change the
   verdict (so there is no purge loop that cannot help).
3. Optional graceful closing of applications the owner listed (``app_closer``; disabled unless authorized).
4. A settle wait (2, 4, 8 s) and a fresh reading.

Stops early when a round changed nothing that could matter, after ``max_rounds`` rounds, or at the deadline. Never: kill or
signal any process, touch unsaved work, purge during inference, weaken a guard, or treat inactive anonymous memory as file
cache. In AGENTIC the verdict is the tiered admission (``prefill_readiness.evaluate_tiered``: the 9 GiB gate, pressure and the
4 GiB headroom, unchanged, against the smallest cold-prefill budget the launch needs). In CHAT only the 9 GiB gate is
prepared for; the guard that follows keeps deciding.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from models.flashnext import prefill_readiness as R
from models.flashnext import purge_recovery as P

GIB = 2 ** 30
LOG_FLAG = "FLASHNEXT_PREP_LOG"
MIN_USEFUL_BYTES = 128 * 2 ** 20            # a step that changes available memory by less than this did not help


@dataclass
class Step:
    round: int
    name: str
    action: str                              # "done", "skipped", "nothing"
    detail: str
    available_before: int = 0
    available_after: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"round": self.round, "name": self.name, "action": self.action, "detail": self.detail,
                "available_before_gib": round(self.available_before / GIB, 2),
                "available_after_gib": round(self.available_after / GIB, 2), "seconds": round(self.seconds, 1)}


@dataclass
class PrepReport:
    admitted: bool
    verdict: R.Verdict | None
    steps: list = field(default_factory=list)
    rounds: int = 0
    seconds: float = 0.0
    stop_reason: str = ""
    app_closing_enabled: bool = False
    readings: list = field(default_factory=list)     # before/after each round: available, anonymous, compressor, swap, headroom

    @property
    def tier(self):
        return self.verdict.tier if self.verdict is not None else None

    def as_dict(self) -> dict:
        return {"admitted": self.admitted, "tier": self.tier.name if self.tier else None, "rounds": self.rounds,
                "seconds": round(self.seconds, 1), "stop_reason": self.stop_reason,
                "app_closing_enabled": self.app_closing_enabled, "readings": self.readings,
                "required_context": R.required_context(), "steps": [s.as_dict() for s in self.steps],
                "verdict": self.verdict.as_dict() if self.verdict is not None else None}

    def failure_report(self) -> str:
        """ONE actionable message for a launch that could not be admitted."""
        verdict = self.verdict
        need = R.required_context()
        lines = [R.render(verdict) if verdict is not None else "no reading", "",
                 *([f"This launch requires a context of {need} tokens (explicit --context-limit), so only tiers that serve it "
                    "were considered; Veystrix will not load the model at a size that cannot hold the request."] if need else []),
                 f"Memory preparation: {self.rounds} round(s) in {self.seconds:.0f}s, stopped because {self.stop_reason}."]
        for step in self.steps:
            lines.append(f"  round {step.round} {step.name}: {step.action}: {step.detail}")
        if verdict is not None:
            gate = verdict.checks[0] if verdict.checks else None
            if gate is not None and not gate.ok and gate.values.get("required"):
                lack = gate.values["required"] - gate.values["available"]
                lines.append(f"The smallest tier is still short by {lack / GIB:.2f} GiB of available memory (needs "
                             f"{gate.values['required'] / GIB:.2f} GiB, has {gate.values['available'] / GIB:.2f} GiB) after every "
                             "step above. Memory held by other applications is not something Veystrix can free on its own.")
        if not self.app_closing_enabled:
            lines.append("Optional: list applications Veystrix may ask to quit gracefully (never forced) in "
                         "~/.flashnext/close-apps.json: {\"enabled\": true, \"apps\": [\"AppName\"]}. It is off until you do.")
        lines.append("Nothing was loaded. No guard was weakened and nothing was terminated.")
        return "\n".join(lines)


def snapshot(label: str, verdict) -> dict:
    """The memory values a recovery decision is made from (GiB): what was there and what the budget left."""
    cats = verdict.categories
    budget = next((c for c in verdict.checks if c.name.startswith(("cold-prefill", "legacy anonymous"))), None)
    row = {"stage": label, "available_gib": round(cats["gate_available"] / GIB, 2),
           "anonymous_gib": round(cats["anonymous"] / GIB, 2), "compressor_gib": round(cats["compressor"] / GIB, 2),
           "swap_used_gib": round(cats["swap_used"] / GIB, 2), "pressure": cats["pressure_level"],
           "admitted": verdict.admitted, "tier": verdict.tier.name if verdict.tier else None}
    if budget is not None:
        row["headroom_gib"] = round(budget.values["headroom"] / GIB, 2)
    return row


class ResourcePreparer:
    def __init__(self, *, agentic: bool, recovery=None, closer=None, reclaim_hooks=(), read=P.read_system, total=None,
                 processes=R.read_processes, clock=time.monotonic, sleep=time.sleep, max_rounds: int = 3,
                 deadline_s: float = 120.0, settle_s: float = 2.0, pause_s: float = 1.0, out=None):
        self.agentic, self.recovery, self.closer = agentic, recovery, closer
        self.reclaim_hooks = list(reclaim_hooks)
        self._read, self._total, self._processes = read, total, processes
        self._clock, self._sleep = clock, sleep
        self.max_rounds, self.deadline_s, self.settle_s, self.pause_s = max_rounds, deadline_s, settle_s, pause_s
        self._out = out or (lambda text: None)

    # -- the verdict being prepared for
    def verdict(self) -> R.Verdict:
        total = self._total or R.physical_memory()
        if self.agentic:
            verdict = R.live_tiered(read=self._read, total=total, processes=self._processes, pause=self.pause_s,
                                    sleep=self._sleep)
            gate = verdict.checks[0] if verdict.checks else None
            self._gate_required = (gate.values or {}).get("required", 0) if gate is not None else 0
            return verdict
        first = self._read()
        self._sleep(self.pause_s)
        reading = self._read()
        gate = R.check_gate(reading)
        pressure = R.check_pressure(reading, max(0, reading.swapouts - first.swapouts))
        verdict = R.Verdict(gate.ok and pressure.ok, [gate, pressure], R.categories(reading, total), [])
        return verdict

    def _available(self) -> int:
        return self._read().available

    def _required(self) -> int:
        cfg = self.recovery.cfg if self.recovery is not None else P.Config()
        base = cfg.start_gate_bytes + cfg.start_margin_bytes
        measured = getattr(self, "_gate_required", 0)       # the smallest tier's measured requirement (AGENTIC only)
        return max(base, int(measured) + cfg.start_margin_bytes) if measured else base

    def prepare(self) -> PrepReport:
        began = self._clock()
        report = PrepReport(False, None, app_closing_enabled=bool(self.closer is not None and self.closer.enabled))
        verdict = report.verdict = self.verdict()
        report.readings.append(snapshot("before", verdict))
        if verdict.admitted:
            report.admitted, report.stop_reason = True, "already admitted"
            report.seconds = self._clock() - began
            return report
        for number in range(1, self.max_rounds + 1):
            if self._clock() - began >= self.deadline_s:
                report.stop_reason = f"the {self.deadline_s:.0f}s deadline was reached"
                break
            report.rounds = number
            progressed = False
            progressed |= self._reclaim_owned(report, number)
            progressed |= self._purge(report, number)
            progressed |= self._close_apps(report, number)
            # let macOS accounting settle: longer after a step that changed something, one base wait when nothing did
            self._sleep(self.settle_s * 2 ** (number - 1) if progressed else self.settle_s)
            verdict = report.verdict = self.verdict()
            report.readings.append(snapshot(f"after round {number}", verdict))
            self._out(f"memory preparation round {number}: gate figure "
                      f"{verdict.categories['gate_available'] / GIB:.2f} GiB, "
                      f"{'admitted' if verdict.admitted else 'not yet admitted'}")
            if verdict.admitted:
                report.admitted, report.stop_reason = True, f"admitted after round {number}"
                break
            if not progressed:
                report.stop_reason = "no step could improve the situation (recovery cannot help)"
                break
        else:
            report.stop_reason = f"{self.max_rounds} rounds did not reach admission"
        report.seconds = self._clock() - began
        self._log(report)
        return report

    # -- steps
    def _reclaim_owned(self, report, number) -> bool:
        before = self._available()
        freed = 0
        for hook in self.reclaim_hooks:
            try:
                freed += int(hook(self._read()) or 0)
            except Exception as error:
                report.steps.append(Step(number, "owned memory", "skipped", f"{type(error).__name__}: {error}", before, before))
        if not self.reclaim_hooks:
            report.steps.append(Step(number, "owned memory", "nothing", "no engine is loaded yet, so Veystrix owns no memory", before, before))
            return False
        report.steps.append(Step(number, "owned memory", "done" if freed else "nothing", f"freed {freed / GIB:.2f} GiB",
                                 before, self._available()))
        return freed >= MIN_USEFUL_BYTES

    def _purge(self, report, number) -> bool:
        if self.recovery is None:
            report.steps.append(Step(number, "file-cache purge", "skipped", "auto purge is off", 0, 0))
            return False
        began = self._clock()
        outcome = self.recovery.attempt(self._required(), "startup")
        record = outcome.record
        before = record.available_before if record else self._available()
        after = record.available_after if record else before
        action = "done" if outcome.action == "purged" else ("nothing" if outcome.action == "ok" else "skipped")
        report.steps.append(Step(number, "file-cache purge", action, outcome.reason, before, after, self._clock() - began))
        return outcome.action == "purged" and (after - before) >= MIN_USEFUL_BYTES

    def _close_apps(self, report, number) -> bool:
        if self.closer is None or not self.closer.enabled:
            return False
        before = self._available()
        began = self._clock()
        results = self.closer.close_allowed()
        closed = [r["app"] for r in results if r["requested"] and r["closed"]]
        kept = [f"{r['app']} ({r['reason']})" for r in results if not r["closed"]]
        detail = (f"closed {', '.join(closed) or 'none'}" + (f"; left open: {', '.join(kept)}" if kept else ""))
        report.steps.append(Step(number, "close allowed apps", "done" if closed else "nothing", detail, before,
                                 self._available(), self._clock() - began))
        return bool(closed)

    def _log(self, report) -> None:
        path = os.path.expanduser(os.environ.get(LOG_FLAG) or "~/.flashnext/prep-log.jsonl")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as handle:
                row = report.as_dict()
                row["verdict"] = {"admitted": row["verdict"]["admitted"], "tier": row["verdict"]["tier"],
                                  "tried": row["verdict"]["tried"]} if row["verdict"] else None
                handle.write(json.dumps({"wall": time.time(), **row}) + "\n")
        except OSError:
            pass


def main(argv=None) -> int:
    """``python -m models.flashnext.resource_prep [--agentic] [--plan]``.

    ``--plan`` only reports what the machine admits now and what preparation could do; it performs no purge and no closing."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--plan", action="store_true", help="report only; change nothing")
    args = parser.parse_args(argv)
    from models.flashnext.app_closer import AppCloser

    preparer = ResourcePreparer(agentic=args.agentic)
    if args.plan:
        verdict = preparer.verdict()
        print(R.render(verdict))
        return 0 if verdict.admitted else 1
    recovery = P.PurgeRecovery(P.Config.from_environment())
    if args.agentic:
        recovery._blocker = R.purge_blocker_tiered
    preparer = ResourcePreparer(agentic=args.agentic, recovery=recovery, closer=AppCloser(), out=print)
    report = preparer.prepare()
    print(json.dumps(report.as_dict(), indent=2, default=str) if report.admitted else report.failure_report())
    return 0 if report.admitted else R.ADMISSION_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
