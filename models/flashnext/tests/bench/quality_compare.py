"""Compare quality-suite results of two execution paths and give a gate verdict (offline, reads JSON only).

    python -m models.flashnext.tests.bench.quality_compare REFERENCE.json CANDIDATE.json --relation same-plan

The relation says what the two paths are supposed to share, which decides what counts as a failure:

* ``same-plan``   the candidate only adds a cache (cached versus uncached, identical chunk plan). Required:
                  identical token ids in every turn of every case. Any difference fails.
* ``reordered``   the candidate changes the execution plan (chunked versus unchunked, a different chunk size or
                  planner). Token identity is not required (bf16 kernels differ with shape), but is reported;
                  required: no case that passed on the reference fails on the candidate, and no category loses
                  pass rate. A first divergence is reported with its position so it can be traced.

Evidence stays in three separate lines in the report: bit-identical, numerically close (comes from
numeric_diagnostics, not from here), task-level.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RELATIONS = ("same-plan", "reordered")


def load(path) -> dict:
    data = json.loads(Path(path).read_text())
    quality = data.get("quality") if isinstance(data, dict) and "quality" in data else data
    if not quality or "cases" not in quality:
        raise ValueError(f"{path}: no quality results")
    return quality


def first_divergence(a: list, b: list):
    for position, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return position
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_case(ref: dict, cand: dict) -> dict:
    turns, identical, divergence = [], True, None
    for index, (r, c) in enumerate(zip(ref["turns"], cand["turns"])):
        same = r["ids"] == c["ids"]
        where = None if same else first_divergence(r["ids"], c["ids"])
        turns.append({"turn": index, "identical": same, "first_divergence": where,
                      "ref_passed": r["passed"], "cand_passed": c["passed"], "ref_generated": r["generated"],
                      "cand_generated": c["generated"]})
        if not same and divergence is None:
            identical, divergence = False, {"turn": index, "token": where}
    if len(ref["turns"]) != len(cand["turns"]):
        identical = False
    return {"case": ref["case"], "category": ref["category"], "identical": identical, "divergence": divergence,
            "ref_passed": ref["passed"], "cand_passed": cand["passed"], "turns": turns}


def by_category(results: list, key: str) -> dict:
    out: dict = {}
    for item in results:
        row = out.setdefault(item["category"], {"passed": 0, "total": 0})
        row["total"] += 1
        row["passed"] += 1 if item[key] else 0
    return out


def compare(reference: dict, candidate: dict, relation: str = "same-plan") -> dict:
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}")
    ref = {c["case"]: c for c in reference["cases"]}
    cand = {c["case"]: c for c in candidate["cases"]}
    shared = [i for i in ref if i in cand]
    cases = [compare_case(ref[i], cand[i]) for i in shared]
    missing = sorted(set(ref) ^ set(cand))
    regressions = [c["case"] for c in cases if c["ref_passed"] and not c["cand_passed"]]
    absolute = [c["case"] for c in cases if not c["ref_passed"] and not c["cand_passed"]]   # fails on both paths
    schema = {}
    for label, results in (("reference", ref), ("candidate", cand)):
        for case_id in shared:
            found = [i for t in results[case_id]["turns"] for i in t.get("schema_issues", [])]
            if found:
                schema.setdefault(case_id, {})[label] = found
    fixes = [c["case"] for c in cases if not c["ref_passed"] and c["cand_passed"]]
    diverged = [c for c in cases if not c["identical"]]
    ref_rates, cand_rates = by_category(cases, "ref_passed"), by_category(cases, "cand_passed")
    losing = sorted(k for k in ref_rates if cand_rates[k]["passed"] < ref_rates[k]["passed"])
    reasons = []
    if missing:
        reasons.append(f"cases not in both runs: {missing}")
    if not cases:
        reasons.append("no common cases")
    if relation == "same-plan":
        if diverged:
            reasons.append("tokens differ from the same-plan reference: "
                           + ", ".join(f"{c['case']}@turn{c['divergence']['turn']}/token{c['divergence']['token']}"
                                       for c in diverged))
    else:
        if regressions:
            reasons.append(f"passed on the reference, fails on the candidate: {regressions}")
        if losing:
            reasons.append(f"pass rate dropped in: {losing}")
    return {
        "relation": relation, "cases": len(cases),
        "bit_identical": not diverged and not missing, "identical_cases": len(cases) - len(diverged),
        "diverged_cases": [{"case": c["case"], **(c["divergence"] or {})} for c in diverged],
        "task_level": {"reference": ref_rates, "candidate": cand_rates, "regressions": regressions,
                       "fixes": fixes, "categories_losing": losing,
                       "absolute_failures": absolute, "schema_issues": schema},
        "verdict": "PASS" if not reasons else "FAIL", "reasons": reasons, "detail": cases}


def text(result: dict) -> str:
    lines = [f"relation: {result['relation']}   verdict: {result['verdict']}",
             f"bit-identical tokens: {result['identical_cases']}/{result['cases']} cases"
             + ("  (all)" if result["bit_identical"] else "")]
    for d in result["diverged_cases"]:
        lines.append(f"  differs: {d['case']} turn {d.get('turn')} token {d.get('token')}")
    t = result["task_level"]
    lines.append("task level (passed/total, reference -> candidate):")
    for category in sorted(t["reference"]):
        r, c = t["reference"][category], t["candidate"][category]
        lines.append(f"  {category:<18} {r['passed']}/{r['total']} -> {c['passed']}/{c['total']}")
    if t["regressions"]:
        lines.append(f"regressions: {t['regressions']}")
    if t["absolute_failures"]:
        lines.append(f"fails on both paths (a model failure, not a regression): {t['absolute_failures']}")
    for case_id, found in t["schema_issues"].items():
        for label, issues in found.items():
            lines.append(f"tool schema issue, {case_id} ({label}): {'; '.join(sorted(set(issues)))}")
    if t["fixes"]:
        lines.append(f"candidate passes where the reference failed: {t['fixes']}")
    lines += [f"reason: {r}" for r in result["reasons"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--relation", choices=RELATIONS, default="same-plan")
    parser.add_argument("--json", help="write the full comparison here (inside results/)")
    args = parser.parse_args(argv)
    result = compare(load(args.reference), load(args.candidate), args.relation)
    print(text(result))
    if args.json:
        from macqwen.results import output_path

        target = output_path("flashnext", "quality-compare", "comparison.json", args.json)
        target.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {target}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
