"""Task-level quality suite for execution-path validation (cases, programmatic checkers, runner).

Purpose: show that an optimization (chunked prefill, prefix cache, a new chunk planner) preserves what the
Q4 model actually does, not only that its states look close. Every case is checked by code, never by a model,
so a result is reproducible. Decoding is greedy (temperature 0), thinking is off, and tokenization is the
production template, so two paths fed the same case differ only through the execution path.

Categories (a case per category is in the "smoke" tier):

* coding              the reply's function is executed against tests in a restricted interpreter
* reasoning           an exact numeric or yes/no answer
* instruction         verifiable formatting constraints (counts, case, JSON shape, forbidden words)
* tool_call           the native tool-call format, with the production tool set (veystrix.tools) rendered by the chat template
                       and read back by the production parser
* multi_turn          facts carried across turns, including a tool result turn
* retrieval_early / retrieval_middle / retrieval_late
                      an exact code planted at a known position of a ~2.2K-token system prompt

Three levels of evidence stay separate (see docs/flashnext/quality-validation-policy.md):
bit-identical (same token ids), numerically close (state and routing diagnostics), task-level (this suite).

    python -m models.flashnext.tests.bench.bench_prefill_memory --conditions F2L --suite smoke ...
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_PROMPT = HERE / "fixtures" / "agent_system_prompt.txt"

SHORT_SYSTEM = "You are a helpful assistant. Answer concisely and exactly as asked."

# The needles: one fact per block, planted early, in the middle, and late in the system prompt. Each block
# also carries a decoy so that a model matching on topic words alone gets it wrong.
NEEDLES = {
    "early": ("staging database password rotation code", "MAPLE-4821",
              "Reference note A. The staging database password rotation code is MAPLE-4821. "
              "The production rotation code is different and is not listed here."),
    "middle": ("on-call escalation phone extension", "7390",
               "Reference note B. The on-call escalation phone extension is 7390. "
               "The billing desk extension, 7319, is unrelated to escalation."),
    "late": ("release branch for the autumn cycle", "heron-2026-q4",
             "Reference note C. The release branch for the autumn cycle is named heron-2026-q4. "
             "The spring branch, finch-2026-q2, was retired in May."),
}


def agent_text() -> str:
    return AGENT_PROMPT.read_text().strip()


def needle_system() -> str:
    """The agent prompt with three reference notes inserted at about 5%, 45% and 100% of its length."""
    paragraphs = agent_text().split("\n\n")
    total = sum(len(p) for p in paragraphs)
    wanted = [("early", 0.05), ("middle", 0.45)]
    out, seen = [], 0
    for paragraph in paragraphs:
        out.append(paragraph)
        seen += len(paragraph)
        for name, fraction in list(wanted):
            if seen / total >= fraction:
                out.append(NEEDLES[name][2])
                wanted.remove((name, fraction))
    out.append(NEEDLES["late"][2])
    return "\n\n".join(out)


TOOLING_SYSTEM = ("You are a coding agent working in a repository. Inspect it with the provided tools before "
                  "answering, and call exactly one tool when a tool is what the request needs.")


def system_prompts() -> dict:
    return {"short": SHORT_SYSTEM, "tooling": TOOLING_SYSTEM, "agent": agent_text(), "needles": needle_system()}


# -- the production tool set (veystrix.tools): the chat template renders it, its parser reads the reply ----

from veystrix.tools import PARAM_TYPES, REQUIRED_PARAMS, TOOLS, parse_tool_calls  # noqa: E402

CALL_BLOCK = re.compile(r"<tool_call>")
CALL_NAME = re.compile(r"<function=([^>\s]+)>")
PARAM_NAME = re.compile(r"<parameter=([^>\s=]+)>")


def tool_schema_issues(text: str) -> list:
    """Every way the tool calls in ``text`` break the production schema, read from the raw text.

    The production parser silently drops a parameter the tool does not define, so a call with an invented argument
    (for example ``working_directory`` on ``run_command``) still parses; this reads the parameter names before that
    happens and reports them. Also: unknown tool, missing required parameter, a value of the wrong type.
    """
    issues = []
    for block in text.split("<tool_call>")[1:]:
        named = CALL_NAME.search(block)
        if not named:
            issues.append("tool_call block without a <function=...> name")
            continue
        name = named.group(1)
        if name not in PARAM_TYPES:
            issues.append(f"unknown tool {name!r}")
            continue
        params = PARAM_NAME.findall(block)
        issues += [f"{name}: unsupported parameter {p!r}" for p in dict.fromkeys(params) if p not in PARAM_TYPES[name]]
        issues += [f"{name}: missing required parameter {r!r}" for r in REQUIRED_PARAMS[name] if r not in params]
        try:
            parse_tool_calls("<tool_call>" + block)
        except ValueError as error:
            issues.append(f"{name}: {error}")
    return issues


# -- checkers: each returns (passed, detail) ---------------------------------------------------

def check_contains(text, all_of=(), none_of=(), any_of=()):
    low = text.lower()
    missing = [s for s in all_of if s.lower() not in low]
    banned = [s for s in none_of if s.lower() in low]
    if any_of and not any(s.lower() in low for s in any_of):
        missing.append("any of " + "|".join(any_of))
    return (not missing and not banned), f"missing {missing} banned {banned}" if (missing or banned) else "ok"


def check_number(text, expected):
    found = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not found:
        return False, "no number"
    answer = re.search(r"answer\s*:\s*(-?\d+(?:\.\d+)?)", text.replace(",", ""), re.I)
    value = float(answer.group(1)) if answer else float(found[-1])
    return abs(value - float(expected)) < 1e-9, f"got {value}, expected {expected}"


def check_yes_no(text, expected):
    answer = re.search(r"answer\s*:\s*(yes|no)\b", text, re.I) or re.search(r"\b(yes|no)\b", text, re.I)
    got = answer.group(1).lower() if answer else None
    return got == expected, f"got {got}, expected {expected}"


def check_bullets(text, count, max_words):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    bullets = [line for line in lines if line.lstrip().startswith("- ")]
    if len(lines) != len(bullets):
        return False, "text outside bullets"
    if len(bullets) != count:
        return False, f"{len(bullets)} bullets, expected {count}"
    long = [b for b in bullets if len(b.lstrip()[2:].split()) >= max_words]
    return not long, f"bullets too long: {len(long)}" if long else "ok"


def check_sentence(text, word, times, ending):
    sentence = text.strip()
    count = len(re.findall(rf"\b{re.escape(word)}\b", sentence, re.I))
    lines = [line for line in sentence.splitlines() if line.strip()]
    ok = count == times and sentence.endswith(ending) and len(lines) == 1
    return ok, f"'{word}' x{count}, ends {sentence[-1:]!r}, lines {len(lines)}"


def check_json(text, keys):
    block = re.search(r"\{.*\}", text, re.S)
    if not block:
        return False, "no JSON object"
    try:
        value = json.loads(block.group(0))
    except ValueError as error:
        return False, f"invalid JSON: {error}"
    for key, kind in keys.items():
        want = {"string": str, "integer": int}[kind]
        if key not in value or not isinstance(value[key], want) or isinstance(value[key], bool):
            return False, f"key {key} is not a {kind}"
    return set(value) == set(keys), "ok" if set(value) == set(keys) else f"keys {sorted(value)}"


def check_caps(text, words):
    cleaned = text.strip().rstrip(".!")
    parts = cleaned.split()
    return (cleaned == cleaned.upper() and len(parts) == words), f"{len(parts)} words, upper={cleaned == cleaned.upper()}"


def check_tool(text, tool, expect=None):
    """Exactly one call, to ``tool``, that the production parser reads, with every required parameter present,
    no unknown parameter, and each ``expect`` parameter (name -> regex) matching. The strict canonical block
    (<function=...> and <parameter=...> closed) is reported in the detail, not required."""
    if len(CALL_BLOCK.findall(text)) != 1:
        return False, f"{len(CALL_BLOCK.findall(text))} tool_call blocks"
    try:
        calls = parse_tool_calls(text)
    except ValueError as error:
        return False, f"parse error: {error}"
    if len(calls) != 1:
        return False, f"parser read {len(calls)} calls"
    name, params = calls[0]
    if name != tool:
        return False, f"called {name}, expected {tool}"
    issues = tool_schema_issues(text)
    if issues:
        return False, "schema: " + "; ".join(issues)
    missing = [key for key in REQUIRED_PARAMS[name] if key not in params]
    if missing:
        return False, f"missing parameters {missing}"
    for key, pattern in (expect or {}).items():
        if not re.search(pattern, str(params.get(key, "")), re.I):
            return False, f"{key}={params.get(key)!r} does not match {pattern!r}"
    strict = "</function>" in text and "</tool_call>" in text
    return True, "ok" if strict else "ok (not closed canonically)"


FORBIDDEN_CALLS = {"exec", "eval", "open", "compile", "__import__", "input", "globals", "locals", "getattr",
                   "setattr", "delattr", "breakpoint", "exit", "quit", "vars", "dir", "memoryview"}
SAFE_BUILTINS = ("abs all any bool chr dict divmod enumerate filter float int isinstance len list map max min "
                 "ord range reversed round set sorted str sum tuple zip print True False None").split()
RUNNER = r"""
import builtins, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
spec = json.load(sys.stdin)
safe = {n: getattr(builtins, n) for n in spec["builtins"] if hasattr(builtins, n)}
scope = {"__builtins__": safe}
results = []
try:
    exec(spec["code"], scope)
except BaseException as error:
    print(json.dumps({"error": f"definition failed: {type(error).__name__}: {error}"})); sys.exit(0)
for test in spec["tests"]:
    try:
        results.append(bool(eval(test, scope)))
    except BaseException as error:
        results.append(False)
print(json.dumps({"results": results}))
"""


def _validated(code: str):
    import ast

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not allowed"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return f"{node.id} is not allowed"
    return None


def extract_code(text: str) -> str:
    block = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (block.group(1) if block else text).strip()


def check_code(text, tests, function):
    """Run the reply's code against ``tests`` in a restricted interpreter: static checks, a whitelist of
    builtins, a CPU limit and a hard timeout. Restricted, not a security sandbox; the code is this model's own
    output on your own machine."""
    code = extract_code(text)
    if f"def {function}" not in code:
        return False, f"no def {function}"
    try:
        problem = _validated(code)
    except SyntaxError as error:
        return False, f"syntax error: {error.msg}"
    if problem:
        return False, problem
    try:
        done = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps(
            {"code": code, "tests": list(tests), "builtins": SAFE_BUILTINS}), capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    try:
        payload = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, f"runner failed: {done.stderr.strip()[-200:]}"
    if "error" in payload:
        return False, payload["error"]
    passed = sum(payload["results"])
    return passed == len(payload["results"]), f"{passed}/{len(payload['results'])} tests"


CHECKERS = {
    "contains": check_contains, "number": check_number, "yes_no": check_yes_no, "bullets": check_bullets,
    "sentence": check_sentence, "json": check_json, "caps": check_caps, "tool": check_tool, "code": check_code,
}


# -- cases ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Turn:
    user: str | None = None                  # a user message, or
    tool_results: tuple = ()                 # tool outputs returned as the next turn
    max_tokens: int = 64
    check: tuple = ("contains", {})          # (checker name, keyword arguments)


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    system: str                              # key of system_prompts()
    turns: tuple
    tier: str = "full"                       # "smoke" cases are also in "full"
    tools: bool = False                      # offer the production tool set through the chat template


def _one(user, check, max_tokens=64, system="short", id_="", category="", tier="full", tools=False):
    return Case(id_, category, system, (Turn(user=user, max_tokens=max_tokens, check=check),), tier, tools)


def build_cases() -> list:
    cases = []
    code = lambda i, user, function, tests, tier="full": cases.append(_one(
        user + " Reply with only the code in a ```python block.", ("code", {"function": function, "tests": tests}),
        200, "short", f"code_{i}", "coding", tier))
    code("palindrome", "Write a Python function `is_palindrome(s)` that returns True if the string reads the same "
         "forwards and backwards, ignoring case and non-alphanumeric characters.", "is_palindrome",
         ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('hello') is False",
          "is_palindrome('') is True"], "smoke")
    code("fizzbuzz", "Write a Python function `fizzbuzz(n)` that returns a list of the strings for 1..n, using "
         "'Fizz' for multiples of 3, 'Buzz' for multiples of 5 and 'FizzBuzz' for both.", "fizzbuzz",
         ["fizzbuzz(15)[-1] == 'FizzBuzz'", "fizzbuzz(3) == ['1', '2', 'Fizz']", "len(fizzbuzz(7)) == 7"])
    code("merge", "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
         "merge_sorted", ["merge_sorted([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5]", "merge_sorted([], [1]) == [1]",
                          "merge_sorted([2, 2], [2]) == [2, 2, 2]"])
    code("wordcount", "Write a Python function `count_words(text)` that returns a dict of lowercase word counts, "
         "splitting on whitespace.", "count_words",
         ["count_words('a b A') == {'a': 2, 'b': 1}", "count_words('') == {}"])

    answer = " Reply with only 'Answer: <number>'."
    for i, (question, number, tier) in enumerate((
            ("A train travels 60 km in 45 minutes. At the same speed, how many km does it travel in 2 hours?", 160, "smoke"),
            ("Tom has 3 boxes with 12 apples each. He gives away 7 apples and then buys 5 more. How many apples does he have?", 34, "full"),
            ("What is the sum of the first 10 positive integers, divided by 5?", 11, "full"))):
        cases.append(_one(question + answer, ("number", {"expected": number}), 24, "short", f"reason_{i + 1}", "reasoning", tier))
    cases.append(_one("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? "
                      "Reply with only 'Answer: yes' or 'Answer: no'.", ("yes_no", {"expected": "yes"}), 16, "short",
                      "reason_4", "reasoning"))

    cases += [
        _one("List exactly 5 bullet points about caching. Each line starts with '- ' and each bullet has fewer than "
             "12 words. No other text.", ("bullets", {"count": 5, "max_words": 12}), 120, "short", "instr_1",
             "instruction", "smoke"),
        _one("Write one sentence about the sea that uses the word 'blue' exactly once and ends with an exclamation "
             "mark. Output only the sentence.", ("sentence", {"word": "blue", "times": 1, "ending": "!"}), 48,
             "short", "instr_2", "instruction"),
        _one("Reply with a JSON object with keys \"name\" (a string) and \"age\" (an integer) for a fictional "
             "person. Output only the JSON.", ("json", {"keys": {"name": "string", "age": "integer"}}), 64, "short",
             "instr_3", "instruction"),
        _one("In all capital letters and in exactly three words, describe the colour of fresh grass.",
             ("caps", {"words": 3}), 16, "short", "instr_4", "instruction"),
    ]

    tool = lambda i, user, name, expect, tier="full": cases.append(_one(
        user, ("tool", {"tool": name, "expect": expect}), 110, "tooling", f"tool_{i}", "tool_call", tier, True))
    tool(1, "What is in the directory src?", "list_dir", {"path": r"\bsrc\b"}, "smoke")
    tool(2, "Find where the text prefix_cache appears under the models folder.", "search",
         {"query": "prefix_cache", "path": "models"})
    tool(3, "Show me lines 10 to 20 of README.md.", "read_file",
         {"path": r"README\.md", "start_line": "^10$", "end_line": "^20$"})
    tool(4, "Run the unit tests with python -m unittest.", "run_command", {"command": "unittest"})
    tool(5, "Find every file whose name ends in .toml.", "find_files", {"pattern": r"toml"})

    cases += [
        Case("multi_1", "multi_turn", "short", (
            Turn(user="My name is Priya and I live in Lisbon. Reply with just OK.", max_tokens=8,
                 check=("contains", {"all_of": ["ok"]})),
            Turn(user="I have 4 cats. How many legs do my cats have in total? Reply with just the number.",
                 max_tokens=8, check=("number", {"expected": 16})),
            Turn(user="What is my name and where do I live? Reply as 'name, city'.", max_tokens=16,
                 check=("contains", {"all_of": ["priya", "lisbon"]}))), "smoke"),
        Case("multi_2", "multi_turn", "short", (
            Turn(user="Let X = 17. Reply 'noted'.", max_tokens=8, check=("contains", {"all_of": ["noted"]})),
            Turn(user="Let Y = X + 8. What is Y? Reply with just the number.", max_tokens=8,
                 check=("number", {"expected": 25})),
            Turn(user="What is X times 2? Reply with just the number.", max_tokens=8,
                 check=("number", {"expected": 34}))), "full"),
        Case("multi_3", "multi_turn", "tooling", (
            Turn(user="What is in the directory src?", max_tokens=110,
                 check=("tool", {"tool": "list_dir", "expect": {"path": r"\bsrc\b"}})),
            Turn(tool_results=("prefix_session.py\nprefix_cache.py\nlinear_state.py",), max_tokens=110,
                 check=("contains", {"any_of": ["prefix_cache", "prefix_session", "linear_state"],
                                     "none_of": ["<tool_call>"]}))),
             "full", True),
    ]

    for position, tier in (("early", "smoke"), ("middle", "full"), ("late", "smoke")):
        topic, code_value, _ = NEEDLES[position]
        cases.append(_one(f"According to the reference notes in your instructions, what is the {topic}? "
                          "Reply with only the value.", ("contains", {"all_of": [code_value]}), 24, "needles",
                          f"retrieve_{position}", f"retrieval_{position}", tier))
    cases.append(_one("According to the reference notes in your instructions, what is the on-call escalation phone "
                      "extension, and what is the release branch for the autumn cycle? Reply as 'extension, branch'.",
                      ("contains", {"all_of": ["7390", "heron-2026-q4"], "none_of": ["7319", "finch"]}), 32, "needles",
                      "retrieve_two", "retrieval_late"))
    return cases


def select(cases, tier="smoke", ids=None):
    if ids:
        wanted = set(ids)
        chosen = [c for c in cases if c.id in wanted]
        if {c.id for c in chosen} != wanted:
            raise ValueError(f"unknown cases: {sorted(wanted - {c.id for c in chosen})}")
        return chosen
    if tier not in ("smoke", "full"):
        raise ValueError("tier must be smoke or full")
    return [c for c in cases if tier == "full" or c.tier == "smoke"]


# -- running -------------------------------------------------------------------------------------

def digest(ids) -> str:
    return hashlib.sha256(repr([int(i) for i in ids]).encode()).hexdigest()[:16]


def run_case(backend, case: Case, systems: dict, *, prefix_report=lambda backend: None, now=time.perf_counter) -> dict:
    """One fresh conversation. Every turn's token ids, text, check outcome, timing and prefix-cache report."""
    turns = []
    for index, turn in enumerate(case.turns):
        if index == 0:
            backend.open_conversation(systems[case.system], turn.user, tools=TOOLS if case.tools else None,
                                      enable_thinking=False)
        elif turn.tool_results:
            backend.append_tool_results(list(turn.tool_results), enable_thinking=False)
        else:
            backend.append_user(turn.user, enable_thinking=False)
        pending = len(backend.pending)
        prompt_digest = digest([*getattr(backend, "tape", ()), *backend.pending])
        began = now()
        text, stats = backend.generate(turn.max_tokens)
        wall = now() - began
        ids = [int(i) for i in (backend.tape[-stats.tokens:] if stats.tokens else [])]
        name, kwargs = turn.check
        passed, detail = CHECKERS[name](text, **kwargs)
        try:
            calls = [{"name": n, "arguments": a} for n, a in parse_tool_calls(text)]
        except ValueError as error:
            calls = [{"error": str(error)}]
        schema_issues = tool_schema_issues(text)
        turns.append({
            "turn": index, "prompt_tokens_added": pending, "prompt_digest": prompt_digest,
            "text": text, "ids": ids, "digest": digest(ids),
            "generated": len(ids), "finish": getattr(stats, "finish", None), "passed": bool(passed), "detail": detail, "tool_calls": calls, "schema_issues": schema_issues,
            "prefill_s": getattr(stats, "prefill_seconds", None), "wall_s": wall, "prefix": prefix_report(backend) if index == 0 else None,
            "pool_release": getattr(backend, "last_pool_release", None) if index == 0 else None,
            "memory_report": getattr(backend, "last_memory_report", None),
            "memory_after": getattr(backend, "last_memory_after", None)})
    return {"case": case.id, "category": case.category, "system": case.system,
            "passed": all(t["passed"] for t in turns), "turns": turns}


def run_suite(backend, cases, systems, *, event=lambda *a, **k: None, save=lambda results: None,
              prefix_report=lambda backend: None, seed=0) -> list:
    """Run ``cases`` in order, each as a fresh conversation, saving after every case so a guard stop keeps
    what finished. Greedy decoding is set explicitly; the seed only matters if a case is run with sampling."""
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:                                      # the unit tests may run without MLX
        pass
    results = []
    for number, case in enumerate(cases):
        if number:
            backend.reset()
        result = run_case(backend, case, systems, prefix_report=prefix_report)
        results.append(result)
        save(results)
        event("quality_case", case=case.id, passed=result["passed"])
    return results


def describe(systems: dict, tokenizer=None) -> dict:
    out = {}
    for key, text in systems.items():
        out[key] = {"sha": hashlib.sha256(text.encode()).hexdigest()[:16], "chars": len(text)}
        if tokenizer is not None:
            out[key]["tokens"] = len(tokenizer.encode(text, add_special_tokens=False))
    return out
