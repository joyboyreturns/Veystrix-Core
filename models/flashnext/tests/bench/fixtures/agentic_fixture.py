"""Reproducible long agentic requests at an exact token size, built with the REAL tokenizer and chat template.

``build(tokenizer, target_tokens)`` returns the OpenAI-style conversation a coding harness would send first: a system
prompt, three read-only tools (glob, grep, read) and a project brief plus a task that needs two sequential tool calls.
The facts the task asks for are deliberately NOT in the brief: they appear only in the tool results the scripted
harness returns (a ``glob`` listing, then ``grep`` lines), so a model that answers without calling the tools cannot be
right, and a long-context failure shows up as a wrong value (a distractor config carries other numbers).

Deterministic: no clock, no randomness beyond a fixed seed, so a fixture is identical on every machine and its
token count is a property of the tokenizer alone (the test asserts the band, it does not assume it).
"""
from __future__ import annotations

import json
import random

TASK = ("Find the relevant configuration file, locate the configured batch size and authentication token expiry, "
        "and report the values. Use available read-only tools.")
# The 8K gate states the order of the two tools. Gate A attempt 2 (unguided task, two config files in the listing) showed the
# model re-globbing with wider patterns for three steps and never calling grep; the guided task asks for the
# glob -> grep sequence the gate must validate. The model still has to emit both calls itself, with correct arguments.
TASK_GUIDED = (TASK + " First use the glob tool to find the gateway configuration file, then use the grep tool on that "
               "file to read the batch size and the token expiry.")
FOLLOW_UP = "Which of those two values would you change first to reduce authentication traffic, and why?"

REAL_FILE = "services/gateway/config/gateway.yaml"
DECOY_FILE = "services/gateway/config/gateway.example.yaml"
BATCH_SIZE, TOKEN_EXPIRY = 1536, 5400
DECOY_BATCH, DECOY_EXPIRY = 64, 900

SYSTEM = ("You are a careful coding agent working in the user's repository. Use the read-only tools to inspect files; "
          "never guess a value you can read. Call one tool at a time, wait for its result, and answer concisely.")

TOOLS = [
    {"type": "function", "function": {"name": "glob", "description": "List files whose path matches a glob pattern.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "glob, e.g. **/*.yaml"},
                                                         "path": {"type": "string", "description": "directory to search"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents for a regular expression.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                         "case_insensitive": {"type": "boolean"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a text file, optionally a line range.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"},
                                                         "limit": {"type": "integer"}}, "required": ["path"]}}},
]

_SERVICES = ["gateway", "ledger", "notifier", "scheduler", "search", "billing", "audit", "ingest", "exporter", "profile"]
_VERBS = ["validates", "retries", "batches", "signs", "caches", "streams", "deduplicates", "rate-limits", "indexes", "replays"]
_OBJECTS = ["session records", "webhook deliveries", "audit events", "invoice lines", "search shards", "queue offsets",
            "device certificates", "export manifests", "profile snapshots", "ledger entries"]
_FAILS = ["a stale lease", "a clock skew above two seconds", "an exhausted connection pool", "a truncated frame",
          "a duplicate idempotency key", "a missing schema migration", "an expired signing key", "a full disk volume"]


def section(index: int) -> list:
    """One deterministic, content-rich module note as a list of lines (the size knob is a line count)."""
    rng = random.Random(7919 * (index + 1))
    name = _SERVICES[index % len(_SERVICES)]
    rev = index // len(_SERVICES) + 1
    lines = [f"### services/{name}-v{rev} (module {index})",
             f"The {name} v{rev} module {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} and {rng.choice(_VERBS)} "
             f"{rng.choice(_OBJECTS)} before handing them to the next stage."]
    for step in range(3):
        lines.append(f"- Failure mode {index}.{step}: {rng.choice(_FAILS)} makes it {rng.choice(_VERBS)} "
                     f"{rng.choice(_OBJECTS)} twice; the runbook says to check queue depth {rng.randint(3, 900)} "
                     f"and alert after {rng.randint(2, 40)} minutes.")
    lines.append(f"Owner rotation {rng.randint(1, 12)}; dashboards: ops/{name}/{rng.randint(100, 999)}; "
                 f"test target `make test-{name}-{rev}` runs {rng.randint(12, 480)} cases.")
    return lines


def brief(sections: int) -> str:
    head = ("# NIMBUS GATEWAY project brief\n"
            "Nimbus is a multi-service backend. This brief summarises each module's behaviour and operational notes. "
            "Configuration lives in per-service YAML files under services/<name>/config/; the example files are "
            "templates and are not what production loads. Details change between releases, so read the actual file "
            "when you need a value.\n")
    out, index = [], 0
    while len(out) < sections:                      # ``sections`` counts lines, about 45 tokens each
        out.extend(section(index))
        index += 1
    return head + "\n".join(out[:sections])


def conversation(sections: int) -> list:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": brief(sections) + "\n\n## Task\n" + TASK}]


def count_tokens(tokenizer, messages) -> int:
    text = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=True)
    return len(tokenizer.encode(text, add_special_tokens=False))


def build(tokenizer, target_tokens: int, tolerance: float = 0.03) -> dict:
    """Smallest brief whose rendered prompt reaches ``target_tokens``; the count must land within ``tolerance``."""
    low, high = 1, 1
    while count_tokens(tokenizer, conversation(high)) < target_tokens:
        low, high = high, high * 2
    while low < high:                                   # tokens grow monotonically with sections
        mid = (low + high) // 2
        if count_tokens(tokenizer, conversation(mid)) < target_tokens:
            low = mid + 1
        else:
            high = mid
    messages = conversation(low)
    tokens = count_tokens(tokenizer, messages)
    if abs(tokens - target_tokens) > tolerance * target_tokens:
        raise ValueError(f"fixture landed at {tokens} tokens for target {target_tokens}")
    return {"messages": messages, "tools": TOOLS, "tokens": tokens, "sections": low}


# ---- the scripted harness side: what the tools return, and what a correct model would do -----------------------------

GLOB_CALL = {"name": "glob", "arguments": {"pattern": "**/gateway*.yaml"}}
GREP_CALL = {"name": "grep", "arguments": {"pattern": "batch_size|token_expiry", "path": REAL_FILE}}


def glob_result() -> str:
    return "\n".join([DECOY_FILE, REAL_FILE])


def grep_result() -> str:
    return f"{REAL_FILE}:14:  batch_size: {BATCH_SIZE}\n{REAL_FILE}:22:  token_expiry_seconds: {TOKEN_EXPIRY}"


def tool_call_text(call: dict) -> str:
    """The Qwen XML form a model emits for a call."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in call["arguments"].items())
    return f"<tool_call>\n<function={call['name']}>\n{body}</function>\n</tool_call>"


def final_answer() -> str:
    return (f"The configuration file is {REAL_FILE}. The batch size is {BATCH_SIZE} and the authentication token "
            f"expiry is {TOKEN_EXPIRY} seconds (the example file's {DECOY_BATCH} and {DECOY_EXPIRY} are templates).")


def check_answer(text: str) -> dict:
    """Mechanical correctness of a final answer: both real values present, and no decoy value stated as the answer.
    A decoy mentioned inside parentheses (as a template value) is allowed; anywhere else it counts as claimed."""
    import re

    outside = re.sub(r"\([^)]*\)", "", text)
    return {"batch_size": str(BATCH_SIZE) in text, "token_expiry": str(TOKEN_EXPIRY) in text,
            "decoy_claimed": str(DECOY_BATCH) in outside or str(DECOY_EXPIRY) in outside}


def as_json(fixture: dict) -> str:
    return json.dumps({"tokens": fixture["tokens"], "sections": fixture["sections"]})
