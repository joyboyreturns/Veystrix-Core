"""Run the REAL DeepSeek Harness client against the REAL Veystrix HTTP server with a SCRIPTED model, and record what the
client sends: input tokens (real tokenizer), max_completion_tokens, tool names, per-turn reuse.

No model checkpoint is loaded (only its tokenizer and chat template). It measures the client and the server protocol,
not any model's intelligence or memory. Usage:

    .venv/bin/python -m models.flashnext.tests.bench.dsh_wire_probe --target 12288 --window 20480 --max-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

DSH = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
CHECKPOINT = os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
KEY = "probe-key"


def patch_yaml(port: int, window: int, max_tokens: int, profile: str) -> str:
    text = f"""- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      veystrix:
        displayName: Veystrix
        apiKeyEnv: VEYSTRIX_API_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:{port}/v1
        models:
          - id: macqwen-flashnext
            name: Qwen3.8 Flash-Next Q4
            contextWindow: {window}
            maxTokens: {max_tokens}
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: veystrix
    model: macqwen-flashnext
- id: session-title-llm
  disabled: true
"""
    if profile == "trimmed":
        text += "".join(f"- id: {i}\n  disabled: true\n" for i in (
            "tool-bash", "tool-pwsh", "tool-jobs", "tool-skill", "tool-subagent", "tool-subagent-fork",
            "tool-subagent-control", "tool-workflow", "tool-working-directory", "goal", "goal-round-driver",
            "command-goal", "tool-goal", "tool-todo", "tool-web"))
        text += "- id: tool-fs\n  config:\n    read: false\n    readImage: false\n    write: false\n    edit: false\n    applyPatch: false\n"
    return text


def make_project(root: Path, decoy: bool = True) -> None:
    """The isolated project. ``decoy=False`` leaves only the real file, so ``glob`` has one unambiguous answer."""
    from models.flashnext.tests.bench.fixtures import agentic_fixture as F
    files = ((F.REAL_FILE, F.BATCH_SIZE, F.TOKEN_EXPIRY), (F.DECOY_FILE, F.DECOY_BATCH, F.DECOY_EXPIRY))
    for name, batch, expiry in (files if decoy else files[:1]):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("service: gateway\nlimits:\n" + "\n" * 11 + f"  batch_size: {batch}\n" + "\n" * 7
                        + f"  token_expiry_seconds: {expiry}\n")


def run(target: int, window: int, max_tokens: int, profile: str, limit: int, timeout: int = 240, task_text: str | None = None, decoy: bool = True) -> dict:
    from transformers import AutoTokenizer
    from macqwen import preferences, server as S
    from macqwen.scheduler import Scheduler
    from macqwen.tests.test_agentic_long_fixtures import RealTokenizerBackend
    from macqwen.tests.test_server import FakeSession
    from models.flashnext.tests.bench.fixtures import agentic_fixture as F

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    overhead = None
    if target and task_text is None:
        overhead = _overhead(tokenizer, profile)         # measured BEFORE this run patches anything (runs do not nest)
    replies = [F.tool_call_text(F.GLOB_CALL), F.tool_call_text(F.GREP_CALL), F.final_answer()]
    session = FakeSession([])
    session.backend = RealTokenizerBackend(tokenizer, replies)
    session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False)
    service = S.ModelService(session, context_limit=limit)
    records = []
    original_chat = S.MacqwenHandler._chat

    def chat(self, payload):
        records.append({"max_completion_tokens": payload.get("max_completion_tokens"),
                        "max_tokens": payload.get("max_tokens"), "tools": [t["function"]["name"] for t in payload.get("tools", [])],
                        "messages": len(payload.get("messages", [])), "roles": [m["role"] for m in payload.get("messages", [])]})
        return original_chat(self, payload)

    original_complete = service.complete

    def complete(messages, tools, max_tokens, *args, **kw):
        out = original_complete(messages, tools, max_tokens, *args, **kw)
        records[-1].update({"prompt_tokens": service.last_prompt_tokens, "served_max_tokens": max_tokens,
                            "reused": service.reused, "rebuilt": service.rebuilt,
                            "prefilled_tokens": session.backend.appended[-1],
                            "calls": [c["name"] for c in out.tool_calls], "budget": service.last_budget})
        return out

    service.complete = complete
    S.MacqwenHandler._chat = chat
    server = S.MacqwenHTTPServer(("127.0.0.1", 0), service, api_key=KEY, scheduler=Scheduler())
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work, home = Path(tmp) / "work", Path(tmp) / "dsh-home"
            work.mkdir(); home.mkdir()
            make_project(work, decoy)
            patch = Path(tmp) / "patch.yml"
            patch.write_text(patch_yaml(server.server_port, window, max_tokens, profile))
            if task_text is not None:
                task = task_text                      # a precomputed task (e.g. the 8K gate's), sized by its own file
            elif target == 0:
                task = TINY_TASK
            else:
                # size the brief so that the harness's FIRST request reaches ``target`` tokens
                fixture = F.build(tokenizer, max(512, target - overhead), tolerance=0.05)
                task = fixture["messages"][1]["content"]
            env = {**os.environ, "DSH_HOME": str(home), "VEYSTRIX_API_KEY": KEY, "NO_PROXY": "127.0.0.1"}
            done = subprocess.run([DSH, "headless", "--patch", str(patch), task], cwd=work, env=env, capture_output=True,
                                  text=True, timeout=timeout)
            return {"target": target, "harness_overhead_tokens": overhead, "target": target, "window": window, "max_tokens": max_tokens, "profile": profile,
                    "server_limit": limit, "exit": done.returncode, "requests": records,
                    "stdout_tail": done.stdout[-600:], "stderr_tail": done.stderr[-400:]}
    finally:
        server.shutdown(); server.server_close()
        S.MacqwenHandler._chat = original_chat


TINY_TASK = "List the configuration files under services."
_OVERHEAD = {}


def _overhead(tokenizer, profile: str) -> int:
    """Tokens the harness adds around a task (its system prompt, tool schemas, framing, runtime-context message),
    measured by one run with a tiny task: first-request prompt tokens minus the task's own tokens."""
    if profile not in _OVERHEAD:
        result = run(0, 40000, 1024, profile, 32768, timeout=120)
        first = next(r for r in result["requests"] if r.get("prompt_tokens"))
        _OVERHEAD[profile] = int(first["prompt_tokens"]) - len(tokenizer.encode(TINY_TASK, add_special_tokens=False))
    return _OVERHEAD[profile]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=4096)
    ap.add_argument("--window", type=int, default=0, help="client contextWindow (default: server limit + 4096)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--profile", choices=("trimmed", "default"), default="trimmed")
    ap.add_argument("--limit", type=int, default=0, help="server context limit (default: target + 4096)")
    ap.add_argument("--task-file", default="", help="use this precomputed task text (the 8K gate's) instead of building one")
    ap.add_argument("--no-decoy", action="store_true", help="the project holds only the real config file")
    a = ap.parse_args()
    limit = a.limit or a.target + 4096
    text = Path(a.task_file).read_text() if a.task_file else None
    result = run(a.target, a.window or limit + 4096, a.max_tokens, a.profile, limit, task_text=text, decoy=not a.no_decoy)
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["exit"] == 0 else 1)
