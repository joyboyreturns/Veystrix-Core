"""End-to-end multi-tool test of DeepSeek Harness against Veystrix server with scripted backend.

Tests the full 3-turn sequence:
- Turn 1: Model emits `glob` tool call.
- Harness: executes `glob` in workspace.
- Turn 2: Model receives glob results and emits `grep` tool call for `batch_size`.
- Harness: executes `grep` in workspace.
- Turn 3: Model receives grep results and produces the final answer.
- Harness: completes with code 0.
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/Users/vincent/Documents/Veystrix")
sys.path.insert(0, str(REPO))

from macqwen import server as S
from macqwen.tests.test_server import FakeBackend, FakeSession
from scratch.fixture_workloads import get_fixture_prompt, TOKENIZER

PORT = 8080
KEY = "test-agentic-multitool-key"
WS = Path("/tmp/veystrix_agentic_4k_fixture/workspace")
DSH_HOME = Path("/tmp/veystrix_agentic_4k_fixture/dsh_home")
DSH = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
BASE_PATCH = REPO / "docs/flashnext/dsh/veystrix-agentic.patch.yml"


class MultiTurnScriptedBackend(FakeBackend):
    call_count = 0
    turns = []

    def __init__(self, pieces):
        super().__init__(pieces)
        self.tokenizer = TOKENIZER

    def encode(self, text):
        return self.tokenizer.encode(text)

    def generate(self, max_tokens, out=None, **kw):
        type(self).call_count += 1
        count = type(self).call_count
        prompt_len = len(self.tape) + len(self.pending)
        type(self).turns.append({"turn": count, "max_tokens": max_tokens, "prompt_len": prompt_len})
        print(f"[ScriptedBackend] turn {count}: prompt_len={prompt_len}, max_tokens={max_tokens}", flush=True)

        if count == 1:
            # Turn 1: Model generates glob tool call to explore workspace
            call_text = (
                '<tool_call>\n'
                '<function=glob>\n'
                '<parameter=pattern>\n'
                '**/*\n'
                '</parameter>\n'
                '</function>\n'
                '</tool_call>'
            )
            if out:
                out(call_text)
            self.tape.extend(self.pending)
            self.pending = []
            self.tape.extend(self.encode(call_text))
            return call_text, SimpleNamespace(
                finish="tool_calls", tokens=25, prompt_tokens=prompt_len,
                prefill_seconds=0.1, prompt_rate=40000.0, seconds=0.1, rate=250.0
            )

        elif count == 2:
            # Turn 2: Model sees file list and generates grep tool call to find batch_size in config
            call_text = (
                '<tool_call>\n'
                '<function=grep>\n'
                '<parameter=path>\n'
                'config/settings.yaml\n'
                '</parameter>\n'
                '<parameter=pattern>\n'
                'batch_size\n'
                '</parameter>\n'
                '</function>\n'
                '</tool_call>'
            )
            if out:
                out(call_text)
            self.tape.extend(self.pending)
            self.pending = []
            self.tape.extend(self.encode(call_text))
            return call_text, SimpleNamespace(
                finish="tool_calls", tokens=35, prompt_tokens=prompt_len,
                prefill_seconds=0.1, prompt_rate=40000.0, seconds=0.1, rate=350.0
            )

        else:
            # Turn 3: Model receives grep results and synthesizes final answer
            answer_text = (
                "Based on the repository investigation using glob and grep:\n"
                "- Discovered workspace structure with config, service/core, service/api, and tests.\n"
                "- Inspected config/settings.yaml using grep and verified pipeline.batch_size is 64.\n"
                "- Core engine in service/core/engine.py coordinates with service/api/auth.py for authentication.\n"
                "All parameters successfully verified."
            )
            if out:
                out(answer_text)
            self.tape.extend(self.pending)
            self.pending = []
            self.tape.extend(self.encode(answer_text))
            return answer_text, SimpleNamespace(
                finish="stop", tokens=55, prompt_tokens=prompt_len,
                prefill_seconds=0.1, prompt_rate=40000.0, seconds=0.1, rate=550.0
            )


def make_patch(tier: str, tmp_dir: Path) -> Path:
    base_text = BASE_PATCH.read_text()
    if tier == "4k":
        window = 12288
    elif tier == "8k":
        window = 16384
    elif tier == "12k":
        window = 24576
    else:
        window = 12288

    modified = re.sub(r"contextWindow:\s*\d+", f"contextWindow: {window}", base_text)
    patch_file = tmp_dir / f"veystrix-{tier}.patch.yml"
    patch_file.write_text(modified)
    return patch_file


def run_tier_test(tier: str, port=PORT):
    MultiTurnScriptedBackend.call_count = 0
    MultiTurnScriptedBackend.turns = []

    tokens, prompt = get_fixture_prompt(tier)
    print(f"\n=======================================================")
    print(f"RUNNING MULTI-TURN HARNESS TEST: Tier {tier.upper()} ({tokens} tokens)")
    print(f"=======================================================")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        patch_path = make_patch(tier, tmp_dir)

        # Context limit and cold prefill limits for server
        if tier == "4k":
            srv_ctx = 8192
            srv_cold = 4096
        elif tier == "8k":
            srv_ctx = 16384
            srv_cold = 8192
        else: # 12k
            srv_ctx = 20480
            srv_cold = 12288

        session = FakeSession(["initial"])
        session.backend = MultiTurnScriptedBackend(["initial"])

        server_thread = threading.Thread(
            target=lambda: S.serve(session, "127.0.0.1", port, KEY, context_limit=srv_ctx, max_cold_prefill=srv_cold),
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)

        # Verify server /health
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=5) as res:
            health = json.loads(res.read())
            print(f"Server /health: {health['status']}, context_limit={health['context_limit']}")

        env = {**os.environ, "DSH_HOME": str(DSH_HOME), "VEYSTRIX_API_KEY": KEY}
        cmd = [DSH, "headless", "--patch", str(patch_path), "--json", prompt]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(WS), env=env, capture_output=True, text=True, timeout=90)
        elapsed = time.time() - t0

        print(f"DSH exited with code {proc.returncode} in {elapsed:.2f}s")
        if proc.returncode != 0:
            print("DSH STDERR:", proc.stderr)
            print("DSH STDOUT (tail):", proc.stdout[-1000:])
            raise RuntimeError(f"Harness failed with returncode {proc.returncode}")

        # Assertions
        assert MultiTurnScriptedBackend.call_count == 3, f"Expected 3 turns, got {MultiTurnScriptedBackend.call_count}"
        for turn_info in MultiTurnScriptedBackend.turns:
            assert turn_info["max_tokens"] >= 1000, f"Turn requested insufficient max_tokens: {turn_info}"

        # Verify stdout contained both tool calls and final answer
        stdout = proc.stdout
        assert '"tool":"glob"' in stdout, "Missing glob tool call in DSH output"
        assert '"tool":"grep"' in stdout, "Missing grep tool call in DSH output"
        assert "batch_size: 64" in stdout or "64" in stdout, "Missing final answer with batch size"
        print(f"SUCCESS: Tier {tier.upper()} multi-turn (glob -> grep -> answer) verified!")


if __name__ == "__main__":
    tier_arg = sys.argv[1] if len(sys.argv) > 1 else "4k"
    run_tier_test(tier_arg)
