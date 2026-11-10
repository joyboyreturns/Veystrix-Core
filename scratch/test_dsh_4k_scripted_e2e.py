"""End-to-end test of DeepSeek Harness against Veystrix server with scripted backend on the 4K fixture."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/Users/vincent/Documents/Veystrix")
sys.path.insert(0, str(REPO))

from veystrix import server as S
from veystrix.tests.test_server import FakeBackend, FakeSession

PORT = 8080
KEY = "test-agentic-e2e-key"
WS = Path("/tmp/veystrix_agentic_4k_fixture/workspace")
DSH_HOME = Path("/tmp/veystrix_agentic_4k_fixture/dsh_home")
DSH = "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
PATCH = REPO / "docs/flashnext/dsh/veystrix-agentic.patch.yml"

# Load the exact calibrated 4K task prompt
from transformers import AutoTokenizer
from scratch.calibrate_fixture import SPEC_HEADER, EXTRA_SECTION, TASK_QUESTION
TASK_4K = SPEC_HEADER + EXTRA_SECTION + TASK_QUESTION

TOK_PATH = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
REAL_TOKENIZER = AutoTokenizer.from_pretrained(TOK_PATH)

class ScriptedBackend(FakeBackend):
    call_count = 0
    received_requests = []

    def __init__(self, pieces):
        super().__init__(pieces)
        self.tokenizer = REAL_TOKENIZER

    def encode(self, text):
        return self.tokenizer.encode(text)

    def generate(self, max_tokens, out=None, **kw):
        type(self).call_count += 1
        count = type(self).call_count
        print(f"[ScriptedBackend] generate turn {count}, max_tokens={max_tokens}", flush=True)

        if count == 1:
            # First turn: emit valid tool call for glob
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
                finish="stop", tokens=25, prompt_tokens=4032,
                prefill_seconds=0.2, prompt_rate=20000.0,
                seconds=0.5, rate=50.0
            )
        else:
            # Second turn: answer using the tool result
            answer_text = (
                "Based on inspecting the workspace files with glob:\n"
                "- Pipeline batch size is defined in config/settings.yaml as 64.\n"
                "- Token leeway is defined in service/api/auth.py as 30 seconds.\n"
                "- Project structure verified: service/core/engine.py coordinates with service/api/auth.py."
            )
            if out:
                out(answer_text)
            self.tape.extend(self.pending)
            self.pending = []
            self.tape.extend(self.encode(answer_text))
            return answer_text, SimpleNamespace(
                finish="stop", tokens=45, prompt_tokens=4150,
                prefill_seconds=0.1, prompt_rate=40000.0,
                seconds=0.5, rate=90.0
            )


def main():
    DSH_HOME.mkdir(parents=True, exist_ok=True)
    session = FakeSession(["initial"])
    session.backend = ScriptedBackend(["initial"])

    server_thread = threading.Thread(
        target=lambda: S.serve(session, "127.0.0.1", PORT, KEY, context_limit=8192, max_cold_prefill=4096),
        daemon=True
    )
    server_thread.start()
    time.sleep(1.0)

    # Verify server is listening
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/health",
        headers={"Authorization": f"Bearer {KEY}"}
    )
    with urllib.request.urlopen(req, timeout=5) as res:
        health_data = json.loads(res.read())
        print("Server /health:", health_data["status"], "context_limit:", health_data["context_limit"])
        assert health_data["context_limit"] == 8192

    # Launch DeepSeek Harness headless
    print("Launching DeepSeek Harness with ~4K prompt...", flush=True)
    env = {**os.environ, "DSH_HOME": str(DSH_HOME), "VEYSTRIX_API_KEY": KEY}
    cmd = [DSH, "headless", "--patch", str(PATCH), "--json", TASK_4K]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(WS), env=env, capture_output=True, text=True, timeout=90)
    duration = time.time() - t0
    print(f"DeepSeek Harness exited with code {proc.returncode} in {duration:.1f}s")
    print("=== DSH STDOUT ===")
    print(proc.stdout[-1500:] if len(proc.stdout) > 1500 else proc.stdout)
    if proc.stderr:
        print("=== DSH STDERR ===")
        print(proc.stderr[-1000:])

    assert proc.returncode == 0, f"Harness failed with code {proc.returncode}"
    assert ScriptedBackend.call_count == 2, f"Expected 2 turns (tool call + answer), got {ScriptedBackend.call_count}"
    print("SUCCESS: Full 4K scripted tool-use round trip verified!")

if __name__ == "__main__":
    main()
