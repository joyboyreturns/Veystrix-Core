"""The REAL Qwen Code client (0.22.0, headless) against the REAL macqwen HTTP server and the REAL tokenizer/chat template.

SCOPE, stated plainly: the model is SCRIPTED (a policy function reads the rendered prompt and returns text). These tests
prove protocol: provider config, bearer auth, streaming, usage, tool-call parsing in both directions, parallel calls,
approval modes, per-request reasoning effort, overflow errors, cancellation and token accounting. They say NOTHING about
whether the real checkpoint can prefill a prompt of this size on this Mac.

Skipped when the Qwen Code binary or the checkpoint tokenizer is absent. Each case runs `qwen -p` in a temp repo with an
isolated HOME, so the user's own ~/.qwen is never read or written.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

from macqwen import preferences
from macqwen.server import MacqwenHTTPServer, ModelService
from macqwen.tests.test_agentic_long_fixtures import TOKENIZER, RealTokenizerBackend
from macqwen.tests.test_server import FakeSession
from models.flashnext.tests.bench.fixtures import agentic_fixture as F

DEFAULT_QWEN = (
    "/opt/homebrew/bin/qwen"
    if os.path.exists("/opt/homebrew/bin/qwen")
    else "/Applications/Qwen Code Desktop.app/Contents/Resources/runtime/qwen-code/bin/qwen"
)
QWEN = os.environ.get("QWEN_CODE_BIN", DEFAULT_QWEN)
MODEL = "macqwen-flashnext"
RUN_TIMEOUT = 240


def tool_call(name, **arguments):
    return F.tool_call_text({"name": name, "arguments": arguments})


class PolicyBackend(RealTokenizerBackend):
    """Real tokenizer; the reply comes from ``policy(prompt_text) -> str`` instead of a fixed list."""

    def __init__(self, tokenizer, policy):
        super().__init__(tokenizer, [])
        self.policy = policy
        self.prompts = []
        self.cancelled = False
        self.completed = 0

    def generate(self, max_tokens, out=None):
        prompt = self.tokenizer.decode(self.tape + self.pending)
        self.prompts.append(prompt)
        text = self.policy(prompt)
        if text == "__SLOW__":
            try:
                for _ in range(400):
                    if out:
                        out(".")
                    time.sleep(0.1)
            except BaseException:
                self.cancelled = True
                raise
            text = "never reached"
        if prompt.rstrip(" ").endswith("<think>\n") and "</think>" not in text:
            text = "(scripted reasoning)\n</think>\n\n" + text      # a real model closes its thinking before answering
        self.replies = [text]
        result = super().generate(max_tokens, out)
        self.completed += 1
        return result


def last_turn(prompt):
    """Everything after the final REAL user message. The chat template renders a tool result as a user turn that starts with
    <tool_response>, so those turns belong to the current exchange and must not end it."""
    parts = prompt.split("<|im_start|>user\n")
    index = len(parts) - 1
    while index > 0 and parts[index].startswith("<tool_response>"):
        index -= 1
    return "<|im_start|>user\n".join(parts[index:])


def tool_responses(prompt):
    return re.findall(r"<tool_response>\n(.*?)\n</tool_response>", last_turn(prompt), re.S)


class Harness:
    """A real MacqwenHTTPServer on an ephemeral port with a policy backend; records what each request carried."""

    def __init__(self, policy, api_key=None, limit=None, max_cold_prefill=None):
        session = FakeSession([])
        session.backend = PolicyBackend(TOKENIZER, policy)
        session.preferences = dict(preferences.DEFAULTS, model="flashnext", thinking_enabled=False, effort="medium")
        self.session, self.backend = session, session.backend
        self.service = ModelService(session, context_limit=limit, max_cold_prefill=max_cold_prefill)
        self.requests = []
        original = self.service.complete

        def recording(messages, tools, max_tokens, *args, **kwargs):
            entry = {"tools": [t["function"]["name"] for t in tools or []], "messages": len(messages),
                     "reasoning": self.service.request_reasoning().as_dict()}
            self.requests.append(entry)
            result = original(messages, tools, max_tokens, *args, **kwargs)
            entry["prompt_tokens"] = result.stats.prompt_tokens
            entry["finish"] = "tool_calls" if result.tool_calls else result.stats.finish
            entry["tool_calls"] = [c["name"] for c in result.tool_calls]
            return result

        self.service.complete = recording
        self.server = MacqwenHTTPServer(("127.0.0.1", 0), self.service, api_key)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


# The shipped ~/.qwen/settings.json excludes Qwen Code's non-essential built-in tools: their schemas alone are ~9K tokens.
EXCLUDED_TOOLS = ["agent", "cron_create", "cron_delete", "cron_list", "enter_worktree", "exit_worktree", "get_goal",
                  "list_agents", "loop_wakeup", "monitor", "notebook_edit", "read_mcp_resource", "record_artifact",
                  "report_findings", "send_message", "skill", "task_stop", "todo_write", "tool_search", "update_goal",
                  "web_fetch", "zoom_image"]


def write_settings(home, port, effort=None, context=65536, exclude=True):
    os.makedirs(os.path.join(home, ".qwen"), exist_ok=True)
    settings = {
        "security": {"auth": {"selectedType": "openai"}},
        "modelProviders": {"openai": [{
            "id": MODEL, "name": "Veystrix test", "baseUrl": f"http://127.0.0.1:{port}/v1", "envKey": "VEYSTRIX_API_KEY",
            "generationConfig": {"timeout": 600000, "streamIdleTimeoutMs": 300000, "maxRetries": 0, "contextWindowSize": context}}]},
        "model": {"name": MODEL},
        "memory": {"enableManagedAutoMemory": False, "enableManagedAutoDream": False, "enableAutoSkill": False},
        "ui": {"enableFollowupSuggestions": False},
    }
    if effort:
        settings["model"]["reasoningEffort"] = effort
    if exclude:
        settings["tools"] = {"exclude": EXCLUDED_TOOLS}
    with open(os.path.join(home, ".qwen", "settings.json"), "w") as handle:
        json.dump(settings, handle)


def failed(out):
    """Qwen Code reports a refused request as exit 0 with result text '[API Error: ...]', so check all three."""
    return out.code != 0 or bool(out.result.get("is_error")) or "API Error" in out.text


def run_qwen(home, repo, prompt, key="local", mode="plan", wait=True):
    path = os.environ.get("PATH", "")
    for p in ("/opt/homebrew/bin", "/usr/local/bin"):
        if p not in path.split(":"):
            path = f"{p}:{path}"
    env = dict(os.environ, HOME=home, VEYSTRIX_API_KEY=key, NO_COLOR="1", PATH=path)
    argv = [QWEN, "-p", prompt, "-o", "json", "--approval-mode", mode]
    if not wait:
        return subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)      # Qwen Code relaunches itself: kill the group, not one pid
    done = subprocess.run(argv, cwd=repo, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    events = []
    try:
        events = json.loads(done.stdout)
    except ValueError:
        pass
    result = next((e for e in reversed(events) if isinstance(e, dict) and e.get("type") == "result"), {})
    err_obj = result.get("error")
    err_msg = err_obj.get("message", "") if isinstance(err_obj, dict) else (str(err_obj) if err_obj else "")
    text = result.get("result") or err_msg or done.stderr
    return SimpleNamespace(code=done.returncode, events=events, result=result, text=text,
                           stdout=done.stdout, stderr=done.stderr)


@unittest.skipIf(not os.path.exists(QWEN), "Qwen Code binary not present")
@unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
class QwenCodeClientTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self._tmp.name, "home")
        self.repo = os.path.join(self._tmp.name, "repo")
        os.makedirs(self.repo)
        with open(os.path.join(self.repo, "notes.txt"), "w") as handle:
            handle.write("alpha\nMARKER-7731\nomega\n")
        with open(os.path.join(self.repo, "other.txt"), "w") as handle:
            handle.write("second file\nMARKER-9902\n")
        self.harness = None

    def tearDown(self):
        if self.harness:
            self.harness.close()
        self._tmp.cleanup()

    def start(self, policy, effort=None, **kwargs):
        self.harness = Harness(policy, **kwargs)
        write_settings(self.home, self.harness.port, effort)
        return self.harness

    # -- provider, auth, streaming, usage ------------------------------------------------------------------------
    def test_models_endpoint_names_the_configured_id_and_reports_context_honestly(self):
        import urllib.request
        harness = self.start(lambda prompt: "x", limit=12288)
        with urllib.request.urlopen(f"http://127.0.0.1:{harness.port}/v1/models") as response:
            card = json.load(response)["data"][0]
        self.assertEqual(card["id"], MODEL)
        self.assertEqual(card["context"]["configured_server_limit"], 12288)
        self.assertIn("native_positional_capability", card["context"])
        self.assertIn("validated_on_this_machine", card["context"])
        self.assertIn("efforts", json.dumps(card["reasoning"]))

    def test_plain_prompt_streams_a_reply_with_usage_and_default_effort(self):
        harness = self.start(lambda prompt: "PLAIN-OK")
        out = run_qwen(self.home, self.repo, "Say hello")
        self.assertEqual(out.code, 0, out.stderr[-400:])
        self.assertFalse(out.result.get("is_error"), out.result)
        self.assertIn("PLAIN-OK", out.text)
        self.assertEqual(len(harness.requests), 1, "exactly one request per prompt with background features off")
        self.assertEqual(harness.requests[0]["reasoning"]["source"], "server-default")
        self.assertGreater(harness.requests[0]["prompt_tokens"], 1000)
        usage = out.result.get("usage") or {}
        self.assertTrue(usage, "usage must reach the client")

    def test_a_wrong_bearer_key_is_refused_and_nothing_is_generated(self):
        harness = self.start(lambda prompt: "SECRET-REPLY", api_key="the-real-key")
        out = run_qwen(self.home, self.repo, "Say hello", key="wrong-key")
        self.assertTrue(failed(out), (out.code, out.text))
        self.assertRegex(out.text, r"401|invalid API key")
        self.assertNotIn("SECRET-REPLY", out.stdout)
        self.assertEqual(harness.backend.completed, 0)
        good = run_qwen(self.home, self.repo, "Say hello", key="the-real-key")
        self.assertIn("SECRET-REPLY", good.text)

    # -- tools ---------------------------------------------------------------------------------------------------
    def test_sequential_read_only_tools_return_results_and_continue(self):
        notes = os.path.join(self.repo, "notes.txt")

        def policy(prompt):
            seen = tool_responses(prompt)
            if not seen:
                return tool_call("glob", pattern="*.txt")
            if len(seen) == 1:
                return tool_call("read_file", file_path=notes)
            found = re.search(r"MARKER-\d+", seen[-1])
            return f"FINAL {found.group(0) if found else 'none'}"

        harness = self.start(policy)
        out = run_qwen(self.home, self.repo, "Find the marker in notes.txt using tools.")
        self.assertEqual(out.code, 0, out.stderr[-400:])
        self.assertIn("FINAL MARKER-7731", out.text)
        self.assertEqual([r["tool_calls"] for r in harness.requests], [["glob"], ["read_file"], []])
        self.assertEqual([r["finish"] for r in harness.requests], ["tool_calls", "tool_calls", "stop"])
        self.assertNotIn("<tool_call>", out.text)
        self.assertNotIn("<function=", out.stdout)
        # the continuation extends the live state: the second and third prompts were not rebuilt from scratch
        self.assertEqual(harness.service.reused, 2)

    def test_two_parallel_tool_calls_in_one_reply_both_run_and_both_results_return(self):
        def policy(prompt):
            seen = tool_responses(prompt)
            if not seen:
                return (tool_call("read_file", file_path=os.path.join(self.repo, "notes.txt")) + "\n"
                        + tool_call("read_file", file_path=os.path.join(self.repo, "other.txt")))
            return "BOTH " + " ".join(sorted(set(re.findall(r"MARKER-\d+", " ".join(seen)))))

        harness = self.start(policy)
        out = run_qwen(self.home, self.repo, "Read both files.")
        self.assertEqual(out.code, 0, out.stderr[-400:])
        self.assertIn("BOTH MARKER-7731 MARKER-9902", out.text)
        self.assertEqual(harness.requests[0]["tool_calls"], ["read_file", "read_file"])
        self.assertEqual(len(tool_responses(harness.backend.prompts[1])), 2)

    def test_a_call_with_an_argument_the_tool_does_not_accept_is_refused_by_the_server_and_nothing_runs(self):
        """The server validates arguments against the advertised schema; the client sees an error, never a half-valid call."""
        def policy(prompt):
            return tool_call("read_file", wrong_argument="x")

        harness = self.start(policy)
        out = run_qwen(self.home, self.repo, "Read notes.txt")
        self.assertTrue(failed(out), out.text)
        self.assertIn("wrong_argument", out.text)
        self.assertEqual(len(harness.requests), 1)
        self.assertNotIn("<tool_call>", out.stdout)

    def test_approved_edit_in_a_temp_repo_changes_the_file_and_plan_mode_does_not(self):
        target = os.path.join(self.repo, "created.txt")

        def policy(prompt):
            if not tool_responses(prompt):
                return tool_call("write_file", file_path=target, content="written by the scripted agent\n")
            return "WROTE"

        harness = self.start(policy)
        run_qwen(self.home, self.repo, "Create created.txt", mode="plan")
        self.assertFalse(os.path.exists(target), "plan mode must not write")
        harness.backend.prompts.clear()
        out = run_qwen(self.home, self.repo, "Create created.txt", mode="yolo")
        self.assertIn("write_file", harness.requests[-2]["tools"] if len(harness.requests) > 1 else [], harness.requests)
        self.assertTrue(os.path.exists(target), (out.text, out.stderr[-300:]))
        with open(target) as handle:
            self.assertEqual(handle.read(), "written by the scripted agent\n")

    # -- reasoning effort ----------------------------------------------------------------------------------------
    def test_effort_changes_low_high_medium_xhigh_low_reach_the_server_and_the_prompt_per_request(self):
        harness = self.start(lambda prompt: "ok")
        sequence = ("low", "high", "medium", "xhigh", "low")
        for level in sequence:
            write_settings(self.home, harness.port, level)             # what `/effort <level>` persists
            out = run_qwen(self.home, self.repo, "hello")
            self.assertEqual(out.code, 0, (level, out.stderr[-300:]))
        self.assertEqual([r["reasoning"]["effort"] for r in harness.requests], list(sequence))
        self.assertTrue(all(r["reasoning"]["wire"] == {"reasoning": {"effort": r["reasoning"]["effort"]}}
                            for r in harness.requests))
        prompts = harness.backend.prompts
        self.assertIn("Reasoning effort is set to low.", prompts[0])
        self.assertNotIn("Reasoning effort is set to", prompts[2])
        self.assertIn("Reasoning effort is set to xhigh.", prompts[3])
        self.assertIn("Reasoning effort is set to low.", prompts[4])
        self.assertEqual(len({prompts[0], prompts[1], prompts[2], prompts[3]}), 4)
        self.assertEqual(harness.session.preferences["effort"], "medium", "nothing was stored on the shared session")
        self.assertFalse(harness.session.preferences["thinking_enabled"])

    def test_max_is_sent_as_xhigh_by_the_client_and_accepted(self):
        harness = self.start(lambda prompt: "ok")
        write_settings(self.home, harness.port, "max")
        out = run_qwen(self.home, self.repo, "hello")
        self.assertEqual(out.code, 0, out.stderr[-300:])
        self.assertEqual(harness.requests[0]["reasoning"]["effort"], "xhigh")

    # -- context accounting, overflow, cancellation --------------------------------------------------------------
    def test_prompt_token_accounting_tracks_the_real_tokenizer_at_4k_8k_12k(self):
        harness = self.start(lambda prompt: "ok", limit=65536, max_cold_prefill=32768)
        baseline = None
        measured = {}
        for size in (0, 4096, 8192, 12288):
            body = F.build(TOKENIZER, size)["messages"][1]["content"] if size else "hello"
            out = run_qwen(self.home, self.repo, body)
            self.assertEqual(out.code, 0, (size, out.stderr[-300:]))
            tokens = harness.requests[-1]["prompt_tokens"]
            measured[size] = tokens
            if size == 0:
                baseline = tokens
        for size in (4096, 8192, 12288):
            body = F.build(TOKENIZER, size)["messages"][1]["content"]
            added = measured[size] - baseline
            self.assertAlmostEqual(added, len(TOKENIZER.encode(body, add_special_tokens=False)), delta=64, msg=(size, measured))
        # Qwen Code's own system prompt plus tool schemas is already this large before any user content:
        self.assertGreater(baseline, 2000, measured)
        self.assertLess(baseline, 16384, "even trimmed, Qwen Code must leave room inside the long tier")
        print(f"\nqwen-code request sizes (server prompt_tokens): {measured}")

    def test_untrimmed_qwen_code_does_not_fit_any_admitted_tier_which_is_why_the_config_excludes_tools(self):
        harness = self.start(lambda prompt: "ok", limit=65536)
        write_settings(self.home, harness.port, exclude=False)
        out = run_qwen(self.home, self.repo, "hello", mode="yolo")
        self.assertEqual(out.code, 0, out.stderr[-300:])
        full = harness.requests[-1]
        self.assertGreaterEqual(len(full["tools"]), 10)
        self.assertGreater(full["prompt_tokens"], 10000)
        write_settings(self.home, harness.port, exclude=True)
        out = run_qwen(self.home, self.repo, "hello", mode="yolo")
        trimmed = harness.requests[-1]
        self.assertLessEqual(len(trimmed["tools"]), 8)
        self.assertLess(trimmed["prompt_tokens"], 12000)
        print(f"\nqwen-code first request: untrimmed {full['prompt_tokens']} tokens / {len(full['tools'])} tools, "
              f"trimmed {trimmed['prompt_tokens']} tokens / {len(trimmed['tools'])} tools")

    def test_a_prompt_over_the_server_limit_is_refused_before_any_generation_with_the_error_visible(self):
        harness = self.start(lambda prompt: "SHOULD-NOT-RUN", limit=1024)
        out = run_qwen(self.home, self.repo, "hello")
        self.assertTrue(failed(out), (out.code, out.text))
        self.assertNotIn("SHOULD-NOT-RUN", out.stdout)
        self.assertEqual(harness.backend.completed, 0)
        self.assertEqual(harness.backend.prompts, [])
        self.assertRegex(out.text.lower(), r"context|too long|maximum|exceed")

    def test_killing_the_client_mid_generation_stops_the_server_side_generation(self):
        harness = self.start(lambda prompt: "__SLOW__")
        process = run_qwen(self.home, self.repo, "hello", wait=False)
        deadline = time.time() + 60
        while time.time() < deadline and not harness.backend.prompts:
            time.sleep(0.2)
        self.assertTrue(harness.backend.prompts, "the request never reached the server")
        time.sleep(1.0)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        deadline = time.time() + 20
        while time.time() < deadline and not harness.backend.cancelled:
            time.sleep(0.2)
        self.assertTrue(harness.backend.cancelled, "generation kept running after the client went away")

    def test_two_sessions_with_different_efforts_do_not_leak_into_each_other(self):
        harness = self.start(lambda prompt: "ok")
        other_home = os.path.join(self._tmp.name, "home2")
        write_settings(other_home, harness.port, "low")
        write_settings(self.home, harness.port, "xhigh")
        for _ in range(2):
            run_qwen(self.home, self.repo, "session A")
            run_qwen(other_home, self.repo, "session B")
        self.assertEqual([r["reasoning"]["effort"] for r in harness.requests], ["xhigh", "low", "xhigh", "low"])


if __name__ == "__main__":
    unittest.main()
