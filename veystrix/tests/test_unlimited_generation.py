"""Checkpoint-free regression tests for uncapped generation across Veystrix."""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from veystrix import agent, context_budget, preferences, reasoning, reasoning_policy
from veystrix.harness_configure import load_yaml12, run as configure_harnesses
from veystrix.harness_models import deepseek_config, qwen_config
from veystrix.model_registry import InstalledModel
from veystrix.server import (
    ClientGone,
    ModelService,
    ModelToolCallError,
    RequestError,
    VeystrixHandler,
    model_card,
    model_entry,
)


class _Tok:
    def __init__(self, rendered="prompt", native_window=None):
        self.rendered = rendered
        if native_window is not None:
            self.model_max_length = native_window

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return self.rendered


class _Backend:
    def __init__(self, pieces, finish="stop", native_window=None):
        self.tokenizer = _Tok(native_window=native_window)
        self.pieces = pieces
        self.finish = finish
        self.pending = []
        self.tape = []
        self._replay_needed = False
        self.reset_count = 0
        self.seen_max_tokens = []
        self.canceled = False

    def reset(self):
        self.reset_count += 1
        self.pending = []
        self.tape = []
        self._replay_needed = False

    def encode(self, text):
        return [ord(c) for c in text]

    def common_prefix(self, ids):
        n = min(len(ids), len(self.tape))
        i = 0
        while i < n and self.tape[i] == ids[i]:
            i += 1
        return i

    def append_tokens(self, ids):
        self.pending.extend(int(t) for t in ids)

    def generate(self, max_tokens, out=None, should_cancel=None):
        self.seen_max_tokens.append(max_tokens)
        text_parts = []
        for piece in self.pieces:
            if should_cancel is not None and should_cancel():
                self.canceled = True
                break
            text_parts.append(piece)
            if out is not None:
                out(piece)
        text = "".join(text_parts)
        prompt_count = len(self.pending) + len(self.tape)
        self.tape.extend(self.pending)
        self.pending = []
        self.tape.extend(self.encode(text))
        tok_count = max(len(self.pieces), 1)
        return text, SimpleNamespace(
            finish=self.finish, tokens=tok_count, prompt_tokens=prompt_count,
            prefill_seconds=0.01, rate=10.0,
        )


class _Session:
    def __init__(self, backend, thinking_enabled=False, effort="medium"):
        self.backend = backend
        self.preferences = dict(
            preferences.DEFAULTS,
            model="flashnext",
            thinking_enabled=thinking_enabled,
            effort=effort,
        )

    def reset(self):
        self.backend.reset()


def _handler_for(service):
    handler = VeystrixHandler.__new__(VeystrixHandler)
    handler.server = SimpleNamespace(
        service=service,
        allowed_origins=(),
        scheduler=SimpleNamespace(snapshot=lambda: {"active": None, "waiting": 0}),
    )
    handler.headers = {}
    handler.wfile = io.BytesIO()
    handler._sse_open = False
    handler._response_model = getattr(service, "model", "veystrix-flashnext")
    handler.status_calls = []
    handler.json_calls = []
    handler.error_calls = []
    handler.sse_events = []

    handler.send_response = lambda status, msg=None: handler.status_calls.append(status)
    handler.send_header = lambda *a, **kw: None
    handler.end_headers = lambda: None
    handler._client_gone = lambda: False

    orig_json = VeystrixHandler._json.__get__(handler, VeystrixHandler)
    orig_error = VeystrixHandler._error.__get__(handler, VeystrixHandler)

    def rec_json(value, status=200, headers=None):
        handler.json_calls.append((value, status, headers))
        return orig_json(value, status, headers)

    def rec_error(message, status=400, code=None, headers=None, kind=None):
        handler.error_calls.append((str(message), status, code))
        return orig_error(message, status, code, headers, kind)

    def rec_sse(value, event=None):
        handler.sse_events.append((event, value))
        prefix = f"event: {event}\n" if event else ""
        handler.wfile.write((prefix + "data: " + json.dumps(value) + "\n\n").encode())

    handler._json = rec_json
    handler._error = rec_error
    handler._sse = rec_sse
    return handler


class SentinelDefaultsTests(unittest.TestCase):
    def test_uncapped_sentinels_everywhere(self):
        self.assertEqual(preferences.DEFAULT_ANSWER_TOKENS, -1)
        self.assertEqual(preferences.DEFAULT_PLAIN_ANSWER_TOKENS, -1)
        self.assertEqual(preferences.DEFAULT_THINK_TOKENS, -1)
        self.assertEqual(preferences.DEFAULTS["max_tokens"], -1)
        self.assertEqual(preferences.DEFAULTS["think_budget"], -1)

        self.assertEqual(preferences.answer_limit({"max_tokens": -1}), -1)
        self.assertEqual(preferences.answer_limit({"max_tokens": None}), -1)
        self.assertEqual(preferences.generation_limit({"max_tokens": -1, "thinking_enabled": True}), -1)
        self.assertEqual(preferences.think_limit({"think_budget": -1, "thinking_enabled": True}), -1)
        self.assertEqual(preferences.separate_think_limit({"think_budget": -1, "thinking_enabled": True}), -1)
        self.assertEqual(preferences.separate_think_limit({"think_budget": None, "thinking_enabled": True}), -1)

        lim = agent.Limits()
        self.assertEqual(lim.max_tokens, -1)
        self.assertEqual(lim.think_tokens, -1)
        # Turn-count ceiling stays intentional and finite.
        self.assertEqual(lim.max_turns, 24)

    def test_context_budget_uncapped_uses_full_remaining_capacity(self):
        for req in (-1, None):
            b = context_budget.evaluate(262144, 1000, req)
            self.assertEqual(b.requested_output, -1)
            self.assertEqual(b.reserved_output, 261144)
            self.assertEqual(b.remaining, 0)
            self.assertTrue(b.continuation_ok)
            self.assertIsNone(context_budget.too_small_for_tools(req, True))

        tight = context_budget.evaluate(4096, 3000, -1)
        self.assertEqual(tight.reserved_output, 1096)
        self.assertFalse(tight.continuation_ok)

    def test_speculative_normalizes_uncapped_sentinel(self):
        from models.flashnext import speculative

        self.assertEqual(
            speculative._normalize_max_tokens(-1),
            speculative.UNLIMITED_GENERATION_HORIZON,
        )
        self.assertEqual(
            speculative._normalize_max_tokens(None),
            speculative.UNLIMITED_GENERATION_HORIZON,
        )
        self.assertEqual(speculative._normalize_max_tokens(77), 77)
        self.assertGreater(speculative.UNLIMITED_GENERATION_HORIZON, 262144)


class HTTPUncappedProtocolTests(unittest.TestCase):
    def test_omitted_and_null_limits_across_all_three_apis(self):
        # 1) Service without configured/native window passes -1 unchanged.
        backend_open = _Backend(["ok"])
        service_open = ModelService(_Session(backend_open))
        handler_open = _handler_for(service_open)

        for route, body in (
            ("_chat", {"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}]}),
            ("_chat", {"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": None}),
            ("_anthropic", {"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}]}),
            ("_responses", {"model": "veystrix-flashnext", "input": "hi", "max_output_tokens": None}),
        ):
            backend_open.seen_max_tokens.clear()
            getattr(handler_open, route)(body)
            self.assertEqual(backend_open.seen_max_tokens, [-1], f"{route} failed uncapped dispatch: {body}")

        # 2) Service with context_limit clamps uncapped requests to real remaining context, never 2048/4096.
        backend_ctx = _Backend(["ok"])
        service_ctx = ModelService(_Session(backend_ctx), context_limit=16384)
        handler_ctx = _handler_for(service_ctx)
        handler_ctx._chat({"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}]})
        prompt_len = len(backend_ctx.encode("prompt"))
        self.assertEqual(backend_ctx.seen_max_tokens, [16384 - prompt_len])

    def test_explicit_positive_limit_honored_and_invalid_rejected(self):
        backend = _Backend(["ok"])
        service = ModelService(_Session(backend), context_limit=16384)
        handler = _handler_for(service)
        handler._chat({"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 500})
        self.assertEqual(backend.seen_max_tokens, [500])

        for bad in (0, -1, True, "500"):
            with self.assertRaises(RequestError):
                handler._limit({"max_tokens": bad})

    def test_models_metadata_honest_no_fake_cap(self):
        svc_open = ModelService(_Session(_Backend(["ok"], native_window=262144)))
        entry_open = model_entry(model_card(svc_open), svc_open)
        self.assertEqual(entry_open["context_window"], 262144)
        self.assertIsNone(entry_open["max_output_tokens"])
        self.assertFalse(entry_open["context_limit_enforced"])
        levels = {item["effort"] for item in model_card(svc_open)["supported_reasoning_levels"]}
        self.assertIn("xhigh", levels)

        svc_enforced = ModelService(_Session(_Backend(["ok"])), context_limit=8192)
        entry_enforced = model_entry(model_card(svc_enforced), svc_enforced)
        self.assertEqual(entry_enforced["context_window"], 8192)
        self.assertTrue(entry_enforced["context_limit_enforced"])
        self.assertEqual(entry_enforced["suggested_client_context_window"], 8192 + 4096)


class TerminationAndBackendTests(unittest.TestCase):
    def test_eos_and_context_boundary_termination(self):
        # EOS termination
        svc_eos = ModelService(_Session(_Backend(["hello"], finish="stop")), context_limit=8192)
        h_eos = _handler_for(svc_eos)
        h_eos._chat({"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(h_eos.json_calls[-1][0]["choices"][0]["finish_reason"], "stop")

        # Context-boundary finish="length" across chat, messages, responses (JSON & streaming)
        svc_len = ModelService(_Session(_Backend(["cut"], finish="length")), context_limit=8192)
        h_len = _handler_for(svc_len)
        h_len._chat({"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(h_len.json_calls[-1][0]["choices"][0]["finish_reason"], "length")

        h_len._anthropic({"model": "veystrix-flashnext", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(h_len.json_calls[-1][0]["stop_reason"], "max_tokens")

        h_len._responses({"model": "veystrix-flashnext", "input": "hi"})
        resp = h_len.json_calls[-1][0]
        self.assertEqual(resp["status"], "incomplete")
        self.assertEqual(resp["incomplete_details"], {"reason": "max_output_tokens"})

        h_len.sse_events.clear()
        h_len._responses({"model": "veystrix-flashnext", "input": "hi", "stream": True})
        self.assertEqual(h_len.sse_events[-1][0], "response.incomplete")
        self.assertEqual(h_len.sse_events[-1][1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})

    def test_streaming_completion_and_client_cancellation(self):
        svc = ModelService(_Session(_Backend(["one ", "two."])))
        h = _handler_for(svc)
        h._chat({
            "model": "veystrix-flashnext",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        chunks = [val for _, val in h.sse_events]
        contents = [c["choices"][0]["delta"].get("content") for c in chunks if c.get("choices")]
        self.assertIn("one ", contents)
        self.assertTrue(any("usage" in c for c in chunks))
        self.assertIn(b"data: [DONE]\n\n", h.wfile.getvalue())

        # Cancellation: both should_cancel hook and streaming gone check abort cleanly.
        backend_cancel = _Backend(["tok1 ", "tok2 ", "tok3 "])
        svc_cancel = ModelService(_Session(backend_cancel))
        svc_cancel.complete([{"role": "user", "content": "hi"}], [], -1, on_text=lambda *_: None, gone=lambda: True)
        self.assertTrue(backend_cancel.canceled)

        backend_stream_cancel = _Backend(["tok1 ", "tok2 "])
        # Simulate a backend that ignores should_cancel and emits pieces anyway.
        def raw_generate(max_tokens, out=None, should_cancel=None):
            if out is not None:
                out("tok1 ")
            return "tok1 ", SimpleNamespace(finish="stop", tokens=1, prompt_tokens=4, prefill_seconds=0.0, rate=1.0)
        backend_stream_cancel.generate = raw_generate
        svc_stream_cancel = ModelService(_Session(backend_stream_cancel))
        with mock.patch("veystrix.server.GONE_CHECK_SECONDS", 0):
            with self.assertRaises(ClientGone):
                svc_stream_cancel.complete([{"role": "user", "content": "hi"}], [], -1, on_text=lambda *_: None, gone=lambda: True)

    def test_interrupted_tool_call_raises_model_tool_call_error(self):
        svc = ModelService(_Session(_Backend(["<tool_call><function=read>"], finish="length")), context_limit=8192)
        with self.assertRaises(ModelToolCallError):
            svc.complete([{"role": "user", "content": "read x"}], [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}], -1)

    def test_backend_context_ceiling_helpers(self):
        from veystrix.backends.flashnext import FlashNextBackend

        b = FlashNextBackend.__new__(FlashNextBackend)
        b.context_limit = 16384
        b.language = None
        b.tokenizer = None
        self.assertEqual(b._context_ceiling(), 16384)

        b2 = FlashNextBackend.__new__(FlashNextBackend)
        b2.context_limit = None
        b2.language = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144))
        b2.tokenizer = None
        self.assertEqual(b2._context_ceiling(), 262144)


class ReasoningAndMemoryGuardTests(unittest.TestCase):
    def test_reasoning_levels_and_uncapped_xhigh(self):
        r_none = reasoning.resolve({"reasoning_effort": "none"}, True, "medium")
        self.assertFalse(r_none.thinking)
        for level in ("low", "medium", "high", "xhigh"):
            r = reasoning.resolve({"reasoning_effort": level}, False, "medium")
            self.assertTrue(r.thinking)
            self.assertEqual(r.effort, level)
        # Checkpoint has no separate 'max' tier: clamps honestly to xhigh.
        r_max = reasoning.resolve({"reasoning_effort": "max"}, False, "medium")
        self.assertTrue(r_max.thinking)
        self.assertEqual(r_max.effort, "xhigh")

        # xhigh never silently caps think budget to 512/2048.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp)
            (ckpt / "config.json").write_text(json.dumps({"reap_prune": {"ratio": 0.5}}))
            self.assertEqual(reasoning_policy.effective_think_budget(-1, "xhigh", ckpt), -1)

    def test_reasoning_streaming_and_ledger_preservation(self):
        svc = ModelService(_Session(_Backend(["<think>", "step ", "1", "</think>", "final answer"]), thinking_enabled=False))
        svc.session.preferences["thinking_enabled"] = True
        svc.session.preferences["effort"] = "xhigh"
        h = _handler_for(svc)
        h._chat({
            "model": "veystrix-flashnext",
            "messages": [{"role": "user", "content": "q1"}],
            "reasoning_effort": "xhigh",
            "stream": True,
        })
        events = [val for _, val in h.sse_events]
        reasoning_deltas = [c["choices"][0]["delta"].get("reasoning_content") for c in events if c.get("choices")]
        content_deltas = [c["choices"][0]["delta"].get("content") for c in events if c.get("choices")]
        self.assertTrue(any(reasoning_deltas))
        self.assertIn("final answer", "".join(x for x in content_deltas if x))

    def test_memory_guard_stops_request_with_503(self):
        from models.flashnext import memory_controller

        svc = ModelService(_Session(_Backend(["ok"])))
        fake_mc = SimpleNamespace(
            can_admit_work=lambda: False,
            current=lambda: SimpleNamespace(stop_reason="host memory critical"),
        )
        with mock.patch.dict(sys.modules, {"models.flashnext.memory_controller": fake_mc}):
            with mock.patch.object(memory_controller, "can_admit_work", return_value=False):
                with mock.patch.object(memory_controller, "current", return_value=SimpleNamespace(stop_reason="host memory critical")):
                    with self.assertRaises(RequestError) as ctx:
                        svc.complete([{"role": "user", "content": "hi"}], [], -1)
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(ctx.exception.code, "memory_critical_stop")


class HarnessConfigGeneratorTests(unittest.TestCase):
    def test_hermes_deepseek_and_qwen_configs_have_no_small_output_caps(self):
        from integrations.hermes import configure_profile as hermes_cfg

        m = InstalledModel("veystrix-flashnext", "flashnext", "flashnext", "/tmp/m", "qwen4_exp", {}, 262144, "test")

        # Qwen Code
        qc = qwen_config({}, [m], port=8080)["modelProviders"]["openai"][0]
        self.assertEqual(qc["generationConfig"]["contextWindowSize"], 262144)
        self.assertNotIn("maxTokens", qc["generationConfig"])
        self.assertNotIn("max_tokens", qc["generationConfig"])

        # DeepSeek Harness generator
        dsh = deepseek_config([{"id": "llm-pi-ai"}], [m], port=8080)[0]["config"]["providers"]["veystrix"]["models"][0]
        self.assertEqual(dsh["contextWindow"], 266240)
        self.assertEqual(dsh["maxTokens"], 262144)
        self.assertEqual(dsh["reasoningEfforts"]["xhigh"], "xhigh")

        # Hermes standalone profile config
        desired = hermes_cfg.desired_config(8080, "veystrix-flashnext", 262144)
        self.assertEqual(desired["model"]["context_length"], 262144)
        self.assertEqual(desired["fallback_providers"], [])
        self.assertNotIn("max_tokens", desired["model"])

        # harness_configure.run writes Qwen, DeepSeek, and Hermes together without output caps
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".qwen").mkdir()
            (root / ".qwen" / "settings.json").write_text("{}")
            dsh_patch = root / ".dsh" / "profiles" / "desktop" / "cordis.patch.yml"
            dsh_patch.parent.mkdir(parents=True)
            dsh_patch.write_text(yaml.safe_dump([{"id": "llm-pi-ai"}]))
            (root / ".hermes").mkdir()

            with mock.patch("veystrix.harness_configure.discover", return_value=[m]):
                reports = configure_harnesses(home=root, port=8080, apply=True)
            self.assertIn("created", reports["Hermes"])
            self.assertIn("updated", reports["Qwen Code"])
            self.assertIn("updated", reports["DeepSeek desktop"])

            qwen_written = json.loads((root / ".qwen" / "settings.json").read_text())
            self.assertNotIn("maxTokens", qwen_written["modelProviders"]["openai"][0]["generationConfig"])

            dsh_written = load_yaml12(dsh_patch.read_text())[0]["config"]["providers"]["veystrix"]["models"][0]
            self.assertEqual(dsh_written["maxTokens"], 262144)
            self.assertEqual(dsh_written["contextWindow"], 266240)

            hermes_written = yaml.safe_load((root / ".hermes" / "profiles" / "veystrix" / "config.yaml").read_text())
            self.assertEqual(
                hermes_written["custom_providers"][0]["models"]["veystrix-flashnext"]["context_length"],
                262144,
            )
            self.assertNotIn("max_tokens", hermes_written["custom_providers"][0]["models"]["veystrix-flashnext"])

    def test_shipped_dsh_patches_preserve_margin_and_uncapped_max_tokens(self):
        dsh_dir = Path(__file__).resolve().parents[2] / "docs" / "flashnext" / "dsh"
        for name in (
            "veystrix-262k.patch.yml",
            "veystrix-terminal.overlay.yml",
            "veystrix-agentic-8k.patch.yml",
            "veystrix-agentic.patch.yml",
        ):
            text = (dsh_dir / name).read_text()
            win = int(re.search(r"^\s*contextWindow:\s*(\d+)", text, re.M)[1])
            mtok = int(re.search(r"^\s*maxTokens:\s*(\d+)", text, re.M)[1])
            self.assertEqual(win - mtok, 4096, f"{name} must keep 4096 margin above maxTokens")
            self.assertGreaterEqual(mtok, 8192, f"{name} must not have an artificial small maxTokens cap")
