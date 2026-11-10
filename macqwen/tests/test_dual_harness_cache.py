"""Dual-harness cache, x-cache-id, diagnostics, and reasoning preservation tests.

Covers:
- x-cache-id header support and namespace isolation
- Tool schema hashing in derive_key
- Static system prompt invariance across reasoning effort levels
- DeepSeek Harness tool-call reasoning preservation and regular-turn omission
- ModelService.cache_diagnostics() reporting
- Post-turn snapshotting into PrefixSession
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from macqwen import conversation, preferences
from macqwen.server import ModelService
from macqwen.sessions import ReasoningLedger, derive_key, explicit_id
from macqwen.tests.test_agentic_long_fixtures import TOKENIZER, RealTokenizerBackend
from macqwen.tests.test_server import FakeSession
from models.flashnext import prefix_cache as P


class DualHarnessCacheTests(unittest.TestCase):
    def test_x_cache_id_header_resolution_and_namespace_isolation(self):
        """Header variations of x-cache-id are recognized and isolated by namespace."""
        messages = [{"role": "user", "content": "Hello world"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        # Various header casings
        headers_lower = {"x-cache-id": "shared-cache-123"}
        headers_upper = {"X-Cache-Id": "shared-cache-123"}
        headers_session = {"X-Session-Id": "shared-cache-123"}

        id_lower = explicit_id(headers_lower)
        id_upper = explicit_id(headers_upper)
        id_session = explicit_id(headers_session)

        self.assertEqual(id_lower, "shared-cache-123")
        self.assertEqual(id_upper, "shared-cache-123")
        self.assertEqual(id_session, "shared-cache-123")

        key_lower = derive_key(messages, explicit=id_lower, tools=tools)
        key_upper = derive_key(messages, explicit=id_upper, tools=tools)
        key_session = derive_key(messages, explicit=id_session, tools=tools)

        # Same namespace + explicit id -> same key
        self.assertEqual(key_lower, key_upper)
        self.assertEqual(key_lower, key_session)
        self.assertTrue(key_lower.startswith("id:shared-cache-123"))

        # Namespace isolation: qwen-code vs deepseek-harness with same x-cache-id
        key_qwen = derive_key(messages, explicit=id_lower, tools=tools, namespace="qwen-code")
        key_dsh = derive_key(messages, explicit=id_lower, tools=tools, namespace="deepseek-harness")

        self.assertNotEqual(key_qwen, key_dsh)
        self.assertTrue(key_qwen.startswith("id:qwen-code:shared-cache-123"))
        self.assertTrue(key_dsh.startswith("id:deepseek-harness:shared-cache-123"))

        # Tool schema isolation: different tools produce different auto keys
        tools_b = [{"type": "function", "function": {"name": "other_tool"}}]
        key_tools_a = derive_key(messages, explicit=None, tools=tools)
        key_tools_b = derive_key(messages, explicit=None, tools=tools_b)
        self.assertNotEqual(key_tools_a, key_tools_b)

    def test_reasoning_system_text_handles_extra_reasoning_effort(self):
        """The reasoning system text accurately passes through base system or prepends extra."""
        base_system = "You are a helpful coding assistant with access to tools."
        text_none, effort_none = conversation.reasoning_system_text(base_system, None)
        text_high, effort_high = conversation.reasoning_system_text(base_system, "high")

        # Base text must match untouched when no extra effort
        self.assertEqual(text_none, base_system)
        self.assertEqual(effort_none, None)
        # For high effort, extra instruction is injected and base system is preserved
        self.assertTrue(text_high.endswith(base_system))
        self.assertIn("Think carefully", text_high)
        self.assertEqual(effort_high, "medium")

    def test_deepseek_harness_tool_call_reasoning_preservation_and_omission(self):
        """DSH sends reasoning on tool-call turns but omits on regular turns; ledger handles both."""
        ledger = ReasoningLedger()
        session_id = "test-dsh-session"

        # Turn 1: Model generated reasoning + tool call
        turn1_reasoning = "I need to inspect the file structure."
        turn1_tool_calls = [{"id": "call_1", "name": "ls", "arguments": "{}"}]
        ledger.record_turn(session_id, "", turn1_tool_calls, turn1_reasoning)

        # Turn 2: Model generated reasoning + final answer
        turn2_reasoning = "File structure looks good. Answering the user."
        ledger.record_turn(session_id, "Here are your files.", [], turn2_reasoning)

        # Scenario: DSH sends history where Turn 1 includes reasoning (tool turn)
        # but Turn 2 omits reasoning (regular turn)
        incoming_messages = [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": turn1_reasoning,  # DSH preserved this
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "call_1"},
            {
                "role": "assistant",
                "content": "Here are your files.",  # DSH omitted reasoning here!
            },
            {"role": "user", "content": "Now summarize file1.txt"},
        ]

        restored_count = ledger.restore_reasoning(session_id, incoming_messages)
        # Turn 1 had reasoning already, Turn 2 was restored from ledger
        self.assertEqual(restored_count, 1)
        self.assertEqual(incoming_messages[1]["reasoning_content"], turn1_reasoning)
        self.assertEqual(incoming_messages[3]["reasoning_content"], turn2_reasoning)

    def test_cache_diagnostics_endpoint_reporting(self):
        """ModelService.cache_diagnostics reports all tier statistics accurately."""
        session = FakeSession([])
        tmp_dir = Path(tempfile.mkdtemp(prefix="diag_test_"))
        try:
            cache = P.PrefixCache("fingerprint_diag", ram_budget=10 * 1024 * 1024, directory=tmp_dir)
            backend = SimpleNamespace(
                _prefix_session=lambda: SimpleNamespace(store=lambda: cache),
                last_prefix=None,
                cache=[],
                tape=(1, 2, 3),
                last_decode_rate=45.0,
            )
            session.backend = backend
            service = ModelService(session)

            diag = service.cache_diagnostics()
            self.assertTrue(diag["cache_enabled"])
            self.assertEqual(diag["ram_cache_bytes"], 0)
            self.assertEqual(diag["ssd_cache_bytes"], 0)
            self.assertEqual(diag["ram_cache_budget_bytes"], 10 * 1024 * 1024)
            self.assertIn("ssd_cache_directory", diag)
            self.assertIn("evictions_ram", diag)
            self.assertIn("evictions_ssd", diag)
            self.assertEqual(diag["reused"], 0)
            self.assertEqual(diag["rebuilt"], 0)
            cache._disk["stored-only"] = {"n": 1000, "bytes": 12, "parent": None, "atime": 0}
            diag = service.cache_diagnostics()
            self.assertEqual(diag["ssd_cache_entries"], 1)
            self.assertIsNone(diag["static_prefix_hits"])
            self.assertEqual(diag["ssd_hits"], 0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
    def test_post_turn_snapshot_is_opt_in_and_never_written_by_default(self):
        """Incident 2026-11-09: a post-turn snapshot (unrestorable, no chunk-plan provenance) was written per turn even with
        the prefix cache off. Default: no write at all."""
        session = FakeSession([])
        mock_store = MagicMock()
        backend = RealTokenizerBackend(TOKENIZER, ["Hello there!"])
        backend._prefix_session = lambda: SimpleNamespace(store=lambda: mock_store)
        backend.cache = [SimpleNamespace(nbytes=100)]
        session.backend = backend
        with patch.dict(os.environ, {"FLASHNEXT_PREFIX_CACHE": "1"}, clear=False):
            os.environ.pop("FLASHNEXT_TURN_SNAPSHOTS", None)
            ModelService(session).complete([{"role": "user", "content": "Hello"}], [], 100)
        self.assertFalse(mock_store.put.called)
        mock_store.reset_mock()
        session.backend = backend = RealTokenizerBackend(TOKENIZER, ["Hello there!"])
        backend._prefix_session = lambda: SimpleNamespace(store=lambda: mock_store)
        backend.cache = [SimpleNamespace(nbytes=100)]
        with patch.dict(os.environ, {"FLASHNEXT_TURN_SNAPSHOTS": "1"}, clear=False):          # flag on, cache off: still none
            os.environ.pop("FLASHNEXT_PREFIX_CACHE", None)
            ModelService(session).complete([{"role": "user", "content": "Hello"}], [], 100)
        self.assertFalse(mock_store.put.called)

    @unittest.skipIf(TOKENIZER is None, "real checkpoint tokenizer not present")
    def test_post_turn_snapshot_saved_when_explicitly_enabled(self):
        """FLASHNEXT_TURN_SNAPSHOTS=1 with the prefix cache on keeps the old behaviour."""
        session = FakeSession([])
        mock_store = MagicMock()
        mock_ps = SimpleNamespace(store=lambda: mock_store)
        backend = RealTokenizerBackend(TOKENIZER, ["Hello there!"])
        backend._prefix_session = lambda: mock_ps
        backend.cache = [SimpleNamespace(nbytes=100)]
        session.backend = backend
        service = ModelService(session)

        # Trigger complete
        with patch.dict(os.environ, {"FLASHNEXT_TURN_SNAPSHOTS": "1", "FLASHNEXT_PREFIX_CACHE": "1"}):
            res = service.complete([{"role": "user", "content": "Hello"}], [], 100)
        self.assertEqual(res.text, "Hello there!")
        self.assertTrue(mock_store.put.called)
        call_kwargs = mock_store.put.call_args[1]
        self.assertEqual(call_kwargs["kind"], "turn_complete")
        self.assertGreater(len(call_kwargs["tokens"]), 0)


if __name__ == "__main__":
    unittest.main()
