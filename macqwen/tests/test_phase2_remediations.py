"""Phase 2 regression tests covering UltraReview findings:
- R2-02: OpenAI tool call index and message name fields
- R2-03: JSON Schema union types, nullability, unhashable type bug fix
- R2-05: XML transport unescaping for array/object arguments (&lt;, &gt;, &amp;)
- R2-08: No unsafe TypeError retries during generation
- R2-09: Accurate cache eligibility & usage accounting
- R2-10: Cold prefill limit enforcement on failed lookup
- R3-05: ReasoningLedger turn alignment across empty reasoning turns
"""
import json
import os
import unittest
from unittest import mock
from types import SimpleNamespace

from macqwen.server import (
    ModelService,
    ModelToolCallError,
    RequestError,
    _coerce_argument,
    _normalize_messages,
    _parse_tool_calls,
    _template_tool_call,
    _validate_call,
    usage_counts,
)
from macqwen.sessions import ReasoningLedger
from models.flashnext.prefix_session import PrefixSession, RecoveryRequiredError


class TestR202OpenAICompatibility(unittest.TestCase):
    def test_tool_call_with_index_is_accepted(self):
        call = {
            "index": 0,
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
        res = _template_tool_call(call, "where")
        self.assertEqual(res["type"], "function")
        self.assertEqual(res["function"]["name"], "test_tool")
        self.assertEqual(res["id"], "call_123")

    def test_messages_with_name_on_user_and_assistant_are_accepted(self):
        messages = [
            {"role": "system", "name": "sys_bot", "content": "hello"},
            {"role": "user", "name": "alice", "content": "hi"},
            {"role": "assistant", "name": "assistant_bot", "content": "hey there"},
        ]
        norm = _normalize_messages(messages)
        self.assertEqual(len(norm), 3)
        self.assertEqual(norm[0]["role"], "system")
        self.assertEqual(norm[1]["role"], "user")
        self.assertEqual(norm[2]["role"], "assistant")


class TestR203JsonSchemaUnionTypes(unittest.TestCase):
    def test_union_string_or_null_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "label": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]},
            },
        }
        # String value should pass
        _validate_call("test", {"label": "active", "count": 5}, schema)
        # Null value should pass without TypeError unhashable list
        _validate_call("test", {"label": None, "count": None}, schema)
        # Wrong type should raise ModelToolCallError
        with self.assertRaises(ModelToolCallError):
            _validate_call("test", {"label": 123}, schema)

    def test_nullable_property(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "nullable": True},
            },
        }
        _validate_call("test", {"name": None}, schema)
        _validate_call("test", {"name": "alice"}, schema)
        with self.assertRaises(ModelToolCallError):
            _validate_call("test", {"name": 456}, schema)

    def test_coerce_argument_union_type(self):
        # Union type parameter coercion
        val = _coerce_argument("hello", ["string", "null"], tool="t", key="p")
        self.assertEqual(val, "hello")
        null_val = _coerce_argument("null", ["string", "null"], tool="t", key="p")
        self.assertIsNone(null_val)


class TestR205XmlUnescaping(unittest.TestCase):
    def test_array_and_object_arguments_unescape_transport_entities(self):
        # JSON string with XML-escaped operators: a &lt; b -> a < b
        raw_xml = '<tool_call><function=edit_file><parameter=edits>[{"old": "a &lt; b", "new": "a &gt; b &amp; c"}]</parameter></function></tool_call>'
        tools = [{
            "type": "function",
            "function": {
                "name": "edit_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "edits": {"type": "array"},
                    },
                },
            },
        }]
        clean, calls = _parse_tool_calls(raw_xml, tools)
        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0]["arguments"])
        self.assertEqual(args["edits"], [{"old": "a < b", "new": "a > b & c"}])


class TestR208NoTypeErrorRetries(unittest.TestCase):
    def test_internal_type_error_does_not_retry_generation(self):
        backend = mock.Mock()
        # Raise TypeError from inside backend.generate
        backend.generate.side_effect = TypeError("internal issue inside mlx kernel")
        backend.tape = []
        backend.pending = []
        backend.encode = lambda s: [1, 2, 3]
        backend.common_prefix = lambda ids: 0
        backend.language = None
        backend.tokenizer = mock.Mock()
        backend.tokenizer.apply_chat_template.return_value = "rendered prompt"

        session = SimpleNamespace(backend=backend, reset=mock.Mock(), preferences={"workspace": ".", "model": "flashnext"})
        service = ModelService(session)
        service._adopt = mock.Mock(return_value=True)
        service.request_reasoning = lambda: SimpleNamespace(thinking=False, effort=None)

        messages = [{"role": "user", "content": "hello"}]
        with self.assertRaises(TypeError) as ctx:
            service.complete(messages, tools=[], max_tokens=10)
        self.assertIn("internal issue inside mlx kernel", str(ctx.exception))
        # backend.generate must be called exactly ONCE: no retry fallback!
        self.assertEqual(backend.generate.call_count, 1)


class TestR209CacheEligibilityAndUsage(unittest.TestCase):
    def test_adopt_ignores_prefix_cache_when_disabled(self):
        from macqwen.backends.flashnext import FlashNextBackend

        backend = mock.MagicMock(spec=FlashNextBackend)
        backend.decoder = None
        backend.encode = lambda s: list(range(len(s.split())))
        backend.common_prefix = lambda ids: 0
        backend.pending = []
        backend.tape = []
        backend._prefix_session = mock.Mock()
        backend._prefix_session.return_value.reusable_tokens.return_value = 50

        session = SimpleNamespace(backend=backend, reset=mock.Mock())
        service = ModelService.__new__(ModelService)
        service.session = session
        service.rebuilt = 0
        service.evictions = 0
        service.warm_tokens = lambda: 0
        service.context_limit = 262144
        service.max_cold_prefill = 32768
        service.max_incremental_prefill = 32768
        service.refused_cold_prefill = 0

        # With FLASHNEXT_PREFIX_CACHE disabled ("0"), reusable_tokens should not be called
        with mock.patch.dict(os.environ, {"FLASHNEXT_PREFIX_CACHE": "0"}):
            service._adopt("prompt word word word")
        self.assertEqual(service.last_cached_tokens, 0)

    def test_usage_counts_reflects_actual_restored_tokens(self):
        service = SimpleNamespace(last_prompt_tokens=100, last_cached_tokens=40)
        stats = SimpleNamespace(prompt_tokens=60, tokens=10)
        u = usage_counts(service, stats)
        self.assertEqual(u["prompt"], 100)
        self.assertEqual(u["cached"], 40)
        self.assertEqual(u["completion"], 10)


class TestR210EnforceColdPrefillLimits(unittest.TestCase):
    def test_prefill_raises_recovery_required_when_actual_restore_fails_on_large_prompt(self):
        prompt = list(range(40000))  # 40K tokens > 32K limit
        mock_store = mock.Mock()
        # Lookup fails to restore: returns hit=None
        mock_store.lookup.return_value = (None, None)
        mock_store.stats = SimpleNamespace(rejected_plan=0)

        ps = PrefixSession(lambda text: [1], lambda: {})
        ps.store = lambda: mock_store

        language = mock.Mock()
        language.make_cache = lambda: []

        with mock.patch.dict(os.environ, {
            "FLASHNEXT_PREFIX_CACHE": "1",
            "FLASHNEXT_PREFIX_CACHE_RESTORE": "1",
            "VEYSTRIX_MAX_COLD_PREFILL": "32768",
        }):
            with self.assertRaises(RecoveryRequiredError) as ctx:
                ps.prefill(language, prompt, [])
            self.assertIn("prohibited on 16 GB hardware", str(ctx.exception))


class TestR305ReasoningLedgerAlignment(unittest.TestCase):
    def test_empty_reasoning_turn_recorded_and_preserves_turn_alignment(self):
        ledger = ReasoningLedger()
        key = "session_1"

        # Turn 0: thinking enabled with reasoning
        ledger.record_turn(key, "answer 0", [], "think step 0")
        # Turn 1: thinking disabled or empty reasoning
        ledger.record_turn(key, "answer 1", [], "")
        # Turn 2: thinking enabled with reasoning
        ledger.record_turn(key, "answer 2", [], "think step 2")

        self.assertEqual(len(ledger.sessions[key]), 3)

        # Incoming request has all 3 assistant turns with reasoning stripped on turns 0 and 2
        messages = [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "answer 0"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "answer 2"},
        ]
        restored = ledger.restore_reasoning(key, messages)
        self.assertEqual(restored, 2)
        self.assertEqual(messages[1].get("reasoning_content"), "think step 0")
        self.assertNotIn("reasoning_content", messages[3])
        self.assertEqual(messages[5].get("reasoning_content"), "think step 2")


if __name__ == "__main__":
    unittest.main()
