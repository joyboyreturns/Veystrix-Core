"""A malformed model tool call: strict by default, pass-through for unattended runs (VEYSTRIX_INVALID_TOOL_CALLS=pass)."""
import json
import os
import unittest
from unittest.mock import patch

from macqwen.server import INVALID_TOOL_CALLS_ENV, ModelToolCallError, _parse_tool_calls

BASH = {"type": "function", "function": {"name": "bash", "description": "run", "parameters": {
    "type": "object", "required": ["command", "description"],
    "properties": {"command": {"type": "string"}, "description": {"type": "string"}, "timeout": {"type": "integer"}}}}}

MISSING = "<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>"
BAD_TYPE = ("<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=description>\nlist\n</parameter>\n"
            "<parameter=timeout>\nsoon\n</parameter>\n</function>\n</tool_call>")
JSON_UNKNOWN = '<tool_call>{"name": "bash", "arguments": {"command": "ls", "description": "d", "bogus": 1}}</tool_call>'
UNDECLARED = "<tool_call><function=rm><parameter=path>/</parameter></function></tool_call>"


class StrictByDefault(unittest.TestCase):
    def test_every_malformed_call_still_fails_the_request(self):
        env = {k: v for k, v in os.environ.items() if k != INVALID_TOOL_CALLS_ENV}
        with patch.dict(os.environ, env, clear=True):
            for text in (MISSING, BAD_TYPE, JSON_UNKNOWN, UNDECLARED):
                with self.assertRaises(ModelToolCallError, msg=text[:40]):
                    _parse_tool_calls(text, [BASH])

    def test_unrecognised_mode_values_stay_strict(self):
        with patch.dict(os.environ, {INVALID_TOOL_CALLS_ENV: "yes"}):
            with self.assertRaises(ModelToolCallError):
                _parse_tool_calls(MISSING, [BASH])


class PassThrough(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {INVALID_TOOL_CALLS_ENV: "pass"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_missing_required_parameter_reaches_the_client_untouched(self):
        text, calls = _parse_tool_calls(MISSING, [BASH])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "bash")
        self.assertEqual(json.loads(calls[0]["arguments"]), {"command": "echo hi"})   # nothing invented
        self.assertEqual(text, "")

    def test_a_badly_typed_value_is_kept_as_the_models_text_and_valid_ones_are_still_typed(self):
        _, calls = _parse_tool_calls(BAD_TYPE, [BASH])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"command": "ls", "description": "list", "timeout": "soon"})

    def test_unknown_parameters_and_undeclared_tools_pass_through_too(self):
        _, calls = _parse_tool_calls(JSON_UNKNOWN, [BASH])
        self.assertEqual(json.loads(calls[0]["arguments"])["bogus"], 1)
        _, calls = _parse_tool_calls(UNDECLARED, [BASH])
        self.assertEqual(calls[0]["name"], "rm")

    def test_a_valid_call_is_unchanged(self):
        ok = MISSING.replace("</function>", "<parameter=description>\nsay hi\n</parameter>\n</function>")
        _, calls = _parse_tool_calls(ok, [BASH])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"command": "echo hi", "description": "say hi"})

    def test_the_pass_is_recorded_in_the_flight_log(self):
        with patch("macqwen.server.flight_recorder.record") as record:
            _parse_tool_calls(MISSING, [BASH])
        record.assert_called_once()
        self.assertEqual(record.call_args.args[0], "invalid_tool_call_passed")


if __name__ == "__main__":
    unittest.main()
