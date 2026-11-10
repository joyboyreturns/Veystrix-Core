"""Local OpenAI and Anthropic compatible HTTP server for MACQWEN."""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
import re
import select
import socket
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

from macqwen.text import (
    CompletedTextBuffer,
    ReasoningStreamFilter,
    ThinkingStreamFilter,
    build_user_encoder,
    extract_reasoning_and_content,
)
from macqwen.tools import ToolCallValidationError, coerce_scalar, _unescape_argument
from macqwen.conversation import (
    content_sentinel,
    reasoning_system_text,
    split_content_slots,
)
from macqwen import context_budget
from macqwen.profiles import system_prompt
from macqwen.scheduler import Busy, ClientGone, Scheduler
from macqwen.sessions import (
    ReasoningLedger,
    SessionRegistry,
    derive_key,
    explicit_id,
)
from macqwen.backends.base import GenerationCancelled
from macqwen import effort_slot
from macqwen import warm as warm_module
from models.flashnext import flight_recorder


MAX_BODY = 16 * 1024 * 1024
# Tokens always kept free for the answer. A prompt that leaves less is refused rather than answered with a stub.
MIN_OUTPUT_RESERVE = context_budget.MIN_OUTPUT_RESERVE
GONE_CHECK_SECONDS = 1.0
SSE_KEEPALIVE_SECONDS = float(os.environ.get("VEYSTRIX_SSE_KEEPALIVE_SECONDS", "15"))
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
XML_FUNCTION = re.compile(
    r"\s*<function=([^>\s]+)>(.*?)(?:</function>|</>)?\s*$", re.DOTALL
)
XML_PARAMETER = re.compile(
    r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL
)


class RequestError(ValueError):
    kind: str | None = None             # explicit OpenAI error type; None derives it from the status

    def __init__(self, message: str, status: int = 400, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class ContextLimitError(RequestError):
    """The prompt does not fit the configured context limit. Nothing is truncated and live state is kept."""

    def __init__(self, message: str, status: int = 400, code: str | None = "context_length_exceeded"):
        super().__init__(message, status, code)


class OversizedPrefillError(ContextLimitError):
    """A fresh cold prefill exceeds the maximum cold prefill limit (default 16,384 tokens)."""

    def __init__(self, message: str, status: int = 400, code: str | None = "oversized_prefill_rejected"):
        super().__init__(message, status, code)


from models.flashnext.prefix_session import RecoveryRequiredError as _BaseRecoveryRequiredError


class RecoveryRequiredError(_BaseRecoveryRequiredError, ContextLimitError):
    """A long conversation lost its state cache and cannot be safely recomputed cold."""

    def __init__(self, message: str, status: int = 400, code: str | None = "recovery_required"):
        super().__init__(message, status=status, code=code)


class ModelToolCallError(RequestError):
    """The MODEL produced a tool call that is malformed or breaks the client's declared schema.

    Not the client's mistake and not an API serialization failure, so it is a ``server_error``, not an
    ``invalid_request_error``. It stays a non-retryable 4xx on purpose: a client that auto-retries 5xx would repeat a
    full (slow) generation that is likely to fail the same way. A harness can feed the message back to the model.
    """

    kind = "server_error"

    def __init__(self, message: str):
        super().__init__(message, 422, "model_tool_call_invalid")


@dataclass
class Completion:
    text: str
    tool_calls: list[dict]
    stats: object
    reasoning_content: str = ""


def usage_counts(service, stats) -> dict:
    """Token accounting every API reports. ``stats.prompt_tokens`` is only what this request prefilled; the prompt the client sent
    is ``service.last_prompt_tokens`` long, and the rest came from cached state (live conversation or SSD snapshot)."""
    prefilled = int(getattr(stats, "prompt_tokens", 0) or 0)
    total = getattr(service, "last_prompt_tokens", None)
    total = max(int(total), prefilled) if isinstance(total, int) and not isinstance(total, bool) else prefilled
    # A request restored from an SSD snapshot reports the whole prompt as prefilled, so the live-state arithmetic alone says
    # "0 cached". ``last_cached_tokens`` is the prefix the server matched (live state or snapshot) when it adopted the request.
    matched = getattr(service, "last_cached_tokens", 0)
    matched = int(matched) if isinstance(matched, int) and not isinstance(matched, bool) else 0
    cached = min(total, max(total - prefilled, matched))
    return {"prompt": total, "cached": cached, "completion": int(getattr(stats, "tokens", 0) or 0)}


def chat_usage(service, stats) -> dict:
    """OpenAI chat usage plus the cache fields clients read: OpenAI ``prompt_tokens_details.cached_tokens`` and DeepSeek's
    ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``. ``prompt_tokens`` counts cached tokens too, as both APIs do."""
    u = usage_counts(service, stats)
    return {"prompt_tokens": u["prompt"], "completion_tokens": u["completion"], "total_tokens": u["prompt"] + u["completion"],
            "prompt_tokens_details": {"cached_tokens": u["cached"]},
            "prompt_cache_hit_tokens": u["cached"], "prompt_cache_miss_tokens": u["prompt"] - u["cached"]}


def responses_usage(service, stats) -> dict:
    u = usage_counts(service, stats)
    return {"input_tokens": u["prompt"], "input_tokens_details": {"cached_tokens": u["cached"]},
            "output_tokens": u["completion"], "total_tokens": u["prompt"] + u["completion"]}


def anthropic_usage(service, stats) -> dict:
    """Anthropic reports uncached input separately from ``cache_read_input_tokens``."""
    u = usage_counts(service, stats)
    return {"input_tokens": u["prompt"] - u["cached"], "cache_read_input_tokens": u["cached"],
            "cache_creation_input_tokens": 0, "output_tokens": u["completion"]}


TEXT_PARTS = ("text", "input_text", "output_text")


def _text_content(content, where: str = "content") -> str:
    """Join text parts. A part that is not text is an error, never dropped."""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise RequestError(f"{where} must be a string or a list of text parts")
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in TEXT_PARTS:
            kind = block.get("type") if isinstance(block, dict) else type(block).__name__
            raise RequestError(
                f"{where} has a {kind!r} part; this server is text-only and "
                "will not discard it"
            )
        parts.append(str(block.get("text", "")))
    return "".join(parts)


def _openai_tools(tools) -> list[dict]:
    result = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            result.append(tool)
            continue
        name = tool.get("name")
        if name:
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            })
    return result


# Message fields the server understands. Anything else with a value is an
# error: a field that changes what the model sees must not be dropped quietly.
MESSAGE_FIELDS = {"role", "content", "tool_calls", "tool_call_id", "reasoning_content", "name"}
TOOL_ONLY_FIELDS = set()  # "name" is supported across all roles per OpenAI spec
IGNORABLE_EMPTY = {"refusal", "audio", "annotations", "function_call"}


def _template_tool_call(call, where: str) -> dict:
    """One OpenAI tool call in the shape the Qwen template renders.

    The wire format carries arguments as a JSON string; the template iterates
    a mapping. Arguments that are not a JSON object are an error.
    """
    if not isinstance(call, dict):
        raise RequestError(f"{where}: a tool call must be an object")
    function = call.get("function")
    if not isinstance(function, dict):
        raise RequestError(f"{where}: tool call has no function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise RequestError(f"{where}: tool call has no function name")
    arguments = function.get("arguments", "")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            raise RequestError(
                f"{where}: arguments of tool call {name!r} are not valid JSON"
            ) from None
    if not isinstance(arguments, dict):
        raise RequestError(
            f"{where}: arguments of tool call {name!r} must be a JSON object"
        )
    extra = sorted(set(call) - {"id", "type", "function", "index"})
    if extra:
        raise RequestError(f"{where}: unsupported tool call fields {extra}")
    item = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call.get("id"):
        item["id"] = call["id"]
    return item


def _normalize_messages(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    normalized = []
    for index, message in enumerate(messages):
        where = f"messages[{index}]"
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError(f"unsupported message role: {role}")
        for key, value in message.items():
            if key in MESSAGE_FIELDS or (key in TOOL_ONLY_FIELDS and role == "tool"):
                continue
            if key in IGNORABLE_EMPTY and value in (None, "", [], {}):
                continue
            raise RequestError(f"{where}: unsupported message field {key!r}")
        item = {"role": "system" if role == "developer" else role,
                "content": _text_content(message.get("content", ""), f"{where}.content")}
        if message.get("tool_calls"):
            if role != "assistant":
                raise RequestError(f"{where}: only assistant messages carry tool_calls")
            if not isinstance(message["tool_calls"], list):
                raise RequestError(f"{where}: tool_calls must be a list")
            item["tool_calls"] = [
                _template_tool_call(call, f"{where}.tool_calls[{number}]")
                for number, call in enumerate(message["tool_calls"])
            ]
        if message.get("tool_call_id"):
            item["tool_call_id"] = message["tool_call_id"]
        if isinstance(message.get("reasoning_content"), str):
            item["reasoning_content"] = message["reasoning_content"]
        normalized.append(item)
    return normalized


def _anthropic_messages(payload: dict) -> list[dict]:
    messages = []
    system = _text_content(payload.get("system", ""))
    if system:
        messages.append({"role": "system", "content": system})
    for source in payload.get("messages", []):
        role = source.get("role")
        content = source.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text = []
        calls = []
        for block in content if isinstance(content, list) else []:
            kind = block.get("type")
            if kind == "text":
                text.append(str(block.get("text", "")))
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
            elif kind == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _text_content(block.get("content", "")),
                })
        if text or calls:
            item = {"role": role, "content": "".join(text)}
            if calls:
                item["tool_calls"] = calls
            messages.append(item)
    return _normalize_messages(messages)


def _responses_messages(payload: dict) -> list[dict]:
    messages = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})
    source = payload.get("input", "")
    if isinstance(source, str):
        messages.append({"role": "user", "content": source})
    elif isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                continue
            kind = item.get("type", "message")
            if kind == "message":
                messages.append({
                    "role": item.get("role", "user"),
                    "content": _text_content(item.get("content", "")),
                })
            elif kind == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": str(item.get("output", "")),
                })
            elif kind == "function_call":
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
    return _normalize_messages(messages)


def _coerce_argument(value: str, kind: str | list | None, *, tool="", key="parameter"):
    # One shared rule: the XML parser here and parse_tool_calls in
    # macqwen.tools both run coerce_scalar, which reverses transport
    # escaping exactly once. String payloads keep their bytes; only
    # typed scalar conversion normalizes whitespace. Bad typed values
    # raise ToolCallValidationError for the caller to report.
    if isinstance(kind, list):
        if "null" in kind and value.strip() == "null":
            return None
        last_exc = None
        for cand in [k for k in kind if k != "null"]:
            try:
                return coerce_scalar(value, cand, tool=tool, key=key)
            except ToolCallValidationError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return coerce_scalar(value, None, tool=tool, key=key)
    if kind == "null" and value.strip() == "null":
        return None
    return coerce_scalar(value, kind, tool=tool, key=key)


INVALID_TOOL_CALLS_ENV = "VEYSTRIX_INVALID_TOOL_CALLS"


def _pass_invalid_tool_calls() -> bool:
    """``pass``: hand a call that breaks the declared schema to the client as an ordinary tool call (flagged in the flight
    log) so a harness can answer with its own tool error and the model retries. Default ``error`` keeps the strict
    contract: the request fails naming the offender. An unattended run uses ``pass``: one bad call must not end a goal."""
    return os.environ.get(INVALID_TOOL_CALLS_ENV, "error").strip().lower() == "pass"


def _validate_call(name: str, arguments, schema: dict | None) -> None:
    """Check one call against the schema the client declared.

    Nothing is repaired or dropped: an unknown tool, an unknown or missing
    parameter, a value outside an enum, or a wrongly typed value raises with
    the offender named so the harness can return it to the model.
    """
    if schema is None:
        raise ModelToolCallError(f"model called tool {name!r}, which this request did not declare")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            raise ModelToolCallError(f"tool {name!r} arguments are not valid JSON") from None
    if not isinstance(arguments, dict):
        raise ModelToolCallError(f"tool {name!r} arguments must be a JSON object")
    properties = schema.get("properties", {}) or {}
    if schema.get("additionalProperties") is not True:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ModelToolCallError(
                f"tool {name!r} has no parameter {unknown[0]!r}; "
                f"accepted: {sorted(properties)}"
            )
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        raise ModelToolCallError(f"tool {name!r} is missing required parameter {missing[0]!r}")

    def _check_type(val, k):
        if k == "string":
            return isinstance(val, str)
        if k == "boolean":
            return isinstance(val, bool)
        if k == "integer":
            return (not isinstance(val, bool)) and (isinstance(val, int) or (isinstance(val, float) and val.is_integer()))
        if k == "number":
            return (not isinstance(val, bool)) and isinstance(val, (int, float))
        if k == "array":
            return isinstance(val, list)
        if k == "object":
            return isinstance(val, dict)
        if k == "null":
            return val is None
        raise ModelToolCallError(f"tool {name!r} parameter {key!r} has unsupported schema type: {k!r}")

    for key, value in arguments.items():
        spec = properties.get(key, {})
        allowed = spec.get("enum")
        if allowed is not None and value not in allowed:
            raise ModelToolCallError(
                f"tool {name!r} parameter {key!r} must be one of {allowed}, got {value!r}"
            )
        kind = spec.get("type")
        nullable = spec.get("nullable", False)
        if value is None:
            if nullable or kind == "null" or (isinstance(kind, list) and "null" in kind):
                continue
            raise ModelToolCallError(f"tool {name!r} parameter {key!r} cannot be null")
        if kind is None:
            continue
        if isinstance(kind, list):
            valid = any(_check_type(value, k) for k in kind if k != "null")
            if not valid:
                raise ModelToolCallError(f"tool {name!r} parameter {key!r} expects {kind}, got {value!r}")
        elif not _check_type(value, kind):
            raise ModelToolCallError(f"tool {name!r} parameter {key!r} expects {kind}, got {value!r}")


def _parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    calls = []
    schemas = {}
    declared = {}
    for tool in _openai_tools(tools):
        function = tool["function"]
        parameters = function.get("parameters", {}) or {}
        properties = parameters.get("properties", {}) or {}
        declared[function["name"]] = parameters
        schemas[function["name"]] = {
            name: value.get("type") for name, value in properties.items()
        }
    blocks = [(match.group(1), match) for match in TOOL_CALL.finditer(text)]
    fallback = False
    if not blocks and "<tool_call>" in text:
        tail = text.rsplit("<tool_call>", 1)[1]
        if "</function>" in tail:
            blocks = [(tail, None)]
            fallback = True
    for block, _match in blocks:
        try:
            value = json.loads(block)
        except (TypeError, ValueError):
            function = XML_FUNCTION.match(block)
            if not function:
                continue
            name, body = function.groups()
            schema = schemas.get(name, {})
            try:
                arguments = {
                    key: _coerce_argument(raw, schema.get(key),
                                         tool=name, key=key)
                    for key, raw in XML_PARAMETER.findall(body)
                }
            except ToolCallValidationError as exc:
                if _pass_invalid_tool_calls():
                    # Keep the model's own text for every parameter; the client's schema check reports the bad one.
                    flight_recorder.record("invalid_tool_call_passed", tool=name, detail=str(exc)[:300])
                    arguments = {}
                    for key, raw in XML_PARAMETER.findall(body):
                        try:
                            arguments[key] = _coerce_argument(raw, schema.get(key), tool=name, key=key)
                        except ToolCallValidationError:
                            arguments[key] = _unescape_argument(raw).strip()
                    calls.append({"id": f"call_{uuid.uuid4().hex}", "name": name, "arguments": json.dumps(arguments)})
                    continue
                # The model sent a typed value its schema rejects. Fail
                # the request with the parameter named so the client
                # harness can feed the message back and the model retries.
                raise ModelToolCallError(str(exc)) from None
            calls.append({"id": f"call_{uuid.uuid4().hex}", "name": name,
                          "arguments": json.dumps(arguments)})
        else:
            name = value.get("name")
            if not name:
                continue
            arguments = value.get("arguments", value.get("parameters", {}))
            calls.append({
                "id": value.get("id", f"call_{uuid.uuid4().hex}"),
                "name": name,
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            })
    if tools:
        for call in calls:
            try:
                _validate_call(call["name"], call["arguments"], declared.get(call["name"]))
            except ModelToolCallError as exc:
                if not _pass_invalid_tool_calls():
                    raise
                flight_recorder.record("invalid_tool_call_passed", tool=call["name"], detail=str(exc)[:300])
    clean = text.rsplit("<tool_call>", 1)[0] if fallback else TOOL_CALL.sub("", text)
    return clean.strip(), calls


def _model_tokenizer(backend):
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        tokenizer = backend.engine.tokenizer
    return tokenizer


DEFAULT_CONTEXT_WINDOW = 32768
# Owner decision 2026-11-09: 32K ceilings (cold, and uncached increment per request). Internal chunks stay 2,048 tokens; the
# 32K figure is an admission ceiling, never one tensor. Only 4K, 8K and 16K have real-checkpoint evidence; 32K does not yet.
MAX_COLD_PREFILL_DEFAULT = 32768
MAX_INCREMENTAL_PREFILL_DEFAULT = 32768
ARCHITECTURAL_CONTEXT_MAXIMUM = 262144

_MODEL_DISPLAY_NAMES = {
    "flashnext": "MACQWEN Flash-Next",
    "bonsai2": "MACQWEN Bonsai-2",
    "k2-horizon": "MACQWEN K2-Horizon",
    "k2_horizon": "MACQWEN K2-Horizon",
    "qwen27b": "MACQWEN Qwen3.8-27B",
}

_BACKEND_DISPLAY_NAMES = {
    "FlashNextBackend": "MACQWEN Flash-Next",
    "BonsaiBackend": "MACQWEN Bonsai-2",
    "K2HorizonBackend": "MACQWEN K2-Horizon",
    "FrankensteinBackend": "MACQWEN Qwen3.8-27B",
}

_WINDOW_ATTRS = (
    "context_window",
    "max_context",
    "n_ctx",
    "max_position_embeddings",
    "max_sequence_length",
    "max_seq_len",
    "n_positions",
)

_WINDOW_HOLDERS = ("language", "model", "engine", "language_model", "_text_model")

_TOKENIZER_ATTRS = (
    "model_max_length",
    "model_max_len",
    "max_position_embeddings",
    "n_positions",
    "context_window",
)


def _service_backend(service):
    session = getattr(service, "session", None)
    if session is not None:
        backend = getattr(session, "backend", None)
        if backend is not None:
            return backend
    return getattr(service, "backend", None)


def _valid_window(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    if not isinstance(value, int) or value < 1:
        return None
    # HF tokenizers use int(1e30) when the checkpoint sets no limit.
    if value >= 10_000_000:
        return None
    return value


def _window_from_object(candidate) -> int | None:
    if candidate is None:
        return None
    for name in _WINDOW_ATTRS:
        try:
            found = _valid_window(getattr(candidate, name, None))
        except Exception:
            continue
        if found is not None:
            return found
    config = getattr(candidate, "config", None)
    if isinstance(config, dict):
        for name in _WINDOW_ATTRS:
            found = _valid_window(config.get(name))
            if found is not None:
                return found
    elif config is not None:
        for name in _WINDOW_ATTRS:
            try:
                found = _valid_window(getattr(config, name, None))
            except Exception:
                continue
            if found is not None:
                return found
    return None


def _tokenizer_windows(tokenizer):
    seen = set()
    current = tokenizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        init_kwargs = getattr(current, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            yield init_kwargs
        current = getattr(current, "_tokenizer", None)


def _context_window_for(service) -> int:
    configured = getattr(service, "context_limit", None)
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured
    backend = _service_backend(service)
    if backend is not None:
        found = _window_from_object(backend)
        if found is not None:
            return found
        for name in _WINDOW_HOLDERS:
            try:
                found = _window_from_object(getattr(backend, name, None))
            except Exception:
                continue
            if found is not None:
                return found
        try:
            tokenizer = _model_tokenizer(backend)
        except Exception:
            tokenizer = getattr(backend, "tokenizer", None)
        if tokenizer is not None:
            for candidate in _tokenizer_windows(tokenizer):
                if isinstance(candidate, dict):
                    for name in _TOKENIZER_ATTRS:
                        found = _valid_window(candidate.get(name))
                        if found is not None:
                            return found
                    continue
                found = _window_from_object(candidate)
                if found is not None:
                    return found
                for name in _TOKENIZER_ATTRS:
                    try:
                        found = _valid_window(getattr(candidate, name, None))
                    except Exception:
                        continue
                    if found is not None:
                        return found
    return DEFAULT_CONTEXT_WINDOW


def _display_name_for(service) -> str:
    slug = str(getattr(service, "model", "") or "")
    backend = _service_backend(service)
    if backend is not None:
        named = _BACKEND_DISPLAY_NAMES.get(type(backend).__name__)
        if named is not None:
            return named
    key = slug[len("macqwen-"):] if slug.startswith("macqwen-") else slug
    named = _MODEL_DISPLAY_NAMES.get(key.lower().replace("_", "-"))
    if named is not None:
        return named
    return slug or "MACQWEN Flash-Next"


def capabilities(service) -> dict:
    """Display name and context window read from backend/tokenizer config.

    Only values the backend or its tokenizer expose are returned;
    otherwise the display falls back to the model slug and the window to
    32768.
    """
    return {
        "display_name": _display_name_for(service),
        "context_window": _context_window_for(service),
    }


def model_card(service) -> dict:
    """Full /v1/models card for the active backend.

    Only display_name and context_window vary by backend; every other
    field keeps the established card shape.
    """
    slug = str(getattr(service, "model", "") or "")
    caps = capabilities(service)
    return {
        "slug": slug,
        "display_name": caps["display_name"],
        "description": "Local low-memory Qwen runtime",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Low reasoning effort"},
            {"effort": "medium", "description": "Medium reasoning effort"},
            {"effort": "high", "description": "High reasoning effort"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "minimal_client_version": [0, 0, 0],
        "supported_in_api": True,
        "priority": 1,
        "upgrade": None,
        "base_instructions": "",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 100000},
        "supports_parallel_tool_calls": False,
        "supports_image_detail_original": False,
        "context_window": caps["context_window"],
        "experimental_supported_tools": [],
    }


_STARTED = int(time.time())


def model_entry(card: dict, service) -> dict:
    """The OpenAI-shaped model object. ``context_window`` is what this server will accept, not the model's maximum."""
    limit = getattr(service, "context_limit", None)
    enforced = isinstance(limit, int) and not isinstance(limit, bool) and limit > 0
    mem = _memory_ledger()
    if mem is not None:
        policy_info = {
            "policy_identifier": mem.get("policy_identifier"),
            "swap_baseline_mb": mem.get("swap_baseline_mb"),
            "current_swap_growth_mb": mem.get("current_swap_growth_mb"),
            "peak_swap_growth_mb": mem.get("peak_swap_growth_mb"),
            "threshold_gib": mem.get("threshold_gib"),
            "threshold_mb": mem.get("threshold_mb"),
            "swap_ceiling_mib": mem.get("swap_ceiling_mib"),
            "swap_ceiling_margin_mib": mem.get("swap_ceiling_margin_mib"),
            "stop_reason": mem.get("stop_reason"),
            "can_admit": mem.get("can_admit", True),
            "probe_availability": mem.get("probe_availability", {}),
            "telemetry_only_checks": mem.get("telemetry_only_checks", []),
        }
    else:
        try:
            from models.flashnext import memory_policy
            policy_info = memory_policy.policy_summary()
        except ImportError:
            policy_info = None
    return {
        "id": card["slug"],
        "object": "model",
        "created": _STARTED,
        "owned_by": "macqwen",
        "context_window": card["context_window"],
        "context_limit_enforced": enforced,
        "max_output_tokens": max(1, card["context_window"] - MIN_OUTPUT_RESERVE) if enforced else None,
        "memory_policy": policy_info,
        # What to configure as the CLIENT's context window: this window plus the client's own hard-coded reserve
        # (DeepSeek Harness subtracts 4096 before computing max_completion_tokens). See macqwen/context_budget.py.
        "suggested_client_context_window": (context_budget.suggested_client_window(card["context_window"])
                                            if enforced else None),
        "client_safety_tokens_assumed": context_budget.CLIENT_SAFETY_TOKENS if enforced else None,
        # The four different numbers, kept apart. Only the server limit and the admitted tier are enforced; a native
        # capability or a configured 262144 never means the machine can run it.
        "context": _context_report(service, limit, enforced),
        "reasoning": {
            "efforts": ["low", "medium", "high", "xhigh"],
            "max": "accepted and clamped to xhigh (no distinct level on this checkpoint)",
            "off": "reasoning.effort=none, reasoning.enabled=false or chat_template_kwargs.enable_thinking=false",
            "wire": ["reasoning.effort", "reasoning_effort", "chat_template_kwargs.enable_thinking", "enable_thinking"],
            "default": "thinking off, effort medium (when the request carries no reasoning field)",
            "applied_to": "the request that carries it; nothing is stored on the server or the session",
            "decoder_thinking_budget": False,
            "response_header": "X-Veystrix-Reasoning",
        },
    }


def _context_report(service, limit, enforced) -> dict:
    backend = _service_backend(service)
    native = context_budget.native_context_window(getattr(backend, "model_path", None)) if backend is not None else None
    if native is None:
        native = 262144
    cold = getattr(service, "max_cold_prefill", None) or MAX_COLD_PREFILL_DEFAULT
    incremental = getattr(service, "max_incremental_prefill", None) or MAX_INCREMENTAL_PREFILL_DEFAULT
    return {
        "native_positional_capability": native,
        "architectural_context_maximum": 262144,
        "configured_server_limit": limit if enforced else 262144,
        "max_cold_prefill": cold,
        "max_incremental_prefill": incremental,
        "memory_admitted_max_cold_prefill": getattr(service, "max_cold_prefill", None),
        "incremental_accumulation_required": True,
        "preferred_internal_chunk_size": 2048,
        "validated_on_this_machine": dict(context_budget.VALIDATION),
        "overflow": f"a prompt over the configured limit is refused with HTTP 400 context_length_exceeded; cold prefill > {cold} (or uncached increment > {incremental}) refused with oversized_prefill_rejected / recovery_required",
        "validated_prefill_tokens_on_real_checkpoint": 16384,
    }


def health(service, scheduler) -> dict:
    """Liveness plus readiness. ``ready`` is false after a GPU lifecycle fault (the process must be restarted)."""
    try:
        from models.flashnext import pool_lifecycle

        fault = pool_lifecycle.lifecycle_fault()
    except Exception:
        fault = None
    loaded = _service_backend(service) is not None
    ready = loaded and not fault
    status = "ok" if ready else "unavailable"
    worker = _worker_state(scheduler)
    controller = None
    try:
        from models.flashnext import memory_controller as _mc
        controller = _mc.current()
    except Exception:
        pass
    if ready and controller is not None and not controller.can_admit():
        ready, status = False, "stopped"          # safety stop: work admission is closed for the rest of this process
    elif ready and worker["stalled"]:
        ready, status = False, "stalled"          # a job is running but nothing has advanced for stall_seconds
    report = {"status": status, "ready": ready, "worker": worker, "model": getattr(service, "model", ""),
              "busy": scheduler.busy, "queue": scheduler.snapshot(),
              "context_limit": getattr(service, "context_limit", None) or 262144,
              "max_cold_prefill": getattr(service, "max_cold_prefill", None) or MAX_COLD_PREFILL_DEFAULT,
              "max_incremental_prefill": getattr(service, "max_incremental_prefill", None) or MAX_INCREMENTAL_PREFILL_DEFAULT}
    if fault:
        report["fault"] = str(fault)
    state = getattr(service, "cache_state", None)
    if callable(state):
        report["cache"] = state()
    mem = _memory_ledger()
    if mem is not None:
        report["memory_policy"] = {
            "policy_identifier": mem.get("policy_identifier"),
            "swap_baseline_mb": mem.get("swap_baseline_mb"),
            "current_swap_growth_mb": mem.get("current_swap_growth_mb"),
            "peak_swap_growth_mb": mem.get("peak_swap_growth_mb"),
            "threshold_gib": mem.get("threshold_gib"),
            "threshold_mb": mem.get("threshold_mb"),
            "swap_ceiling_mib": mem.get("swap_ceiling_mib"),
            "swap_ceiling_margin_mib": mem.get("swap_ceiling_margin_mib"),
            "stop_reason": mem.get("stop_reason"),
            "can_admit": mem.get("can_admit", True),
            "probe_availability": mem.get("probe_availability", {}),
            "telemetry_only_checks": mem.get("telemetry_only_checks", []),
        }
    else:
        try:
            from models.flashnext import memory_policy
            report["memory_policy"] = memory_policy.policy_summary()
        except ImportError:
            pass
    return report


STALL_SECONDS = float(os.environ.get("VEYSTRIX_STALL_SECONDS", "180"))
RESTART_COMMAND = "./chat.sh --mode agentic --pool-gb 0 --context-limit 262144 --max-cold-prefill 32768 --max-incremental-prefill 32768"


def _memory_stopped() -> bool:
    try:
        from models.flashnext import memory_controller
        return memory_controller.is_stopped()
    except ImportError:
        return False


def _safety_stop_error(prefix: str = "request cancelled") -> RequestError:
    """The actionable failure a client gets when the memory safety stop ends its request (never a silent hang)."""
    reason = "memory limit exceeded"
    try:
        from models.flashnext import memory_controller
        c = memory_controller.current()
        if c is not None and c.stop_reason:
            reason = c.stop_reason
    except ImportError:
        pass
    return RequestError(
        f"{prefix} by the memory safety stop: {reason}. This server will exit and accepts no further work. "
        f"Do NOT retry automatically: restart it deliberately with `{RESTART_COMMAND}` once memory pressure has eased.",
        503, "memory_critical_stop")


def _worker_state(scheduler, clock=time.monotonic) -> dict:
    """Forward progress of the inference worker. ``stalled`` means a job is active and no streamed layer or decoded token has
    completed for ``stall_seconds``: the server must not call itself healthy then. Heavy prefill is not a stall: every
    streamed MoE layer ticks the counter."""
    try:
        from models.flashnext import progress
        snap = progress.PROGRESS.snapshot(clock)
    except Exception:
        snap = {"stage": "unknown", "seconds_since_progress": 0.0, "progress_ticks": 0, "in_request": False}
    busy = bool(getattr(scheduler, "busy", False))
    snap["busy"] = busy
    snap["stall_seconds"] = STALL_SECONDS
    snap["stalled"] = bool(busy and snap.get("seconds_since_progress", 0.0) > STALL_SECONDS)
    return snap


def _anthropic_stop(result) -> str:
    if getattr(getattr(result, "stats", None), "finish", None) == "length":
        return "max_tokens"
    if getattr(result, "tool_calls", None):
        return "tool_use"
    return "end_turn"


def _log_request(backend, stats, wall: float, first_piece, calls: int, started: float, text=None, extra=None) -> None:
    """Opt-in per-request record (MACQWEN_REQUEST_LOG=1 for stderr, or a JSONL path)."""
    target = os.environ.get("MACQWEN_REQUEST_LOG")
    if not target:
        return
    try:
        prefix = getattr(backend, "last_prefix", None)
        row = {
            "time": time.time(), "wall_seconds": round(wall, 3),
            "ttft_seconds": None if first_piece is None else round(first_piece - started, 3),
            "prompt_tokens": getattr(stats, "prompt_tokens", None),
            "prefill_seconds": round(getattr(stats, "prefill_seconds", 0.0), 3),
            "prefill_tokens_per_second": round(getattr(stats, "prompt_rate", 0.0), 2),
            "completion_tokens": getattr(stats, "tokens", None),
            "decode_seconds": round(getattr(stats, "seconds", 0.0), 3),
            "decode_tokens_per_second": round(getattr(stats, "rate", 0.0), 3),
            "finish": getattr(stats, "finish", None), "tool_calls": calls,
            "prefix": prefix.as_dict() if hasattr(prefix, "as_dict") else prefix,
        }
        if text is not None:
            row["raw_output"] = text[:8000]          # what the model generated, before any parsing (diagnosis of tool calls)
        if extra:
            row.update(extra)
        line = json.dumps(row, default=str)
        if target == "1":
            print("request " + line, file=sys.stderr, flush=True)
            return
        with open(target, "a") as handle:
            handle.write(line + "\n")
    except Exception as exc:
        print(f"Warning: _log_request failed: {exc}", file=sys.stderr, flush=True)


def _memory_ledger():
    """The staged memory controller's counters (swap at launch, peak swap increase, recoveries); None when no guard runs."""
    try:
        from models.flashnext import memory_controller
        return memory_controller.ledger()
    except Exception:
        return None


class ModelService:
    """Own one loaded model and serve requests from one warm cache."""

    def __init__(self, session, context_limit=None, max_cold_prefill=None, max_incremental_prefill=None):
        self.session = session
        self.model = f"macqwen-{session.preferences['model']}"
        self.reused = 0
        self.rebuilt = 0
        self.evictions = 0                  # live states replaced by a request that did not extend them
        self.refused_cold_prefill = 0
        self._request = threading.local()   # per-request controls (reasoning), set by the handler thread before complete()
        self.last_reasoning = None          # macqwen.reasoning.Reasoning of the last request that reached the model
        # Tokens a prompt may hold. None keeps the model's advertised window (default 262144).
        self.context_limit = context_limit
        # Largest single prefill (a cold rebuild, or the new tokens of a continuation) that the admitted memory tier
        # allows. None means no tier limit. Requests above it are refused before any state changes.
        self.max_cold_prefill = max_cold_prefill
        self.max_incremental_prefill = max_incremental_prefill
        self.last_prompt_tokens = None
        self.last_cached_tokens = 0
        self.last_divergence = None         # why the last request could not reuse the live tape (see _adopt)
        self.last_budget = None             # context_budget.Budget (as dict) of the last request that reached the model
        self.sessions = SessionRegistry()
        self.reasoning_ledger = ReasoningLedger()
        self.session_hint = None            # set by the HTTP layer for the request being served

    def warm_tokens(self) -> int:
        backend = getattr(self.session, "backend", None)
        try:
            return len(getattr(backend, "tape", ()) or ()) if backend is not None else 0
        except TypeError:
            return 0

    def cache_diagnostics(self) -> dict:
        backend = getattr(self.session, "backend", None)
        last_prefix = getattr(backend, "last_prefix", None)
        prefix_dict = last_prefix.as_dict() if hasattr(last_prefix, "as_dict") else last_prefix
        store = None
        st = None
        if backend is not None and hasattr(backend, "_prefix_session"):
            try:
                store = backend._prefix_session().store()
                st = store.stats
            except Exception:
                pass

        active_bytes = 0
        if backend is not None and getattr(backend, "cache", None):
            try:
                for c in backend.cache:
                    if hasattr(c, "nbytes"):
                        active_bytes += c.nbytes
                    elif hasattr(c, "state") and isinstance(c.state, (list, tuple)):
                        active_bytes += sum(getattr(a, "nbytes", 0) for a in c.state if a is not None)
                    elif hasattr(c, "cache") and isinstance(c.cache, (list, tuple)):
                        active_bytes += sum(getattr(a, "nbytes", 0) for a in c.cache if a is not None)
            except Exception:
                pass

        ram_bytes = store._ram_bytes if store else 0
        ram_entries = len(store._ram) if store else 0
        ram_budget = store.ram_budget if store else 0
        ssd_bytes = sum(m["bytes"] for m in store._disk.values()) if store else 0
        ssd_entries = len(store._disk) if store else 0
        ssd_budget = store.disk_budget if store else 0
        ssd_dir = str(store.directory) if store and store.directory else None

        swap_growth_mb = 0.0
        swap_base_mb = 0.0
        swap_peak_growth_mb = 0.0
        try:
            from models.flashnext import memory_controller
            c = memory_controller.current()
            if c:
                swap_growth_mb = round(c.swap_increase_mb(), 2)
                swap_base_mb = round(getattr(c, "baseline_swap_used_mb", 0.0), 2)
                swap_peak_growth_mb = round(getattr(c, "_peak_swap_growth_mb", 0.0), 2)
        except Exception:
            pass

        last_prefill_tok = getattr(last_prefix, "prefilled_tokens", 0) if last_prefix else 0
        last_prefill_sec = getattr(last_prefix, "prefill_seconds", 0.0) if last_prefix else 0.0
        matched_prefix_tok = getattr(last_prefix, "hit_tokens", 0) if last_prefix else 0
        last_decode_rate = getattr(backend, "last_decode_rate", None)

        return {
            "cache_enabled": store is not None,
            "ram_cache_bytes": ram_bytes,
            "ram_cache_entries": ram_entries,
            "ram_cache_budget_bytes": ram_budget,
            "ssd_cache_bytes": ssd_bytes,
            "ssd_cache_entries": ssd_entries,
            "ssd_cache_budget_bytes": ssd_budget,
            "ssd_cache_directory": ssd_dir,
            "active_model_state_bytes": active_bytes,
            "matched_prefix_tokens": matched_prefix_tok,
            "reused_tokens": st.reused_tokens if st else 0,
            "ram_hits": st.ram_hits if st else 0,
            "ssd_hits": st.ssd_hits if st else 0,
            "static_prefix_hits": sum(1 for m in store._disk.values() if m.get("n", 0) > 0) if store else 0,
            "session_continuation_hits": self.reused,
            "reasoning_state_restoration_hits": getattr(self.reasoning_ledger, "restored_turns_total", 0),
            "prefix_mismatches": getattr(st, "prefix_mismatches", 0) if st else 0,
            "state_invalids": getattr(st, "state_invalids", 0) if st else 0,
            "evicted_hits": getattr(st, "evicted_hits", 0) if st else 0,
            "restore_failures": getattr(st, "restore_failures", 0) if st else 0,
            "cold_lookups": getattr(st, "cold_lookups", 0) if st else 0,
            "evictions_ram": getattr(st, "evicted_ram", 0) if st else 0,
            "evictions_ssd": getattr(st, "evicted_ssd", 0) if st else 0,
            "ssd_writes": getattr(st, "ssd_writes", 0) if st else 0,
            "ssd_bytes_written": getattr(st, "ssd_bytes_written", 0) if st else 0,
            "ssd_bytes_read": getattr(st, "ssd_bytes_read", 0) if st else 0,
            "writes_skipped": getattr(st, "writes_skipped", 0) if st else 0,
            "restore_seconds": getattr(st, "restore_seconds", 0.0) if st else 0.0,
            "snapshot_seconds": getattr(last_prefix, "snapshot_seconds", 0.0) if last_prefix else 0.0,
            "newly_computed_prefill_tokens": last_prefill_tok,
            "actual_prefill_seconds": last_prefill_sec,
            "decode_tokens_per_second": last_decode_rate,
            "current_swap_growth_mb": swap_growth_mb,
            "swap_baseline_mb": swap_base_mb,
            "peak_swap_growth_mb": swap_peak_growth_mb,
            "corruption_recovery_errors": (getattr(st, "state_invalids", 0) + getattr(st, "restore_failures", 0)) if st else 0,
            "reused": self.reused,
            "rebuilt": self.rebuilt,
            "evictions": self.evictions,
            "refused_cold_prefill": self.refused_cold_prefill,
            "warm_tokens": self.warm_tokens(),
            "max_cold_prefill": self.max_cold_prefill,
            "max_incremental_prefill": self.max_incremental_prefill or MAX_INCREMENTAL_PREFILL_DEFAULT,
            "sessions": self.sessions.snapshot(),
            "last_budget": self.last_budget,
            "last_divergence": self.last_divergence,
            "last_prefix": prefix_dict,
            "memory_controller": _memory_ledger(),
        }

    def cache_state(self) -> dict:
        return self.cache_diagnostics()

    def _check_context(self, tokens: int) -> None:
        self.last_prompt_tokens = tokens
        limit = self.context_limit or ARCHITECTURAL_CONTEXT_MAXIMUM
        if limit and tokens + MIN_OUTPUT_RESERVE > limit:
            raise ContextLimitError(
                "context limit exceeded: " + context_budget.overflow_guidance(tokens, limit)
                + " Nothing was truncated and nothing was dropped.")

    def _check_prefill(self, tokens: int, total_tokens: int = 0, matched_prefix: int = 0,
                       is_cold: bool = True, outcome: str = "COLD", incremental: bool = False,
                       live_extension: bool = False) -> None:
        """Refuse a prefill larger than the allowed prefill limit. Runs before any state changes.

        ``incremental`` (the request extends live or restored state) is judged against ``max_incremental_prefill``; a cold or
        rebuilt prompt against ``max_cold_prefill``. Both default to 32,768 and neither is a licence to split a rejected
        cold prompt into chunks: chunking (2,048 tokens) happens only inside an admitted prefill."""
        if incremental:
            limit = self.max_incremental_prefill
            effective_limit = limit if limit is not None else MAX_INCREMENTAL_PREFILL_DEFAULT
            # A memory tier admitted at startup (max_cold_prefill) bounds every prefill, incremental ones included.
            effective_limit = min(effective_limit,
                                  self.max_cold_prefill if self.max_cold_prefill is not None else MAX_COLD_PREFILL_DEFAULT)
        else:
            limit = self.max_cold_prefill
            effective_limit = limit if limit is not None else MAX_COLD_PREFILL_DEFAULT

        if tokens > effective_limit:
            self.refused_cold_prefill += 1
            if live_extension:
                raise OversizedPrefillError(
                    f"this request adds {tokens} uncached tokens on top of {matched_prefix} tokens of reusable state "
                    f"(total prompt: {total_tokens}), which exceeds the per-request incremental prefill limit of "
                    f"{effective_limit} tokens. The 262K window is filled incrementally, never in one step: send "
                    "less new content per turn (for example fewer or smaller tool outputs). Nothing was run and the "
                    "live state is unchanged.",
                    code="oversized_prefill_rejected")
            if total_tokens > MAX_COLD_PREFILL_DEFAULT:
                if is_cold or matched_prefix == 0:
                    raise OversizedPrefillError(
                        f"this request needs {tokens} tokens of cold prefill (total prompt: {total_tokens}), "
                        f"which exceeds the maximum cold prefill limit of {effective_limit} tokens. Native 262K context "
                        f"requires incremental state accumulation (max {self.max_incremental_prefill or MAX_INCREMENTAL_PREFILL_DEFAULT} "
                        "uncached tokens per turn), not cold prefill. "
                        "Do not split a giant cold prompt across chunks to bypass this limit. Nothing was run and "
                        "the live state is unchanged.",
                        code="oversized_prefill_rejected",
                    )
                else:
                    raise RecoveryRequiredError(
                        f"session state could not be recovered from memory or SSD cache (total prompt: {total_tokens} "
                        f"tokens, matched prefix: {matched_prefix} tokens, uncached tokens: {tokens}, outcome: {outcome}). "
                        f"Re-prefilling {tokens} tokens cold would exceed the safe prefill limit of {effective_limit} "
                        "tokens and is prohibited on 16 GB hardware. A compatible state snapshot is required for recovery. "
                        "Shorten the prompt or resume from a saved session snapshot.",
                        code="recovery_required",
                    )
            raise ContextLimitError(
                f"this request needs {tokens} tokens prefilled but this server was admitted for at most {effective_limit} per "
                "prefill under the memory available at startup. Nothing was truncated, dropped or run; the live state "
                "is unchanged. Shorten the conversation, or restart when more memory is free.",
                code="prefill_budget_exceeded")

    @staticmethod
    def _substitute_slots(normalized, tokenizer):
        """Stand sentinels in for marker-carrying user/system content.

        Returns (render_messages, slot_texts). Assistant history is the
        model's own output, so its genuine thinking structure keeps joint
        encoding; user, system, and tool content is untrusted and splits
        when it carries control markers. No markers anywhere means no
        slots, and the prompt renders exactly as before.
        """
        try:
            codec = build_user_encoder(tokenizer)
        except (AttributeError, TypeError):
            return normalized, []
        render_messages, slot_texts = [], []
        for message in normalized:
            item = dict(message)
            content = item.get("content", "")
            if (
                isinstance(content, str)
                and item.get("role") in ("user", "system", "tool")
                and codec[0].search(content) is not None
            ):
                item["content"] = content_sentinel(len(slot_texts))
                slot_texts.append(content)
            render_messages.append(item)
        if not slot_texts:
            return normalized, []
        return render_messages, slot_texts

    def _adopt_split(self, rendered, slot_texts, tokenizer, effort_text="") -> bool:
        """Adopt a sentinel render with content-safe slot encoding.

        Returns False when the template transformed a sentinel or the
        backend cannot encode structurally: the caller adopts jointly.
        """
        try:
            codec = build_user_encoder(tokenizer)
        except (AttributeError, TypeError):
            return False
        encode = getattr(self.session.backend, "encode", None)
        if encode is None:
            return False
        parts = split_content_slots(rendered, len(slot_texts))
        if parts is None:
            return False
        safe = codec[1]
        ids = list(encode(parts[0]))
        for content, structural in zip(slot_texts, parts[1:]):
            ids.extend(safe(content))
            ids.extend(encode(structural))
        self._adopt(rendered, ids=ids, effort_text=effort_text)
        return True

    def _with_effort_turn(self, ids, effort_text: str):
        """Insert the effort turn right after the system message's ``<|im_end|>\\n`` (see macqwen/effort_slot.py)."""
        if not effort_text:
            return ids
        from models.flashnext import prefix_session

        encode = self.session.backend.encode
        marker = [int(t) for t in encode(prefix_session.IM_END)]
        ids = [int(t) for t in ids]
        width = len(marker)
        for index in range(len(ids) - width + 1):
            if ids[index:index + width] == marker:
                cut = index + width
                return ids[:cut] + [int(t) for t in encode(effort_text)] + ids[cut:]
        flight_recorder.record("effort_turn_not_inserted", reason="no system end marker in the prompt")
        return ids

    def _adopt(self, rendered: str, ids=None, effort_text: str = "") -> bool:
        """Feed the prompt, keeping the cache when it holds a prefix.

        A client resends the whole conversation on every call. When the new
        prompt extends what the cache already holds, only the new tokens need
        a prefill. When it diverges the cache has to go: the recurrent layers
        carry state that cannot be rewound. The prefill size is checked against
        the admitted tier BEFORE anything is reset, so a refusal keeps the live state.
        """
        session = self.session
        backend = session.backend
        encode = getattr(backend, "encode", None)
        common_prefix = getattr(backend, "common_prefix", None)
        if encode is None or common_prefix is None:
            session.reset()
            backend.append_text(rendered)
            self.rebuilt += 1
            return False
        if ids is None:
            ids = encode(rendered)
        if effort_slot.placement() == "late":
            ids = self._with_effort_turn(ids, effort_text)
        self._check_context(len(ids))
        replay = bool(getattr(backend, "_replay_needed", False))
        shared = 0
        extends = False
        if not replay and not backend.pending and backend.tape:
            shared = common_prefix(ids)
            extends = shared == len(backend.tape) and shared < len(ids)

        cached_tokens = 0
        cache_outcome = "COLD"
        if not extends:
            prefix_session = getattr(backend, "_prefix_session", None)
            if callable(prefix_session):
                can_restore = True
                try:
                    from macqwen.backends.flashnext import FlashNextBackend
                    if isinstance(backend, FlashNextBackend):
                        from models.flashnext import prefix_session as ps_mod
                        if getattr(backend, "decoder", None) is not None or not (ps_mod.enabled() and ps_mod.restore_enabled()):
                            can_restore = False
                except ImportError:
                    pass
                if can_restore:
                    try:
                        cached_tokens = int(prefix_session().reusable_tokens(ids))
                        cache_outcome = "SNAPSHOT_HIT" if cached_tokens > 0 else "MISS"
                    except Exception:
                        cached_tokens = 0
                        cache_outcome = "ERROR"
        needed_prefill = len(ids) - shared if extends else max(0, len(ids) - cached_tokens)
        matched_prefix = shared if extends else cached_tokens
        self._last_extends = bool(extends)
        self.last_cached_tokens = int(matched_prefix)
        had_state = self.warm_tokens() > 0 or bool(getattr(self, "session_hint", None))
        is_cold = (not extends) and (matched_prefix == 0) and (not had_state)
        self._check_prefill(
            tokens=needed_prefill,
            total_tokens=len(ids),
            matched_prefix=matched_prefix,
            is_cold=is_cold,
            outcome=cache_outcome,
            incremental=bool(extends or cached_tokens > 0),
            # "The increment is too big" only when most of the prompt is already cached; a mostly uncached prompt is a lost state.
            live_extension=bool(extends and shared * 2 >= len(ids)),
        )
        flight_recorder.record("adopt", prompt_tokens=len(ids), matched_tokens=matched_prefix, needed_prefill=needed_prefill,
                               extends=bool(extends), outcome=cache_outcome, warm_tokens=self.warm_tokens(),
                               session_hint=bool(getattr(self, "session_hint", None)))
        if self.warm_tokens() and not extends:
            self.evictions += 1
            # Diagnostic only. A rebuild of a long conversation is expensive (12K tokens of prefill) and the cause is
            # invisible from outside: either another conversation took the state, or the client's history re-encodes
            # differently from the tokens the model really generated (thinking text not echoed back, a non-canonical
            # BPE split). Record where the two sequences first differ so a real run can tell which.
            tape = list(getattr(backend, "tape", ()) or ())
            if tape and not replay and not backend.pending:
                where = shared
                window = lambda seq: [int(t) for t in seq[max(0, where - 3):where + 4]]   # noqa: E731
                self.last_divergence = {"tape_tokens": len(tape), "prompt_tokens": len(ids), "first_difference": where,
                                        "tape_window": window(tape), "prompt_window": window(ids),
                                        "tape_is_prefix_of_prompt": False}
        if replay:
            # The transcript may still be a prefix, but the live cache is
            # deliberately invalid. Reusing it would make a tool result
            # appear incremental while the backend silently re-prefills the
            # entire conversation on the next generation.
            session.reset()
            backend.append_tokens(ids)
            self.rebuilt += 1
            return False
        if extends:
            backend.append_tokens(ids[shared:])
            self.reused += 1
            return True
        session.reset()
        backend.append_tokens(ids)
        self.rebuilt += 1
        return False

    def request_reasoning(self):
        """The reasoning controls of the request this thread is serving, or the server defaults (never stored on the session)."""
        from macqwen import reasoning as R

        current = getattr(self._request, "reasoning", None)
        if current is not None:
            return current
        prefs = self.session.preferences
        return R.resolve({}, prefs["thinking_enabled"], prefs["effort"])

    def complete(self, messages, tools, max_tokens, on_text=None, gone=None) -> Completion:
        try:
            from models.flashnext import memory_controller
            if not memory_controller.can_admit_work():
                c = memory_controller.current()
                reason = c.stop_reason if c and c.stop_reason else "memory controller stopped work admission"
                raise RequestError(f"request rejected: {reason}", status=503, code="memory_critical_stop")
        except ImportError:
            pass
        session = self.session
        backend = session.backend
        reasoning = self.request_reasoning()
        self.last_reasoning = reasoning
        hint, self.session_hint = self.session_hint, None
        had_state = self.warm_tokens() > 0
        key = None
        try:
            normalized = _normalize_messages(messages)
            system_index = next(
                (index for index, message in enumerate(normalized)
                 if message["role"] == "system"),
                None,
            )
            if system_index is None:
                current_prompt = getattr(session, "current_system_prompt", None)
                prompt = (
                    current_prompt()
                    if callable(current_prompt)
                    else system_prompt("plain", session.preferences["workspace"])
                )
                normalized.insert(0, {
                    "role": "system", "content": prompt,
                })
                system_index = 0
            effort_text = ""
            if effort_slot.placement() == "late":
                # Render exactly as "medium" (no effort sentence, nothing prepended), then insert the effort sentence as its own
                # system turn right after the system message: the tools block and the system prompt, and the state cached at
                # the end of the system message, are then the same for every effort.
                system_text, template_effort = normalized[system_index]["content"], "medium"
                effort_text = effort_slot.block_text(reasoning.effort, reasoning.thinking)
            else:
                system_text, template_effort = reasoning_system_text(
                    normalized[system_index]["content"],
                    reasoning.effort,
                )
            key = derive_key(normalized, hint, tools=tools)
            self.reasoning_ledger.restore_reasoning(key, normalized)
            normalized[system_index] = dict(normalized[system_index], content=system_text)
            tokenizer = _model_tokenizer(backend)
            render_messages, slot_texts = self._substitute_slots(
                normalized, tokenizer
            )
            rendered = tokenizer.apply_chat_template(
                render_messages,
                tools=_openai_tools(tools) or None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=reasoning.thinking,
                reasoning_effort=template_effort,
            )
            if slot_texts:
                before = self.reused
                if not self._adopt_split(rendered, slot_texts, tokenizer, effort_text=effort_text):
                    raise RequestError(
                        "unable to safely encode content placeholders"
                    )
            else:
                before = self.reused
                self._adopt(rendered, effort_text=effort_text)
            self.sessions.activate(key, prompt_tokens=self.last_prompt_tokens or 0, reused=self.reused > before,
                                   had_state=had_state)
            stream_filter = ReasoningStreamFilter(
                reasoning.thinking
            )
            words = CompletedTextBuffer()
            hold = bool(tools)

            began = time.perf_counter()
            first = []

            checked = [time.perf_counter()]

            import inspect

            on_text_takes_kind = False
            if on_text is not None:
                try:
                    ot_sig = inspect.signature(on_text)
                    on_text_takes_kind = (
                        "kind" in ot_sig.parameters
                        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ot_sig.parameters.values())
                    )
                except Exception:
                    pass

            def send_text(chunk, kind="content"):
                if on_text is None:
                    return
                if on_text_takes_kind:
                    on_text(chunk, kind=kind)
                elif kind == "content":
                    on_text(chunk)

            def streamed(piece: str) -> None:
                if not first:
                    first.append(time.perf_counter())
                    flight_recorder.record("first_token", seconds_since_start=round(first[0] - began, 3))
                if _memory_stopped():
                    raise _safety_stop_error("generation cancelled")
                if gone is not None:                # a closed client ends the generation instead of finishing it unseen
                    now = time.perf_counter()
                    if now - checked[0] >= GONE_CHECK_SECONDS:
                        checked[0] = now
                        if gone():
                            raise ClientGone("client disconnected during generation")
                for kind, item in stream_filter.feed(piece):
                    if kind == "thinking":
                        send_text(item, kind="thinking")
                    else:
                        for complete in words.feed(item):
                            send_text(None if hold else complete, kind="content")

            def cancel_check() -> bool:
                try:
                    from models.flashnext import memory_controller
                    if memory_controller.is_stopped():
                        return True
                except ImportError:
                    pass
                if gone is not None and gone():
                    return True
                return False

            requested_output = max_tokens
            server_limit = self.context_limit or 262144
            if self.last_prompt_tokens:
                available_for_output = max(1, server_limit - self.last_prompt_tokens)
                max_tokens = min(max_tokens, available_for_output)
            budget = context_budget.evaluate(self.context_limit, self.last_prompt_tokens or 0, requested_output)
            self.last_budget = budget.as_dict()
            gen_kwargs = {
                "max_tokens": max_tokens,
                "out": streamed if (on_text is not None or gone is not None) else None,
            }
            progress_mark = [time.monotonic(), -1]

            def prefill_progress(done: int, total: int) -> None:
                # Throttled lifecycle record: the last line before a stall says how far prefill got.
                now = time.monotonic()
                if done != progress_mark[1] and (now - progress_mark[0] >= 10.0 or done >= total or done == 0):
                    progress_mark[0], progress_mark[1] = now, done
                    flight_recorder.record("prefill_progress", done=int(done), total=int(total),
                                           elapsed=round(time.perf_counter() - began, 1))

            flight_recorder.record("generate_begin", prompt_tokens=self.last_prompt_tokens, max_tokens=max_tokens,
                                   thinking=reasoning.thinking, effort=reasoning.effort if reasoning.thinking else None)

            gen_params = inspect.signature(backend.generate).parameters
            call_kwargs = dict(gen_kwargs)
            if "should_cancel" in gen_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in gen_params.values()):
                call_kwargs["should_cancel"] = cancel_check
            if "on_prefill_progress" in gen_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in gen_params.values()):
                call_kwargs["on_prefill_progress"] = prefill_progress

            text, stats = backend.generate(**call_kwargs)

            last_prefix = getattr(backend, "last_prefix", None)
            if last_prefix is not None:
                self.last_cached_tokens = int(getattr(last_prefix, "hit_tokens", 0) or 0)
            elif not getattr(self, "_last_extends", False):
                self.last_cached_tokens = 0

            flight_recorder.record("generate_end", finish=getattr(stats, "finish", None),
                                   prompt_tokens=getattr(stats, "prompt_tokens", None),
                                   completion_tokens=getattr(stats, "tokens", None),
                                   prefill_seconds=round(getattr(stats, "prefill_seconds", 0.0), 2),
                                   decode_tokens_per_second=round(getattr(stats, "rate", 0.0), 2))
            if on_text is not None:
                for kind, item in stream_filter.finish():
                    if kind == "thinking":
                        send_text(item, kind="thinking")
                    else:
                        for complete in words.feed(item):
                            send_text(None if hold else complete, kind="content")
                for complete in words.finish():
                    send_text(None if hold else complete, kind="content")

            extracted_reasoning, raw_visible = extract_reasoning_and_content(
                text, reasoning.thinking
            )
            visible, calls = _parse_tool_calls(raw_visible, tools)
            if key is not None:
                self.reasoning_ledger.record_turn(key, visible, calls, extracted_reasoning)
            # Post-turn snapshot: OPT-IN (FLASHNEXT_TURN_SNAPSHOTS=1, with the prefix cache enabled). Off by default since
            # 2026-11-09: a state produced by decode steps has no chunk-plan provenance, so the exact-prefix lookup never
            # accepts it (plan None is "unknown history"), yet each one cost a ~120-400 MB serialisation and SSD write per
            # turn, and it ran even with the prefix cache disabled (the incident server wrote one at 12:41:12). Turn
            # boundaries are snapshotted for free during the next request's prefill instead (message-boundary anchors).
            if (os.environ.get("FLASHNEXT_TURN_SNAPSHOTS") == "1" and os.environ.get("FLASHNEXT_PREFIX_CACHE") == "1"
                    and backend is not None and hasattr(backend, "_prefix_session")):
                try:
                    ps = backend._prefix_session()
                    if ps is not None and getattr(backend, "tape", None) and getattr(backend, "cache", None):
                        store = ps.store()
                        executed = getattr(backend, "last_prefix", None)
                        plan = tuple(executed.executed_edges) if executed and hasattr(executed, "executed_edges") else None
                        store.put(
                            tokens=tuple(backend.tape),
                            cache_list=backend.cache,
                            language=getattr(backend, "language", None),
                            plan=plan,
                            kind="turn_complete",
                        )
                except Exception:
                    pass
            _log_request(backend, stats, time.perf_counter() - began,
                         first[0] if first else None, text.count("<tool_call>"), began, text=text,
                         extra={"context_budget": self.last_budget, "divergence": self.last_divergence,
                                "reused": self.reused, "rebuilt": self.rebuilt, "evictions": self.evictions,
                                "turn_memory": getattr(backend, "last_turn_memory", None),
                                "reasoning": reasoning.as_dict()})
            if (tools and not calls and getattr(stats, "finish", None) == "length"
                    and raw_visible.rfind("<tool_call>") > raw_visible.rfind("</tool_call>")):
                # The model began a tool call and the output limit ended it. Handing the client the raw marker as
                # ordinary text would look like an answer; nothing was executed and no call is invented.
                raise ModelToolCallError(
                    f"the model began a tool call but the output limit of {max_tokens} tokens ended it first; "
                    "no tool was called. Raise max_tokens / max_completion_tokens (the client may be computing "
                    "it from a context window that leaves too little room).")
            if hold and on_text is not None and visible:
                final_words = CompletedTextBuffer()
                for complete in final_words.feed(visible) + final_words.finish():
                    try:
                        on_text(complete, kind="content")
                    except TypeError:
                        on_text(complete)
            return Completion(visible, calls, stats, reasoning_content=extracted_reasoning)
        except ContextLimitError:
            flight_recorder.record("request_refused", kind="context_or_prefill_limit")
            raise
        except GenerationCancelled:
            session.reset()
            self.sessions.dropped()
            stopped = _memory_stopped()
            flight_recorder.record("request_cancelled", memory_stop=stopped)
            if stopped:
                raise _safety_stop_error() from None
            raise
        except BaseException as error:
            session.reset()
            self.sessions.dropped()
            flight_recorder.record("request_error", error=type(error).__name__, detail=str(error)[:300])
            from models.flashnext.request_admission import RequestMemoryRefused
            if isinstance(error, RequestMemoryRefused):
                raise RequestError(str(error), status=503, code="memory_admission_refused") from error
            raise


# Chat Completions request fields. Honoured ones are consumed by the server;
# ACCEPTED ones are harmless to the result and carry no model-visible meaning;
# IGNORED ones are sampling controls the server fixes per process (see
# --temperature), so they are listed in the X-Veystrix-Ignored-Fields header
# instead of vanishing. Everything else is refused by name.
CHAT_HONOURED = {"model", "messages", "stream", "stream_options", "max_tokens",
                 "max_completion_tokens", "tools",
                 # per-request reasoning controls, see macqwen/reasoning.py (applied to this request only)
                 "reasoning", "reasoning_effort", "chat_template_kwargs", "enable_thinking"}
CHAT_ACCEPTED = {"store", "user", "metadata"}
CHAT_IGNORED = {"temperature", "top_p", "presence_penalty", "frequency_penalty",
                "seed", "parallel_tool_calls"}


def _check_chat_fields(payload: dict) -> list[str]:
    """Refuse fields the server cannot honour; return the ones it ignores."""
    ignored = []
    for key, value in payload.items():
        if key in CHAT_HONOURED or key in CHAT_ACCEPTED or value is None:
            continue
        if key in CHAT_IGNORED:
            ignored.append(key)
        elif key == "tool_choice" and value == "auto":
            continue
        elif key == "n" and value == 1:
            continue
        else:
            raise RequestError(
                f"request field {key!r} is not supported by this server and "
                "was not applied"
            )
    stream_options = payload.get("stream_options")
    if stream_options is not None:
        if not isinstance(stream_options, dict):
            raise RequestError("stream_options must be an object")
        extra = sorted(set(stream_options) - {"include_usage"})
        if extra:
            raise RequestError(f"stream_options field {extra[0]!r} is not supported")
    return sorted(ignored)


class MacqwenHTTPServer(ThreadingHTTPServer):
    """Connections are handled by threads; inference is serialized by ``scheduler`` (see macqwen/scheduler.py)."""

    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, service, api_key=None, allowed_origins=(), scheduler=None):
        super().__init__(address, MacqwenHandler)
        self.service = service
        self.api_key = api_key
        self.allowed_origins = tuple(allowed_origins)
        self.scheduler = scheduler or Scheduler()


class MacqwenHandler(BaseHTTPRequestHandler):
    server: MacqwenHTTPServer
    protocol_version = "HTTP/1.1"
    _sse_open = False

    def log_message(self, _format, *_args):
        return

    def _authorized(self) -> bool:
        key = self.server.api_key
        if not key:
            return True
        # constant time, so a wrong key cannot be discovered byte by byte
        offered = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if offered.startswith(prefix) and hmac.compare_digest(
                offered[len(prefix):], key):
            return True
        return hmac.compare_digest(self.headers.get("x-api-key", "") or "", key)

    def _allowed_origin(self) -> str | None:
        """Which Origin, if any, may read this reply.

        A browser page on any site can reach a server bound to localhost.
        Answering every Origin with `*` would let any site you visit use this
        model and read its output, so an Origin is echoed back only when it
        was allowed explicitly with --allow-origin. Requests without an
        Origin header, which is every non-browser client, are unaffected.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return None
        allowed = self.server.allowed_origins
        if "*" in allowed or origin in allowed:
            return origin
        return None

    def _reject_foreign_origin(self) -> bool:
        """Refuse a browser request from an Origin that was not allowed."""
        if self.headers.get("Origin") and self._allowed_origin() is None:
            self._error(
                "this origin is not allowed; start the server with "
                "--allow-origin to permit browser clients",
                403,
            )
            return True
        return False

    def _headers(self, status=200, content_type="application/json"):
        from models.flashnext import progress

        progress.PROGRESS.response_started()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        origin = self._allowed_origin()
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        ignored = getattr(self, "_ignored_fields", None)
        if ignored:
            self.send_header("X-Veystrix-Ignored-Fields", ",".join(ignored))
        applied = getattr(self, "_reasoning", None)
        if applied is not None:
            self.send_header("X-Veystrix-Reasoning", applied.header())

    def _json(self, value, status=200, headers=None):
        body = json.dumps(value).encode()
        self._headers(status)
        self.send_header("Content-Length", str(len(body)))
        for name, text in (headers or {}).items():
            self.send_header(name, str(text))
        self.end_headers()
        self.wfile.write(body)

    ERROR_TYPES = {400: "invalid_request_error", 404: "invalid_request_error", 405: "invalid_request_error",
                   409: "invalid_request_error", 413: "invalid_request_error", 401: "authentication_error",
                   403: "permission_error", 429: "rate_limit_error"}

    def _error(self, message, status=400, code=None, headers=None, kind=None):
        """OpenAI error object: message, type, param and code are always present."""
        kind = kind or self.ERROR_TYPES.get(status, "server_error" if status >= 500 else "invalid_request_error")
        if code == "memory_critical_stop":
            # OpenAI-style SDKs honour this: a safety stop is never retried automatically, whatever maxRetries says.
            headers = {**(headers or {}), "x-should-retry": "false"}
        self._json({"error": {"message": str(message), "type": kind, "param": None, "code": code}}, status, headers)

    def _busy(self, exc):
        self._error(exc, 503, "server_busy", {"Retry-After": exc.retry_after})

    def _client_gone(self) -> bool:
        """True when the peer has closed its end. Never consumes data."""
        try:
            sock = self.connection
            readable, _, _ = select.select([sock], [], [], 0)
            return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
        except BlockingIOError:
            return False
        except (OSError, ValueError):
            return True

    def _drain_payload(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if 0 < length <= MAX_BODY:
                self.rfile.read(length)
        except Exception:
            pass

    def _payload(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError("invalid Content-Length") from exc
        if length < 1 or length > MAX_BODY:
            raise RequestError("request body size is invalid", 413)
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RequestError("request body must contain JSON") from exc
        if not isinstance(value, dict):
            raise RequestError("request body must be an object")
        return value

    def _sse_start(self):
        self._headers(content_type="text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._sse_open = True

    def _sse(self, value, event=None):
        prefix = f"event: {event}\n" if event else ""
        with self._write_lock():
            self.wfile.write((prefix + "data: " + json.dumps(value) + "\n\n").encode())
            self.wfile.flush()

    def _heartbeat(self):
        with self._write_lock():
            self.wfile.write(b": generating\n\n")
            self.wfile.flush()

    def _write_lock(self):
        lock = self.__dict__.get("_sse_lock")
        if lock is None:
            lock = self.__dict__.setdefault("_sse_lock", threading.Lock())
        return lock

    def _complete_with_keepalive(self, service, messages, tools, max_tokens, on_text):
        """``service.complete`` while a side thread keeps the SSE stream alive.

        A cold prefill can take many minutes and produces no token, so without this the stream is silent from the first
        role chunk until the first decoded token and a client idle timeout (Qwen Code's ``streamIdleTimeoutMs``) measures
        the prefill, not the connection. The comment line is invisible to SSE parsers. A failed write only ends the
        keep-alive: the cancellation checks still decide whether the work stops."""
        stop = threading.Event()

        def beat():
            while not stop.wait(SSE_KEEPALIVE_SECONDS):
                try:
                    self._heartbeat()
                except (OSError, ValueError):
                    return

        ticker = threading.Thread(target=beat, name="sse-keepalive", daemon=True)
        ticker.start()
        try:
            return service.complete(messages, tools, max_tokens, on_text, gone=self._client_gone)
        finally:
            stop.set()
            ticker.join(timeout=2.0)

    def _sse_chat_error(self, message):
        try:
            self._sse({"error": {"message": str(message),
                                 "type": getattr(message, "kind", None) or "server_error",
                                 "param": None, "code": getattr(message, "code", None)}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse_anthropic_error(self, message):
        try:
            self._sse({"type": "error", "error": {
                "type": "api_error", "message": str(message)}}, "error")
            self._sse({"type": "message_stop"}, "message_stop")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse_responses_error(self, message, ident=None):
        try:
            response = {"id": ident or "resp_unknown", "object": "response",
                        "status": "failed",
                        "error": {"message": str(message),
                                  "type": "server_error"}}
            self._sse({"type": "response.failed", "response": response},
                       "response.failed")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self._headers(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self._reject_foreign_origin():
            return
        if not self._authorized():
            self._error("invalid API key", 401, "invalid_api_key")
            return
        path = urlsplit(self.path).path
        service = self.server.service
        if path == "/health":
            self._json(health(service, self.server.scheduler))
        elif path == "/ready":
            report = health(service, self.server.scheduler)
            self._json(report, 200 if report["ready"] else 503)
        elif path in ("/cache/diagnostics", "/v1/cache/diagnostics"):
            self._json(service.cache_diagnostics())
        elif path == "/v1/models":
            card = model_card(service)
            self._json({"object": "list", "data": [model_entry(card, service)], "models": [card]})
        elif path.startswith("/v1/models/"):
            card = model_card(service)
            wanted = path[len("/v1/models/"):]
            if wanted == card["slug"]:
                self._json(model_entry(card, service))
            else:
                self._error(f"the model {wanted!r} does not exist", 404, "model_not_found")
        else:
            self._error("route not found", 404, "not_found")

    ROUTES = ("/v1/chat/completions", "/v1/messages", "/v1/responses")

    def do_POST(self):
        if self._reject_foreign_origin():
            return
        if not self._authorized():
            self._error("invalid API key", 401, "invalid_api_key")
            return
        self._sse_open = False
        self._ignored_fields = None
        self._reasoning = None
        path = urlsplit(self.path).path
        if path not in self.ROUTES:
            self._drain_payload()
            self._error("route not found", 404, "not_found")
            return
        try:
            payload = self._payload()
        except RequestError as exc:
            self._error(exc, exc.status, exc.code, kind=exc.kind)
            return
        except (BrokenPipeError, ConnectionResetError):
            return
        try:
            self.server.scheduler.run(lambda: self._handle_post(path, payload), gone=self._client_gone, path=path)
        except Busy as exc:
            try:
                self._busy(exc)
            except (BrokenPipeError, ConnectionResetError):
                pass
        except ClientGone:
            pass

    def _handle_post(self, path, payload):
        """Runs on the inference thread, one request at a time."""
        from models.flashnext import progress

        try:
            self.server.service.session_hint = explicit_id(self.headers, payload)
        except Exception:
            pass

        progress.PROGRESS.request_begin(getattr(self, "connection", None), path)
        try:
            tools_value = payload.get("tools")
            flight_recorder.record(
                "request_begin", path=path, seq=progress.PROGRESS._request,
                user_agent=str(self.headers.get("User-Agent", ""))[:80],
                session_header=bool(self.server.service.session_hint),
                stream=bool(payload.get("stream")),
                messages=len(payload.get("messages") or payload.get("input") or []),
                tools=len(tools_value) if isinstance(tools_value, list) else 0,
                reasoning_fields={k: payload[k] for k in ("reasoning_effort", "reasoning", "enable_thinking")
                                  if k in payload},
                max_tokens=payload.get("max_completion_tokens", payload.get("max_tokens")),
                waiting=self.server.scheduler.snapshot().get("waiting"))
        except Exception:
            pass
        try:
            if path == "/v1/chat/completions":
                self._chat(payload)
            elif path == "/v1/messages":
                self._anthropic(payload)
            else:
                self._responses(payload)
        except RequestError as exc:
            if not getattr(self, "_sse_open", False):
                try:
                    self._error(exc, exc.status, exc.code, kind=exc.kind)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        except (BrokenPipeError, ConnectionResetError, ClientGone):
            pass
        except GenerationCancelled:
            if not getattr(self, "_sse_open", False):
                try:
                    self._error("generation cancelled", 503, "generation_cancelled")
                except (BrokenPipeError, ConnectionResetError):
                    pass
        except Exception as exc:
            if not getattr(self, "_sse_open", False):
                try:
                    self._error(str(exc), 500, "internal_error")
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            progress.PROGRESS.request_end()
            flight_recorder.record("request_end", path=path)

    DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("VEYSTRIX_DEFAULT_MAX_OUTPUT_TOKENS", "2048"))

    def _limit(self, payload, default=None):
        if default is None:
            default = self.DEFAULT_MAX_OUTPUT_TOKENS
        value = payload.get(
            "max_output_tokens",
            payload.get("max_completion_tokens", payload.get("max_tokens", default)),
        )
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RequestError("max_tokens must be a positive integer")
        return min(value, capabilities(self.server.service)["context_window"])

    def _chat(self, payload):
        service = self.server.service
        self._ignored_fields = _check_chat_fields(payload)
        self._reasoning = self._resolve_reasoning(payload)
        local = getattr(service, "_request", None)
        if local is None:                       # a minimal stand-in service (tests): give it the per-request slot
            local = service._request = threading.local()
        local.reasoning = self._reasoning
        try:
            return self._chat_body(payload, service)
        finally:
            local.reasoning = None

    def _resolve_reasoning(self, payload):
        from macqwen import reasoning as R

        service = self.server.service
        prefs = getattr(getattr(service, "session", None), "preferences", None) or {}
        try:
            return R.resolve(payload, bool(prefs.get("thinking_enabled", False)), prefs.get("effort", "medium"))
        except R.ReasoningError as exc:
            raise RequestError(str(exc), 400, "unsupported_reasoning_effort")

    def _chat_body(self, payload, service):
        messages = _normalize_messages(payload.get("messages"))
        tools = payload.get("tools") or []
        warm_request = self.headers.get(warm_module.WARM_HEADER, "") == "1"
        if not warm_request:
            # Remember this client's static prefix (system message + tools) so ./serve.sh --warm can pre-cache every effort.
            warm_module.remember(str(self.headers.get("User-Agent", "")), messages, tools)
        cap = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if isinstance(cap, int) and not isinstance(cap, bool) and not warm_request:
            message = context_budget.too_small_for_tools(cap, bool(tools))
            if message:
                raise RequestError(message, 400, "max_tokens_too_small")
        ident = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if payload.get("stream"):
            max_tokens = 1 if warm_request else self._limit(payload)
            self._sse_start()
            try:

                def chunk(delta, finish=None):
                    self._sse({
                        "id": ident, "object": "chat.completion.chunk", "created": created,
                        "model": service.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    })

                chunk({"role": "assistant", "content": ""})

                def on_text(text, kind="content"):
                    if text is None:
                        self._heartbeat()
                    elif kind == "thinking":
                        chunk({"reasoning_content": text})
                    else:
                        chunk({"content": text})

                result = self._complete_with_keepalive(service, messages, tools, max_tokens, on_text)
                for index, call in enumerate(result.tool_calls):
                    chunk({"tool_calls": [{
                        "index": index, "id": call["id"], "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }]})
                chunk({}, "tool_calls" if result.tool_calls else result.stats.finish)
                if (payload.get("stream_options") or {}).get("include_usage"):
                    self._sse({
                        "id": ident, "object": "chat.completion.chunk", "created": created,
                        "model": service.model, "choices": [],
                        "usage": chat_usage(service, result.stats),
                    })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ClientGone):
                pass
            except Exception as exc:
                self._sse_chat_error(exc)
            return
        result = service.complete(messages, tools, self._limit(payload), gone=self._client_gone)
        message = {"role": "assistant", "content": result.text or None}
        if getattr(result, "reasoning_content", None):
            message["reasoning_content"] = result.reasoning_content
        if result.tool_calls:
            message["tool_calls"] = [{
                "id": call["id"], "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            } for call in result.tool_calls]
        self._json({
            "id": ident, "object": "chat.completion", "created": created,
            "model": service.model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if result.tool_calls else result.stats.finish}],
            "usage": chat_usage(service, result.stats),
        })

    def _anthropic(self, payload):
        service = self.server.service
        messages = _anthropic_messages(payload)
        tools = payload.get("tools") or []
        ident = f"msg_{uuid.uuid4().hex}"
        if payload.get("stream"):
            max_tokens = self._limit(payload)
            self._sse_start()
            try:
                self._sse({"type": "message_start", "message": {
                    "id": ident, "type": "message", "role": "assistant",
                    "model": service.model, "content": [], "stop_reason": None,
                    "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0},
                }}, "message_start")
                text_open = [False]

                def on_text(text):
                    if text is None:
                        self._heartbeat()
                        return
                    if not text_open[0]:
                        self._sse({"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "text", "text": ""}},
                                  "content_block_start")
                        text_open[0] = True
                    self._sse({"type": "content_block_delta", "index": 0,
                               "delta": {"type": "text_delta", "text": text}},
                              "content_block_delta")

                result = self._complete_with_keepalive(service, messages, tools, max_tokens, on_text)
                index = 0
                if text_open[0]:
                    self._sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
                    index = 1
                for call in result.tool_calls:
                    self._sse({"type": "content_block_start", "index": index,
                               "content_block": {"type": "tool_use", "id": call["id"],
                                                 "name": call["name"], "input": {}}},
                              "content_block_start")
                    self._sse({"type": "content_block_delta", "index": index,
                               "delta": {"type": "input_json_delta",
                                         "partial_json": call["arguments"]}},
                              "content_block_delta")
                    self._sse({"type": "content_block_stop", "index": index},
                              "content_block_stop")
                    index += 1
                stop = _anthropic_stop(result)
                self._sse({"type": "message_delta", "delta": {"stop_reason": stop,
                           "stop_sequence": None}, "usage": anthropic_usage(service, result.stats)},
                          "message_delta")
                self._sse({"type": "message_stop"}, "message_stop")
            except (BrokenPipeError, ConnectionResetError, ClientGone):
                pass
            except Exception as exc:
                self._sse_anthropic_error(exc)
            return
        result = service.complete(messages, tools, self._limit(payload), gone=self._client_gone)
        content = []
        if result.text:
            content.append({"type": "text", "text": result.text})
        for call in result.tool_calls:
            try:
                arguments = json.loads(call["arguments"])
            except ValueError:
                arguments = {"raw": call["arguments"]}
            content.append({"type": "tool_use", "id": call["id"],
                            "name": call["name"], "input": arguments})
        self._json({
            "id": ident, "type": "message", "role": "assistant",
            "model": service.model, "content": content,
            "stop_reason": _anthropic_stop(result),
            "stop_sequence": None,
            "usage": anthropic_usage(service, result.stats),
        })

    def _responses(self, payload):
        service = self.server.service
        messages = _responses_messages(payload)
        tools = payload.get("tools") or []
        ident = f"resp_{uuid.uuid4().hex}"
        if payload.get("stream"):
            max_tokens = self._limit(payload)
            self._sse_start()
            try:
                base = {"id": ident, "object": "response", "created_at": int(time.time()),
                        "status": "in_progress", "error": None,
                        "incomplete_details": None, "model": service.model, "output": []}
                sequence = [0]

                def event(kind, **values):
                    sequence[0] += 1
                    self._sse({"type": kind, "sequence_number": sequence[0], **values}, kind)

                event("response.created", response=base)
                event("response.in_progress", response=base)
                item_id = f"msg_{uuid.uuid4().hex}"
                opened = [False]

                def on_text(text):
                    if text is None:
                        self._heartbeat()
                        return
                    if not opened[0]:
                        item = {"id": item_id, "type": "message", "status": "in_progress",
                                "role": "assistant", "content": []}
                        event("response.output_item.added", output_index=0, item=item)
                        event("response.content_part.added", item_id=item_id,
                              output_index=0, content_index=0,
                              part={"type": "output_text", "text": "", "annotations": []})
                        opened[0] = True
                    event("response.output_text.delta", item_id=item_id,
                          output_index=0, content_index=0, delta=text)

                result = self._complete_with_keepalive(service, messages, tools, max_tokens, on_text)
                output = []
                if result.text:
                    message = {"id": item_id, "type": "message", "status": "completed",
                               "role": "assistant", "content": [{"type": "output_text",
                               "text": result.text, "annotations": []}]}
                    output.append(message)
                    event("response.output_text.done", item_id=item_id, output_index=0,
                          content_index=0, text=result.text)
                    event("response.content_part.done", item_id=item_id, output_index=0,
                          content_index=0, part=message["content"][0])
                    event("response.output_item.done", output_index=0, item=message)
                for index, call in enumerate(result.tool_calls, start=len(output)):
                    item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex}",
                            "call_id": call["id"], "name": call["name"],
                            "arguments": call["arguments"], "status": "completed"}
                    output.append(item)
                    pending = dict(item, arguments="", status="in_progress")
                    event("response.output_item.added", output_index=index, item=pending)
                    event("response.function_call_arguments.delta", item_id=item["id"],
                          output_index=index, delta=call["arguments"])
                    event("response.function_call_arguments.done", item_id=item["id"],
                          output_index=index, arguments=call["arguments"])
                    event("response.output_item.done", output_index=index, item=item)
                completed = dict(base, status="completed", output=output,
                                 usage=responses_usage(service, result.stats))
                event("response.completed", response=completed)
            except (BrokenPipeError, ConnectionResetError, ClientGone):
                pass
            except Exception as exc:
                self._sse_responses_error(exc, ident)
            return
        result = service.complete(messages, tools, self._limit(payload), gone=self._client_gone)
        output = []
        if result.text:
            output.append({"id": f"msg_{uuid.uuid4().hex}", "type": "message",
                           "status": "completed", "role": "assistant",
                           "content": [{"type": "output_text", "text": result.text,
                                        "annotations": []}]})
        for call in result.tool_calls:
            output.append({"type": "function_call", "id": f"fc_{uuid.uuid4().hex}",
                           "call_id": call["id"], "name": call["name"],
                           "arguments": call["arguments"], "status": "completed"})
        self._json({
            "id": ident, "object": "response", "created_at": int(time.time()),
            "status": "completed", "model": service.model, "output": output,
            "usage": responses_usage(service, result.stats),
        })


def serve(session, host: str, port: int, api_key: str | None = None,
          allowed_origins=(), context_limit=None, max_cold_prefill=None,
          max_incremental_prefill=None, scheduler=None) -> None:
    """Reset the model when the server starts and when it stops.

    HTTP is served from a background thread; this thread runs the inference jobs, so everything bound to the
    inference thread stays on it.
    """
    session.reset()
    server = MacqwenHTTPServer(
        (host, port), ModelService(session, context_limit, max_cold_prefill, max_incremental_prefill), api_key, allowed_origins,
        scheduler)
    recorder = flight_recorder.install()
    if recorder.enabled:
        print(f"flight recorder: {recorder.path}")
    flight_recorder.record("server_start", host=host, port=server.server_port, context_limit=context_limit,
                           max_cold_prefill=max_cold_prefill or MAX_COLD_PREFILL_DEFAULT,
                           max_incremental_prefill=max_incremental_prefill or MAX_INCREMENTAL_PREFILL_DEFAULT)
    try:
        from models.flashnext import memory_controller as _mc
        _mc.register_worker_probe(lambda: server.scheduler.busy)
    except ImportError:
        pass
    print(f"server ready at http://{host}:{server.server_port}")
    if allowed_origins:
        print(f"browser origins allowed: {', '.join(allowed_origins)}")
    if context_limit:
        print(f"context limit: {context_limit} tokens (longer prompts are refused, never truncated)")
    if max_cold_prefill:
        print(f"memory tier: single prefills up to {max_cold_prefill} tokens (larger ones are refused, never truncated)")
    print(f"one request runs at a time; up to {server.scheduler.max_waiting} wait at most "
          f"{server.scheduler.max_wait:g}s, then get 503 server_busy")
    print("OpenAI: /v1/chat/completions and /v1/responses")
    print("Anthropic: /v1/messages")
    print("Status: /health, /ready, /v1/models")
    stop = threading.Event()
    listener = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                                name="http-listener", daemon=True)
    listener.start()
    try:
        server.scheduler.serve_jobs(stop)
    except KeyboardInterrupt:
        print("\nserver stopped")
    finally:
        stop.set()
        flight_recorder.record("server_stop")
        server.shutdown()
        server.server_close()
        session.reset()
        flight_recorder.uninstall()
