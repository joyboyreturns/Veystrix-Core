"""Which conversation the single live model state belongs to, and what switching between conversations cost.

The model holds ONE live conversation state (a KV cache plus recurrent state that cannot be rewound). Several clients, or one
client with side requests (a title, a summary, a subagent), can therefore interleave requests from different conversations.
This module does not hold model state and cannot corrupt it. It only names conversations and counts what happens so the cost
is visible and the behaviour is explicit:

* a request is served whatever its length or shape; nothing is refused for being short or tool-less;
* a request that does not extend the live conversation REPLACES it: the next request of the replaced conversation is rebuilt
  from its full exact token sequence (the prefix cache supplies any stored prefix). Isolation comes from that exact-token
  reconstruction, not from keeping state: a conversation can never see another conversation's cache;
* each switch is counted, and the replaced conversation is marked cold until it is rebuilt.

A conversation is identified by an explicit id when the client sends one (``X-Veystrix-Session`` header, or
``metadata.session_id`` / ``metadata.conversation_id`` in the body), otherwise by a digest of its first system message and its
first user message, which are identical on every turn of one conversation and differ between unrelated ones.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict


KNOWN_SESSION_HEADERS = (
    "x-cache-id", "X-Cache-Id", "X-Cache-ID",
    "x-session-id", "X-Session-Id",
    "X-Veystrix-Session", "x-veystrix-session",
    "session-id", "Session-Id",
    # pi-ai affinity headers. DeepSeek Harness 0.2.2-rc.1 does NOT let a profile enable them on an openai-completions route
    # (sendSessionAffinityHeaders is "withhold" in its compat gate), so a DSH conversation is identified by transcript
    # digest (system + first user message + tool names), not by header. Kept for clients that do send them.
    "session_id", "x-session-affinity",
)


def derive_key(messages, explicit: str | None = None, tools: list | None = None, namespace: str = "default") -> str:
    """Stable id of the conversation these (normalized) messages belong to.

    Combines:
    - Namespace isolation
    - Explicit cache/session hint (where provided)
    - Deterministic transcript digest (system prompt + first user message + tool schemas)
    """
    if explicit:
        return f"id:{namespace}:{explicit[:64]}" if namespace != "default" else f"id:{explicit[:64]}"
    system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    first_user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    tool_keys = sorted(t.get("function", {}).get("name", "") for t in (tools or [])) if tools else []
    digest = hashlib.sha256(
        json.dumps([namespace, system, first_user, tool_keys], default=str, ensure_ascii=False).encode()
    ).hexdigest()
    return f"auto:{digest[:16]}"


def explicit_id(headers=None, payload=None) -> str | None:
    """The client's own session id or cache id, if it sent one."""
    value = None
    if headers is not None:
        for h in KNOWN_SESSION_HEADERS:
            value = headers.get(h)
            if value:
                break
    if not value and isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict):
            value = meta.get("cache_id") or meta.get("session_id") or meta.get("conversation_id")
        if not value:
            value = payload.get("cache_id") or payload.get("session_id") or payload.get("conversation_id")
    return str(value).strip()[:64] if value and str(value).strip() else None


class SessionRegistry:
    def __init__(self, capacity: int = 16, clock=time.time):
        self.capacity, self._clock = capacity, clock
        self.items: OrderedDict[str, dict] = OrderedDict()
        self.live: str | None = None
        self.switches = 0

    def activate(self, key: str, *, prompt_tokens: int, reused: bool, had_state: bool) -> dict:
        """Record that ``key`` now owns the live state after a successful adoption."""
        now = self._clock()
        entry = self.items.pop(key, None) or {"id": key, "first_seen": now, "requests": 0, "rebuilds": 0, "cold": False}
        if self.live is not None and self.live != key and had_state:
            self.switches += 1
            replaced = self.items.get(self.live)
            if replaced is not None:
                replaced["cold"] = True
                replaced["replaced_by"] = key
        entry["requests"] += 1
        entry["rebuilds"] += 0 if reused else 1
        entry["cold"] = False
        entry["last_seen"], entry["last_prompt_tokens"] = now, prompt_tokens
        self.items[key] = entry
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        self.live = key
        return entry

    def dropped(self) -> None:
        """The live state was discarded (a failed or cancelled request)."""
        if self.live is not None and self.live in self.items:
            self.items[self.live]["cold"] = True
        self.live = None

    def snapshot(self) -> dict:
        return {"live": self.live, "switches": self.switches,
                "known": [{k: v for k, v in e.items() if k in ("id", "requests", "rebuilds", "cold", "last_prompt_tokens")}
                          for e in reversed(self.items.values())][:8]}


def _norm_tool_args(arg):
    if isinstance(arg, str):
        try:
            return json.loads(arg) if arg.strip() else {}
        except (ValueError, TypeError):
            return arg.strip()
    return arg if arg is not None else {}


def tool_calls_match(tc1, tc2) -> bool:
    """Check if two sets of tool calls match in function name and arguments."""
    c1 = tc1 or []
    c2 = tc2 or []
    if len(c1) != len(c2):
        return False
    for a, b in zip(c1, c2):
        fa = a.get("function", a) if isinstance(a, dict) else {}
        fb = b.get("function", b) if isinstance(b, dict) else {}
        if fa.get("name") != fb.get("name"):
            return False
        if _norm_tool_args(fa.get("arguments", "")) != _norm_tool_args(fb.get("arguments", "")):
            return False
    return True


class ReasoningLedger:
    """Retains server-generated exact reasoning text per turn and per session.

    When clients (such as older harnesses or web clients) send history without
    reasoning_content, the ledger matches visible assistant messages against
    prior generations and unambiguously reconstructs reasoning_content so the
    template renders identical token IDs and preserves the KV cache.
    """

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.restored_turns_total: int = 0
        # maps session_key -> list of recorded assistant turns:
        # [{"content": str, "tool_calls": list, "reasoning_content": str, "time": float}]
        self.sessions: OrderedDict[str, list[dict]] = OrderedDict()

    def record_turn(self, key: str, content: str, tool_calls: list, reasoning_content: str, clock=time.time) -> None:
        """Record the generation outcome of an assistant turn."""
        if not key or not reasoning_content:
            return
        turns = self.sessions.pop(key, [])
        turns.append({
            "content": content or "",
            "tool_calls": list(tool_calls or []),
            "reasoning_content": reasoning_content,
            "time": clock(),
        })
        self.sessions[key] = turns
        while len(self.sessions) > self.capacity:
            self.sessions.popitem(last=False)

    def restore_reasoning(self, key: str, messages: list[dict]) -> int:
        """Match assistant messages in the request against the ledger for this session.

        Reconstructs missing or empty reasoning_content in-place on messages.
        Returns the count of restored assistant turns.
        """
        if not key or key not in self.sessions:
            return 0
        recorded_turns = self.sessions[key]
        if not recorded_turns:
            return 0

        # Find all assistant messages in the incoming request
        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]
        if not assistant_indices:
            return 0

        # If incoming assistant messages exceed recorded turns, match is ambiguous
        if len(assistant_indices) > len(recorded_turns):
            return 0

        # Verify exact sequence match: each assistant turn must match recorded turn
        matches = []
        for turn_idx, msg_idx in enumerate(assistant_indices):
            msg = messages[msg_idx]
            rec = recorded_turns[turn_idx]
            # Match content
            msg_content = msg.get("content") or ""
            rec_content = rec.get("content") or ""
            if msg_content != rec_content:
                return 0
            # Match tool calls
            msg_calls = msg.get("tool_calls") or []
            rec_calls = rec.get("tool_calls") or []
            if not tool_calls_match(msg_calls, rec_calls):
                return 0
            matches.append((msg_idx, rec["reasoning_content"]))

        # All assistant messages match unambiguously! Populate or validate reasoning_content
        restored = 0
        for msg_idx, reasoning_content in matches:
            msg = messages[msg_idx]
            existing = msg.get("reasoning_content")
            if not existing:
                msg["reasoning_content"] = reasoning_content
                restored += 1
            elif isinstance(existing, str) and existing != reasoning_content:
                # Client passed reasoning (e.g. DSH tool-call turn or client round-trip).
                # Standardize to exact recorded reasoning so template renders bit-identically.
                if existing.strip() == reasoning_content.strip() or reasoning_content.startswith(existing.strip()):
                    msg["reasoning_content"] = reasoning_content
                    restored += 1

        # If the incoming conversation branched or rolled back (fewer assistant turns than recorded),
        # trim recorded turns to match the current conversation depth
        if len(assistant_indices) < len(recorded_turns):
            self.sessions[key] = recorded_turns[:len(assistant_indices)]

        if restored:
            self.restored_turns_total += restored
        return restored

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self.sessions.clear()
        else:
            self.sessions.pop(key, None)
