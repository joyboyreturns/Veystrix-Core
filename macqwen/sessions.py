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


def derive_key(messages, explicit: str | None = None) -> str:
    """Stable id of the conversation these (normalized) messages belong to."""
    if explicit:
        return "id:" + explicit[:64]
    system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    first_user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    digest = hashlib.sha256(json.dumps([system, first_user], default=str, ensure_ascii=False).encode()).hexdigest()
    return "auto:" + digest[:16]


def explicit_id(headers=None, payload=None) -> str | None:
    """The client's own session id, if it sent one."""
    value = None
    if headers is not None:
        value = headers.get("X-Veystrix-Session") or headers.get("x-veystrix-session")
    if not value and isinstance(payload, dict):
        meta = payload.get("metadata")
        if isinstance(meta, dict):
            value = meta.get("session_id") or meta.get("conversation_id")
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
