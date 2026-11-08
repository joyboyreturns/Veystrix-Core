"""Per-request reasoning controls: what a client asked for, what the checkpoint can do, and what was applied.

Wire contracts accepted (all optional, all applied to THIS request only; nothing is stored on the session or the server):

* ``reasoning: {"effort": "low|medium|high|xhigh|max|none"}``      Qwen Code 0.22.0 (``/effort``), measured on the wire
* ``reasoning_effort: "..."``                                       OpenAI flat form
* ``chat_template_kwargs: {"enable_thinking": bool}``                the checkpoint's own chat-template switch
* ``enable_thinking: bool``                                          the same switch, flat (DashScope style)
* ``reasoning: {"enabled": false}`` / ``reasoning: false``          off

What the checkpoint really supports (Qwen3.8-Flash-Next chat template, ``enable_thinking`` and ``reasoning_effort``):
thinking ON or OFF, and three effort names in the template: ``xhigh`` (default), ``medium``, ``low``. Each only changes one
sentence of the system prompt (xhigh: "think carefully ... validate key assumptions"; low: "keep your thinking brief";
medium: nothing) and, with it, the model's prompt-following behaviour. There is NO decoder-level thinking budget. This module
therefore implements four distinct, honest levels and says so:

    low -> template ``low``;  medium -> template ``medium``;  xhigh -> template ``xhigh``;
    high -> template ``medium`` plus the repository's own stopping-rule sentence (``conversation.EXTRA_REASONING``);
    max -> ``xhigh`` (Qwen Code already clamps ``max`` to ``xhigh`` for generic OpenAI endpoints; the server clamps it too and
           reports it, so there is no fifth level);  none/off/false -> thinking off (``enable_thinking=False``).

Conflicts are never silent: an explicit disable beats an effort in the same request and is reported in ``notes``; an unknown
effort name is a 400, not an ignored field. No field at all means the server default (``preferences``: thinking off, effort
medium), reported as ``source: server-default``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LEVELS = ("low", "medium", "high", "xhigh")
SUPPORTED_NAMES = (*LEVELS, "max", "none")
OFF_NAMES = ("none", "off", "disabled", "false", "0")
ALIASES = {"minimal": "low", "med": "medium", "extra-high": "xhigh", "extrahigh": "xhigh", "x-high": "xhigh"}
CLAMPS = {"max": ("xhigh", "max has no distinct level on this checkpoint (the template knows xhigh, medium and low); using xhigh"),
          "minimal": ("low", "minimal has no distinct level; using low")}
TEMPLATE_KWARGS = {"enable_thinking"}
WIRE_FIELDS = ("reasoning", "reasoning_effort", "chat_template_kwargs", "enable_thinking")


class ReasoningError(ValueError):
    """A reasoning control the server cannot honour; the caller turns it into a 400."""


@dataclass(frozen=True)
class Reasoning:
    thinking: bool                    # effective enable_thinking for the chat template
    effort: str                       # effective level in LEVELS (used only when thinking is on)
    requested: str | None             # the client's effort string after normalisation, or None
    source: str                       # which wire field decided it, or "server-default"
    wire: dict = field(default_factory=dict)    # the raw reasoning-related fields as received
    notes: tuple = ()

    def as_dict(self) -> dict:
        return {"thinking": self.thinking, "effort": self.effort if self.thinking else None, "requested": self.requested,
                "source": self.source, "wire": self.wire, "notes": list(self.notes)}

    def header(self) -> str:
        level = f"effort={self.effort}" if self.thinking else "effort=n/a"
        return f"thinking={'on' if self.thinking else 'off'} {level} source={self.source}"


def _name(value, where: str) -> str:
    if isinstance(value, bool):
        return "none" if value is False else "medium"
    if not isinstance(value, str) or not value.strip():
        raise ReasoningError(f"{where} must be one of {', '.join(SUPPORTED_NAMES)}")
    key = value.strip().lower()
    return ALIASES.get(key, key)


def _check(name: str, where: str) -> str:
    if name in OFF_NAMES:
        return "none"
    if name not in SUPPORTED_NAMES:
        raise ReasoningError(f"{where}={name!r} is not supported by this checkpoint; use one of "
                             f"{', '.join(SUPPORTED_NAMES)} (low, medium, high and xhigh are distinct; max is clamped to xhigh)")
    return name


def resolve(payload: dict, default_thinking: bool, default_effort: str) -> Reasoning:
    """Resolve the controls of one chat request. Raises ``ReasoningError`` for anything it cannot apply."""
    wire = {key: payload[key] for key in WIRE_FIELDS if payload.get(key) is not None}
    notes: list[str] = []
    requested = None
    source = "server-default"
    thinking = bool(default_thinking)
    effort = default_effort if default_effort in LEVELS else "medium"

    reasoning = payload.get("reasoning")
    effort_name = None
    enabled = None
    if isinstance(reasoning, dict):
        unknown = sorted(set(reasoning) - {"effort", "enabled", "exclude", "max_tokens"})
        if unknown:
            raise ReasoningError(f"reasoning.{unknown[0]} is not supported by this server")
        if reasoning.get("max_tokens") is not None:
            raise ReasoningError("reasoning.max_tokens (a thinking budget) is not supported: the checkpoint has no "
                                 "decoder-level thinking budget; use reasoning.effort")
        if reasoning.get("effort") is not None:
            effort_name = _check(_name(reasoning["effort"], "reasoning.effort"), "reasoning.effort")
            source = "reasoning.effort"
        if reasoning.get("enabled") is not None:
            enabled = bool(reasoning["enabled"])
            source = source if effort_name else "reasoning.enabled"
    elif isinstance(reasoning, (str, bool)):
        effort_name = _check(_name(reasoning, "reasoning"), "reasoning")
        source = "reasoning"
    elif reasoning is not None:
        raise ReasoningError("reasoning must be an object, a string or a boolean")
    if payload.get("reasoning_effort") is not None:
        flat = _check(_name(payload["reasoning_effort"], "reasoning_effort"), "reasoning_effort")
        if effort_name is not None and flat != effort_name:
            notes.append(f"reasoning_effort={flat!r} differs from reasoning.effort={effort_name!r}; the nested form was used")
        else:
            effort_name = flat if effort_name is None else effort_name
            source = "reasoning_effort" if source == "server-default" else source

    kwargs = payload.get("chat_template_kwargs")
    explicit_thinking = None
    if kwargs is not None:
        if not isinstance(kwargs, dict):
            raise ReasoningError("chat_template_kwargs must be an object")
        unknown = sorted(set(kwargs) - TEMPLATE_KWARGS)
        if unknown:
            raise ReasoningError(f"chat_template_kwargs.{unknown[0]} is not supported by this server (only "
                                 f"{', '.join(sorted(TEMPLATE_KWARGS))}); it was not applied")
        if kwargs.get("enable_thinking") is not None:
            explicit_thinking = bool(kwargs["enable_thinking"])
            source = "chat_template_kwargs.enable_thinking"
    if payload.get("enable_thinking") is not None:
        flat_thinking = bool(payload["enable_thinking"])
        if explicit_thinking is not None and flat_thinking != explicit_thinking:
            notes.append("enable_thinking differs from chat_template_kwargs.enable_thinking; the template kwarg was used")
        elif explicit_thinking is None:
            explicit_thinking = flat_thinking
            source = "enable_thinking" if source == "server-default" else source

    if effort_name in CLAMPS or effort_name == "max":
        target, text = CLAMPS.get(effort_name, CLAMPS["max"])
        notes.append(text)
        effort_name = target
    requested = effort_name

    off = (effort_name == "none") or (enabled is False) or (explicit_thinking is False)
    if off:
        if effort_name not in (None, "none") and (explicit_thinking is False or enabled is False):
            notes.append(f"effort {effort_name!r} was requested together with an explicit thinking disable; thinking is off")
        thinking = False
        if source == "server-default":
            source = "reasoning.effort"
    elif effort_name is not None or enabled is True or explicit_thinking is True:
        thinking = True
        if effort_name is not None:
            effort = effort_name
    return Reasoning(thinking=thinking, effort=effort, requested=requested, source=source, wire=wire, notes=tuple(notes))
