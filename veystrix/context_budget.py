"""Context budgeting that keeps five different numbers apart.

* advertised client window: what a client believes the model accepts (DeepSeek Harness ``contextWindow``)
* server window: what this server really accepts (``--context-limit``, lowered by the admitted memory tier)
* input tokens: the real tokenizer's count of the rendered prompt (the client only estimates it)
* reserved output: tokens kept for the answer (``max_completion_tokens`` after clamping)
* continuation headroom: room left for later tool results and turns of the same conversation

DeepSeek Harness (pi-ai ``clampMaxTokensToContext``) asks for
``min(maxTokens, contextWindow - estimated_input - CLIENT_SAFETY_TOKENS)`` and floors it at 1, with
``CLIENT_SAFETY_TOKENS = 4096`` hard-coded. A window equal to the server window therefore makes the client ask for one
token near the limit, which cannot hold a tool call. The advertised window must be the server window PLUS that safety
margin; ``suggested_client_window`` computes it and the server publishes it on ``/v1/models``.

Pure functions, no model and no I/O, so the rules are unit-testable and the numbers in docs are generated, not typed.
"""
from __future__ import annotations

from dataclasses import dataclass

CLIENT_SAFETY_TOKENS = 4096        # DeepSeek Harness, measured in docs/flashnext/dsh/veystrix-agentic.patch.yml
MIN_OUTPUT_RESERVE = 256           # the server refuses a prompt that leaves less than this for the answer
MIN_TOOL_OUTPUT = 64               # a tool-using request asking for less cannot hold even a short call
CONTINUATION_WARN_TOKENS = 2048    # below this much room, the next tool result may not fit


def suggested_client_window(server_window: int, safety: int = CLIENT_SAFETY_TOKENS) -> int:
    """The ``contextWindow`` to configure in the client so that its own reserve is on top of the server's window."""
    return int(server_window) + int(safety)


def client_max_completion(advertised_window: int, estimated_input: int, max_tokens: int,
                          safety: int = CLIENT_SAFETY_TOKENS) -> int:
    """What DeepSeek Harness would request (for documentation and tests)."""
    return max(1, min(int(max_tokens), int(advertised_window) - int(estimated_input) - int(safety)))


@dataclass(frozen=True)
class Budget:
    server_window: int
    input_tokens: int
    requested_output: int
    reserved_output: int
    remaining: int                  # server_window - input_tokens - reserved_output
    continuation_ok: bool           # remaining >= CONTINUATION_WARN_TOKENS

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def evaluate(server_window: int | None, input_tokens: int, requested_output: int | None) -> Budget:
    """Budget of one request. ``server_window`` None means no enforced limit (nothing is claimed)."""
    req = -1 if requested_output is None or int(requested_output) < 0 else int(requested_output)
    if not server_window:
        return Budget(0, int(input_tokens), req, req, 0, True)
    available = max(0, int(server_window) - int(input_tokens))
    if req < 0:
        reserved = max(1, available)
        remaining = max(0, int(server_window) - int(input_tokens) - reserved)
        return Budget(int(server_window), int(input_tokens), req, reserved, remaining,
                      available >= CONTINUATION_WARN_TOKENS)
    reserved = max(1, min(req, available))
    remaining = int(server_window) - int(input_tokens) - reserved
    return Budget(int(server_window), int(input_tokens), req, reserved, remaining,
                  remaining >= CONTINUATION_WARN_TOKENS)


def too_small_for_tools(max_tokens: int | None, has_tools: bool) -> str | None:
    """An error message when a tool-using request asks for an output cap that cannot hold a tool call."""
    if max_tokens is None or int(max_tokens) < 0:
        return None
    if has_tools and int(max_tokens) < MIN_TOOL_OUTPUT:
        return (f"max_completion_tokens={max_tokens} cannot hold a tool call (at least {MIN_TOOL_OUTPUT} are needed). "
                "The client computed it from a context window that leaves no room for the answer. Set the client's "
                "context window to this server's context_window plus the client's own safety margin "
                f"(DeepSeek Harness: {CLIENT_SAFETY_TOKENS}); GET /v1/models returns it as "
                "suggested_client_context_window.")
    return None


def overflow_guidance(input_tokens: int, server_window: int) -> str:
    """What to do when the prompt does not fit. Nothing is truncated by the server."""
    over = int(input_tokens) + MIN_OUTPUT_RESERVE - int(server_window)
    return (f"the prompt is {input_tokens} tokens and the server window is {server_window}; it is {over} tokens too long "
            f"once {MIN_OUTPUT_RESERVE} are reserved for the answer. The server never truncates a system prompt, tool "
            "schema or tool result. Compact on the client: drop or summarise old tool results first, then old turns; "
            "or disable unused tools (their schemas are in every request); or restart with a larger --context-limit "
            "(sizes above the validated ones are staged, see docs/flashnext/long-context-agentic-audit-2026-11-08.md).")


# What has really been executed on the 16 GB MacBook Air M4 with the real 113 GB checkpoint, kept apart from what the
# checkpoint could do (native) and what the server is configured to accept (limit). Update only from preserved evidence.
VALIDATION = {
    "largest_real_cold_prefill_tokens": 3974,
    "evidence": "Current recovery: streamed reply after 3974-token cold prefill; see docs/flashnext/CODEX_RUNTIME_RECOVERY_PROGRESS.md",
    "real_8k": "Current recovery: 7974-token cold prefill cancelled after 3984 tokens under sustained memory pressure",
    "above_validated": "Unverified in current recovery; larger historical reports require independent evidence review",
}


def native_context_window(model_path) -> int | None:
    """``max_position_embeddings`` of the checkpoint's text config, or None when it cannot be read."""
    import json
    import os

    try:
        with open(os.path.join(str(model_path), "config.json")) as handle:
            config = json.load(handle)
        value = (config.get("text_config") or config).get("max_position_embeddings")
        return int(value) if isinstance(value, int) and value > 0 else None
    except (OSError, ValueError, TypeError):
        return None
