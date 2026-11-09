"""Where the reasoning-effort sentence goes in the prompt, and why it is not where the checkpoint's template puts it.

The Qwen3.8-Flash-Next chat template writes the effort sentence as the FIRST text of the system message, before the tool
schemas and the client's system prompt. Everything after it depends on it, so ``low`` and ``xhigh`` shared nothing with
``medium`` (and ``high`` diverged right after the tools): every effort switch re-prefilled the whole ~6-10K-token
prefix cold, about 5 minutes on this machine.

``late`` (the default) renders the prompt exactly as ``medium`` and then inserts the effort sentence as its own short system
turn immediately AFTER the system message:

    <|im_start|>system \n tools block + client system prompt <|im_end|> \n
    <|im_start|>system \n Reasoning effort is set to low. ... <|im_end|> \n          <- only for low / high / xhigh
    <|im_start|>user ...

* ``medium`` and thinking-off have no extra turn: their tokens are identical to what they always were, so every cache entry
  written before this change stays valid, and the state stored at the end of the system message (an anchor of the chunk
  plan in every cache mode) is exactly the shared prefix of every effort.
* ``low``, ``high``, ``xhigh`` (and ``max``, clamped to ``xhigh``) add the turn. The tools and the whole system prompt before
  it are the same tokens for every effort, so a switch re-prefills only the turn and what follows it.

Cost of the trade: this is not the layout the checkpoint was trained with (the same sentence, read after the system prompt
instead of before it, as a second system turn). ``VEYSTRIX_EFFORT_PLACEMENT=template`` restores the checkpoint's own layout
exactly. Switching the effort in the middle of a conversation still re-prefills the history after the system message
(attention state depends on every earlier token); the system prompt itself is not recomputed.
"""
from __future__ import annotations

import os

from macqwen.conversation import EXTRA_REASONING

PLACEMENT_ENV = "VEYSTRIX_EFFORT_PLACEMENT"

# Verbatim from the checkpoint's chat_template.jinja (a test compares them with the template when the file is present).
TEMPLATE_SENTENCES = {
    "xhigh": ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
              "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."),
    "low": ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
            "unnecessary elaboration."),
}
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def placement() -> str:
    """``late`` (default) or ``template``. Anything unrecognised means ``late``."""
    return "template" if os.environ.get(PLACEMENT_ENV, "late").strip().lower() == "template" else "late"


def sentence(effort: str, thinking: bool) -> str:
    """The effort sentence for a request, or "" (medium, or thinking off, which has no effort sentence)."""
    if not thinking:
        return ""
    return TEMPLATE_SENTENCES.get(effort) or EXTRA_REASONING.get(effort) or ""


def block_text(effort: str, thinking: bool) -> str:
    """The whole inserted system turn, or "" when the request has none."""
    text = sentence(effort, thinking)
    return f"{IM_START}system\n{text}{IM_END}\n" if text else ""
