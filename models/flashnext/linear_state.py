"""Release each GatedDeltaNet layer's prefill inputs as the layer finishes (opt-in).

A GatedDeltaNet layer keeps a small convolution state, ``cache[0]``, which is a
slice of that layer's ``conv_input`` (``tokens x 10,240`` bf16, about 20 KB per
token). The state is lazy: nothing the layer returns depends on it, so a
prefill that evaluates only the layer outputs (the streaming MoE does exactly
that, once per layer) leaves the state unevaluated. Its inputs stay alive
until something evaluates it, which in the full-logit path is the first decode
step. Measured on a GatedDeltaNet layer with the checkpoint's shapes and random
weights: 42 to 47 KB per prompt token held per layer until the state is
evaluated, then 3.3 MB (the real state size) once it is.

Across the 36 linear-attention layers that is a prefill allocation growing with
the prompt, on the order of 1 MB per token: about 1 GB at 1K tokens, and
several GB at 8K to 10K. Evaluating ``cache[0]`` right after the layer frees
the inputs while the prefill is still running.

The operations are unchanged and only the moment they run moves, so every
value, state and output is bit-identical. ``FLASHNEXT_EVAL_LINEAR_STATE=1``
turns it on. Off by default. Decode calls (one token) are never touched.
"""
from __future__ import annotations

import os

import mlx.core as mx

FLAG = "FLASHNEXT_EVAL_LINEAR_STATE"
ENABLED = [os.environ.get(FLAG, "0") == "1"]
MIN_TOKENS = 64            # below this the inputs are small and decode must stay lean
_APPLIED = [False]
_ORIGINAL = {}


def set_enabled(value: bool) -> None:
    ENABLED[0] = bool(value)


def _evaluate_state(cache) -> None:
    """Evaluate the convolution state if the cache has one. Returns nothing."""
    try:
        state = cache[0]
    except (TypeError, IndexError, KeyError):
        return
    if state is not None:
        mx.eval(state)


def call(self, inputs, mask=None, cache=None):
    output = _ORIGINAL["call"](self, inputs, mask=mask, cache=cache)
    if ENABLED[0] and cache is not None and int(inputs.shape[1]) >= MIN_TOKENS:
        _evaluate_state(cache)
    return output


def apply() -> bool:
    """Wrap the Qwen4 GatedDeltaNet call once (flag read at call time)."""
    if _APPLIED[0]:
        return False
    from mlx_vlm.models.qwen4_exp import language

    _ORIGINAL["call"] = language.Qwen4ExpGatedDeltaNet.__call__
    language.Qwen4ExpGatedDeltaNet.__call__ = call
    _APPLIED[0] = True
    return True


def restore() -> None:
    """Undo ``apply`` (tests)."""
    if not _APPLIED[0]:
        return
    from mlx_vlm.models.qwen4_exp import language

    language.Qwen4ExpGatedDeltaNet.__call__ = _ORIGINAL["call"]
    _APPLIED[0] = False
