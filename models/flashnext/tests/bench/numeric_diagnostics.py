"""Numerical diagnostics for comparing two prefill plans (offline tool, no production change).

Chunked and unchunked prefill are not bit-identical, because kernel shapes depend on row counts. This module
measures HOW different two executions are instead of accepting "bf16 drift" on faith:

* final-token logits: largest difference among the top candidates against the top-token margin, so it says
  whether the argmax could have flipped;
* expert routing: per layer and token, the router's top-(k+1) probabilities, how many tokens changed their
  selected expert set, and how close to a tie each changed token was in the reference run;
* state: per-layer relative error of attention KV, indexer keys, recurrent state and convolution state,
  with exact comparison of position tensors, and the KV error per token position (a hand-off bug shows as a
  localized jump, drift as a smooth profile);
* generation: greedy tokens, first divergence, and the top-1 versus top-2 logit margin at each step.

Everything here only reads values. The recorders wrap callables and return their results unchanged.

    from models.flashnext.tests.bench import numeric_diagnostics as N
    ref = N.execute(language, ids, chunk_tokens=0, decode_tokens=16)
    other = N.execute(language, ids, chunk_tokens=2048, policy="balanced", anchors=(2046,), decode_tokens=16)
    report = N.compare(ref, other)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from veystrix.sampling import Sampler, Sampling


# -- recorders -------------------------------------------------------------------------------

class MarginSampler(Sampler):
    """A Sampler that also records the top-1 and top-2 logits and ids of every row it is asked to sample.

    Sampling itself is the parent's, unchanged (the row is only read). ``records`` holds
    ``(top1_id, top2_id, top1_logit, top2_logit)`` per call, in call order.
    """

    def __init__(self, settings=None, sink=None):
        super().__init__(settings)
        self.records = sink if sink is not None else []

    def __call__(self, logits):
        row = logits.reshape(-1).astype(mx.float32)
        order = mx.argpartition(-row, kth=1)[:2]
        values = row[order]
        mx.eval(order, values)
        ids, vals = order.tolist(), values.tolist()
        if vals[0] < vals[1]:
            ids, vals = ids[::-1], vals[::-1]
        self.records.append((ids[0], ids[1], vals[0], vals[1]))
        return super().__call__(logits)


@contextmanager
def margin_sampler(module, sink):
    """Make ``module.Sampler`` (for example ``veystrix.backends.flashnext``) record margins into ``sink``."""
    original = module.Sampler

    class Recording(MarginSampler):
        def __init__(self, settings=None):
            super().__init__(settings, sink)

    module.Sampler = Recording
    try:
        yield sink
    finally:
        module.Sampler = original


class _Tap(nn.Module):
    """Stands in for a router ``gate`` module: same output, plus a record of the top-(k+1) probabilities."""

    def __init__(self, inner, recorder, layer, k):
        super().__init__()
        object.__setattr__(self, "_inner", inner)        # not a child module: parameters() are untouched
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "_k", k)

    def __call__(self, x):
        logits = self._inner(x)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        order = mx.argsort(-probs, axis=-1)[..., : self._k + 1]
        top = mx.take_along_axis(probs, order, axis=-1)
        mx.eval(order, top)
        self._recorder.add(self._layer, np.array(order).reshape(-1, self._k + 1),
                           np.array(top).reshape(-1, self._k + 1))
        return logits


class RoutingRecorder:
    """Records, per layer, the router's top-(k+1) experts and probabilities for every token routed.

    ``stage`` labels what is being run ("prefill" or "decode"). Chunks of a prefill append in order, so the
    prefill arrays have one row per prompt token.
    """

    def __init__(self):
        self.stage = "prefill"
        self.data: dict = {}          # (stage, layer) -> [index arrays], [probability arrays]
        self._installed = []

    def add(self, layer, indices, probs):
        slot = self.data.setdefault((self.stage, layer), ([], []))
        slot[0].append(indices)
        slot[1].append(probs)

    def get(self, stage, layer):
        slot = self.data.get((stage, layer))
        if slot is None:
            return None, None
        return np.concatenate(slot[0]), np.concatenate(slot[1])

    def layers(self, stage="prefill"):
        return sorted(layer for s, layer in self.data if s == stage)

    @contextmanager
    def install(self, language):
        for index, layer in enumerate(language.layers):
            block = getattr(layer, "mlp", None)
            if block is None or not hasattr(block, "gate") or not hasattr(block, "top_k"):
                continue
            original = block["gate"]
            block["gate"] = _Tap(original, self, index, int(block.top_k))
            self._installed.append((block, original))
        try:
            yield self
        finally:
            for block, original in self._installed:
                block["gate"] = original
            self._installed.clear()


# -- execution ---------------------------------------------------------------------------------

@dataclass
class Execution:
    label: str
    n_prompt: int
    chunk_spans: list
    state: dict                        # name -> numpy array (KV trimmed to the cache offset)
    prefill_margin: tuple              # (top1 id, top2 id, top1 logit, top2 logit) of the seeding row
    prefill_top: list                  # top-5 (id, logit) of the seeding row
    tokens: list                       # greedy tokens: the seeded one, then the decoded ones
    margins: list                      # one (id1, id2, logit1, logit2) per token
    routing: RoutingRecorder | None = None
    top_logits: dict = field(default_factory=dict)


def state_arrays(cache, language=None) -> dict:
    """Every cache array (KV cut to the cache offset) plus position state, as numpy arrays."""
    from models.flashnext import prefix_cache as P

    out = {}
    for index, entry in enumerate(cache):
        if P.kind_of(entry) == "arrays":
            for slot, value in enumerate(entry.cache):
                if value is not None:
                    out[f"l{index}.s{slot}"] = value
        else:
            n = int(entry.offset)
            out[f"l{index}.keys"] = entry.keys[..., :n, :]
            out[f"l{index}.values"] = entry.values[..., :n, :]
            out[f"l{index}.index_keys"] = entry.index_keys
            out[f"l{index}.index_position_ids"] = entry.index_position_ids
    if language is not None:
        for name, attr in P.LANG_FIELDS:
            value = getattr(language, attr, None)
            if value is not None:
                out[f"lang.{name}"] = value
    mx.eval(list(out.values()))
    return {name: np.array(value.astype(mx.float32)) if value.dtype == mx.bfloat16 else np.array(value)
            for name, value in out.items()}


def execute(language, ids, *, chunk_tokens=0, policy="grid", anchors=(), decode_tokens=0,
            record_routing=True, label="") -> Execution:
    """Prefill ``ids`` under a chunk plan, then decode greedily, recording everything above.

    ``anchors`` are boundaries that must be chunk edges (and are reported as boundaries). Decoding uses the
    plain language-model call, one token at a time on the same cache.
    """
    from models.flashnext import prefill

    env = {prefill.FLAG_CHUNK_TOKENS: str(chunk_tokens), prefill.FLAG_CHUNK_POLICY: policy}
    sampler = MarginSampler(Sampling.greedy_settings())
    recorder = RoutingRecorder() if record_routing else None
    spans = []
    real_chunk, real_target = prefill._prefill_chunk, prefill._prefill_target

    def chunk_spy(lang, chunk_ids, cache):
        spans.append(int(chunk_ids.shape[1]))
        return real_chunk(lang, chunk_ids, cache)

    def target_spy(lang, chunk_ids, cache, *args, **kwargs):
        spans.append(int(chunk_ids.shape[1]))
        return real_target(lang, chunk_ids, cache, *args, **kwargs)

    cache = language.make_cache()
    ids = [int(t) for t in ids]
    language._position_ids = None
    language._rope_deltas = None
    context = recorder.install(language) if recorder else _null()
    with mock.patch.dict(os.environ, env), mock.patch.object(prefill, "_prefill_chunk", chunk_spy), \
            mock.patch.object(prefill, "_prefill_target", target_spy), context:
        _, _, token = prefill.prefill_target(
            language, mx.array([ids]), cache, sampler=sampler, boundaries=tuple(anchors), anchors=tuple(anchors))
        mx.eval(token)
        tokens = [int(token.item())]
        if recorder:
            recorder.stage = "decode"
        for _ in range(decode_tokens):
            out = language(mx.array([[tokens[-1]]]), cache=cache)
            logits = out.logits[:, -1, :]
            mx.eval(logits)
            step = sampler(logits)
            mx.eval(step)
            tokens.append(int(step.item()))
    state = state_arrays(cache, language)
    margins = list(sampler.records)
    top = {"prefill_row": margins[0]}
    return Execution(label=label, n_prompt=len(ids), chunk_spans=spans, state=state,
                     prefill_margin=margins[0], prefill_top=[], tokens=tokens, margins=margins,
                     routing=recorder, top_logits=top)


@contextmanager
def _null():
    yield None


# -- comparison --------------------------------------------------------------------------------

def tensor_error(a: np.ndarray, b: np.ndarray) -> dict:
    """Error of ``b`` against ``a``. Relative terms use the tensor's own scale so near-zero values do not
    inflate them."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"mismatch": f"{a.shape} {a.dtype} vs {b.shape} {b.dtype}"}
    if a.dtype.kind in "iu":
        return {"identical": bool(np.array_equal(a, b)), "max_abs": float(np.max(np.abs(a.astype(np.int64) - b))) if a.size else 0.0,
                "rel_l2": 0.0 if np.array_equal(a, b) else float("inf")}
    x, y = a.astype(np.float64), b.astype(np.float64)
    diff = x - y
    norm = np.linalg.norm(x)
    return {"identical": bool(np.array_equal(a, b)), "max_abs": float(np.max(np.abs(diff))) if a.size else 0.0,
            "rel_l2": float(np.linalg.norm(diff) / norm) if norm else 0.0}


def position_profile(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Relative L2 error per token position along ``axis``."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    other = tuple(i for i in range(x.ndim) if i != axis % x.ndim)
    num = np.sqrt(np.sum((x - y) ** 2, axis=other))
    den = np.sqrt(np.sum(x ** 2, axis=other))
    return num / np.maximum(den, 1e-12)


def _kind(name: str) -> str:
    if name.startswith("lang."):
        return "position"
    suffix = name.split(".", 1)[1]
    return {"keys": "kv", "values": "kv", "index_keys": "indexer", "index_position_ids": "position",
            "s0": "conv", "s1": "recurrent"}.get(suffix, "other")


def compare_logits(ref: Execution, other: Execution) -> dict:
    id1, id2, l1, l2 = ref.prefill_margin
    o1, o2, ol1, ol2 = other.prefill_margin
    return {
        "top1_same": id1 == o1, "ref_margin": l1 - l2, "other_margin": ol1 - ol2,
        "top1_logit_diff": abs(l1 - ol1), "top2_logit_diff": abs(l2 - ol2) if id2 == o2 else None,
        # The argmax could only have changed if the perturbation reached half the margin.
        "decision_safe": (l1 - l2) > 2.0 * max(abs(l1 - ol1), abs(l2 - ol2) if id2 == o2 else 0.0),
    }


def compare_routing(ref: Execution, other: Execution, stage="prefill") -> dict:
    """Per layer: tokens whose selected expert SET changed, and how near a tie each was (reference run)."""
    if ref.routing is None or other.routing is None:
        return {}
    layers = {}
    all_gap_flip, all_gap_keep = [], []
    for layer in sorted(set(ref.routing.layers(stage)) & set(other.routing.layers(stage))):
        ri, rp = ref.routing.get(stage, layer)
        oi, _ = other.routing.get(stage, layer)
        n = min(len(ri), len(oi))
        k = ri.shape[1] - 1
        ref_sets = np.sort(ri[:n, :k], axis=1)
        other_sets = np.sort(oi[:n, :k], axis=1)
        changed = np.any(ref_sets != other_sets, axis=1)
        gap = (rp[:n, k - 1] - rp[:n, k]) / np.maximum(rp[:n, k - 1], 1e-30)      # k-th versus (k+1)-th, relative
        layers[layer] = {"tokens": int(n), "flips": int(changed.sum()),
                         "median_flip_gap": float(np.median(gap[changed])) if changed.any() else None,
                         "median_other_gap": float(np.median(gap[~changed])) if (~changed).any() else None}
        all_gap_flip.extend(gap[changed].tolist())
        all_gap_keep.extend(gap[~changed].tolist())
    tokens = sum(v["tokens"] for v in layers.values())
    flips = sum(v["flips"] for v in layers.values())
    return {"layers": layers, "tokens_routed": tokens, "flips": flips,
            "flip_fraction": flips / tokens if tokens else 0.0,
            "median_flip_gap": float(np.median(all_gap_flip)) if all_gap_flip else None,
            "median_kept_gap": float(np.median(all_gap_keep)) if all_gap_keep else None,
            "max_flip_gap": float(np.max(all_gap_flip)) if all_gap_flip else None}


def compare_state(ref: Execution, other: Execution) -> dict:
    rows = {}
    for name in sorted(set(ref.state) & set(other.state)):
        a, b = ref.state[name], other.state[name]
        if name == "lang.position_ids" and a.ndim == b.ndim and a.shape[:-1] == b.shape[:-1]:
            # The language model keeps the position ids its FIRST call computed and reads them only at cache
            # offset 0; a chunked run's first call is shorter. Compare what both hold.
            n = min(a.shape[-1], b.shape[-1])
            a, b = a[..., :n], b[..., :n]
        rows[name] = {**tensor_error(a, b), "kind": _kind(name)}
    missing = sorted(set(ref.state) ^ set(other.state))
    by_kind = {}
    for name, row in rows.items():
        item = by_kind.setdefault(row["kind"], {"tensors": 0, "identical": 0, "worst_rel_l2": 0.0, "worst_max_abs": 0.0})
        item["tensors"] += 1
        item["identical"] += bool(row.get("identical"))
        if "rel_l2" in row and np.isfinite(row["rel_l2"]):
            item["worst_rel_l2"] = max(item["worst_rel_l2"], row["rel_l2"])
        item["worst_max_abs"] = max(item["worst_max_abs"], row.get("max_abs", 0.0))
    positions_exact = all(row.get("identical") for row in rows.values() if row["kind"] == "position")
    return {"tensors": rows, "missing": missing, "by_kind": by_kind, "positions_exact": positions_exact,
            "all_identical": not missing and all(row.get("identical") for row in rows.values())}


def kv_profile(ref: Execution, other: Execution, layer: int) -> np.ndarray:
    return position_profile(ref.state[f"l{layer}.keys"], other.state[f"l{layer}.keys"], axis=2)


BF16_NOISE = 1e-3          # per-position relative L2 of ordinary bf16 rounding differences (bf16 epsilon is 3.9e-3)
FP32_NOISE = 1e-6          # about five times fp32 chunk-versus-whole drift (2e-7 to 5e-7)


def noise_floor(array: np.ndarray) -> float:
    """The smallest per-position relative error that counts as a difference for this tensor's precision.
    bf16 state is stored as float32 here, but its low 16 bits are all zero, which tells them apart."""
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)
    return BF16_NOISE if flat.size and not np.any(flat & 0xFFFF) else FP32_NOISE


def boundary_spikes(ref: Execution, other: Execution, window: int = 8) -> dict:
    """Does the KV error jump right after a chunk edge of ``other``?

    Per full-attention layer and internal edge: the mean per-position error over the ``window`` positions
    starting at the edge, divided by that layer's median per-position error (floored at the precision's noise). Drift is smooth, so the ratio
    stays near or below one; a state or conv hand-off defect at the edge gives a large ratio. This is the
    discriminating test, because a defect confined to a few positions can leave the aggregate error, the
    logits and even the generated tokens almost unchanged.
    """
    edges, total = [], 0
    for width in other.chunk_spans[:-1]:
        total += width
        edges.append(total)
    worst = {"ratio": 0.0, "layer": None, "edge": None}
    rows = {}
    layers = sorted(int(name.split(".")[0][1:]) for name in other.state
                    if name.endswith(".keys") and name in ref.state)
    skipped = []
    for layer in layers:
        if ref.state[f"l{layer}.keys"].shape != other.state[f"l{layer}.keys"].shape:
            skipped.append(layer)                           # reported as a shape mismatch by compare_state
            continue
        profile = kv_profile(ref, other, layer)
        # The floor keeps a layer whose error is exactly zero over most positions (the causal prefix of an
        # early edge, or a tiny model) from turning ordinary rounding at the edge into a huge ratio.
        median = max(float(np.median(profile)), noise_floor(ref.state[f"l{layer}.keys"]))
        for edge in edges:
            if edge + window > len(profile):
                continue
            ratio = float(profile[edge:edge + window].mean() / median)
            rows[(layer, edge)] = ratio
            if ratio > worst["ratio"]:
                worst = {"ratio": ratio, "layer": layer, "edge": edge}
    return {"edges": edges, "worst": worst, "ratios": rows, "skipped_layers": skipped}


def compare_generation(ref: Execution, other: Execution) -> dict:
    n = min(len(ref.tokens), len(other.tokens))
    divergence = next((i for i in range(n) if ref.tokens[i] != other.tokens[i]), None)
    row = {"ref_tokens": len(ref.tokens), "other_tokens": len(other.tokens), "identical": ref.tokens == other.tokens,
           "first_divergence": divergence}
    if divergence is not None:
        r, o = ref.margins[divergence], other.margins[divergence]
        row.update({"ref_margin_at_divergence": r[2] - r[3], "other_margin_at_divergence": o[2] - o[3],
                    "diverged_to_second_choice": other.tokens[divergence] == r[1] and ref.tokens[divergence] == r[0]})
    return row


def compare(ref: Execution, other: Execution) -> dict:
    """One report: logits, routing, state and generation."""
    state = compare_state(ref, other)
    return {"ref": ref.label, "other": other.label, "chunks": {"ref": ref.chunk_spans, "other": other.chunk_spans},
            "logits": compare_logits(ref, other), "routing": compare_routing(ref, other),
            "state": state, "generation": compare_generation(ref, other),
            "boundaries": boundary_spikes(ref, other)}


def summary(report: dict) -> dict:
    """The numbers that decide whether a difference looks like drift, without the per-layer tables."""
    state = report["state"]
    routing = report["routing"]
    return {
        "bit_identical": bool(state["all_identical"] and report["generation"]["identical"]),
        "positions_exact": state["positions_exact"],
        "state_worst_rel_l2_by_kind": {k: v["worst_rel_l2"] for k, v in state["by_kind"].items()},
        "routing_flip_fraction": routing.get("flip_fraction"),
        "flips_are_near_ties": (routing.get("median_flip_gap") is not None and routing.get("median_kept_gap") is not None
                                and routing["median_flip_gap"] < routing["median_kept_gap"]) if routing.get("flips") else None,
        "logits_decision_safe": report["logits"]["decision_safe"],
        "max_boundary_spike_ratio": report["boundaries"]["worst"]["ratio"],
        "shape_mismatches": [n for n, row in state["tensors"].items() if "mismatch" in row],
        "tokens_identical": report["generation"]["identical"],
        "first_divergence": report["generation"]["first_divergence"],
    }
