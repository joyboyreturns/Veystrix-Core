"""Layer-major (windowed) out-of-core prefill. Opt-in: ``FLASHNEXT_PREFILL_ORDER=layer``; the default ``chunk`` is unchanged.

Why. A MoE prompt chunk of 2,048 tokens routes about 400 of a layer's 512 experts, so every chunk re-reads about 60 GB of the
checkpoint (docs/flashnext/architecture-overhaul.md has the arithmetic). The chunk-major loop

    chunk 1 -> layers 0..47, chunk 2 -> layers 0..47, ...

therefore pays the whole SSD pass once per chunk, and its per-layer router sync leaves the GPU idle while the reads run. Here a
*window* of several chunks runs layer-major,

    layer 0 -> chunks 1..n, layer 1 -> chunks 1..n, ...

so a layer's experts are read ONCE per window and reused by every chunk of the window. The window is bounded (default 8,192
tokens): the activation it keeps resident between layers is ``tokens x hc_count x hidden`` bf16 (20 KB per token, 164 MB at
8K), and a window end is a safe cut: every layer's cache is at the same position, so a snapshot or an interruption keeps
everything before it.

Exactness. Layer ``l``'s output for chunk ``c`` depends only on layer ``l-1``'s output for chunk ``c`` and layer ``l``'s own
cache after chunk ``c-1``. Both orders honour exactly those dependencies, with the same chunk shapes, masks and position ids, so
the per-layer arithmetic is the same operations on the same inputs. The one difference is the expert operand: the window
hands ``gather_qmm`` all of the layer's experts and uses the router's indices directly, where the chunk-major path hands it
only that chunk's experts and remapped indices. ``gather_qmm`` computes each (token, expert) product from that expert's own
rows, so the result does not depend on which other experts sit in the operand; tests check this bit for bit on a small model, and
the real checkpoint must pass the token-digest gate before it is promoted. Until it does, the cache fingerprint carries
``identity()`` so layer-major snapshots never mix with chunk-major ones.

What it does not do. A layer-major window cannot snapshot at a chunk edge inside it (the later layers have not reached it), so
the cadence of resumable cuts is the window, not the chunk; an interruption loses at most one window of work. Windows still end
at the system-prompt boundary and prefer message anchors. Adaptive chunk re-planning runs between windows only.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import mlx.core as mx

FLAG_ORDER = "FLASHNEXT_PREFILL_ORDER"                  # "chunk" (default) | "layer"
FLAG_WINDOW = "FLASHNEXT_PREFILL_WINDOW_TOKENS"         # tokens per layer-major window
FLAG_HYBRID = "FLASHNEXT_PREFILL_HYBRID_SMALL_WINDOWS"  # opt-in: reference path for small windows
FLAG_PREFETCH = "FLASHNEXT_PREFILL_LAYER_PREFETCH"      # opt-in: read layer l+1's experts while layer l computes
FLAG_TAIL_CHUNK = "FLASHNEXT_PREFILL_TAIL_CHUNK"        # "1" (default): a trailing tiny chunk runs chunk-major after the last window
ORDERS = ("chunk", "layer")
PLAN_VERSION = "layer-major-v1"
DEFAULT_WINDOW_TOKENS = 8192
MIN_WINDOW_TOKENS = 1024
MAX_WINDOW_TOKENS = 32768
HYBRID_SMALL_WINDOW_TOKENS = 2048
# A chunk of at most this many tokens takes the decode-style MoE path (``expert_cache`` uses the window only above 8 rows),
# which touches the 2 GB decode pool as a Metal buffer for the first time: measured +1.9 GB of MLX memory and +0.4 s at layer 1,
# held for the rest of the window (results/flashnext/20261110-005029-prefill-profile-manual). It is run after the window instead.
TINY_CHUNK_TOKENS = 8


def order() -> str:
    value = (os.environ.get(FLAG_ORDER, "") or "chunk").strip().lower()
    if value not in ORDERS:
        raise ValueError(f"{FLAG_ORDER} must be one of {', '.join(ORDERS)}")
    return value


def enabled() -> bool:
    return order() == "layer"


def window_tokens() -> int:
    raw = os.environ.get(FLAG_WINDOW, "") or str(DEFAULT_WINDOW_TOKENS)
    try:
        value = int(float(raw))
    except ValueError as error:
        raise ValueError(f"{FLAG_WINDOW} must be a number of tokens") from error
    if not MIN_WINDOW_TOKENS <= value <= MAX_WINDOW_TOKENS:
        raise ValueError(f"{FLAG_WINDOW} must be between {MIN_WINDOW_TOKENS} and {MAX_WINDOW_TOKENS}")
    return value


def identity() -> str:
    """The execution-plan identity for the cache fingerprint: empty for chunk-major (existing keys unchanged)."""
    if not enabled():
        return ""
    hybrid = "/hybrid-small-v1" if hybrid_enabled() else ""
    return f"{PLAN_VERSION}/w{window_tokens()}{hybrid}"


def prefetch_enabled() -> bool:
    value = os.environ.get(FLAG_PREFETCH, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_PREFETCH} must be 0 or 1")
    return value == "1"


def tail_chunk_enabled() -> bool:
    value = os.environ.get(FLAG_TAIL_CHUNK, "1") or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_TAIL_CHUNK} must be 0 or 1")
    return value == "1"


def peel_tail(group):
    """Split a final window into (chunks run layer-major, trailing tiny chunk or None).

    Only a window of more than one chunk whose last chunk has at most ``TINY_CHUNK_TOKENS`` tokens is split. The tail then runs
    through the reference chunk-major path after every layer of the window has finished, so it sees exactly the cache state the
    reference gives it and takes the same executor: results are unchanged."""
    group = list(group)
    if len(group) > 1 and tail_chunk_enabled() and group[-1][1] - group[-1][0] <= TINY_CHUNK_TOKENS:
        return group[:-1], group[-1]
    return group, None


def hybrid_enabled() -> bool:
    value = os.environ.get(FLAG_HYBRID, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_HYBRID} must be 0 or 1")
    return value == "1"


# ---------------------------------------------------------------------------------------------------------------- planning

def plan_windows(spans, offset: int, window: int, required=(), anchors=()):
    """Group chunk ``spans`` (relative ``(start, end)`` pairs covering the suffix) into windows of whole chunks.

    A window holds at most ``window`` tokens unless a single chunk is larger. It ends at a ``required`` position (the
    system-prompt boundary) when one falls inside it, otherwise at the last anchor in its second half, otherwise at the last
    chunk edge that fits. Returns a list of span lists; positions are absolute only for ``required`` and ``anchors``.
    """
    spans = [(int(a), int(b)) for a, b in spans]
    if not spans:
        return []
    required = sorted({int(p) for p in required})
    anchors = sorted({int(p) for p in anchors})
    windows, index = [], 0
    while index < len(spans):
        first = spans[index][0]
        limit = first + int(window)
        stop = index
        while stop + 1 < len(spans) and spans[stop + 1][1] <= limit:
            stop += 1                                   # spans[index..stop] fit (the first chunk always does)
        forced = next((k for k in range(index, stop + 1) if offset + spans[k][1] in required), None)
        if forced is not None:
            stop = forced
        elif stop + 1 < len(spans):
            half = first + int(window) // 2
            nice = [k for k in range(index, stop + 1) if offset + spans[k][1] in anchors and spans[k][1] > half]
            if nice:
                stop = nice[-1]
        windows.append(spans[index:stop + 1])
        index = stop + 1
    return windows


# ---------------------------------------------------------------------------------------------------------- cost model

def expert_pass_bytes(experts: int, layers: int, expert_bytes: int) -> int:
    """Bytes of one full pass over every expert of ``layers`` layers."""
    return int(experts) * int(layers) * int(expert_bytes)


def chunk_major_bytes(tokens: int, chunk: int, layers: int, experts_read_per_layer: float, expert_bytes: int) -> int:
    """Physical expert bytes of the chunk-major order: every chunk reads its routed experts in every layer."""
    chunks = -(-int(tokens) // int(chunk))
    return int(chunks * layers * experts_read_per_layer * expert_bytes)


def layer_major_bytes(tokens: int, window: int, layers: int, experts: int, expert_bytes: int) -> int:
    """Physical expert bytes of the layer-major order: every window reads each layer's full expert set once."""
    windows = -(-int(tokens) // int(window))
    return windows * expert_pass_bytes(experts, layers, expert_bytes)


def activation_bytes(tokens: int, hc_count: int, hidden: int, dtype_bytes: int = 2) -> int:
    """Resident hidden states of one window between layers."""
    return int(tokens) * int(hc_count) * int(hidden) * int(dtype_bytes)


def predicted_seconds(tokens: int, compute_tok_s: float, io_bytes: int, io_gb_s: float, overlapped: bool) -> float:
    """Wall time when compute and SSD reads run at the given rates, serialized or perfectly overlapped."""
    compute = tokens / compute_tok_s
    io = io_bytes / (io_gb_s * 1e9)
    return max(compute, io) if overlapped else compute + io


# ----------------------------------------------------------------------------------------------------- expert window

class ExpertWindow:
    """Supplies one layer's complete expert set to every chunk of a window (see ``expert_cache._LAYER_WINDOW``).

    ``ring`` is the prefill ring (``prefill_ring.PrefillRing``) or None. With a ring the layer's experts are read into one
    half with direct I/O and handed out as zero-copy views; without one they are read with the layer's own path (a copy per
    layer). Either way ``end_layer`` must be called only after ``mx.eval`` has returned for every graph that used them: the
    ring's hazard contract (a half is not recycled while the GPU may read it) depends on it.
    """

    def __init__(self, ring=None):
        self.ring = ring
        self._layer = None
        self._weights = None
        self._prefetched = {}                   # layer_id -> (projections, pending reads); at most two layers ahead
        self.stats = {"layers": 0, "bytes": 0, "ring_layers": 0, "fallback_layers": 0, "read_seconds": 0.0,
                      "prefetch_layers": 0, "prefetch_used": 0}
        self.layer_reads = []                   # trace only: [layer_id, read seconds, MLX active MB before, after]

    def prefetch(self, glu) -> bool:
        """Queue the next layer's complete expert read on the worker threads without waiting for it (opt-in).

        Fallback path only: each layer's destination is a fresh array, so the reads can never write into memory the layer
        being computed still uses; with a ring (one half) or the pool-reuse path this does nothing. One layer ahead at most:
        about one more layer of operands (1.5 GB at 512 experts) is resident while it overlaps. The bytes read and their order
        are those of the blocking read, so results are identical; only when the wait happens changes. At the start of layer
        ``l`` the window holds layer ``l`` (queued earlier, consumed by its MoE call) and layer ``l+1`` (queued now): never
        more than two, never the same layer twice.
        """
        if (glu is None or self.ring is not None or not prefetch_enabled() or getattr(glu, "layer_id", -1) < 0
                or glu.layer_id in self._prefetched or len(self._prefetched) >= 2):
            return False
        from models.flashnext import expert_cache, expert_io_stats

        pool = getattr(glu.gate_proj.cache.store, "_flashnext_expert_pool", None)
        if expert_io_stats.pool_reuse_enabled() and pool is not None and pool._slot_of:
            return False                        # the pool-reuse read path owns this layer
        projections = (glu.gate_proj, glu.up_proj, glu.down_proj)
        experts = list(range(glu.gate_proj.num_experts))
        self._prefetched[glu.layer_id] = (projections, expert_cache.submit_projection_tasks(projections, experts))
        self.stats["prefetch_layers"] += 1
        return True

    def drain(self) -> None:
        """Wait for any queued prefetch to stop writing, then drop it. Safe to call at any time and on every exit path."""
        held, self._prefetched = self._prefetched, {}
        if not held:
            return
        from models.flashnext import expert_cache

        def futures(item):
            if hasattr(item, "futures"):
                yield from item.futures
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    yield from futures(sub)
            else:
                yield item

        for future in futures([pending for _, pending in held.values()]):
            try:
                expert_cache._resolve_future(future, None)
            except Exception as error:
                # A consumed layer's failure is raised by the real wait. This one belongs to a layer that was never consumed
                # (the window ended or aborted first): quiescence is all that matters, but it is counted and logged.
                self.stats["prefetch_drain_errors"] = self.stats.get("prefetch_drain_errors", 0) + 1
                logging.getLogger("flashnext.layer_major").warning("orphaned layer prefetch failed: %r", error)

    def weights_for(self, glu):
        layer = glu.layer_id
        if self._layer == layer and self._weights is not None:
            return self._weights
        began = time.perf_counter()
        try:
            active_before = round(mx.get_active_memory() / 1e6)
        except Exception:
            active_before = None
        experts = list(range(glu.gate_proj.num_experts))
        weights = None
        if layer in self._prefetched:
            from models.flashnext import expert_cache

            projections, pending = self._prefetched.pop(layer)
            raw = expert_cache._await_projection_tasks(pending)
            weights = [projection.cache.to_mx(chunks) for projection, chunks in zip(projections, raw)]
            self.stats["fallback_layers"] += 1
            self.stats["prefetch_used"] += 1
        elif self.ring is not None:
            weights = self.ring.stage(glu.gate_proj.cache.store, glu.gate_proj.cache.prefix.rsplit(".", 1)[0], experts)
            if weights is not None:
                self.stats["ring_layers"] += 1
        if weights is None:
            weights = glu._read_weights((glu.gate_proj, glu.up_proj, glu.down_proj), experts)
            self.stats["fallback_layers"] += 1
        self._layer, self._weights = layer, weights
        self.stats["layers"] += 1
        self.stats["read_seconds"] += time.perf_counter() - began
        try:
            self.layer_reads.append([layer, round(time.perf_counter() - began, 3), active_before,
                                     round(mx.get_active_memory() / 1e6)])
        except Exception:
            pass
        try:
            store = glu.gate_proj.cache.store
            prefix = glu.gate_proj.cache.prefix.rsplit(".", 1)[0]
            self.stats["bytes"] += len(experts) * sum(
                store.refs[f"{prefix}.{p}.{k}"].row_bytes for p in ("gate_proj", "up_proj", "down_proj")
                for k in ("weight", "scales", "biases"))
        except Exception:
            pass
        return weights

    def end_layer(self) -> None:
        """Drop this layer's operands. Call after the GPU work that used them has completed."""
        self._layer = self._weights = None
        if self.ring is not None:
            self.ring._retire()

    def abort_layer(self) -> None:
        """Fence before releasing a layer whose evaluation did not complete.

        A failed dispatch or evaluation may still have GPU work in flight.
        Keep the operands alive until the ring's normal abort fence succeeds;
        the surrounding ring session retains its pool if fencing fails.
        """
        self.drain()                            # no read may still be writing when the operands are released
        if self.ring is not None:
            self.ring.abort()
        self._layer = self._weights = None


# --------------------------------------------------------------------------------------------------------------- driver

def _position_ids(language, inputs, offset: int, cache):
    """The position ids ``LanguageModel.__call__`` builds for a text-only chunk that starts at ``offset``.

    The first chunk (``offset == 0``) goes through ``get_rope_index`` exactly as the model does and records the same state;
    later chunks add the cache offset to the stored rope deltas.
    """
    batch, length = inputs.shape
    if offset == 0 or language._rope_deltas is None:
        position_ids, deltas = language.get_rope_index(inputs, None, None, None)
        language._rope_deltas = mx.zeros((batch, 1), dtype=deltas.dtype)
        language._position_ids = position_ids
        return position_ids
    delta = mx.array(offset + language._rope_deltas)
    if delta.ndim == 0:
        delta = mx.expand_dims(delta, axis=0)
    delta = mx.tile(delta, (batch, 1)) if delta.shape[0] < batch else delta[:batch]
    position_ids = mx.broadcast_to(mx.arange(length).reshape(1, -1), (batch, length))
    position_ids = mx.add(position_ids, delta)
    if language._position_ids is not None and language._position_ids.ndim == 3:
        position_ids = mx.broadcast_to(position_ids[None, ...], (3, batch, length))
    return position_ids


@dataclass
class WindowReport:
    tokens: int = 0
    chunks: int = 0
    seconds: float = 0.0
    layer_seconds: list = field(default_factory=list)
    layer_reads: list = field(default_factory=list)         # [layer, read s, active MB before, after] per streamed layer
    chunk_seconds: list = field(default_factory=list)       # per layer: seconds of each chunk's evaluation (trace only)
    chunk_memory_mb: list = field(default_factory=list)     # per layer: MLX active MB after each chunk's evaluation (trace only)
    layer_memory_mb: list = field(default_factory=list)     # [active, cache, peak] MLX MB at each layer end (trace only)
    expert_bytes: int = 0
    read_seconds: float = 0.0
    ring_layers: int = 0
    fallback_layers: int = 0
    prefetch_layers: int = 0


def _streamed_glu(layer):
    from models.flashnext import expert_cache

    glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
    return glu if isinstance(glu, expert_cache.StreamingSwitchGLU) else None


def run_window(language, ids, cache, chunk_spans, offset: int, window=None, checkpoint=None, after_layer=None):
    """Advance every layer's cache over ``ids`` (shape ``(1, T)``) layer-major and return the final hidden states.

    ``chunk_spans`` are relative ``(start, end)`` chunks covering ``ids``; ``offset`` is how many tokens the cache already
    holds. Returns ``(chunk_outputs, report)`` where ``chunk_outputs`` is the list of per-chunk hidden states after the last
    layer and the head's hyper-connection mixer is NOT applied (the caller applies it to the chunk it needs).
    ``window`` is an ``ExpertWindow`` or None (None leaves the layers' own expert path in charge, which is what the
    chunk-major order uses). ``checkpoint(layer)`` runs at each layer end after the layer has been evaluated (a safe point).
    """
    from models.flashnext import expert_cache

    body = language.model
    layers = body.layers
    hc = body.args.hc_count
    count = len(layers)
    report = WindowReport(tokens=int(ids.shape[1]), chunks=len(chunk_spans))
    # ExpertWindow is reused across windows; report only this invocation's
    # traffic and timings rather than its lifetime totals.
    stats_before = dict(window.stats) if window is not None else None
    reads_before = len(window.layer_reads) if window is not None and hasattr(window, "layer_reads") else 0
    began = time.perf_counter()
    inputs = [ids[:, a:b] for a, b in chunk_spans]
    states = []
    positions = []
    for (a, b), chunk in zip(chunk_spans, inputs):
        states.append(mx.tile(body.embed_tokens(chunk), (1, 1, hc)))
        positions.append(_position_ids(language, chunk, offset + a, cache))
    mx.eval(states)
    fa_masks = [None] * len(chunk_spans)
    ssm_masks = [None] * len(chunk_spans)
    from mlx_vlm.models.qwen4_exp import language as q4

    previous = expert_cache._LAYER_WINDOW[0]
    expert_cache._LAYER_WINDOW[0] = window
    try:
        for index, layer in enumerate(layers):
            layer_began = time.perf_counter()
            prefetch = getattr(window, "prefetch", None)
            if prefetch is not None and index + 1 < count:
                prefetch(_streamed_glu(layers[index + 1]))      # overlaps this layer's compute (opt-in, no-op otherwise)
            chunk_times = []
            chunk_memory = []
            for c, chunk in enumerate(inputs):
                chunk_began = time.perf_counter()
                if index == body.fa_idx:
                    # The model builds the mask from the first full-attention cache before any layer runs, i.e. at this
                    # chunk's start offset; the cache is at exactly that offset here too.
                    fa_masks[c] = q4._create_qwen3_5_attention_mask(states[c], cache[body.fa_idx])
                if index == body.ssm_idx:
                    ssm_masks[c] = q4._create_qwen3_5_ssm_mask(states[c], cache[body.ssm_idx])
                mask = ssm_masks[c] if layer.is_linear else fa_masks[c]
                states[c] = layer(states[c], chunk, mask=mask, cache=cache[index], position_ids=positions[c])
                # One evaluation per (layer, chunk): bounds the lazy graph and keeps the cache arrays concrete.
                mx.eval(states[c], cache[index].state)
                chunk_times.append(round(time.perf_counter() - chunk_began, 3))
                try:
                    chunk_memory.append(round(mx.get_active_memory() / 1e6))
                except Exception:
                    chunk_memory.append(None)
            report.chunk_seconds.append(chunk_times)
            report.chunk_memory_mb.append(chunk_memory)
            if window is not None:
                window.end_layer()
            if os.environ.get("FLASHNEXT_PREFILL_CLEAR_CACHE", "1") != "0":
                mx.clear_cache()
            report.layer_seconds.append(round(time.perf_counter() - layer_began, 3))
            try:
                report.layer_memory_mb.append([round(mx.get_active_memory() / 1e6), round(mx.get_cache_memory() / 1e6),
                                               round(mx.get_peak_memory() / 1e6)])
            except Exception:
                pass
            if checkpoint is not None:
                checkpoint(index)
            if after_layer is not None:
                after_layer(index, count)
    except BaseException:
        if window is not None:
            window.abort_layer()
        raise
    finally:
        expert_cache._LAYER_WINDOW[0] = previous
        drain = getattr(window, "drain", None)
        if drain is not None:
            drain()
    report.seconds = time.perf_counter() - began
    if window is not None:
        report.expert_bytes = window.stats["bytes"] - stats_before["bytes"]
        report.read_seconds = window.stats["read_seconds"] - stats_before["read_seconds"]
        report.ring_layers = window.stats["ring_layers"] - stats_before["ring_layers"]
        report.fallback_layers = window.stats["fallback_layers"] - stats_before["fallback_layers"]
        report.prefetch_layers = window.stats.get("prefetch_used", 0) - stats_before.get("prefetch_used", 0)
        report.layer_reads = list(getattr(window, "layer_reads", [])[reads_before:])
    return states, report
