"""Flash-Next prompt ingestion without the interactive chat."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import mlx.core as mx

from models.flashnext import memory_controller
from models.flashnext.qsa_chunk import QSA_CHUNK_THRESHOLD


PREFILL_FULL_LOGITS_MAX_TOKENS = int(
    os.environ.get("FLASHNEXT_PREFILL_FULL_LOGITS_MAX_TOKENS", "8192")
)
PREFILL_RELEASE_BYTES = int(
    os.environ.get("FLASHNEXT_PREFILL_RELEASE_BYTES", "256000000")
)
PREFILL_CLEAR_CACHE = os.environ.get("FLASHNEXT_PREFILL_CLEAR_CACHE", "1") != "0"
# Opt-in: split one long prefill into internal chunks that share the cache. 0 = off (one call).
# Chunking changes kernel shapes, so outputs are numerically equivalent but NOT bit-identical to
# the single call; validate with a digest before relying on it.
FLAG_CHUNK_TOKENS = "FLASHNEXT_PREFILL_CHUNK_TOKENS"
MIN_CHUNK_TOKENS = 64
# Largest accepted chunk. A chunk's transient memory grows with its length (about 0.27 MB per token plus
# attention scores), so the opt-in is bounded; see long_context_budget.py.
MAX_CHUNK_TOKENS = 2048
# FLASHNEXT_PREFILL_CHUNK_POLICY: how a prompt is cut into chunks of at most the chunk size.
#   "grid" (default)  edges at every multiple of the chunk size in absolute position plus the requested
#                     boundaries; the original strategy, unchanged.
#   "balanced"        each segment between boundaries is cut into the fewest equal pieces of at most the
#                     chunk size, so a boundary near a grid line costs no extra tiny chunk.
FLAG_CHUNK_POLICY = "FLASHNEXT_PREFILL_CHUNK_POLICY"
#   "budgeted"        a deterministic, position-aware plan: each chunk is the largest of CANDIDATE_CHUNKS (up to the
#                     chunk size) whose estimated extra MLX memory at that position fits FLASHNEXT_PREFILL_CHUNK_BUDGET_MB.
#                     Chunks shrink as the cached context grows because attention scores scale with chunk x length.
#                     It depends only on the prompt end, the anchors and the budget, never on live memory, so cold and
#                     restored runs share it (and the budget is part of the cache fingerprint).
CHUNK_POLICIES = ("grid", "balanced", "budgeted")
FLAG_CHUNK_BUDGET_MB = "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB"
DEFAULT_CHUNK_BUDGET_MB = 1000
CANDIDATE_CHUNKS = (2048, 1024, 512, 256)
# Short prompts keep full-sequence logits although the decoder reads only the
# last row. Projecting only the last row changed those logits by up to 0.125
# (2026-09-22), so it is not exact and was removed.


def chunk_tokens() -> int:
    """Requested chunk size from the environment (0 = off). Bad values raise."""
    raw = os.environ.get(FLAG_CHUNK_TOKENS, "0") or "0"
    try:
        value = int(float(raw))
    except ValueError as error:
        raise ValueError(f"{FLAG_CHUNK_TOKENS} must be a number of tokens") from error
    if value < 0 or (0 < value < MIN_CHUNK_TOKENS):
        raise ValueError(f"{FLAG_CHUNK_TOKENS} must be 0 or at least {MIN_CHUNK_TOKENS}")
    if value > MAX_CHUNK_TOKENS:
        raise ValueError(f"{FLAG_CHUNK_TOKENS} must be at most {MAX_CHUNK_TOKENS}")
    return value


def chunk_policy() -> str:
    """Chunking policy from the environment ("grid" when unset). Bad values raise."""
    value = (os.environ.get(FLAG_CHUNK_POLICY, "") or "grid").strip().lower()
    if value not in CHUNK_POLICIES:
        raise ValueError(f"{FLAG_CHUNK_POLICY} must be one of {', '.join(CHUNK_POLICIES)}")
    return value


def chunk_budget_bytes() -> int:
    """Per-chunk memory budget of the "budgeted" policy, in bytes. Bad values raise."""
    raw = os.environ.get(FLAG_CHUNK_BUDGET_MB, "") or str(DEFAULT_CHUNK_BUDGET_MB)
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{FLAG_CHUNK_BUDGET_MB} must be a number of MB") from error
    if value < 100:
        raise ValueError(f"{FLAG_CHUNK_BUDGET_MB} must be at least 100 MB")
    return int(value * 1e6)


def budgeted_chunk(position: int, cap: int, budget: int) -> int:
    """Largest candidate chunk <= ``cap`` whose estimated increment at ``position`` fits ``budget`` (smallest if none)."""
    from models.flashnext.prefill_budget import chunk_increment_bytes

    candidates = [c for c in CANDIDATE_CHUNKS if c <= cap] or [min(CANDIDATE_CHUNKS)]
    for c in candidates:
        if chunk_increment_bytes(c, position) <= budget:
            return c
    return candidates[-1]


def budgeted_edges(end: int, cap: int, budget: int, anchors=(), start: int = 0) -> list:
    """Internal edges of the "budgeted" plan over ``[start, end]`` (greedy from ``start``; ``start=0`` is the cold plan).

    Anchors are cut points. A final piece shorter than MIN_CHUNK_TOKENS is merged into the previous one unless
    that edge is an anchor; the rule depends only on the last edge and ``end``, so the plan before it is prefix-stable.
    """
    wanted = sorted({int(a) for a in anchors if start < int(a) < end})
    edges, position = [], int(start)
    while position < end:
        chunk = budgeted_chunk(position, cap, budget)
        stop = min(position + chunk, end)
        for a in wanted:
            if position < a < stop:
                stop = a
                break
        if stop < end:
            edges.append(stop)
        position = stop
    if edges and end - edges[-1] < MIN_CHUNK_TOKENS and edges[-1] not in wanted:
        edges.pop()
    return edges


def balanced_edges(end: int, chunk: int, anchors=()) -> list:
    """Internal chunk edges of the cold plan over ``[0, end]`` (the policy "balanced").

    ``anchors`` are absolute positions that must be edges (the system-message boundary, message ends). Each
    segment between consecutive anchors is cut into the fewest equal pieces of at most ``chunk`` tokens (the
    first ``length % pieces`` pieces carry the extra token). The result depends only on ``end``, ``chunk`` and
    the anchors, never on how many tokens a cache already holds, so a run restored at any edge of this plan
    continues with exactly the cold run's remaining pieces.
    """
    points = [0] + sorted({int(a) for a in anchors if 0 < int(a) < end}) + [end]
    edges = []
    for start, stop in zip(points, points[1:]):
        length = stop - start
        pieces = -(-length // chunk) if chunk else 1
        base, extra = divmod(length, pieces)
        position = start
        for index in range(pieces):
            position += base + (1 if index < extra else 0)
            edges.append(position)
    return edges[:-1]


def plan_chunks(total: int, chunk: int, boundaries=(), offset: int = 0, policy=None, anchors=None):
    """Spans ``[(start, end), ...]`` covering ``range(total)``.

    ``offset`` is how many tokens the cache already holds, so the prompt being planned ends at the absolute
    position ``offset + total``.

    Policy "grid" (the default): edges fall at every multiple of ``chunk`` in *absolute* position and at each
    requested boundary, which is also absolute. A run restored from a snapshot at position ``offset``
    therefore uses the same spans as a cold run that passed that position, which is what makes the two
    bit-identical.

    Policy "balanced": the cold plan over ``[0, offset + total]`` is computed from ``anchors`` (default:
    ``boundaries``) and the restored run takes its tail. Pass the anchors of the whole prompt, including those
    at or before ``offset``. When ``offset`` is an edge of the cold plan the spans are exactly the cold run's
    remaining spans; otherwise the first span is shortened to the next edge and the rest still match.
    """
    policy = chunk_policy() if policy is None else policy
    end = offset + total
    if policy == "budgeted":
        cold = budgeted_edges(end, chunk or max(CANDIDATE_CHUNKS), chunk_budget_bytes(),
                              boundaries if anchors is None else anchors)
        points = [0] + [e - offset for e in cold if offset < e < end] + [total]
        return [(a, b) for a, b in zip(points, points[1:]) if b > a]
    if policy == "balanced":
        cold = balanced_edges(end, chunk, boundaries if anchors is None else anchors)
        points = [0] + [e - offset for e in cold if offset < e < end] + [total]
        return [(a, b) for a, b in zip(points, points[1:]) if b > a]
    edges = set()
    if chunk:
        first = (offset // chunk + 1) * chunk
        edges.update(range(first, offset + total, chunk))
    wanted = {b for b in boundaries if offset < b < offset + total}
    edges.update(wanted)
    ordered = sorted(edges)
    # A last chunk shorter than MIN_CHUNK_TOKENS is merged into the one before it (a 1-token tail would
    # run the decode-shaped kernels). A requested boundary is never dropped. The rule depends only on the
    # absolute last edge and the end, so a restored run and the cold run still split alike.
    if ordered and offset + total - ordered[-1] < MIN_CHUNK_TOKENS and ordered[-1] not in wanted:
        ordered.pop()
    points = [0] + [e - offset for e in ordered] + [total]
    return [(a, b) for a, b in zip(points, points[1:]) if b > a]


def _prefill_chunk(language, ids, cache) -> None:
    """Run one non-final chunk: advance every cache, keep nothing else."""
    output = language(ids, cache=cache, return_hidden=True, skip_logits=True)
    mx.eval([entry.state for entry in cache])
    output = None
    if PREFILL_CLEAR_CACHE:
        mx.clear_cache()


def _prefill_target(
    language,
    ids,
    cache,
    want_logits: bool = False,
    want_hidden: bool = False,
    sampler=None,
    total_tokens: int | None = None,
):
    """Push one target prompt through the model and seed its next token.

    The returned tuple is ``(logits, hidden_states, token)``. Requested
    tensors are retained for callers that need them. The token always comes
    from the exact target path, using ``sampler`` when supplied.

    Short prompts keep the normal full-logit call. Large prompts skip
    full-sequence logits unless the caller explicitly requests them.
    """
    def pick(row):
        if sampler is None:
            return mx.argmax(row, axis=-1).astype(mx.uint32)
        return sampler(row).astype(mx.uint32)

    full_logits_limit = min(
        QSA_CHUNK_THRESHOLD,
        PREFILL_FULL_LOGITS_MAX_TOKENS,
    )
    # ``total_tokens`` is the whole prompt when this is the last chunk of a chunked prefill, so the
    # choice between full logits and the fused head is the one the unchunked call would make.
    short_prompt = int(total_tokens if total_tokens is not None else ids.shape[1]) <= full_logits_limit
    full_logits = want_logits or short_prompt
    call = {
        "cache": cache,
        "return_hidden": want_hidden or not full_logits,
        "skip_logits": not full_logits,
    }
    # MTP relies on the model's explicit hidden-sink path. Keep this
    # keyword absent for ordinary large prefills, which use the fast path.
    if want_hidden:
        call["capture_layer_ids"] = []
    output = language(ids, **call)
    logits = output.logits if full_logits else None
    hidden_states = output.hidden_states if (want_hidden or not full_logits) else None

    if full_logits:
        mx.eval(logits)
        if hidden_states:
            mx.eval(hidden_states)
        token = pick(logits[:, -1, :])
        mx.eval(token)
        release_cache = int(logits.nbytes) >= PREFILL_RELEASE_BYTES
        if release_cache or (not short_prompt and PREFILL_CLEAR_CACHE):
            mx.clear_cache()
    else:
        last_hidden = hidden_states[-1][:, -1:]
        if sampler is None:
            # The fused head skips materialising logits, which is why this
            # path exists. Only a sampler needs them, and only for the final
            # row.
            token = language.speculative_argmax_from_hidden(
                last_hidden
            ).reshape(-1).astype(mx.uint32)
        else:
            token = pick(language.speculative_logits_from_hidden(last_hidden))
        mx.eval(token, [entry.state for entry in cache])
        # A short prompt leaves no large temporaries. Keep the allocator's
        # buffer cache, as the full-logit short path always has.
        if PREFILL_CLEAR_CACHE and not short_prompt:
            mx.clear_cache()

    # Drop the model output wrapper while retaining only what the caller
    # requested. This keeps the allocator release effective for large calls.
    output = None
    if want_logits:
        retained_logits = logits
    else:
        retained_logits = None
    retained_hidden = hidden_states if want_hidden else None
    return retained_logits, retained_hidden, token


def prefill_language(language, ids, cache, sampler=None, **options):
    """Compatibility wrapper for the standard target prefill entry point.

    ``options`` pass ``boundaries``, ``on_boundary`` and ``offset`` through to ``prefill_target``.
    """
    _logits, _hidden, token = prefill_target(
        language, ids, cache, sampler=sampler, **options
    )
    return None, token


@dataclass
class PlanTrace:
    """What a prefill really did: the chunk edges that ran (absolute positions) and per-chunk measurements.

    ``edges`` are the internal edges actually executed; a snapshot taken at a boundary records the edges before it
    from here, so a run the live guard re-planned can never pose as the cold plan's state. ``chunks`` are measurements
    (peak MLX memory above the memory resident before the chunk, wall seconds, host headroom when read)."""
    policy: str = ""
    cap: int = 0
    budget_mb: float = 0.0
    planned_edges: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    stepdowns: list = field(default_factory=list)
    deviated: bool = False
    throttles: list = field(default_factory=list)      # bounded waits at chunk edges while the memory controller was ELEVATED

    def edges_before(self, position: int) -> tuple:
        return tuple(e for e in self.edges if e < position)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("policy", "cap", "budget_mb", "planned_edges", "edges", "chunks",
                                              "stepdowns", "deviated", "throttles")}


def _active_memory():
    """``(active_bytes, peak_bytes)`` before a chunk. The peak counter is never reset here: the benchmark harness and the
    guards read the process-wide peak, and a per-chunk reset would corrupt it."""
    try:
        return int(mx.get_active_memory()), int(mx.get_peak_memory())
    except Exception:
        return 0, 0


def _measured(trace, start, end, seconds, before, available=None, pressure=None):
    """Record one chunk. The increment above the pre-chunk resident memory is exact only when the chunk set a new
    process peak; otherwise the chunk stayed below an earlier peak and only an upper bound is known (None)."""
    if trace is None:
        return
    active, peak_before = before
    try:
        peak = int(mx.get_peak_memory())
    except Exception:
        peak = 0
    trace.chunks.append({"start": start, "end": end, "tokens": end - start, "seconds": round(seconds, 3),
                         "increment_peak_mb": round(max(0, peak - active) / 1e6, 1) if peak > peak_before else None,
                         "process_peak_mb": round(peak / 1e6, 1), "resident_before_mb": round(active / 1e6, 1),
                         "available_gb": None if available is None else round(available / 2 ** 30, 2),
                         "pressure": pressure, "memory_state": memory_controller.state()})


def _uncache_reads_when_elevated(language, trace=None) -> None:
    """At a chunk edge while the memory controller is ELEVATED, read the next chunk's expert rows with F_NOCACHE so they stop
    filling the file cache (the bytes and their order are unchanged, so results are identical). The backend restores
    buffered reads when the request ends (``_release_between_turns``). Never raises.

    OPT-IN (``FLASHNEXT_UNCACHE_WHEN_ELEVATED=1``), off by default since 2026-11-09. In the failed real 8K run both stretches of
    zero page-ins began right after an ELEVATED episode, where this switch turned on, and it stayed on until the request ended:
    F_NOCACHE reads bypass the file cache, so vm_stat stops counting them as page-ins and nothing shows whether the prefill
    kept its pace. It also removed the compression churn, so it may be good, but it was unvalidated in the run that failed
    and is not worth the risk until the per-chunk log (``FLASHNEXT_CHUNK_LOG``) can compare it with buffered reads."""
    if os.environ.get("FLASHNEXT_UNCACHE_WHEN_ELEVATED", "0") != "1" or not memory_controller.elevated():
        return
    try:
        from models.flashnext import pool_lifecycle

        store = pool_lifecycle.store_of(language)
        if store is not None and hasattr(store, "set_uncached_reads") and not getattr(store, "_uncached", False):
            store.set_uncached_reads(True)
            if trace is not None:
                trace.throttles.append({"uncached_reads": True, "state": memory_controller.state()})
    except Exception:
        pass


CHUNK_LOG_FLAG = "FLASHNEXT_CHUNK_LOG"


def _log_chunk(start, end, seconds, data, trace=None) -> None:
    """Append one JSON line per finished chunk to ``$FLASHNEXT_CHUNK_LOG`` (when set), AS IT FINISHES. The per-request chunk
    trace is only written when the request ends, so a prefill that never ends (the 8K gate, 2026-11-08: no request row after 15
    minutes) left no record of which chunk was slow. Never raises."""
    path = os.environ.get(CHUNK_LOG_FLAG)
    if not path:
        return
    try:
        import json

        row = {"wall": round(time.time(), 2), "start": start, "end": end, "tokens": end - start,
               "seconds": round(seconds, 2), "memory_state": memory_controller.state(),
               "available_gb": None if not data else round(data["available"] / 2 ** 30, 2),
               "pressure": None if not data else data["pressure"], "mlx_active_gb": round(mx.get_active_memory() / 2 ** 30, 2),
               "mlx_peak_gb": round(mx.get_peak_memory() / 2 ** 30, 2)}
        with open(path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
    except Exception:
        pass


def _run_chunks(language, ids, cache, want_logits, want_hidden, sampler, boundaries, on_boundary, offset,
                anchors=None, trace=None, scheduler=None):
    """The prefill body. One call unless chunking or boundaries split it.

    With a ``scheduler`` (budgeted policy + FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS) the live guard may re-plan the REST of the
    prompt between chunks. Chunks are cut only at plan edges, so every edge a snapshot or the cache sees is recorded."""
    total = int(ids.shape[1])
    spans = [(0, total)]
    chunk = chunk_tokens()
    if not (want_logits or want_hidden):
        spans = plan_chunks(total, chunk, boundaries, offset, anchors=anchors) or [(0, total)]
    if trace is not None:
        trace.planned_edges = [offset + e for _, e in spans[:-1]]
    # A short tool-result prefill is one span and has no chunk edge, and it skips the F_NOCACHE ring (under 512 tokens).
    # If the controller is still ELEVATED or RECOVERING from the previous turn, start it with uncached reads.
    _uncache_reads_when_elevated(language, trace)
    if len(spans) == 1:
        began, active = time.perf_counter(), _active_memory()
        result = _prefill_target(language, ids, cache, want_logits, want_hidden, sampler)
        _measured(trace, offset, offset + total, time.perf_counter() - began, active)
        return result
    absolute_end = offset + total
    wanted = tuple(boundaries if anchors is None else anchors)
    boundary_set = set(boundaries)
    index = 0
    while index < len(spans) - 1:
        start, end = spans[index]
        began, active = time.perf_counter(), _active_memory()
        _prefill_chunk(language, ids[:, start:end], cache)
        seconds = time.perf_counter() - began
        if trace is not None:
            trace.edges.append(offset + end)
        if on_boundary is not None and (offset + end) in boundary_set:
            on_boundary(offset + end)
        # A safe boundary: the chunk is evaluated and no GPU work is in flight. While the memory controller reports
        # ELEVATED, wait (bounded) for the machine to calm down before the next chunk starts.
        waited = memory_controller.checkpoint("prefill-chunk")
        if waited["waited_s"] > 0 and trace is not None:
            trace.throttles.append({"at": offset + end, **waited})
        _uncache_reads_when_elevated(language, trace)
        data = scheduler.read_system() if scheduler is not None else None
        _log_chunk(offset + start, offset + end, seconds, data)
        _measured(trace, offset + start, offset + end, seconds, active,
                  None if data is None else data["available"], None if data is None else data["pressure"])
        if scheduler is not None and data is not None:
            before = scheduler.current_chunk_size
            cap = scheduler.observe_and_adapt(data["available"], data["pressure"])
            if cap != before:
                # Re-plan the remainder from this edge with the new cap. Chunks before it are untouched.
                rest = budgeted_edges(absolute_end, cap, chunk_budget_bytes(), wanted, start=offset + end)
                points = [end] + [e - offset for e in rest] + [total]
                spans[index + 1:] = [(a, b) for a, b in zip(points, points[1:]) if b > a]
                if trace is not None:
                    trace.deviated = True
                    trace.stepdowns.append({"at": offset + end, "cap_from": before, "cap_to": cap,
                                            "available_gb": round(data["available"] / 2 ** 30, 2),
                                            "pressure": data["pressure"]})
        index += 1
    start, end = spans[-1]
    began, active = time.perf_counter(), _active_memory()
    result = _prefill_target(language, ids[:, start:end], cache, want_logits, want_hidden, sampler,
                             total_tokens=total + offset)
    _log_chunk(offset + start, offset + end, time.perf_counter() - began, None)
    _measured(trace, offset + start, offset + end, time.perf_counter() - began, active)
    return result


def prefill_target(
    language,
    ids,
    cache,
    want_logits: bool = False,
    want_hidden: bool = False,
    sampler=None,
    boundaries=(),
    on_boundary=None,
    offset: int = 0,
    anchors=None,
    trace=None,
):
    """Push one target prompt through the model and seed its next token.

    ``trace`` (a ``PlanTrace``) is filled with the chunks that really ran; see its docstring.

    With ``FLASHNEXT_PREFILL_RING_EXPERTS`` and ``FLASHNEXT_PREFILL_CACHE_LIMIT_MB``
    both unset (the default) this is exactly ``_prefill_target``. The cache
    limit (``prefill_memory``) caps MLX's allocator cache for the call; the ring
    (``prefill_ring``) holds the in-pool read ring, or with
    ``FLASHNEXT_PREFILL_RING_SCRATCH=1`` a temporary pool on a first turn.

    ``FLASHNEXT_PREFILL_CHUNK_TOKENS`` (or ``boundaries``) splits the call into chunks that share the
    cache; ``on_boundary(position)`` runs after the chunk that ends at each absolute boundary, with the
    cache evaluated, so a caller can snapshot it. ``offset`` is how many tokens the cache already holds.
    ``anchors`` (policy "balanced") are the absolute positions of the whole prompt that must be chunk edges,
    including those before ``offset``; ``boundaries`` is the subset the caller wants reported. Not used when
    logits or hidden states are requested.
    """
    from models.flashnext import prefill_memory, prefill_ring

    ring_on = prefill_ring.enabled()
    cap_on = prefill_memory.enabled()
    scheduler = None
    if trace is not None:
        trace.policy, trace.cap, trace.budget_mb = chunk_policy(), chunk_tokens(), chunk_budget_bytes() / 1e6
    if chunk_policy() == "budgeted" and chunk_tokens():
        from models.flashnext import prefill_scheduler

        if prefill_scheduler.adaptive_enabled():
            scheduler = prefill_scheduler.AdaptivePrefillScheduler(max_chunk=chunk_tokens())
            scheduler.start_at(chunk_tokens())

    def body():
        return _run_chunks(language, ids, cache, want_logits, want_hidden, sampler,
                           tuple(boundaries), on_boundary, int(offset),
                           None if anchors is None else tuple(anchors), trace, scheduler)

    if not ring_on and not cap_on:
        return body()
    with prefill_memory.cache_cap():
        if not ring_on:
            return body()
        with prefill_ring.session(language, int(ids.shape[1])) as ring:
            result = body()
            if ring is not None:
                # Nothing the caller keeps may still read ring memory lazily.
                mx.eval([item for item in result if item is not None])
                prefill_ring.fence()
            return result
