"""Glue between one prefill and the prefix cache (opt-in: ``FLASHNEXT_PREFIX_CACHE=1``).

``PrefixSession.prefill`` is what the backend calls instead of a plain prefill for a fresh conversation:

1. look up the longest stored token prefix that is strictly shorter than the prompt and restore it,
2. run the prefill for the remaining tokens, stopping at the configured boundaries (the end of the
   system message, and optionally every chunk edge) so the state there can be snapshotted,
3. report what was reused, what was computed and what restoring cost.

It never approximates: a prefix is used only when every token and the whole engine fingerprint match.
Native MTP and the fused decoder keep their own caches and are not covered (the backend skips this path
for them).
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from . import prefill as prefill_module
from . import prefix_cache as P
from . import prefix_profiles

FLAG = "FLASHNEXT_PREFIX_CACHE"
FLAG_DIR = "FLASHNEXT_PREFIX_CACHE_DIR"
FLAG_RAM_GB = "FLASHNEXT_PREFIX_CACHE_RAM_GB"
FLAG_DISK_GB = "FLASHNEXT_PREFIX_CACHE_DISK_GB"
FLAG_MIN_TOKENS = "FLASHNEXT_PREFIX_CACHE_MIN_TOKENS"
FLAG_BOUNDARIES = "FLASHNEXT_PREFIX_CACHE_BOUNDARIES"
# Test control. "0" keeps the session's chunk plan (boundaries and anchors) but never restores and never writes a
# snapshot: the cold, uncached run that has exactly the chunk plan a cached run would continue. Default "1".
FLAG_RESTORE = "FLASHNEXT_PREFIX_CACHE_RESTORE"
DEFAULT_DIR = "~/.flashnext/prefix-cache"
IM_END = "<|im_end|>\n"


def enabled() -> bool:
    return os.environ.get(FLAG, "0") == "1"


def restore_enabled() -> bool:
    return os.environ.get(FLAG_RESTORE, "1") != "0"


def _number(name: str, default: str) -> float:
    raw = os.environ.get(name, default) or default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def boundary_mode() -> str:
    mode = os.environ.get(FLAG_BOUNDARIES, "system") or "system"
    if mode not in ("system", "chunks", "messages", "all"):
        raise ValueError(f"{FLAG_BOUNDARIES} must be 'system', 'chunks', 'messages' or 'all'")
    return mode


def anchor_mode() -> str:
    """Where the chunk plan is forced to cut. "all" is "messages" plus extra snapshots at the plan's own chunk edges: the
    anchors, and so every plan, are the same, which is why stored entries stay valid when the mode changes between them."""
    mode = boundary_mode()
    return "messages" if mode == "all" else mode


class RecoveryRequiredError(RuntimeError):
    """A long conversation lost its state cache and cannot be safely recomputed cold."""

    def __init__(self, message: str, status: int = 400, code: str | None = "recovery_required"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


@dataclass
class Report:
    prompt_tokens: int
    hit_tokens: int = 0
    prefilled_tokens: int = 0
    tier: str | None = None
    outcome: str = "COLD"                               # RAM_HIT, SSD_HIT, PREFIX_MISMATCH, STATE_INVALID, EVICTED, RESTORE_FAILED, COLD
    lookup_seconds: float = 0.0
    restore_seconds: float = 0.0
    prefill_seconds: float = 0.0
    bytes_loaded: int = 0
    bytes_restored: int = 0
    snapshots: list = field(default_factory=list)       # token counts stored during this prefill
    snapshot_seconds: float = 0.0                       # time spent storing them (SSD write included)
    snapshots_skipped: list = field(default_factory=list)  # {position, reason}; a skipped boundary needs recomputation
    plan_rejections: int = 0                            # stored prefixes skipped for an incompatible chunk history
    plan_edges: list = field(default_factory=list)      # internal chunk edges of this prompt's cold plan
    executed_edges: list = field(default_factory=list)  # internal chunk edges that really ran (differs if the guard re-planned)
    chunk_trace: list = field(default_factory=list)     # per-chunk measurements (seconds, peak increment, headroom)
    execution_order: str = ""
    execution_windows: list = field(default_factory=list)
    execution_segments: list = field(default_factory=list)
    plan_deviated: bool = False                         # the live guard changed the plan mid-prefill
    stepdowns: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PrefixSession:
    """One per backend. ``compatibility`` returns the dict that fingerprints the engine and checkpoint."""

    def __init__(self, encode, compatibility):
        self._encode = encode
        self._compatibility = compatibility
        self._store: P.PrefixCache | None = None

    # ---- configuration ----
    def store(self) -> P.PrefixCache:
        if self._store is None:
            policy = prefill_module.chunk_policy()
            # The balanced plan depends on where the anchors are, which depends on these settings, so they
            # are part of the key. The default "grid" policy keeps the original key unchanged.
            plan = {} if policy == "grid" else {
                "chunk_policy": policy, "boundaries": anchor_mode(),
                "min_tokens": int(_number(FLAG_MIN_TOKENS, "512"))}
            if policy == "budgeted":
                # The plan is a function of this budget and the candidate sizes; a different budget is a different plan.
                plan["chunk_budget_mb"] = prefill_module.chunk_budget_bytes() / 1e6
                plan["chunk_candidates"] = list(prefill_module.CANDIDATE_CHUNKS)
            from . import layer_major, quantized_kv
            # Layer-major execution is a different (versioned) execution plan until its states are proven equal on the real
            # checkpoint: its snapshots never mix with chunk-major ones. Chunk-major keeps the existing key.
            plan = {**plan, **({"prefill_order": layer_major.identity()} if layer_major.enabled() else {})}
            kv_part = quantized_kv.fingerprint_parts()      # bf16: empty (unchanged); else dtype, bits, group size
            fp = P.fingerprint(
                **self._compatibility(), prefix_cache=P.FORMAT_VERSION,
                snapshot_chain_semantics=2,
                chunk=prefill_module.chunk_tokens(), **plan, **kv_part,
                # Prefill settings that change numerics are part of the key: a prefix computed one way
                # is not the same state as one computed another way.
                ring=os.environ.get("FLASHNEXT_PREFILL_RING_EXPERTS", "0"),
                qsa_chunk=os.environ.get("FLASHNEXT_QSA_QUERY_CHUNK", "2048"),
            )
            directory = os.path.expanduser(os.environ.get(FLAG_DIR) or DEFAULT_DIR)
            if prefix_profiles.enabled():
                # Opt-in profile manager: same exact snapshots, plus frequency-aware eviction, pins, a size budget and
                # a free-space reserve (its own settings, FLASHNEXT_PREFIX_PROFILES_*).
                self._store = prefix_profiles.ProfiledPrefixCache(
                    fp, int(_number(FLAG_RAM_GB, "0.75") * 1e9), Path(directory),
                    config=self.execution_profile(), **prefix_profiles.settings_from_env())
            else:
                self._store = P.PrefixCache(
                    fp, int(_number(FLAG_RAM_GB, "0.75") * 1e9), Path(directory),
                    int(_number(FLAG_DISK_GB, "24") * 1e9))
        return self._store

    @staticmethod
    def execution_profile() -> dict:
        """The memory and execution settings a snapshot was produced under (recorded in its profile; the numeric ones
        are already in the fingerprint)."""
        from . import expert_pool

        return {"chunk_tokens": prefill_module.chunk_tokens(), "chunk_policy": prefill_module.chunk_policy(),
                "ring_experts": os.environ.get("FLASHNEXT_PREFILL_RING_EXPERTS", "0"),
                "expert_pool_gb": expert_pool.configured_gb(),
                "memory_manager": os.environ.get("FLASHNEXT_MEMORY_MANAGER", "0") == "1"}

    def system_boundary(self, prompt) -> int | None:
        """Token count just after the first ``<|im_end|>\\n`` (the end of the system message)."""
        marker = list(self._encode(IM_END))
        if not marker:
            return None
        width = len(marker)
        for index in range(len(prompt) - width + 1):
            if prompt[index] == marker[0] and list(prompt[index:index + width]) == marker:
                return index + width
        return None

    def message_boundaries(self, prompt) -> list:
        """Token counts just after every ``<|im_end|>\\n``: the end of the system message, each user, assistant
        and tool message (a tool result arrives inside a user turn and ends the same way)."""
        marker = list(self._encode(IM_END))
        if not marker:
            return []
        width, found, index = len(marker), [], 0
        while index <= len(prompt) - width:
            if prompt[index] == marker[0] and list(prompt[index:index + width]) == marker:
                found.append(index + width)
                index += width
            else:
                index += 1
        return found

    def anchors(self, prompt) -> tuple:
        """Positions the chunk plan must cut at, whatever the cache already holds (the structural boundaries
        a snapshot is taken at): the system-message end, or every message end in "messages" mode."""
        minimum = int(_number(FLAG_MIN_TOKENS, "512"))
        if boundary_mode() in ("messages", "all"):
            chosen = set(self.message_boundaries(prompt))
        else:
            system = self.system_boundary(prompt)
            chosen = set() if system is None else {system}
        return tuple(sorted(b for b in chosen if minimum <= b < len(prompt)))

    def boundaries(self, prompt, offset: int) -> tuple:
        """Absolute positions to snapshot at (after ``offset``, before the last token)."""
        minimum = int(_number(FLAG_MIN_TOKENS, "512"))
        chosen = set(self.anchors(prompt))
        if boundary_mode() in ("chunks", "all"):
            chunk = prefill_module.chunk_tokens()
            if prefill_module.chunk_policy() == "budgeted":
                chosen.update(prefill_module.budgeted_edges(
                    len(prompt),
                    chunk or max(prefill_module.CANDIDATE_CHUNKS),
                    prefill_module.chunk_budget_bytes(),
                    self.anchors(prompt),
                ))
            elif chunk:
                if prefill_module.chunk_policy() == "balanced":
                    # Every internal edge of the cold plan (they depend on the prompt's end and anchors).
                    chosen.update(prefill_module.balanced_edges(len(prompt), chunk, self.anchors(prompt)))
                else:
                    chosen.update(range(chunk, len(prompt), chunk))
        return tuple(sorted(b for b in chosen if max(minimum, offset + 1) <= b < len(prompt)))

    def cold_edges(self, prompt) -> tuple:
        """Internal chunk edges of this prompt's cold plan (the run that has no cache to start from)."""
        spans = prefill_module.plan_chunks(
            len(prompt), prefill_module.chunk_tokens(), self.boundaries(prompt, 0), 0,
            anchors=self.anchors(prompt))
        return tuple(end for _, end in spans[:-1])

    @staticmethod
    def compatible(edges, n_tokens, plan) -> bool:
        """May a state stored at ``n_tokens`` with chunk history ``plan`` continue a prompt whose cold plan has
        the internal edges ``edges``? Only if the cold plan cuts at ``n_tokens`` and made exactly the stored
        cuts before it, so the continuation is the cold run's tail and the state is what the cold run held
        there. An unknown history (None) is never compatible."""
        if plan is None or n_tokens not in edges:
            return False
        return tuple(e for e in edges if e < n_tokens) == tuple(plan)

    # ---- the prefill ----
    def reusable_tokens(self, prompt) -> int:
        """Leading tokens the next ``prefill`` would restore (0 with restore off or no compatible entry). Side-effect
        free; used to size the memory a prefill will need before it starts."""
        if not enabled() or not restore_enabled():
            return 0
        prompt = [int(t) for t in prompt]
        edges = self.cold_edges(prompt)
        return self.store().peek(prompt, accept=lambda n, plan: self.compatible(edges, n, plan))

    def prefill(self, language, prompt, cache, sampler=None):
        """Returns ``(token, cache, report)``. ``cache`` must be a fresh, empty cache list."""
        prompt = [int(t) for t in prompt]
        store = self.store()
        report = Report(prompt_tokens=len(prompt))
        edges = self.cold_edges(prompt)
        report.plan_edges = list(edges)
        restore = restore_enabled()
        system = self.system_boundary(prompt) if restore else None
        if system is not None and system >= int(_number(FLAG_MIN_TOKENS, "512")):
            store.observe(prompt[:system])               # an exact system prompt seen again is a recognised profile
        rejected_before = store.stats.rejected_plan
        lookup_began = time.perf_counter()
        if restore:
            res = store.lookup(prompt, language.make_cache, language=language,
                               accept=lambda n, plan: self.compatible(edges, n, plan))
            hit, restored = res[0], res[1]
            outcome = getattr(res, "outcome", ("RAM_HIT" if hit and hit.tier == "ram" else "SSD_HIT" if hit else "COLD"))
        else:                                   # same plan, no restore, no snapshot (test control)
            hit, restored = None, None
            outcome = "COLD"
        report.lookup_seconds = time.perf_counter() - lookup_began
        report.outcome = outcome
        report.plan_rejections = store.stats.rejected_plan - rejected_before
        offset = 0
        if hit is not None:
            cache, offset = restored, hit.n_tokens
            report.hit_tokens, report.tier = hit.n_tokens, hit.tier
            report.restore_seconds, report.bytes_loaded = hit.seconds, hit.bytes_loaded
            report.bytes_restored = hit.bytes_loaded

        needed = len(prompt) - offset
        max_cold = int(os.environ.get("VEYSTRIX_MAX_COLD_PREFILL") or 32768)
        max_inc = int(os.environ.get("VEYSTRIX_MAX_INCREMENTAL_PREFILL") or max_cold)
        # Use the incremental allowance only after an actual successful restore;
        # a failed or incompatible lookup remains subject to the cold limit.
        effective_limit = max_inc if offset > 0 else max_cold
        if needed > effective_limit:
            outcome_str = getattr(res, "outcome", outcome) if restore else "COLD"
            raise RecoveryRequiredError(
                f"session state could not be recovered from memory or SSD cache (total prompt: {len(prompt)} "
                f"tokens, matched prefix: {offset} tokens, uncached tokens: {needed}, outcome: {outcome_str}). "
                f"Prefilling {needed} uncached tokens would exceed the configured "
                f"{'incremental' if offset else 'cold'} prefill limit of {effective_limit} "
                "tokens and is prohibited on 16 GB hardware. A compatible state snapshot is required for recovery. "
                "Shorten the prompt or resume from a saved session snapshot.",
                code="recovery_required",
            )
        boundaries = self.boundaries(prompt, offset)

        last = []
        held = contextlib.ExitStack()
        trace = prefill_module.PlanTrace()

        def history(position):
            """Chunk edges before ``position`` that really produced this state. Without a re-plan that is the cold plan
            exactly. After one it is the restored prefix's cold edges (the lookup proved they match) followed by the
            edges that ran, which a later cold plan will not match, so the snapshot can only miss, never mislead."""
            if not trace.deviated:
                return tuple(e for e in edges if e < position)
            base = tuple(e for e in edges if e <= offset) if offset else ()
            return base + trace.edges_before(position)

        def snapshot(position, frozen=None):
            if not restore:
                return
            from models.flashnext import memory_controller, flight_recorder
            usage = store.usage()
            reason = ("memory_pressure" if memory_controller.forbid_optional() else
                      "ram_budget_exceeded" if usage["ram_bytes"] > usage["ram_budget"] else None)
            if reason is not None:
                report.snapshots_skipped.append({"position": position, "reason": reason})
                flight_recorder.record("prefix_snapshot_skipped", position=position, reason=reason)
                return
            # Chain each snapshot to the previous one (or to the restored hit) so KV is stored once.
            stored = time.perf_counter()
            flight_recorder.record("prefix_snapshot_begin", position=position)
            snapshot_plan = history(position)
            entry = store.put(prompt[:position], cache if frozen is None else frozen, parent=last[-1] if last else hit, language=language,
                              plan=snapshot_plan,
                              kind="system" if position == system else None, auto_parent=False)
            flight_recorder.record("prefix_snapshot_end", position=position, seconds=round(time.perf_counter() - stored, 3))
            report.snapshot_seconds += time.perf_counter() - stored
            if entry.plan != snapshot_plan:
                # A token-identical entry produced under different chunking is
                # not the state we just computed. Do not chain a delta to it.
                flight_recorder.record("prefix_snapshot_plan_conflict", position=position)
                return
            held.enter_context(store.lease((entry.key,)))   # the next snapshot chains to it: keep it until we finish
            last.append(entry)
            report.snapshots.append(position)

        suffix = mx.array(prompt[offset:])[None]
        report.prefilled_tokens = len(prompt) - offset
        import logging
        _prefix_log = logging.getLogger("flashnext.prefix_session")
        _prefix_log.info(
            "prefix cache lookup: outcome=%s matching_prefix_length=%d restored_token_count=%d newly_prefilled_token_count=%d lookup_seconds=%.4f restore_seconds=%.4f bytes_restored=%d",
            report.outcome, report.hit_tokens, report.hit_tokens, report.prefilled_tokens, report.lookup_seconds, report.restore_seconds, report.bytes_restored,
        )
        began = time.perf_counter()
        with held:                                       # leases end however the prefill ends (error, cancellation)
            if hit is not None:
                held.enter_context(store.lease(hit.chain))
            _, _, token = prefill_module.prefill_target(
                language, suffix, cache, sampler=sampler, boundaries=boundaries,
                on_boundary=snapshot, offset=offset, anchors=self.anchors(prompt), trace=trace,
                required=tuple(x for x in (system,) if x is not None))
        report.executed_edges, report.chunk_trace = list(trace.edges), list(trace.chunks)
        report.execution_order, report.execution_windows = trace.order, list(trace.windows)
        report.execution_segments = list(trace.layers)
        report.plan_deviated, report.stepdowns = trace.deviated, list(trace.stepdowns)
        report.prefill_seconds = time.perf_counter() - began
        if restore:
            store.record_prefill(report.prefilled_tokens, report.prefill_seconds)
        return token, cache, report
