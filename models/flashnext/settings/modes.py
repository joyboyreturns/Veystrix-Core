"""The two user-facing operating modes of the one Flash-Next runtime: CHAT and AGENTIC.

A mode is configuration, not a second implementation. Both modes load the same model through the same
backend, quantized expert representation, Metal kernels, cache-integrity rules and safety supervisor
(``ChatGuard``); they differ in the environment preset below, the chat profile, and whether the local
server is started. A mode is chosen at launch and the process is not reconfigured live: the flags it sets
are read at import or at the first request, so changing mode means a new process (and a new model load).

Defaults contain only what has real-model evidence. Anything offline-tested only is an explicit ``--enable``
opt-in and is labelled as such in the startup banner. Pure Python with no dependencies, so the launcher
(which may run under the system interpreter) can import it.

Evidence behind the choices (see docs/flashnext/performance-engineering-report-2026-11-08.md):

* Decode pool. Recorded 0 GB against 6 GB decode runs: 3.42-3.47 against 3.79-3.93 tok/s. No decode run at
  2 GB exists. Offline replay of a recorded route trace: LRU hits 0% at 1 GB, about 47% at 2 GB. So the choices
  are 0 or 2 GB, never an in-between size, and 2 GB is a memory-safe default whose decode speed the first real
  session measures. Above 2 GB is refused here (the 12.7 GB Metal limit leaves no room on a 16 GB machine).
* AGENTIC prefill is the ``F2Br`` configuration that passed on the real model at 2K: balanced 2,048-token
  chunks, the 512-expert scratch ring, eager linear-state evaluation, the exact prefix cache, and the decode-pool
  release before a fresh long prefill. Cached against uncached under the same chunk plan was bit-identical.
* CHAT keeps one-call prefill, so its numerics equal the reference digest, and uses live conversation state
  instead of serialized snapshots.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

CHAT, AGENTIC = "chat", "agentic"
MODES = (CHAT, AGENTIC)
PROFILE = {CHAT: "plain", AGENTIC: "agent"}

POOL_CAP_GB = 2.0
DEFAULT_POOL_GB = 2.0
START_GATE_GB = 9.0               # ChatGuard refuses to start below this much available memory
PREFIX_DIR = "~/.flashnext/prefix-cache"
PREFIX_RAM_GB = 0.75
PREFIX_DISK_GB = 50.0             # SSD cache quota; oldest unleased entries are evicted first (LRU) once it is full
QUOTA_ENV = "VEYSTRIX_CACHE_GB"    # owner override of the quota, e.g. VEYSTRIX_CACHE_GB=80 ./chat.sh --mode agentic ...
DISK_RESERVE_GB = 50.0
MIN_PREFIX_DISK_GB = 1.0
# The 2K path has real-model evidence; 4K is a budget estimate (about 7.1 GB peak, calibrated at 1K only).
VALIDATED_PROMPT_TOKENS = 2048
VALIDATED_PREFILL_TOKENS = 16384   # largest cold prefill run on the real checkpoint (2026-11-09: 15,819 tokens, 176 MB swap growth)
AGENTIC_CONTEXT_LIMIT = 262144    # native window of the checkpoint (owner decision 2026-11-09); prefill ceilings below are separate
# Prefill ceilings (owner decision 2026-11-09). Admission ceilings only: prefill still runs in 2,048-token chunks. Real-checkpoint
# evidence exists to 16K cold prefill (4K, 8K, 16K); 32K is implemented and unit-tested but NOT validated on the real checkpoint.
DEFAULT_MAX_COLD_PREFILL = 32768
DEFAULT_MAX_INCREMENTAL_PREFILL = 32768
LONG_CONTEXT_LIMIT = 16384        # with --enable long-context; admission may lower it (prefill_readiness.LONG_TIER)

# Flags every mode sets. ChatGuard is the safety supervisor; native MTP is rejected by recorded results;
# eager linear-state evaluation reproduced the 1K digest and cut retained memory (0.99 to 0.24 GB per 1K tokens).
COMMON = {
    "FLASHNEXT_CHAT_GUARD": "1",
    "FLASHNEXT_NATIVE_MTP": "off",
    "FLASHNEXT_EVAL_LINEAR_STATE": "1",
}
# Flags that switch the prefill and memory machinery. The mode owns them: inherited values are replaced,
# so a research flag left in ~/.frankenstein/env cannot change a mode silently.
PREFILL_OFF = {
    "FLASHNEXT_PREFILL_RING_EXPERTS": "0",
    "FLASHNEXT_PREFILL_RING_SCRATCH": "0",
    "FLASHNEXT_PREFILL_CHUNK_TOKENS": "0",
    "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL": "0",
    "FLASHNEXT_PREFIX_CACHE": "0",
}
AGENTIC_PREFILL = {
    "FLASHNEXT_PREFILL_RING_EXPERTS": "512",
    "FLASHNEXT_PREFILL_RING_SCRATCH": "1",
    "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048",
    "FLASHNEXT_PREFILL_CHUNK_POLICY": "balanced",
    "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL": "1",
    "FLASHNEXT_PREFIX_CACHE": "1",
    "FLASHNEXT_PREFIX_CACHE_BOUNDARIES": "all",   # message ends + every chunk edge: a cancelled or crashed prefill resumes
}
EXPERIMENTAL_OFF = {
    "FLASHNEXT_KV_CACHE_DTYPE": "bf16",             # bf16 KV is the reference; quantized KV is an explicit opt-in below
    "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS": "0",
    "FLASHNEXT_MEMORY_MANAGER": "0",
    "FLASHNEXT_PREFIX_PROFILES": "0",
    "FLASHNEXT_AUTO_MEMORY_RECLAIM": "0",
    "FLASHNEXT_SHORT_PREFILL_POOL_REUSE": "0",
    "FLASHNEXT_PREFILL_ORDER": "chunk",             # chunk-major is the reference order; layer-major is an explicit opt-in below
    "FLASHNEXT_PREFILL_HYBRID_SMALL_WINDOWS": "0",
    "FLASHNEXT_PREFILL_LAYER_PREFETCH": "0",        # layer prefetch overlaps reads with compute; explicit opt-in below
    "FLASHNEXT_UNCACHED_EXPERT_READS": "0",         # buffered reads are the reference; uncached reads are an explicit opt-in below
    "FLASHNEXT_EXPERT_POOL_BUFFERED_FILL": "0",     # decode pool fills bypass the file cache (F_NOCACHE); buffered fills are an opt-in below
    "FLASHNEXT_NGRAM_DEDUP": "0",                  # bounded per-lookup dedup is an unmeasured prototype
}
# Offline-tested only. Each is explicit (--enable NAME), never a default, and labelled in the banner.
EXPERIMENTAL = {
    "ngram-dedup": ({"FLASHNEXT_NGRAM_DEDUP": "1"},
                    "deduplicate repeated n-gram rows within each lookup; offline exactness tests only, real-checkpoint speed unmeasured"),
    "long-context": ({"FLASHNEXT_PREFILL_CHUNK_POLICY": "budgeted", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048",
                      "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB": "1000", "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS": "1"},
                     "deterministic memory-budgeted chunk plan (2048/1024/512/256) with live step-down and a 'long' "
                     "admission tier (16K context, 14K cold prefill); offline-tested only, no real run above 2K"),
    "layer-major": ({"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_PREFILL_WINDOW_TOKENS": "8192"},
                    "layer-major prefill: each window of up to 8,192 tokens reads a layer's experts once for all its chunks "
                    "(about 3x less SSD traffic at 16K); snapshots only at window ends; offline-tested bit-identical on a "
                    "small model, no real-checkpoint run yet (docs/flashnext/architecture-overhaul.md)"),
    "layer-major-4k": ({"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_PREFILL_WINDOW_TOKENS": "4096"},
                       "experimental layer-major prefill with a 4,096-token window; real-checkpoint validation pending"),
    "layer-major-hybrid-4k": ({"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_PREFILL_WINDOW_TOKENS": "4096",
                              "FLASHNEXT_PREFILL_HYBRID_SMALL_WINDOWS": "1"},
                             "experimental 4K layer-major windows with chunk-major execution for windows up to 2K; unmeasured"),
    "layer-prefetch": ({"FLASHNEXT_PREFILL_LAYER_PREFETCH": "1"},
                       "layer-major only: read layer l+1's experts while layer l computes. With the prefill ring (the agentic "
                       "preset) the scratch ring gets a second half (+1.5 GB while prefilling) and the read goes to it with direct "
                       "I/O; without a ring it reads into fresh arrays (+1.5 GB). Same bytes in the same order, so results are "
                       "identical. The fresh-array form measured no gain at pool 0 (0.989x); the two-half ring form is unmeasured "
                       "on the real checkpoint"),
    "uncached-reads": ({"FLASHNEXT_UNCACHED_EXPERT_READS": "1"},
                       "read every expert row with F_NOCACHE so streaming does not fill the file cache or push the server's own "
                       "pages into the compressor; same bytes in the same order, so results are identical; speed effect unmeasured"),
    "uncached-prefill-reads": ({"FLASHNEXT_UNCACHED_EXPERT_READS": "prefill"},
                               "F_NOCACHE expert reads during prefill only; decode goes back to buffered reads so reused experts can "
                               "hit the page cache (the all-phases uncached mode measured decode 1.1-2.0 tok/s); unmeasured"),
    "pool-buffered-fill": ({"FLASHNEXT_EXPERT_POOL_BUFFERED_FILL": "1"},
                           "decode expert-pool fills read through the file cache so free memory acts as a second-level cache for "
                           "experts the pool evicted; same bytes, same results; replay predicts fewer SSD bytes per token; unmeasured live"),
    "kv-affine8": ({"FLASHNEXT_KV_CACHE_DTYPE": "affine8"},
                   "8-bit affine attention KV (about 1.9x smaller K/V); saves ~0.15 GB at 12K; quality on the real "
                   "checkpoint unmeasured, never mixed with bf16 snapshots"),
    "kv-affine4": ({"FLASHNEXT_KV_CACHE_DTYPE": "affine4"},
                   "4-bit affine attention KV; synthetic attention error about 10% (worst rows 40-60% with outlier "
                   "channels), saves ~0.2 GB at 12K: NOT recommended, kept for measurement"),
    "memory-manager": ({"FLASHNEXT_MEMORY_MANAGER": "1", "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL": "0"},
                       "adaptive pool sizing and release; no real-model run yet (F2Bm pending)"),
    "prefix-profiles": ({"FLASHNEXT_PREFIX_PROFILES": "1"},
                        "frequency-aware pinning and SSD eviction for system prompts; offline-tested only"),
    "auto-reclaim": ({"FLASHNEXT_AUTO_MEMORY_RECLAIM": "1"},
                     "request-to-request memory reclamation; offline-tested only"),
    "short-pool-reuse": ({"FLASHNEXT_SHORT_PREFILL_POOL_REUSE": "1"},
                         "reuse resident pool experts in short prefills; replay predicts under 5% hits"),
}
OWNED = frozenset({*COMMON, *PREFILL_OFF, *AGENTIC_PREFILL, *EXPERIMENTAL_OFF,
                   "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB", "FLASHNEXT_PREFILL_WINDOW_TOKENS", "FLASHNEXT_KV_GROUP_SIZE", "FLASHNEXT_KV_BITS",
                   "FLASHNEXT_EXPERT_POOL_GB", "FLASHNEXT_EXPERT_POOL_AUTO", "FLASHNEXT_PREFIX_CACHE_DIR", "FLASHNEXT_PREFIX_CACHE_RAM_GB",
                   "FLASHNEXT_PREFIX_CACHE_DISK_GB", "FLASHNEXT_PREFIX_PROFILES_MAX_GB",
                   "FLASHNEXT_PREFIX_PROFILES_FREE_RESERVE_GB", "FLASHNEXT_WATCHDOG"})


class ModeError(ValueError):
    """The requested mode or option is not allowed."""


@dataclass(frozen=True)
class ModePlan:
    mode: str
    profile: str
    serve: bool
    env: dict
    pool_gb: float
    pool_reason: str
    context_limit: int | None
    enabled: tuple = ()
    warnings: tuple = ()
    prefix_dir: str | None = None
    prefix_disk_gb: float = 0.0
    labels: dict = field(default_factory=dict)
    pool_auto: bool = False
    context_required: bool = False      # the context limit was stated by the owner: admission may not settle for less
    max_cold_prefill: int | None = None
    max_incremental_prefill: int | None = None


def choose_pool_gb(requested, available_gb=None):
    """Return ``(gb, reason)``. ``None`` or ``"auto"``: up to 2 GB, decided at the first decode from real headroom.
    A number: 0, or a fixed size up to 2 GB allocated without that check. Never above the cap."""
    if requested is not None and str(requested).lower() != "auto":
        try:
            gb = float(requested)
        except ValueError:
            raise ModeError(f"pool size must be 'auto' or a number of GB, not {requested!r}") from None
        if gb < 0:
            raise ModeError("pool size must not be negative")
        if gb > POOL_CAP_GB:
            raise ModeError(f"pool size {gb:g} GB is above the {POOL_CAP_GB:g} GB cap; a larger pool needs a "
                            "validated memory budget (the recorded 6 GB arms ran near the 3 GiB guard line)")
        if gb == 0:
            return 0.0, "requested: no decode pool (SSD streaming, memory-safe; recorded 3.42-3.47 tok/s)"
        reason = ("fixed, allocated with no headroom check: the first real chat at a fixed 2 GB stopped the "
                  "compressor guard (2.94 GiB available) right after the pool was created")
        if gb < 2.0:
            reason += "; under 2 GB the recorded replay showed LRU hit rates near zero, so this may only cost RAM"
        return gb, reason
    if available_gb is not None and available_gb < START_GATE_GB:
        return 0.0, f"available memory {available_gb:.1f} GB is under the {START_GATE_GB:g} GB start gate"
    return DEFAULT_POOL_GB, ("auto: a 2 GB pool only if available memory covers it plus the 3 GiB guard floor and "
                             "0.5 GiB margin when the first token is decoded, otherwise SSD streaming without a pool")


def _free_disk_gb(directory) -> float:
    path = Path(directory).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def _positive(value, name):
    if value is None:
        return None
    if not isinstance(value, int) or value < 1:
        raise ModeError(f"{name} must be a positive number of tokens")
    return value


def cache_quota_gb(environ=None) -> float:
    """SSD cache quota in GB: PREFIX_DISK_GB unless VEYSTRIX_CACHE_GB holds a positive number."""
    raw = (os.environ if environ is None else environ).get(QUOTA_ENV)
    try:
        value = float(raw) if raw not in (None, "") else PREFIX_DISK_GB
    except ValueError:
        return PREFIX_DISK_GB
    return value if value > 0 else PREFIX_DISK_GB


def plan_mode(mode, *, pool_gb=None, enable=(), available_gb=None, prefix_dir=None, free_disk_gb=None,
              context_limit=None, serve=None, max_cold_prefill=None, max_incremental_prefill=None):
    """Build the plan for a mode. ``serve`` defaults to True for AGENTIC (local model server)."""
    if mode not in MODES:
        raise ModeError(f"unknown mode {mode!r}; choose chat or agentic")
    enable = tuple(dict.fromkeys(enable))
    for name in enable:
        if name not in EXPERIMENTAL:
            raise ModeError(f"unknown experimental feature {name!r}; choose from {', '.join(EXPERIMENTAL)}")
    if len({"layer-major", "layer-major-4k", "layer-major-hybrid-4k"} & set(enable)) > 1:
        raise ModeError("choose only one layer-major window size")
    if {"uncached-reads", "uncached-prefill-reads"} <= set(enable):
        raise ModeError("choose one of uncached-reads (all phases) and uncached-prefill-reads (prefill only)")
    if "layer-prefetch" in enable and not {"layer-major", "layer-major-4k", "layer-major-hybrid-4k"} & set(enable):
        raise ModeError("layer-prefetch needs a layer-major mode (layer-major, layer-major-4k or layer-major-hybrid-4k)")
    if mode == CHAT and "prefix-profiles" in enable:
        raise ModeError("prefix-profiles needs the AGENTIC prefix cache")
    if mode == CHAT and "long-context" in enable:
        raise ModeError("long-context is an AGENTIC feature (CHAT keeps one-call prefill and its reference numerics)")
    if mode == CHAT and {"layer-major", "layer-major-4k", "layer-major-hybrid-4k"} & set(enable):
        raise ModeError("layer-major is an AGENTIC feature (needs watchdog supervision and agentic mode)")
    if len({"kv-affine4", "kv-affine8"} & set(enable)) > 1:
        raise ModeError("choose one quantized KV mode: kv-affine4 or kv-affine8")
    gb, reason = choose_pool_gb(pool_gb, available_gb)
    auto = (pool_gb is None or str(pool_gb).lower() == "auto") and gb > 0
    warnings = []
    env = {**COMMON, **PREFILL_OFF, **EXPERIMENTAL_OFF, "FLASHNEXT_EXPERT_POOL_GB": f"{gb:g}",
           "FLASHNEXT_EXPERT_POOL_AUTO": "1" if auto else "0"}
    directory, disk_gb = None, 0.0
    if mode == AGENTIC:
        env.update(AGENTIC_PREFILL)
        env["FLASHNEXT_WATCHDOG"] = "1"      # out-of-process stall watchdog (AGENTIC serves clients that cannot wait forever)
        directory = os.path.expanduser(prefix_dir or PREFIX_DIR)
        free = _free_disk_gb(directory) if free_disk_gb is None else float(free_disk_gb)
        disk_gb = min(cache_quota_gb(), max(0.0, free - DISK_RESERVE_GB))
        if disk_gb < MIN_PREFIX_DISK_GB:
            # A zero SSD budget means "unlimited" to the cache, so the cache is switched off instead.
            env["FLASHNEXT_PREFIX_CACHE"] = "0"
            warnings.append(f"prefix cache OFF: only {free:.0f} GB free on the cache volume, "
                            f"{DISK_RESERVE_GB:g} GB is kept in reserve")
            directory, disk_gb = None, 0.0
        else:
            env.update({"FLASHNEXT_PREFIX_CACHE_DIR": directory,
                        "FLASHNEXT_PREFIX_CACHE_RAM_GB": f"{PREFIX_RAM_GB:g}",
                        "FLASHNEXT_PREFIX_CACHE_DISK_GB": f"{disk_gb:g}",
                        "FLASHNEXT_PREFIX_PROFILES_MAX_GB": f"{disk_gb:g}",
                        "FLASHNEXT_PREFIX_PROFILES_FREE_RESERVE_GB": f"{DISK_RESERVE_GB:g}"})
    labels = {}
    for name in enable:
        flags, label = EXPERIMENTAL[name]
        env.update(flags)
        labels[name] = label
    if "prefix-profiles" in enable and env["FLASHNEXT_PREFIX_CACHE"] != "1":
        env["FLASHNEXT_PREFIX_PROFILES"] = "0"
        warnings.append("prefix-profiles ignored because the prefix cache is off")
    default_limit = LONG_CONTEXT_LIMIT if "long-context" in enable else AGENTIC_CONTEXT_LIMIT
    cold = _positive(max_cold_prefill, "max cold prefill")
    incremental = _positive(max_incremental_prefill, "max incremental prefill")
    limit = context_limit if context_limit is not None else (default_limit if mode == AGENTIC else None)
    if mode == AGENTIC:
        explicit_cold = cold or max_cold_prefill_from_environment()
        explicit_incremental = incremental or max_incremental_prefill_from_environment()
        if limit and explicit_cold and explicit_cold > limit:
            raise ModeError(f"max cold prefill {explicit_cold} exceeds the context limit {limit}")
        if limit and explicit_incremental and explicit_incremental > limit:
            raise ModeError(f"max incremental prefill {explicit_incremental} exceeds the context limit {limit}")
        # Defaults never exceed the window (a smaller --context-limit lowers them); explicit values are honoured or refused.
        cold = explicit_cold or min(DEFAULT_MAX_COLD_PREFILL, limit or DEFAULT_MAX_COLD_PREFILL)
        incremental = explicit_incremental or min(DEFAULT_MAX_INCREMENTAL_PREFILL, limit or DEFAULT_MAX_INCREMENTAL_PREFILL)
    if limit is not None and limit < 1:
        raise ModeError("context limit must be a positive number of tokens")
    if mode == AGENTIC and cold and cold > VALIDATED_PREFILL_TOKENS:
        warnings.append(f"max cold prefill {cold} is above the {VALIDATED_PREFILL_TOKENS}-token prefill that has real-checkpoint "
                        "evidence; it is implemented and unit-tested only. Raise it step by step and watch swap growth.")
    return ModePlan(mode, PROFILE[mode], mode == AGENTIC if serve is None else bool(serve), env, gb, reason, limit,
                    enable, tuple(warnings), directory, disk_gb, labels, auto,
                    context_required=context_limit is not None or mode == AGENTIC,
                    max_cold_prefill=cold, max_incremental_prefill=incremental)


def apply_mode(environment: dict, plan: ModePlan) -> dict:
    """Write the plan into ``environment``. Owned keys the plan does not set are removed (their default is off)."""
    for key in OWNED:
        environment.pop(key, None)
    environment.update(plan.env)
    environment["VEYSTRIX_MODE"] = plan.mode
    environment["VEYSTRIX_ENABLE"] = ",".join(plan.enabled)
    if plan.context_limit:
        environment["VEYSTRIX_CONTEXT_LIMIT"] = str(plan.context_limit)
    else:
        environment.pop("VEYSTRIX_CONTEXT_LIMIT", None)
    if plan.context_limit and plan.context_required:
        environment["VEYSTRIX_CONTEXT_REQUIRED"] = "1"       # read by prefill_readiness.required_context
    else:
        environment.pop("VEYSTRIX_CONTEXT_REQUIRED", None)   # an inherited value must not turn a default into a requirement
    if plan.mode == AGENTIC:
        environment["VEYSTRIX_MAX_COLD_PREFILL"] = str(plan.max_cold_prefill or DEFAULT_MAX_COLD_PREFILL)
        environment["VEYSTRIX_MAX_INCREMENTAL_PREFILL"] = str(plan.max_incremental_prefill or DEFAULT_MAX_INCREMENTAL_PREFILL)
    else:
        environment.pop("VEYSTRIX_MAX_COLD_PREFILL", None)
        environment.pop("VEYSTRIX_MAX_INCREMENTAL_PREFILL", None)
    return environment


def plan_from_environment(environ=None):
    """Rebuild the plan a launcher applied, from the process environment itself (what actually runs).

    Returns None when no mode was applied (a direct ``session.py`` or benchmark launch)."""
    environ = os.environ if environ is None else environ
    mode = environ.get("VEYSTRIX_MODE")
    if mode not in MODES:
        return None
    env = {key: environ[key] for key in OWNED if key in environ}
    enabled = tuple(name for name in environ.get("VEYSTRIX_ENABLE", "").split(",") if name in EXPERIMENTAL)
    try:
        gb = float(env.get("FLASHNEXT_EXPERT_POOL_GB", "0") or 0)
    except ValueError:
        gb = 0.0
    try:
        disk = float(env.get("FLASHNEXT_PREFIX_CACHE_DISK_GB", "0") or 0)
    except ValueError:
        disk = 0.0
    auto = env.get("FLASHNEXT_EXPERT_POOL_AUTO") == "1"
    return ModePlan(mode, PROFILE[mode], mode == AGENTIC, env, gb,
                    choose_pool_gb("auto")[1] if auto else "from the launch environment",
                    context_limit_from_environment(environ), enabled, (),
                    env.get("FLASHNEXT_PREFIX_CACHE_DIR"), disk,
                    {name: EXPERIMENTAL[name][1] for name in enabled}, auto,
                    context_required=environ.get("VEYSTRIX_CONTEXT_REQUIRED") == "1",
                    max_cold_prefill=max_cold_prefill_from_environment(environ),
                    max_incremental_prefill=max_incremental_prefill_from_environment(environ))


def context_limit_from_environment(environ=None):
    raw = (os.environ if environ is None else environ).get("VEYSTRIX_CONTEXT_LIMIT")
    try:
        value = int(raw) if raw else None
    except ValueError:
        return None
    return value if value and value > 0 else None


def max_cold_prefill_from_environment(environ=None):
    """Largest single cold prefill (tokens) the admitted memory tier allows; None means no tier limit."""
    raw = (os.environ if environ is None else environ).get("VEYSTRIX_MAX_COLD_PREFILL")
    try:
        value = int(raw) if raw else None
    except ValueError:
        return None
    return value if value and value > 0 else None


def max_incremental_prefill_from_environment(environ=None):
    """Largest single uncached incremental prefill (tokens) allowed; None means the default (32768)."""
    raw = (os.environ if environ is None else environ).get("VEYSTRIX_MAX_INCREMENTAL_PREFILL")
    try:
        value = int(raw) if raw else None
    except ValueError:
        return None
    return value if value and value > 0 else None


def banner(plan: ModePlan, *, checkpoint=None, available_gb=None, guard_trace=None, auto_purge=None) -> str:
    """Startup summary: mode, checkpoint, pool, prefix cache, chunk policy, memory readiness, supervisor."""
    env = plan.env
    rows = [f"Veystrix {plan.mode.upper()}  (profile {plan.profile}, one shared Flash-Next runtime)"]
    rows.append(f"  checkpoint     {checkpoint or 'default'}")
    pool = ("auto, up to " if plan.pool_auto else "") + f"{plan.pool_gb:g} GB" if plan.pool_gb else "off"
    rows.append(f"  decode pool    {pool}  ({plan.pool_reason})")
    if env.get("FLASHNEXT_PREFIX_CACHE") == "1":
        profiles = "profiles ON" if env.get("FLASHNEXT_PREFIX_PROFILES") == "1" else "profiles off"
        rows.append(f"  prefix cache   exact RAM+SSD, {plan.prefix_dir}, RAM {PREFIX_RAM_GB:g} GB, "
                    f"SSD {plan.prefix_disk_gb:g} GB, free reserve {DISK_RESERVE_GB:g} GB, {profiles}")
    else:
        rows.append("  prefix cache   off" + ("  (live conversation state is reused instead)"
                                              if plan.mode == CHAT else ""))
    if env.get("FLASHNEXT_PREFILL_CHUNK_TOKENS", "0") != "0":
        rows.append(f"  chunk policy   {env.get('FLASHNEXT_PREFILL_CHUNK_POLICY', 'grid')} "
                    f"{env['FLASHNEXT_PREFILL_CHUNK_TOKENS']}-token chunks, {env['FLASHNEXT_PREFILL_RING_EXPERTS']}-expert "
                    "scratch ring")
    else:
        rows.append("  chunk policy   one call per prompt (reference numerics)")
    from models.flashnext import memory_policy
    swap_policy = memory_policy.is_swap_growth_policy({**env, "VEYSTRIX_MODE": plan.mode})
    if swap_policy:
        if available_gb is not None and available_gb >= START_GATE_GB:
            ready = (
                f"READY ({available_gb:.1f} GB available; start gate waived under swap-growth-only policy; "
                "stops only at +2.5 GiB swap growth)"
            )
        else:
            ready = (
                f"swap-growth-only (start gate waived; "
                + (f"available {available_gb:.1f} GB is telemetry-only; " if available_gb is not None else "")
                + "stops only at +2.5 GiB swap growth)"
            )
    elif available_gb is None:
        ready = "not measured"
    else:
        ready = f"{available_gb:.1f} GB available, start gate {START_GATE_GB:g} GB: " + (
            "READY" if available_gb >= START_GATE_GB else "BELOW GATE")
    rows.append(f"  memory         {ready}")
    if swap_policy:
        guard = "ON (swap growth +2.5 GiB, thermal, battery stops; pressure/available are telemetry)" if env.get("FLASHNEXT_CHAT_GUARD") == "1" else "OFF"
    else:
        guard = "ON (memory, swap, thermal, battery stops)" if env.get("FLASHNEXT_CHAT_GUARD") == "1" else "OFF"
    rows.append(f"  supervisor     {guard}" + (f"  trace {guard_trace}" if guard_trace else ""))
    if auto_purge is not None:
        rows.append("  auto purge   " + ("ON  (sudo -n /usr/sbin/purge only when memory is short and the engine is idle)"
                                         if auto_purge else "off"))
    if plan.context_limit:
        rows.append(f"  context limit  {plan.context_limit} tokens (requests above it are refused, never truncated)")
    if plan.mode == AGENTIC:
        rows.append(f"  prefill limits cold {plan.max_cold_prefill or DEFAULT_MAX_COLD_PREFILL}, uncached increment "
                    f"{plan.max_incremental_prefill or DEFAULT_MAX_INCREMENTAL_PREFILL} tokens per request (admission ceilings; "
                    "internal chunks stay 2048; real-checkpoint evidence to 16384 only)")
    for name, label in plan.labels.items():
        rows.append(f"  EXPERIMENTAL   {name}: {label}")
    for warning in plan.warnings:
        rows.append(f"  warning        {warning}")
    if plan.mode == AGENTIC:
        rows.append("  status         exact system-prompt cache and 2K chunked prefill: real-model evidence; "
                    "DeepSeek Harness: protocol verified offline, no full-model run has passed yet (the first stopped in cold prefill); "
                    "10K: NOT verified")
    return "\n".join(rows)
