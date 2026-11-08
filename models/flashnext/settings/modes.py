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
PREFIX_DISK_GB = 20.0
DISK_RESERVE_GB = 50.0
MIN_PREFIX_DISK_GB = 1.0
# The 2K path has real-model evidence; 4K is a budget estimate (about 7.1 GB peak, calibrated at 1K only).
VALIDATED_PROMPT_TOKENS = 2048
AGENTIC_CONTEXT_LIMIT = 8192
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
    "FLASHNEXT_PREFIX_CACHE_BOUNDARIES": "system",
}
EXPERIMENTAL_OFF = {
    "FLASHNEXT_KV_CACHE_DTYPE": "bf16",             # bf16 KV is the reference; quantized KV is an explicit opt-in below
    "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS": "0",
    "FLASHNEXT_MEMORY_MANAGER": "0",
    "FLASHNEXT_PREFIX_PROFILES": "0",
    "FLASHNEXT_AUTO_MEMORY_RECLAIM": "0",
    "FLASHNEXT_SHORT_PREFILL_POOL_REUSE": "0",
}
# Offline-tested only. Each is explicit (--enable NAME), never a default, and labelled in the banner.
EXPERIMENTAL = {
    "long-context": ({"FLASHNEXT_PREFILL_CHUNK_POLICY": "budgeted", "FLASHNEXT_PREFILL_CHUNK_TOKENS": "2048",
                      "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB": "1000", "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS": "1"},
                     "deterministic memory-budgeted chunk plan (2048/1024/512/256) with live step-down and a 'long' "
                     "admission tier (16K context, 14K cold prefill); offline-tested only, no real run above 2K"),
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
                   "FLASHNEXT_PREFILL_CHUNK_BUDGET_MB", "FLASHNEXT_KV_GROUP_SIZE", "FLASHNEXT_KV_BITS",
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


def plan_mode(mode, *, pool_gb=None, enable=(), available_gb=None, prefix_dir=None, free_disk_gb=None,
              context_limit=None, serve=None):
    """Build the plan for a mode. ``serve`` defaults to True for AGENTIC (local model server)."""
    if mode not in MODES:
        raise ModeError(f"unknown mode {mode!r}; choose chat or agentic")
    enable = tuple(dict.fromkeys(enable))
    for name in enable:
        if name not in EXPERIMENTAL:
            raise ModeError(f"unknown experimental feature {name!r}; choose from {', '.join(EXPERIMENTAL)}")
    if mode == CHAT and "prefix-profiles" in enable:
        raise ModeError("prefix-profiles needs the AGENTIC prefix cache")
    if mode == CHAT and "long-context" in enable:
        raise ModeError("long-context is an AGENTIC feature (CHAT keeps one-call prefill and its reference numerics)")
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
        disk_gb = min(PREFIX_DISK_GB, max(0.0, free - DISK_RESERVE_GB))
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
    limit = context_limit if context_limit is not None else (default_limit if mode == AGENTIC else None)
    if limit is not None and limit < 1:
        raise ModeError("context limit must be a positive number of tokens")
    if mode == AGENTIC and limit and limit > VALIDATED_PROMPT_TOKENS:
        warnings.append(f"context limit {limit} is above the {VALIDATED_PROMPT_TOKENS}-token path that has "
                        "real-model evidence; 4K, 8K and 10K are staged, unvalidated")
    return ModePlan(mode, PROFILE[mode], mode == AGENTIC if serve is None else bool(serve), env, gb, reason, limit,
                    enable, tuple(warnings), directory, disk_gb, labels, auto, context_required=context_limit is not None)


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
                    context_required=environ.get("VEYSTRIX_CONTEXT_REQUIRED") == "1")


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
    if available_gb is None:
        ready = "not measured"
    else:
        ready = f"{available_gb:.1f} GB available, start gate {START_GATE_GB:g} GB: " + (
            "READY" if available_gb >= START_GATE_GB else "BELOW GATE")
    rows.append(f"  memory         {ready}")
    guard = "ON (memory, swap, thermal, battery stops)" if env.get("FLASHNEXT_CHAT_GUARD") == "1" else "OFF"
    rows.append(f"  supervisor     {guard}" + (f"  trace {guard_trace}" if guard_trace else ""))
    if auto_purge is not None:
        rows.append("  auto purge   " + ("ON  (sudo -n /usr/sbin/purge only when memory is short and the engine is idle)"
                                         if auto_purge else "off"))
    if plan.context_limit:
        rows.append(f"  context limit  {plan.context_limit} tokens (requests above it are refused, never truncated)")
    for name, label in plan.labels.items():
        rows.append(f"  EXPERIMENTAL   {name}: {label}")
    for warning in plan.warnings:
        rows.append(f"  warning        {warning}")
    if plan.mode == AGENTIC:
        rows.append("  status         exact system-prompt cache and 2K chunked prefill: real-model evidence; "
                    "DeepSeek Harness: protocol verified offline, no full-model run has passed yet (the first stopped in cold prefill); "
                    "10K: NOT verified")
    return "\n".join(rows)
