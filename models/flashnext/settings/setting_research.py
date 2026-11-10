"""Internal and research-only runtime switches used by benchmarks."""
from __future__ import annotations

import os

from macqwen.backend_settings import Setting


def env(key, default="0"):
    return lambda _backend: os.environ.get(key, default)


def flag(key):
    return lambda _backend: os.environ.get(key, "0") == "1"


def text(raw):
    return raw


def integer(raw):
    return int(raw)


def positive_integer(raw):
    value = int(raw)
    if value <= 0:
        raise ValueError("value must be positive")
    return value


def choice(options, name):
    def parse(raw):
        if raw not in options:
            raise ValueError(f"{name} must be one of: {', '.join(options)}")
        return raw
    return parse


def norm_convention(raw):
    value = str(raw).strip().lower()
    if value in {"", "auto"}:
        return ""
    if value in {"one", "zero"}:
        return value
    raise ValueError("norm-convention must be auto, one, or zero")


def research(name, key, default="0", lifecycle="startup", parser=text, unset=None):
    """``default`` is the chat default; ``unset`` is what the runtime does
    when the variable is absent, when the two differ."""
    return Setting(
        name, (key,), default, parser, lifecycle, "research", "research-only", "flashnext",
        env(key, default if unset is None else unset), None, flag(key), None, key, __file__,
    )


SETTINGS = (
    research("prewarm", "FLASHNEXT_PREWARM", lifecycle="startup"),
    research("slab-min-slots", "FLASHNEXT_SLAB_MIN_SLOTS", "4", "startup", integer),
    research("slab-max-slots", "FLASHNEXT_SLAB_MAX_SLOTS", "6", "startup", integer),
    research("slab-num-layers", "FLASHNEXT_SLAB_NUM_LAYERS", "12", "startup", integer),
    research("slab-pack-require-existing", "FLASHNEXT_SLAB_PACK_REQUIRE_EXISTING", "0", "startup"),
    research("slab-pack-expected-path", "FLASHNEXT_SLAB_PACK_EXPECTED_PATH", "", "startup"),
    research("swap-resident", "FLASHNEXT_SWAP_RESIDENT"),
    research("track-resident", "FLASHNEXT_TRACK_RESIDENT"),
    research("pin-parts", "FLASHNEXT_PIN_PARTS", "all"),
    research("shared-read-buffer", "FLASHNEXT_SHARED_READ_BUFFER"),
    research("compile", "FLASHNEXT_COMPILE"),
    research("qsa-cache-pooled-keys", "FLASHNEXT_QSA_CACHE_POOLED_KEYS", "1", parser=choice(("0", "1"), "qsa-cache-pooled-keys"), unset="0"),
    research("qsa-scatter-decode", "FLASHNEXT_QSA_SCATTER_DECODE", "1", parser=choice(("0", "1"), "qsa-scatter-decode"), unset="0"),
    research("overlap", "FLASHNEXT_OVERLAP", "0", parser=choice(("0", "1"), "overlap"), unset="1"),
    research("expert-lock-gb", "FLASHNEXT_EXPERT_LOCK_GB", "0", "startup", text),
    research("rdahead", "FLASHNEXT_RDAHEAD", "0", parser=choice(("0", "1"), "rdahead"), unset="1"),
    research("wired-gb", "FLASHNEXT_WIRED_GB", "0", "startup", float),
    research("swap-max-rows", "FLASHNEXT_SWAP_MAX_ROWS", "4", "startup", integer),
    research("profile-io", "FLASHNEXT_PROFILE_IO"),
    research("prefill-trace-layers", "FLASHNEXT_PREFILL_TRACE_LAYERS", "0", "startup",
             choice(("0", "1"), "prefill-trace-layers"), unset="0"),
    research("profile-boundaries", "FLASHNEXT_PROFILE_BOUNDARIES"),
    research("profile-boundary", "FLASHNEXT_PROFILE_BOUNDARY", ""),
    research("stream-pack-chunk", "FLASHNEXT_STREAM_PACK_CHUNK", "2", "startup", integer, unset="0"),
    research("physical-miss-trace", "FLASHNEXT_PHYSICAL_MISS_TRACE", "0", "startup"),
    research("physical-miss-profile", "FLASHNEXT_PHYSICAL_MISS_PROFILE", "~/.cache/flashnext/physical-misses.json", "startup"),
    research("physical-miss-min-samples", "FLASHNEXT_PHYSICAL_MISS_MIN_SAMPLES", "1", "startup", integer),
    research("gpu-keepwarm-iters", "FLASHNEXT_GPU_KEEPWARM_ITERS", "60000", "startup", integer),
    research("gpu-keepwarm-period-ms", "FLASHNEXT_GPU_KEEPWARM_PERIOD_MS", "0.5", "startup", float),
    research("gpu-keepwarm-stream-pack", "FLASHNEXT_GPU_KEEPWARM_STREAM_PACK", "1", "startup"),
    research("small-sidecar", "FLASHNEXT_SMALL_SIDECAR", "", "startup"),
    research("one-sync", "FLASHNEXT_ONE_SYNC", "1", parser=choice(("0", "1"), "one-sync"), unset="0"),
    research("compile-gdn", "FLASHNEXT_COMPILE_GDN", "1", parser=choice(("0", "1"), "compile-gdn"), unset="0"),
    research("expert-pool-gb", "FLASHNEXT_EXPERT_POOL_GB", "6", "startup", float, unset="0"),
    research("expert-pool-lock", "FLASHNEXT_EXPERT_POOL_LOCK"),
    research("prefill-prefetch-gb", "FLASHNEXT_PREFILL_PREFETCH_GB", "0", "startup", float, unset="0"),
    research("prefill-ring-experts", "FLASHNEXT_PREFILL_RING_EXPERTS", "0", "startup", integer, unset="0"),
    research("prefill-ring-min-tokens", "FLASHNEXT_PREFILL_RING_MIN_TOKENS", "512", "startup", integer, unset="512"),
    research("prefill-ring-margin-gb", "FLASHNEXT_PREFILL_RING_MARGIN_GB", "0.5", "startup", float, unset="0.5"),
    research("prefill-ring-transient-gb", "FLASHNEXT_PREFILL_RING_TRANSIENT_GB", "2.0", "startup", float, unset="2.0"),
    research("prefill-ring-scratch", "FLASHNEXT_PREFILL_RING_SCRATCH", "0", "startup", choice(("0", "1"), "prefill-ring-scratch"), unset="0"),
    research("prefill-cache-limit-mb", "FLASHNEXT_PREFILL_CACHE_LIMIT_MB", "0", "startup", integer, unset="0"),
    research("prefix-cache", "FLASHNEXT_PREFIX_CACHE", "1", parser=choice(("0", "1"), "prefix-cache"), unset="0"),
    research("prefix-cache-dir", "FLASHNEXT_PREFIX_CACHE_DIR"),
    research("prefix-cache-ram-gb", "FLASHNEXT_PREFIX_CACHE_RAM_GB", "0.75", "startup", float, unset="0.75"),
    research("prefix-cache-disk-gb", "FLASHNEXT_PREFIX_CACHE_DISK_GB", "20", "startup", float, unset="20"),
    research("disk-reserve-gb", "FLASHNEXT_DISK_RESERVE_GB", "50", "startup", float, unset="50"),
    research("prefix-cache-min-tokens", "FLASHNEXT_PREFIX_CACHE_MIN_TOKENS", "512", "startup", integer, unset="512"),
    research("prefix-cache-boundaries", "FLASHNEXT_PREFIX_CACHE_BOUNDARIES", "system", "startup", choice(("system", "chunks", "messages", "all"), "prefix-cache-boundaries"), unset="system"),
    research("prefill-chunk-tokens", "FLASHNEXT_PREFILL_CHUNK_TOKENS", "0", "startup", integer, unset="0"),
    research("prefill-chunk-policy", "FLASHNEXT_PREFILL_CHUNK_POLICY", "grid", "startup", choice(("grid", "balanced"), "prefill-chunk-policy"), unset="grid"),
    research("prefill-release-decode-pool", "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL", "1", "startup", choice(("0", "1"), "prefill-release-decode-pool"), unset="0"),
    research("memory-manager", "FLASHNEXT_MEMORY_MANAGER", "1", "startup", choice(("0", "1"), "memory-manager"), unset="0"),
    research("memory-short-uncached", "FLASHNEXT_MEMORY_SHORT_UNCACHED", "1", "startup", choice(("0", "1"), "memory-short-uncached"), unset="0"),
    research("prefix-profiles", "FLASHNEXT_PREFIX_PROFILES", "1", "startup", choice(("0", "1"), "prefix-profiles"), unset="0"),
    research("prefix-profiles-max-gb", "FLASHNEXT_PREFIX_PROFILES_MAX_GB", "10", "startup", float, unset="10"),
    research("prefix-profiles-free-reserve-gb", "FLASHNEXT_PREFIX_PROFILES_FREE_RESERVE_GB", "50", "startup", float, unset="50"),
    research("prefix-profiles-pin-after", "FLASHNEXT_PREFIX_PROFILES_PIN_AFTER", "3", "startup", integer, unset="3"),
    research("prefix-high-value-min-tokens", "FLASHNEXT_HIGH_VALUE_MIN_TOKENS", "512", "startup", integer, unset="512"),
    research("prefix-profiles-half-life-h", "FLASHNEXT_PREFIX_PROFILES_HALF_LIFE_H", "72", "startup", float, unset="72"),
    research("prefix-cache-restore", "FLASHNEXT_PREFIX_CACHE_RESTORE", "0", "startup", choice(("0", "1"), "prefix-cache-restore"), unset="1"),
    research("eval-linear-state", "FLASHNEXT_EVAL_LINEAR_STATE", "1", parser=choice(("0", "1"), "eval-linear-state"), unset="0"),
    research("compile-hc", "FLASHNEXT_COMPILE_HC", "1", parser=choice(("0", "1"), "compile-hc"), unset="0"),
    research("compile-norm", "FLASHNEXT_COMPILE_NORM", "1", parser=choice(("0", "1"), "compile-norm"), unset="0"),
    research("stream-embed", "FLASHNEXT_STREAM_EMBED", "1", "startup", choice(("0", "1"), "stream-embed"), unset="0"),
    research("ngram-parallel-min", "FLASHNEXT_NGRAM_PARALLEL_MIN", "16", "startup", integer, unset="0"),
    research("ngram-dedup", "FLASHNEXT_NGRAM_DEDUP", "0", "startup", choice(("0", "1"), "ngram-dedup"), unset="0"),
    research("io-qos", "FLASHNEXT_IO_QOS", "user-interactive", "startup", choice(("default", "user-interactive", "user-initiated", "utility"), "io-qos"), unset="default"),
    research("norm-weight-cache", "FLASHNEXT_NORM_WEIGHT_CACHE", "1", parser=choice(("0", "1"), "norm-weight-cache"), unset="0"),
    research("slab-counts", "FLASHNEXT_SLAB_COUNTS", "turn", "startup", choice(("turn", "cumulative"), "slab-counts")),
    research("slab-profile", "FLASHNEXT_SLAB_PROFILE", "frozen", "startup", choice(("rolling", "frozen"), "slab-profile"), unset="rolling"),
    research("slab-counts-decay", "FLASHNEXT_SLAB_COUNTS_DECAY", "0.9", "startup", float),
    research("slab-pack-max-age-days", "FLASHNEXT_SLAB_PACK_MAX_AGE_DAYS", "14", "startup", float),
    Setting(
        "metal-g64", ("FLASHNEXT_METAL_G64",), "1", choice(("0", "1"), "metal-g64"),
        "startup", "runtime", "public", "flashnext",
        env("FLASHNEXT_METAL_G64", "1"), None,
        lambda _backend: os.environ.get("FLASHNEXT_METAL_G64", "1") == "1",
        None, "FLASHNEXT_METAL_G64", __file__,
    ),
    Setting(
        "slab-g64", ("FLASHNEXT_SLAB_G64",), "0", choice(("0", "1"), "slab-g64"),
        "startup", "storage", "research-only", "flashnext",
        env("FLASHNEXT_SLAB_G64", "0"), None,
        flag("FLASHNEXT_SLAB_G64"), None, "FLASHNEXT_SLAB_G64", __file__,
    ),
    Setting(
        "norm-convention", ("FLASHNEXT_NORM_CONVENTION",), "", norm_convention,
        "startup", "runtime", "research-only", "flashnext",
        env("FLASHNEXT_NORM_CONVENTION", ""), None,
        lambda _backend: bool(os.environ.get("FLASHNEXT_NORM_CONVENTION")),
        None, "FLASHNEXT_NORM_CONVENTION", __file__,
    ),
    Setting(
        "qsa-dense-mask-max-bytes", ("FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES",),
        512 * 1024 * 1024, positive_integer, "startup", "runtime", "research-only",
        "flashnext",
        lambda _backend: positive_integer(os.environ.get(
            "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES", str(512 * 1024 * 1024)
        )),
        None, lambda _backend: "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES" in os.environ,
        None, "FLASHNEXT_QSA_DENSE_MASK_MAX_BYTES", __file__,
    ),
    Setting(
        "uncache-when-elevated", ("FLASHNEXT_UNCACHE_WHEN_ELEVATED",), "0",
        choice(("0", "1"), "uncache-when-elevated"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_UNCACHE_WHEN_ELEVATED", "0"), None,
        flag("FLASHNEXT_UNCACHE_WHEN_ELEVATED"), None, "FLASHNEXT_UNCACHE_WHEN_ELEVATED", __file__,
    ),
    Setting(
        "layer-prefetch", ("FLASHNEXT_PREFILL_LAYER_PREFETCH",), "0",
        choice(("0", "1"), "layer-prefetch"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_PREFILL_LAYER_PREFETCH", "0"), None,
        flag("FLASHNEXT_PREFILL_LAYER_PREFETCH"), None, "FLASHNEXT_PREFILL_LAYER_PREFETCH", __file__,
    ),
    Setting(
        "uncached-expert-reads", ("FLASHNEXT_UNCACHED_EXPERT_READS",), "0",
        choice(("0", "1", "prefill"), "uncached-expert-reads"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_UNCACHED_EXPERT_READS", "0"), None,
        lambda _backend: os.environ.get("FLASHNEXT_UNCACHED_EXPERT_READS", "0") != "0", None, "FLASHNEXT_UNCACHED_EXPERT_READS", __file__,
    ),
    Setting(
        "pool-buffered-fill", ("FLASHNEXT_EXPERT_POOL_BUFFERED_FILL",), "0",
        choice(("0", "1"), "pool-buffered-fill"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_EXPERT_POOL_BUFFERED_FILL", "0"), None,
        flag("FLASHNEXT_EXPERT_POOL_BUFFERED_FILL"), None, "FLASHNEXT_EXPERT_POOL_BUFFERED_FILL", __file__,
    ),
    Setting(
        "auto-memory-reclaim", ("FLASHNEXT_AUTO_MEMORY_RECLAIM",), "0",
        choice(("0", "1"), "auto-memory-reclaim"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_AUTO_MEMORY_RECLAIM", "0"), None,
        flag("FLASHNEXT_AUTO_MEMORY_RECLAIM"), None, "FLASHNEXT_AUTO_MEMORY_RECLAIM", __file__,
    ),
    Setting(
        "auto-purge", ("FLASHNEXT_AUTO_PURGE",), "0",
        choice(("0", "1"), "auto-purge"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_AUTO_PURGE", "0"), None,
        flag("FLASHNEXT_AUTO_PURGE"), None, "FLASHNEXT_AUTO_PURGE", __file__,
    ),
    Setting(
        "kv-cache-dtype", ("FLASHNEXT_KV_CACHE_DTYPE",), "bf16",
        choice(("bf16", "affine4"), "kv-cache-dtype"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_KV_CACHE_DTYPE", "bf16"), None,
        lambda _backend: os.environ.get("FLASHNEXT_KV_CACHE_DTYPE", "bf16") == "affine4",
        None, "FLASHNEXT_KV_CACHE_DTYPE", __file__,
    ),
    Setting(
        "kv-group-size", ("FLASHNEXT_KV_GROUP_SIZE",), "64",
        integer, "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_KV_GROUP_SIZE", "64"), None,
        lambda _backend: "FLASHNEXT_KV_GROUP_SIZE" in os.environ,
        None, "FLASHNEXT_KV_GROUP_SIZE", __file__,
    ),
    Setting(
        "kv-bits", ("FLASHNEXT_KV_BITS",), "4",
        integer, "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_KV_BITS", "4"), None,
        lambda _backend: "FLASHNEXT_KV_BITS" in os.environ,
        None, "FLASHNEXT_KV_BITS", __file__,
    ),
    Setting(
        "prefill-adaptive-chunks", ("FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS",), "0",
        choice(("0", "1"), "prefill-adaptive-chunks"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS", "0"), None,
        flag("FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS"), None, "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS", __file__,
    ),
    Setting(
        "prefill-order", ("FLASHNEXT_PREFILL_ORDER",), "chunk",
        choice(("chunk", "layer"), "prefill-order"), "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_PREFILL_ORDER", "chunk"), None,
        lambda _backend: os.environ.get("FLASHNEXT_PREFILL_ORDER", "chunk") != "chunk", None,
        "FLASHNEXT_PREFILL_ORDER", __file__,
    ),
    Setting(
        "prefill-window-tokens", ("FLASHNEXT_PREFILL_WINDOW_TOKENS",), "8192",
        integer, "startup", "runtime", "research-only",
        "flashnext", env("FLASHNEXT_PREFILL_WINDOW_TOKENS", "8192"), None,
        lambda _backend: "FLASHNEXT_PREFILL_WINDOW_TOKENS" in os.environ, None,
        "FLASHNEXT_PREFILL_WINDOW_TOKENS", __file__,
    ),
)
