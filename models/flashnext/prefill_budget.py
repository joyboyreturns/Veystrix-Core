"""Long-prompt MLX memory model for the 16 GB M4 (production module; formerly only in tests/bench).

The scheduler and prefill planner import this, so it must not live under tests. Every term is tagged below.

Original header: Projected MLX memory for a long prompt (offline, no model).

Every term is tagged: ``structural`` comes from the config and the code,
``measured`` from a real run on this machine, ``estimate`` is arithmetic on
those. Nothing here is a measurement of a 2K+ prompt. Run it to print the table:

    python -m models.flashnext.tests.bench.long_context_budget
"""
from __future__ import annotations

from dataclasses import dataclass

GB = 1e9
METAL_LIMIT = 12_713_115_648          # measured: max recommended working set
GUARD_HEADROOM = 500_000_000          # the child guard stops within 0.5 GB of the limit

# Model shape (config.json, checkpoint qwen4_exp).
HIDDEN, HC = 2560, 4
LAYERS, FULL_LAYERS = 48, 12
LINEAR_LAYERS = LAYERS - FULL_LAYERS
CONV_DIM = 2 * 16 * 128 + 48 * 128                      # 10,240
GDN_STATE_BYTES = 48 * 128 * 128 * 4 + 3 * CONV_DIM * 2  # fp32 recurrent state + conv state
KV_BYTES_PER_TOKEN = 2 * 2 * 256 * 2                    # K and V, 2 kv heads, head_dim 256, bf16
INDEX_BYTES_PER_TOKEN = 128 * 2 + 12                    # indexer key bf16 + MRoPE position ids
VOCAB = 248_320
LOGITS_FULL_MAX_TOKENS = 2048                           # prefill.py: min(QSA threshold, 8192)

# Measured on the first-turn F2 runs (results of 2026-11-07).
WEIGHTS = 2_794_742_880                                 # MLX active after load
SCRATCH_RING = 1_622_032_384                            # 512-expert scratch pool as mapped
DECODE_POOL = 2_000_000_000
ALLOCATOR_CACHE_PEAK = 1_350_000_000                    # steady 1.31 GB, peak 1.35 GB at 1K

# Fit of end-of-prefill active memory minus the weights, from real runs at
# 129, 257 and 1,025 tokens: 0.233, 0.345 and 0.992 GB.
RETAINED_INTERCEPT = 0.12 * GB                          # about the GatedDeltaNet state
RETAINED_SLOPE = 0.85e6                                 # bytes per token, mechanism: see linear_state.py

# Per-token prefill transients. Hyper-connection streams: [tokens, 4 x 2560] bf16
# with about 8 live copies (estimate). MoE: ring-arm working memory above the pool
# was 0.19 GB per 1K tokens (synthetic, MoE only).
HYPER_TRANSIENT = HIDDEN * HC * 2 * 8
MOE_TRANSIENT = 190_000
SDPA_PAIR_BYTES = 24 * 2                                # unfused scores: 24 heads x bf16 per (q, k) pair
QSA_QUERY_CHUNK = 2048                                  # qsa_chunk.py


@dataclass(frozen=True)
class Budget:
    tokens: int
    weights: float
    scratch_ring: float
    retained_as_is: float
    retained_fixed: float
    state_kv: float
    logits: float
    transient: float
    attention_scores: float
    allocator_cache: float

    def peak(self, fixed: bool) -> float:
        retained = self.retained_fixed if fixed else self.retained_as_is
        return (self.weights + self.scratch_ring + retained + self.state_kv + self.logits
                + self.transient + self.attention_scores + self.allocator_cache)

    def fits(self, fixed: bool) -> bool:
        return self.peak(fixed) <= METAL_LIMIT - GUARD_HEADROOM


def state_and_kv(tokens: int) -> float:
    """Persistent cache after prefill: GatedDeltaNet state plus KV and indexer keys."""
    return (LINEAR_LAYERS * GDN_STATE_BYTES
            + FULL_LAYERS * tokens * (KV_BYTES_PER_TOKEN + INDEX_BYTES_PER_TOKEN))


def logits_bytes(tokens: int) -> float:
    return tokens * VOCAB * 2 if tokens <= LOGITS_FULL_MAX_TOKENS else 0.0


def attention_score_bytes(tokens: int, query_chunk: int = QSA_QUERY_CHUNK) -> float:
    """Unfused SDPA scores for one query chunk against all keys (measured scaling 48 B per pair)."""
    return min(tokens, query_chunk) * tokens * SDPA_PAIR_BYTES


def budget(tokens: int, chunk: int = 0) -> Budget:
    """``chunk`` > 0 models FLASHNEXT_PREFILL_CHUNK_TOKENS: per-token transients, attention scores and
    retained prefill inputs are bounded by the chunk (each chunk evaluates the cache states and frees its
    temporaries), while the persistent KV and state still grow with all ``tokens``."""
    persistent = state_and_kv(tokens)
    live = min(tokens, chunk) if chunk else tokens
    if chunk:
        return Budget(
            tokens=tokens, weights=WEIGHTS, scratch_ring=SCRATCH_RING,
            retained_as_is=persistent, retained_fixed=persistent, state_kv=0.0,
            logits=0.0,                                     # the last chunk uses the fused head above 2,048
            transient=(HYPER_TRANSIENT + MOE_TRANSIENT) * live,
            attention_scores=min(live, QSA_QUERY_CHUNK) * tokens * SDPA_PAIR_BYTES,
            allocator_cache=ALLOCATOR_CACHE_PEAK)
    return Budget(
        tokens=tokens,
        weights=WEIGHTS,
        scratch_ring=SCRATCH_RING,
        retained_as_is=RETAINED_INTERCEPT + RETAINED_SLOPE * tokens,
        retained_fixed=persistent,
        state_kv=0.0,       # the as-is retained fit already contains the persistent state
        logits=logits_bytes(tokens),
        transient=(HYPER_TRANSIENT + MOE_TRANSIENT) * tokens,
        attention_scores=attention_score_bytes(tokens),
        allocator_cache=ALLOCATOR_CACHE_PEAK,
    )




# ---- incremental (in-run) model -------------------------------------------------------------------------
# While a prefill is running the weights, the scratch ring and the allocator are ALREADY resident, so the host
# "available" figure already excludes them. The question for the next chunk is only what it ADDS.
PER_TOKEN_TRANSIENT = HYPER_TRANSIENT + MOE_TRANSIENT      # estimate, bytes per live token (353,840)
KV_BYTES_PER_TOKEN_TOTAL = FULL_LAYERS * (KV_BYTES_PER_TOKEN + INDEX_BYTES_PER_TOKEN)   # bf16 KV + indexer, all layers


def chunk_increment_bytes(chunk: int, position: int, kv_bytes_per_token: float | None = None) -> int:
    """Estimated extra MLX memory (above what is resident before the chunk) while a chunk of ``chunk`` tokens runs
    with ``position`` tokens already cached. ``estimate``: transient per token (hyper-connection streams and MoE),
    unfused attention scores (24 heads x bf16 per query/key pair, the QSA mask bounds queries at 2,048), and the
    KV the chunk appends. The 48 B/pair term is measured (see SDPA_PAIR_BYTES); the per-token transient is not."""
    chunk, position = int(chunk), int(position)
    queries = min(chunk, QSA_QUERY_CHUNK)
    scores = queries * (position + chunk) * SDPA_PAIR_BYTES
    kv = chunk * (KV_BYTES_PER_TOKEN_TOTAL if kv_bytes_per_token is None else kv_bytes_per_token)
    return int(chunk * PER_TOKEN_TRANSIENT + scores + kv)
