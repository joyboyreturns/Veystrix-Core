"""End-to-end prefill memory and speed on the real checkpoint (not yet approved)."""
from pathlib import Path

from veystrix.testsuite.api import COMMON_METRICS, TestSpec


def command(config, result_path: Path) -> list[str]:
    record = config.output("prefill-memory.jsonl")
    return [str(config.python), "-m", "models.flashnext.tests.bench.bench_prefill_memory",
            "--sizes", "1k", "--rounds", "3", "--record", str(record)]


TEST = TestSpec(
    id="prefill-memory", title="First-turn and later-turn prefill memory, scratch pool and ring",
    category="diagnostic",
    explanation=(
        "Runs fresh child processes for the baseline, the first-turn scratch pool, the "
        "ring in a fresh pool, the allocator cache cap, and the in-pool ring on a second "
        "turn, at 1K, 2K and 4K prompt tokens, in reversed rounds under a safety watchdog."
    ),
    why=(
        "The ring and the scratch pool pass offline tests on synthetic tensors only. This "
        "measures peak memory, SSD reads, compression, swap, speed and exact output on the "
        "real model before any default changes."
    ),
    script=command,
    metrics=COMMON_METRICS + (
        "prefill tokens/s on the tokens actually prefilled", "MLX peak, active and cache memory",
        "SSD bytes read", "compression and swap deltas", "scratch acquire and dispose time",
        "first token and KV-cache hash", "battery, power mode and thermal state",
    ),
    controls={
        "plan": "docs/flashnext/prefetch-integration.md section 14",
        "conditions": "F0-F4 first turn, S0-S2 second turn; flags from the production constants",
        "baseline": "the production chat environment with every opt-in flag unset",
        "processes": "one fresh child per arm; no shared allocator cache or pool",
        "ordering": "three or more rounds; odd rounds reversed; guarded F1 last, 1K only",
        "second turn": "turn 1 runs under baseline flags as setup; flags apply to turn 2 only",
        "sampling": "greedy; closed thinking block; 128 decode tokens",
        "exactness": "first token, KV-cache hash and token digest equal the baseline's",
        "safety stops": "battery below 25%, Low Power Mode, memory pressure, swap, throttling, exit 137",
        "power": "final numbers need AC power and Low Power Mode off; otherwise preliminary",
    },
    source="models/flashnext/tests/bench/bench_prefill_memory.py",
    promotion=False,
    status="deferred",
)
