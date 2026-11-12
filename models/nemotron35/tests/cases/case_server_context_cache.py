from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_context_cache"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint]


TEST = TestSpec(
    id="server-context-cache", title="Cold and warm 8K and 32K server requests", category="diagnostic",
    explanation="Runs one cold request and one identical warm replay at each context length.",
    why="Measures prefill, decode, cache reuse and swap with a requested expert pool.",
    script=command, metrics=COMMON_METRICS, promotion=False,
    controls={"sampling": "greedy", "output_cap": "32", "pool_gb": "2", "swap_ceiling_mib": "from environment (3072 by default)"},
    source="models/flashnext/tests/bench/bench_server_context_cache.py",
)
