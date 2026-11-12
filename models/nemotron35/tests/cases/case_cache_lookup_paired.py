from veystrix.testsuite.api import TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("nemotron35", "bench_cache_lookup_paired")]


TEST = TestSpec(
    id="cache-lookup-paired", title="Synthetic 100K prefix lookup", category="diagnostic",
    explanation="Compares exact prefix selection over 100K-token tapes with unrelated namespaces present.",
    why="Isolates metadata and prefix matching from checkpoint load, tensor restoration and decode.",
    script=command, metrics=("cold_lookup_seconds", "warm_lookup_seconds"), promotion=False,
    controls={"fixture": "synthetic metadata only", "rounds": "3"},
    source="models/nemotron35/tests/bench/bench_cache_lookup_paired.py",
)
