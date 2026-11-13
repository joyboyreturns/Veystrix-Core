from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_short_paired"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint, "--performance-v2"]


TEST = TestSpec(
    id="performance-v2", title="Paired prompt lookup and prefill options", category="diagnostic",
    explanation="Compares all five opt-in performance options with fresh, guarded servers and exact token digests.",
    why="Measures decode speed, first-text latency and physical reads on a repeated coding workload.",
    script=command, metrics=COMMON_METRICS, promotion=False,
    controls={"sampling": "greedy", "max_tokens": "24", "rounds": "3", "order": "AB, BA, AB"},
    source="models/flashnext/tests/bench/bench_server_short_paired.py",
)
