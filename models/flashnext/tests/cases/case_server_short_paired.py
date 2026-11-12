from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_short_paired"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint]


TEST = TestSpec(
    id="server-short-paired", title="Paired short server requests", category="diagnostic",
    explanation="Runs fresh baseline and candidate servers in alternating order with isolated caches.",
    why="Measures the single-request path through the production HTTP protocol.",
    script=command, metrics=COMMON_METRICS, promotion=False,
    controls={"sampling": "greedy", "max_tokens": "8", "rounds": "3"},
    source="models/flashnext/tests/bench/bench_server_short_paired.py",
)
