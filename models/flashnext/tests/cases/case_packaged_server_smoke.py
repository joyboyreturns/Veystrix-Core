from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_short_paired"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint, "--packaged"]


TEST = TestSpec(
    id="packaged-server-smoke", title="Signed app server smoke", category="diagnostic",
    explanation="Loads the installed checkpoint with the signed app interpreter and completes a streamed request.",
    why="Checks the packaged entry point, cache diagnostics, and normal response delivery.",
    script=command, metrics=COMMON_METRICS, promotion=False,
    controls={"sampling": "greedy", "max_tokens": "8", "rounds": "1"},
    source="models/flashnext/tests/bench/bench_server_short_paired.py",
)
