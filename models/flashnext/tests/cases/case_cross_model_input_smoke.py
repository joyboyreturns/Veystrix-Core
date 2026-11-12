from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_short_paired"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint, "--cross-model"]


TEST = TestSpec(
    id="cross-model-input-smoke", title="Signed app cross-model input reuse", category="diagnostic",
    explanation="Runs the same authenticated input through Flash-Next, Nemotron and Flash-Next in one router.",
    why="Checks that input bytes can be reused across both switches while model state remains scoped.",
    script=command, metrics=COMMON_METRICS, promotion=False,
    controls={"max_tokens": "8", "resident_models": "one"},
    source="models/flashnext/tests/bench/bench_server_short_paired.py",
)
