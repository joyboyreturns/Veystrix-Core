"""Supervised exactness gate for the opt-in layer-major prefill path."""
from macqwen.testsuite.api import TestSpec, bench_module


def command(config, _result_path):
    return [str(config.python), "-m", bench_module("flashnext", "bench_layer_gate"),
            "--checkpoint", str(config.checkpoint)]


TEST = TestSpec(
    id="layer-major-gate", title="Supervised layer-major state equivalence",
    category="diagnostic",
    explanation="Three reversed pairs of fresh supervised AGENTIC servers run a 2K–4K coding prompt with a tool schema in context.",
    why="Real-checkpoint state and token equivalence is required before any layer-major performance comparison.",
    script=command,
    metrics=("all attention and recurrent state hashes", "position state", "prompt and generated token IDs",
             "chunk edges", "physical read bytes", "MLX peak memory", "watchdog and swap guard"),
    controls={"sampling": "greedy", "pool": "0 GB in both arms", "ordering": "three reversed pairs",
              "cache": "separate empty prefix store per arm; OS file cache is not purged",
              "tools": "schema in prompt only; tool-enabled protocol requires at least 64 output tokens",
              "gate": "missing evidence or any mismatch stops immediately",
              "headroom": "postponed (exit 2, nothing started) unless available memory covers floor + observed footprint + margin"},
    source="models/flashnext/tests/bench/bench_layer_gate.py", promotion=False,
)
