"""Supervised gate for the experimental hybrid prefill plan."""
from veystrix.testsuite.api import TestSpec, bench_module


def command(config, _result_path):
    return [str(config.python), "-m", bench_module("flashnext", "bench_layer_gate"),
            "--checkpoint", str(config.checkpoint), "--candidate", "layer-major-hybrid-4k"]


TEST = TestSpec(
    id="layer-hybrid-gate", title="Supervised hybrid prefill state equivalence",
    category="diagnostic",
    explanation="Three reversed pairs of supervised AGENTIC servers compare chunk-major with the opt-in hybrid plan.",
    why="The hybrid heuristic and its state must pass the exact real-checkpoint gate before speed testing.",
    script=command,
    metrics=("attention and recurrent state hashes", "position state", "final-row logits", "generated token IDs",
             "chunk edges and executed order", "physical read bytes", "MLX peak memory", "safety telemetry"),
    controls={"sampling": "greedy", "pool": "0 GB", "ordering": "three reversed pairs",
              "cache": "separate empty prefix store per arm; OS file cache is not purged",
              "safety": "stop on the first safety or exactness failure; no automatic retry",
              "tools": "tool schema in prompt context; actual tool protocol is a separate test",
              "headroom": "postponed (exit 2, nothing started) unless available memory covers floor + observed footprint + margin"},
    source="models/flashnext/tests/bench/bench_layer_gate.py", promotion=False,
)
