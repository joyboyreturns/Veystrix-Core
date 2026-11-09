"""Opt-in, fail-closed state probe for supervised layer-order experiments.

This diagnostic executes in the server only when its explicit output path is
set. It records hashes, shapes and dtypes, never prompt text or state tensors.
Capturing hashes is outside prefill timing and adds work, so these arms are
correctness diagnostics, not promotion-quality throughput measurements.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

FLAG = "VEYSTRIX_LAYER_GATE_PROBE"


def _array_record(array):
    import mlx.core as mx
    import numpy as np

    if array is None:
        return None
    raw = array.view(mx.uint8)
    mx.eval(raw)
    # Hash the exported buffer directly: tobytes() would allocate another
    # full-size host copy of each cache tensor during a memory-sensitive run.
    data = np.ascontiguousarray(np.asarray(raw))
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(memoryview(data)).hexdigest()}


def state_record(cache, tokens, language):
    """Capture every supported cache field and language position state, or raise."""
    from models.flashnext import prefix_cache as P

    if not cache or not tokens:
        raise ValueError("empty cache or prompt cannot establish state equivalence")
    entry = P.capture(cache, tokens, "layer-gate", language=language)
    layers = []
    for layer in entry.layers:
        if layer.kind == "arrays":
            layers.append({"kind": layer.kind, "states": [_array_record(a) for a in layer.states]})
        else:
            layers.append({"kind": layer.kind,
                           "kv": {key: _array_record(value) for key, value in sorted(layer.kv.items())}})
    return {"token_count": len(tokens), "layers": layers,
            "language": {name: _array_record(getattr(language, attr, None)) for name, attr in P.LANG_FIELDS}}


def record_logits(array):
    """Append the actual final-row logits selected by the prefill sampler."""
    from macqwen.results import ensure_inside

    path = ensure_inside(Path(os.environ[FLAG]))
    with path.open("r+", encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps({"event": "logits", "value": _array_record(array)}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Probe:
    def __init__(self, path):
        from macqwen.results import ensure_inside
        from models.flashnext.diskio import disk_bytes_read
        import mlx.core as mx

        self.path = ensure_inside(Path(path))
        # Reserve the path without leaving a descriptor open if inference is
        # cancelled or the model raises before the final diagnostic record.
        with self.path.open("x", encoding="utf-8"):
            pass
        self.before_disk = disk_bytes_read()
        self.started = time.perf_counter()
        mx.reset_peak_memory()

    def write(self, record):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def prefilled(self, backend, token, prefill_seconds):
        import mlx.core as mx
        from models.flashnext.diskio import disk_bytes_read

        disk = disk_bytes_read()
        peak = mx.get_peak_memory()
        tokens = list(backend.tape)
        prefix = getattr(backend, "last_prefix", None)
        self.write({"event": "prefilled", "first_token": int(token.item()),
                    "prompt_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                    "state": state_record(backend.cache, tokens, backend.language),
                    "prefix": prefix.as_dict() if hasattr(prefix, "as_dict") else prefix,
                    "prefill_seconds": prefill_seconds, "peak_bytes": peak,
                    "physical_bytes": disk - self.before_disk if disk >= 0 and self.before_disk >= 0 else None})

    def finished(self, produced, stats):
        from models.flashnext.diskio import disk_bytes_read

        disk = disk_bytes_read()
        self.write({"event": "finished", "token_ids": [int(t) for t in produced],
                    "finish": stats.finish, "generation_seconds": stats.seconds,
                    "tail_seconds": stats.tail_seconds, "tail_tokens": stats.tail_tokens,
                    "wall_seconds": time.perf_counter() - self.started,
                    "physical_bytes_total": disk - self.before_disk if disk >= 0 and self.before_disk >= 0 else None})
