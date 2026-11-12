"""Paired prefix selection over synthetic manifests. No checkpoint or tensor state is loaded."""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile

from veystrix.measurement import MeasurementRun, source_hash
from veystrix.results import ROOT, output_path

PREFIX = 102_400
ENTRIES = 32
REPEATS = 15
IDENTITY = {"model": "synthetic-cache-lookup"}

CHILD = """
import json, sys, time
from pathlib import Path
from statistics import median
import numpy as np
from models.nemotron35.prefix_cache import PrefixCache
from veystrix.universal_cache import CacheManager
root, length, repeats, identity = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
manager = CacheManager(root, reserve_bytes=0)
cache = PrefixCache(root, identity, quota_bytes=2**30, manager=manager)
query = np.arange(length + 1, dtype='<u4')
times = []
for _ in range(repeats + 1):
    started = time.perf_counter()
    assert cache.reusable_tokens(query) == length
    times.append(time.perf_counter() - started)
print(json.dumps({'cold_lookup_seconds': times[0], 'warm_lookup_seconds': median(times[1:]), 'seconds': times}))
"""


def fixture(root):
    import numpy as np
    from models.nemotron35.prefix_cache import PrefixCache
    from veystrix.universal_cache import CacheManager

    manager = CacheManager(root, reserve_bytes=0)
    cache = PrefixCache(root, IDENTITY, quota_bytes=2**30, manager=manager)
    for i in range(ENTRIES):
        count = PREFIX if i == 0 else PREFIX - i * 64
        ids = np.arange(count, dtype="<u4")
        ids[0] = i
        key = cache._key(ids)
        data, sidecar = cache._paths(key)
        data.write_bytes(b"synthetic")
        sidecar.write_bytes(b"VXT1" + ids.tobytes())
    other = manager.namespace("flashnext", "1" * 64)
    for i in range(ENTRIES):
        key = f"{i:064x}"
        (other / f"{key}.safetensors").write_bytes(b"synthetic")
        (other / f"{key}.json").write_text(json.dumps({
            "key": key, "fingerprint": other.name, "parent": None, "file_bytes": 9, "n_tokens": PREFIX,
        }))


def main():
    provenance = Path(os.environ["VEYSTRIX_BENCH_PROVENANCE"]).resolve()
    sources = {"baseline": Path(os.environ["VEYSTRIX_BENCH_BASELINE_SOURCE"]).resolve(), "candidate": ROOT}
    result = output_path("nemotron35", "cache-lookup-paired", "lookup-arms.jsonl")
    manifests = {name: source_hash(provenance / f"{name}-files.json") for name in sources}
    run = MeasurementRun(result, runtime="nemotron35", experiment="cache-lookup-paired",
                         metadata={"synthetic": True, "prefix_tokens": PREFIX, "entries_per_namespace": ENTRIES,
                                   "warm_repetitions": REPEATS, "source_manifest_sha256": manifests,
                                   "revision": os.environ["VEYSTRIX_BENCH_REVISION"]})
    run.start()
    arms = {name: [] for name in sources}
    for index, order in enumerate((tuple(sources), tuple(reversed(sources)), tuple(sources)), start=1):
        for name in order:
            arm_id = f"{index}-{name}"
            with tempfile.TemporaryDirectory(prefix="veystrix-cache-lookup-") as scratch:
                fixture(Path(scratch))
                command = [sys.executable, "-c", CHILD, scratch, str(PREFIX), str(REPEATS), json.dumps(IDENTITY)]
                process = subprocess.run(command, cwd=sources[name], text=True, capture_output=True,
                                         env=os.environ | {"PYTHONPATH": str(sources[name])}, timeout=60)
                metrics = json.loads(process.stdout) if process.returncode == 0 else {}
                run.arm(arm_id=arm_id, condition=name, round_index=index, command=command, metrics=metrics,
                        status="raw" if metrics else "failed", source_root=str(sources[name]))
                if not metrics:
                    run.failure(process.stderr, arm_id=arm_id)
                    return 1
                arms[name].append(metrics)
                print(f"{arm_id} cold {metrics['cold_lookup_seconds'] * 1000:.2f} ms "
                      f"warm {metrics['warm_lookup_seconds'] * 1000:.2f} ms", flush=True)
    run.finish("diagnostic", medians={name: {key: median(row[key] for row in rows)
                                           for key in ("cold_lookup_seconds", "warm_lookup_seconds")}
                                     for name, rows in arms.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
