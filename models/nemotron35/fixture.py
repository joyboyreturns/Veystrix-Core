"""Create/load tiny local MLX fixtures. Not a real-checkpoint converter or loader."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import mlx.core as mx
from .expert_store import ExpertStore
from .hybrid import StreamingHybrid
from .model import make_synthetic_model
from .streaming_moe import StreamedExperts

_FIELDS = ('up_w', 'up_scales', 'up_biases', 'down_w', 'down_scales', 'down_biases')

def create_fixture(directory, seed=7):
    """Create a new directory; refuse to overwrite existing fixtures or user data."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    mx.random.seed(seed)
    records, specs, data = {}, {}, bytearray()
    for expert in range(4):
        up, down = mx.random.normal((32, 32)) * 0.1, mx.random.normal((32, 32)) * 0.1
        tensors = (*mx.quantize(up, group_size=32, bits=4), *mx.quantize(down, group_size=32, bits=4))
        mx.eval(*tensors)
        import numpy as np
        arrays = [np.asarray(t) for t in tensors]
        record = b''.join(a.tobytes() for a in arrays)
        records[expert] = (len(data), len(record), hashlib.sha256(record).hexdigest())
        specs[expert] = {key: (a.shape, a.dtype.name) for key, a in zip(_FIELDS, arrays)}
        data.extend(record)
    (root / 'experts.bin').write_bytes(data)
    index = {'format': 'veystrix-synthetic-nemotron-v1', 'records': records, 'specs': specs}
    (root / 'index.json').write_text(json.dumps(index, sort_keys=True))
    model = _make(root, index)
    mx.eval(model.parameters())
    model.save_weights(str(root / 'dense.safetensors'))
    return model

def _make(root, index):
    if index.get('format') != 'veystrix-synthetic-nemotron-v1':
        raise ValueError('only the tiny synthetic fixture format is supported')
    records = {int(k): v for k, v in index['records'].items()}
    specs = {int(k): v for k, v in index['specs'].items()}
    if set(records) != set(range(4)) or set(specs) != set(records):
        raise ValueError('fixture requires four synthetic experts')
    args = make_synthetic_model(width=32).args
    store = ExpertStore(root / 'experts.bin', records, budget_bytes=128 * 1024)
    return StreamingHybrid(args, lambda _layer: StreamedExperts(store, specs))

def load_fixture(directory):
    root = Path(directory)
    index_path = root / 'index.json'
    if index_path.stat().st_size > 1024 * 1024:
        raise ValueError('synthetic index too large')
    index = json.loads(index_path.read_text())
    model = _make(root, index)
    model.load_weights(str(root / 'dense.safetensors'), strict=True)
    return model
