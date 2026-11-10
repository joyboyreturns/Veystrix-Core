"""Model-free synthetic tests; no checkpoint or network access."""
import unittest
import mlx.core as mx
import numpy as np
from .model import make_synthetic_model
from .config import ConfigurationError, inspect_checkpoint

class SyntheticModelTests(unittest.TestCase):
    def test_missing_checkpoint_fails_without_network(self):
        with self.assertRaisesRegex(ConfigurationError, "no download"):
            inspect_checkpoint('/definitely/missing/nemotron')

    def test_full_prefill_matches_incremental_hybrid_state(self):
        model = make_synthetic_model()
        x = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        whole_cache, split_cache = model.make_cache(), model.make_cache()
        whole = model(x, cache=whole_cache)
        parts = [model(x[:, i:i+1], cache=split_cache) for i in range(4)]
        split = mx.concatenate(parts, axis=1)
        mx.eval(whole, split)
        np.testing.assert_allclose(np.asarray(whole), np.asarray(split), rtol=1e-4, atol=1e-4)

    def test_snapshot_restore_and_prefix_mismatch(self):
        import tempfile
        from pathlib import Path
        from .state import save, restore, SnapshotError
        model = make_synthetic_model()
        cache = model.make_cache()
        model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.safetensors"
            identity = {"model": "synthetic", "tokenizer": "fixture", "precision": "fp32", "execution": "reference"}
            save(path, cache, identity, [1, 2, 3])
            recovered = restore(path, identity, [1, 2, 3])
            a = model(mx.array([[4]], dtype=mx.int32), cache=cache)
            b = model(mx.array([[4]], dtype=mx.int32), cache=recovered)
            mx.eval(a, b)
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            with self.assertRaises(SnapshotError):
                restore(path, identity, [1, 2, 4])
            path.write_bytes(b"damaged")
            with self.assertRaises(SnapshotError):
                restore(path, identity, [1, 2, 3])

    def test_chunked_prefill_matches_incremental(self):
        model = make_synthetic_model()
        x = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        chunk_cache, step_cache = model.make_cache(), model.make_cache()
        chunk = mx.concatenate([model(x[:, :2], cache=chunk_cache), model(x[:, 2:], cache=chunk_cache)], axis=1)
        step = mx.concatenate([model(x[:, i:i+1], cache=step_cache) for i in range(4)], axis=1)
        mx.eval(chunk, step)
        np.testing.assert_allclose(np.asarray(chunk), np.asarray(step), rtol=1e-4, atol=1e-4)

class SyntheticStorageTests(unittest.TestCase):
    def test_expert_reads_lru_and_corruption(self):
        import hashlib, tempfile
        from pathlib import Path
        from .expert_store import ExpertStore
        a, b = b'1234', b'abcd'
        sha = lambda data: hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'experts.bin'
            path.write_bytes(a + b)
            store = ExpertStore(path, {'a': (0, 4, sha(a)), 'b': (4, 4, sha(b))}, 4)
            self.assertEqual(store.get('a'), a)
            self.assertEqual(store.get('a'), a)
            self.assertEqual(store.get('b'), b)
            self.assertEqual(store.get('a'), a)
            self.assertEqual((store.hits, store.misses, store.bytes_read, store.resident_bytes), (1, 3, 12, 4))
            damaged = ExpertStore(path, {'x': (0, 4, sha(b))}, 4)
            with self.assertRaisesRegex(ValueError, 'damaged'):
                damaged.get('x')
            oversized = ExpertStore(path, {'x': (0, 8, sha(a + b))}, 0, max_record_bytes=4)
            with self.assertRaisesRegex(ValueError, 'oversized'):
                oversized.get('x')

    def test_mlx_affine_four_bit_round_trip(self):
        x = mx.array(np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32))
        packed, scales, biases = mx.quantize(x, group_size=32, bits=4)
        restored = mx.dequantize(packed, scales, biases, group_size=32, bits=4)
        mx.eval(restored)
        np.testing.assert_allclose(np.asarray(restored), np.asarray(x), atol=0.07, rtol=0)

class ReferenceMathTests(unittest.TestCase):
    def test_router_selects_full_topk_with_uncorrected_weights(self):
        from mlx_lm.models.nemotron_h import group_expert_select
        gates = mx.array([[0.0, 1.0, -1.0, 0.5]])
        bias = mx.array([1.0, 0.0, 0.0, 0.0])
        indices, weights = group_expert_select(gates, bias, 2, 1, 1, 2.5, True)
        mx.eval(indices, weights)
        indices = np.asarray(indices)[0]
        scores = 1 / (1 + np.exp(-np.asarray(gates)[0]))
        expected = np.argsort(-(scores + np.asarray(bias)))[:2]
        self.assertEqual(set(indices.tolist()), set(expected.tolist()))
        np.testing.assert_allclose(np.asarray(weights)[0], scores[indices] / scores[indices].sum() * 2.5, atol=1e-6)

    def test_nvidia_six_of_128_routing_preserves_selection_and_weights(self):
        from mlx_lm.models.nemotron_h import group_expert_select
        logits_np = np.linspace(-5, 5, 128, dtype=np.float32)[None, :]
        bias_np = np.zeros(128, dtype=np.float32)
        bias_np[3] = 2.0
        ids, weights = group_expert_select(mx.array(logits_np), mx.array(bias_np), 6, 1, 1, 2.5, True)
        mx.eval(ids, weights)
        selected = np.asarray(ids)[0]
        original = 1 / (1 + np.exp(-logits_np[0]))
        expected = np.argsort(-(original + bias_np))[:6]
        self.assertEqual(set(selected.tolist()), set(expected.tolist()))
        np.testing.assert_allclose(np.asarray(weights)[0], original[selected] / original[selected].sum() * 2.5, atol=1e-6)

    def test_mamba_ssm_matches_scalar_recurrence(self):
        from mlx_lm.models.ssm import ssm_update
        x = np.array([[[[0.2]], [[0.4]], [[-0.1]]]], dtype=np.float32)
        a_log = np.array([0.0], dtype=np.float32)
        b = np.array([[[[0.3]], [[0.6]], [[0.1]]]], dtype=np.float32)
        c = np.array([[[[0.2]], [[0.4]], [[0.7]]]], dtype=np.float32)
        dt = np.array([[[0.1], [0.2], [0.3]]], dtype=np.float32)
        d = np.array([0.5], dtype=np.float32)
        bias = np.array([0.0], dtype=np.float32)
        y, state = ssm_update(*map(mx.array, (x, a_log, b, c, d, dt, bias)), None, (0.0, float('inf')), None)
        mx.eval(y, state)
        expected, current = [], 0.0
        for t in range(3):
            delta = np.logaddexp(0, dt[0, t, 0])
            current = np.exp(-delta) * current + x[0, t, 0, 0] * delta * b[0, t, 0, 0]
            expected.append(current * c[0, t, 0, 0] + x[0, t, 0, 0] * d[0])
        np.testing.assert_allclose(np.asarray(y).ravel(), expected, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(float(np.asarray(state).ravel()[0]), current, places=6)

class SyntheticRuntimeTests(unittest.TestCase):
    def test_generate_and_cancel_clear_hybrid_state(self):
        from .runtime import SyntheticRuntime, Cancelled
        runtime = SyntheticRuntime(make_synthetic_model())
        logits = runtime.prefill([1, 2, 3, 4], chunk_size=2)
        output = runtime.greedy_decode(logits, 3)
        self.assertEqual(len(output), 3)
        self.assertEqual(runtime.tokens[:4], [1, 2, 3, 4])
        with self.assertRaises(Cancelled):
            runtime.prefill([1], cancelled=lambda: True)
        self.assertEqual(runtime.tokens, [])
        self.assertTrue(all(entry.state == [None, None] for entry in runtime.cache if hasattr(entry, 'cache')))

class MemoryPlanningTests(unittest.TestCase):
    def test_budget_counts_recurrent_kv_and_fails_closed(self):
        from .memory import estimate, require_budget
        config = dict(layers_block_type=['mamba', 'attention'], mamba_num_heads=4,
                      mamba_head_dim=4, ssm_state_size=32, n_groups=2, conv_kernel=4,
                      num_key_value_heads=2, head_dim=4)
        result = estimate(config=config, context_tokens=10, non_expert_bytes=100,
                          expert_pool_bytes=200, activation_bytes=300, reserve_bytes=400)
        self.assertEqual(result.attention_kv_bytes, 10 * 2 * 2 * 4 * 2)
        self.assertGreater(result.mamba_state_bytes, 0)
        require_budget(result, result.total_bytes)
        with self.assertRaises(MemoryError):
            require_budget(result, result.total_bytes - 1)

class ComponentTests(unittest.TestCase):
    def test_moe_combines_all_selected_experts_and_shared(self):
        model = make_synthetic_model()
        moe = model.backbone.layers[2].mixer
        x = mx.array(np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 2, 16))
        indices, weights = moe.gate(x)
        expected = (moe.switch_mlp(x, indices) * weights[..., None]).sum(axis=-2) + moe.shared_experts(x)
        got = moe(x)
        mx.eval(got, expected, indices)
        self.assertEqual(indices.shape[-1], 2)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))

    def test_attention_kv_incremental(self):
        from mlx_lm.models.cache import KVCache
        model = make_synthetic_model()
        attention = model.backbone.layers[1].mixer
        x = mx.array(np.linspace(-1, 1, 64, dtype=np.float32).reshape(1, 4, 16))
        full_cache, step_cache = KVCache(), KVCache()
        full = attention(x, mask='causal', cache=full_cache)
        pieces = [attention(x[:, i:i+1], cache=step_cache) for i in range(4)]
        incremental = mx.concatenate(pieces, axis=1)
        mx.eval(full, incremental)
        np.testing.assert_allclose(np.asarray(full), np.asarray(incremental), rtol=1e-5, atol=1e-6)
        self.assertEqual(step_cache.offset, 4)

class StreamingExpertsTests(unittest.TestCase):
    def test_ssd_affine_experts_match_resident_quantized_computation(self):
        import hashlib, tempfile
        from pathlib import Path
        from .expert_store import ExpertStore
        from .streaming_moe import StreamedExperts
        import mlx.nn as nn
        rng = np.random.default_rng(42)
        records, specs, raw = {}, {}, bytearray()
        reference = []
        for expert in range(3):
            up = mx.array(rng.normal(size=(32, 32)).astype(np.float32) / 5)
            down = mx.array(rng.normal(size=(32, 32)).astype(np.float32) / 5)
            tensors = (*mx.quantize(up, group_size=32, bits=4), *mx.quantize(down, group_size=32, bits=4))
            mx.eval(*tensors)
            names = ('up_w','up_scales','up_biases','down_w','down_scales','down_biases')
            arrays = [np.asarray(t) for t in tensors]
            record = b''.join(a.tobytes() for a in arrays)
            records[expert] = (len(raw), len(record), hashlib.sha256(record).hexdigest())
            specs[expert] = {n: (a.shape, a.dtype.name) for n, a in zip(names, arrays)}
            raw.extend(record)
            reference.append(tensors)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'synthetic-experts.bin'
            path.write_bytes(raw)
            store = ExpertStore(path, records, len(raw))
            streamed = StreamedExperts(store, specs)
            x = mx.array(rng.normal(size=(1, 2, 32)).astype(np.float32))
            indices = mx.array([[[0, 2], [2, 1]]], dtype=mx.int32)
            actual = streamed(x, indices)
            expected = []
            for row, ids in enumerate(((0, 2), (2, 1))):
                values = []
                for expert in ids:
                    u, us, ub, d, ds, db = reference[expert]
                    hidden = nn.relu2(mx.quantized_matmul(x[0,row:row+1], u, us, ub, group_size=32, bits=4))
                    values.append(mx.quantized_matmul(hidden, d, ds, db, group_size=32, bits=4)[0])
                expected.append(mx.stack(values))
            expected = mx.stack(expected)[None]
            mx.eval(actual, expected)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            self.assertEqual(store.misses, 3)
            self.assertEqual(store.hits, 1)
            self.assertLessEqual(store.resident_bytes, store.budget)
            # The same streamed implementation can serve the upstream hybrid MoE layer.
            model = make_synthetic_model(width=32)
            model.backbone.layers[2].mixer.switch_mlp = streamed
            # Synthetic router has four experts; include record 3 as a fourth expert.
            store.records[3] = store.records[0]
            specs[3] = specs[0]
            tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
            full_cache, step_cache = model.make_cache(), model.make_cache()
            full = model(tokens, cache=full_cache)
            step = mx.concatenate([model(tokens[:, i:i+1], cache=step_cache) for i in range(4)], axis=1)
            mx.eval(full, step)
            np.testing.assert_allclose(np.asarray(full), np.asarray(step), atol=1e-4, rtol=1e-4)
            from .hybrid import StreamingHybrid
            hybrid = StreamingHybrid(model.args, lambda layer: streamed)
            # No upstream resident SwitchMLP or routed expert weight arrays are constructed.
            self.assertEqual([layer.kind for layer in hybrid.layers], ["M", "*", "E"])
            hybrid.layers[2].mixer.gate.weight = mx.random.normal((4, 32)) * 0.1
            first_cache, second_cache = hybrid.make_cache(), hybrid.make_cache()
            first = hybrid(tokens, cache=first_cache)
            second = mx.concatenate([hybrid(tokens[:, i:i+1], cache=second_cache) for i in range(4)], axis=1)
            mx.eval(first, second)
            np.testing.assert_allclose(np.asarray(first), np.asarray(second), atol=1e-4, rtol=1e-4)

class LocalCheckpointFixtureTests(unittest.TestCase):
    def test_tiny_checkpoint_loads_dense_and_streams_experts(self):
        import tempfile
        from pathlib import Path
        from .fixture import create_fixture, load_fixture
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture'
            original = create_fixture(path)
            restored = load_fixture(path)
            tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
            expected = original(tokens, cache=original.make_cache())
            actual = restored(tokens, cache=restored.make_cache())
            mx.eval(actual, expected)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            with self.assertRaises(FileExistsError):
                create_fixture(path)

class SyntheticAdapterTests(unittest.TestCase):
    def test_adapter_executes_streamed_fixture_and_cancels_safely(self):
        import tempfile
        from pathlib import Path
        from veystrix.backends.base import GenerationCancelled
        from veystrix.backends.nemotron35 import SyntheticBackend
        from .fixture import create_fixture
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'fixture'
            create_fixture(path)
            backend = SyntheticBackend(path)
            backend.append_tokens([1, 2, 3])
            text, stats = backend.generate(2)
            self.assertEqual(len(text.split()), 2)
            self.assertEqual((stats.prompt_tokens, stats.tokens), (3, 2))
            self.assertEqual(len(backend.tape), 5)
            backend.append_tokens([4])
            with self.assertRaises(GenerationCancelled):
                backend.generate(2, should_cancel=lambda: True)
            self.assertEqual((backend.tape, backend.pending), ([], []))

class CliSafetyTests(unittest.TestCase):
    def test_nemotron_launcher_never_initializes_runtime(self):
        import sys
        from unittest.mock import patch
        from veystrix import cli
        with patch.object(sys, 'argv', ['chat.sh', '--model', 'nemotron35-lightning', '--checkpoint', '/missing']), \
             patch.object(cli, '_interpreter') as interpreter:
            with self.assertRaisesRegex(SystemExit, 'no download'):
                cli.main()
            interpreter.assert_not_called()

    def test_check_missing_checkpoint_does_not_initialize_runtime(self):
        import sys
        from unittest.mock import patch
        from veystrix import cli
        with patch.object(sys, 'argv', ['chat.sh', '--model', 'nemotron35-lightning', '--checkpoint', '/missing', '--check']), \
             patch.object(cli, '_interpreter') as interpreter:
            with self.assertRaisesRegex(SystemExit, 'no download'):
                cli.main()
            interpreter.assert_not_called()

class CheckpointMetadataTests(unittest.TestCase):
    def test_exact_architecture_and_wrong_order(self):
        from .config import EXPECTED, LAYER_ORDER, validate_config, ConfigurationError
        config = dict(EXPECTED, layers_block_type=list(LAYER_ORDER))
        validate_config(config)
        config["layers_block_type"][0], config["layers_block_type"][1] = config["layers_block_type"][1], config["layers_block_type"][0]
        with self.assertRaisesRegex(ConfigurationError, "layer order"):
            validate_config(config)

    def test_nvfp4_is_not_silently_treated_as_affine(self):
        from .config import EXPECTED, LAYER_ORDER, validate_config, ConfigurationError
        config = dict(EXPECTED, layers_block_type=list(LAYER_ORDER), quantization_config={"format": "NVFP4"})
        with self.assertRaisesRegex(ConfigurationError, "NVFP4"):
            validate_config(config)

class SafeTensorIndexTests(unittest.TestCase):
    def test_metadata_offsets_and_path_rejection_without_tensor_loading(self):
        import tempfile
        from pathlib import Path
        from .safetensors_index import index_shard, IndexError
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'weights.safetensors'
            mx.save_safetensors(str(path), {'a': mx.array([[1.0, 2.0]])})
            index = index_shard(root, path.name)
            self.assertEqual((index['a']['dtype'], index['a']['shape'], index['a']['length']), ('F32', (1, 2), 8))
            with self.assertRaises(IndexError):
                index_shard(root, '../weights.safetensors')
            path.write_bytes(b'broken')
            with self.assertRaises(IndexError):
                index_shard(root, path.name)

    def test_sharded_index_validates_tensor_membership(self):
        import json, tempfile
        from pathlib import Path
        from .safetensors_index import index_checkpoint, IndexError
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            mx.save_safetensors(str(path / 'model-1.safetensors'), {'layer.expert.weight': mx.ones((2, 2))})
            manifest = path / 'model.safetensors.index.json'
            manifest.write_text(json.dumps({'weight_map': {'layer.expert.weight': 'model-1.safetensors'}}))
            self.assertEqual(index_checkpoint(path)['layer.expert.weight']['shape'], (2, 2))
            manifest.write_text(json.dumps({'weight_map': {'missing': 'model-1.safetensors'}}))
            with self.assertRaisesRegex(IndexError, 'missing from shard'):
                index_checkpoint(path)

class AffineCheckpointIndexTests(unittest.TestCase):
    def test_per_expert_affine_tensors_read_from_safetensors(self):
        import json, tempfile
        from pathlib import Path
        from .affine_index import AffineExpertIndex
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            matrices = [mx.array(np.arange(1024, dtype=np.float32).reshape(32, 32) / 1000)] * 2
            fields = {}
            for matrix, name in zip(matrices, ('up_proj', 'down_proj')):
                for item, part in zip(mx.quantize(matrix, group_size=32, bits=4), ('weight', 'scales', 'biases')):
                    fields[f'backbone.layers.1.mixer.experts.0.{name}.{part}'] = item
            mx.save_safetensors(str(root / 'part.safetensors'), fields)
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'part.safetensors' for key in fields}}))
            index = AffineExpertIndex(root, hidden_dim=32, intermediate_dim=32, group_size=32)
            self.assertEqual(index.validate_inventory([1], 1)['tensors'], 6)
            with self.assertRaisesRegex(ValueError, 'missing'):
                index.validate_inventory([1], 2)
            actual = index.read(1, 0)
            for key, value in fields.items():
                name = '.'.join(key.split('.')[-2:])
                mx.eval(value, actual[name])
                np.testing.assert_array_equal(np.asarray(value), np.asarray(actual[name]))
            with self.assertRaisesRegex(ValueError, 'missing'):
                index.read(1, 1)
            from .streaming_moe import IndexedAffineExperts
            streamed = IndexedAffineExperts(index, layer=1, max_resident_bytes=100000)
            x = mx.ones((1, 2, 32))
            output = streamed(x, mx.array([[[0], [0]]]))
            u, d = (mx.quantize(matrices[0], group_size=32, bits=4), mx.quantize(matrices[1], group_size=32, bits=4))
            import mlx.nn as nn
            expected = mx.quantized_matmul(nn.relu2(mx.quantized_matmul(x, *u, group_size=32, bits=4)), *d, group_size=32, bits=4)
            mx.eval(output, expected)
            # Per-token vs batched MLX matmul can choose different float32 reduction orders.
            np.testing.assert_allclose(np.asarray(output[:, :, 0, :]), np.asarray(expected), rtol=1e-6, atol=1e-6)
            self.assertEqual((streamed.misses, streamed.hits), (1, 1))
            self.assertLessEqual(streamed.resident_bytes, streamed.max_resident_bytes)

    def test_half_precision_affine_metadata_round_trips(self):
        import json, tempfile
        from pathlib import Path
        from .affine_index import AffineExpertIndex
        for precision in (mx.float16, mx.bfloat16):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                w = mx.array(np.linspace(-1, 1, 1024, dtype=np.float32).reshape(32, 32)).astype(precision)
                fields = {}
                for name in ('up_proj', 'down_proj'):
                    for tensor, part in zip(mx.quantize(w, group_size=32, bits=4), ('weight', 'scales', 'biases')):
                        fields[f'backbone.layers.1.mixer.experts.0.{name}.{part}'] = tensor
                mx.save_safetensors(str(root / 'part.safetensors'), fields)
                (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {key: 'part.safetensors' for key in fields}}))
                actual = AffineExpertIndex(root, hidden_dim=32, intermediate_dim=32, group_size=32).read(1, 0)
                for key, tensor in fields.items():
                    got = actual['.'.join(key.split('.')[-2:])]
                    self.assertEqual(tensor.dtype, got.dtype)
                    mx.eval(tensor, got)
                    np.testing.assert_array_equal(np.asarray(tensor.astype(mx.float32)), np.asarray(got.astype(mx.float32)))
