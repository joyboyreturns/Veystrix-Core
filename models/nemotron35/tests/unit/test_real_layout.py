"""Model-free tests for the real-layout (stacked affine 4-bit) path.

All checkpoints here are tiny random synthetic ones written in the identical
tensor layout. These tests do not touch, load or validate any real weights.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from models.nemotron35 import inventory as inv
from models.nemotron35.config import EXPECTED, LAYER_ORDER, ConfigurationError, validate_config
from models.nemotron35.runtime import Cancelled
from models.nemotron35.streamed_model import load_streamed
from models.nemotron35.tiny_checkpoint import (TINY_CONFIG, permissive_config_check, write_checkpoint)

_ref_dir = None
_ref_model = None


def setUpModule():
    global _ref_dir, _ref_model
    _ref_dir = tempfile.mkdtemp(prefix='nemotron-tiny-')
    _ref_model = write_checkpoint(Path(_ref_dir) / 'ck')


def tearDownModule():
    shutil.rmtree(_ref_dir, ignore_errors=True)


def f32(x):
    return x.astype(mx.float32)


def copy_checkpoint():
    dest = Path(tempfile.mkdtemp(prefix='nemotron-copy-', dir=_ref_dir)) / 'ck'
    shutil.copytree(Path(_ref_dir) / 'ck', dest)
    return dest


def real_config():
    config = dict(EXPECTED)
    config.update(layers_block_type=list(LAYER_ORDER), vocab_size=131072,
                  quantization={'group_size': 64, 'bits': 4, 'mode': 'affine'}, n_group=1)
    return config


class InventoryTests(unittest.TestCase):
    def test_full_size_inventory_matches_published_index_totals(self):
        expected = inv.expected_tensors(real_config())
        self.assertEqual(len(expected), 729)
        sizes = inv.estimate_bytes(real_config())
        # published total_size 17,775,339,392 differs only by the float32 router bias (23 * 128 * 2 extra bytes)
        self.assertEqual(sizes['total_bytes'] + 23 * 128 * 2, 17775339392)
        self.assertEqual(sizes['routed_expert_bytes'], 16523329536)

    def test_rejects_nvfp4_and_non_affine(self):
        for q in ({'mode': 'nvfp4', 'bits': 4, 'group_size': 16}, {'mode': 'affine', 'bits': 8, 'group_size': 64}):
            cfg = real_config()
            cfg['quantization'] = q
            with self.assertRaises(inv.CheckpointError):
                inv.expected_tensors(cfg)

    def test_valid_tiny_checkpoint_verifies(self):
        summary = inv.verify_checkpoint(Path(_ref_dir) / 'ck', config_check=permissive_config_check)
        self.assertEqual(summary['tensors'], 63)
        self.assertGreater(summary['routed_expert_bytes'], 0)

    def test_strict_public_config_refuses_tiny_checkpoint(self):
        with self.assertRaises(inv.CheckpointError):
            inv.verify_checkpoint(Path(_ref_dir) / 'ck')

    def _mutate_index(self, ck, fn):
        path = ck / 'model.safetensors.index.json'
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data))

    def test_missing_tensor_in_index_fails(self):
        ck = copy_checkpoint()
        self._mutate_index(ck, lambda d: d['weight_map'].pop('backbone.norm_f.weight'))
        with self.assertRaisesRegex(inv.CheckpointError, 'inventory|missing'):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_unexpected_tensor_in_index_fails(self):
        ck = copy_checkpoint()
        self._mutate_index(ck, lambda d: d['weight_map'].update({'mtp.layers.0.x': 'model-00001-of-00002.safetensors'}))
        with self.assertRaises(inv.CheckpointError):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_total_size_mismatch_fails(self):
        ck = copy_checkpoint()
        self._mutate_index(ck, lambda d: d['metadata'].update(total_size=d['metadata']['total_size'] + 1))
        with self.assertRaisesRegex(inv.CheckpointError, 'total_size'):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_truncated_shard_fails(self):
        ck = copy_checkpoint()
        shard = ck / 'model-00001-of-00002.safetensors'
        shard.write_bytes(shard.read_bytes()[:-16])
        with self.assertRaises(inv.CheckpointError):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_trailing_bytes_fail(self):
        ck = copy_checkpoint()
        with open(ck / 'model-00002-of-00002.safetensors', 'ab') as handle:
            handle.write(b'\0' * 8)
        with self.assertRaisesRegex(inv.CheckpointError, 'extent'):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_missing_shard_and_partial_download_fail(self):
        ck = copy_checkpoint()
        (ck / 'model-00002-of-00002.safetensors').unlink()
        with self.assertRaises(inv.CheckpointError):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)
        ck = copy_checkpoint()
        (ck / 'x.incomplete').write_bytes(b'')
        with self.assertRaisesRegex(inv.CheckpointError, 'partial'):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_unreferenced_shard_fails(self):
        ck = copy_checkpoint()
        shutil.copy(ck / 'model-00001-of-00002.safetensors', ck / 'model-00009-of-00002.safetensors')
        with self.assertRaisesRegex(inv.CheckpointError, 'not named'):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_symlinked_shard_fails(self):
        ck = copy_checkpoint()
        target = ck / 'model-00001-of-00002.safetensors'
        real = ck.parent / 'elsewhere.bin'
        shutil.move(target, real)
        os.symlink(real, target)
        with self.assertRaises(inv.CheckpointError):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_wrong_shape_fails(self):
        ck = copy_checkpoint()
        config = json.loads((ck / 'config.json').read_text())
        config['n_routed_experts'] = 4
        (ck / 'config.json').write_text(json.dumps(config))
        with self.assertRaises(inv.CheckpointError):
            inv.verify_checkpoint(ck, config_check=permissive_config_check)

    def test_hash_verification_detects_corruption(self):
        ck = copy_checkpoint()
        meta = ck / '.cache' / 'huggingface' / 'download'
        meta.mkdir(parents=True)
        for shard in ck.glob('*.safetensors'):
            (meta / f'{shard.name}.metadata').write_text(f'commit\n{inv.sha256_file(shard)}\n0\n')
        self.assertEqual(inv.verify_checkpoint(ck, config_check=permissive_config_check, hashes=True,
                                               require_hashes=True)['hashes_verified'], 2)
        shard = ck / 'model-00001-of-00002.safetensors'
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1  # flip one payload bit: structure stays valid, hash must catch it
        shard.write_bytes(bytes(data))
        with self.assertRaisesRegex(inv.CheckpointError, 'SHA-256'):
            inv.verify_checkpoint(ck, config_check=permissive_config_check, hashes=True)

    def test_required_hashes_missing_fails(self):
        with self.assertRaisesRegex(inv.CheckpointError, 'no recorded SHA-256'):
            inv.verify_checkpoint(Path(_ref_dir) / 'ck', config_check=permissive_config_check,
                                  hashes=True, require_hashes=True)


class StreamedModelTests(unittest.TestCase):
    def setUp(self):
        self.streamed = load_streamed(Path(_ref_dir) / 'ck', pool_bytes=64 * 1024,
                                      config_check=permissive_config_check)

    def tearDown(self):
        self.streamed.close()

    def test_dense_core_excludes_routed_experts(self):
        summary = inv.verify_checkpoint(Path(_ref_dir) / 'ck', config_check=permissive_config_check)
        self.assertEqual(self.streamed.dense_bytes, summary['dense_bytes'])

    def test_prefill_logits_match_in_memory_reference(self):
        tokens = mx.array([[1, 5, 9, 3, 7, 2, 11, 4, 6, 8]])
        a, b = f32(_ref_model(tokens)), f32(self.streamed.model(tokens))
        self.assertLess(float(mx.abs(a - b).max()), 0.05 * float(mx.abs(a).max()))
        self.assertTrue(bool(mx.all(mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1))))

    def test_cached_decode_matches_reference_exactly(self):
        ref_cache, cache = _ref_model.make_cache(), self.streamed.model.make_cache()
        for token in (1, 5, 9, 3, 7, 2):
            x = mx.array([[token]])
            a, b = f32(_ref_model(x, cache=ref_cache)), f32(self.streamed.model(x, cache=cache))
            self.assertLess(float(mx.abs(a - b).max()), 1e-2)
            self.assertEqual(int(mx.argmax(a[0, -1])), int(mx.argmax(b[0, -1])))

    def test_chunked_prefill_matches_single_pass(self):
        tokens = [1, 5, 9, 3, 7, 2, 11, 4]
        whole = f32(self.streamed.model(mx.array([tokens])))[:, -1]
        cache = self.streamed.model.make_cache()
        for start in (0, 3, 6):
            out = self.streamed.model(mx.array([tokens[start:start + 3]]), cache=cache)
        self.assertLess(float(mx.abs(whole - f32(out)[:, -1]).max()), 0.05 * float(mx.abs(whole).max()))

    def test_pool_is_bounded_and_counts_reads(self):
        for token in range(20):
            self.streamed.model(mx.array([[token, (token * 7) % 64, (token * 3) % 64]]))
        stats = self.streamed.pool.stats()
        self.assertLessEqual(stats['resident_bytes'], stats['budget_bytes'])
        self.assertGreater(stats['misses'], 0)
        self.assertGreater(stats['bytes_read'], 0)

    def test_pool_smaller_than_one_expert_still_correct(self):
        self.streamed.close()
        self.streamed = load_streamed(Path(_ref_dir) / 'ck', pool_bytes=0, config_check=permissive_config_check)
        tokens = mx.array([[1, 5, 9, 3]])
        a, b = f32(_ref_model(tokens)), f32(self.streamed.model(tokens))
        self.assertLess(float(mx.abs(a - b).max()), 0.05 * float(mx.abs(a).max()))
        self.assertEqual(self.streamed.pool.stats()['resident_bytes'], 0)

    def test_cancellation_while_streaming(self):
        self.streamed.pool.cancel_check = lambda: True
        with self.assertRaises(Cancelled):
            self.streamed.model(mx.array([[1, 2, 3]]))

    def test_truncated_shard_after_load_fails_closed(self):
        ck = copy_checkpoint()
        streamed = load_streamed(ck, pool_bytes=0, config_check=permissive_config_check)
        try:
            streamed.reader.close()
            for shard in ck.glob('*.safetensors'):
                shard.write_bytes(shard.read_bytes()[:64])
            from models.nemotron35.streamed_experts import ExpertReadError
            with self.assertRaises(ExpertReadError):
                streamed.model(mx.array([[1, 2, 3]]))
        finally:
            streamed.close()

    def test_load_refuses_damaged_checkpoint_before_reading_weights(self):
        ck = copy_checkpoint()
        with open(ck / 'model-00001-of-00002.safetensors', 'ab') as handle:
            handle.write(b'x')
        with self.assertRaises(inv.CheckpointError):
            load_streamed(ck, pool_bytes=0, config_check=permissive_config_check)

    def test_public_config_validator_still_strict(self):
        with self.assertRaises(ConfigurationError):
            validate_config(TINY_CONFIG)


if __name__ == '__main__':
    unittest.main()
