"""Full-size structure checks that allocate no weights (MLX arrays stay lazy and are never evaluated)."""
import json
import os
import unittest
from pathlib import Path

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import nemotron_h

from models.nemotron35 import inventory as inv
from models.nemotron35.config import EXPECTED, LAYER_ORDER
from models.nemotron35.streamed_experts import StreamedSwitchMLP
from models.nemotron35.streamed_model import _stub_switch_mlp

REAL = os.environ.get("VEYSTRIX_NEMOTRON_TEST_CHECKPOINT", "")


def full_config():
    config = dict(EXPECTED)
    config.update(layers_block_type=list(LAYER_ORDER), vocab_size=131072, n_group=1, max_position_embeddings=262144,
                  intermediate_size=1856, attention_bias=False, mamba_proj_bias=False, mlp_bias=False,
                  layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True, num_hidden_layers=52,
                  quantization={'group_size': 64, 'bits': 4, 'mode': 'affine'})
    return config


class FullSizeStructureTests(unittest.TestCase):
    def test_model_parameters_match_inventory_exactly(self):
        config = full_config()
        expected = inv.expected_tensors(config)
        args = nemotron_h.ModelArgs.from_dict(config)
        with _stub_switch_mlp(lambda *a, **k: StreamedSwitchMLP(None, 0, 64)):
            model = nemotron_h.Model(args)
        dense = {n for n in expected if '.switch_mlp.' not in n}
        nn.quantize(model, group_size=64, bits=4, mode='affine',
                    class_predicate=lambda path, m: hasattr(m, 'to_quantized') and f'{path}.scales' in dense)
        actual = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
        self.assertEqual(set(actual), dense)
        for name in dense:
            self.assertEqual(actual[name], tuple(expected[name][0]), name)

    @unittest.skipUnless(REAL, "set VEYSTRIX_NEMOTRON_TEST_CHECKPOINT to a downloaded checkpoint (metadata only)")
    def test_real_checkpoint_metadata_verifies_without_loading_weights(self):
        summary = inv.verify_checkpoint(REAL)
        self.assertEqual(summary['tensors'], 729)
        self.assertEqual(summary['total_bytes'], 17775339392)


if __name__ == '__main__':
    unittest.main()
