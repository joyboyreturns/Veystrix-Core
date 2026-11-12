import os
import unittest
from unittest.mock import patch
from argparse import Namespace

from models.flashnext.overnight import server_command
from models.nemotron35 import admission


class ModelContextDefaultsTests(unittest.TestCase):
    def test_nemotron_launcher_defers_context_to_backend_and_forwards_shared_root(self):
        command = server_command(Namespace(model='nemotron35-lightning', host='127.0.0.1', port=8080,
                                           pool_gb='auto', checkpoint='/model', prefix_dir='/cache'))
        self.assertNotIn('--context-limit', command)
        self.assertEqual(command[command.index('--prefix-dir') + 1], '/cache')

    def test_nemotron_default_and_memory_guard(self):
        config = {'max_position_embeddings': 262144, 'layers_block_type': ['mamba', 'attention'],
                  'mamba_num_heads': 2, 'mamba_head_dim': 4, 'ssm_state_size': 8,
                  'n_groups': 1, 'conv_kernel': 4, 'num_key_value_heads': 2, 'head_dim': 4}
        machine = admission.Machine(total=16 * admission.GIB, available=12 * admission.GIB,
                                    swap_used=0, pressure=1, gpu_limit=12 * admission.GIB)
        plan = admission.make_plan(config, machine, dense_bytes=0, pool_bytes=0)
        self.assertEqual(plan.context_tokens, 262144)
        with self.assertRaises(admission.AdmissionRefused):
            admission.make_plan(config, machine, dense_bytes=13 * admission.GIB, pool_bytes=0)


    def test_lightning_int8_defaults_to_one_million_with_64k_prefill_limits(self):
        from models.nemotron35.config import EXPECTED, LAYER_ORDER
        from veystrix.session import _apply_backend_limits

        config = dict(EXPECTED, layers_block_type=LAYER_ORDER, max_position_embeddings=262_144)
        gib = admission.GIB
        machine = admission.Machine(16 * gib, 10 * gib, 0, 1, 12 * gib)
        plan = admission.make_plan(config, machine, dense_bytes=gib, pool_bytes=0,
                                   int8_kv=True, lightning=True)
        self.assertEqual(plan.context_tokens, 1_048_576)
        backend = Namespace(context_limit=plan.context_tokens, max_cold_prefill=admission.PREFILL_LIMIT,
                            max_incremental_prefill=admission.PREFILL_LIMIT)
        with patch.dict(os.environ, {"VEYSTRIX_MAX_INCREMENTAL_PREFILL": "1048576"}, clear=True):
            _apply_backend_limits(backend)
            self.assertEqual(os.environ["VEYSTRIX_CONTEXT_LIMIT"], "1048576")
            self.assertEqual(os.environ["VEYSTRIX_MAX_COLD_PREFILL"], "65536")
            self.assertEqual(os.environ["VEYSTRIX_MAX_INCREMENTAL_PREFILL"], "65536")
