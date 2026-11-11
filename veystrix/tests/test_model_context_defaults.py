import unittest
from argparse import Namespace

from models.flashnext.overnight import server_command
from models.nemotron35 import admission


class ModelContextDefaultsTests(unittest.TestCase):
    def test_nemotron_launcher_requests_full_window(self):
        command = server_command(Namespace(model='nemotron35-lightning', host='127.0.0.1', port=8080,
                                           pool_gb='auto', checkpoint='/model'))
        self.assertEqual(command[command.index('--context-limit') + 1], '262144')

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

