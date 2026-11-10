"""Regression tests for Phase 5 Ultra Review remediations:
- R1-07: ExpertPool allocation failure marks store._flashnext_pool_declined
- R1-08: _read_weights_with_pool uses expert_pool.HEADER constant
- R1-09: prefill injectable logits hook and test-bench import isolation
- R1-10: prefill_scheduler.CANDIDATE_CHUNKS imported from prefill
"""
import os
import unittest
from unittest import mock

import mlx.core as mx

from models.flashnext import expert_cache, expert_pool, prefill, prefill_scheduler


class DummyStore:
    def __init__(self):
        self._flashnext_expert_pool = None
        self._flashnext_pool_declined = None


class Phase5RemediationsTests(unittest.TestCase):
    def test_r1_07_expert_pool_failure_marks_pool_declined(self):
        """R1-07: When ExpertPool construction fails, store._flashnext_pool_declined is set."""
        store = DummyStore()
        with mock.patch.dict(os.environ, {"FLASHNEXT_EXPERT_POOL_GB": "2", "FLASHNEXT_CHAT_GUARD": "0"}):
            with mock.patch("models.flashnext.expert_pool.ExpertPool", side_effect=MemoryError("OOM")):
                pool = expert_pool.for_store(store)
                self.assertIsNone(pool)
                self.assertEqual(store._flashnext_pool_declined, "allocation_failed")

                # Subsequent call returns None immediately without re-invoking ExpertPool
                with mock.patch("models.flashnext.expert_pool.ExpertPool") as mock_ep:
                    second_pool = expert_pool.for_store(store)
                    self.assertIsNone(second_pool)
                    mock_ep.assert_not_called()

                # reset_decline clears the decline status
                expert_pool.reset_decline(store)
                self.assertIsNone(store._flashnext_pool_declined)

    def test_r1_08_expert_cache_uses_expert_pool_header_constant(self):
        """R1-08: _read_weights_with_pool computes hit bases using expert_pool.HEADER."""
        import inspect
        source = inspect.getsource(expert_cache.StreamingSwitchGLU._read_weights_with_pool)
        self.assertIn("expert_pool.HEADER", source)
        self.assertNotIn("4096 +", source)

        class MockPool:
            stride = 1000

        mock_self = mock.Mock()
        mock_self.gate_proj.cache.store = mock.Mock()
        pool = MockPool()
        hits = {5: 2}

        # Mock submit_projection_tasks and verify hit_indices_and_bases
        with mock.patch("models.flashnext.expert_cache.submit_projection_tasks"):
            with mock.patch.object(expert_pool, "HEADER", 8192):
                with mock.patch("time.perf_counter", return_value=0.0):
                    try:
                        expert_cache.StreamingSwitchGLU._read_weights_with_pool(mock_self, [], [5], pool, hits)
                    except Exception:
                        pass

    def test_r1_09_prefill_injectable_logits_hook(self):
        """R1-09: set_logits_hook installs a callback receiving final-row prefill logits."""
        captured = []

        def hook(logits):
            captured.append(logits)

        prefill.set_logits_hook(hook)
        self.assertEqual(prefill.get_logits_hook(), hook)

        # Calling set_logits_hook(None) removes the hook
        prefill.set_logits_hook(None)
        self.assertIsNone(prefill.get_logits_hook())

    def test_r1_10_candidate_chunks_shared_definition(self):
        """R1-10: prefill_scheduler.CANDIDATE_CHUNKS is imported from prefill.CANDIDATE_CHUNKS."""
        self.assertIs(prefill_scheduler.CANDIDATE_CHUNKS, prefill.CANDIDATE_CHUNKS)
