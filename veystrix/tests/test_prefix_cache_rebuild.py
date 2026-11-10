"""Tests for static prefix cache reuse across rebuilt requests and explicit lookup outcomes."""
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from veystrix.server import ModelService, ContextLimitError
from veystrix.tests.test_server import FakeSession
from models.flashnext import prefix_cache as P
from models.flashnext.prefix_cache import (
    OUTCOME_RAM_HIT,
    OUTCOME_SSD_HIT,
    OUTCOME_PREFIX_MISMATCH,
    OUTCOME_STATE_INVALID,
    OUTCOME_EVICTED,
    OUTCOME_RESTORE_FAILED,
    OUTCOME_COLD,
)
from models.flashnext.prefix_session import PrefixSession, Report


class FakeArray:
    def __init__(self, shape, dtype="bfloat16"):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.nbytes = 1000

    def __getitem__(self, item):
        return self


class FakeKV:
    def __init__(self, offset=1000):
        self.offset = offset
        self.keys = FakeArray((1, 2, offset, 256))
        self.values = FakeArray((1, 2, offset, 256))
        self.index_keys = FakeArray((1, offset, 128))
        self.index_position_ids = FakeArray((1, offset), dtype="int32")


class FakeArrays:
    def __init__(self):
        self.cache = [FakeArray((1, 48, 128, 128)), FakeArray((1, 3, 1024))]


class PrefixCacheOutcomeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name
        self.fp = "test-fp-1234"
        self.cache = P.PrefixCache(self.fp, ram_budget=10 * 1024 * 1024, directory=self.cache_dir, disk_budget=50 * 1024 * 1024)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cold_outcome_on_empty_cache(self):
        ids = list(range(1000))
        hit, caches = self.cache.lookup(ids, lambda: [])
        self.assertIsNone(hit)
        self.assertIsNone(caches)
        res = self.cache.lookup(ids, lambda: [])
        self.assertEqual(res.outcome, OUTCOME_COLD)

    def test_ram_hit_outcome(self):
        tokens = tuple(range(600))
        entry = P.Entry(
            key=P.chain_key(self.fp, tokens),
            parent=None,
            n_tokens=len(tokens),
            parent_tokens=0,
            token_delta=tokens,
            layers=[],
            plan=(200, 400),
        )
        self.cache._admit(entry)
        self.cache._plans[entry.key] = entry.plan
        self.cache._tokens[entry.key] = tokens
        self.cache._lengths.add(len(tokens))

        prompt = list(range(800))
        res = self.cache.lookup(prompt, lambda: [], accept=lambda n, plan: True)
        self.assertEqual(res.outcome, OUTCOME_RAM_HIT)
        self.assertIsNotNone(res.hit)
        self.assertEqual(res.hit.n_tokens, 600)
        self.assertEqual(res.hit.tier, "ram")

    def test_prefix_mismatch_outcome_when_plan_rejected(self):
        tokens = tuple(range(600))
        entry = P.Entry(
            key=P.chain_key(self.fp, tokens),
            parent=None,
            n_tokens=len(tokens),
            parent_tokens=0,
            token_delta=tokens,
            layers=[],
            plan=(200, 400),
        )
        self.cache._admit(entry)
        self.cache._plans[entry.key] = entry.plan
        self.cache._tokens[entry.key] = tokens
        self.cache._lengths.add(len(tokens))

        prompt = list(range(800))
        # accept rejects this candidate
        res = self.cache.lookup(prompt, lambda: [], accept=lambda n, plan: False)
        self.assertEqual(res.outcome, OUTCOME_PREFIX_MISMATCH)
        self.assertIsNone(res.hit)

    def test_evicted_outcome_when_entry_dropped(self):
        tokens = tuple(range(600))
        key = P.chain_key(self.fp, tokens)
        self.cache._evicted_keys[key] = 12345.0
        self.cache._lengths.add(len(tokens))

        prompt = list(range(800))
        res = self.cache.lookup(prompt, lambda: [])
        self.assertEqual(res.outcome, OUTCOME_EVICTED)
        self.assertIsNone(res.hit)


class PrefixCacheRebuildAdoptionTests(unittest.TestCase):
    def test_adopt_subtracts_reusable_tokens_before_check_prefill(self):
        session = FakeSession([])
        backend = MagicMock()
        backend.pending = []
        backend.tape = []
        backend.encode = lambda text: list(range(len(text)))
        backend.common_prefix = lambda ids: 0

        # Simulate 8,000 reusable prefix tokens
        mock_prefix = MagicMock()
        mock_prefix.reusable_tokens.return_value = 8000
        backend._prefix_session = lambda: mock_prefix

        session.backend = backend
        # Context limit: 16384, max_cold_prefill: 4096
        # A request of 10,000 tokens has 8,000 cached tokens -> only 2,000 cold tokens
        service = ModelService(session, context_limit=16384, max_cold_prefill=4096,
                               max_incremental_prefill=4096)

        # Mock prompt of 10,000 tokens
        prompt_ids = list(range(10000))
        # With reusable 8,000 tokens: needed_prefill is 10000 - 8000 = 2000 <= 4096 -> passes!
        adopted = service._adopt("dummy", ids=prompt_ids)
        self.assertFalse(adopted)  # extends was False, so it's a rebuilt turn
        self.assertEqual(service.rebuilt, 1)
        self.assertEqual(service.refused_cold_prefill, 0)

    def test_adopt_refuses_when_uncached_tokens_exceed_budget(self):
        session = FakeSession([])
        backend = MagicMock()
        backend.pending = []
        backend.tape = []
        backend.encode = lambda text: list(range(len(text)))
        backend.common_prefix = lambda ids: 0

        # Simulate only 2,000 reusable tokens out of 10,000
        mock_prefix = MagicMock()
        mock_prefix.reusable_tokens.return_value = 2000
        backend._prefix_session = lambda: mock_prefix

        session.backend = backend
        service = ModelService(session, context_limit=16384, max_cold_prefill=4096,
                               max_incremental_prefill=4096)

        prompt_ids = list(range(10000))
        # 10000 - 2000 = 8000 > the 4096 incremental limit -> refuse.
        with self.assertRaises(ContextLimitError) as cm:
            service._adopt("dummy", ids=prompt_ids)
        self.assertEqual(service.refused_cold_prefill, 1)
        self.assertIn("prefill_budget_exceeded", getattr(cm.exception, "code", ""))


if __name__ == "__main__":
    unittest.main()
