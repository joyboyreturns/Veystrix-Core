"""Synthetic test suite for Native 262,144-token Context without 262K Cold Prefill.

Verifies:
1. Cold requests at 32K, 64K, 128K, 200K, 262K without cache are rejected immediately
   with OversizedPrefillError (code="oversized_prefill_rejected") without model execution.
2. Incremental accumulation ladder succeeds from 8K up to 262K when each step's
   uncached prefill is <= 16,384 tokens.
3. Lost or evicted cache on long-context conversations (>16K) raises RecoveryRequiredError
   (code="recovery_required") rather than computing a giant cold prefill.
4. SSD/RAM snapshot restoration restores prefix state and permits continuation.
5. Exact 262,144 context boundaries and output token reservations are enforced.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from veystrix import preferences, context_budget
from veystrix.server import (
    ModelService,
    ContextLimitError,
    OversizedPrefillError,
    RecoveryRequiredError,
    MIN_OUTPUT_RESERVE,
    MAX_COLD_PREFILL_DEFAULT,
    MAX_INCREMENTAL_PREFILL_DEFAULT,
    ARCHITECTURAL_CONTEXT_MAXIMUM,
)
from veystrix.tests.test_server import FakeSession


class MockTokenBackend:
    def __init__(self):
        self.tokenizer = MagicMock()
        self.tokenizer.apply_chat_template = lambda messages, **kwargs: "mock_rendered"
        self.tape: list[int] = []
        self.pending: list[int] = []
        self._replay_needed = False
        self.reset_count = 0
        self.model_path = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
        self._prefix_mock = None
        self._mock_ids: list[int] = []
        self.generate = MagicMock(return_value=("output", MagicMock(tokens=10, prompt_tokens=0, finish="stop")))

    def encode(self, text: str) -> list[int]:
        if self._mock_ids:
            return list(self._mock_ids)
        return [1] * len(text)

    def common_prefix(self, ids: list[int]) -> int:
        shared = 0
        for a, b in zip(self.tape, ids):
            if a == b:
                shared += 1
            else:
                break
        return shared

    def append_tokens(self, tokens: list[int]) -> None:
        self.tape.extend(tokens)

    def append_text(self, text: str) -> None:
        pass

    def reset(self) -> None:
        self.reset_count += 1
        self.tape.clear()
        self.pending.clear()

    def _prefix_session(self):
        return self._prefix_mock


class Incremental262KContextTests(unittest.TestCase):
    def setUp(self):
        self.backend = MockTokenBackend()
        self.session = FakeSession([])
        self.session.backend = self.backend
        self.session.preferences = dict(
            preferences.DEFAULTS,
            model="flashnext",
            thinking_enabled=False,
            effort="medium",
        )
        self.service = ModelService(self.session, context_limit=262144)

    def test_cold_oversized_prefill_rejected_at_ladder_steps(self):
        """Cold requests above 32,768 tokens must be rejected without model computation (owner decision 2026-11-09)."""
        ladder = [32769, 49152, 65536, 131072, 200000, 250000]
        for size in ladder:
            with self.subTest(size=size):
                self.backend.reset()
                self.backend._mock_ids = list(range(size))
                self.backend.generate.reset_mock()

                # Verify rejection before model execution
                with self.assertRaises(OversizedPrefillError) as cm:
                    self.service.complete([{"role": "user", "content": "cold_prompt"}], [], 1024)

                err = cm.exception
                self.assertEqual(err.code, "oversized_prefill_rejected")
                self.assertEqual(err.status, 400)
                self.assertIn("exceeds the maximum cold prefill limit", str(err))
                self.assertIn("32768", str(err))
                self.assertIn("incremental state accumulation", str(err))
                self.backend.generate.assert_not_called()

    def test_incremental_accumulation_ladder_succeeds_up_to_262k(self):
        """Context grows incrementally from 8K up to 260K+ tokens within the 16K per-turn prefill limit."""
        # Step 1: Initial cold turn within 16K limit (e.g. 8,192 tokens)
        self.backend.reset()
        t1_ids = list(range(8192))
        self.backend._mock_ids = t1_ids
        res = self.service.complete([{"role": "user", "content": "t1"}], [], 512)
        self.assertEqual(len(self.backend.tape), 8192)
        self.assertEqual(self.service.rebuilt, 1)
        self.assertEqual(self.service.reused, 0)

        # Step 2: Incremental turn (+8,192 new tokens -> 16,384 total)
        t2_ids = list(range(16384))
        self.backend._mock_ids = t2_ids
        self.service.complete([{"role": "user", "content": "t2"}], [], 512)
        self.assertEqual(len(self.backend.tape), 16384)
        self.assertEqual(self.service.reused, 1)

        # Step 3: Incremental turn (+16,384 new tokens -> 32,768 total)
        t3_ids = list(range(32768))
        self.backend._mock_ids = t3_ids
        self.service.complete([{"role": "user", "content": "t3"}], [], 512)
        self.assertEqual(len(self.backend.tape), 32768)
        self.assertEqual(self.service.reused, 2)

        # Step 4: Incremental turn (+16,384 new tokens -> 49,152 total)
        t4_ids = list(range(49152))
        self.backend._mock_ids = t4_ids
        self.service.complete([{"role": "user", "content": "t4"}], [], 512)
        self.assertEqual(len(self.backend.tape), 49152)
        self.assertEqual(self.service.reused, 3)

        # Step 5: Incremental turn (+16,384 new tokens -> 65,536 total)
        t5_ids = list(range(65536))
        self.backend._mock_ids = t5_ids
        self.service.complete([{"role": "user", "content": "t5"}], [], 512)
        self.assertEqual(len(self.backend.tape), 65536)
        self.assertEqual(self.service.reused, 4)

        # Step 6: Multi-step jump to 128K via increments
        curr = 65536
        while curr < 128000:
            step = min(16000, 128000 - curr)
            curr += step
            self.backend._mock_ids = list(range(curr))
            self.service.complete([{"role": "user", "content": f"t_{curr}"}], [], 512)

        self.assertEqual(len(self.backend.tape), 128000)

        # Step 7: Jump to 200K via increments
        while curr < 200000:
            step = min(16000, 200000 - curr)
            curr += step
            self.backend._mock_ids = list(range(curr))
            self.service.complete([{"role": "user", "content": f"t_{curr}"}], [], 512)

        self.assertEqual(len(self.backend.tape), 200000)

        # Step 8: Jump to 260,000 tokens
        while curr < 260000:
            step = min(15000, 260000 - curr)
            curr += step
            self.backend._mock_ids = list(range(curr))
            self.service.complete([{"role": "user", "content": f"t_{curr}"}], [], 512)

        self.assertEqual(len(self.backend.tape), 260000)
        self.assertGreater(self.service.reused, 10)
        self.assertEqual(self.service.refused_cold_prefill, 0)

    def test_lost_cache_on_long_context_raises_recovery_required(self):
        """A 200,000-token session that lost its cache raises RecoveryRequiredError."""
        # Simulate an ongoing session by setting warm tokens and then resetting tape (cache loss)
        self.backend.tape = list(range(198000))
        self.service.complete([{"role": "user", "content": "prior"}], [], 512)

        # Now simulate cache eviction or mismatch where only 1,000 tokens match
        self.backend.tape = list(range(1000))
        # Request arrives with 200,000 tokens
        self.backend._mock_ids = list(range(200000))

        with self.assertRaises(RecoveryRequiredError) as cm:
            self.service.complete([{"role": "user", "content": "lost_cache_turn"}], [], 512)

        err = cm.exception
        self.assertEqual(err.code, "recovery_required")
        self.assertEqual(err.status, 400)
        self.assertIn("session state could not be recovered", str(err))
        self.assertIn("200000", str(err))
        self.assertIn("prohibited on 16 GB hardware", str(err))

    def test_ram_or_ssd_snapshot_restoration_allows_continuation(self):
        """When an SSD/RAM snapshot restores a 64K prefix, a 66K prompt prefils only 2K delta."""
        self.backend.reset()
        # Mock prefix session reporting 64,000 reusable tokens
        mock_ps = MagicMock()
        mock_ps.reusable_tokens.return_value = 64000
        self.backend._prefix_mock = mock_ps

        # 66,000 token prompt -> needs only 2,000 incremental tokens prefilled
        self.backend._mock_ids = list(range(66000))
        self.backend.generate.reset_mock()

        # Should pass because needed_prefill (2,000) <= max_incremental_prefill (16,384)
        self.service.complete([{"role": "user", "content": "snapshot_continuation"}], [], 512)
        self.backend.generate.assert_called_once()
        self.assertEqual(self.service.refused_cold_prefill, 0)

    def test_context_boundary_and_output_reservation_at_262144(self):
        """Total context (prompt + reserved output) must not exceed 262,144."""
        # 1. Prompt at 261,888 tokens + 256 reserve = 262,144 -> passes context check
        self.backend.reset()
        self.backend.tape = list(range(260000))  # warm tape so incremental prefill is small
        self.backend._mock_ids = list(range(261888))
        self.service._check_context(261888)
        self.assertEqual(self.service.last_prompt_tokens, 261888)

        # 2. Prompt at 261,889 tokens + 256 reserve = 262,145 > 262,144 -> raises ContextLimitError
        with self.assertRaises(ContextLimitError) as cm:
            self.service._check_context(261889)
        self.assertEqual(cm.exception.code, "context_length_exceeded")
        self.assertIn("context limit exceeded", str(cm.exception))

        # 3. Output clamping when prompt is near boundary
        # 261,800 prompt tokens + 1,000 requested output -> clamped to 344 tokens
        self.backend._mock_ids = list(range(261800))
        self.backend.tape = list(range(260000))
        self.service.complete([{"role": "user", "content": "near_limit"}], [], max_tokens=1000)

        call_kwargs = self.backend.generate.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 344)
        self.assertEqual(self.service.last_budget["reserved_output"], 344)
        self.assertEqual(self.service.last_budget["remaining"], 0)


if __name__ == "__main__":
    unittest.main()
