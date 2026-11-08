"""The offline long-context budget: structure, calibration at 1K, and ordering."""
from __future__ import annotations

import unittest

from models.flashnext.tests.bench import long_context_budget as B


class BudgetTests(unittest.TestCase):
    def test_conv_dim_and_state_sizes_match_the_config(self):
        self.assertEqual(B.CONV_DIM, 10_240)
        self.assertEqual(B.GDN_STATE_BYTES, 3_145_728 + 3 * 10_240 * 2)
        self.assertEqual(B.LINEAR_LAYERS, 36)

    def test_persistent_state_is_small_and_linear_in_tokens(self):
        one, ten = B.state_and_kv(1000), B.state_and_kv(10_000)
        self.assertLess(ten, 0.5e9)
        self.assertAlmostEqual((ten - one) / 9000, B.FULL_LAYERS * (B.KV_BYTES_PER_TOKEN + B.INDEX_BYTES_PER_TOKEN), delta=1)

    def test_full_logits_only_up_to_the_prefill_threshold(self):
        self.assertEqual(B.logits_bytes(1025), 1025 * 248_320 * 2)
        self.assertGreater(B.logits_bytes(2048), 1.0e9)
        self.assertEqual(B.logits_bytes(2049), 0.0)
        self.assertEqual(B.logits_bytes(10_000), 0.0)
        self.assertAlmostEqual(10_000 * 248_320 * 2 / 1e9, 4.97, places=2)   # the figure in the brief

    def test_calibrated_to_the_measured_1k_run(self):
        # Measured at 1,025 tokens: MLX active 5.96 GB plus allocator cache 1.35 GB at the end of prefill.
        measured = 5.96e9 + 1.35e9
        projected = B.budget(1025).peak(False)
        self.assertAlmostEqual(projected / measured, 1.0, delta=0.10)
        self.assertGreaterEqual(projected, measured, "the projection must not undershoot the measured run")

    def test_retained_fit_reproduces_the_three_measured_points(self):
        for tokens, measured in ((129, 0.233e9), (257, 0.345e9), (1025, 0.992e9)):
            fit = B.RETAINED_INTERCEPT + B.RETAINED_SLOPE * tokens
            self.assertAlmostEqual(fit, measured, delta=0.06e9)

    def test_as_is_grows_faster_than_with_the_fix(self):
        for small, large in ((1024, 2048), (2048, 4096), (4096, 8192), (8192, 10_000)):
            gap_small = B.budget(small).peak(False) - B.budget(small).peak(True)
            gap_large = B.budget(large).peak(False) - B.budget(large).peak(True)
            self.assertGreater(gap_large, gap_small)

    def test_ten_thousand_tokens_do_not_fit_as_is_and_fit_with_the_fix(self):
        self.assertFalse(B.budget(10_000).fits(False))
        self.assertTrue(B.budget(10_000).fits(True))
        self.assertFalse(B.budget(8192).fits(False))

    def test_attention_scores_are_bounded_by_the_query_chunk(self):
        self.assertEqual(B.attention_score_bytes(1024), 1024 * 1024 * 48)
        self.assertEqual(B.attention_score_bytes(10_000), B.QSA_QUERY_CHUNK * 10_000 * 48)

    def test_chunking_bounds_the_transients_but_not_the_cache(self):
        whole, chunked = B.budget(10_000), B.budget(10_000, 2048)
        self.assertLess(chunked.transient, whole.transient / 4)
        self.assertLess(chunked.attention_scores, whole.attention_scores * 1.01)
        self.assertEqual(B.state_and_kv(10_000), chunked.retained_fixed)
        self.assertLess(chunked.peak(True), whole.peak(True))
        self.assertEqual(B.budget(1500, 2048).transient, B.budget(1500).transient, "a short prompt is one chunk")

    def test_smaller_chunks_use_less_memory_at_10k(self):
        self.assertLess(B.budget(10_000, 1024).peak(True), B.budget(10_000, 2048).peak(True))

    def test_table_lists_every_size(self):
        text = B.table()
        for size in ("1024", "2048", "4096", "8192", "10000"):
            self.assertIn(size, text)


if __name__ == "__main__":
    unittest.main()
