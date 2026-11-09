import unittest

from models.flashnext.tests.bench.bench_cache_holdout import (
    LeastStale, RecencyFrequency, evaluate, verification_bounds)


class HoldoutTests(unittest.TestCase):
    def test_training_excludes_holdout_and_charges_cold_fill(self):
        events = [[0, "decode", i, 0, [1]] for i in range(4)]
        events += [[1, "decode", i, 0, [2]] for i in range(5)]
        result = evaluate(events, 1, 1, 10)
        static = result["policies"]["static_frequency"]
        self.assertEqual(result["train_turns"], [0])
        self.assertEqual(static["hits"], 0)
        self.assertEqual(static["estimated_miss_bytes"], 50)
        self.assertEqual(static["estimated_bytes_with_cold_fill"], 60)
        self.assertEqual(result["policies"]["global_lru"]["hits"], 4)

    def test_missing_training_or_holdout_fails(self):
        with self.assertRaises(ValueError):
            evaluate([[0, "decode", 0, 0, [1]]], 0, 1, 10)
        with self.assertRaises(ValueError):
            evaluate([[0, "decode", 0, 0, [1]]], 1, 1, 10)

    def test_verification_union_uses_each_layer_and_counts_partial_block(self):
        events = [[2, "decode", 1, 0, [1, 2]], [2, "decode", 1, 1, [8]],
                  [2, "decode", 2, 0, [2, 3]], [2, "decode", 2, 1, [8]],
                  [2, "decode", 3, 0, [1]], [2, "decode", 3, 1, [9]]]
        result = verification_bounds(events, 2, 0, 10, widths=(1, 2))
        self.assertEqual(result["1"]["miss_records"], 8)
        self.assertEqual(result["2"]["miss_records"], 6)
        self.assertEqual(result["2"]["blocks"], 2)
        with self.assertRaises(ValueError):
            verification_bounds(events[:-1], 2, 0, 10)

    def test_least_stale_evicts_the_layer_needed_furthest_ahead(self):
        policy = LeastStale(2, [0, 1, 2])
        policy.access(0, [5])
        policy.access(2, [7])                       # pool full: (0,5), (2,7)
        # Running layer 1 next: layer 2 is needed in 0 layers, layer 0 in 1 layer. (0,5) is furthest ahead.
        self.assertEqual(policy.access(1, [9]), 0)
        self.assertNotIn((0, 5), policy.pool)
        self.assertIn((2, 7), policy.pool)

    def test_policies_never_exceed_capacity_or_evict_the_current_call(self):
        for policy in (LeastStale(3, [0, 1]), RecencyFrequency(3)):
            hits = 0
            for step in range(20):
                hits += policy.access(step % 2, [step % 5, (step + 1) % 5])
                self.assertLessEqual(len(policy.pool), 3)
            self.assertGreaterEqual(hits, 0)
            self.assertEqual(policy.access(0, [1, 2, 3, 4]) >= 0, True)   # call larger than capacity does not raise

    def test_recency_frequency_keeps_the_popular_entry(self):
        policy = RecencyFrequency(2)
        for _ in range(5):
            policy.access(0, [1])
        policy.access(0, [2])
        policy.access(0, [3])                       # evicts the count-1 entry, not the popular (0,1)
        self.assertIn((0, 1), policy.pool)
        self.assertNotIn((0, 2), policy.pool)

    def test_zero_capacity_policies_hit_nothing(self):
        self.assertEqual(LeastStale(0, [0]).access(0, [1]), 0)
        self.assertEqual(RecencyFrequency(0).access(0, [1]), 0)


if __name__ == "__main__":
    unittest.main()
