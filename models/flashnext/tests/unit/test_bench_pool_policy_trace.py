import gzip
import json
import os
import tempfile
import unittest

from models.flashnext.tests.bench import bench_pool_policy_trace as B


def decode(*layers_experts):
    return [[0, "decode", i, layer, list(experts)] for i, (layer, experts) in enumerate(layers_experts)]


class PolicyTests(unittest.TestCase):
    def test_lru_hand_counted(self):
        # 2 slots, one layer. Accesses: [1,2] miss miss; [1] hit; [3] evicts 2; [2] misses again.
        events = decode((0, (1, 2)), (0, (1,)), (0, (3,)), (0, (2,)))
        hit, _ = B.replay(events, 2, "lru")
        self.assertEqual(hit, 1 / 5)

    def test_lru_cyclic_scan_larger_than_pool_never_hits_but_bip_does(self):
        # Textbook scan: 10 experts walked cyclically one per event against 8 slots, so LRU always evicts the next one.
        events = decode(*[(0, (i % 10,)) for i in range(200)])
        lru, _ = B.replay(events, 8, "lru")
        bip, _ = B.replay(events, 8, "bip8")
        self.assertEqual(lru, 0.0)
        self.assertGreater(bip, 0.1)

    def test_belady_is_an_upper_bound(self):
        events = decode(*[(i % 3, tuple((i * j) % 7 for j in range(1, 4))) for i in range(200)])
        best = B.replay(events, 6, "belady")[0]
        for name in ("lru", "arc", "bip8"):
            self.assertLessEqual(B.replay(events, 6, name)[0], best + 1e-12, name)

    def test_current_event_entries_are_not_victims(self):
        pool = B.Lru(2)
        pool.access(0, (1, 2))
        pool.access(0, (3, 4))                      # 4 entries wanted, 2 slots: first two are admitted, rest skipped
        self.assertLessEqual(len(pool.pool), 2)

    def test_prefill_only_queries_and_never_fills(self):
        events = [[0, "prefill", 0, 0, [1, 2]], [0, "decode", 0, 0, [1]], [0, "prefill", 0, 0, [1, 2]]]
        decode_rate, prefill_rate = B.replay(events, 4, "lru")
        self.assertEqual(decode_rate, 0.0)           # first decode access is a miss
        self.assertEqual(prefill_rate, 1 / 4)        # only expert 1 resident for the second prefill

    def test_all_policies_never_exceed_slots(self):
        events = decode(*[(i % 2, tuple((i + j) % 11 for j in range(4))) for i in range(300)])
        for name in ("lru", "arc", "bip8"):
            pool = B.make_policy(name, 5, [])
            for _, _, _, layer, experts in events:
                pool.access(layer, experts)
            size = len(pool.pool) if hasattr(pool, "pool") else len(pool.t1) + len(pool.t2)
            self.assertLessEqual(size, 5, name)

    def test_main_reads_trace_file(self):
        data = {"record_bytes": 1_000_000, "events": decode((0, (1, 2)), (0, (1, 2)))}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "t.json.gz")
            with gzip.open(path, "wt") as handle:
                json.dump(data, handle)
            events, record = B.load_trace(path)
            self.assertEqual((len(events), record), (2, 1_000_000))
            B.main([path, "--gb", "0.01", "--policies", "lru"])


if __name__ == "__main__":
    unittest.main()
