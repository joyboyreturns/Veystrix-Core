import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.flashnext import decode_profile, expert_cache


def _events(tokens=3, layers=4, step=0.01):
    out = []
    now = 100.0
    for _ in range(tokens):
        for layer in range(layers):
            out.append((now, layer, [layer, layer + 1]))
            now += step
    return out


class GroupingTests(unittest.TestCase):
    def test_tokens_split_when_layer_restarts(self):
        tokens = decode_profile.tokens_from_events(_events(3, 4))
        self.assertEqual(len(tokens), 3)
        self.assertEqual([len(t["layers"]) for t in tokens], [4, 4, 4])
        self.assertEqual(tokens[0]["routes"]["2"], [2, 3])

    def test_summary_token_ms_uses_next_token_start(self):
        events = _events(3, 4, 0.01)
        summary = decode_profile.summarize(events, 99.0, 100.12)
        self.assertEqual(len(summary["tokens"]), 3)
        self.assertAlmostEqual(summary["tokens"][0]["ms"], 40.0, places=3)
        self.assertAlmostEqual(summary["tokens"][2]["ms"], 40.0, places=3)
        self.assertEqual(len(summary["tokens"][0]["layer_gap_ms"]), 3)

    def test_empty_events(self):
        self.assertEqual(decode_profile.summarize([], 0.0, 1.0)["tokens"], [])

    def test_split_uses_recorded_layer_ids_for_sparse_traces(self):
        from models.flashnext.tests.bench.bench_decode_profile import split

        events = [(100.0, 2, [1]), (100.01, 5, [2]),
                  (101.0, 2, [1]), (101.02, 5, [2])]
        body = decode_profile.summarize(events, 99.0, 102.0)
        self.assertEqual(body["tokens"][1]["layer_ids"], [2, 5])
        self.assertEqual(split(body)["layer_gap_ms_mean"], {"2": 20.0})
        # Profiles recorded before layer_ids existed still retain route order.
        for token in body["tokens"]:
            del token["layer_ids"]
        self.assertEqual(split(body)["layer_gap_ms_p95"], {"2": 20.0})


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        decode_profile._STATE.update(active=False, events=[])
        expert_cache.set_route_trace(None)

    def tearDown(self):
        expert_cache.set_route_trace(None)
        decode_profile._STATE.update(active=False, events=[])

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(decode_profile.FLAG_DECODE_PROFILE_DIR, None)
            self.assertFalse(decode_profile.enabled())
            self.assertFalse(decode_profile.begin())
            self.assertIsNone(decode_profile.finish())
            self.assertIsNone(expert_cache._ROUTE_TRACE[0])

    def test_records_and_writes_then_uninstalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {decode_profile.FLAG_DECODE_PROFILE_DIR: tmp}):
                self.assertTrue(decode_profile.begin())
                self.assertIs(expert_cache._ROUTE_TRACE[0], decode_profile._record)
                for layer in range(3):
                    expert_cache._ROUTE_TRACE[0](layer, [1, 2, 3])
                for layer in range(3):
                    expert_cache._ROUTE_TRACE[0](layer, [4, 5, 6])
                path = decode_profile.finish()
                self.assertIsNotNone(path)
                body = json.loads(Path(path).read_text())
                self.assertEqual(len(body["tokens"]), 2)
                self.assertEqual(body["tokens"][1]["routes"]["0"], [4, 5, 6])
                self.assertIn("timers", body)
                self.assertIn("host_windows", body)
                self.assertIn("expert_io", body)
                self.assertIsNone(expert_cache._ROUTE_TRACE[0])
                self.assertFalse(decode_profile.active())

    def test_begin_is_not_reentrant(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {decode_profile.FLAG_DECODE_PROFILE_DIR: tmp}):
                self.assertTrue(decode_profile.begin())
                self.assertFalse(decode_profile.begin())
                decode_profile.finish()

    def test_finish_failure_does_not_raise_and_clears_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {decode_profile.FLAG_DECODE_PROFILE_DIR: tmp}):
                decode_profile.begin()
                with mock.patch.object(Path, "write_text", side_effect=OSError("disk")):
                    self.assertIsNone(decode_profile.finish())
                self.assertFalse(decode_profile.active())
                self.assertIsNone(expert_cache._ROUTE_TRACE[0])


if __name__ == "__main__":
    unittest.main()
