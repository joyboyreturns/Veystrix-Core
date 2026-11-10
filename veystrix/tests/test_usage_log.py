import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from veystrix import usage_log


class UsageLogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "usage" / "requests.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def counts(self, prompt=100, cached=60, completion=20):
        return {"prompt": prompt, "cached": cached, "completion": completion}

    def stats(self, **kw):
        return SimpleNamespace(**{"prefill_seconds": 1.5, "seconds": 4.0, "finish": "stop", **kw})

    def test_record_has_counts_timings_and_nothing_else(self):
        record = usage_log.build_record("veystrix-flashnext", self.counts(), self.stats(), 6.2, now=1.7e9, warm=False)
        self.assertEqual(record, {"ts": 1.7e9, "model": "veystrix-flashnext", "prompt": 100, "cached": 60, "uncached": 40,
                                  "completion": 20, "total": 120, "wall_s": 6.2, "prefill_s": 1.5, "decode_s": 4.0,
                                  "finish": "stop", "warm": False})

    def test_append_writes_one_json_line_per_call_with_private_permissions(self):
        for n in range(3):
            self.assertTrue(usage_log.append(usage_log.build_record("m", self.counts(completion=n), self.stats(), 1.0), self.path))
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([r["completion"] for r in rows], [0, 1, 2])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_env_zero_turns_recording_off_and_env_path_is_used(self):
        service = SimpleNamespace(model="m")
        with mock.patch.dict(os.environ, {usage_log.ENV: "0"}):
            usage_log.record_request(service, self.stats(), 1.0, lambda s, st: self.counts())
        self.assertFalse(self.path.exists())
        with mock.patch.dict(os.environ, {usage_log.ENV: str(self.path)}):
            usage_log.record_request(service, self.stats(), 1.0, lambda s, st: self.counts())
        self.assertEqual(len(self.path.read_text().splitlines()), 1)

    def test_a_failing_write_or_counts_function_never_raises(self):
        blocked = Path(self.dir.name) / "file"
        blocked.write_text("x")
        self.assertFalse(usage_log.append({"a": 1}, blocked / "child" / "requests.jsonl"))
        with mock.patch.dict(os.environ, {usage_log.ENV: str(self.path)}):
            usage_log.record_request(SimpleNamespace(model="m"), self.stats(), 1.0, lambda s, st: 1 / 0)
        self.assertFalse(self.path.exists())

    def test_rotation_archives_and_never_loses_a_record(self):
        with mock.patch.object(usage_log, "MAX_BYTES", 200):
            for _ in range(30):
                usage_log.append(usage_log.build_record("m", self.counts(), self.stats(), 1.0), self.path)
        files = list(self.path.parent.glob("requests*.jsonl"))
        self.assertGreater(len(files), 2)          # several archives, none overwritten
        self.assertEqual(sum(len(p.read_text().splitlines()) for p in files), 30)

    def test_warm_flag_is_per_thread_and_defaults_to_false(self):
        usage_log.set_warm(True)
        self.assertTrue(usage_log.build_record("m", self.counts(), self.stats(), 1.0)["warm"])
        usage_log.set_warm(False)
        self.assertFalse(usage_log.build_record("m", self.counts(), self.stats(), 1.0)["warm"])


if __name__ == "__main__":
    unittest.main()
