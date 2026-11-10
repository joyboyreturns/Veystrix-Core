"""Per-request memory readiness, purge wiring, /health telemetry and the last-prefill record the warm-up reads."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.nemotron35 import admission, readiness
from models.nemotron35.tests.unit.test_backend import machine
from models.nemotron35.tiny_checkpoint import TINY_CONFIG, write_checkpoint

GIB = 2 ** 30
CFG = dict(TINY_CONFIG, max_position_embeddings=262144)


def plan(pool=2 * GIB):
    return admission.make_plan(CFG, machine(), dense_bytes=1_252_009_856, pool_bytes=pool)


class Reader:
    """A scripted machine: each read pops the next state (the last one repeats)."""
    def __init__(self, *states):
        self.states, self.calls = list(states), 0

    def __call__(self):
        self.calls += 1
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


class EnsureReadyTests(unittest.TestCase):
    def test_roomy_machine_is_admitted_without_any_recovery(self):
        reclaim, purge = [], []
        report = readiness.ensure_ready(Reader(machine(available=8 * GIB)), plan=plan(), resident_pool_bytes=0,
                                        reclaim=lambda: reclaim.append(1) or 0,
                                        purge=lambda n: purge.append(n) or ("ok", ""))
        self.assertTrue(report.admitted)
        self.assertEqual((reclaim, purge, report.steps), ([], [], []))

    def test_owned_memory_is_released_first_and_can_be_enough(self):
        reader = Reader(machine(available=GIB, pressure=2), machine(available=6 * GIB, pressure=1))
        purged = []
        report = readiness.ensure_ready(reader, plan=plan(), resident_pool_bytes=GIB, reclaim=lambda: 2 * GIB,
                                        purge=lambda n: purged.append(n) or ("ok", ""))
        self.assertEqual([s.name for s in report.steps], ["owned memory"])
        self.assertEqual(purged, [])           # no purge when releasing owned memory was enough
        self.assertTrue(report.admitted)

    def test_purge_runs_after_owned_memory_and_settles_before_the_next_reading(self):
        reader = Reader(machine(available=GIB, pressure=2), machine(available=GIB, pressure=2),
                        machine(available=6 * GIB, pressure=1))
        slept = []
        report = readiness.ensure_ready(reader, plan=plan(), resident_pool_bytes=0, reclaim=lambda: 0,
                                        purge=lambda n: ("purged", "recovered 5000 MB"), sleep=slept.append)
        self.assertEqual([s.name for s in report.steps], ["owned memory", "file-cache purge"])
        self.assertEqual(slept, [readiness.SETTLE_SECONDS])
        self.assertTrue(report.admitted)

    def test_short_under_pressure_is_refused_with_what_was_tried(self):
        reader = Reader(machine(available=GIB, pressure=2))
        with self.assertRaises(admission.AdmissionRefused) as caught:
            readiness.ensure_ready(reader, plan=plan(), resident_pool_bytes=0, reclaim=lambda: 0,
                                   purge=lambda n: ("skipped", "sudo -n /usr/sbin/purge is not permitted"))
        text = str(caught.exception)
        self.assertIn("owned memory", text)
        self.assertIn("not permitted", text)
        self.assertIn("pressure level 2", text)

    def test_short_without_pressure_proceeds(self):
        report = readiness.ensure_ready(Reader(machine(available=GIB // 2, pressure=1)), plan=plan(),
                                        resident_pool_bytes=0, reclaim=lambda: 0, purge=lambda n: ("skipped", "off"))
        self.assertTrue(report.admitted)

    def test_critical_pressure_never_attempts_a_purge(self):
        purged = []
        with self.assertRaises(admission.AdmissionRefused):
            readiness.ensure_ready(Reader(machine(available=GIB // 2, pressure=4)), plan=plan(), resident_pool_bytes=0,
                                   reclaim=lambda: 0, purge=lambda n: purged.append(n) or ("purged", ""))
        self.assertEqual(purged, [])

    def test_a_failing_recovery_step_never_fails_the_request_by_itself(self):
        def boom():
            raise RuntimeError("x")
        report = readiness.ensure_ready(Reader(machine(available=GIB // 2, pressure=1)), plan=plan(),
                                        resident_pool_bytes=0, reclaim=boom, purge=lambda n: 1 / 0)
        self.assertTrue(report.admitted)
        self.assertEqual({s.action for s in report.steps}, {"skipped"})

    def test_requirement_counts_unfilled_pool_but_bounds_it(self):
        small = readiness.required_bytes(plan(2 * GIB), 2 * GIB)
        big = readiness.required_bytes(plan(4 * GIB), 0)
        self.assertEqual(big - small, readiness.MAX_POOL_GROWTH_BYTES)


class BackendIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import models.nemotron35.tests.unit.test_speed_features as sf
        cls.dir = tempfile.mkdtemp(prefix="nemotron-ready-")
        write_checkpoint(Path(cls.dir) / "ck")
        cls.sf, cls.old = sf, sf._dir
        sf._dir = cls.dir

    @classmethod
    def tearDownClass(cls):
        cls.sf._dir = cls.old
        shutil.rmtree(cls.dir, ignore_errors=True)

    def make(self, **kw):
        return self.sf.make(**kw)

    def test_pressure_refusal_happens_before_any_work_and_leaves_state_untouched(self):
        backend = self.make()
        try:
            backend._current_machine = lambda: machine(available=GIB // 2, pressure=2)
            backend.append_text("hello world")
            with self.assertRaises(admission.AdmissionRefused):
                backend.generate(2)
            self.assertEqual(backend.cache_tokens, 0)
            self.assertTrue(backend.pending)        # the prompt is kept so the request can be retried
        finally:
            backend.close()

    def test_purge_is_off_unless_enabled_and_reports_why(self):
        backend = self.make()
        try:
            with mock.patch.dict("os.environ", {"FLASHNEXT_AUTO_PURGE": "0"}):
                self.assertEqual(backend._purge(GIB)[0], "skipped")
        finally:
            backend.close()

    def test_health_snapshot_and_last_prefix_have_the_fields_the_server_and_warmup_read(self):
        backend = self.make()
        try:
            backend.append_text("".join(chr(65 + i % 26) for i in range(24)))
            backend.generate(3)
            last = backend.last_prefix
            self.assertEqual((last.hit_tokens, last.prefilled_tokens, last.outcome), (0, 24, "cold"))
            self.assertEqual(last.prompt_tokens, 24)
            snap = backend.health_snapshot()
            for key in ("pool", "prefetch", "prefix_cache", "last_prefix", "readiness", "prefill"):
                self.assertIn(key, snap)
            self.assertEqual(snap["last_prefix"]["prompt_tokens"], 24)
            json.dumps(snap)                          # must be serialisable for /health
        finally:
            backend.close()

    def test_second_request_with_a_shared_prefix_reports_a_prefix_hit(self):
        backend = self.make(snapshot_tokens=8)
        backend.prefix_cache.min_tokens = 4
        try:
            text = "".join(chr(65 + (i * 3) % 26) for i in range(24))
            backend.append_text(text)
            backend.generate(2)
            backend.reset()
            backend.append_text(text + "ZZZZ")
            backend.generate(2)
            last = backend.last_prefix
            self.assertGreater(last.hit_tokens, 0)
            self.assertEqual(last.outcome, "ssd_hit")
            self.assertEqual(last.hit_tokens + last.prefilled_tokens, last.prompt_tokens)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
