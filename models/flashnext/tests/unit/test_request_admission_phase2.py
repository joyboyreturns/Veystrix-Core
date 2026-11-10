import os
import time
import unittest
from unittest import mock

from models.flashnext import memory_controller as MC, purge_recovery as P, request_admission as A


class RequestAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"FLASHNEXT_PREFILL_ORDER": "layer"}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.watch = mock.patch("models.flashnext.progress.memory_watch_ready", return_value=True)
        self.watch.start()
        self.addCleanup(self.watch.stop)
        controller = MC.Controller(MC.Policy.from_environment())
        controller.last["mono_t"] = time.monotonic()
        MC.install(controller)
        self.addCleanup(MC.install, None)

    def reading(self, *, free=12, inactive=0, anonymous=0, swap=1, pressure=1):
        page = 2**20
        return P.Reading(page=page, free=free * 1024, inactive=inactive * 1024,
                         anonymous=anonymous * 1024, swap_used=swap * 1024 * page, pressure=pressure)

    def test_only_swap_at_the_ceiling_refuses(self):
        # Owner decision 2026-11-10: swap in use is the only memory limit.
        A.check(8192, read=lambda: self.reading())
        A.check(20, read=lambda: self.reading(swap=2.9))
        with self.assertRaisesRegex(A.RequestMemoryRefused, "swap"):
            A.check(20, read=lambda: self.reading(swap=3.0))
        with self.assertRaisesRegex(A.RequestMemoryRefused, "swap"):
            A.check(20, read=lambda: self.reading(swap=3.5))

    def test_low_memory_pressure_and_request_size_do_not_refuse(self):
        low = lambda: self.reading(free=1, inactive=8, anonymous=8, pressure=2, swap=1)
        A.check(20, read=low)
        A.check(16384, max_output_tokens=50000, snapshot_count=13, read=low)

    def test_missing_probe_and_stale_guard_fail_closed(self):
        with self.assertRaisesRegex(A.RequestMemoryRefused, "unavailable"):
            A.check(1, read=lambda: (_ for _ in ()).throw(ValueError("bad probe")))
        MC.current().last["mono_t"] = time.monotonic() - 4
        with self.assertRaisesRegex(A.RequestMemoryRefused, "stale"):
            A.check(1, read=lambda: self.reading())

    def test_missing_watchdog_refuses(self):
        with mock.patch("models.flashnext.progress.memory_watch_ready", return_value=False):
            with self.assertRaisesRegex(A.RequestMemoryRefused, "watchdog"):
                A.check(1, read=lambda: self.reading())

    def test_candidate_defaults_and_explicit_rollback(self):
        p = MC.Policy.from_environment({"FLASHNEXT_PREFILL_ORDER": "layer"})
        self.assertEqual((p.swap_ceiling_mib, p.swap_ceiling_margin_mib), (3072, 0))
        self.assertEqual(MC.Policy.from_environment({}).swap_ceiling_mib, 0)
        self.assertEqual(MC.Policy.from_environment({"FLASHNEXT_PREFILL_ORDER": "layer", MC.FLAG_SWAP_CEILING: "0"}).swap_ceiling_mib, 0)
        A.check(8192, environ={"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_REQUEST_ADMISSION": "0"}, read=lambda: None)

    def test_service_maps_refusal_to_retriable_error_and_can_retry(self):
        from veystrix.server import ModelService, RequestError
        from veystrix.tests.test_server import FakeSession

        session = FakeSession(["answer"])
        service = ModelService(session)
        original = session.backend.generate
        def refuse(*args, **kwargs):
            raise A.RequestMemoryRefused("insufficient headroom; retry")
        session.backend.generate = refuse
        with self.assertRaises(RequestError) as caught:
            service.complete([{"role": "user", "content": "hello"}], [], 10)
        self.assertEqual((caught.exception.status, caught.exception.code), (503, "memory_admission_refused"))
        session.backend.generate = original
        self.assertEqual(service.complete([{"role": "user", "content": "hello"}], [], 10).text, "answer")

    def test_candidate_backend_missing_request_state_refuses_before_generation(self):
        from veystrix.backends.flashnext import FlashNextBackend

        backend = object.__new__(FlashNextBackend)
        backend._generate = mock.Mock()
        with self.assertRaisesRegex(A.RequestMemoryRefused, "cannot estimate request memory"):
            backend.generate(5)
        backend._generate.assert_not_called()
