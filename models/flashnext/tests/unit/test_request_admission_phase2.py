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

    def test_safe_and_short_requests(self):
        A.check(8192, read=lambda: self.reading())
        with self.assertRaisesRegex(A.RequestMemoryRefused, "headroom"):
            A.check(20, read=lambda: self.reading(free=1))

    def test_inactive_anonymous_is_not_free_headroom(self):
        with self.assertRaisesRegex(A.RequestMemoryRefused, "headroom"):
            A.check(8192, read=lambda: self.reading(free=3, inactive=8, anonymous=8))

    def test_swap_and_pressure_even_on_short_request(self):
        with self.assertRaisesRegex(A.RequestMemoryRefused, "swap"):
            A.check(20, read=lambda: self.reading(swap=2.4))
        with self.assertRaisesRegex(A.RequestMemoryRefused, "pressure"):
            A.check(20, read=lambda: self.reading(pressure=2))

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

    def test_16k_requires_more_headroom(self):
        with mock.patch.object(P, "request_required_bytes", return_value=5 * A.GIB):
            A.check(8192, read=lambda: self.reading(free=8))
            with self.assertRaisesRegex(A.RequestMemoryRefused, "headroom"):
                A.check(16384, read=lambda: self.reading(free=8))

    def test_output_and_snapshot_staging_have_distinct_reserves(self):
        with mock.patch.object(P, "request_required_bytes", return_value=5 * A.GIB):
            A.check(8192, max_output_tokens=1, read=lambda: self.reading(free=8))
            with self.assertRaisesRegex(A.RequestMemoryRefused, "headroom"):
                A.check(8192, max_output_tokens=50000, read=lambda: self.reading(free=8))
            with self.assertRaisesRegex(A.RequestMemoryRefused, "headroom"):
                A.check(8192, snapshot_count=13, read=lambda: self.reading(free=8))

    def test_candidate_defaults_and_explicit_rollback(self):
        p = MC.Policy.from_environment({"FLASHNEXT_PREFILL_ORDER": "layer"})
        self.assertEqual((p.swap_ceiling_mib, p.swap_ceiling_margin_mib), (3072, 768))
        self.assertEqual(MC.Policy.from_environment({}).swap_ceiling_mib, 0)
        self.assertEqual(MC.Policy.from_environment({"FLASHNEXT_PREFILL_ORDER": "layer", MC.FLAG_SWAP_CEILING: "0"}).swap_ceiling_mib, 0)
        A.check(8192, environ={"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_REQUEST_ADMISSION": "0"}, read=lambda: None)

    def test_service_maps_refusal_to_retriable_error_and_can_retry(self):
        from macqwen.server import ModelService, RequestError
        from macqwen.tests.test_server import FakeSession

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
        from macqwen.backends.flashnext import FlashNextBackend

        backend = object.__new__(FlashNextBackend)
        backend._generate = mock.Mock()
        with self.assertRaisesRegex(A.RequestMemoryRefused, "cannot estimate request memory"):
            backend.generate(5)
        backend._generate.assert_not_called()
