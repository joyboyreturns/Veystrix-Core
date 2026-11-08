"""The decode pool is optional: it is allocated only when real headroom covers it (first real chat, 2026-11-08)."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from models.flashnext import expert_pool as E

GIB = 2 ** 30
POOL = 1_999_880_192                       # the pool the real run created (651 slots)


class Store(SimpleNamespace):
    pass


def reader(available, pressure=1):
    calls = []

    def read():
        calls.append(1)
        return int(available * GIB), pressure
    read.calls = calls
    return read


class Env:
    def __enter__(self):
        self.patch = mock.patch.dict(os.environ, {"FLASHNEXT_EXPERT_POOL_GB": "2", "FLASHNEXT_EXPERT_POOL_AUTO": "1",
                                                   "FLASHNEXT_CHAT_GUARD": "1"})
        self.patch.start()
        E.set_budget_gb(None)
        return self

    def __exit__(self, *exc):
        self.patch.stop()
        E._HEADROOM_READER[0] = None


class DecisionTests(unittest.TestCase):
    def test_the_guard_floor_is_the_guards_own_number(self):
        from models.flashnext.tests.bench import prefill_memory_harness as H
        self.assertEqual(E.POOL_FLOOR_BYTES, int(H.COMPRESSOR_STOP_AVAILABLE_GIB * 2 ** 30))

    def test_the_measured_conditions_are_refused(self):
        # t=21.7 s in the trace: 4.35 GiB available, compressor lump still present. A 2 GB pool left 2.94 GiB.
        allowed, reason = E.headroom_decision(int(4.35 * GIB), POOL)
        self.assertFalse(allowed)
        self.assertIn("4.35 GiB available", reason)
        self.assertIn("3 GiB guard floor", reason)

    def test_comfortable_headroom_is_allowed(self):
        self.assertTrue(E.headroom_decision(int(6.4 * GIB), POOL)[0])      # the idle reading before the prefill
        self.assertTrue(E.headroom_decision(int(E.POOL_FLOOR_BYTES + E.POOL_MARGIN_BYTES + POOL), POOL)[0])
        self.assertFalse(E.headroom_decision(int(E.POOL_FLOOR_BYTES + E.POOL_MARGIN_BYTES + POOL) - 1, POOL)[0])

    def test_pressure_refuses_even_with_room(self):
        self.assertFalse(E.headroom_decision(12 * GIB, POOL, pressure=2)[0])


class ForStoreTests(unittest.TestCase):
    def test_insufficient_headroom_returns_no_pool_and_never_constructs_one(self):
        with Env():
            E._HEADROOM_READER[0] = reader(4.35)
            store = Store()
            with mock.patch.object(E, "ExpertPool") as constructor:
                self.assertIsNone(E.for_store(store, 32))
            constructor.assert_not_called()
            self.assertFalse(store._flashnext_pool_decision["allowed"])

    def test_the_refusal_is_remembered_so_decode_does_not_poll_memory_per_layer(self):
        with Env():
            read = reader(4.35)
            E._HEADROOM_READER[0] = read
            store = Store()
            with mock.patch.object(E, "ExpertPool"):
                for _ in range(48 * 10):                       # ten tokens of layer calls
                    self.assertIsNone(E.for_store(store, 32))
            self.assertEqual(len(read.calls), 1)

    def test_the_next_request_looks_again_and_can_allocate(self):
        with Env():
            read = reader(4.35)
            E._HEADROOM_READER[0] = read
            store = Store()
            fake = SimpleNamespace(slots=651, _mm=b"x" * 10)
            with mock.patch.object(E, "ExpertPool", return_value=fake) as constructor:
                self.assertIsNone(E.for_store(store, 32))
                E.reset_decline(store)                          # a new request
                E._HEADROOM_READER[0] = reader(6.5)
                self.assertIs(E.for_store(store, 32), fake)
                self.assertIs(E.for_store(store, 32), fake)     # created once
            constructor.assert_called_once()

    def test_sufficient_headroom_allocates_the_two_gb_pool(self):
        with Env():
            E._HEADROOM_READER[0] = reader(7.0)
            fake = SimpleNamespace(slots=651, _mm=b"x")
            with mock.patch.object(E, "ExpertPool", return_value=fake) as constructor:
                self.assertIs(E.for_store(Store(), 32), fake)
            self.assertEqual(constructor.call_args.args[1], int(2 * 1e9))

    def test_a_fixed_pool_is_not_headroom_checked(self):
        with Env():
            os.environ["FLASHNEXT_EXPERT_POOL_AUTO"] = "0"
            E._HEADROOM_READER[0] = reader(1.0)
            fake = SimpleNamespace(slots=651, _mm=b"x")
            with mock.patch.object(E, "ExpertPool", return_value=fake):
                self.assertIs(E.for_store(Store(), 32), fake)

    def test_a_guarded_allocation_failure_still_raises_and_leaves_nothing_behind(self):
        with Env():
            E._HEADROOM_READER[0] = reader(7.0)
            store = Store()
            with mock.patch.object(E, "ExpertPool", side_effect=MemoryError("no")):
                with self.assertRaises(MemoryError):
                    E.for_store(store, 32)
            self.assertIsNone(getattr(store, "_flashnext_expert_pool", None))

    def test_cancellation_during_construction_propagates_and_a_retry_can_succeed(self):
        from macqwen.backends.base import GenerationCancelled
        with Env():
            E._HEADROOM_READER[0] = reader(7.0)
            store = Store()
            fake = SimpleNamespace(slots=651, _mm=b"x")
            with mock.patch.object(E, "ExpertPool", side_effect=[GenerationCancelled(), fake]):
                with self.assertRaises(GenerationCancelled):
                    E.for_store(store, 32)
                self.assertIsNone(getattr(store, "_flashnext_expert_pool", None))
                self.assertIsNone(getattr(store, "_flashnext_pool_declined", None))
                self.assertIs(E.for_store(store, 32), fake)

    def test_no_pool_flag_means_no_pool_and_no_memory_read(self):
        with Env():
            os.environ["FLASHNEXT_EXPERT_POOL_GB"] = "0"
            read = reader(1.0)
            E._HEADROOM_READER[0] = read
            self.assertIsNone(E.for_store(Store(), 32))
            self.assertEqual(read.calls, [])

    def test_allocator_cache_is_trimmed_before_measuring(self):
        with Env():
            order = []
            E._HEADROOM_READER[0] = lambda: (order.append("read"), (int(9 * GIB), 1))[1]
            with mock.patch.object(E, "_trim_allocator_cache", lambda: order.append("trim")), \
                    mock.patch.object(E, "ExpertPool", return_value=SimpleNamespace(slots=1, _mm=b"x")):
                E.for_store(Store(), 32)
            self.assertEqual(order, ["trim", "read"])


class DecodeRoutingTests(unittest.TestCase):
    def test_a_declined_pool_sends_the_call_to_the_normal_read_path(self):
        from models.flashnext.expert_cache import StreamingSwitchGLU
        fake = SimpleNamespace(gate_proj=SimpleNamespace(cache=SimpleNamespace(store=Store()), group_size=32))
        with mock.patch("models.flashnext.expert_pool.for_store", return_value=None):
            self.assertFalse(StreamingSwitchGLU._pool_usable(fake))
        with mock.patch("models.flashnext.expert_pool.for_store", return_value=object()):
            self.assertTrue(StreamingSwitchGLU._pool_usable(fake))

    def test_the_call_site_checks_the_pool_before_taking_the_pool_pass(self):
        import inspect
        from models.flashnext.expert_cache import StreamingSwitchGLU
        self.assertIn("self._expert_pool_on and self._pool_usable()", inspect.getsource(StreamingSwitchGLU.__call__))


class ChatTurnsWithoutAPoolTests(unittest.TestCase):
    def test_repeated_short_chat_turns_continue_correctly_when_the_pool_is_declined(self):
        import mlx.core as mx
        from macqwen.tests.test_flashnext_backend import FlashNextBackendTests
        from models.flashnext.tests.unit.test_prefix_cache import build_model

        with Env():
            E._HEADROOM_READER[0] = reader(4.35)
            backend = FlashNextBackendTests().backend()
            backend.language = build_model()
            backend.cache = backend.language.make_cache()
            backend.tokenizer.decode = lambda ids: "".join(chr(65 + int(i) % 26) for i in ids)
            backend.stops = {1}
            lengths = []
            for tokens in ([12, 13, 14], [15, 16], [17, 18], [19]):
                backend.pending = list(tokens)
                backend.generate(3, out=lambda piece: None)
                mx.synchronize()
                lengths.append(len(backend.tape))
            self.assertEqual(lengths, sorted(set(lengths)))          # the conversation only grows
            self.assertGreater(lengths[-1], 8)
            self.assertIsNone(getattr(E.for_store(Store(), 32), "x", None))


if __name__ == "__main__":
    unittest.main()


class TracePhaseTests(unittest.TestCase):
    def test_generate_labels_prefill_decode_and_idle_even_when_cancelled(self):
        from macqwen.tests.test_flashnext_backend import FlashNextBackendTests
        from models.flashnext import trace_phase

        seen = []
        phase = SimpleNamespace(set=seen.append)
        trace_phase.register(phase)
        self.addCleanup(trace_phase.register, None)
        backend = FlashNextBackendTests().backend()

        def fake_generate(max_tokens, out, on_prefilled, *rest):
            on_prefilled()
            raise RuntimeError("cancelled")

        backend._generate = fake_generate
        backend._after_request = lambda: seen.append("cleanup")
        with self.assertRaises(RuntimeError):
            backend.generate(1)
        self.assertEqual(seen, ["prefill", "decode", "cleanup", "idle"])

    def test_without_a_registered_tracer_it_is_a_no_op(self):
        from models.flashnext import trace_phase
        trace_phase.register(None)
        trace_phase.set_phase("decode")
