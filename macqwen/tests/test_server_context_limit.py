import unittest

from macqwen.server import MIN_OUTPUT_RESERVE, ContextLimitError, ModelService, capabilities
from macqwen.tests.test_server import FakeSession


class ContextLimitTests(unittest.TestCase):
    def test_a_prompt_inside_the_limit_runs_and_output_is_clamped_to_the_remaining_room(self):
        session = FakeSession(["hi"])
        seen = []
        original = session.backend.generate
        session.backend.generate = lambda max_tokens, out=None: (seen.append(max_tokens), original(max_tokens, out))[1]
        service = ModelService(session, context_limit=len("rendered prompt") + MIN_OUTPUT_RESERVE + 10)
        service.complete([{"role": "user", "content": "x"}], [], 4096)
        self.assertEqual(seen, [MIN_OUTPUT_RESERVE + 10])

    def test_an_oversized_prompt_is_refused_without_truncation_and_keeps_live_state(self):
        session = FakeSession(["hello"])
        limit = 1000
        service = ModelService(session, context_limit=limit)
        service.complete([{"role": "user", "content": "one"}], [], 10)
        tape, resets = list(session.backend.tape), session.backend.reset_count
        session.backend.tokenizer.rendered = "x" * (limit - MIN_OUTPUT_RESERVE + 1)
        with self.assertRaises(ContextLimitError) as caught:
            service.complete([{"role": "user", "content": "two"}], [], 10)
        self.assertIn(str(limit), str(caught.exception))
        self.assertEqual(caught.exception.code, "context_length_exceeded")
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual((session.backend.tape, session.backend.reset_count), (tape, resets))
        self.assertEqual(session.backend.pending, [])

    def test_a_prompt_that_leaves_less_than_the_output_reserve_is_refused_not_answered_with_a_stub(self):
        limit = 1000
        for tokens, allowed in ((limit - MIN_OUTPUT_RESERVE, True), (limit - MIN_OUTPUT_RESERVE + 1, False)):
            with self.subTest(tokens=tokens):
                session = FakeSession(["ok"])
                session.backend.tokenizer.rendered = "x" * tokens
                service = ModelService(session, context_limit=limit)
                if allowed:
                    service.complete([{"role": "user", "content": "x"}], [], 4096)
                else:
                    with self.assertRaises(ContextLimitError):
                        service.complete([{"role": "user", "content": "x"}], [], 4096)

    def test_the_advertised_window_is_the_configured_limit(self):
        self.assertEqual(capabilities(ModelService(FakeSession(["a"]), context_limit=4096))["context_window"], 4096)

    def test_no_limit_keeps_the_previous_behaviour(self):
        session = FakeSession(["hello"])
        service = ModelService(session)
        service.complete([{"role": "user", "content": "x"}], [], 10)
        self.assertIsNone(service.context_limit)


if __name__ == "__main__":
    unittest.main()
