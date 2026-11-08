"""The five context numbers stay separate; the DeepSeek Harness reserve is accounted for, not assumed."""
import unittest

from macqwen import context_budget as C


class ClientWindowTests(unittest.TestCase):
    def test_a_client_window_equal_to_the_server_window_asks_for_one_token_near_the_limit(self):
        # the recorded failure: window 4096, ~900-token request, harness reserve 4096
        self.assertEqual(C.client_max_completion(4096, 902, 1024), 1)
        self.assertEqual(C.client_max_completion(8192, 4300, 1024), 1)

    def test_the_suggested_window_leaves_room_for_the_requested_output_whenever_the_prompt_fits_the_server(self):
        for server in (8192, 12288, 16384):
            advertised = C.suggested_client_window(server)
            for real_input in range(500, server - C.MIN_OUTPUT_RESERVE - 1024, 997):
                for error in (0.8, 1.0, 1.25):                      # the client only ESTIMATES input tokens
                    estimate = int(real_input * error)
                    asked = C.client_max_completion(advertised, estimate, 1024)
                    self.assertGreaterEqual(asked, min(1024, max(1, server - estimate)), (server, real_input, error))
                    self.assertGreaterEqual(asked, C.MIN_TOOL_OUTPUT if estimate < server - 1024 else 1)

    def test_server_clamp_never_exceeds_the_room_left(self):
        for window, prompt, asked in ((16384, 12000, 4096), (16384, 16000, 1024), (8192, 100, 1024)):
            b = C.evaluate(window, prompt, asked)
            self.assertLessEqual(prompt + b.reserved_output, max(window, prompt + 1))
            self.assertEqual(b.remaining, window - prompt - b.reserved_output)

    def test_continuation_headroom_flag(self):
        self.assertTrue(C.evaluate(16384, 12288, 1024).continuation_ok)       # 3072 left for tool results
        self.assertFalse(C.evaluate(16384, 14500, 1024).continuation_ok)      # 860 left: the next tool result may not fit
        self.assertTrue(C.evaluate(None, 99999, 1024).continuation_ok)        # no enforced limit: nothing claimed

    def test_tiny_output_cap_with_tools_is_rejected_without_tools_it_is_fine(self):
        self.assertIsNotNone(C.too_small_for_tools(1, True))
        self.assertIsNone(C.too_small_for_tools(1, False))
        self.assertIsNone(C.too_small_for_tools(C.MIN_TOOL_OUTPUT, True))

    def test_overflow_guidance_names_the_numbers_and_the_client_side_remedy(self):
        text = C.overflow_guidance(17000, 16384)
        for needle in ("17000", "16384", "never truncates", "Compact on the client"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
