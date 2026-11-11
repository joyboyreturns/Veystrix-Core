import json
import unittest
from unittest.mock import patch

from models.flashnext.overnight import health


class Response:
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read(self): return json.dumps({"ready": True}).encode()


class SupervisorHealthAuthTests(unittest.TestCase):
    def test_health_probe_uses_configured_server_key(self):
        with patch.dict("os.environ", {"VEYSTRIX_SERVER_API_KEY": "test-key"}), \
             patch("urllib.request.urlopen", return_value=Response()) as fetch:
            self.assertEqual(health("127.0.0.1", 1234), {"ready": True})
            self.assertEqual(fetch.call_args.args[0].get_header("Authorization"), "Bearer test-key")

    def test_health_probe_works_without_key(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("urllib.request.urlopen", return_value=Response()) as fetch:
            self.assertEqual(health("127.0.0.1", 1234), {"ready": True})
            self.assertIsNone(fetch.call_args.args[0].get_header("Authorization"))
