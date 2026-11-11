"""Installed checkpoint identity cannot route to a different loaded backend."""
import unittest
from unittest.mock import patch

from veystrix.model_registry import InstalledModel, installation_id
from veystrix.tests.test_server_runtime import Running, request, CHAT


class InstalledModelsAPITests(unittest.TestCase):
    def test_inventory_marks_only_exact_loaded_checkpoint(self):
        run = Running(self, api_key="secret")
        installed = InstalledModel(installation_id("flashnext", "/models/a"), "flashnext", "A", "/models/a", "qwen4_exp", {}, None, "metadata_only")
        run.service.installed_model_id = installed.id
        with patch("veystrix.model_registry.discover", return_value=[installed]):
            self.assertEqual(request(run.port, "GET", "/v1/installed-models")[0], 401)
            status, data, _ = request(run.port, "GET", "/v1/installed-models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertEqual(data["data"][0]["id"], installed.id)
        self.assertTrue(data["data"][0]["loaded"])

    def test_identity_is_reflected_on_all_compatible_wire_formats(self):
        run = Running(self)
        run.service.installed_model_id = installation_id("flashnext", "/models/a")
        for route, payload in (
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}),
            ("/v1/responses", {"input": "hi"}),
        ):
            with self.subTest(route=route):
                status, result, _ = request(run.port, "POST", route, {**payload, "model": run.service.installed_model_id})
                self.assertEqual(status, 200, result)
                self.assertEqual(result["model"], run.service.installed_model_id)

    def test_exact_active_installation_id_is_accepted_but_another_is_not(self):
        run = Running(self)
        run.service.installed_model_id = installation_id("flashnext", "/models/a")
        other = installation_id("flashnext", "/models/b")
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "model": other})
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))
        status, body, _ = request(run.port, "POST", "/v1/chat/completions", {**CHAT, "model": run.service.installed_model_id})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], run.service.installed_model_id)

if __name__ == "__main__":
    unittest.main()
