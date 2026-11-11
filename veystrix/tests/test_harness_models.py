import unittest
from veystrix.harness_models import qwen_config, deepseek_config
from veystrix.model_registry import InstalledModel


def m(x): return InstalledModel(x, "flashnext", x, "/"+x, "qwen4_exp", {}, None, "test")


class HarnessConfigTests(unittest.TestCase):
    def test_catalog_uses_native_window_with_a_safe_fallback(self):
        from dataclasses import replace
        full = replace(m('full'), native_context=262144)
        small = replace(m('small'), native_context=8192)
        entries = qwen_config({}, [full, small, m('unknown')])['modelProviders']['openai']
        self.assertEqual([x['generationConfig']['contextWindowSize'] for x in entries], [262144, 8192, 32768])
        patch = [{'id': 'llm-pi-ai'}]
        models = deepseek_config(patch, [full, small, m('unknown')])[0]['config']['providers']['veystrix']['models']
        self.assertEqual([x['contextWindow'] for x in models], [266240, 12288, 36864])
        self.assertEqual([x['maxTokens'] for x in models], [262144, 8192, 32768])
        self.assertEqual(models[0]['reasoningEfforts']['xhigh'], 'xhigh')
        self.assertEqual(models[0]['reasoningEfforts']['max'], 'max')

    def test_qwen_merge_preserves_unrelated_providers_and_is_idempotent(self):
        original = {"modelProviders": {"openai": [{"id": "other", "baseUrl": "https://example.test"}], "other": [{"id": "x"}]}, "model": {"name": "other"}}
        result = qwen_config(original, [m("a"), m("b")])
        self.assertEqual(result["model"], original["model"])
        self.assertEqual(result["modelProviders"]["other"], original["modelProviders"]["other"])
        self.assertEqual([x["id"] for x in result["modelProviders"]["openai"]], ["other", "a", "b"])
        self.assertEqual(qwen_config(result, [m("a"), m("b")]), result)
        self.assertEqual(len(original["modelProviders"]["openai"]), 1)
    def test_qwen_does_not_replace_foreign_entry(self):
        with self.assertRaises(ValueError): qwen_config({"modelProviders": {"openai": [{"id": "a", "baseUrl": "https://other"}]}}, [m("a")])
    def test_deepseek_preserves_other_plugins_and_provider(self):
        old = [{"id": "llm-pi-ai", "config": {"providers": {"other": {"api": "other"}, "veystrix": {"baseURL": "http://127.0.0.1:8080/v1", "extra": True}}}}, {"id": "ui", "config": {"x": 1}}]
        new = deepseek_config(old, [m("a"), m("b")])
        self.assertEqual(new[1], old[1])
        self.assertEqual(new[0]["config"]["providers"]["other"], {"api": "other"})
        self.assertTrue(new[0]["config"]["providers"]["veystrix"]["extra"])
        self.assertEqual([x["id"] for x in new[0]["config"]["providers"]["veystrix"]["models"]], ["a", "b"])
        self.assertEqual(deepseek_config(new, [m("a"), m("b")]), new)
    def test_deepseek_rejects_ambiguous_plugins_and_foreign_provider(self):
        with self.assertRaises(ValueError): deepseek_config([], [m("a")])
        with self.assertRaises(ValueError): deepseek_config([{"id": "llm-pi-ai", "config": {"providers": {"veystrix": {"baseURL": "https://elsewhere"}}}}], [m("a")])

class HarnessWriteTests(unittest.TestCase):
    def test_configure_creates_backup_and_preserves_unrelated_settings(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from veystrix.harness_configure import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / '.qwen' / 'settings.json'; path.parent.mkdir()
            original = '{"model":{"name":"other"},"other":{"keep":true}}\n'; path.write_text(original)
            with patch('veystrix.harness_configure.discover', return_value=[m('a'), m('b')]):
                result = run(home=root, apply=True)
                self.assertIn('updated', result['Qwen Code'])
                backups = list(path.parent.glob('settings.json.veystrix-backup-*'))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), original)
                self.assertEqual(json.loads(path.read_text())['other'], {'keep': True})
                self.assertEqual(run(home=root, apply=True)['Qwen Code'], 'unchanged')

    def test_hermes_refresh_uses_dedicated_profile_and_preserves_main_config(self):
        import tempfile, yaml
        from pathlib import Path
        from unittest.mock import patch
        from veystrix.harness_configure import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root / '.hermes'; home.mkdir()
            main = home / 'config.yaml'; main.write_text('model: external\n')
            with patch('veystrix.harness_configure.discover', return_value=[m('a'), m('b')]):
                result = run(home=root, apply=True)
                self.assertIn('created', result['Hermes'])
                profile = yaml.safe_load((home/'profiles/veystrix/config.yaml').read_text())
                self.assertEqual(set(profile['custom_providers'][0]['models']), {'a','b'})
                self.assertEqual(main.read_text(), 'model: external\n')
                self.assertIn('unchanged', run(home=root, apply=True)['Hermes'])

    def test_deepseek_yaml_off_key_stays_a_string(self):
        from veystrix.harness_configure import load_yaml12
        parsed = load_yaml12('reasoningEfforts:\n  off: none\n  enabled: false\n')
        self.assertEqual(parsed['reasoningEfforts']['off'], 'none')
        self.assertIs(parsed['reasoningEfforts']['enabled'], False)
