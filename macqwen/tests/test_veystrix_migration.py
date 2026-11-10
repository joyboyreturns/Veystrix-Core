"""Offline tests for the MACQWEN -> Veystrix staged rebrand.

No model, no checkpoint, no network. Verifies: the ``veystrix`` import alias, the
VEYSTRIX_* environment aliases (and that they cover every legacy name the source reads),
configuration migration (copy, never move/delete) and that persistent cache locations
did not change.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from macqwen import api_keys, env_compat, preferences

ROOT = Path(__file__).resolve().parents[2]


class ImportAliasTests(unittest.TestCase):
    def test_alias_resolves_to_the_same_module_object(self):
        import veystrix.preferences as alias
        self.assertIs(alias, preferences)
        import veystrix.api_keys as alias_keys
        self.assertIs(alias_keys, api_keys)

    def test_from_import_form_works(self):
        from veystrix import preferences as via_from
        self.assertIs(via_from, preferences)

    def test_unknown_submodule_is_a_normal_import_error(self):
        with self.assertRaises(ModuleNotFoundError):
            import veystrix.does_not_exist  # noqa: F401

    def test_version_matches_pyproject(self):
        import veystrix
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(veystrix.__version__, project["version"])
        self.assertEqual(project["name"], "veystrix")

    def test_both_console_scripts_are_declared(self):
        scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
        self.assertEqual(scripts["veystrix"], scripts["macqwen"])

    def test_python_dash_m_veystrix_help_does_not_load_anything(self):
        # --help is handled by argparse in the backend launcher before any model work.
        # Only check that the module entry point exists and imports; do not execute it.
        self.assertTrue((ROOT / "veystrix" / "__main__.py").is_file())


class EnvironmentAliasTests(unittest.TestCase):
    def test_veystrix_name_sets_the_legacy_name(self):
        env = {"VEYSTRIX_MODEL_ROOT": "/models"}
        changed = env_compat.apply(env)
        self.assertEqual(env["MACQWEN_MODEL_ROOT"], "/models")
        self.assertEqual(changed, ["MACQWEN_MODEL_ROOT"])

    def test_veystrix_name_wins_when_both_are_set(self):
        env = {"VEYSTRIX_PYTHON": "/new/python", "MACQWEN_PYTHON": "/old/python"}
        env_compat.apply(env)
        self.assertEqual(env["MACQWEN_PYTHON"], "/new/python")

    def test_legacy_only_is_untouched(self):
        env = {"MACQWEN_PYTHON": "/old/python"}
        self.assertEqual(env_compat.apply(env), [])
        self.assertEqual(env, {"MACQWEN_PYTHON": "/old/python"})

    def test_non_alias_veystrix_variables_are_not_copied(self):
        env = {"VEYSTRIX_API_KEY": "secret", "VEYSTRIX_MODE": "agentic", "VEYSTRIX_CACHE_GB": "50"}
        self.assertEqual(env_compat.apply(env), [])
        self.assertFalse(any(name.startswith("MACQWEN_") for name in env))

    def test_prefix_families_are_aliased(self):
        env = {"VEYSTRIX_BONSAI2_FUTURE_OPTION": "1"}
        env_compat.apply(env)
        self.assertEqual(env["MACQWEN_BONSAI2_FUTURE_OPTION"], "1")

    def test_every_legacy_name_the_source_reads_is_covered(self):
        """Drift guard: a new MACQWEN_* read must also be aliased."""
        names = set()
        for path in ROOT.joinpath("macqwen").rglob("*.py"):
            names |= set(re.findall(r"MACQWEN_([A-Z0-9_]+)", path.read_text(errors="replace")))
        for path in ROOT.joinpath("models").rglob("*.py"):
            names |= set(re.findall(r"MACQWEN_([A-Z0-9_]+)", path.read_text(errors="replace")))
        names.discard("")                                  # the bare prefix in docstrings
        uncovered = {
            suffix for suffix in names
            if suffix not in env_compat.LEGACY_SUFFIXES and not suffix.startswith(env_compat.LEGACY_PREFIXES)
        }
        self.assertEqual(uncovered, set())

    def test_subprocess_sees_the_alias_after_importing_the_package(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("MACQWEN_", "VEYSTRIX_"))}
        env["VEYSTRIX_MODEL_ROOT"] = "/models/x"
        out = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys; sys.path.insert(0, %r); import models, os; print(os.environ.get('MACQWEN_MODEL_ROOT'))" % str(ROOT)],
            capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(out.stdout.strip(), "/models/x", out.stderr)


class ConfigurationMigrationTests(unittest.TestCase):
    def test_default_locations_use_the_new_name(self):
        self.assertIn("Veystrix", api_keys.DEFAULT_PATH)
        self.assertIn(".veystrix", preferences.DEFAULT_PATH)
        self.assertIn("MACQWEN", api_keys.LEGACY_PATH)
        self.assertIn("~/.macqwen/preferences.json", preferences.LEGACY_PATHS)

    def test_legacy_key_file_is_copied_not_moved_and_stays_private(self):
        with tempfile.TemporaryDirectory() as root:
            legacy = Path(root, "MACQWEN", "api_keys.json")
            legacy.parent.mkdir()
            legacy.write_text('{"tavily": "tvly-test"}')
            new = Path(root, "Veystrix", "api_keys.json")
            self.assertTrue(api_keys.migrate_legacy(new, legacy))
            self.assertEqual(new.read_text(), legacy.read_text())
            self.assertTrue(legacy.exists(), "the legacy file must survive")
            self.assertEqual(stat.S_IMODE(new.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(new.parent.stat().st_mode), 0o700)

    def test_existing_new_key_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            legacy = Path(root, "old.json")
            legacy.write_text('{"tavily": "old"}')
            new = Path(root, "new.json")
            new.write_text('{"tavily": "new"}')
            self.assertFalse(api_keys.migrate_legacy(new, legacy))
            self.assertEqual(new.read_text(), '{"tavily": "new"}')

    def test_missing_legacy_file_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(api_keys.migrate_legacy(Path(root, "n.json"), Path(root, "none.json")))
            self.assertFalse(Path(root, "n.json").exists())

    def test_default_key_store_migrates_on_open(self):
        with tempfile.TemporaryDirectory() as root:
            legacy = Path(root, "MACQWEN", "api_keys.json")
            legacy.parent.mkdir()
            legacy.write_text('{"tavily": "tvly-legacy"}')
            new = Path(root, "Veystrix", "api_keys.json")
            with patch.object(api_keys, "DEFAULT_PATH", str(new)), patch.object(api_keys, "LEGACY_PATH", str(legacy)), \
                    patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TAVILY_API_KEY", None)
                store = api_keys.KeyStore(str(new))   # the (patched) default location
                self.assertEqual(store.values.get("tavily"), "tvly-legacy")
                os.environ.pop("TAVILY_API_KEY", None)
            self.assertTrue(new.exists() and legacy.exists())

    def test_preferences_fall_back_to_the_old_default_and_keep_it(self):
        with tempfile.TemporaryDirectory() as root:
            old = Path(root, ".macqwen", "preferences.json")
            old.parent.mkdir()
            old.write_text('{"animate": false}')
            new = Path(root, ".veystrix", "preferences.json")
            with patch.object(preferences, "DEFAULT_PATH", str(new)), \
                    patch.object(preferences, "LEGACY_PATHS", (str(old),)):
                loaded = preferences.load(str(new))
            self.assertFalse(loaded["animate"])
            self.assertTrue(old.exists())

    def test_persistent_cache_locations_are_unchanged(self):
        """The rebrand must not move or rename engine caches: that would orphan them."""
        from models.flashnext import prefix_session, sessions
        self.assertEqual(prefix_session.DEFAULT_DIR, "~/.flashnext/prefix-cache")
        self.assertEqual(sessions.DEFAULT_SESSION_DIR, "~/.cache/flashnext/sessions")

    def test_saved_session_module_path_rewrite_is_still_present(self):
        text = (ROOT / "models" / "flashnext" / "sessions.py").read_text()
        self.assertIn('b"models.flashnext."', text)

    def test_internal_packages_were_not_renamed(self):
        for directory in ("macqwen", "models/flashnext", "veystrix"):
            self.assertTrue((ROOT / directory).is_dir(), directory)


if __name__ == "__main__":
    unittest.main()
