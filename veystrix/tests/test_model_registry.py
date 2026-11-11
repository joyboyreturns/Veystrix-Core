import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix import model_registry as registry


class ModelRegistryTests(unittest.TestCase):
    def test_distinct_stable_ids_for_two_checkpoints_and_exact_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one", "two"):
                path = root / name
                path.mkdir()
                (path / "config.json").write_text(json.dumps({
                    "model_type": "qwen4_exp", "max_position_embeddings": 262144,
                    "quantization": {"bits": 4}}))
            choices = [("flashnext", root / "one"), ("flashnext", root / "two")]
            with patch.object(registry, "installed_checkpoints", return_value=choices):
                first = registry.discover(root)
                second = registry.discover(root)
                self.assertEqual(first, second)
                self.assertEqual(len({item.id for item in first}), 2)
                self.assertEqual(registry.resolve(first[0].id, root), first[0])
                self.assertIsNone(registry.resolve("veystrix-flashnext", root))
                self.assertEqual(first[0].native_context, 262144)
                self.assertEqual(first[0].verification, "metadata_and_shard_presence_only")

    def test_alias_paths_do_not_duplicate_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "one"
            path.mkdir()
            (root / "alias").symlink_to(path, target_is_directory=True)
            with patch.object(registry, "installed_checkpoints", return_value=[
                ("flashnext", path), ("flashnext", root / "alias")]):
                self.assertEqual(len(registry.discover(root)), 1)
