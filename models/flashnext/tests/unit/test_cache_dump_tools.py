"""Cache dumps, their comparison, and the prefix-cache integrity checksums used by the real-model runs."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from models.flashnext import prefix_cache as P
from models.flashnext.tests.bench import bench_prefill_memory as B
from models.flashnext.tests.bench import compare_cache_dumps as C
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


class DumpTests(Base):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        mx.random.seed(2)
        cls.ids = [int(t) for t in mx.random.randint(2, VOCAB, (100,)).tolist()]

    def filled(self):
        caches = self.model.make_cache()
        out = self.model(mx.array([self.ids]), cache=caches)
        mx.eval(out.logits, [c.state for c in caches])
        return caches

    def test_a_dump_holds_every_cache_array_and_the_position_state(self):
        caches = self.filled()
        B.dump_cache(caches, self.model, self.tmp / "d.safetensors")
        loaded = mx.load(str(self.tmp / "d.safetensors"))
        self.assertIn("lang.position_ids", loaded)
        self.assertIn("lang.rope_deltas", loaded)
        kinds = [P.kind_of(c) for c in caches]
        for index, kind in enumerate(kinds):
            if kind == "qsa_kv":
                for name in ("keys", "values", "index_keys", "index_position_ids"):
                    self.assertIn(f"l{index}.{name}", loaded)
                self.assertEqual(loaded[f"l{index}.keys"].shape[2], 100)
            else:
                self.assertIn(f"l{index}.s0", loaded)
                self.assertIn(f"l{index}.s1", loaded)

    def test_two_dumps_of_the_same_state_compare_identical(self):
        caches = self.filled()
        B.dump_cache(caches, self.model, self.tmp / "a.safetensors")
        B.dump_cache(caches, self.model, self.tmp / "b.safetensors")
        result = C.compare(str(self.tmp / "a.safetensors"), str(self.tmp / "b.safetensors"))
        self.assertTrue(result["all_identical"])
        self.assertEqual(result["different"], [])

    def test_a_dump_that_differs_in_one_element_is_reported_with_its_size(self):
        caches = self.filled()
        B.dump_cache(caches, self.model, self.tmp / "a.safetensors")
        tensors = mx.load(str(self.tmp / "a.safetensors"))
        name = next(k for k in tensors if k.endswith(".s1"))
        changed = tensors[name]
        flat = changed.reshape(-1)
        flat = mx.concatenate([flat[:1] + 0.5, flat[1:]])
        tensors[name] = flat.reshape(changed.shape)
        mx.save_safetensors(str(self.tmp / "b.safetensors"), tensors)
        result = C.compare(str(self.tmp / "a.safetensors"), str(self.tmp / "b.safetensors"))
        self.assertFalse(result["all_identical"])
        self.assertEqual(len(result["different"]), 1)
        row = result["different"][0]
        self.assertEqual((row["tensor"], row["differing_elements"]), (name, 1))
        self.assertAlmostEqual(row["max_abs"], 0.5, places=5)

    def test_missing_tensors_and_shape_changes_are_not_identical(self):
        mx.save_safetensors(str(self.tmp / "a.safetensors"), {"x": mx.zeros((2, 3)), "y": mx.zeros((1,))})
        mx.save_safetensors(str(self.tmp / "b.safetensors"), {"x": mx.zeros((3, 2))})
        result = C.compare(str(self.tmp / "a.safetensors"), str(self.tmp / "b.safetensors"))
        self.assertFalse(result["all_identical"])
        self.assertEqual(result["only_in_one"], ["y"])
        self.assertIn("shape or dtype differs", result["different"][0]["reason"])

    def test_negative_zero_is_a_byte_difference(self):
        mx.save_safetensors(str(self.tmp / "a.safetensors"), {"x": mx.array([0.0], dtype=mx.float32)})
        mx.save_safetensors(str(self.tmp / "b.safetensors"), {"x": mx.array([-0.0], dtype=mx.float32)})
        self.assertFalse(C.compare(str(self.tmp / "a.safetensors"), str(self.tmp / "b.safetensors"))["all_identical"])

    def test_dtype_mismatch_is_not_identical(self):
        mx.save_safetensors(str(self.tmp / "a.safetensors"), {"x": mx.zeros((2,), dtype=mx.float32)})
        mx.save_safetensors(str(self.tmp / "b.safetensors"), {"x": mx.zeros((2,), dtype=mx.bfloat16)})
        self.assertFalse(C.compare(str(self.tmp / "a.safetensors"), str(self.tmp / "b.safetensors"))["all_identical"])


class ChecksumTests(Base):
    def test_checksums_detect_a_changed_snapshot_and_ignore_an_unchanged_one(self):
        from types import SimpleNamespace

        model = build_model()
        mx.random.seed(4)
        ids = [int(t) for t in mx.random.randint(2, VOCAB, (100,)).tolist()]
        caches = model.make_cache()
        mx.eval(model(mx.array([ids]), cache=caches).logits, [c.state for c in caches])
        store = P.PrefixCache(P.fingerprint(model="t"), 1 << 40)
        store.put(ids, caches, language=model)
        session = SimpleNamespace(store=lambda: store)
        first = B.store_checksums(session)
        self.assertEqual(first, B.store_checksums(session))
        entry = next(iter(store._ram.values()))
        entry.layers[0].states[0][...] = 1.0                    # corrupt it in place
        self.assertNotEqual(B.store_checksums(session), first)


if __name__ == "__main__":
    unittest.main()
