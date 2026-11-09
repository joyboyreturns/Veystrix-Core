"""Opt-in uncached expert reads: same bytes, no file-cache fill, and the end-of-request restore cannot undo the mode."""
import fcntl
import os
import tempfile
import threading
import unittest
from unittest import mock

from models.flashnext import store as store_module
from models.flashnext.settings import modes


def make_store(directory, always, prefill=False):
    store = store_module.SafeTensorStore.__new__(store_module.SafeTensorStore)
    store.dir = directory
    store._fds = {}
    store._fd_lock = threading.Lock()
    store._rdahead = True
    store._uncached_always = always
    store._uncached_prefill = prefill
    store._uncached = always
    return store


class UncachedReadsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with open(os.path.join(self.tmp.name, "shard"), "wb") as handle:
            handle.write(b"x" * 64)

    def _fd_calls(self, store, action):
        calls = []
        real = fcntl.fcntl
        with mock.patch.object(store_module.fcntl, "fcntl",
                               side_effect=lambda fd, cmd, arg=0: calls.append((cmd, arg)) or real(fd, cmd, arg)):
            action(store)
        return calls

    def test_default_opens_buffered_and_always_opens_with_nocache(self):
        buffered = make_store(self.tmp.name, False)
        calls = self._fd_calls(buffered, lambda s: s._fd("shard"))
        self.assertNotIn((fcntl.F_NOCACHE, 1), calls)
        always = make_store(self.tmp.name, True)
        calls = self._fd_calls(always, lambda s: s._fd("shard"))
        self.assertIn((fcntl.F_NOCACHE, 1), calls)
        for store in (buffered, always):
            for fd in store._fds.values():
                os.close(fd)

    def test_end_of_request_restore_cannot_turn_the_always_mode_off(self):
        always = make_store(self.tmp.name, True)
        always._fd("shard")
        calls = self._fd_calls(always, lambda s: s.set_uncached_reads(False))
        self.assertTrue(always._uncached)
        self.assertEqual(calls, [(fcntl.F_NOCACHE, 1)])
        conditional = make_store(self.tmp.name, False)
        conditional._fd("shard")
        conditional.set_uncached_reads(True)
        calls = self._fd_calls(conditional, lambda s: s.set_uncached_reads(False))
        self.assertFalse(conditional._uncached)
        self.assertEqual(calls, [(fcntl.F_NOCACHE, 0)])
        for store in (always, conditional):
            for fd in store._fds.values():
                os.close(fd)

    def test_bytes_read_are_identical_in_both_modes(self):
        results = []
        for always in (False, True):
            store = make_store(self.tmp.name, always)
            fd = store._fd("shard")
            results.append(os.pread(fd, 64, 0))
            os.close(fd)
        self.assertEqual(results[0], results[1])

    def test_prefill_scope_is_uncached_only_while_a_prefill_runs(self):
        store = make_store(self.tmp.name, False, prefill=True)
        fd = store._fd("shard")
        self.assertFalse(store._uncached)                          # decode and idle: buffered
        calls = self._fd_calls(store, lambda s: s.begin_prefill_reads())
        self.assertTrue(store._uncached)
        self.assertEqual(calls, [(fcntl.F_NOCACHE, 1)])
        calls = self._fd_calls(store, lambda s: s.end_prefill_reads())
        self.assertFalse(store._uncached)
        self.assertEqual(calls, [(fcntl.F_NOCACHE, 0)])
        os.close(fd)
        # Other modes are untouched by the scope hooks.
        for always, prefill in ((True, False), (False, False)):
            other = make_store(self.tmp.name, always, prefill)
            self._fd_calls(other, lambda s: (s.begin_prefill_reads(), s.end_prefill_reads()))
            self.assertEqual(other._uncached, always)

    def test_store_environment_values_are_parsed_and_validated(self):
        index = os.path.join(self.tmp.name, "model.safetensors.index.json")
        with open(index, "w") as handle:
            handle.write('{"weight_map": {}}')
        for value, always, prefill in (("0", False, False), ("1", True, False), ("prefill", False, True)):
            with mock.patch.dict(os.environ, {"FLASHNEXT_UNCACHED_EXPERT_READS": value}):
                store = store_module.SafeTensorStore(self.tmp.name)
            self.addCleanup(store.close)
            self.assertEqual((store._uncached_always, store._uncached_prefill, store._uncached), (always, prefill, always))
        with mock.patch.dict(os.environ, {"FLASHNEXT_UNCACHED_EXPERT_READS": "yes"}):
            with self.assertRaises(ValueError):
                store_module.SafeTensorStore(self.tmp.name)

    def test_prefill_target_scopes_the_reads_on_success_and_on_failure(self):
        from types import SimpleNamespace
        from models.flashnext import prefill, pool_lifecycle
        store = mock.Mock()
        with mock.patch.object(pool_lifecycle, "store_of", return_value=store):
            with mock.patch.object(prefill, "_prefill_target_checked", return_value="done") as inner:
                self.assertEqual(prefill.prefill_target(SimpleNamespace(), "ids", "cache", want_logits=True), "done")
            inner.assert_called_once_with(mock.ANY, "ids", "cache", want_logits=True)
            self.assertEqual([c[0] for c in store.method_calls], ["begin_prefill_reads", "end_prefill_reads"])
            store.reset_mock()
            with mock.patch.object(prefill, "_prefill_target_checked", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    prefill.prefill_target(SimpleNamespace(), "ids", "cache")
            self.assertEqual([c[0] for c in store.method_calls], ["begin_prefill_reads", "end_prefill_reads"])

    def test_prefill_mode_is_explicit_and_cannot_combine_with_the_all_phase_mode(self):
        flags, label = modes.EXPERIMENTAL["uncached-prefill-reads"]
        self.assertEqual(flags, {"FLASHNEXT_UNCACHED_EXPERT_READS": "prefill"})
        plan = modes.plan_mode("agentic", enable=("uncached-prefill-reads",), available_gb=12, free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_UNCACHED_EXPERT_READS"], "prefill")
        with self.assertRaises(modes.ModeError):
            modes.plan_mode("agentic", enable=("uncached-reads", "uncached-prefill-reads"), available_gb=12, free_disk_gb=200)

    def test_mode_is_explicit_opt_in_and_not_a_default(self):
        self.assertEqual(modes.EXPERIMENTAL_OFF["FLASHNEXT_UNCACHED_EXPERT_READS"], "0")
        flags, label = modes.EXPERIMENTAL["uncached-reads"]
        self.assertEqual(flags, {"FLASHNEXT_UNCACHED_EXPERT_READS": "1"})
        self.assertIn("unmeasured", label)
        plan = modes.plan_mode("agentic", enable=("uncached-reads", "layer-major-4k"), available_gb=12, free_disk_gb=200)
        self.assertEqual(plan.env["FLASHNEXT_UNCACHED_EXPERT_READS"], "1")
        self.assertEqual(modes.plan_mode("agentic", available_gb=12, free_disk_gb=200).env["FLASHNEXT_UNCACHED_EXPERT_READS"], "0")


class PrefillScopeSafetyTests(unittest.TestCase):
    """begin/end are reference counted and always restore buffered reads; the wrapper keeps the target's signature."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with open(os.path.join(self.tmp.name, "shard"), "wb") as handle:
            handle.write(b"x" * 64)

    def test_nested_prefills_restore_only_when_the_outermost_ends(self):
        store = make_store(self.tmp.name, False, prefill=True)
        store.begin_prefill_reads()
        store.begin_prefill_reads()
        self.assertTrue(store._uncached)
        store.end_prefill_reads()
        self.assertTrue(store._uncached)          # an inner end must not turn the outer prefill's mode off
        store.end_prefill_reads()
        self.assertFalse(store._uncached)

    def test_unbalanced_end_never_goes_negative_or_leaves_the_mode_on(self):
        store = make_store(self.tmp.name, False, prefill=True)
        store.end_prefill_reads()
        self.assertEqual(store._prefill_depth, 0)
        store.begin_prefill_reads()
        store.end_prefill_reads()
        self.assertFalse(store._uncached)

    def test_exception_and_cancellation_inside_prefill_restore_buffered_reads(self):
        from models.flashnext import prefill, pool_lifecycle

        store = make_store(self.tmp.name, False, prefill=True)
        for error in (RuntimeError("boom"), KeyboardInterrupt()):
            with mock.patch.object(pool_lifecycle, "store_of", return_value=store), \
                 mock.patch.object(prefill, "_prefill_target_checked", side_effect=error):
                with self.assertRaises(type(error)):
                    prefill.prefill_target(object(), [1, 2, 3], [])
            self.assertFalse(store._uncached)
            self.assertEqual(store._prefill_depth, 0)

    def test_wrapper_keeps_the_public_signature(self):
        import inspect
        from models.flashnext import prefill

        names = list(inspect.signature(prefill.prefill_target).parameters)
        self.assertEqual(names[:3], ["language", "ids", "cache"])
        for expected in ("want_logits", "want_hidden", "sampler", "boundaries", "on_boundary", "offset", "anchors", "trace",
                         "required"):
            self.assertIn(expected, names)
        self.assertEqual(prefill.prefill_target.__name__, "prefill_target")
        self.assertTrue(prefill.prefill_target.__doc__)


class DrainReportsOrphanedFailuresTests(unittest.TestCase):
    def test_orphaned_prefetch_failure_is_counted_and_logged(self):
        from models.flashnext import expert_cache, layer_major

        window = layer_major.ExpertWindow.__new__(layer_major.ExpertWindow)
        window.stats = {"prefetch_layers": 1}
        window._prefetched = {3: (("p",), ["future-a"])}
        with mock.patch.object(expert_cache, "_resolve_future", side_effect=OSError("short read")), \
             self.assertLogs("flashnext.layer_major", level="WARNING") as logs:
            window.drain()
        self.assertEqual(window.stats["prefetch_drain_errors"], 1)
        self.assertIn("short read", logs.output[0])
        self.assertEqual(window._prefetched, {})


if __name__ == "__main__":
    unittest.main()
