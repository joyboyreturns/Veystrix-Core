import threading
import unittest

from veystrix.model_coordinator import ModelCoordinator, ModelNotFound, ModelBusy, HandoffRequired, LifecycleFault
from veystrix.model_registry import InstalledModel


def record(name):
    return InstalledModel(name, "flashnext", name, "/models/" + name, "qwen4_exp", {}, None, "metadata_only")


class Driver:
    def __init__(self):
        self.loaded = None
        self.events = []
        self.fail_unload = False
        self.fail_load = False
    def load(self, model, cancelled):
        assert self.loaded is None
        self.events.append(("load", model.id))
        self.loaded = model.id
        if self.fail_load:
            raise RuntimeError("load failed")
    def unload(self):
        self.events.append(("unload", self.loaded))
        if self.fail_unload:
            raise RuntimeError("worker still alive")
        self.loaded = None
    def ready(self, model):
        return self.loaded == model.id


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.models = {x: record(x) for x in ("a", "b")}
        self.driver = Driver()
        self.c = ModelCoordinator(self.models.get, self.driver)

    def test_switch_waits_for_explicit_owner_and_unloads_before_load(self):
        with self.c.request("a", owner="coding"): pass
        with self.assertRaises(HandoffRequired):
            with self.c.request("b", owner="background"): pass
        self.assertEqual(self.driver.events, [("load", "a")])
        with self.c.request("b", owner="coding"): pass
        self.assertEqual(self.driver.events, [("load", "a"), ("unload", "a"), ("load", "b")])
        self.assertEqual(self.c.snapshot().model_id, "b")

    def test_unknown_or_ambiguous_ids_never_unload(self):
        with self.c.request("a"): pass
        with self.assertRaises(ModelNotFound):
            with self.c.request("missing", handoff=True): pass
        self.assertEqual(self.driver.loaded, "a")

    def test_failed_unload_poisons_coordinator(self):
        with self.c.request("a", owner="user"): pass
        self.driver.fail_unload = True
        with self.assertRaises(LifecycleFault):
            with self.c.request("b", owner="user"): pass
        self.assertEqual(self.c.snapshot().state, "fault")
        self.assertNotIn(("load", "b"), self.driver.events)
        with self.assertRaises(LifecycleFault):
            with self.c.request("a"): pass

    def test_partial_load_cleanup_is_required(self):
        self.driver.fail_load = True
        with self.assertRaises(LifecycleFault):
            with self.c.request("a"): pass
        self.assertIsNone(self.driver.loaded)
        self.assertEqual(self.c.snapshot().state, "unloaded")

    def test_cannot_switch_during_a_stream(self):
        entered, done = threading.Event(), threading.Event()
        def stream():
            with self.c.request("a", owner="user"):
                entered.set()
                done.wait(2)
        thread = threading.Thread(target=stream)
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaises(ModelBusy):
                with self.c.request("b", owner="user", timeout=0.05): pass
            self.assertEqual(self.driver.loaded, "a")
        finally:
            done.set(); thread.join(2)

    def test_cancel_before_load_and_explicit_handoff(self):
        with self.assertRaises(ModelBusy):
            with self.c.request("a", cancelled=lambda: True): pass
        self.assertEqual(self.driver.events, [])
        with self.c.request("a", owner="one"): pass
        with self.c.request("b", owner="two", handoff=True): pass
        self.assertEqual(self.c.snapshot().owner, "two")

    def test_unowned_session_cannot_be_evicted_without_handoff(self):
        with self.c.request("a"): pass
        with self.assertRaises(HandoffRequired):
            with self.c.request("b", owner="new"): pass
        self.assertEqual(self.driver.loaded, "a")

    def test_readiness_error_faults_before_any_switch(self):
        with self.c.request("a", owner="one"): pass
        def broken(_model):
            raise RuntimeError("probe unavailable")
        self.driver.ready = broken
        with self.assertRaises(LifecycleFault):
            with self.c.request("a", owner="one"): pass
        self.assertEqual(self.c.snapshot().state, "fault")

    def test_failed_load_with_failed_cleanup_blocks_future_loads(self):
        self.driver.fail_load = True
        self.driver.fail_unload = True
        with self.assertRaises(LifecycleFault):
            with self.c.request("a"): pass
        self.assertEqual(self.c.snapshot().state, "fault")
        with self.assertRaises(LifecycleFault):
            with self.c.request("b", handoff=True): pass
        self.assertNotIn(("load", "b"), self.driver.events)

    def test_identity_loss_refuses_routing(self):
        with self.c.request("a"): pass
        self.driver.loaded = "different"
        with self.assertRaises(LifecycleFault):
            with self.c.request("a"): pass
        self.assertEqual(self.c.snapshot().state, "fault")
