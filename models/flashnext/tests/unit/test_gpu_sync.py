"""GPU/GIL deadlock regression (first full-checkpoint AGENTIC run, 2026-11-08). No checkpoint is loaded.

Every GPU scenario runs in a child process with its own timeout and an in-child faulthandler, so a deadlock fails one
test with the Python stacks instead of freezing the suite.
"""
import ast
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CHILD_TIMEOUT = 40

PRELUDE = '''
import faulthandler, sys, time
sys.path.insert(0, %(root)r)
import numpy as np
import mlx.core as mx
faulthandler.dump_traceback_later(%(inner)d, exit=True)

def heavy(x, rounds=40):
    w = mx.random.normal((2048, 2048))
    y = x
    for _ in range(rounds):
        y = mx.tanh(y @ w)
    return y.sum()

def python_owned_operand():
    """A graph whose only reference to a Python-owned buffer is the in-flight job (the expert-read pattern)."""
    host = np.random.default_rng(0).standard_normal((2048, 2048), dtype=np.float32)
    x = mx.from_dlpack(host, copy=False)
    y = heavy(x)
    del x, host
    return y
'''


def metal_available() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import mlx.core as mx; raise SystemExit(0 if mx.metal.is_available() else 3)"],
                           capture_output=True, timeout=60, cwd=ROOT)
    return probe.returncode == 0


def run_child(body: str, timeout: int = CHILD_TIMEOUT):
    code = PRELUDE % {"root": str(ROOT), "inner": timeout - 6} + textwrap.dedent(body)
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return done.returncode, done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as expired:
        return None, (expired.stdout or b"").decode(errors="replace") if isinstance(expired.stdout, bytes) else (expired.stdout or ""), \
            "child exceeded its timeout", True


@unittest.skipUnless(metal_available(), "needs the Metal build of MLX")
class DeadlockReproductionTests(unittest.TestCase):
    def test_raw_synchronize_with_a_python_owned_operand_deadlocks_and_names_the_call(self):
        # The known bad pattern. If a future MLX stops deadlocking this, the canary says so and the guard test below
        # still protects the production code.
        code, out, err, hung = run_child("""
            y = python_owned_operand()
            mx.async_eval(y)
            mx.synchronize()
            print("COMPLETED")
        """, timeout=22)
        if code == 0 and "COMPLETED" in out:
            self.skipTest("this MLX no longer deadlocks mx.synchronize with a DLPack operand; the canary is obsolete")
        self.assertIn("Timeout", err + out)            # the in-child faulthandler fired, with Python stacks
        self.assertIn("<no Python frame>", err + out)  # Metal's completion thread: native only, waiting for the GIL
        self.assertIn("in <module>", err + out)        # the main thread is parked in the synchronize call

    def test_plain_arrays_synchronize_without_trouble(self):
        code, out, err, hung = run_child("""
            y = heavy(mx.random.normal((2048, 2048)))
            mx.async_eval(y)
            mx.synchronize()
            print("COMPLETED")
        """)
        self.assertEqual((code, hung), (0, False), err)
        self.assertIn("COMPLETED", out)

    def test_gpu_sync_completes_with_a_python_owned_operand(self):
        code, out, err, hung = run_child("""
            from models.flashnext import gpu_sync
            y = python_owned_operand()
            mx.async_eval(y)
            gpu_sync.synchronize()
            print("COMPLETED")
        """)
        self.assertEqual((code, hung), (0, False), err)
        self.assertIn("COMPLETED", out)

    def test_the_barrier_really_waits_for_earlier_work(self):
        code, out, err, hung = run_child("""
            from models.flashnext import gpu_sync
            y = python_owned_operand()
            mx.async_eval(y)
            waited = gpu_sync.barrier()
            began = time.perf_counter()
            y.item()                                   # must already be finished: nothing is left to wait for
            after = time.perf_counter() - began
            print("WAITED", waited, "AFTER", after)
        """)
        self.assertEqual((code, hung), (0, False), err)
        waited = float(out.split("WAITED")[1].split()[0])
        after = float(out.split("AFTER")[1].split()[0])
        self.assertGreater(waited, 0.05, "the barrier returned before the queued work could have run")
        self.assertLess(after, 0.05, "work was still running after the barrier")

    def test_the_barrier_covers_a_second_stream(self):
        code, out, err, hung = run_child("""
            from models.flashnext import gpu_sync
            side = mx.new_stream(mx.gpu)
            with mx.stream(side):
                w = mx.random.normal((2048, 2048))
                z = mx.zeros((2048, 2048)) + 1
                for _ in range(40):
                    z = mx.tanh(z @ w)
                z = z.sum()
            mx.async_eval(z)
            gpu_sync.barrier(extra_streams=[side])
            began = time.perf_counter()
            z.item()
            print("AFTER", time.perf_counter() - began)
        """)
        self.assertEqual((code, hung), (0, False), err)
        self.assertLess(float(out.split("AFTER")[1].split()[0]), 0.05)

    EXIT_PRELUDE = """
        import atexit
        from models.flashnext import gpu_sync
        %(install)s
        host = np.random.default_rng(0).standard_normal((2048, 2048), dtype=np.float32)
        x = mx.from_dlpack(host, copy=False)
        w = mx.random.normal((2048, 2048)); y = x
        for _ in range(40): y = mx.tanh(y @ w)
        mx.async_eval(y.sum()); del x, y, host
    """

    def test_exit_hook_stream_cleanup_hangs_with_work_in_flight_unless_drained(self):
        # mlx_vlm registers mx.clear_streams() at exit; it waits with the GIL held, like mx.synchronize.
        raw = run_child(textwrap.dedent(self.EXIT_PRELUDE % {"install": ""}) + "\nmx.clear_streams()\nprint('COMPLETED')\n", timeout=22)
        if raw[0] == 0 and "COMPLETED" in raw[1]:
            self.skipTest("this MLX no longer hangs clear_streams with work in flight")
        self.assertIn("Timeout", raw[2] + raw[1])
        drained = run_child(textwrap.dedent(self.EXIT_PRELUDE % {"install": ""}) + "\ngpu_sync.drain()\nmx.clear_streams()\nprint('COMPLETED')\n")
        self.assertEqual((drained[0], drained[3]), (0, False), drained[2])
        self.assertIn("COMPLETED", drained[1])

    def test_the_installed_exit_drain_lets_a_real_interpreter_exit_with_work_in_flight(self):
        for attempt in range(3):
            done = run_child(textwrap.dedent(self.EXIT_PRELUDE % {"install": "gpu_sync.install_exit_drain()"}) + "\nprint('BODY DONE', flush=True)\n")
            self.assertEqual((done[0], done[3]), (0, False), f"attempt {attempt}: " + done[2][-3000:])
            self.assertIn("BODY DONE", done[1])

    def test_the_backend_installs_the_exit_drain(self):
        self.assertIn("gpu_sync.install_exit_drain()", (ROOT / "veystrix/backends/flashnext.py").read_text())
        from models.flashnext import gpu_sync
        self.assertTrue(callable(gpu_sync.install_exit_drain))

    def test_stress_many_requests_with_threads_exceptions_and_cleanup_barriers(self):
        # The production shape: worker threads fill host buffers, the main thread wraps them zero-copy, queues pipelined
        # steps, sometimes abandons them (cancellation or error), and a cleanup barrier follows every request.
        code, out, err, hung = run_child("""
            from concurrent.futures import ThreadPoolExecutor
            from models.flashnext import gpu_sync
            gpu_sync.install_exit_drain()

            def fill(seed):
                host = np.empty((512, 512), dtype=np.float32)
                host[:] = np.random.default_rng(seed).standard_normal((512, 512), dtype=np.float32)
                return host

            w = mx.random.normal((512, 512))
            with ThreadPoolExecutor(4) as pool:
                for request in range(150):
                    futures = [pool.submit(fill, request * 8 + i) for i in range(4)]
                    pending = []
                    try:
                        for number, future in enumerate(futures):
                            x = mx.from_dlpack(future.result(), copy=False)
                            y = x
                            for _ in range(6):
                                y = mx.tanh(y @ w)
                            mx.async_eval(y)
                            pending.append(y)
                            del x, y
                            if request %% 7 == 3 and number == 2:
                                raise RuntimeError("cancelled mid request")
                    except RuntimeError:
                        pass
                    finally:
                        gpu_sync.synchronize()
                    pending.clear()
            print("STRESS DONE")
        """.replace("%%", "%"), timeout=90)
        self.assertEqual((code, hung), (0, False), err[-6000:] + out[-2000:])
        self.assertIn("STRESS DONE", out)


class ProductionCodeTests(unittest.TestCase):
    def production_files(self):
        for base in ("veystrix", "models/flashnext"):
            for path in (ROOT / base).rglob("*.py"):
                parts = path.relative_to(ROOT).parts
                if "tests" in parts or any(part.startswith("test_") for part in parts[-1:]):
                    continue
                yield path

    def test_nothing_in_production_calls_mx_synchronize(self):
        offenders = []
        for path in self.production_files():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "synchronize":
                    target = node.value
                    if (isinstance(target, ast.Name) and target.id in {"mx", "mlx"}) or \
                            (isinstance(target, ast.Attribute) and target.attr == "core"):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
                if isinstance(node, ast.ImportFrom) and node.module in {"mlx.core", "mlx"}:
                    if any(alias.name == "synchronize" for alias in node.names):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} (import)")
        self.assertEqual(offenders, [], "use models.flashnext.gpu_sync.synchronize(); mx.synchronize holds the GIL while it waits")

    def test_the_request_cleanup_and_lifecycle_paths_use_the_safe_barrier(self):
        from models.flashnext import gpu_sync, pool_lifecycle, prefill_ring

        self.assertIs(pool_lifecycle._gpu_synchronize(), gpu_sync.synchronize)
        backend = (ROOT / "veystrix/backends/flashnext.py").read_text()
        self.assertEqual(backend.count("gpu_sync.synchronize"), 4)        # request cleanup, after_request, before_prefill, _release_between_turns
        self.assertIn("from .gpu_sync import synchronize", (ROOT / "models/flashnext/prefill_ring.py").read_text())
        self.assertTrue(callable(prefill_ring.fence))

    def test_a_failed_barrier_poisons_the_process_like_before(self):
        from unittest import mock

        from models.flashnext import gpu_sync, pool_lifecycle

        with mock.patch.object(gpu_sync, "barrier", side_effect=RuntimeError("device lost")), \
                mock.patch.object(pool_lifecycle, "mark_lifecycle_fault") as mark:
            with self.assertRaises(RuntimeError):
                gpu_sync.synchronize()
        mark.assert_called_once()
        self.assertIn("GPU synchronization failed", mark.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
