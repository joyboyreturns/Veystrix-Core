"""GPU synchronization that cannot deadlock against MLX's completion handler.

The failure (2026-11-08, first full-checkpoint AGENTIC run, preserved in
``results/flashnext/20261108-agentic-smoke-retry-gpu-hang/``):

* ``mx.synchronize()`` (and ``mx.synchronize(stream)``) wait for the command buffer with the Python GIL held.
* An MLX array built from Python-owned memory (``mx.from_dlpack(numpy_buffer, copy=False)``: the per-layer expert reads in
  ``expert_cache`` and ``store``, the decode pool in ``expert_pool``, the slab in ``slab_pack``) is released by Metal's
  completion handler once the GPU has finished with it. Its deleter drops a Python reference, which needs the GIL.
* If the buffer being waited on is the one whose handler is releasing such an array, the main thread waits for the
  handler and the handler waits for the GIL: a permanent deadlock. Nothing else is wrong; it only needs a request
  whose last pipelined decode step is still in flight when the request's cleanup synchronizes.

Reproduced offline without any checkpoint (``tests/unit/test_gpu_sync.py``): plain arrays synchronize fine, a DLPack-backed
array deadlocks ``mx.synchronize`` with the same native stacks as the real run, and ``mx.eval`` of a trailing op does not,
because ``mx.eval`` releases the GIL while it waits. A job queued behind earlier work on the same stream cannot finish
before it, so waiting for a tiny trailing job on every stream in use is a complete barrier.

Rules enforced by a source scan test: production code never calls ``mx.synchronize``; it calls ``synchronize()`` here.
The buffers keep their zero-copy ownership. Nothing is copied, nothing is freed early, and no native wait is made GIL-free
by a global setting: only the one blocking call is replaced.
"""
from __future__ import annotations

import atexit
import time

from . import pool_lifecycle

_EXIT_INSTALLED = [False]


def _streams(mx, extra=()):
    streams = [mx.default_stream(mx.default_device())]
    try:                                    # the keep-warm kernel runs on its own stream when that option is on
        from . import expert_cache

        warm = expert_cache._KEEPWARM_STATE.get("stream")
        if warm is not None:
            streams.append(warm)
    except Exception:
        pass
    streams.extend(extra)
    seen, unique = set(), []
    for stream in streams:
        key = repr(stream)
        if key not in seen:
            seen.add(key)
            unique.append(stream)
    return unique


def barrier(extra_streams=()) -> float:
    """Wait for every job queued so far on every stream in use. Releases the GIL while waiting. Returns seconds waited."""
    import mlx.core as mx

    from . import progress

    began = time.perf_counter()
    with progress.PROGRESS.waiting("sync", "gpu barrier"):
        for stream in _streams(mx, extra_streams):
            with mx.stream(stream):
                mx.eval(mx.zeros((1,)) + 1)
    return time.perf_counter() - began


def synchronize(extra_streams=()) -> None:
    """The replacement for ``mx.synchronize`` in production code. A failure poisons the process like before."""
    pool_lifecycle.synchronize_or_fault(lambda: barrier(extra_streams))


def drain(extra_streams=(), settle: float = 0.05) -> None:
    """``synchronize`` plus time for Metal's completion handlers to finish releasing operands.

    The barrier returns when the GPU is done; the handler that frees a Python-backed operand can still be running and
    still needs the GIL. Sleeping releases the GIL, so the handler completes. Use before anything that waits with the GIL
    held (interpreter exit, the ``mlx_vlm`` stream cleanup).
    """
    import gc

    synchronize(extra_streams)
    gc.collect()
    time.sleep(settle)
    synchronize(extra_streams)


def _drain_at_exit() -> None:
    try:
        drain()
    except BaseException:
        pass                       # never fail an interpreter exit


def install_exit_drain() -> None:
    """Drain the GPU and Metal's completion handlers before ``mlx_vlm``'s exit hook waits with the GIL.

    ``mlx_vlm`` registers ``mx.clear_streams()`` at exit. That call waits for the streams with the GIL held, so a job still
    in flight (or a completion handler still releasing a Python-backed operand) deadlocks the shutdown exactly like
    ``mx.synchronize`` did (reproduced: 6 of 6 hangs without the drain, 0 of 6 with it). atexit runs hooks last-in first-out,
    so ``mlx_vlm`` is imported first to register its hook, then this one is registered after it.
    """
    if _EXIT_INSTALLED[0]:
        return
    try:
        import mlx_vlm  # noqa: F401  (registers its own exit hook first)
    except Exception:
        pass
    atexit.register(_drain_at_exit)
    _EXIT_INSTALLED[0] = True
