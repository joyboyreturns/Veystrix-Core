# GPU/GIL deadlock: diagnosis, repair and out-of-process watchdog

Status 2026-11-08. Offline only. **No full-checkpoint run has confirmed the repair.** Evidence for the original failure is
preserved in `results/flashnext/20261108-agentic-smoke-retry-gpu-hang/`; see also
`agentic-smoke-retry-gpu-hang-2026-11-08.md`.

## 1. Mechanism (reproduced offline, no checkpoint)

* `mx.synchronize()` (also the per-stream form, and `mx.clear_streams()`, which `mlx_vlm` registers as an `atexit` hook)
  waits on Metal command buffers **with the Python GIL held**.
* An MLX array built from Python-owned memory (`mx.from_dlpack(numpy_buffer, copy=False)`) is released by Metal's completion
  handler once the GPU is done with it. Its deleter drops a Python reference, which needs `PyGILState_Ensure`.
* If the waiter is the thread that holds the GIL while the handler is the one that must take it, neither can proceed.
  The process is alive, silent and unkillable by any Python-level guard (the `ChatGuard` tracer is itself a Python thread
  and is starved by the same lock).
* `mx.eval`, `.item()`, `tolist()` and numpy conversion release the GIL while waiting and are not affected.

Production DLPack sites: `expert_cache.py` (`_StreamedPackRead.to_mx`), `store.py`, `expert_pool.py`, `slab_pack.py`.
A second GPU stream exists for the optional keep-warm kernel.

What is **not** established: which exact Python call held the GIL in the real run. The inference is the request-cleanup
`mx.synchronize` after a request whose last pipelined decode step was still in flight. The request log was never written,
so this cannot be confirmed from the artifacts.

## 2. Repair: `models/flashnext/gpu_sync.py`

* `barrier()` queues a trivial op on every stream in use and `mx.eval`s it. Jobs on one stream complete in order, so waiting
  for the trailing job is a complete barrier, and `mx.eval` releases the GIL.
* `synchronize()` wraps the barrier in `pool_lifecycle.synchronize_or_fault`, so a failed barrier poisons the process
  exactly as a failed `mx.synchronize` did.
* `drain()` adds `gc.collect()` and a short sleep (GIL released) so completion handlers finish releasing operands.
  `install_exit_drain()` registers it after `mlx_vlm`'s hook so it runs first at interpreter exit.
* Nothing is copied, nothing is freed early, no global GIL setting is changed. Q4 weights, routing and every memory, swap,
  compressor, Metal and thermal guard are untouched.
* Call sites replaced: request cleanup, `after_request`, `before_prefill` (backend), `pool_lifecycle.make_room`,
  `prefill_ring.fence`. A source-scan test fails if production code calls `mx.synchronize` again.

Tests: `models/flashnext/tests/unit/test_gpu_sync.py` (12). Every GPU scenario runs in a child process with its own
timeout and in-child faulthandler. Included: the raw deadlock reproduced with the same native stacks (canary, skips itself
if a future MLX fixes it), barrier completeness, second stream, exit hook with and without the drain, and a 150-request
stress with worker threads, abandoned requests and a cleanup barrier after each. The exit-hook variant hung in 6 of 6
trials without the drain and 0 of 20 with it.

## 3. Out-of-process watchdog: `models/flashnext/watchdog.py`, `progress.py`

Why out of process: a deadlock that holds the GIL stops every Python thread in the worker, including timers and the
existing guard tracer. Only another process can observe it.

Design:

* The worker writes a small seqlock record in a private memory-mapped file: stage, request sequence, a **progress counter
  that only real work advances** (a completed streamed MoE layer in prefill, a decoded token in decode), the time of the
  last advance, token count and whether a response has started. No timer thread ticks it, because a ticking thread would
  hide exactly this failure.
* The watchdog keys on process identity `(start_sec, start_usec)` from `proc_pidinfo`, so a recycled pid is never
  signalled. It only ever signals the worker it was started for, never another application.
* Stage-specific deadlines in seconds (defaults: starting 300, loading 900, request 60, prefill 180, decode 60, sync 60,
  cleanup 60, stopping 60), configurable. A stage past its deadline is not killed if the process shows external activity
  in the recent window (CPU time or disk bytes read via `proc_pid_rusage`): slow SSD-bound work is distinguished from a
  stall. A per-stage hard limit (defaults: starting 900, loading 3600, request 300, prefill 1800, decode 600, sync 300,
  cleanup 300, stopping 180) ends even a stage that still looks active but whose progress counter has not moved. The
  defaults are conservative guesses, not measured limits; the idle stage has no deadline.
* On a stall, in order: write a report; `SIGUSR1` so the worker's `faulthandler` dumps Python stacks (works with the GIL
  held); a 3 s native `/usr/bin/sample`; tell the client; `SIGTERM`; wait out a grace period; `SIGKILL` only if the worker
  is still alive. Before every signal the process identity is re-checked.
* The worker passes the client socket to the watchdog (`SCM_RIGHTS`), so the watchdog can tell the client before
  terminating the worker. Before any response bytes: `503` with a JSON error (`code: worker_stalled`) and
  `Retry-After: 30`. After streaming began, a terminating error event in the client's own protocol (chat completions:
  `data: {"error": ...}` then `data: [DONE]`; `/v1/messages` and `/v1/responses` have their own failure events).
* Enabled by `FLASHNEXT_WATCHDOG=1` (set by AGENTIC mode). Other knobs: `FLASHNEXT_WATCHDOG_DIR`, `_DEADLINES`, `_HARD`,
  `_ACTIVITY_WINDOW`, `_POLL`, `_GRACE`, `_KILL_WAIT`, `_SAMPLE`. With the flag unset every worker-side call is a no-op.

Tests: `models/flashnext/tests/unit/test_watchdog.py` (29) including real processes: a worker frozen with the GIL held,
the known deadlock reproduced and bounded, pid-reuse refusal, healthy slow work left alone, and client error delivery.
The real-process group passed 4 consecutive runs.

## 4. Audit notes

* Lease and cancellation: the barrier runs in `finally`, so an abandoned or errored request still fences before its
  buffers can be reused. This audit read the code and exercised it in the stress test; it was not run against the real
  model.
* Python-owned buffers keep zero-copy ownership; the fix changes only how the wait is performed.
* Remaining exposure: any new native call that waits with the GIL held and can overlap a Python-backed operand would reopen
  the problem. The source-scan test covers `mx.synchronize` only. `mx.clear_streams` is covered at exit by the drain.

## 5. What is and is not verified

| Claim | Evidence |
| --- | --- |
| Raw `mx.synchronize` deadlocks with a DLPack operand | Offline reproduction, native stacks match the real run |
| `gpu_sync` removes that deadlock | Offline tests and stress, no real-model run |
| Exit hook hang fixed | 0 of 20 hangs offline |
| Watchdog bounds a frozen worker | Real-process tests |
| Repair works on the 113 GB checkpoint | **Not verified** |
