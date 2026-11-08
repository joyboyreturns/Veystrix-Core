# AGENTIC smoke retry: prefill passed, then a GIL/Metal-completion deadlock at the start of decode (2026-11-08, battery 65%, PRELIMINARY)

Result: **FAIL, hang, no retry.** Admission passed, the cold prefill completed with healthy memory, and the process froze at the start of decode. No token was produced and no tool call happened. The guard did not fire (it was frozen too, see below). The harness waited 10 minutes; I stopped it.
Preserved: `results/flashnext/20261108-agentic-smoke-retry-gpu-hang/` (guard trace `veystrix-chat-13285.jsonl`, `server-stack-sample.txt` from `sample`, `server.log`, `task.jsonl`, `admission-log.jsonl`, `purge-log.jsonl`).

Run: same configuration as the first attempt (`./chat.sh --mode agentic --pool-gb 0 --auto-purge`, real backend, 127.0.0.1:8080, key set, context limit 4096, `dsh headless` with the glob+grep patch, 898-token request), on battery by instruction, Safari closed, nothing else running.

## What worked
* Admission (new rule): ADMITTED, headroom 4.18 GiB (other anonymous 2.69 GiB); the gate was 8.77 GiB and one startup purge brought it to 11.36 GiB (+2.8 GiB, 0.23 s). Same start condition as the failure but with 3.6 GiB less other anonymous memory.
* Cold prefill of 898 tokens: **18.5 s** (about 49 tok/s), against more than 46 s without finishing in the failed run. Page-ins during prefill 18.3 GiB.
* Memory during prefill: minimum available **4.68 GiB** (3.26 before), compressor growth peak **3.15 GiB** (5.51 before), pressure level 1 throughout, MLX peak 4.11 GiB, process RSS peak 4.10 GiB, swap-out counter flat, swap used 773.5 MB.

## The hang
At trace time 32.0 s the trace labelled the first decode row; the last row is t=33.2 s, then nothing for 10 minutes. `sample` of the server shows a deadlock, not a slow GPU:
* **Main thread** (holding the Python GIL): inside an MLX operation, `CommandEncoder::synchronize` -> `-[_MTLCommandBuffer waitUntilCompleted]`.
* **Metal completion thread** (`com.Metal.CompletionQueueDispatch`): inside the completion callback, destroying an `mlx::core::array::Data`; the destructor calls into the Python extension, which calls `PyGILState_Ensure` and waits for the GIL.
* The main thread waits for the command buffer to complete; the buffer cannot complete until its callback returns; the callback waits for the GIL the main thread holds. Three more Python threads are blocked in `take_gil` behind it, including the guard's tracer (which was in the middle of a `vm_stat` subprocess call), which is why the trace stopped.
* Most likely cause (inference, not proven): the freed array wraps Python-owned memory through DLPack or a numpy buffer (`mx.from_dlpack(np.frombuffer(...), copy=False)` in `expert_pool.py` and `expert_cache.py`), whose deleter needs the GIL. It is a latent race: five CHAT answers and many earlier runs passed. Which exact Python call held the GIL while synchronizing is not identified; the frames are inlined in the sample.

## Why the guard did not stop it
The runtime guard is a Python thread. A deadlock that holds the GIL starves it: no sample, no stop. This is not a resource stop and not a weakened guard, but it shows the guard cannot see this failure class. No guard threshold was touched.

## Shutdown
Harness client killed first; server stopped on SIGTERM within 2 s (its signal handler ran once the deadlocked thread was released by process teardown); port 8080 free; no model process left.

## Next
1. Confirm the mechanism offline with a regression test that frees a DLPack-backed array in a Metal completion handler while the main thread synchronizes without releasing the GIL (needs the real MLX build, no checkpoint).
2. Candidate fixes, none applied: make every synchronization point release the GIL; give DLPack-backed arrays a deleter that does not need the GIL; and add an out-of-process heartbeat watchdog so a frozen interpreter is killed and a trace written. Each needs approval because they touch the decode path and the safety layer.
3. Only then repeat the smoke test under the same conditions, once.

## Update 2026-11-08 (offline follow-up)

The mechanism was reproduced offline without a checkpoint and the blocking call was replaced; an out-of-process watchdog
now bounds any future stall. See `gpu-gil-deadlock-and-watchdog.md` for the diagnosis, the repair, the exit-hook variant
and the watchdog design, and `agentic-smoke-retry-proposal-2026-11-08.md` for the request to run one guarded retry. The
repair is **not** confirmed on the real model, and which Python call held the GIL in this run remains inferred.
