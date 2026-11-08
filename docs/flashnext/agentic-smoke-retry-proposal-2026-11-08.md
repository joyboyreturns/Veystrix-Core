# Proposal: one guarded full-model AGENTIC smoke test (APPROVED AND USED; see the result note)

> Update 2026-11-08: the owner approved one attempt. It ran once (battery, preliminary) and is documented in
> `agentic-smoke-retry-4-2026-11-08.md`: prefill, decode start and shutdown passed; the harness's own one-token output
> cap prevented a tool call. The cause is fixed offline. A further run needs a new approval. The text below is the
> original proposal, kept as written.

Status 2026-11-08. This is a request for a decision. **Nothing here has been run against the 113 GB checkpoint, and it
will not be until the owner approves.** Two real attempts exist: the first stopped by the memory guard in cold prefill
(`first-agentic-smoke-guard-stop-2026-11-08.md`), the second by the GPU/GIL deadlock
(`agentic-smoke-retry-gpu-hang-2026-11-08.md`).

## What has changed since the second attempt (all offline)

| Change | Evidence |
| --- | --- |
| Deadlock mechanism reproduced and the blocking call replaced (`gpu_sync`) | Offline reproduction with the real native stacks, 12 tests, 0 of 20 exit-hook hangs |
| Out-of-process watchdog bounds any future stall and tells the client | 29 tests including real processes and the known deadlock |
| Admission gate (9 GiB plus pressure, swap and cold-prefill budget) before any load | Passed on the second attempt; unchanged |
| One-request scheduler, bounded busy answer, disconnect handling, `/ready` | 30 new tests plus a `serve()` run with a fake model |
| Output reserve, standard errors, model-versus-client error split, warm-conversation guard | Tests in `test_server_runtime.py` |
| Regression suites | See `gpu-gil-deadlock-and-watchdog.md` and the final run recorded in the session report |

What is **not** known: which Python call held the GIL in the real run (inferred, not observed), and whether the repair
removes the real failure. The watchdog is the safety net for that uncertainty, not a proof of the fix.

## The test

One attempt, one process, the same minimal harmless DeepSeek Harness task as the previous attempt (a short system
prompt, one harmless tool call, one tool-result continuation; see `agentic-harness.md` for the exact command, whose
`--patch` flag must precede `--json`). Launch: `./chat.sh --mode agentic --pool-gb 0`. The AGENTIC default is an
automatic decode pool of up to 2 GB when memory allows; this test overrides it with the verified 0 GB fallback so the
repair is not confounded with a pool decision and no pool is allocated without real headroom. Otherwise the preset is
unchanged: balanced 2,048-token prefill, exact prefix cache, 4096 context limit, title request disabled.

### Preconditions (all must hold; any miss means do not launch)

1. Owner approval for this specific run, given after reading this document.
2. Plugged in with Low Power Mode off for a final-quality run, or an explicit owner statement that a battery run is
   accepted as preliminary (as for the second attempt). The power state is recorded either way.
3. Nothing heavy open (the second attempt was first blocked by a browser holding memory); the readiness gate reports
   admitted, not just "passed earlier".
4. Free SSD space above the existing reserve; checkpoint path unchanged; no purge during inference.
5. `FLASHNEXT_WATCHDOG=1` (AGENTIC sets it) and the watchdog log directory writable.
6. `git status` is recorded at launch so the code revision under test is known (nothing is committed or pushed by this
   work; the revision is the working tree).

### What is watched

* The existing guards, unchanged (memory pressure, swap, compressor growth, Metal, thermal). Any stop ends the test.
* The watchdog stage deadlines (starting 300 s, loading 900 s, request 60 s, prefill 180 s, decode 60 s, sync 60 s,
  cleanup 60 s). The earlier measured 898-token cold prefill took 18.5 s, far inside the prefill deadline. SSD-bound work
  with external activity is not killed at the soft deadline; the hard limits still apply.
* A total wall-clock cap of 20 minutes for the whole attempt, enforced by the person or session running it.

### Stop rules

* Any guard stop, watchdog action, memory pressure at level 2 or above, swap-out, or thermal warning: stop, collect
  evidence, **do not retry automatically**, diagnose offline.
* A hang that the watchdog ends is itself a finding: the watchdog report, Python frames and the 3 s native sample are the
  deliverable, and the next step is offline.
* Never terminate anything except the Veystrix worker and its own watchdog.

### Success criteria

1. The harness completes the task: at least one tool call in the model's output, executed by the harness, and the answer
   after the tool result.
2. No guard stop and no watchdog action.
3. Every request reaches idle (`/health` shows `busy: false`); the server exits cleanly on shutdown within 60 s with no
   hang in the exit hook.
4. Records saved under `results/flashnext/`: guard trace, watchdog log and progress stages, request log, `/health`
   snapshots before and after, power state, memory samples, `git status`.

Even a full success proves one short task. It does not validate a long session, 4K or larger contexts, the prefix cache
across processes, or Hermes and OpenClaw compatibility; those remain separate, individually approved steps.

## Decision requested

Approve or decline this single attempt. Alternatives if declined: (a) a CHAT-only sanity run on the real model (it
exercises the repaired barrier and exit hook but not the server or the watchdog under load); (b) more offline work first,
for example a longer fake-model soak of the server and watchdog together.
