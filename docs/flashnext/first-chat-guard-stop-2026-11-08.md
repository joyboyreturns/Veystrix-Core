# First full-checkpoint CHAT: compressor guard stop (2026-11-08, battery 77%, PRELIMINARY)

Trace (preserved): `results/flashnext/20261108-first-chat-guard-stop/veystrix-chat-85653.jsonl` (94 rows, 0.25 s samples).
Run: `./chat.sh --mode chat --pool-gb 2 --auto-purge`. Output reached "A rainbow", then the guard stopped the process:
`compressor grew 2.04 GiB since the arm began (limit 2 GiB) while only 2.94 GiB was available (floor 3 GiB)`.
The trace's `phase` stayed `init` the whole run (the chat never set phases), so stages below are placed from MLX memory jumps and the
`pool.exists` field. That gap is fixed (see "Changes").

## Memory timeline (GiB; t in seconds from the guard's start)

| t | Stage | MLX active | Available | Compressor (growth) | File-backed | Process RSS |
|---|---|---|---|---|---|---|
| 0.0 | process start, after the startup purge | 0 | 9.15 | 0.86 (0) | 0.59 | 0.04 |
| 1.2 to 2.4 | weights load | 0 to 2.60 | 8.93 to 6.40 | 0.86 (0) | 0.87 to 3.55 | to 2.92 |
| 2.6 to 10.3 | idle at the prompt (about 8 s) | 2.60 | 6.4 to 6.5 | 0.86 (0) | 3.6 to 3.85 | 2.92 |
| 10.6 | prefill begins | 2.61, peak 3.29 by 10.9 | 6.07 | 0.89 | 3.86 | 3.31 |
| 12.0 | compressor lump | 2.66 | 4.77 | **3.24 (2.38)** | 4.74 | **1.19** |
| 12.07 | `compressor_warning` | | 4.77 | growth 2.43 (limit 2.0) | | |
| 12.4 to 21.7 | prefill (about 11 s) | 2.63 to 2.77, peak 3.36 | **4.13 to 5.1** | 4.2 to 4.9 (**2.7 to 4.07**) | 4.0 to 5.0 | 0.46 to 0.94 |
| 21.7 | just before the pool | 2.77 | 4.35 | 4.21 (3.35) | 4.66 | 0.74 |
| 22.0 | **pool created** (1,999,880,192 bytes, 651 slots) | **4.61 (+1.84)** | **3.21** | 4.27 (3.41) | 2.90 | 0.55 |
| 22.4 to 23.2 | first decode tokens | 4.67 to 4.74 | 3.17, 3.03, 2.97, **2.94** | 3.39, 2.85, 2.85, **2.90 (2.04)** | 2.85 to 2.79 | 0.93 to 2.07 |
| 23.23 | **guard stop** | 4.74 | **2.94** | growth **2.04** | 2.79 | 2.07 |

Swap: no swap-out (counter flat), 8 pages swapped in (harmless), swap used constant at 773.5 MiB, pressure level 1 throughout. Battery 77%,
Low Power Mode off, no thermal warning. MLX peak 4.76 GiB, allocator cache at most 0.15 GiB (nothing to reclaim there).
Startup purge (`~/.flashnext/purge-log.jsonl`): one purge before the load, 7.75 to 9.91 GB available (+2.17 GB) in 0.20 s, file-backed 4.86 to 0.51 GB.
No other purge fired (the idle monitor saw 6.4 GiB, above its 5.1 GiB threshold, and never acts during a request).

## Root cause
Two things together, in this order:
1. **Precondition, before any pool existed:** at the first prefill (t=10.6 to 12.0) the compressor grew 2.4 GiB and then to 4.07 GiB, while the
   process RSS collapsed from 2.92 to 0.46 GiB. The most likely reading is that macOS compressed the engine's own idle resident weights (about 2.4 GiB,
   untouched for 8 s) as the prefill's buffered expert reads grew the file cache. This is the same first-prefill compressor lump seen earlier (F0: +2.75 GB at 128
   tokens). It sat above the 2 GiB limit for 10 s as a warning only, because available memory stayed above 3 GiB (4.13 minimum).
2. **Trigger:** creating the decode pool at the first decode token took available memory from 4.35 to 3.21 GiB within 0.3 s and to 2.94 GiB by the next tokens
   (the pool's pages are touched at creation and by the first token's expert fill, about 1.4 GiB per token of reads). That crossed the 3 GiB floor while the
   compressor was still 2.04 GiB above its start (it was decaying from 4.07). The stop was marginal: 0.04 GiB over the growth limit, 0.06 GiB under the floor.

The pool is therefore the proximate cause and the compressor lump the necessary precondition; neither alone would have fired the guard. Prefill itself,
with heavy buffered streaming and growth of 4 GiB, held at least 4.13 GiB available. No external memory change is visible: system-wide anonymous memory and
file cache moved only in step with our own load and prefill, and the purge preceded the run.

Uncertain: whether the 2.4 GiB compressed was the engine's weights or partly other processes' memory (system anonymous memory fell 3.3 GiB while only our
RSS is attributable); why available fell a full 1.14 GiB in the very first 0.3 s after creation, before a token could have been read (lazy page touching by the
Metal buffer wrap is a hypothesis); and whether the no-pool path stays above the floor for a whole 64-token answer (it did through an 11 s prefill, but is unmeasured for decode).

## Changes (all uncommitted, opt-in or default-safe)
* `FLASHNEXT_EXPERT_POOL_AUTO=1` (CHAT and AGENTIC default; `--pool-gb auto`): the pool, still created lazily at the first decode, is allocated only if
  `available - pool >= 3 GiB guard floor + 0.5 GiB margin` (a 2 GB pool needs 5.36 GiB) and pressure is below 2, measured after trimming the allocator cache.
  Otherwise there is no pool: `StreamingSwitchGLU._pool_usable()` sends decode to the ordinary read path (the pool-0 path), the refusal is printed once and remembered
  for the request (no per-layer polling) and re-checked at the next request. Under the numbers measured in this trace (4.35 GiB) the pool is refused.
* A fixed `--pool-gb 2` still allocates unconditionally and the banner now says that it stopped the guard in the first real chat. 1 GB is not substituted.
* The safety trace now labels `prefill`, `decode` and `idle`, so the next trace places every reading.
* Guards are unchanged (9 GiB start, compressor and available floor, swap, thermal, battery, GPU fence). No purge runs during inference.

Offline tests (`test_pool_headroom.py`): the measured conditions are refused and 6.4 GiB allowed, the guard's own 3 GiB constant is asserted, refusal is sticky per
request and retried on the next, a fixed pool is not checked, allocation failure and cancellation during construction leave nothing behind, no-pool decode routing,
repeated short chat turns with the pool declined, trace phases including a cancelled request.
