# Autonomous memory preparation and tiered AGENTIC admission

Status 2026-11-08. Offline only. Nothing here has been validated on the 113 GB checkpoint beyond the two earlier real AGENTIC
runs whose traces it uses.

## 1. The problem it answers

The 2026-11-08 AGENTIC admission refused a launch with 8.87 GiB available (gate 9.00) and 3.54 GiB cold-prefill headroom
(needs 4.00), because other processes held 3.58 GiB of anonymous memory. Nothing was loaded. The refusal was correct, but it
asked a person to free a few hundred MB by hand, and it assumed one fixed 6.0 GiB cold-prefill peak for every launch.

## 2. What the 898-token fixture actually costs (measured, real model)

From the guard traces in `results/flashnext/` (`mlx_*`, `rss`, `available_mb`, `compressor_growth_mb`, `anonymous_mb`):

| | Completed prefill (retry run) | Stopped prefill (first run) |
| --- | --- | --- |
| MLX peak | 4.11 GiB | 4.62 GiB |
| MLX active, max | 3.89 GiB | 4.42 GiB |
| MLX allocator cache, max | 0.31 GiB | 0.67 GiB |
| Process RSS, max | 4.10 GiB | 3.72 GiB |
| Compressor growth, max | 3.15 GiB | 5.51 GiB |
| Available memory, minimum | 4.68 GiB | 3.26 GiB |
| Pressure, max | 1 (normal) | 2 (stopped) |
| Other processes' anonymous memory at start | about 2.8 GiB | about 8.9 GiB |

The first run is the one that stopped: 8.9 GiB of other anonymous memory is what the budget refuses. The retry run had about
2.8 GiB and completed the prefill at normal pressure. The earlier 2,046-token real arm peaked at 5.40 GiB.

The peak scales with the tokens prefilled. Full-sequence logits alone are 248,320 x 2 B = 0.47 MiB per token (kept on purpose;
last-row projection changed logits, see `prefill-memory-investigation.md`), and the per-layer transient grows with the chunk.
The two measured points (898 tokens at 4.62 GiB, 2,046 at 5.40 GiB) imply 0.68 MiB per token, which is above the logits slope
alone, so interpolating between them is conservative.

## 3. Tiered admission

`prefill_readiness.cold_prefill_peak_bytes(tokens)` interpolates linearly between the two measured peaks, never goes below
the 898-token value, never above 6.0 GiB, adds the 0.3 GiB of process memory and **rounds up** to 0.25 GiB.

| Tier | Context | Single cold prefill up to | Budget |
| --- | --- | --- | --- |
| standard | 4096 | 4096 tokens | 6.00 GiB (unchanged) |
| compact | 2048 | 1536 tokens | 5.50 GiB |
| minimal | 1536 | 1024 tokens | 5.25 GiB |

Admission takes the largest tier that passes the **unchanged** rules: 9 GiB start gate, pressure and swap-out, and 4 GiB of
headroom. No threshold was lowered. The server then **enforces** the admitted tier: a request whose context exceeds it, or
whose single prefill (a rebuild, or the new tokens of a continuation) exceeds it, is refused with `400
prefill_budget_exceeded` before any state changes, so the assumption is never exceeded silently.

Evidence limits, stated plainly: two real runs at 898 tokens and one at 2,046. Nothing between is measured (hence the rounding
up), nothing above 2,046 is measured (hence the standard tier keeps 6.0 GiB), and the 0.75 compression ratio comes from one
run. The runtime guards (compressor, swap, pressure, thermal, GPU) are untouched and still stop a run that the model
mispredicts.

## 4. The preparation procedure (`resource_prep.py`, runs before every flashnext model load)

Bounded: at most 3 rounds, 120 s, settle waits of 2, 4 and 8 s after a helpful step (one base wait after an idle round). Per
round, in order:

1. **Veystrix-owned memory** (allocator cache, RAM prefix snapshots, expert staging). Nothing is loaded at startup, so this
   is recorded as a no-op; the same hooks serve an idle manager.
2. **File-cache purge** through `purge_recovery` only: never overlapping a load or a request, needs `sudo -n` permission
   (present here), skipped when file cache cannot change the verdict (a purge against 3.58 GiB of anonymous memory would
   only cost SSD reads later), backoff after an ineffective purge. Now on by default for flashnext; `--no-auto-purge` or
   `FLASHNEXT_AUTO_PURGE=0` turns it off.
3. **Optional graceful app closing** (`app_closer.py`), **off**. It needs `~/.flashnext/close-apps.json` with a literal
   `"enabled": true` and a list of names, and then only asks those apps to quit the way a person would (an app with
   unsaved work keeps its save prompt and is reported as still running, never forced). A protected list (window server,
   Finder, Dock, Siri, terminals, Claude Code, login window) can never be closed even when listed, and names are restricted
   to plain characters.
4. Settle, read again, decide.

It stops early when no step could help, so there is no endless ineffective loop. AGENTIC then exits 69 with **one** report
(what was read, what each step did, how much unrelated anonymous memory would have to drop, the optional allowlist). CHAT
only prepares for the 9 GiB gate; the guard that follows keeps deciding.

Never, by construction and by a source-scan test: signalling or killing any process, purging during inference, touching
unsaved work, weakening a guard, or counting inactive anonymous memory as file cache.

## 5. Current readiness on this machine (read-only evaluation, 2026-11-08 ~18:40)

Gate figure 8.35 GiB (short by 0.65 GiB, up to 4.75 GiB of file cache recoverable), pressure 1, swap used 0.72 GiB with no
swap-out, other anonymous memory 3.89 GiB. Standard and compact tiers are not admitted by the budget; the **minimal** tier
passes the headroom rule (4.09 GiB against 4.00 GiB). So preparation would purge file cache (closing the gate) and admit at the
minimal tier, context 1536 and cold prefill up to 1024 tokens. The margin is small and moves with what else is running.

## 5b. First real use (2026-11-08, battery)

On the real launch the preparation purged 2,318 MB of file cache by itself (8.93 to 11.09 GiB available, 0.2 s), the tiered
admission chose **compact** (headroom 4.15 GiB), and the run needed about 4.1 GiB of MLX memory against the 5.5 GiB budget,
with pressure 1 and no swap-out. No manual step was needed. See `agentic-smoke-retry-4-2026-11-08.md`.

## 6. Working-set levers: what was and was not changed

Unchanged and not adopted, because each needs an approved real-model arm and none has a measured speed or quality result:
512 MB allocator cache cap (`FLASHNEXT_PREFILL_CACHE_LIMIT_MB`, up to about 0.8 GB resident, unmeasured), streamed embeddings
(`FLASHNEXT_STREAM_EMBED=1`, up to 0.4 GB), a smaller scratch ring (rejected by a real run: 256 experts fell back to buffered
reads), per-layer expert counts for a possibly smaller ring (needs the counter's data). The proven 0 GB expert pool and the
balanced 2,048-token chunk plan stay as they are. The only change to the working set is that a launch no longer reserves
memory for a prefill it will not run.

## 7. Tests

`models/flashnext/tests/unit/test_resource_prep.py` (28): tier arithmetic and invariants; tight memory admits a smaller tier
under the unchanged 4 GiB rule while the gate and pressure still refuse every tier; recoverable file cache is purged and the
launch admitted; unrelated anonymous memory is never purged against and yields one report; an ineffective purge runs once;
the deadline and round bounds; no purge during a request or load, under critical pressure or without permission; CHAT only
prepares for the gate; the log is private; nothing can signal a process; the app closer is off by default (every disabled
form), lists only, protects, rejects injection and leaves unsaved-work apps alone; startup wiring and CLI default.
