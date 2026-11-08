# Automatic cache recovery (`FLASHNEXT_AUTO_PURGE=1`)

Opt-in, off by default, works in CHAT and AGENTIC (same process path). Implemented in `models/flashnext/purge_recovery.py`.
It exists so a blocked start (or a squeezed idle engine) recovers without a password prompt or a manual `sudo purge`.
It is **not** a periodic purge: `FLASHNEXT_PURGE_CHECK_SECONDS` (default 5) is how often memory is *looked at*.

```sh
FLASHNEXT_AUTO_PURGE=1 FLASHNEXT_PURGE_CHECK_SECONDS=5 ./chat.sh --mode chat --pool-gb 2
./chat.sh --mode chat --pool-gb 2 --auto-purge          # same thing; the banner shows "auto purge   ON"
python -m models.flashnext.purge_recovery               # accounting and the decision, purges nothing
python -m models.flashnext.purge_recovery --recover     # run the startup recovery once (purges only if justified)
```

## When a purge runs (every condition must hold)
1. The engine is idle: no model load, no request (prefill, decode, GPU work and expert reads all happen inside `generate`, which now holds
   an activity record for the whole call including its cleanup), no other purge. A request that arrives mid-purge waits for it.
2. Available memory (`free + inactive + speculative`, the quantity `ChatGuard` gates on) is below the need: 9 GiB (9,663,676,416 bytes) plus a
   128 MiB margin at startup; `FLASHNEXT_PURGE_IDLE_HEADROOM_GB` (default 5.5 GB, the memory manager's constrained threshold) while idle.
3. A purge could close the gap: the file-backed pages not already counted as available are at least the shortfall and 256 MiB.
4. No swap-out is in progress, pressure is below critical (level 4), `sudo -n` is permitted.
5. Veystrix-owned memory was reclaimed first: allocator cache always; RAM prefix snapshots move to SSD only if pressure is 2 or more or
   available is under 3.5 GB. The decode pool is never touched. If that alone fixes the gap, no purge happens.
6. The backoff allows it. Ineffective purge (under 256 MiB recovered, or a failure): wait 30 s, 60 s, 120 s ... capped at 30 min; one effective purge
   resets it. Even after an effective purge at least 60 s pass, so refill-then-purge cannot loop tightly. No daily count limit.

Only `sudo -n /usr/sbin/purge` is executed (argv, no shell, no terminal input, 30 s timeout), plus `sudo -n -l /usr/sbin/purge`, which lists the
permission and runs nothing. If sudo wants a password the feature records "not permitted", skips, and re-checks every 10 minutes. It never
prompts and never edits sudoers or SIP. On this machine `sudo -n -l` shows `(root) NOPASSWD: /usr/sbin/purge` already present (you added it);
I did not touch it.

Startup: the recovery loop runs **before** the guard starts and before the model loads, waits up to 90 s (`FLASHNEXT_PURGE_STARTUP_WAIT_SECONDS`),
and then hands over to the unchanged guard, which applies its own 9 GiB rule. Idle: a daemon monitor ticks every 5 s while the chat prompt or the
server is waiting, skips the tick entirely when busy, and on exit lets an in-flight purge finish before stopping.

## Records
`~/.flashnext/purge-log.jsonl` (mode 0600, `FLASHNEXT_PURGE_LOG`): one `purge` row per attempt (available before and after, recovered bytes, duration,
file-backed before and after, owned memory reclaimed, return code, next allowed in), a `refill` row when the next purge or shutdown arrives (file cache
regrown, SSD bytes the engine read since), and a `summary` row at exit (count, effective, failed, seconds, skip reasons).

## Why the gate can fail with green pressure and about 7 GB of cached files
Validated on live `vm_stat` and `sysctl vm.page_*` counts: `file_backed + anonymous == active + inactive + speculative` holds exactly, so the gate's
`free + inactive + speculative` counts nothing twice and needs no correction. But it is a queue measure, not a cache measure:
* It leaves out **active** file-backed pages (recently used cache, such as the checkpoint's own shard reads). Activity Monitor's "Cached Files"
  (`file_backed + purgeable`) includes them, so it can read several GB while the gate reads low. A purge frees them and raises the gate figure by up to
  `file_backed - speculative`; it adds nothing for inactive pages, which are already counted.
* It includes **inactive anonymous** pages (application memory that must be compressed before reuse), so it can read higher than what is free to use.
* Pressure is green (level 1) because macOS can still reclaim cache; the gate asks for more headroom than macOS needs.
The bounds are exposed by `accounting()`: the lower bound assumes every inactive page is file cache (gain 0), the upper bound that none is. The
inactive split is not available from `vm_stat`, so a purge is attempted on the upper bound and the measured result teaches the backoff. The 9 GiB rule,
swap, thermal, battery and GPU guards are unchanged.
Recorded example (`phase2-engineering-evidence.md` section 9): available 7.79 GB, file-backed 5,125 MB, swap 797 MB, pressure 1: the gate blocked while
up to about 5 GB was recoverable.

## Verified, and not
* Offline: 27 tests (repeated purges, no daily limit, minimum spacing, unavailable sudo, failing and ineffective purges with exponential backoff and cap,
  concurrent request and purge, two attempts never overlapping, cancelled request releases the gate, load blocks purge, startup flow, idle monitor, shutdown
  during a purge, argv and stdin of the real runner, private log, accounting on a real `vm_stat` sample, backend wiring).
* Live, no purge: the status command on this machine reports the identity true and `sudo -n` permitted.
* **Not exercised:** a real purge through this code, the effect of a real purge on available memory, and the refill and SSD cost after one. They are
  recorded in the log the first time the feature fires. The cost to expect: the next request re-reads shard pages the cache had held.
