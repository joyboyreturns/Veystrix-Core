# Overnight operation, DSH stream timeout, 50 GB cache, decode (2026-11-09)

Follow-up to `deepseek-swap-incident-2026-11-09.md`. Branch `development3.8`, nothing committed. Real checkpoint, AC power, Low Power Mode off.

## What was wrong and what changed

| Item | Finding | Change |
|---|---|---|
| DSH "pi-ai stream idle timeout" | DSH aborts a stream after `streamIdleTimeoutMs` (default 300,000) without a real chunk. SSE comments (`: generating`) do not count. A cold 16K prefill takes ~440 s. | DSH profiles (headless + desktop): `streamIdleTimeoutMs` and `timeoutMs` 7,200,000. Composed config checked with `dsh --dump-config`. **Restart the DSH desktop app to load it.** Qwen Code `~/.qwen/settings.json`: same two values (its 900,000 was ~ one 32K cold prefill). Backups `*.pre-timeout.bak`. |
| Retry across a restart | DSH retries only listed codes. | DSH: `retryableCodes: [TRANSPORT]`, 20 retries, 15-60 s apart (~15-20 min): connection refused/reset while the supervisor restarts. SERVER (503) and TIMEOUT are never retried. Server marks `memory_critical_stop` with `x-should-retry: false`. |
| Malformed tool call ended a DSH run | Real session: model called `bash` without `description`; the server's strict validation failed the whole request (400) and `dsh` exited 1. | `VEYSTRIX_INVALID_TOOL_CALLS=pass` (set by `serve.sh`; default stays `error`): the call goes to the client unchanged (flagged `invalid_tool_call_passed` in the flight log), nothing is invented. Real DSH vs a stub: DSH answered with its own tool error, the model retried, exit 0. |
| SSD cache | 24 GB quota. | 50 GB (`modes.PREFIX_DISK_GB`), `VEYSTRIX_CACHE_GB` overrides. Oldest-accessed unleased leaf is evicted first (existing LRU); 50 GB free-disk reserve still applies. Live: budget 50,000,000,000 B, 15 entries, 2.9 GB. |
| Launch it yourself | none | `./serve.sh`, `./serve.sh --overnight`, `./serve.sh --status` (`models/flashnext/overnight.py`). |

## Supervisor (`./serve.sh --overnight`)

Runs the one supported `chat.sh --mode agentic` line under `caffeinate -ims`. Never retries a request, never deletes swapfiles or caches.
Owner stop (SIGTERM/Ctrl-C), exit 70 and exit 71 (port already served) do not restart. Guard stop (75) and admission refusal (69): 180 s cooldown,
wait for swap to stop moving, at most 4 per 6 h. Any other exit: 20 s doubling to 600 s, at most 8 per 30 min. `/health` gone for 5 min after ready: kill and restart.
When a budget runs out it stops and records why. Logs: `~/.flashnext/logs/supervisor.jsonl`, `supervisor-status.json`, `server-launch-*.log` (12 kept).

**Policy decision to know about:** you asked that the system not stop; the earlier rule was no automatic retry after a safety stop. The supervisor restarts the
*server* after a guard stop (bounded, after settling), never the request. If you want a guard stop to end the night instead, use `--max-guard-stops 0`.

Bug found by real use and fixed: a second launch while a server was running logged a false "ready" (the other server's `/health`) and would crash-loop.
Now refused (exit 71), and ready is only reported while the supervisor's own child is alive.

## Real runs (all under the supervisor, swap guard active)

* Real `dsh` session 1, cold, 5.6K-token prompt: 258 s, exit 0, swap +1.0 GB peak.
* Real `dsh` session 2 (new session): 5,385 of 5,603 tokens restored from SSD in 0.28 s, 218 prefilled. This is the cross-session system-prompt reuse. (That run failed on the malformed `bash` call, fixed above.)
* Real `dsh` tool loop, two more new sessions after a restart: both exit 0 with correct answers; session start `SNAPSHOT_HIT` 5,385 tokens; each later tool turn prefilled 20-112 tokens on the live state; swap growth 0.
* `kill -9` of the server: supervisor restarted it, ready again after ~50 s; SSD cache intact.
* Unit suites (Haiku, unverified by me beyond counts): macqwen 640 OK, flashnext unit 1,611 OK (3 skipped).
* Second `./serve.sh` while one runs: refused, nothing started.

## Decode speed

No exact acceleration found. Same script, 96 tokens, temperature 0: pool 0 tail 3.13 and 2.16 tok/s; pool 2 GB 2.68 and 2.23 tok/s, so the 2 GB pool gives nothing measurable (default stays 0). Decode reads ~0.8 GB/token from SSD; exact native MTP was measured slower on this machine (`research.md`); the faster routes (top-k cuts, no PLE) change output and were rejected before. The practical levers are fewer decode tokens (reasoning effort off or low) and the cache, which removes prefill, not decode. Not run: 24K/32K cold prefill, a real guard stop under pressure (offline-tested only).

## Not done / open

* Sampling is the saved chat preference (temperature 1.0, top_p 0.95: the card's thinking-mode values). Lower temperature may reduce malformed tool calls; not changed.
* No client namespace partition of the SSD cache; no real 32K run; DSH desktop must be restarted to read the new profile.
