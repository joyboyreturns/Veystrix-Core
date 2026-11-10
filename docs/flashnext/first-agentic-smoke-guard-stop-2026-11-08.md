# First full-checkpoint AGENTIC smoke test: guard stop during the first prefill (2026-11-08, battery 67-69%, PRELIMINARY)

Result: **FAIL, resource stop, no retry.** The tool-call round trip did not happen: the cold system-prompt prefill had not finished when the safety guard stopped the process (`memory pressure level 2`, exit 75). The model never produced a token.

Run: `VEYSTRIX_SERVER_API_KEY=... VEYSTRIX_REQUEST_LOG=... ./chat.sh --mode agentic --pool-gb 0 --auto-purge`, real `FlashNextBackend`, 127.0.0.1:8080, context limit 4096, guard on, prefix cache on (RAM 0.75 GB / SSD 20 GB), balanced 2048 chunks, 512-expert ring. Harness: real `dsh headless` with `docs/flashnext/dsh/veystrix-agentic.patch.yml` (exposed tools: `glob`, `grep`; request 898 tokens), fresh `DSH_HOME`, workspace with only `alpha.txt` and `beta.md`.
Preserved: `results/flashnext/20261108-first-agentic-smoke-guard-stop/` (guard trace `veystrix-chat-7444.jsonl`, 302 rows at 0.25 s; `server.log`; `task.jsonl`; `purge-log.jsonl`). The per-request log was never written because the request did not finish.

Before the run: suites green (VEYSTRIX 465, FlashNext unit 1,335, 3 skipped); gate READY at 10.3 GB after one startup purge (+1.8 GB, 0.18 s); pressure normal, swap used 773.5 MB and flat, Low Power Mode off, battery 69%, no thermal warning. One earlier attempt failed in 1 s on my own harness command-line order (`--patch` must precede `--json`); it sent no request and is not a resource retry.

## Timeline (GiB; t = seconds from guard start)
| t | stage | MLX active | anonymous | compressor (growth) | available | process RSS |
|---|---|---|---|---|---|---|
| 0 | start, after purge | 0 | 6.31 | 0.97 (0) | 9.59 | 0.04 |
| 3 to 35 | model loaded, idle | 2.77 | 8.8 | 0.98 | 6.3 to 6.4 | 3.09 |
| 37.8 | prefill begins | 4.06 | **4.7** | **3.07 to 3.9 (+2.1 to +2.9)** | 5.1 to 4.5 | 1.64 |
| 41 to 75 | prefill | 2.9 to 3.0 (peak 4.62) | 4.0 to 3.0 | 4.2 to 5.4 (+3.2 to +4.4) | 4.1 to 4.8 | 1.0 to 1.7 |
| 80 to 83 | prefill | 3.0 to 3.2 | 3.6 to 2.45 | 5.9 to 6.5 (+4.9 to +5.5) | 3.5 to 3.3 | 1.0 to 1.8 |
| 82.8 | **pressure level 2, stop** | 3.03 | | 6.48 (+5.51) | 3.34 | 1.19 |

No swap-out (counter flat), no purge during the request, available memory never went below the 3 GiB floor (minimum 3.26). The stop rule that fired is the macOS pressure level (stop at 2), unchanged.

## Findings
1. The 9 GiB gate said 9.59 GiB, but about 2.5 GiB of that was other processes' idle anonymous memory (inactive anonymous pages count as "available"). Anonymous memory was 6.3 GiB at start with Safari/WebKit and the Claude client running. When the prefill started, macOS compressed it: anonymous 8.8 to 4.7 GiB, compressor +2.9 GiB within seconds, of which only about 1.5 GiB was this process (RSS 3.09 to 1.64). This is the known first-prefill compressor lump (CHAT showed +2.4 GiB; F0 +2.75 GB), and it was larger here.
2. The prefill then ran for more than 45 s and was not done. A cold prefill streams almost every expert: earlier runs read about 50 GB for 1K tokens in 25 s and 150 GB for 2K tokens in 65 s. File-backed memory held 4 to 5.3 GiB throughout, and the compressor kept growing (to +5.5 GiB, against +2.6 to +3.2 GiB in the earlier 1K and 2K runs) until macOS reached pressure level 2.
3. Earlier real 1K and 2K prefills passed on a cleaner machine (AC, fewer apps) with a minimum available of 3.18 to 3.49 GiB: the same thin margin. Today's machine state (battery, about 6.3 GiB of other anonymous memory) tipped it. This is not evidence of a harness or protocol problem: the request was well formed and the server was in prefill.
4. AGENTIC differs from CHAT here only in that its first request is a cold prefill of about 900 tokens (about 50 to 100 GB of expert reads) instead of about 30 tokens.

Unverified: how much of the 2.9 GiB lump was other processes versus this one (only RSS is attributable); whether the 898-token prompt would have finished in about 30 s and 100 GB on a quiet machine; the chunk plan actually used (the request log was not reached).

## Not done, deliberately
No retry, no change to guards or limits, no further optimization flags, no 2K cache test.

## Next bottleneck
Cold-prefill memory headroom: the first-prefill compressor lump plus file-cache churn, on a machine whose gate figure includes inactive anonymous memory. Candidates, none applied: (a) run plugged in with Safari and other heavy apps closed, then repeat the identical smoke test; (b) make the readiness figure subtract inactive anonymous memory so the gate reports what a prefill can actually use; (c) the investigation items already proposed in `prefill-memory-investigation.md` (cap the file cache, stream the embedding, per-layer expert counter); (d) a smaller chunk size, which trades time for working set.

Follow-up: the admission rule that came out of this analysis is documented in `agentic-prefill-readiness.md` (offline only; the full model has not been run again).
