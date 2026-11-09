# Overnight architecture results (2026-11-09 to 2026-11-10)

Engineering report for the prefill and decode work on Qwen3.8-Flash-Next 4-bit (MacBook Air M4, 16 GB, fanless). Everything
below was measured on the real checkpoint (`/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`) with the supervised harness
(`serve.sh` server, independent watchdog, swap/power/thermal stops, uncached prefill reads, reversed pairs) unless it says
"unit test". Raw arms are under `results/flashnext/`; design and per-run detail are in `docs/flashnext/architecture-overhaul.md`.
All of it is **opt-in**; the default server is unchanged chunk-major. Nothing was pushed; commits are local on `development3.8`.

## 1. Headline

| Measured on this checkpoint and prompt | Reference (chunk-major) | Opt-in path | Ratio | Exact? |
|---|---|---|---|---|
| 8,139-token cold prefill, tok/s (median of 3 reversed pairs) | 48.3 | **89.5** (95.2 / 89.5 / 87.6) | 1.853x | yes, 3/3 pairs |
| same, physical prefill reads | 277-279 GB | 89 GB | 0.32x | |
| same, MLX peak | 8.16 GB | 9.74 GB | +1.6 GB | |
| 16,139-token cold prefill, tok/s (3 pairs) | 44.7 | **69.2** (67.4 / 69.4 / 69.2) | 1.547x | yes, 3/3 pairs |
| same, physical prefill reads | 538 GB | 177 GB | 0.33x | |
| same, MLX peak | 10.35 GB | 11.92 GB | +1.6 GB | |
| Decode tok/s, 2.7K context (3 pairs) | 2.77 | **3.60** (`pool-buffered-fill`) | 1.30x | yes |
| Decode tok/s, 8K context (3 pairs, ratios 1.36 / 0.97 / 1.14) | 2.50 | 2.84 (`pool-buffered-fill`) | 1.13x | yes |
| Decode tok/s over three consecutive turns (2 pairs; see below) | 2.13-2.71 | 2.56-3.50 (`pool-buffered-fill`) | 1.19-1.33x per turn | text identical |

Targets were 80-100 tok/s cold prefill at 8K-16K and 5 tok/s decode. **8K prefill meets the target (89.5); 16K does not (69.2);
decode does not (3.6 at 2.7K, 2.8 at 8K).** These are one prompt on one machine; the second window at 16K is slower because
attention covers 8K+ context and the machine is thermally throttled (OS thermal pressure 2 for most of each arm).

Sustained decode (`bench_sustained_decode.py`, `results/flashnext/20261110-040253-sustained-decode-manual/`, 2 reversed pairs, one process per arm, three consecutive turns of 82 / 64 / 96 generated
tokens on a 2.5K-token coding prompt, swap guards on, swap flat at 1.12 GiB, identical generated text in all four arms; server-side decode tok/s): default 2.64 / 2.28 / 2.32 and 2.71 / 2.16 / 2.13;
`pool-buffered-fill` 3.50 / 2.80 / 2.94 and 3.27 / 2.56 / 2.64. Per-turn ratios 1.33 / 1.23 / 1.27 and 1.21 / 1.19 / 1.24, so the buffered fill is a steady **+19% to +33% (median about +24%)**
across the session with no further warm-up gain, but **sustained decode is 2.6-3.5 tok/s, not 5**; the 3.60 of the 32-token A/B is the early, shortest-context part of this curve (speed falls slightly as context grows).
(The client-side per-turn rates the script also prints count streamed text chunks, not tokens, and read low; use the server-side figures above.)

Cached follow-up (single observation, not a benchmark): a streamed tool-result turn in the same process matched 2,424 of
2,452 prompt tokens and prefilled 28 (5.8 s to first token). A restore in a **new process** from the SSD tier restored 2,107
tokens and prefilled 632 (`bench_restore_check.py`).

## 2. What was built

1. **Layer-major prefill** (`models/flashnext/layer_major.py`, `prefill.py`): for a window of up to 8,192 tokens each layer's experts
   are read once and used by every chunk of the window (reference order re-reads them per chunk). Chunk shapes, masks, position ids
   and per-layer cache order are unchanged, so results are bit-identical (unit tests on a small model; real-checkpoint gate).
2. **Two-half scratch ring overlap** (`--enable layer-prefetch`, `prefill_ring.py`): the next layer's experts are read with direct I/O
   into the ring's second half while the current layer computes (+1.5 GB while prefilling). First measurement was a guard bug that
   overlapped only alternate layers (1.277x); the corrected guard gave 1.465x at 8K.
3. **Trailing tiny chunk** (`layer_major.peel_tail`, rollback `FLASHNEXT_PREFILL_TAIL_CHUNK=0`): a <=8-token tail runs chunk-major after the
   window; it had triggered the decode-pool path mid-window (+1.9 GB MLX, +0.4 s). 1.214x without overlap, peak 7.48 GB.
4. **Inner snapshots** (`prefix_cache.freeze_cache`, `prefix_session.snapshot(position, frozen)`, rollback
   `FLASHNEXT_PREFILL_INNER_SNAPSHOTS=0`): a boundary on a chunk edge inside a window (the system prompt) is snapshotted from per-layer
   frozen caches instead of ending the window there. This removed the second 75 GB expert pass at 8K: 70.75 -> 89.5 tok/s.
5. **Buffered pool fills** (`--enable pool-buffered-fill`, `FLASHNEXT_EXPERT_POOL_BUFFERED_FILL=1`): same bytes, page cache allowed for pool fills: +30% decode at 2.7K,
   +13% at 8K, reads -36% / -29%.
6. **Measurement tooling**: `bench_layer_gate.py` (paired, exactness-gated; 16K prompt shape added), `bench_prefill_profile.py` (per layer/chunk),
   `bench_decode_profile.py` + `decode_profile.py` (decode wall time split), `bench_sustained_decode.py`, `bench_restore_check.py`, `bench_integration_smoke.py`, OS thermal-pressure sampling.

## 3. Bottleneck breakdown

* **Decode, pool 2 GB, 2.7K context:** 385 ms/token = SSD wait 290 ms (75%), GPU drain 72 ms (19%), host Python about 6 ms (2%); 0.73 GB read per token
  at about 2.5 GB/s. The earlier "0.4 s non-read" figure was a pool-0 artefact. Policies (LRU, LFU, per-layer) are within a few points of each
  other on a replayed 83-token route trace; capacity is the lever (about +9 hit points per extra GB).
* **8K prefill, layer-major + overlap + inner snapshots:** one 8,132-token window of about 84 s plus a 1.6 s tail. Reads are almost entirely hidden
  (expert wait about 1 s per window); what is left is GPU compute, which slows with depth as OS thermal pressure reaches 2.
  Layer 1 (PLE n-gram rows) costs about 8 s per prefill (random row reads, about 0.5 s fixed per call) and is not yet overlapped.
* **16K:** window 1 86 s, window 2 148 s (reads 88 GB per window, attention over 8K+ context, throttled).

### Decode cost model (ESTIMATE from measured inputs; not a measurement)

Inputs, all measured above: GPU drain about 72 ms and host about 6 ms per token; physical reads 0.73 GB/token at pool 2 GB (about 0.47 GB/token with `pool-buffered-fill`, -36%); the drive delivers about 2.5 GB/s
to decode while it is blocked and about 3.0 GB/s in the best prefill windows. With no overlap, token time = bytes / bandwidth + 78 ms. At 0.47 GB/token and the **best observed 3.0 GB/s** that is 157 + 78 = 235 ms = **4.3 tok/s at 2.7K context**
(measured 3.6; the gap is the 2.5 GB/s decode bandwidth). **5 tok/s needs a 200 ms token, i.e. about 0.37 GB/token at 3.0 GB/s: roughly half of today's 0.73 GB/token**, which the route replay prices at about +5 GB of effective
expert cache (about +9 hit points per GB), not available under the 3 GiB swap rule on this 16 GB machine. Overlap of reads with the 72 ms GPU drain could recover at most that 72 ms (about +1 tok/s at the same bytes) but earlier
look-ahead prefetch lost, and that remains untested for a new predictor. 8-10 tok/s would need about 0.15-0.2 GB/token or a faster drive. So the remaining decode gap is a physical capacity and bandwidth limit of this machine, not a
software defect that is still on the table, unless a read-saving idea (e.g. exact speculative verification that shares expert reads across several tokens, which measured slower when last tried) is shown to pay off.

## 4. Tests

* flashnext unit discovery: **1,754 tests, OK (3 skipped)** on the final tree; macqwen: **686 tests, OK** (run before the last two small flashnext-only edits, which do not touch macqwen).
* New offline tests: layer-major equivalence incl. inner snapshots (stored state equals chunk-major byte for byte, survives in-place KV updates,
  restore resumes exactly), staging/overlap (9), peel-tail, flag validation, restore-check validator.
* Real-checkpoint, exact: 8K pairs x3, 16K pairs x3, 2.7K decode A/B, 8K decode A/B, restore from an inner snapshot in a new process (SSD_HIT).

## 5. What failed, stopped or is unproven (read this before enabling anything)

* **Memory margin is thin.** 8K: swap 1.2-1.7 GiB, pressure 1 (one sample at 2), minimum available 1.2-3.3 GB. 16K: swap peaked **2.78 GiB** (limit 3), pressure 2
  in every arm, minimum available 1.4-2.0 GB (chunk-major arms reached 2.75 GiB too). Do not lengthen or repeat 16K runs without clearing swap first.
* **Integration smoke stopped on memory safety (`results/flashnext/20261110-034221-integration-smoke-manual/`).** Through the normal server with
  `--enable layer-major --enable layer-prefetch` and `--no-auto-purge`: the tool-call turn (valid `read_file` call) and the streamed tool-result follow-up passed;
  the next 8K cold prefill in the same process drove swap from 1.45 to 3.33 GiB in about 10 s and was stopped by the server guard and the harness.
  **That exceeded the owner's 3 GiB ceiling by about 250 MiB for a few seconds** (poll lag against a roughly 340 MiB/s spike). The cancel-mid-prefill,
  Anthropic and Responses steps were therefore not exercised. Cause unproven (hypothesis: file cache left by buffered decode plus the layer-major working set; the
  readiness model uses chunk-major peak figures and treats file cache as reclaimable). Not retried. Until a layer-major-aware pre-prefill purge/admission exists and
  is tested, do not use layer-major in a long-lived multi-request server.
* **Integration smoke under a new premise passed (`results/flashnext/20261110-035709-integration-smoke-manual/`).** Same server flags, but production auto-purge **on**, the absolute swap guard
  on (`FLASHNEXT_SWAP_CEILING_MIB=3072`, margin 768 MiB, confirmed live through `/health` before any request) and a smaller cancel prompt (4,638 tokens instead of 8,139). Results: tool-call turn valid;
  streamed tool-result follow-up matched 2,424 of 2,452 tokens, 40 SSE events, `[DONE]`, coherent text; **client disconnect 15 s into a 4.6K layer-major prefill: `request_cancelled` (no memory stop),
  server ready again 2 s later, next request answered**; `/v1/messages` and `/v1/responses` both 200 with correct shapes. Swap stayed at 1,262-1,270 MiB (flat), pressure level 1, minimum available
  3.9 GB. This does **not** explain or retire the earlier stop: the premise differs (purge on, guard on, 4.6K instead of 8K), a pre-request purge was not seen in the flight recorder (memory was ample), and the
  8K-after-decode scenario that breached the ceiling has not been re-run. Layer-major remains opt-in. Observation: a 471-token cold prompt took 25 s (about 19 tok/s) in this layer-major server. This is probably **not specific to layer-major**: with 10 of 512 experts routed per token, the union over 471 tokens
covers most of every layer's experts (about 99% if routing were uniform; real routing is skewed, so less), so chunk-major also reads close to a full 75 GB pass. Chunk-major was not measured on this prompt, so the
comparison is open; the hybrid mode (`layer-major-hybrid-4k`, chunk-major below 2,048 tokens) is unmeasured.
* An earlier version of that smoke was invalid (a one-request state probe raised on the second request); fixed in the harness.
* Not validated: real Qwen Code or DeepSeek Harness runs with the opt-ins, cancel of an **8K** layer-major prefill (4.6K was tested), restart-and-resume with
  the opt-ins in a multi-turn session, reasoning-effort variants, 262K (only the architectural limit exists; no 262K conversation was run), the combination of
  `layer-major` with `pool-buffered-fill`, prompts other than the repository-excerpt coding prompt, battery operation (all numbers above are on AC power).
* 4K windows: two of three pairs valid (+2.8% / +12.5%); the third stopped because I created a source file mid-run (gate fails closed). Hybrid small-window mode was never measured.
* Previously rejected or unchanged: pool eviction policies (no gain over LRU), one-layer routing look-ahead (earlier negative result), native MTP (earlier 33-42% slower, 55.6% acceptance;
  not re-measured), a 1 GB LRU pool (thrashes). A pool above 2 GB is not supported by `serve.sh` / `choose_pool_gb` and was not tried.

## 6. Launch commands

Reference (unchanged default; supervised, swap/thermal/battery stops, auto-purge):

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2            # http://127.0.0.1:8080/v1
./serve.sh --overnight --pool-gb 2                        # restarts after a crash or guard stop (bounded, logged)
```

Optimized prefill (opt-in; the exact configuration of the 8K and 16K gates, minus their private ports/cache; see the caveats in section 5):

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable layer-major --enable layer-prefetch
```

Optimized decode (opt-in, measured on the chunk-major reference path):

```bash
VEYSTRIX_API_KEY=local ./serve.sh --pool-gb 2 --enable uncached-prefill-reads --enable pool-buffered-fill
```

Rollbacks: drop the `--enable` flags; `FLASHNEXT_PREFILL_INNER_SNAPSHOTS=0`, `FLASHNEXT_PREFILL_TAIL_CHUNK=0`, `FLASHNEXT_EXPERT_POOL_BUFFERED_FILL=0`. Layer-major snapshots carry a
distinct cache identity and never mix with chunk-major ones; no existing cache was deleted.

Measurement commands (supervised, three reversed pairs, exactness enforced):

```bash
.venv/bin/python -m models.flashnext.tests.bench.bench_layer_gate --source-tokens 6000 --candidate layer-major --prefetch --pool-gb 2 --reads prefill --power-rules ac     # 8K
.venv/bin/python -m models.flashnext.tests.bench.bench_layer_gate --source-tokens 14000 --candidate layer-major --prefetch --pool-gb 2 --reads prefill --power-rules ac    # 16K
.venv/bin/python -m models.flashnext.tests.bench.bench_layer_gate --decode-ab --pool-gb 2 --candidate-pool-gb 2 --candidate-enable pool-buffered-fill --reads prefill
.venv/bin/python -m models.flashnext.tests.bench.bench_restore_check
```

Qwen Code: provider `modelProviders.openai[]` in `~/.qwen/settings.json` with `baseUrl` `http://127.0.0.1:8080/v1`, `envKey` `VEYSTRIX_API_KEY`
(`docs/flashnext/qwen-code-integration-2026-11-09.md`). DeepSeek Harness:
`DSH_HOME=<profile> VEYSTRIX_API_KEY=local dsh headless --patch docs/flashnext/dsh/veystrix-agentic.patch.yml "task"`
(other patches in `docs/flashnext/dsh/`). Neither client was run against the opt-in paths.

## 7. Ranked remaining work

0. Done offline after the smoke stop (unit-tested, **not yet validated live**): (a) `FLASHNEXT_SWAP_CEILING_MIB=3072` (+ `FLASHNEXT_SWAP_CEILING_MARGIN_MIB`, default 256) adds an
   absolute swap guard to the server's memory controller on top of the shipped baseline-relative 2.5 GiB rule, which with 1.5 GiB already in swap allows about 4 GiB in total. It is **off
   by default** (it would refuse to run when swap is already high for other reasons) and cannot bound a fast spike (the smoke saw about 340-420 MiB/s while a stop takes seconds), so
   it narrows the overshoot rather than preventing it. (b) The pre-request purge requirement (`purge_recovery.request_required_bytes`) adds 1.6 GB (the measured layer-major peak
   delta at 8K and 16K) when `FLASHNEXT_PREFILL_ORDER=layer` and the prefill exceeds 1,024 tokens; production's auto-purge was off in the smoke, so this has not been exercised.
1. Layer-major-aware admission (remaining): a purge (the passwordless `sudo -n /usr/sbin/purge` rule exists) before a large cold prefill, layer-major peak (9.7 GB, +1.5 GB with overlap) in the
   readiness figures, and a proactive margin so a fast swap spike cannot overshoot 3 GiB; then re-run the smoke (cancel, Anthropic, Responses) and a 3-turn session.
2. Decode capacity: a pool above 2 GB (needs an owner decision on memory) or a native read path; this is the only lever the profile supports (75% SSD wait).
3. Overlap the layer-1 n-gram row reads (about 8 s of an 8K prefill) with the previous window's compute.
4. 16K: a window-2 specific plan (it is compute- and thermally limited, not read-limited), thermal-aware chunking.
5. Real-client validation (Qwen Code, DeepSeek Harness) of the opt-ins; reasoning-effort and tool-call continuity across a restart.
6. Re-measure native MTP and n-gram drafting at depth 2-4 with the current checkpoint before drawing a conclusion either way.
