# Architecture overhaul: layer-major prefill, decode data plane, speculative verification (2026-11-09)

Status: **layer-major prefill is implemented behind an opt-in flag and verified bit-identical on a small model, checkpoint-free.
No real-checkpoint number in this document is new.** Every predicted rate below is a model calibrated on earlier owner-run
measurements, not a measurement of the new code. The owner has authorized the controlled plan in section 9.

## 1. Execution graph today (what a request does)

```
HTTP (macqwen/server.py) -> template render -> _adopt (live state | SSD snapshot | rebuild) -> backend.generate
  prefix_session.prefill: lookup snapshot -> plan chunks (balanced, 2048, anchors) -> prefill_target
    for each chunk (CHUNK-MAJOR):                                   models/flashnext/prefill.py _run_chunks
      language(ids_chunk, cache) -> Qwen4ExpModel: embed -> tile x4 hyper-connection streams ->
        for layer in 48 layers:  [PLE] -> attention (36 GatedDeltaNet, 12 sparse-attention) -> MoE
          MoE: router (GPU) -> mx.eval(inds) + tolist()  <- host sync, GPU idle
               read the chunk's routed experts (~400/512) from SSD into the ring half (F_NOCACHE preadv)  <- GPU idle
               gather_qmm over those rows -> shared expert -> combine
      mx.eval(cache states); on_boundary -> snapshot to RAM/SSD
  decode: one token = 48 layers x (router sync + 10 experts' reads + custom Metal executor)
```

Planes (existing, kept): compute = MLX/Metal ops and the custom executor; data = `store.py`, `expert_pool.py`,
`prefill_ring.py`/`pool_ring.py` (hazard-tracked leases), physical-I/O counters; state = GatedDeltaNet/KV caches and
`prefix_cache.py` snapshots with plan provenance; control = scheduler, `memory_controller`, watchdog, cancellation.
Layer-major changes only the order the compute plane walks (layers x chunks) and what the data plane hands it (the whole
layer once per window). The state and control planes keep their contracts.

## 2. The model facts that decide the design

48 layers (12 sparse attention, 36 GatedDeltaNet), hidden 2,560 x 4 hyper-connection streams, 512 routed experts, top-10,
moe_intermediate 640, 3,072,000-byte expert record (gate/up/down weight+scales+biases). One full pass over every expert is
**75.5 GB**. Per token the model touches 480 expert records (1.47 GB) in decode.

## 3. Measured baselines used (from `results/flashnext/*`, owner-run)

| Fact | Value | Source |
| --- | --- | --- |
| 1,025-token single call | 25.3 s, 50.3 GB staged, 16.6 s read waits (3.03 GB/s), 8.7 s other | `20261107-134606-prefill-memory-manual` |
| experts touched per layer | 341 of 512 at 1K tokens; about 400 at 2K (60 GB per chunk) | same; `long-context-engine.md` |
| 2K cold prefill | 61 tok/s, peak MLX 5.8 GB | `20261108-002253-prefill-memory-manual` |
| 4K / 8K / 16K | 54.2 / 47.0 / 48.1 tok/s (35.8 under memory pressure) | your brief, supervised runs |
| decode, pool off | 3.04 tok/s, about 0.8 GB physical per token | brief; replay below |

Read-amplification, stated plainly: **a prompt of N tokens in 2,048-token chunks reads about N/2048 x 60 GB**. A 16K prompt
reads about 470 GB to use a 75 GB model. The chunk loop is not compute-limited: at 3 GB/s the 60 GB per chunk is 19.6 s of a
36 s chunk. The 125 tok/s compute rate in the model below is what the 1K arm leaves after read waits (1,025 / 8.7 s = 118)
and the 2K arm after its waits (about 127).

## 4. Layer-major correctness: checkpoint-free evidence, real model pending

`Qwen4ExpModel.__call__` is a plain loop: layer `l`'s output for a chunk depends only on layer `l-1`'s output for the same
chunk and on layer `l`'s own cache after the previous chunk. GatedDeltaNet recurrent state and conv state, the PLE n-gram
context, and the sparse-attention KV all live in per-layer caches, so the per-layer chunk order (c1, c2, ...) that causality
needs is preserved when the outer loop is the layer. Masks come from the first attention/SSM cache offset at chunk start;
position ids from the chunk's absolute offset. Nothing crosses layers except the hidden state (20 KB per token).

Checked, not argued: `test_layer_major.py` runs a small Qwen4-Exp model (PLE linear layers, sparse attention, MoE) both ways
and compares every cache array and the hidden states **byte for byte**: equal chunks, uneven chunks, resume from a restored
prefix, single-chunk windows, two windows in sequence, the `prefill_target` path with its token. A negative control (shifted
ids, different chunk shapes) fails the comparison, so the test is sensitive. The one operand difference is that the window gives
`gather_qmm` all 512 experts with the router's indices where chunk-major gives the chunk's experts with remapped indices; the
test runs real `StreamingSwitchGLU` layers at the real row sizes (hidden 2560, intermediate 640, Q4/G32) and the outputs are
bit-identical across chunks that touch different expert subsets. **Open:** the real checkpoint must still pass the exact
token-digest gate; until then layer-major snapshots carry `layer-major-v1/w<window>` in the cache fingerprint and never mix
with chunk-major ones.

## 5. Cost model

`models/flashnext/layer_major.py` (`chunk_major_bytes`, `layer_major_bytes`, `activation_bytes`, `predicted_seconds`),
SSD 3.0 GB/s, compute 125 tok/s, 400 experts read per chunk-layer for chunk-major, all 512 for layer-major:

| prompt | window | chunk-major bytes | layer-major bytes | traffic ratio | chunk-major tok/s | layer-major serial | layer-major overlapped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 4,096 | 118 GB | 75 GB | 1.6x | 56.8 | 70.7 | 125 |
| 8,192 | 8,192 | 236 GB | 75 GB | 3.1x | 56.8 | 90.3 | 125 |
| 16,384 | 8,192 | 472 GB | 151 GB | 3.1x | 56.8 | 90.3 | 125 |
| 16,384 | 16,384 | 472 GB | 75 GB | 6.2x | 56.8 | 104.9 | 125 |

Activations: 168 MB per 8,192-token window, 671 MB per 32K. The model reproduces chunk-major's 54-61 tok/s because it was
calibrated on those arms; the layer-major columns are extrapolation. Attention cost grows with context (the 47-48 tok/s at
8K-16K), so expect the real compute rate to fall below 125 at long context. "Serial" is what this implementation does (one ring
half: read the layer, compute it, read the next). "Overlapped" needs a second half and an asynchronous read of layer
`l+1` during layer `l`, which the all-experts read makes simple (no router dependency); it is **not built**, see section 8.

Bytes that layer-major does *not* save: attention weights, shared experts, embeddings stay resident. It reads about 20%
more than the chunk's routed set at small windows (all 512 instead of about 400); at 8K the routed union is close to 512.

## 6. What was implemented

* `models/flashnext/layer_major.py`: window planner, `ExpertWindow` (one layer's complete expert set per window, ring-staged
  with a copy-path fallback), the layer-major driver `run_window`, the cost model.
* `expert_cache._LAYER_WINDOW` hook in `StreamingSwitchGLU.__call__`: with a window active, a prompt-sized call uses the
  resident full layer and the router's indices directly. **No router sync, no `tolist()`, no per-chunk read**; the progress
  observer still runs at each MoE call, so cancellation and the watchdog tick are unchanged.
* `prefill._run_layer_major`, branch in `_run_chunks` (`FLASHNEXT_PREFILL_ORDER=layer`), `_prefill_target(forward=...)` so the
  final token comes from the identical head code, `PlanTrace.order/windows/layers`.
* Windows end at the system-prompt boundary (`required`), prefer a message anchor in their second half, and are otherwise
  whole chunks up to `FLASHNEXT_PREFILL_WINDOW_TOKENS` (default 8,192, range 1,024-32,768).
* `--enable layer-major` (chat.sh and `serve.sh --overnight --enable layer-major`), settings registered, mode default `chunk`.
* `--enable layer-major-hybrid-4k` is a separate, unmeasured experiment. Windows of at most 2,048 tokens execute their
  original chunks through the reference path instead of loading every expert in the layer; larger windows use layer-major.
  The plan fingerprint carries `hybrid-small-v1`, so snapshots do not mix with either other order. Checkpoint-free tests compare
  token and complete cache state under matched chunk boundaries and check snapshots on cancellation. The 2,048-token cutoff is
  a heuristic, not evidence of a physical-I/O or latency improvement. It has not passed the real-checkpoint gate.
* Memory-controller checkpoints now run after every layer (finer than per chunk).
* On a layer-major exception, the staged expert ring is fenced before its lease is retired. A failed fence leaves the operands
  retained for the enclosing ring session's failure handling instead of marking a potentially active buffer reusable.
  Checkpoint-free tests cover the ordering and the failed-fence path.

Rollback: omit `--enable layer-major` (default) or set `FLASHNEXT_PREFILL_ORDER=chunk`; the chunk-major code path is unchanged.

Trade-offs, stated: (1) a snapshot can be taken only at a window end, so "boundaries: all" yields snapshots every window, not
every 2,048 tokens; (2) an interrupted prefill loses up to one window (about 90 s at 8,192) instead of one chunk; (3) the
adaptive chunk scheduler does not re-plan inside a window; (4) progress percentages from the layer counter are approximate in
the first window (cancellation and the watchdog are exact).

## 7. Decode: analysis (no code change justified yet)

Physical lower bound: 480 records x 3.07 MB = 1.47 GB logical per token. At 3 GB/s the bound for hit rate h is
`3.0 / (1.47 x (1 - h))` tok/s: h=0 -> 2.0; h=0.466 -> 3.8 (measured 3.04, 80% of it); h=0.636 -> 5.6; h=0.783 -> 9.4.
So **5 tok/s needs about 63% of expert reads served from RAM**, and 6-8 tok/s needs 68-78%: a larger or smarter resident
set, not faster kernels. Replay of the only recorded route trace (older checkpoint, 3.21 MB records, one session, not this
checkpoint; `bench_pool_policy_trace`):

| pool | slots | LRU | ARC | BIP8 | Belady (upper bound) |
| --- | --- | --- | --- | --- | --- |
| 1 GB | 311 | **0.0%** | 28.7% | 28.8% | 53.9% |
| 2 GB | 622 | 46.6% | 48.9% | 40.4% | 67.9% |
| 4 GB | 1,245 | 63.6% | 63.6% | 52.1% | 78.3% |

Reading: LRU thrashes completely at 1 GB (scan pollution) and an admission-aware policy (ARC) recovers 28.7%; at the
production 2 GB the gain is +2.3 points (about 0.03 GB per token, under 0.1 tok/s); 63% needs 4 GB of pool, which the
16 GB machine and the swap incident do not allow. Decision: leave LRU/2 GB as is. To go further, capture a **route trace on
the current checkpoint** (the existing `set_route_trace`, decode only) and replay it; that is part of the section 9 request.
Not changed: router synchronization and kernel dispatch (needs a profiled run, same rule as `AGENT_INVARIANTS.md`: no `mx.eval`
changes on the decode path without a controlled measurement).

An additional offline holdout replay trains frequency pins on turns 0–1 and evaluates turn 2 of that same older-checkpoint
trace, starting each simulated cache empty. At 311 / 622 / 1,245 slots, global LRU hit 0.0% / 48.0% / 64.7%, ARC hit
28.6% / 50.4% / 64.4%, and equal per-layer LRU hit 34.5% / 51.7% / 65.5%. Static frequency pins hit 11.0% / 13.5% /
17.0% before charging their cold-fill traffic. See `results/flashnext/20261109-190028-cache-holdout-manual/`.
These are simulated route hits, not measured physical misses or throughput; the source has different expert sizes and routing
than the target. The 4 GB point is a simulation, not authorization to allocate that pool on this machine. Per-layer partitioning
is a candidate for a current-checkpoint trace and safe, paired experiment, not a production change. At 311 slots some partitions
contain fewer slots than an event's expert set; the replay assumes uncached misses can execute via separate transient storage
and does not account for that storage or its I/O. Production feasibility and memory cost remain unproven.

Two more offline policies were added to the same holdout replay (`bench_cache_holdout.py`, 7 unit tests) and run on the same
older-checkpoint trace (`results/flashnext/20261109-193910-cache-holdout-manual/cache-holdout.json`). `LeastStale` is a
layer-cycle approximation of SpecMD's idea, not its system: decode visits layers in a fixed cycle, so the victim is the entry whose
layer is needed furthest ahead. `RecencyFrequency` is a lightweight decayed-count baseline for FlashMoE-style replacement,
without a learned predictor. Simulated hit rate on the held-out turn:

| Slots | Global LRU | ARC | Per-layer LRU | Least-Stale (approx.) | Recency-frequency | Static pins |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 311 | 0.0% | 28.6% | 34.5% | 32.7% | 0.0% | 11.0% |
| 622 | 48.0% | 50.4% | 51.7% | 51.0% | 49.0% | 13.5% |
| 1,245 | 64.7% | 64.4% | 65.5% | 65.2% | 64.8% | 17.0% |

Reading: neither new policy beats equal per-layer LRU, and Least-Stale only matches ARC. Recency-frequency thrashes at the 1 GB
size like global LRU (scan pollution). The best spread between any two policies at the production 2 GB size is 3.7 points
(global LRU to per-layer LRU), about 0.05 GB of demand traffic per token, or roughly 0.1 tok/s under the bound above. So the
offline evidence says replacement policy is a second-order decode lever on this trace and **no policy change is justified**;
the first-order lever remains pool size, which the 16 GB machine limits. These are simulated route hits on a different checkpoint
and one session, not physical misses; the Least-Stale result in particular does not test SpecMD's full system. Next decode
evidence needs a current-checkpoint route trace and measured physical reads, which are subject to the headroom premise in
section 9.

## 8. Speculative decoding (native MTP): stays disabled, with the arithmetic

Verifying k draft tokens reads the union of their experts per layer. With 10 experts per token and consecutive-token overlap
well under 100%, two positions read about 1.7x the bytes of one (union of two top-10 sets), three about 2.3x. A verification
step yields `1 + sum(P(first j drafts accepted))` target tokens on average, before output-limit effects. Under a simplified
I/O-only comparison, this expected count must exceed the expert-byte ratio; at one draft and a 1.7 ratio, acceptance would
need to exceed 70%. For two drafts and a 2.3 ratio, the sum of the one- and two-draft prefix-acceptance probabilities must
exceed 1.3; that is possible, and draft, cache and verification costs make the real threshold higher. The existing exact MTP path measured slower in earlier paired runs
(`research.md`), the draft layer adds its own 1.63 GB expert pool, and decode is read-bound. Therefore `FLASHNEXT_NATIVE_MTP=off`
stays. A real acceptance and union-size measurement would be needed to reopen it, and it is listed in section 9 as optional.

An idealized replay of the held-out older-checkpoint route trace groups 1–7 target positions into per-layer unions, assumes
perfect acceptance and free drafting, and simulates a cold global LRU. With the 622-slot pool, simulated demand is 643 MB per
output token for singleton steps, then 694 / 844 / 769 / 700 / 666 / 636 MB for 2–7-position blocks, respectively. With no
pool it falls from 1,237 to 636 MB over the same range. See
`results/flashnext/20261109-190616-cache-holdout-manual/cache-holdout.json`. This is a route-union cost model: sorted union
order changes replacement, it excludes draft, GPU verification and rollback costs, and it is neither physical I/O nor evidence
of actual acceptance on the target checkpoint. It shows why cache state can reverse a naive batch-size argument; it does not
justify enabling MTP or ruling out a different exact design.

## 9. Authorized real-checkpoint measurement plan (not yet run)

Goal: does layer-major prefill beat chunk-major on the real model, with an identical token digest?

1. **Digest gate, 2K-4K tokens, greedy:** chunk-major vs layer-major (window 4,096), fresh process each, 3 pairs, reversed
   order alternation. Pass = identical token digest and identical final-cache hash. Fail = stop, no performance runs.
2. **Prefill A/B, 8K cold prompt** (never 32K or 262K): chunk-major vs layer-major windows 4,096 and 8,192, 3 pairs each,
   reversed order. Record tok/s, per-layer seconds, `ring.wait_seconds`, staged GB, `ssd_prefill_bytes`, peak MLX, compressor
   and swap growth, power state. Stop on any guard stop; no retry.
3. **16K only if 8K passes and memory headroom allows:** one pair.
4. **Decode route trace** (optional, 128 tokens, pool 0 and 2 GB): feeds the policy replay in section 7.

Expected cost: about 25 to 40 minutes of model time in total for steps 1-2. Run plugged in, Low Power Mode off, swap baseline
recorded, `FLASHNEXT_CHAT_GUARD=1` (swap +2.5 GiB stop) active. Results go through `macqwen.results` per the measurement standard.

The existing `prefill-memory` benchmark is not this gate: it directly instantiates the backend instead of using the supervised
AGENTIC launcher, and its comparison tolerates an unavailable cache hash. A gate must capture and compare all attention and
recurrent state, position metadata, final-row logits and generated token IDs under the supervised server, and fail closed when any required
evidence is missing. The new `layer-major-gate` terminal case does this with separate prefix caches, greedy sampling, live
watchdog/guard checks and a 4K candidate window. It includes a tool schema in prompt context; real tool calls need a separate
protocol test because the server requires at least 64 output tokens for tool-enabled requests.

On 2026-11-09, two startup attempts failed before inference (the first request violated the tool-output minimum; the second
detected sampled preferences). After those setup issues were fixed, a 2,739-token reference prefill completed, but the external
monitor stopped the run when available memory fell below its 3,000 MiB floor around the end of decoding. The supervisor, server and
independent watchdog exited. The recorded reference prefill was 90.37 s, 137.25 GB process physical reads and 4.91 GiB MLX peak;
32 tokens were generated in 13.51 s. This single interrupted diagnostic is not a valid A/B, is not a cold-throughput promotion
result, and says nothing about candidate correctness or speed. The candidate and subsequent arms were not run. See
`results/flashnext/20261109-185459-layer-major-gate/`. The copied safety trace records minimum available memory of 2,693.5 MiB
in decode and peak compressor use of 6,363.6 MiB during prefill, versus 244.2 MiB at startup. It also records 842 swapout pages
over the arm; swap usage did not breach the +2.5 GiB stop. This supports a resource stop, not an inference about which process
caused the system-wide compression. Respect the stop; do not automatically retry it.
The gate now records each safety sample and stops more conservatively at 3 GiB available, or if compressor growth exceeds
2 GiB while available memory is under 4 GiB. The state probe also avoids a redundant full-size host byte copy during hashing.
Final-row logit capture was added after this interrupted reference, so the old artifact cannot supply a complete pair for the
strengthened gate.

Measurement correction: `ExpertWindow` is reused across windows. Its counters are lifetime totals, so each `WindowReport` now
subtracts the counters at entry; otherwise a multi-window trace would count earlier expert reads repeatedly. These are staged
byte counts, not proof of physical SSD bytes; physical reads must be measured independently.

### New safe measurement premise after the stop (2026-11-09, offline analysis; nothing re-run)

What the stopped trace shows (`reference-safety-trace.jsonl`, 360 samples; system-wide numbers, not per-process attribution):
available memory 8,540.2 MiB before launch, about 6,209 MiB once MLX held 2.98 GB of weights, about 4,000–4,700 MiB through
prefill, and a minimum of 2,693.5 MiB in decode (MLX peak 5.27 GB, RSS 1.9 GB). The compressor went from 244 to about 3,785 MiB in
the first two seconds of loading, i.e. macOS compressed other processes' pages to make room. Pressure level stayed 1 (normal)
and swap fell from 1,349.8 to 972.4 MiB, so the OS itself never signalled trouble. The stop came from the gate's own 3 GiB floor:
**one arm needs about 5.85 GiB of headroom (8,540.2 − 2,693.5 MiB), and this machine idles at 8.3–8.5 GiB available.** The
floor was crossed by arithmetic, so the same start condition will cross it again regardless of which arm runs.

New premise, enforced in `bench_layer_gate.py` (`headroom_reason`) before any process, tokenizer or result directory exists:

* start only if available memory ≥ floor 3,072 + observed footprint 7,300 (raised from 6,000 after the 7,125 MiB drop of the owner-run gate; first measured 5,847) + margin 512 =
  **10,884 MiB for the reference arm and 11,908 MiB for the layer-major arm** (originally 9,584 and 10,608). The candidate adds 1,024 MiB because it holds a
  1,500 MiB layer operand instead of about 1,170 MiB plus window activations; that allowance is an estimate, not a
  measurement, to be replaced by one once a candidate arm has run. A postponed gate exits with code 2 and records nothing that
  could be mistaken for a run.
* the in-flight floors (3 GiB available, 2 GiB compressor growth below 4 GiB available, +2.5 GiB swap, AC, Low Power Mode,
  thermal) are unchanged. The headroom budget is an addition; the footprint may only be raised by later evidence, never lowered
  to admit a run.
* **Today it refuses**: this Mac read 8,292 MiB available, 1,292 MiB below the reference requirement and 2,316 MiB below the
  candidate requirement. Nothing was launched. The user must free that memory (quit memory-heavy applications, or restart
  the Mac and launch nothing else) before the gate can run; the agent does not stop unrelated processes. This is a machine
  condition, not a statement about correctness or speed.
* the gate's old reference probe still cannot form a pair (no final-row logits), so a full 3-pair gate is still required; no
  result from the stopped arm is reused.

Purge (owner instruction, 2026-11-09: purge automatically whenever RAM is needed). Two parts. **Server:** already built in, no
new code: `FLASHNEXT_AUTO_PURGE` defaults to 1 for flashnext in `macqwen/cli.py` (so `serve.sh` has it), and
`purge_recovery.py` purges `sudo -n /usr/sbin/purge` at startup, before requests and from an idle monitor, only when the engine
is idle, memory is short, enough file cache exists and pressure allows (`docs/flashnext/auto-purge.md`, whose "off by default"
header is stale). The passwordless rule is installed (`/etc/sudoers.d/veystrix-purge`; `sudo -n -l /usr/sbin/purge` lists it).
**Gate:** it had been launched with `--no-auto-purge` and no purge step, so a short start was simply refused. It now calls
`prepare_headroom()`: if the budget is short and the machine is otherwise safe (AC, thermal, pressure level 1), it runs one
purge before any server exists, waits 3 s, re-samples, and applies the same budget again; a purge that frees too little, a
failed purge, or an unsafe machine still postpones the gate. Arms keep `--no-auto-purge` so a purge never lands inside a run;
the purge is recorded in each arm's evidence. Measured: one purge raised available memory from 9,239 to 12,050 MiB (+2.8 GiB)
with pressure level 1 and swap unchanged (compressor unchanged at 411 MiB). That would satisfy the 10,608 MiB candidate
requirement, but the memory after a purge is file cache that refills during the run, and the stopped run's 5.85 GiB drop was
measured without one, so the premise remains a conservative estimate until a real arm is measured. A purge also changes
the OS file-cache state; the gate records it, and its result is still an exactness diagnostic, not a cold-throughput number.

**Second stop (owner-run gate, 2026-11-09 14:20 UTC, `results/flashnext/20261109-194853-layer-gate-manual/`).** The owner ran the
gate with the headroom premise in place. The first reference arm (`r1-chunk`) started at 11,426 MiB available (no purge was
needed), compressor 381 MiB, swap 932.4 MiB, pressure level 1, AC, Low Power Mode off. During the request the server held
available memory at about 4,300–5,800 MiB (a 7 GiB drop from launch, larger than the 5.85 GiB the premise assumed) and the
compressor rose to 4,285–5,174 MiB. On the last sample macOS reported **pressure level 2** and the gate stopped with
"memory pressure or swap telemetry is unsafe or unreadable" (the rule requires level 1). Swap did not grow (932.4 MiB throughout),
available memory never approached the 3 GiB floor, no candidate arm ran, the supervisor, server and watchdog exited, and no
pair, state hash or token comparison exists. It is a resource stop, not a correctness or performance result, and it was not retried.

What it changes. The binding constraint is not the available-memory figure the headroom budget checks; it is macOS reaching
pressure level 2 while the compressor holds about 5 GiB, with the server running a 2.7K-token prefill. Two independent runs
(start 8.5 GiB and start 11.4 GiB) therefore ended in a memory stop, so more headroom at launch did not remove the stop; it moved
it. The memory controller treats level 2 as survivable in product use, but the gate deliberately stops at it. That rule is a
safeguard and has not been relaxed. The headroom premise above is therefore **not sufficient** and its 6,000 MiB footprint
was raised to 7,300 MiB on this evidence (required start 10,884 / 11,908 MiB). Even so the budget cannot predict level 2:
that arm stopped with 4.5 GiB still available, so the premise remains necessary, not sufficient.

Decision needed from the owner before any further live gate: either (a) free more of the machine first (quit applications so far
fewer pages need compressing, or run after a fresh restart with only the terminal open) and expect the gate to stop again if level 2
recurs, or (b) explicitly accept level 2 (warn) as non-stopping for this diagnostic while keeping the +2.5 GiB swap stop, the 3 GiB
available floor and the compressor-growth rule, which is a weakening that only the owner can authorize. Neither is assumed here.

**Third stop (agent-run under the owner's explicit instruction to proceed, `results/flashnext/20261109-195541-layer-gate-manual/`).**
Start 12,036 MiB available (no purge needed), compressor 463 MiB, pressure 1. The same first reference (chunk-major) arm stopped
with "compressor grew over 2 GiB while available memory fell below 4 GiB": compressor 463 → 5,662 MiB (+5.2 GiB) and available
3,715 MiB on the last sample, pressure level 1 and swap flat (924.4 MiB) throughout; file-backed memory rose from 0.7 GB at
launch to about 5 GB during the request. The Mac was otherwise idle (no process above 550 MB; `top` lists nothing worth closing), so
the pressure is the server's own: three runs, three memory stops, all in the reference arm before layer-major ran.

Working hypothesis (not yet proven): with `--pool-gb 0` the expert reads are buffered `pread`s, so about 137 GB streamed through
the page cache fills it to ~5 GB and the kernel compresses the server's own pages to make room. The project's own earlier notes
agree that buffered reads grow the file cache by the bytes read and that `F_NOCACHE` removed the compression churn
(`memory_manager.py`, `prefill.py::_uncache_reads_when_elevated`). New opt-in mode **`--enable uncached-reads`**
(`FLASHNEXT_UNCACHED_EXPERT_READS=1`, `store.py`): every expert shard descriptor is opened `F_NOCACHE` for the whole run, and the
memory manager's end-of-request `set_uncached_reads(False)` cannot turn it off. Same bytes in the same order, so results are
identical (`test_uncached_reads.py`: descriptor modes, restore cannot undo it, identical bytes, off by default in every mode,
recorded as execution-only, not in the cache identity). The gate now runs **both arms** with it (`--buffered-reads` restores the old
configuration), so the comparison stays symmetric. Its speed effect is unmeasured and no speed claim is made; whether it removes
the compressor growth is exactly what the next gate attempt tests.

**Fourth attempt: the exactness gate PASSED (`results/flashnext/20261109-200252-layer-gate-manual/`).** With
`--enable uncached-reads` in both arms (this is not the production default of buffered reads), three fresh reversed pairs
(r1 chunk→layer, r2 layer→chunk, r3 chunk→layer) ran the supervised AGENTIC server on the real checkpoint with a 2,739-token
coding prompt, greedy sampling, pool 0, isolated caches, the watchdog live, the +2.5 GiB swap stop confirmed and every safety
sample recorded. In every pair the gate's fail-closed comparison passed, and I re-read the raw arms: identical prompt hash, first
token (40), final-row logits hash, 32 generated token IDs, executed chunk edges (1,054 / 2,107 / 2,732), and all 36 recurrent and
12 attention cache-array hashes plus position state. The candidate (`layer-major-v1/w4096`) ran two layer-major windows
(0–2,107 and 2,107–2,739). Pressure stayed at level 1 in all six arms; swap never moved; compressor peak was 3.8–4.0 GiB in the
chunk-major arms and 0.6–0.7 GiB in the layer-major arms; minimum available was 4.4–4.8 GiB (chunk) and 5.2–5.7 GiB (layer).
This is exactness at one prompt length and one window setting only: it says nothing about other lengths, the 8K and 16K windows,
tool calls, or sampled decoding.

Raw arms, 2,739 tokens, pool 0, F_NOCACHE reads, a correctness-gate prompt (not a cold-throughput promotion result; the OS file
cache did not hold experts because reads bypass it, but probe overhead applies):

| Arm | prefill s (3 arms) | tok/s (median) | spread | physical GB | MLX peak GB |
| --- | --- | --- | --- | --- | --- |
| chunk-major | 93.80 / 92.49 / 92.65 | 29.56 | 1.4% | 142.4 | 5.27 |
| layer-major 4K window | 114.03 / 116.98 / 116.01 | 23.61 | 2.5% | 163.8 | 5.49 |

So at this size layer-major is **slower (0.80x) and reads 15% more**, as the cost model's "small windows read all 512 experts"
caveat predicts: the required system-prompt boundary cut the prompt into a 2,107-token window and a 632-token tail window, and
each window reads the whole 75.5 GB expert set. The measured candidate peak is +0.22 GB over the reference (the headroom budget
had assumed +1,024 MiB). The 2.7K chunk-major rate (29.6 tok/s) is well below the historical 54–61 tok/s runs, which had the
expert pool and ring and larger chunks; the two are different configurations and are not compared here.

**8K attempts and the owner's memory rule.** Two 8,139-token (production 2,048-chunk plan) reference arms stopped
(`20261109-202044-…`, `20261109-202253-…`) the same way: after two calm 1,054-token chunks (MLX 3.1 GB active, compressor
606 MiB, 7 GB available) the first full 2,048-token chunk made the compressor jump 606 → 5,822 MiB inside one 5 s sample,
file-backed memory fall 4.1 → 1.6 GB and macOS report pressure level 2, with swap unchanged and available above the 3 GiB
floor. The 2.7K gate never saw it because its chunks were about 1,054 tokens. Engine finding, independent of the gate: on this
16 GB Mac a production-size 2,048-token chunk drives macOS into heavy compression even from a clean 12 GB start. The owner then
set the rule for these runs: **"the only rule I want is that it doesn't reach 3 GB swap."** The gate's default is now
`--memory-rules swap-only`: swap in use must stay below 3 GiB (stop at 3,072 MiB) and the server's own growth stop is lowered to
`3 GiB − baseline − 50 MiB` (about 2.1 GiB at the current 0.9 GB baseline; never above 2.5); the 3 GiB available floor,
pressure-level-1 and compressor-growth stops are not applied in flight. Kept: AC power, Low Power Mode and thermal checks, the
independent watchdog, stop on the first exactness failure, and a stop at critical pressure (level 4, where macOS may kill
processes), which is a retained guard beyond the owner's wording. `--memory-rules strict` restores the original gate.

**8K paired comparison under the swap-only rule (partial: `results/flashnext/20261109-203753-layer-gate-manual/`).** An 8,139-token
prompt, production 2,048 chunk plan (edges 1,054 / 2,107 / 4,116 / 6,124 / 8,132), pool 0, F_NOCACHE reads in both arms. Pair 1
(chunk then layer) **passed the exact comparison at 8K** (identical prompt hash, first token 40, final-row logits, 32 token IDs,
all cache and position hashes; candidate `layer-major-v1/w4096` ran three layer windows 0–2,107 / 2,107–6,124 / 6,124–8,139).
Arm r2-layer also completed; arm r2-chunk then **stopped at its preflight because the Mac was on battery power** (AC is required
in both memory-rule sets; I did not change it), so there is one chunk-major arm, not three. Raw arms:

| Arm | prefill s | tok/s | physical GB | MLX peak GB | min avail MiB | max compressor MiB | max pressure | max swap MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 chunk-major | 208.4 | 39.05 | 279.5 | 6.33 | 3,543 | 5,323 | 2 | 2,525 |
| r1 layer-major 4K | 190.2 | 42.78 | 247.4 | 6.57 | 4,268 | 2,544 | 1 | 1,813 |
| r2 layer-major 4K | 193.1 | 42.15 | 247.4 | 6.57 | 4,539 | 2,460 | 1 | 1,781 |

This is **not** a completed A/B (one reference arm). It indicates layer-major about 8–10% faster at 8K, with 11.5% fewer bytes read,
far less than the 3x traffic cut the cost model predicts for large windows, because three windows each reread the full 75.5 GB
set (the required system-prompt boundary forces a 2,107-token first window). Per-window segment timings show the structure:
each window spends 31–34 s blocked on its 75.5 GB read (about 2.3 GB/s) before computing at roughly 90–100 tok/s, so
**the reads are serial with compute**. Hiding them is the lever. In the reference arm swap rose from 1,522 to 2,525 MiB (under the
3 GiB ceiling) and pressure reached level 2; the layer-major arms stayed at level 1.

**New opt-in: `--enable layer-prefetch` (`FLASHNEXT_PREFILL_LAYER_PREFETCH=1`, needs a layer-major mode).** At the start of layer
`l` the window queues layer `l+1`'s complete expert read on the existing worker threads (fresh destination arrays, so it cannot
overwrite memory the running layer uses), so the read overlaps layer `l`'s attention and MoE compute; layer `l`'s MoE call then
takes its already-read operands. Never more than two layers pending, fallback path only (ring or pool-reuse layers are
untouched), same bytes in the same order, drained on every failure path (`abort_layer`, end of window). It adds about 1.5 GB
resident while overlapped. `test_layer_major.py::LayerPrefetchTests` (real streamed layers): bit-identical output, prefetch used,
no-op when off / with a ring / duplicate, abort waits for queued reads, stale entries not consumed early. **Unmeasured on the real
checkpoint**; the gate takes `--prefetch` to run the candidate arm with it, with exactness enforced on every pair.

**8K paired comparison with `layer-prefetch` (complete: `results/flashnext/20261109-205829-layer-gate-manual/`).** Exactness passed
on all three reversed pairs (identical tokens, logits and cache/position hashes at 8,139 tokens, `layer-major-v1/w4096` plus
prefetch). Speed did not improve. Raw arms (prefill s, chunk-major then layer-major+prefetch): r1 232.8 / 257.3, r2 259.0 / 261.8,
r3 271.6 / 274.0; medians 31.42 vs 31.09 tok/s (ratio 0.989); spreads 15.0% and 6.4%; physical GB 273–280 vs 246–248.
The prefetch did hide the reads (blocked read seconds per window: 24.9 / 22.3 / 1.8 against 33.6 / 30.7 / 32.0 without it) but
total window time rose (the last window 97.7 s against 61.7 s), i.e. compute slowed by more than the reads saved. The memory
picture differs: layer arms with prefetch reached a 5.9–6.9 GB compressor and pressure level 2, against 2.5 GB and level 1 for the
earlier layer arms without it. The prefetch buffers (about 1.5 GB, numpy arrays outside MLX's peak counter) are the likely cause,
but this is not proven by an isolated test. The run is also confounded by machine drift: the chunk-major arms themselves slowed
233 → 259 → 272 s across the run as swap in use and compressor size grew, so the ratio above is a within-run comparison, not a
clean measurement of the prefetch penalty. **Result: prefetch as built does not help on this 16 GB machine; it stays opt-in and
is not promoted.** A candidate follow-up is to prefetch only while the memory controller reports a normal state, or to prefetch
half-layers to cap the extra footprint; neither is built.

**Swap ceiling stop (`results/flashnext/20261109-213137-layer-gate-manual/`).** A complete no-prefetch 8K A/B was started and the
owner's own rule stopped it, as designed: "swap in use reached 3256 MiB (owner ceiling 3072 MiB)" in the first chunk-major arm. Swap
in use was 1,896 MiB at launch (it was 908 MiB at the first run of the day, 1,522 and about 2,100 MiB later: other processes' pages
sit in swap and macOS does not release them between my runs) and the reference arm's 2,048-token chunks each added swap
(1,872 → 2,462 → 2,435 → 2,890 → 3,256 MiB) while the compressor surged to 7.7 GB. After the stop swap is 2,103 MiB, so about
970 MiB of room remains: not enough for an 8K reference arm. Not retried; the rule and the process list are the owner's. What would
change it: a restart or closing the applications whose pages are swapped out (I do not stop unrelated processes). The finding also
holds for the product: on this Mac one production-style 8K cold prefill costs on the order of 1 GiB of swap growth, which is
what the server's own growth stop is for. Because this run produced no completed arm, the no-prefetch 8K A/B remains one chunk-major
arm (208.4 s, 39.05 tok/s) against two layer-major arms (190.2 s and 193.1 s; 42.78 and 42.15 tok/s) from the partial run above,
and the pair order was not reversed three times: it is an indication (layer-major about 8–10% faster at 8K), not a validated
measurement.

**Decode measured on the current checkpoint (from the completed gate arms, pool 0, F_NOCACHE in all phases).** The probe's
`finished` event carries real decode timings. Across the 15 completed arms decode ran at **1.1–2.0 tok/s** (32-token horizon;
tail rate over the last 23 tokens 1.2–2.0), about 1.06 GB of physical reads per token (33 GB per 31 tokens), at both 2.7K and 8K
context, in both execution orders (2.7K chunk-major 1.76–1.83 tok/s, layer-major 1.47–1.49; 8K 1.12–1.49). That is below the
historical 3.04 tok/s pool-off figure and far below the 5 tok/s target. Reason found in the design, not yet isolated by a
paired test: `uncached-reads` applies F_NOCACHE to decode too, and decode re-reads the same experts, so it discards the
page-cache hits that buffered reads get. That motivated **`uncached-prefill-reads`** (`FLASHNEXT_UNCACHED_EXPERT_READS=prefill`):
F_NOCACHE only while a prefill runs (begin and end hooks around `prefill_target`, also on failure), buffered reads for decode.
Same bytes, so results are identical (`test_uncached_reads.py`). The gate has `--decode-ab` (reference = F_NOCACHE all phases,
candidate = prefill only, both chunk-major, exactness enforced on every pair) to measure the decode effect.

**Decode read-mode A/B (complete: `results/flashnext/20261109-214254-layer-gate-manual/`).** Three reversed pairs, 2,739-token
prompt, pool 0, both arms chunk-major, exactness enforced and passed on every pair (identical prompt hash, first token, final
logits, state and all 32 generated token IDs). Reference = F_NOCACHE in all phases; candidate = F_NOCACHE during prefill only
(`uncached-prefill-reads`). Decode, 32-token horizon (31 timed tokens), raw per-arm: reference 1.77 / 1.10 / 1.31 tok/s (median
1.31, spread 50.5%), candidate 1.75 / 1.80 / 1.66 (median 1.75, spread 8.0%); decode physical reads 1.074 GB/token (identical in
all three reference arms) against 0.52–0.57 GB/token. Pairwise ratios 0.99, 1.64, 1.27: the bytes halve in every pair but the speed
gain is uneven and absent in pair 1, so the median 1.33x carries a wide reference spread. **Scope of the claim:** the reference
was the all-phase uncached mode that I introduced for the gate, not the production default (buffered reads); the candidate's decode
reads are those of the production default. So this documents that all-phase F_NOCACHE throttled decode and that **production
decode on this checkpoint at pool 0 is about 1.75 tok/s and 0.52–0.57 GB read per token**, not a gain over production. A simple
fit of the two conditions (reads at about 3 GB/s) leaves about 0.4 s per token that is not SSD reads (about 48 layers x 8 ms),
which is now the larger part of decode at pool 0. The 5 tok/s target is 2.9x away. Prefill seconds were equal in the two arms
(they use the same F_NOCACHE prefill); minimum available memory was lower in the candidate (3.6 vs 5.1 GB) as the page cache fills
during decode. One prompt, one decode horizon, no tool-call or long-context decode.

**Pool A/B at 2.7K (complete: `results/flashnext/20261109-215935-layer-gate-manual/`).** Three reversed pairs, 2,739-token prompt,
both arms chunk-major with prefill-only uncached reads, reference pool 0 vs candidate expert pool 2 GB (the production default is a
pool, so every pool-0 number above understates production). Exactness passed on every pair (identical state, logits and all 32
token IDs). Raw arms, pool 0 / pool 2 GB: prefill 106.6 / 110.3 / 97.4 s against 70.2 / 70.7 / 69.7 s (median 26.1 vs 39.0 tok/s,
1.5x); decode 2.01 / 2.22 / 1.82 against 2.61 / 2.56 / 2.60 tok/s (median 2.01 vs 2.60 = 1.29x; spreads 19.9% and 2.3%); decode
physical reads 0.46–0.48 against 0.76–0.77 GB per token; MLX peak 5.28 against 7.33 GB; swap flat (about 1.5 GB), pressure level 1,
minimum available 4.1–5.0 GB against 5.9–6.1 GB. Reading: the pool path is faster although it reads **more** bytes per token. At
pool 0 the OS page cache already serves about 68% of decode reads (0.47 of 1.47 GB logical); the pool's 47% hit rate leaves 0.77 GB,
which equals the earlier simulation (1.47 x (1 - 0.47) = 0.78). The pool wins through a cheaper non-read path (direct slot reads,
the zero-copy prefill ring), not through fewer bytes. Pool-0 decode itself varied between 1.75 tok/s (previous A/B) and 2.01 here,
so run-to-run variation of about 15% exists. One prompt, 32-token horizon.

**Pool A/B at 8K (complete: `results/flashnext/20261109-221415-layer-gate-manual/`).** Three reversed pairs, 8,139-token prompt
(production 2,048 chunk plan), both arms chunk-major with prefill-only uncached reads, pool 0 vs pool 2 GB, exactness enforced and
passed on all three pairs. Raw arms, pool 0 / pool 2 GB: cold prefill 266.2 / 262.1 / 267.1 s (30.6 / 31.1 / 30.5 tok/s) against
184.5 / 184.7 / 182.1 s (44.1 / 44.1 / 44.7 tok/s): median 30.58 vs 44.12 tok/s, **1.44x**, spreads 11.4% / 0.8%; decode (32-token
horizon) 1.69 / 1.62 / 1.51 against 2.44 / 2.45 / 2.46 tok/s, median 1.62 vs 2.45 = **1.51x**; decode physical reads 0.57–0.64
against 0.797 GB per token; prefill physical reads equal (277.7–279.3 vs 279.0 GB, so the prefill gain is not fewer bytes); MLX
peak 6.33 vs 8.16 GB. **Memory was better with the pool, not worse:** the pool arms never grew swap and stayed at pressure level
1 (minimum available 3.7–4.8 GB), while the pool-0 arms grew swap by 0.3–0.75 GB, reached pressure level 2 and fell to 2.5–2.9 GB
available. So the production-default 2 GB pool is both the fastest and the safest configuration measured here. This is the
production default on this checkpoint, not a new gain: it re-confirms about 44 tok/s cold prefill at 8K (historical 47) and
about 2.45 tok/s decode, against targets of 80–100 and 5. One prompt, one machine, uncached prefill reads.

**Layer-major with the 2 GB pool at 8K (partial: `results/flashnext/20261109-224255-layer-gate-manual/`).** Both arms pool 2 GB,
prefill-only uncached reads, 8,139 tokens. Pair 1 (chunk-major then layer-major `w4096`) **passed the exact comparison**. Raw arms:
chunk-major 186.9 s (43.5 tok/s), 279.0 GB read, MLX peak 8.16 GB; layer-major 167.3 s (48.6 tok/s), 247.4 GB read, MLX peak
9.10 GB; swap flat (about 1.0 GB), pressure level 1, minimum available 3.2–3.4 GB in both. That is layer-major about **12%
faster** with 11% fewer bytes at pool 2 GB, one pair only. In the layer-major arm every layer of the three windows (0–2,107,
2,107–6,124, 6,124–8,139; 43.1 / 72.0 / 51.8 s) was staged through the ring (48 ring layers, 0 fallback), i.e. the zero-copy path the
pool-0 gate could not use. The run then stopped at the AC-power rule when the charger was disconnected (22:51, battery 80%,
discharging) during pair 2's layer arm; no arm was retried. **Not a validated A/B** (one chunk-major arm, one layer-major arm,
pair order not reversed).

**Layer-major with the 2 GB pool at 8K, second run (2 of 3 pairs; `results/flashnext/20261109-225535-layer-gate-manual/`).**
Launched on battery (80%), the charger connected during pair 2. Pool 2 GB, prefill-only uncached reads, 8,139 tokens, reversed pair
order (chunk, layer | layer, chunk). **Pair 1 and pair 2 both passed the exact comparison** (token IDs, final logits, every cache
hash). Raw arms (prefill s / tok/s | decode tok/s | prefill GB read | MLX peak): r1 chunk 153.7 / 52.96 | 2.52 | 279.0 | 8.16 GB;
r1 layer 149.5 / 54.46 | 2.50 | 247.5 | 9.10 GB; r2 layer 151.8 / 53.63 | 2.56 | 247.5 | 9.10 GB; r2 chunk 170.7 / 47.67 | 2.30 |
277.7 | 8.16 GB. Pair ratios: **+2.8%** and **+12.5%**; per-condition medians (n = 2) layer 54.05 vs chunk 50.31 tok/s (+7.4%). Chunk
spread is 11%, so the gain is **not statistically established**; the r2 chunk arm is the slowest of the four and followed the AC
switch and a drifting machine (compare r1 chunk 153.7 s). What holds regardless of drift: 11% fewer physical prefill bytes
(247.5 vs about 278 GB, identical in both layer arms), about 0.94 GB higher MLX peak, decode unchanged (about 2.5 tok/s either
way), swap +0.1 GB at most, pressure level 1 throughout. Absolute rate here (53 tok/s) is above the earlier pool-2 pair (44 tok/s)
with the same code, which shows how far machine state moves the absolute numbers; only within-run pairs are comparable.
Decision: layer-major stays opt-in; it is exact, a small positive (reads -11%), and nowhere near 80-100 tok/s.

**Process fault (mine).** The gate stopped before round 3 with "checkpoint or source changed during the comparison": while the
run was in progress I created `models/flashnext/decode_profile.py` and its test under the fingerprinted source tree. The gate
failed closed as designed. No arm was affected (all four arms and both validations completed before the check), but round 3 was
not run and the experiment is formally incomplete. Rule from now on: no file creation or edits under `models/` or `macqwen/`
while a gate runs; draft new code in the scratchpad and move it in afterwards.

**Decode time distribution on the current checkpoint (new, 2026-11-09 night;
`results/flashnext/20261109-231546-decode-profile-manual/`).** Supervised `serve.sh` server, chunk-major reference, pool 2 GB,
prefill-only F_NOCACHE, 2,739-token prompt, 96 decode tokens, AC power. New opt-in instrumentation (`decode_profile.py`,
`FLASHNEXT_DECODE_PROFILE_DIR`, with the existing `FLASHNEXT_PROFILE_IO` and `FLASHNEXT_HOST_WINDOW`) records per-token and
per-layer wall time, the expert_cache timers and the real routes. Four arms: plain 2.56, io-profiled 2.51, host-window 2.53, plain
again 2.20 tok/s. The profiled arms are within 2% of the first plain arm (the profiler does not distort the run); the last plain
arm is 14% slower after about 12 minutes of load (machine drift, recorded, not averaged away). Safety: swap 0.98-1.02 GiB and flat,
pressure level 1, minimum available 3.9 GiB in every arm.

| Per decode token (385 ms median, 83 profiled tokens) | ms | share of wall | what it is |
|---|---:|---:|---|
| `io_wait` (main thread blocked on expert reads) | 288-291 | about 75% | SSD reads, 47.5 read calls/token |
| `score_sync` (GPU drain at the router eval) | 72 | about 19% | GPU compute of attention/GDN/router/shared work since the last sync |
| `topk_python` + `ngram_wait` + `shared_expert` | about 6 | about 1.5% | host Python |
| host-only windows (both devices idle) | 2.2 | 0.6% | `keep_loop` 2.06 ms, everything else under 0.1 ms |
| layer-to-layer gap | 6.4-8.9 ms each, mean 7.9, sum 371 | | uniform across 47 layers |

Timers can overlap, so they are shown beside wall time, not added. Reads: 0.73 GB/token physical, so decode runs at about 2.5 GB/s
of the drive while blocked, and **the SSD wait is the decode ceiling at pool 2 GB**. This corrects an earlier inference in this
document and in the second-opinion report: the "0.4 s/token of non-read time" was measured at pool 0, where the fallback path
(copies, `to_mx`) and the page cache change the split. It does not describe the production pool path, where host Python is about 1.5%
and the GPU about 19%.

Route trace (first current-checkpoint trace; 83 tokens, about 376 distinct (layer, expert) per token because adaptive top-k drops
slots): only 25.6% of a token's experts were also used by the previous token in the same layer. Offline replay (hit rate over
the whole trace, which includes the cold start, so absolute rates are pessimistic; ranking is what matters):

| pool | global LRU (production) | per-layer LRU | decayed LFU | Belady (offline optimal) |
|---|---:|---:|---:|---:|
| 2 GB (651 slots) | 28.6% | 29.1% | 27.0% | 56.1% |
| 3 GB | 37.3% | 38.1% | 42.0% | 63.0% |
| 4 GB | 47.5% | 46.5% | 48.8% | 67.2% |
| 6 GB | 58.0% | 57.3% | 58.6% | 72.1% |

Policies are within a few points of each other and even the oracle gains 27 points at 2 GB: **capacity, not policy, is the lever**
(about +9 points per extra GB; SSD GB/token 0.82 at 2 GB, 0.61 at 4 GB, 0.48 at 6 GB in the replay). The pool is F_NOCACHE-filled, so
free file-cache memory never serves an evicted expert. A victim-cache replay (pool 2 GB plus an LRU of recently read records
standing in for the OS file cache) gives 0.82 / 0.79 / 0.73 / 0.62 GB/token for 0 / 2 / 3 / 4 GB of file cache, i.e. a file cache
of size X is worth about as much as growing the pool by X, without reserving memory. The replay is calibrated: pool 0 with the
roughly 9 GB of free memory it has measured 68% page-cache hits earlier, and the same simulator predicts 0.4 GB/token for that
size. Candidate from this: `--enable pool-buffered-fill` (`FLASHNEXT_EXPERT_POOL_BUFFERED_FILL=1`) makes decode pool fills read
through the file cache on their own descriptors (bytes and order unchanged, prefill ring keeps its uncached descriptors). Hypothesis:
fewer physical bytes per token and a shorter `io_wait`, results bit-identical. Unmeasured live until the A/B below.

**Engineering-risk audit (offline; a read-only static audit by a second agent, each finding verified against the code by me before
acting; report kept in the session scratchpad).**

| Risk raised | Verified? | Action |
|---|---|---|
| `prefill_target` wrapper (`*args, **kwargs`) hides the public signature, defaults and docstring | yes | `functools.update_wrapper(..., assigned=("__doc__","__annotations__"))` at module end, so `inspect.signature` follows `__wrapped__`; name stays `prefill_target`; test added; all 30+ callers pass positionally or by keyword and are unaffected |
| `begin/end_prefill_reads` is a boolean (an inner end would end an outer prefill's mode) | real in principle, **not reachable in the server**: generation is serialized by the scheduler (a waiting request gets 503 `server_busy`) | reference-counted anyway (`_prefill_depth`, lock); tests: nesting, unbalanced end, exception and `KeyboardInterrupt` inside prefill both restore buffered reads |
| Elevated-pressure uncache switch can stay on after an abort | the switch is opt-in (`FLASHNEXT_UNCACHE_WHEN_ELEVATED=1`, off by default) and `_after_request` runs in a `finally` that restores buffered reads | no change |
| `ExpertWindow.drain()` swallows orphaned prefetch errors | partly: a consumed layer's error is raised by the real wait; only a layer that was never consumed was silent | now counted (`stats["prefetch_drain_errors"]`) and logged at WARNING; test added |
| Execution-only flags excluded from cache identity | confirmed correct: `FLASHNEXT_UNCACHED_EXPERT_READS`, layer-prefetch and the new pool buffered fill do not change any byte or order; layer-major plan identity `layer-major-v1/w<window>[/hybrid-small-v1]` is in the fingerprint so layer-major and chunk-major snapshots never cross-load (`prefix_session.py:112-147`, load rejects a fingerprint mismatch) | none; the real-model cross-mode snapshot test is still not run |
| Gate probe cost lands inside `prefill_seconds` | partly: `record_logits` (host copy of 0.5 MB of logits, SHA-256, one fsync) runs inside `prefill_target`; the full cache-state hash runs **after** the prefill clock stops and before the decode clock starts | cost is one eval that the first token needs anyway plus about 1 ms of hashing and one fsync: about 0.01% of a 150 s prefill, identical in both arms of every pair; not changed, disclosed here |

**Buffered pool fills, decode A/B at 2.7K (complete: `results/flashnext/20261109-233548-layer-gate-manual/`).** First measured
decode gain of the overhaul. Three reversed pairs (chunk, layer | layer, chunk | chunk, layer), AC power, 2,739-token prompt, both arms
chunk-major, pool 2 GB, prefill-only F_NOCACHE; the "layer" arm adds `--enable pool-buffered-fill` (reference arm: uncached pool
fills, the production behaviour). **Exactness passed on all three pairs** (token IDs, final logits, every cache hash). Raw arms,
reference / candidate: decode over the 31-token horizon 2.772 / 2.879 / 2.740 against 3.620 / 3.586 / 3.597 tok/s (medians 2.772 vs
3.597, **1.298x**; every pair favours the candidate, pair ratios 1.31, 1.25, 1.31; spreads 5.0% vs 0.9%); 23-token tail 2.655 vs 3.583;
decode physical reads 0.778 / 0.732 / 0.778 against 0.499 / 0.502 / 0.499 GB/token (**-36%**); prefill unchanged (40.31 vs 40.14
tok/s, 0.996x, the flag does not touch prefill); MLX peak 7.33 GB in both. Memory: swap flat (968-976 MiB) in all six arms, pressure
level 1, available after the arm 4.8-5.9 GB (candidate 5.0-5.2). Page-ins during the arm: about 3 GB (reference) against about
13 GB (candidate): the file cache is now actually working, which is the mechanism the replay predicted (its estimate for a 4 GB
victim tier was 0.62 GB/token; the machine did better, 0.50). Limits: one prompt, 32-token horizon, 2.7K context, AC power, drift
inside the run is small (reference prefill 66.1-68.0 s). Not yet shown: 8K context (running next), a long decode (the cache keeps
warming), and behaviour under real memory pressure.

**Buffered pool fills, decode A/B at 8K (complete: `results/flashnext/20261109-234803-layer-gate-manual/`).** Same design with the
8,139-token prompt (production 2,048-token chunks), AC power throughout, three reversed pairs, **exactness passed on all three**.
Raw arms, reference / candidate: decode 2.20 / 2.52 / 2.50 against 2.99 / 2.45 / 2.84 tok/s (medians 2.50 vs 2.84, **+13.4%**;
pair ratios 1.36, 0.97, 1.14; the reference spread is 14% and includes a slow first arm); decode physical reads 0.810 / 0.810 /
0.810 against 0.555 / 0.605 / 0.578 GB/token (**-29% median**, consistent in all three pairs); prefill 47.35 vs 47.17 tok/s
(0.996x, untouched). MLX peak 8.16 GB in all arms; swap flat (952-968 MiB), pressure level 1, available after the arm 4.1-4.6 GB,
compressor 1.1-1.65 GB (candidate arms 1.4-1.65 against 1.1-1.4). Reading: the physical-byte reduction is reliable at both context
lengths (-36% at 2.7K, -29% at 8K); the speed gain is clear at 2.7K (+30%, all pairs) and positive but **not firmly established at
8K** (+13% on medians, one pair negative, noise of the same size). Decision: stays **opt-in** (`--enable pool-buffered-fill`); it is
exact, memory-safe in these runs and helps, but the owner-facing default is not changed on this evidence (one prompt, 32-token horizon).

**Layer-major with 8,192-token windows at 8K (complete, first validated prefill gain; `results/flashnext/20261110-001110-layer-gate-manual/`).**
AC power, pool 2 GB, prefill-only F_NOCACHE, 8,139-token prompt, three reversed pairs, **exactness passed on all three** (token IDs,
final logits, every cache hash; candidate plan `layer-major-v1/w8192`, windows [0, 2,107] and [2,107, 8,139], 48 ring layers and 0
fallback layers in both). Raw arms, chunk-major / layer-major: prefill 174.2 / 174.8 / 171.1 s (46.7 / 46.6 / 47.6 tok/s) against
150.5 / 148.8 / 150.4 s (54.1 / 54.7 / 54.1 tok/s): medians 46.72 vs 54.12, **1.159x**, pair ratios 1.158, 1.174, 1.137, spreads
2.1% and 1.1%; prefill physical reads 279.1 / 277.7 / 277.6 against 168.3 / 167.2 / 168.3 GB (**-40%**); MLX peak 8.16 vs 8.91 GB
(+0.75 GB); decode unchanged (2.50 vs 2.54 tok/s). Memory: pressure level 1 in all arms, available after the arm 4.7-4.8 GB
(chunk) and 6.3-6.5 GB (layer); swap rose 0.10-0.16 GiB in each layer arm (944 to 1,261 MiB over the run), never near the
3 GiB ceiling. Why only +16% for -40% bytes: the two windows each read 75.5 GB in 26.3 s (2.9 GB/s) but took 42.6 s and 107.6 s, so
the prefill is **no longer I/O-bound**: about 97 of 150 s are compute and other work, and the second window (6,032 tokens, 107.6 s)
runs at 56 tok/s end to end (about 74 tok/s for its non-read part). Reaching 80-100 tok/s now needs compute work (per-layer split
below), not fewer bytes. Limits: one prompt, AC power, 2 GB pool. For comparison the 4K-window plan measured 54.05 vs 50.31 tok/s over two
pairs in an earlier run (+7%, reads -11%), so the 8K window is the better plan on this prompt, but the two runs are different
sessions and are not an A/B of 4K against 8K. 16K, the boundary policy and the cross-mode snapshot test are unmeasured. Status: validated exact and faster on this checkpoint at 8K;
**stays opt-in** (`--enable layer-major`).

**Where the 8K layer-major prefill time goes (3 single-arm profiles with the new per-layer trace; `results/flashnext/20261110-003731-`,
`-004140-prefill-profile-manual/`; one arm each, not paired, not a speed result: 58.6-55.8 tok/s with drift from 137 to 146 s).**
Trace-only additions (`layer_seconds`, `attention_layers`, `layer_memory_mb`, `layer_reads`, `chunk_seconds` in the execution
segments; no extra synchronisation, no change to results). Findings, in order of size:

1. **Expert staging is not the bottleneck any more.** Constant 0.52-0.56 s per layer (75.5 GB in 26.2-26.3 s per window, 2.9 GB/s)
   in both windows; it adds no MLX memory (the ring lives in the pool).
2. **Compute slows steadily with layer depth inside the big window.** Window 1 (2,107 tokens): every layer 0.82-0.91 s, flat.
   Window 2 (6,032 tokens): linear layers 1.39 s early to about 2.1 s late, attention layers 1.8 to 3.4 s, the same factor for both
   types. Per-chunk times grow together, so it is not one slow kernel. Machine state during window 2 (safety trace): available memory
   fell from 6.7 to 3.0-4.5 GB, the compressor rose from 0.5 to 3.4 GB and file-backed memory fell from 3.9 to 1.3 GB at the start of
   window 2 and stayed there; pressure level 1 (2 in one earlier arm). The ramp is smooth and the compressor level is steady
   through it, so I attribute it to sustained load on a fanless machine (thermal), **not proven**; thermal pressure is now sampled
   (`notifyutil -g com.apple.system.thermalpressurelevel`, 0 to 4, no root needed) in every harness sample, with a stop at 3.
3. **Layer 1 (the model's per-layer-embedding layer, `ple_layer_ids = [2]`) costs about 8 s per 8K prefill** (about 6%): 2.6/2.0/2.0
   s for each 2,008-token chunk plus 0.54 s for an 8-token chunk, against 0.83/0.26/0.26/0.05 s for ordinary layers. Its 16
   n-gram rows per token are read from disk one random row at a time (`FLASHNEXT_NGRAM_PARALLEL_MIN=16` is already on in the served
   launch); it has a fixed cost of about 0.5 s per call and about 0.45 ms per token. The same cost exists in chunk-major. A batched,
   deduplicated look-ahead of the window's rows (the ids depend only on the token IDs) is exact and could save most of it.
4. **Window 2 holds 1.8 GB more MLX memory from layer 1 to the window end** (active 4.9 to 6.7 GB at layer 1, no growth in window 1),
   exactly when macOS starts compressing; its source is not yet identified. Peak 8.45 GB appears at the first attention layer.

**Thermal evidence for finding 2 (arm 4, `results/flashnext/20261110-004626-prefill-profile-manual/`, 61.1 tok/s, 133 s).** OS thermal
pressure read 0 until 63 s, 1 at 73-83 s and **2 (heavy) from 93 s to the end**, exactly while the window-2 layers slowed (every
fourth non-attention layer: 1.40, 1.41, 1.41, 1.40, 1.47, 1.47, 1.56, 1.58, 1.67, 1.78, 1.85 s). The compressor was already steady at
3.2-3.6 GB from 73 s, so memory state does not explain the ramp; thermal pressure does. Consequence: on this fanless machine sustained
prefill runs at full speed for about 60-90 s and then throttles, so a long prefill averages 10-20% below its cool-state rate, and any
comparison must interleave arms (the reversed-pair design already does) rather than trust arm order. It also means the compute
needed per token, not only the bytes, sets the long-prompt ceiling. Early (cool) layers of window 2 cost 1.4 s with 0.54 s of that
the unoverlapped expert read, so about 0.85 s per layer for 6,032 tokens (about 7,100 tokens/s of layer compute); overlapping the
reads with compute and removing the layer-1 and memory costs is where the remaining headroom is.

**Finding 4 resolved (arm 5, `results/flashnext/20261110-005029-prefill-profile-manual/`, per-chunk memory trace).** The +1.9 GB of
MLX memory at layer 1 of window 2 appears after the **fourth chunk, the 7-token tail (tokens 8,132-8,139)**, not the large chunks:
active memory after the layer-1 chunks was 4,775 / 4,734 / 4,837 MB and 6,737 MB after the 7-token chunk, and that chunk costs 0.43-0.54
s at layer 1 against 0.05 s at ordinary layers. A chunk of 8 rows or fewer takes the decode-style MoE path
(`expert_cache.__call__` uses the window only above 8 rows), which uses the 2 GB decode pool as a Metal buffer for the first time
(about the pool's size). In layer-major that happens in the middle of the window, so the 1.9 GB stays for the rest of it and
coincides with the start of memory compression (compressor 0.5 to 2.7-3.4 GB); chunk-major pays the same cost but only at the very
end. Fix: `layer_major.peel_tail` runs a trailing chunk of at most 8 tokens through the reference chunk-major path **after** the
last window (`prefill._run_layer_major`), where it sees exactly the cache state and executor the reference gives it, so results are
unchanged (unit test compares token and every cache array bit for bit with chunk-major, and the executed edges). Rollback switch:
`FLASHNEXT_PREFILL_TAIL_CHUNK=0` restores the old schedule. Predicted effect: window 2 no longer carries +1.9 GB, so less
compression and a smaller peak; speed effect unmeasured until the paired gate below.

**Layer-major (8,192-token windows) with the tail-chunk fix, paired gate at 8K (complete, exact; `results/flashnext/20261110-005904-layer-gate-manual/`).**
AC power, pool 2 GB, prefill-only F_NOCACHE, 8,139-token prompt, three reversed pairs, **exactness passed on all three** (token IDs, final
logits, every cache hash, executed edges). Segments of the candidate arms: [0, 2,107] layer, [2,107, 8,132] layer, [8,132, 8,139] chunk
(the 7-token tail, 1.6 s). Raw arms, chunk-major / layer-major: prefill 159.2 / 172.2 / 175.6 s (51.1 / 47.3 / 46.4 tok/s) against
144.9 / 131.9 / 141.9 s (56.2 / 61.7 / 57.4 tok/s): medians 47.27 vs 57.37 tok/s, **1.214x**, pair ratios 1.099, 1.305, 1.238; spreads
10.1% and 9.6% (thermal drift: pressure reached 2 in every layer arm); prefill physical reads 279 / 277 / 278 against 168 / 168 / 168 GB
(**-40%**); **MLX peak 7.48 GB against 8.16 GB for the reference** (8.91 GB before the fix); decode unchanged (2.50 vs 2.47 tok/s). Memory:
pressure level 1 in all arms, minimum available 3.2-3.6 GB in the layer arms, compressor at the end 0.67-0.73 GB (1.6-2.1 GB peak
during the arm; it was 3.4 GB sustained before the fix), swap flat to decreasing (1,415 to 1,279 MiB over the run). Compared with the same
gate before the fix (54.1 vs 46.7, 1.159x, peak 8.91 GB) the candidate gained about 6% and lost 1.4 GB of footprint; the two runs are
different sessions, so the 6% is indicative, not an A/B of the fix. Status: **validated exact and faster at 8K on this checkpoint and
this prompt**, lower peak memory than the reference; still opt-in (`--enable layer-major`) pending 16K, the boundary policy and the
cross-mode snapshot test. Remaining prefill budget per window: expert reads are unoverlapped (about 26 s per window, 58 s of about
140), layer 1 (n-gram rows) about 8 s, thermal throttling 10-20%.

**Two-half ring overlap, first measurement (partial overlap; `results/flashnext/20261110-012814-layer-gate-manual/`).** New opt-in form of
`--enable layer-prefetch` with the prefill ring: a second ring half (scratch pool +1.5 GB while prefilling) so layer l+1's rows are read
with direct I/O while layer l computes (`PrefillRing.stage_begin` / `stage_finish` / `cancel_stage`, `ExpertWindow._prefetch_ring`;
bytes, order and destination layout are those of the blocking stage). Paired gate with `--candidate layer-major --prefetch`, 8K, AC
power, pool 2 GB: **exactness passed on all three pairs**; prefill 62.0 / 57.1 / 64.6 tok/s against chunk-major 52.5 / 48.5 / 47.2 (medians
61.98 vs 48.53, **1.277x**; pair ratios 1.180, 1.176, 1.368; the previous layer-major run without overlap was 1.214x in another
session); MLX peak 9.05 GB (+1.5 GB), minimum available 1.8-2.1 GB, compressor up to 3.4-6.8 GB, pressure level 1, swap flat (1,255 to
1,279 MiB). The per-layer wait trace showed **only every other layer overlapped**: waits of 0.0 s on odd layers and 0.8-1.0 s on even
layers (the blocking read also got slower, 0.95 s against 0.53 s, under the higher memory pressure). Cause, a bug of mine: the guard
"one begun staging at a time" refused to begin layer l+2 at the start of layer l+1 because layer l+1's own staging had not been
consumed yet. Two outstanding stagings are legal with two halves (one waiting, one filling; `begin_fill` refuses when no half is
free), so the guard now allows two. Window 1 (2,107 tokens) stays read-bound at about 26 s (compute 0.3 s per layer against 0.53 s
read); window 2 is compute-bound, so a fully overlapped schedule is predicted at about 105 s for the 8K prefill (about 77 tok/s)
against the measured 126-143 s; the corrected guard is re-measured next.

**Two-half ring overlap with the corrected guard: paired gate at 8K (complete, exact; `results/flashnext/20261110-015335-layer-gate-manual/`).**
Same design as the partial-overlap run, `--candidate layer-major --prefetch`, AC power, pool 2 GB, 8,139-token prompt, three reversed
pairs, **exactness passed on all three** (token IDs, final logits, every cache hash, executed edges). Raw arms, chunk-major /
layer-major + overlap: prefill 161.7 / 172.8 / 168.5 s (50.3 / 47.1 / 48.3 tok/s) against 110.8 / 115.0 / 115.7 s (73.4 / 70.7 /
70.3 tok/s): medians 48.30 vs 70.75 tok/s, **1.465x**, pair ratios 1.458, 1.502, 1.456, spreads 6.7% and 4.4%; prefill physical reads
279 / 277 / 277 against 168 / 168 / 168 GB; decode unchanged (2.49 vs 2.51 tok/s). The per-layer trace shows the overlap working:
window 2 (tokens 2,107-8,132) waited **1.0 s in total** for expert reads (it was 26.3 s without overlap), window 1 (2,107 tokens, compute
0.3 s per layer against 0.53 s of read) still waited 9.7-10.8 s because it is read-bound; window times 29.6-30.1 s and 79.2-83.6 s
against 42.6 s and 94-108 s without overlap. Memory cost: MLX peak 9.05 GB against 8.16 GB for the reference (+1.5 GB, the second half),
minimum available 1.9-2.7 GB, compressor 3.3-3.7 GB at its peak, **pressure level 1, swap flat at 1,255 MiB** in every arm (OS thermal
pressure reached 2 in each layer arm, as in the earlier runs). Status: validated exact and faster at 8K (**about 70 tok/s cold
prefill against 47 for the reference**) on this checkpoint and prompt; the machine has little headroom while it runs (about 2 GB
available at the minimum), so it stays **opt-in** (`--enable layer-major --enable layer-prefetch`). Not yet measured: 16K, a 3 GiB-swap
stress, other applications competing for memory. Remaining prefill budget at 8K: window 1 is read-bound (the forced system-prompt
boundary makes the 75.5 GB pass happen twice, about 26 s), layer 1 (n-gram rows) about 8 s, thermal throttling in window 2.

**Inner snapshots: one window across the system-prompt boundary (complete, exact; `results/flashnext/20261110-022214-layer-gate-manual/`).**
Ending a window at the system-prompt boundary made every expert be read a second time (window 1 was read-bound, about 26 s). A boundary on a
chunk edge inside a window is now snapshotted without ending the window: each layer is frozen (`prefix_cache.freeze_cache`, a new Python object
over the same immutable arrays, cut to the layer's offset) the moment it finishes the chunk that ends at the boundary, and the snapshot is assembled
after the window (`prefix_session.snapshot(position, frozen)`). Default on inside layer-major, rollback `FLASHNEXT_PREFILL_INNER_SNAPSHOTS=0`
(the old cut at `required` positions). Unit tests: the stored state at both edges equals the chunk-major state byte for byte, stays correct after
later chunks update the live KV buffers, and a restore from it resumes to the same token and cache.
Paired 8K gate, `--candidate layer-major --prefetch`, AC, pool 2 GB, 8,139 tokens, three reversed pairs, **exactness passed on all three**:
chunk-major 155.7 / 170.5 / 168.5 s (52.3 / 47.7 / 48.3 tok/s) against layer-major + overlap + inner snapshots 85.5 / 90.9 / 92.9 s
(**95.2 / 89.5 / 87.6 tok/s**): medians 48.3 vs **89.5 tok/s, 1.853x**, pair ratios 1.82 / 1.88 / 1.81, spread 8.2%; prefill physical reads 277-279
vs 89 GB; the prefill is one 8,132-token window (83.8 s) plus the 7-token tail (1.6 s); MLX peak 9.74 vs 8.16 GB. Machine: swap 1.2-1.7 GiB in every arm
(limit 3 GiB), memory pressure level 1 except one sample at level 2 in the first layer arm (minimum available 1.2 GB in that arm, 2.2-3.3 GB in the
others), compressor up to 3.8 GB, OS thermal pressure up to 2 in every arm (chunk-major too). Status: validated exact and faster on this checkpoint
and prompt at 8K; the memory margin while it runs is thin (about 1-3 GB available at the minimum), so it stays **opt-in**
(`--enable layer-major --enable layer-prefetch`). Not yet measured: 16K, a restart from the stored system snapshot after an inner-snapshot
prefill on the real checkpoint (unit-tested on the small model only), competing applications.

**16K paired gate (complete, exact; `results/flashnext/20261110-025312-layer-gate-manual/`).** Same configuration as the 8K gate above
(`--candidate layer-major --prefetch`, AC, pool 2 GB, uncached prefill reads), 16,139-token prompt (`--source-tokens 14000`; the gate's
workload-shape check was widened to 12,288-20,480 tokens for this run, no memory or swap rule changed), three reversed pairs, exactness
passed on all three. Chunk-major 361.0 / 361.5 / 359.7 s (**44.7 tok/s**, spread 0.5%), layer-major + overlap + inner snapshots
239.5 / 232.7 / 233.3 s (67.4 / 69.4 / 69.2, median **69.2 tok/s**, spread 2.9%): **1.547x**. Physical prefill reads 538 vs 177 GB; MLX peak
10.35 vs 11.92 GB. Trace of a layer arm: window 1 (0-8,119) 86.4 s, window 2 (8,119-16,132) 148.4 s, 7-token tail 4.2 s; snapshots
at every planned boundary. The 80-100 tok/s target is **not** met at 16K (69 tok/s): the second window computes slower (attention over 8K+
context, thermal pressure 2 for most of each arm) and each window still reads every expert once (88 GB each).
**Machine margin is thin at 16K.** Swap in use reached 2,848 MB (2.78 GiB, limit 3 GiB) in the first layer arm and 2.2-2.7 GiB in the
chunk-major arms, memory pressure level 2 in every arm, minimum available 1.4-2.0 GB. The gate did not stop, but the run is within about 0.2 GiB of the
owner's swap ceiling; part of that is ambient drift (swap started each arm at 1.2-1.6 GiB). Do not repeat or lengthen 16K runs without first
clearing swap headroom; layer-major stays opt-in.

Consequence for the roadmap: double-buffering the layer reads (section 10, item 2) adds a 1.5 GiB half on top of a footprint the
machine already barely accommodates, so it is not worth building before the exactness gate has passed and a measured candidate
footprint exists. Decode cache-policy work needs a current-checkpoint route trace, which needs the same headroom.

### Research checks (2026-11-09)

- [Layered Prefill](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html)
  schedules contiguous layer groups to interleave prefill and decode. The current windowed implementation instead runs a
  complete prefill window layer by layer; it does not implement the paper's concurrent-serving scheduler. The paper's reported
  results are not measurements on this Mac or checkpoint.
- [SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching) motivates Least-Stale through predictable
  layer access. Its hit-rate results do not establish a win here; compare against actual current-checkpoint physical misses.
- [moe-offload](https://github.com/Jakevin/moe-offload) offers per-layer cache allocation and optional bounded prefetch;
  [Vates](https://github.com/AMOS144/Vates) combines native demand reads, per-layer allocation and verified MTP. Their different
  workloads and configurations are hypotheses for experiments, not transferable throughput numbers.
- [FlashMoE](https://arxiv.org/abs/2601.17063) and [SP-MoE](https://arxiv.org/abs/2510.10302) motivate measuring cache
  replacement and draft-aware expert traffic, respectively. Neither removes the need to include predictor and verification costs.
- [MLX Metal kernel documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) specifies that
  `ensure_row_contiguous` defaults to true and can copy non-contiguous inputs. Any change to that option needs correct stride
  handling and measurement with the locally installed MLX; a view alone does not prove a copy-free execution path.

The current ring reads with `preadv` into an anonymous, Metal-wrapped pool and exposes contiguous part-major slices. Existing
checkpoint-free experiments in `pool_ring.py` report a materialized copy for strided expert-record views and no such penalty for
compact views; this is evidence for that synthetic case, not proof that every downstream real-checkpoint operation is copy-free.
The fallback store can instead construct MLX arrays from NumPy or use `mx.from_dlpack(..., copy=False)`. These paths differ in
ownership and copy behavior and require separate measured comparisons before changing defaults.

## 10. Next bottleneck and next steps (not done)

1. Compute-bound after layer-major: about 125 tok/s on this model before attention growth. Skipping the last layer's MoE for
   non-final windows (its output is unused; 1/48 of compute and one 1.5 GB read per window) is exact and cheap.
2. Double-buffered asynchronous layer reads (ring `halves=2`, +1.6 GB scratch) to move from "serial" to "overlapped" in the table:
   roughly +20% at 8K windows. Needs its own hazard tests and a memory check.
3. Union-only reads (route first, read the union) save the unused 10-20% at small windows.
4. Per-window snapshots only; capturing recurrent state at inner chunk edges would restore chunk-edge resumability.

## Commands

```bash
./serve.sh --overnight --enable layer-major            # opt-in layer-major prefill (supervised)
./chat.sh --mode agentic --enable layer-major ...      # same, direct launch (use serve.sh for the swap safeguards)
FLASHNEXT_PREFILL_WINDOW_TOKENS=4096 ...               # smaller window: less lost work, more traffic
curl -s localhost:8080/v1/cache/diagnostics | jq .last_prefix   # prefill report incl. executed edges
```
