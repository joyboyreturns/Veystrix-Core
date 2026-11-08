# Prefix profiles and managed SSD cache (opt-in, offline-validated)

Status: implemented and tested offline only (small random-weight model, injected clock and disk space). Never run on the real
checkpoint. Off by default: `FLASHNEXT_PREFIX_PROFILES=1` (plus `FLASHNEXT_PREFIX_CACHE=1`) turns it on. Production defaults,
`main`, Q4, routing and the checkpoint are untouched.

Code: `models/flashnext/prefix_profiles.py` (new), small hooks in `prefix_cache.py` and `prefix_session.py`.
Tests: `tests/unit/test_prefix_profiles.py` (34), `tests/unit/test_prefix_endurance.py` (5, including a 100-request run).

## What it is, and what it is not

The exact snapshots, exact matching and exact restoration are the existing `PrefixCache` (SHA-256 chain over the engine
fingerprint and every token id, chunk-plan provenance, lossless storage). Nothing here matches approximately, compresses
state or changes what is computed. The profile manager adds records and housekeeping around those snapshots.

A **profile** is the record for one stored snapshot:

| Field | Where it lives |
|---|---|
| exact token-prefix identity (chain key, token count), model/configuration fingerprint | manifest (immutable) |
| chunk-plan provenance (`plan_edges`) | manifest and inside the checksummed data file |
| KV deltas, GatedDeltaNet recurrent and convolution state, position state; SSD location and size | `<key>.safetensors` |
| hit count, last-used time, time saved, `kind` ("system"), pin, execution profile (`config`) | `index.profiles` sidecar |
| estimated recomputation cost | computed: tokens over the learned prefill rate (default 60 tok/s, learned from real prefills) |
| prefix sightings (`seen`) | sidecar |

The sidecar is only metadata: it is atomic, checksummed and has a `.bak`. Losing it loses counters and pins, never state;
the manager rebuilds from the manifests.

## How a repeated system prompt is recognised

No heuristic. The system boundary is an anchor of every chunk plan, so every request with the same system prompt reaches
the same exact key. `PrefixSession.prefill` calls `observe(prompt[:system_boundary])` before the lookup; the count of
sightings and the hit count decide. A snapshot of the system message that has been seen or hit `pin_after` (default 3) times
is pinned automatically (`"auto"`); pins may take at most half of the budget. `pin()` / `unpin()` give manual pins.

For a matching request the existing path runs: longest exact valid prefix (plan-compatible), RAM tier if resident else SSD,
prefill only the suffix. The warm decode pool and pool release are the memory manager's job and are unchanged.

## SSD cache management

Limits (all configurable): `FLASHNEXT_PREFIX_PROFILES_MAX_GB` (10), `..._FREE_RESERVE_GB` (50), `..._PIN_AFTER` (3),
`..._HALF_LIFE_H` (72). Hot RAM budget: the existing `FLASHNEXT_PREFIX_CACHE_RAM_GB` (0.75).

* **Before every write**, room is made inside the byte budget and the free-space reserve. If protection prevents it, or the
  reserve cannot be restored even by evicting everything evictable (the volume is full for other reasons), nothing is
  evicted and the SSD write is skipped (the snapshot stays in RAM, `writes_skipped` counts it). No futile wiping.
* **Eviction order**: snapshots of another fingerprint or format first (they cannot hit under this configuration), then the
  lowest value, `(1 + hits) x recompute_seconds x 0.5^(age/half-life) / bytes`; oldest first on a tie. This is LRU informed
  by reuse frequency and by how expensive the state is to rebuild per byte.
* **Never evicted**: pinned entries, leased entries, and every ancestor of those. Only leaves are dropped, so a chain is
  never left without its parent. A lease covers the chain being restored (released in a `finally`, so a cancellation cannot
  leak it) and the snapshots a running prefill extends (held until the prefill ends, however it ends).
* **RAM tier**: ranked by the same value, so pinned and frequently used profiles stay hot and cold ones go to the SSD tier
  first (written out before they leave RAM, so nothing is lost). `shrink_ram` (used by the memory manager under pressure)
  uses the same order and skips leased entries. `warm_ram()` reloads the most valuable snapshots after a restart.
* **Atomic writes and checksums**: unchanged `write_entry` (temporary name, fsync, rename, manifest last); every restore
  verifies size, SHA-256, embedded key, plan, token digest.
* **Crash recovery at startup**: temporary files of an interrupted write and data files that never got a manifest are
  removed (after a 60 s grace in case another process is writing); manifests without data or with a wrong size are dropped;
  chains with a missing ancestor are dropped; the budget is enforced again, so a lowered limit takes effect across restarts.
  Entries of another fingerprint are kept (not deleted at startup) and become the first eviction candidates.
* **Deletion scope**: only files in the cache directory named exactly `<64 hex>.json` / `.safetensors`, the matching
  temporary names and `index.profiles` (+ `.bak`). No symlink is followed or removed. A directory that holds anything else
  and has no marker file (`.veystrix-prefix-cache`) is refused (`UnsafeDirectory`). Model weights and user files are never
  reachable by the delete path.
* **Records** (`report()`): lookups, hits, misses, restored tokens, RAM/SSD hits, SSD bytes read/written, evictions (count and
  bytes), writes skipped, orphans removed, invalidated, time saved (estimated recompute minus restore time), pins,
  config drift, index state.

## Tests

* Records: identity, cost, execution profile, frequency, time saved, restart persistence (counters and pins), config drift.
* Recognition: repeated system prompt recognised and pinned, only system snapshots auto-pin, pin cap, changed prompt is new.
* Eviction: value order (a frequently used old entry survives a new unused one), oldest first among unused, pinned never
  evicted and an unfittable write skipped, pinned child protects its chain, no orphaned child, leased entries kept and lease
  released when the block raises, RAM pressure moves cold entries to SSD while pinned ones stay resident.
* Limits: reserve evicts until it fits, futile reserve evicts nothing, oversize write skipped.
* Crash recovery: leftover temporaries (and the grace period), orphan data, manifest without data, truncated data, flipped
  byte, `scrub`, corrupt index (backup, then manifest rebuild), tampered checksum, write interrupted between data and
  manifest, lowered budget at restart, pinned entry surviving it.
* Safety: foreign fingerprints kept then evicted first, non-cache files never touched, symlink not followed, eviction never
  touches weights.
* Bit-identity: SSD-restored continuation equals the cold run with the same boundary (logits and every cache array).
* Endurance (100 requests): three-plus system prompts (one dominant), different conversations sharing it, short tool-result
  continuations, RAM shrink to 0/0.4/0.9 MB, a ~12-snapshot SSD budget reached repeatedly, restarts, cancellations inside the
  prefill after a snapshot, every result bit-identical (token and all cache arrays) to a cold same-plan run; after every
  request: no leaked lease, exact byte accounting, SSD budget respected, no orphaned child, every file accounted for, no stale
  record or token tape; the dominant prompt is auto-pinned and hits every time; MLX memory flat between request 40 and 99.
* Cancellation during restore, during an interrupted write, incompatible configuration (chunk plan, numerics setting).

## Remaining risks

* Not exercised on the real checkpoint: entry sizes are 0.17 GB per snapshot there against 0.5 MB in the tests; sizes of
  the 50 GB reserve and 10 GB budget interact with real files only through `shutil.disk_usage` (injected in tests).
* Recompute cost is linear in tokens (prefill rate learned from real prefills). Real cost is mildly super-linear (attention);
  it only orders eviction.
* A RAM-only snapshot whose SSD write was skipped is lost on eviction from RAM (a miss, never wrong state); a child of it
  invalidates itself on the next lookup.
* One engine process per cache directory. Concurrent processes are only protected by the 60 s orphan grace; there is no
  cross-process lock.
* `kind="system"` comes from `PrefixSession.system_boundary` (first `<|im_end|>\n`). A harness that sends several system
  messages gets the first one recognised.
