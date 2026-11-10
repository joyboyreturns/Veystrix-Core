# UltraReview Remediation Ledger

Tracking all 30 Ultra Review findings across reviews 1, 2, and 3.

| ID | Phase | File | Function/Lines | Severity | Status | Commit |
|---|---|---|---|---|---|---|
| R1-01 | Phase 4: Supervision & Failure Recovery | `models/flashnext/auto_reclaim.py` | L187-L209 | normal | verified | 9cac8f5 |
| R1-02 | Phase 4: Supervision & Failure Recovery | `models/flashnext/overnight.py` | L137-L141 | normal | verified | 9cac8f5 |
| R1-03 | Phase 3: Persistent Cache & Recovery | `models/flashnext/prefix_session.py` | L213-L220 | normal | verified | fe4682a |
| R1-04 | Phase 4: Supervision & Failure Recovery | `models/flashnext/flight_recorder.py` | L43-L43 | normal | verified | 9cac8f5 |
| R1-05 | Phase 2: Correctness & Tool-Calling | `models/flashnext/expert_cache.py` | L1836-L1844 | normal | verified | 7f0da26 |
| R1-06 | Phase 2: Correctness & Tool-Calling | `models/flashnext/prefill.py` | L392-L394 | nit | verified | 7f0da26 |
| R1-07 | Phase 5: Lower-Risk Engine Findings | `models/flashnext/expert_pool.py` | L313-L332 | nit | verified | 9f7985c |
| R1-08 | Phase 5: Lower-Risk Engine Findings | `models/flashnext/expert_cache.py` | L1623-L1623 | nit | verified | 9f7985c |
| R1-09 | Phase 5: Lower-Risk Engine Findings | `models/flashnext/prefill.py` | L246-L250 | nit | verified | 9f7985c |
| R1-10 | Phase 5: Lower-Risk Engine Findings | `models/flashnext/prefill_scheduler.py` | L39-L39 | nit | verified | 9f7985c |
| R2-01 | Phase 4: Supervision & Failure Recovery | `macqwen/server.py` | L928-L930 | normal | verified | 9cac8f5 |
| R2-02 | Phase 2: Correctness & Tool-Calling | `macqwen/server.py` | L223-L223 | normal | verified | 7f0da26 |
| R2-03 | Phase 2: Correctness & Tool-Calling | `macqwen/server.py` | L419-L419 | normal | verified | 7f0da26 |
| R2-04 | Phase 4: Supervision & Failure Recovery | `macqwen/server.py` | L1473-L1476 | normal | verified | 9cac8f5 |
| R2-05 | Phase 2: Correctness & Tool-Calling | `macqwen/server.py` | L355-L357 | normal | verified | 7f0da26 |
| R2-06 | Phase 3: Persistent Cache & Recovery | `models/flashnext/prefix_cache.py` | L832-L841 | normal | verified | fe4682a |
| R2-07 | Phase 3: Persistent Cache & Recovery | `models/flashnext/prefix_profiles.py` | L382-L383 | normal | verified | fe4682a |
| R2-08 | Phase 2: Correctness & Tool-Calling | `macqwen/server.py` | L1459-L1469 | normal | verified | 7f0da26 |
| R2-09 | Phase 2: Correctness & Tool-Calling | `macqwen/server.py` | L1245-L1270 | normal | verified | 7f0da26 |
| R2-10 | Phase 2: Correctness & Tool-Calling | `models/flashnext/prefix_session.py` | L251-L268 | normal | verified | 7f0da26 |
| R3-01 | Phase 4: Supervision & Failure Recovery | `macqwen/cli.py` | L37-L65 | normal | verified | 9cac8f5 |
| R3-02 | Phase 4: Supervision & Failure Recovery | `models/flashnext/chat_guard.py` | L121-L124 | normal | verified | 9cac8f5 |
| R3-03 | Phase 4: Supervision & Failure Recovery | `macqwen/cli.py` | L303-L304 | normal | verified | 9cac8f5 |
| R3-04 | Phase 4: Supervision & Failure Recovery | `models/flashnext/request_admission.py` | L47-L53 | normal | verified | 9cac8f5 |
| R3-05 | Phase 2: Correctness & Tool-Calling | `macqwen/sessions.py` | L154-L189 | normal | verified | 7f0da26 |
| R3-06 | Phase 3: Persistent Cache & Recovery | `macqwen/cache_tools.py` | L114-L140 | normal | verified | fe4682a |
| R3-07 | Phase 4: Supervision & Failure Recovery | `macqwen/session.py` | L649-L661 | normal | verified | 9cac8f5 |
| R3-08 | Phase 3: Persistent Cache & Recovery | `macqwen/cache_tools.py` | L101-L151 | nit | verified | fe4682a |
| R3-09 | Phase 3: Persistent Cache & Recovery | `macqwen/cache_tools.py` | L156-L158 | nit | verified | fe4682a |
| R3-10 | Phase 4: Supervision & Failure Recovery | `macqwen/session.py` | L1153-L1157 | pre_existing | verified | 9cac8f5 |

## Detailed Finding Specifications

### R1-01 — `models/flashnext/auto_reclaim.py` (L187-L209)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
`post_request_cleanup` is an after-request cleanup hook, but its prefix-snapshot persistence (`prefix_store.shrink_ram(0)` in the CRITICAL and CONSTRAINED branches) has no exception isolation. Any write failure surfaces as `after_request()['error']`, and the backend turns that into a process-wide lifecycle fault, so every later request is refused until restart. Fix: catch and report non-GPU failures (shrink_ram, clear_cache) inside post_request_cleanup and let only GPU-synchronization failures propagate.
- **Reviewer Reasoning**:
Trigger: FLASHNEXT_AUTO_MEMORY_RECLAIM=1 (`--enable auto-reclaim`, which by itself makes `memory_manager.enabled()` true), plus a write failure while persisting a RAM snapshot. On a 16 GB Mac, available memory under 5.5 GB after a request is common, so the CONSTRAINED branch calls `prefix_store.shrink_ram(0)` after almost every request. That calls `_persist`, then `write_entry` (mx.save_safetensors, fsync, os.replace, a manifest write). Any OSError or RuntimeError there, such as disk full between the free-space check and the write, EIO, or a permission problem on the cache directory, propagates out of `post_request_cleanup`. `MemoryManager.after_request` catches it into `done['error']`. `FlashNextBackend._after_request` (macqwen/backends/flashnext.py:724-726) then calls `pool_lifecycle.mark_lifecycle_fault(...)`, and every later `generate()` raises 'previous memory cleanup failed; restart the process'. Without auto-reclaim, `after_request` never touches the prefix store, and `_release_between_turns` wraps its own `shrink_ram` in try/except without poisoning anything.

**Verification:** normal. Trigger: FLASHNEXT_AUTO_MEMORY_RECLAIM=1 with a prefix store that has an SSD directory, the reclaimer in CONSTRAINED or CRITICAL state, and `write_entry` raising while a RAM snapshot is persisted (EIO, permissions, a removed volume, or ENOSPC between the free-space check and the write).

How it happens:
- In the new `models/flashnext/auto_reclaim.py`, line 188 (CRITICAL) and line 206 (CONSTRAINED) call `evicted = prefix_store.shrink_ram(0)` with no try/except. By contrast, the `set_uncached_reads` call at lines 155-159 is wrapped.
- **Current-code applicability**: Confirmed present in `models/flashnext/auto_reclaim.py:187-209` where `shrink_ram(0)` called during after-request cleanup lacked exception handling, leading `FlashNextBackend._after_request()` to mark a process-wide lifecycle fault poisoning future requests.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r1_01_auto_reclaim_shrink_ram_failure_is_isolated`.
- **Planned fix**: Wrap `prefix_store.shrink_ram(0)` in try/except in `auto_reclaim.py` so disk/cache errors are non-fatal, and restrict lifecycle fault marking in `macqwen/backends/flashnext.py:_after_request` to GPU synchronization and Metal device errors.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r1_01_auto_reclaim_shrink_ram_failure_is_isolated`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R1-02 — `models/flashnext/overnight.py` (L137-L141)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
The supervisor's crash-loop stop can never fire with the shipped defaults, so a server that crashes on every start is restarted forever. `Policy.decide` stops only when more than `max_restarts` (8) restarts fall inside `window_s` (1800 s). Its own backoff spaces restarts at least 20+40+80+160+320+600+600+600 = 2420 s apart, and `wait_until_settled` adds 15 s or more to each gap, so 9 restarts never fit in 30 minutes. Pruning old entries also shrinks `len(self.restarts)`, which resets the cooldown, so nothing ever ends the loop.
- **Reviewer Reasoning**:
Trigger: `./serve.sh --overnight` with defaults, `--max-restarts 8 --window-minutes 30`, and a server that exits non-zero (not 69/70/71/75) on every start, e.g. a startup exception or a /health loss after ready, which returns -9. Restart k only happens after a cooldown of min(600, 20*2^(k-1)) s plus at least one 15 s settle sleep. For 9 entries to be inside the 1800 s window, the 8 gaps between them must each be at least the cooldown for the entries already present (20, 40, 80, 160, 320, 600, 600, 600), which adds up to 2420 s. That is more than 1800, so `len(self.restarts) > self.max_restarts` is never true. Once old entries are pruned, the list settles at about 4 to 5 entries, the cooldown drops back to 160 to 320 s, and the supervisor relaunches the 113 GB model every few minutes all night. The module docstring promises the opposite: 'more than --max-restarts restarts inside --window-minutes -> the supervisor stops and says why (crash loop)'. The unit test passes only because it calls decide() at t=0,1,2,3 and ignores the cooldowns.

**Verification:** normal. Trigger: `serve.sh --overnight` with the default `--max-restarts 8 --window-minutes 30` and a server that exits with a code other than 0/69/70/71/75 on every start. In that case the documented crash-loop stop never fires, and the supervisor keeps restarting the server forever, roughly every 10 minutes.

Mechanism in models/flashnext/overnight.py, which is new in this diff (476 added lines):
- Line 137 `self.restarts.append(now)` runs after `_prune` at line 128 (`now - queue[0] > window`).
- **Current-code applicability**: Confirmed present in `models/flashnext/overnight.py:137-141` where exponential backoffs pushed restarts beyond 1800s window, pruning restarts and preventing `len(self.restarts) > 8` from ever triggering.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r1_02_overnight_crash_loop_detects_consecutive_failures`.
- **Planned fix**: Track `consecutive_failures` in `Policy.decide`, resetting it only after a stable run duration, and trigger crash loop stop when `consecutive_failures > max_restarts`.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r1_02_overnight_crash_loop_detects_consecutive_failures`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R1-03 — `models/flashnext/prefix_session.py` (L213-L220)
- **Phase**: Phase 3: Persistent Cache & Recovery
- **Severity**: normal
- **Reported Problem**:
The new "budgeted" chunk policy cuts at positions that `prefix_session.boundaries()` never predicts. That function only special-cases "balanced" and treats every other policy as grid (`range(chunk, len(prompt), chunk)`). Under budgeted, message anchors shift every later edge off the 2048 grid, so in boundary mode "chunks"/"all" `on_boundary` almost never fires inside a message and the chunk-edge resume snapshots are silently lost.
- **Reviewer Reasoning**:
AGENTIC with `--enable long-context` sets FLASHNEXT_PREFILL_CHUNK_POLICY=budgeted and FLASHNEXT_PREFILL_CHUNK_TOKENS=2048, and the AGENTIC preset sets FLASHNEXT_PREFIX_CACHE_BOUNDARIES=all, whose stated purpose is that a cancelled or crashed prefill resumes. `prefix_session.boundaries()` (prefix_session.py:201-208) adds the grid points 2048, 4096, 6144, … as snapshot boundaries. `_run_chunks` only calls `on_boundary` when `(offset + end) in boundary_set`, and `plan_chunks` (budgeted branch) cuts with `budgeted_edges`, which stops at each message anchor and then continues in 1024/2048 steps from that anchor. Example: a system prompt ending at 2500 followed by a 12K-token user or tool message gives edges 2500, 3524, 4548, … None of them equals 4096, 6144, …, so no snapshot is taken inside the 12K message. Under the grid and balanced policies, boundaries do match the plan's edges. Under budgeted, a cancellation or crash at token 13K in long-context mode resumes from the last message end and redoes up to about 12K tokens of cold prefill (minutes of SSD streaming on this machine).

**Verification:** normal. Trigger: AGENTIC mode run with `--enable long-context`. That turns on the budgeted policy with a chunk size of 2048 (modes.py:98-99), and the AGENTIC preset keeps `FLASHNEXT_PREFIX_CACHE_BOUNDARIES=all` (modes.py:78, commented "message ends + every chunk edge: a cancelled or crashed prefill resumes").

How it happens:
- prefix_session.py:201-208 (`boundaries()`) handles only "balanced" specially. Every other policy, budgeted included, gets `chosen.update(range(chunk, len(prompt), chunk))`, which adds the grid points 2048, 4096, 6144 and so on.
- **Current-code applicability**: Confirmed present in `prefix_session.py:207-220` where `boundaries()` previously fell through to grid `range(chunk, len(prompt), chunk)` under `budgeted` chunk policy, causing intermediate snapshot callbacks to never match `plan_chunks` cuts.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase3_remediations.py:test_r1_03_boundaries_budgeted_policy_aligns_with_budgeted_edges`.
- **Planned fix**: Explicitly check `if prefill_module.chunk_policy() == "budgeted"` in `boundaries()` and evaluate `prefill_module.budgeted_edges(len(prompt), chunk or max(prefill_module.CANDIDATE_CHUNKS), prefill_module.chunk_budget_bytes(), self.anchors(prompt))` to keep snapshot callbacks aligned with actual cuts.
- **Regression test**: `macqwen/tests/test_phase3_remediations.py:test_r1_03_boundaries_budgeted_policy_aligns_with_budgeted_edges`
- **Status**: verified
- **Associated commit**: fe4682a

### R1-04 — `models/flashnext/flight_recorder.py` (L43-L43)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
`_open` calls `os.makedirs(os.path.dirname(self.path))` without checking the result. If FLASHNEXT_FLIGHT_LOG is a bare filename, dirname is '' and makedirs raises FileNotFoundError on every write.
- **Reviewer Reasoning**:
With FLASHNEXT_FLIGHT_LOG=flight.jsonl (a relative bare name, which `resolve_path` accepts), `os.makedirs('', ...)` raises FileNotFoundError. `record()` catches it as OSError and only increments `failures`, so it never opens the file. The flight recorder writes nothing, and a stalled request leaves no lifecycle record, which is the reason this module exists. Fix: skip makedirs when the directory is empty, or resolve the path with os.path.abspath first.

**Verification:** nit. Trigger: an operator sets FLASHNEXT_FLIGHT_LOG to a bare relative filename such as `flight.jsonl`. resolve_path (line 94-95) returns `os.path.expanduser(configured)` unchanged, so nothing makes the path absolute. `_open` then runs `directory = os.path.dirname(self.path)` (line 42), which gives `''`, and `os.makedirs(directory, mode=0o700, exist_ok=True)` (line 43). In CPython, `os.makedirs('')` calls `mkdir('')`, which fails with ENOENT. `exist_ok` only suppresses the error when `os.path.isdir(name)` is true, and `isdir('')` is False, so FileNotFoundError is raised.
- **Current-code applicability**: Confirmed present in `models/flashnext/flight_recorder.py:43` where `os.makedirs(os.path.dirname(self.path))` failed with FileNotFoundError when configured with a bare filename because `dirname` was `""`.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r1_04_flight_recorder_handles_bare_filename`.
- **Planned fix**: Guard `os.makedirs` with `if directory: os.makedirs(directory, ...)`.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r1_04_flight_recorder_handles_bare_filename`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R1-05 — `models/flashnext/expert_cache.py` (L1836-L1844)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
The new capacity guard in `_pool_pass` (and in `ExpertPool.assign`) compares distinct experts against `pool.slots`, which still counts the slots leased to the prefill ring. Under a lease, a routed set larger than the slots actually usable passes the guard, and assign evicts experts it assigned earlier in the same call.
- **Reviewer Reasoning**:
Condition: the prefill ring is leased in the decode pool (non-scratch, up to half the slots), the pool is small, and a decode-shaped (8 tokens or fewer) MoE call runs during the lease. An example is the tiny tail chunk that layer_major.peel_tail runs inside the ring session. Example: 160 slots with 80 leased, and 8 tokens x top-k route about 80 or more distinct experts. That passes `len(wanted) > pool.slots` and assign's own check, but `_free` and the unleased LRU only cover 80. assign then pops LRU entries that were just assigned in this call, so two experts share one slot and the kernel silently uses the wrong expert weights. This is the exact substitution the comment says the guard prevents. Fix: compare against the usable capacity (slots minus the leased range, e.g. len(_free) + len(_slot_of)) in both _pool_pass and assign, and fall back to the streamed path when it is exceeded.

**Verification:** normal: the trigger is narrow but the result is silently wrong. It needs three things together: the prefill ring is on and leased inside the decode pool, the pool is configured below about 0.49 GB (FLASHNEXT_EXPERT_POOL_GB), and a prefill chunk of 8 tokens or fewer runs during the lease.

**How it happens**
- `pool_ring.lease_range` (pool_ring.py:100-109) takes slots out of use. It deletes `_slot_of` entries at or above `first` and filters `_free` to `slot < first`. It does not reduce `pool.slots`.
- The new guard in `_pool_pass` is `if len(wanted) > pool.slots:` (expert_cache.py:1836).
- **Current-code applicability**: Confirmed present in `models/flashnext/expert_cache.py:1836-1844` where `len(wanted) > pool.slots` didn't account for slots leased to prefill ring.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r1_05_pool_capacity_respects_leased_slots`.
- **Planned fix**: Compute `usable = len(pool._free) + len(pool._slot_of)` and guard against `len(wanted) > usable`, safely streaming without slot aliasing or dropped experts.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r1_05_pool_capacity_respects_leased_slots`
- **Status**: verified
- **Associated commit**: 7f0da26

### R1-06 — `models/flashnext/prefill.py` (L392-L394)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: nit
- **Reported Problem**:
The single-span prefill path calls `_prefill_target` without `total_tokens=offset + total`, while the multi-span path passes the absolute prompt end. A run restored from a prefix snapshot therefore picks between full logits and the fused head using only the suffix length, and the cold run uses the whole prompt length.
- **Reviewer Reasoning**:
Trigger: prefix sessions (`prefix_session.py:308` passes `offset`) with the default limit `min(QSA_CHUNK_THRESHOLD=2048, 8192)`. Take a 3000-token prompt with the system boundary at 1000. The cold run splits into the spans [(0,1000),(1000,3000)], and its last chunk gets `total_tokens=3000`, so it uses the fused head. The restored run has offset=1000 and a 2000-token suffix with no boundary inside it. That is one span, so `short_prompt` is computed from `ids.shape[1]=2000 <= 2048` and it takes the full-logits path. The file's own comment says these two paths differ by up to 0.125 in the logits, so a cache hit can produce a different first token than the cold run. This breaks the bit-identical guarantee that `plan_chunks` and the prefix cache rely on, and the base code has no such cold/restored split. A correct fix makes every final call, including the single-span one, decide `short_prompt` from `offset + total`.

**Verification:** nit. Trigger: a prefix-session restore (offset > 0) whose suffix is at most 2048 tokens with no boundary inside it, while the whole prompt is longer than 2048 tokens.

How it happens:
- `_run_chunks` takes the single-span shortcut at prefill.py:415-417 and calls `_prefill_target(language, ids, cache, want_logits, want_hidden, sampler)` without `total_tokens`.
- The multi-span tail at prefill.py:463-464 passes `total_tokens=total + offset`. The layer-major path does the same with `total_tokens=absolute_end` (line 529).
- **Current-code applicability**: Confirmed present in `models/flashnext/prefill.py:417` where single-span restored prefill omitted `total_tokens`.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r1_06_single_span_restored_prefill_passes_total_tokens`.
- **Planned fix**: When `offset > 0`, pass `total_tokens=total + offset` so fused-head vs full-logits path selection matches cold prefill.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r1_06_single_span_restored_prefill_passes_total_tokens`
- **Status**: verified
- **Associated commit**: 7f0da26

### R1-07 — `models/flashnext/expert_pool.py` (L313-L332)
- **Phase**: Phase 5: Lower-Risk Engine Findings
- **Severity**: nit
- **Reported Problem**:
After a pool allocation failure (with FLASHNEXT_CHAT_GUARD unset), for_store stores None in `store._flashnext_expert_pool` and records no refusal. As a result, every later call to `for_store` tries to build `ExpertPool` again. `_pool_usable()` calls it once per MoE layer per decode step, so a failing allocation is retried roughly 48 times per token.
- **Reviewer Reasoning**:
Condition: `ExpertPool(...)` raises MemoryError, RuntimeError or OSError (anonymous mmap or `mx.from_dlpack` failing under memory pressure), and the 1 GB fallback also fails or the target is 1 GB or less. `for_store` then sets `store._flashnext_expert_pool = None`, and the `pool is None and enabled()` guard is true again on the next call. `StreamingSwitchGLU.__call__` -> `_pool_usable()` -> `for_store` therefore repeats the multi-GB mmap and Metal wrap on every MoE layer of every decode token, while the machine is already short of memory. Decode becomes far slower and memory use churns. Before this change the failure was raised once. A correct fix remembers the failure, for example by setting `_flashnext_pool_declined` (which `reset_decline` already clears per request) or a sentinel, so later calls return None at once.

**Verification:** nit. Trigger: with FLASHNEXT_CHAT_GUARD unset and the pool enabled, `ExpertPool(...)` raises MemoryError, RuntimeError or OSError, and either the 1 GB fallback also fails or the target is 1 GB or less.

**Mechanism (verified in the diff).** In expert_pool.py `for_store`, the `except (MemoryError, RuntimeError, OSError)` branch ends with `pool = None` and then `store._flashnext_expert_pool = pool`. It never sets `store._flashnext_pool_declined`.
- **Current-code applicability**: Confirmed present in `models/flashnext/expert_pool.py:313-336` where failed pool allocation set `store._flashnext_expert_pool = None` without marking `store._flashnext_pool_declined`, causing repeated allocation attempts per token.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase5_remediations.py:test_r1_07_expert_pool_failure_marks_pool_declined`.
- **Planned fix**: Set `store._flashnext_pool_declined = "allocation_failed"` when `pool is None` so subsequent calls return None immediately, cleared on `reset_decline(store)`.
- **Regression test**: `macqwen/tests/test_phase5_remediations.py:test_r1_07_expert_pool_failure_marks_pool_declined`
- **Status**: verified
- **Associated commit**: 9f7985c

### R1-08 — `models/flashnext/expert_cache.py` (L1623-L1623)
- **Phase**: Phase 5: Lower-Risk Engine Findings
- **Severity**: nit
- **Reported Problem**:
`_read_weights_with_pool` hard-codes the pool header as the literal `4096` (`4096 + hits[expert] * pool.stride`) instead of using `expert_pool.HEADER`, which `ExpertPool.fill` and `prefill_ring` already use.
- **Reviewer Reasoning**:
The slot byte layout now lives in two places. If `expert_pool.HEADER` ever changes (for example, to page-align on 16 KiB pages), `fill()` writes rows at `HEADER + offset + slot*stride` while this path reads at `4096 + ...`. Pool-reuse hits would then copy wrong bytes into the expert operands with no error, giving wrong weights and wrong outputs. Using `expert_pool.HEADER` (or a pool method that returns a slot's base) removes the duplication.

**Verification:** nit: the claim is accurate. In models/flashnext/expert_cache.py:1622, `_read_weights_with_pool` works out each hit slot's base as `(i, 4096 + hits[expert] * pool.stride)`. The constant already exists as `HEADER = 4096` in models/flashnext/expert_pool.py:42. `ExpertPool.fill` uses it to write slots (expert_pool.py:203, `offset = HEADER + self.offsets[(projection, part)]`), and so does prefill_ring.py (lines 188, 189, 604 and 643, all `expert_pool.HEADER`).
- **Current-code applicability**: Confirmed present in `models/flashnext/expert_cache.py:1622` where `4096` was hardcoded instead of using `expert_pool.HEADER`.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase5_remediations.py:test_r1_08_expert_cache_uses_expert_pool_header_constant`.
- **Planned fix**: Import `expert_pool` and compute slot base with `expert_pool.HEADER + hits[expert] * pool.stride`.
- **Regression test**: `macqwen/tests/test_phase5_remediations.py:test_r1_08_expert_cache_uses_expert_pool_header_constant`
- **Status**: verified
- **Associated commit**: 9f7985c

### R1-09 — `models/flashnext/prefill.py` (L246-L250)
- **Phase**: Phase 5: Lower-Risk Engine Findings
- **Severity**: nit
- **Reported Problem**:
Production prefill code imports a test-bench module (models.flashnext.tests.bench.layer_gate_probe) behind a VEYSTRIX_* env var, in two places in `_prefill_target`.
- **Reviewer Reasoning**:
Cost: the runtime now depends on the tests/bench package being present and importable. A packaged build that excludes tests fails with ImportError mid-prefill whenever VEYSTRIX_LAYER_GATE_PROBE is set. The probe also uses a different env prefix from every other FLASHNEXT_* flag, which makes it easy to miss in settings/modes. Simpler form: an injectable logits hook (like set_route_trace in expert_cache) that the bench installs, so production code imports nothing from tests.

**Verification:** nit. The core claim holds. The diff adds two lazy imports of a test-bench module to production `_prefill_target` in models/flashnext/prefill.py: line 245 (`if os.environ.get("VEYSTRIX_LAYER_GATE_PROBE"): from models.flashnext.tests.bench.layer_gate_probe import record_logits`) and line 266 (the same import before `record_logits(final_logits)`). An injectable callback, like the existing `set_route_trace` at expert_cache.py:261, would let the bench install the hook so prefill imports nothing from tests.

Two of the stated costs are wrong:
(1) A packaged build would not hit ImportError.
- **Current-code applicability**: Confirmed present in `models/flashnext/prefill.py:245,266` where production prefill directly imported a test-bench module.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase5_remediations.py:test_r1_09_prefill_injectable_logits_hook`.
- **Planned fix**: Provide injectable `set_logits_hook`/`get_logits_hook` callbacks and isolate bench test imports with exception handling.
- **Regression test**: `macqwen/tests/test_phase5_remediations.py:test_r1_09_prefill_injectable_logits_hook`
- **Status**: verified
- **Associated commit**: 9f7985c

### R1-10 — `models/flashnext/prefill_scheduler.py` (L39-L39)
- **Phase**: Phase 5: Lower-Risk Engine Findings
- **Severity**: nit
- **Reported Problem**:
`prefill_scheduler.CANDIDATE_CHUNKS` re-declares `prefill.CANDIDATE_CHUNKS` instead of importing it. The prefill copy feeds `budgeted_chunk` and is part of the prefix-cache fingerprint (`chunk_candidates` in prefix_session.store); the scheduler copy is not.
- **Reviewer Reasoning**:
Cost: the budgeted plan's chunk sizes are defined in two places. `_run_chunks` passes the scheduler's cap (chosen from prefill_scheduler.CANDIDATE_CHUNKS) to `budgeted_edges`, which then filters prefill.CANDIDATE_CHUNKS. If one tuple is edited (say a 768 or 4096 candidate is added to prefill.py, which also changes the fingerprint), the live guard steps between caps the planner does not have. A re-plan can then land on a different size than intended without any fingerprint change on the scheduler side. Importing the single tuple from prefill (or prefill_budget) removes the drift.

**Verification:** nit: the duplication is real and a maintainer could reasonably act on it. Today both copies are the same tuple, so nothing breaks.

- `models/flashnext/prefill_scheduler.py:39` declares `CANDIDATE_CHUNKS = (2048, 1024, 512, 256)`.
- `models/flashnext/prefill.py:44` declares the same tuple separately.
- The prefill copy is used by `budgeted_edges` (prefill.py:88, `candidates = [c for c in CANDIDATE_CHUNKS if c <= cap]`) and by the cold plan at prefill.py:159.
- **Current-code applicability**: Confirmed present in `models/flashnext/prefill_scheduler.py:39` where `CANDIDATE_CHUNKS` was defined separately from `models.flashnext.prefill`.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase5_remediations.py:test_r1_10_candidate_chunks_shared_definition`.
- **Planned fix**: Import `CANDIDATE_CHUNKS` in `prefill_scheduler.py` from `models.flashnext.prefill`.
- **Regression test**: `macqwen/tests/test_phase5_remediations.py:test_r1_10_candidate_chunks_shared_definition`
- **Status**: verified
- **Associated commit**: 9f7985c

### R2-01 — `macqwen/server.py` (L928-L930)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
The new opt-in `_log_request` (MACQWEN_REQUEST_LOG) runs inside `complete()`'s try block after a successful generation, with no exception handling around `open(target, "a")` or the write. Any OSError, such as a missing directory, an unexpanded `~` path, a full disk or a permission error, turns a finished generation into a failed request. The `except BaseException` handler also resets the live conversation state. Fix: wrap the logging in its own try/except so a logging failure never fails the request.
- **Reviewer Reasoning**:
Condition: MACQWEN_REQUEST_LOG is set to a path that cannot be opened for append, for example a results directory that was removed, a quoted `~/...` path (no expanduser is applied), or ENOSPC. The bench/smoke scripts and docs use this variable. Every request generates fully, and then `open()` raises. `complete()`'s `except BaseException` calls `session.reset()` and `self.sessions.dropped()` and re-raises, so the client gets a 500 internal_error (or an SSE error) and the output is thrown away. The next request has to rebuild the conversation from scratch, which for a long agentic conversation can mean `recovery_required`. In the base no request log existed, so this failure mode is new. The neighbouring hooks (`warm_module.remember`, the post-turn snapshot) are wrapped and never raise.

**Verification:** normal. Condition: MACQWEN_REQUEST_LOG is set to a path that cannot be appended to (missing directory, a literal `~` path, permission denied, ENOSPC). Every request then generates fully and still fails.

How it happens:
- `_log_request` is new in this diff; the base `macqwen/server.py` has no `_log_request` or `REQUEST_LOG`.
- When the target is not "1", it ends with a bare `with open(target, "a") as handle: handle.write(line + "\n")` at server.py:930-931. Nothing catches errors around it, and nothing expands `~`.
- It is called at server.py:1522, inside `complete()`'s outer `try`.
- **Current-code applicability**: Confirmed present in `macqwen/server.py:928-930` where `_log_request` lacked exception handling and path expansion, failing requests if log write failed.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r2_01_log_request_failure_does_not_abort_generation`.
- **Planned fix**: Wrap `_log_request` in a non-fatal try/except logging a diagnostic warning without failing the completed generation.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r2_01_log_request_failure_does_not_abort_generation`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R2-02 — `macqwen/server.py` (L223-L223)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
New allow-lists reject OpenAI-spec fields that the old code passed through. `_template_tool_call` rejects any tool-call key other than id/type/function (e.g. `index`, which this server itself emits in streamed tool_call deltas). `_normalize_messages` rejects `name` on user, assistant and system messages. Fix: accept and ignore spec-valid fields that do not change rendering, which covers the 2 sites listed. Same pattern at 2 sites (macqwen/server.py:223, macqwen/server.py:192). [also at: macqwen/server.py:243 - `_normalize_messages` now rejects any message field outside {role, content, tool_calls, tool_call_id, reasoning_content}, with `name` allowed only on tool messages.]
- **Reviewer Reasoning**:
Condition: a client keeps the `index` key when it rebuilds tool calls from streamed deltas (the server sends `"index": i` in every tool_calls chunk), or sends `name` on a user or assistant message as the OpenAI spec allows. Every later turn of that conversation gets HTTP 400 "unsupported tool call fields ['index']" or "unsupported message field 'name'". The agent loop stops after its first tool call, whereas the base forwarded `message["tool_calls"]` verbatim and ignored unknown keys.

**Verification:** normal. Trigger: an OpenAI-compatible client sends the next turn with a field the spec allows but the new allow-lists do not accept. One case is `index` left on rebuilt tool calls. The server itself emits it at server.py:2010 (`"index": index, "id": call["id"], "type": "function"`), and stream accumulators such as openai-node's keep it in the final message. The other case is `name` on a user, assistant or system message.
- **Current-code applicability**: Confirmed present in `macqwen/server.py:223` where `name` was rejected in non-tool messages and `index` in tool calls.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r2_02_message_allows_name_field_and_tool_call_index`.
- **Planned fix**: Allow `name` in `MESSAGE_FIELDS` and allow `index` in tool call mappings.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r2_02_message_allows_name_field_and_tool_call_index`
- **Status**: verified
- **Associated commit**: 7f0da26

### R2-03 — `macqwen/server.py` (L419-L419)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
`_validate_call` does `kind in kinds` on the dict `kinds`. When a declared property uses a JSON-Schema type array (e.g. `"type": ["string", "null"]`), that raises TypeError: unhashable type 'list', instead of a ModelToolCallError.
- **Reviewer Reasoning**:
Trigger: a client declares a tool whose parameter has `"type": ["string","null"]` (common in MCP and hand-written schemas), and the model calls that tool. `_parse_tool_calls` → `_validate_call` raises TypeError. That is not a ModelToolCallError, so it escapes even VEYSTRIX_INVALID_TOOL_CALLS=pass mode, the session is reset, and the client gets a 500 internal_error after a full generation. The base did no schema validation and returned the call. Fix: normalise `kind` (handle list or other non-string types) before the dict lookup, so unknown type shapes are accepted or reported as ModelToolCallError.

**Verification:** Severity: normal. This fails whenever a client declares a tool parameter with a JSON-Schema type array (e.g. `"type": ["string","null"]`, which MCP and Pydantic-generated schemas often produce) and the model calls that tool with that parameter.

How it happens, citing macqwen/server.py:
- Line 431: `_parse_tool_calls` stores the client's raw `parameters` in `declared[...]`. `_openai_tools` (lines 169-187) copies the schema as-is and never normalises `type`.
- Line 488 calls `_validate_call` for every call. Inside it, line 413 sets `kind = spec.get("type")`, which here is a list.
- **Current-code applicability**: Confirmed present in `macqwen/server.py:419` where union types in schemas raised unhashable TypeError during dict lookup.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r2_03_validate_call_handles_union_schema_types`.
- **Planned fix**: Handle list/tuple types in schema `type` definitions and support nullable properties before dict lookup.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r2_03_validate_call_handles_union_schema_types`
- **Status**: verified
- **Associated commit**: 7f0da26

### R2-04 — `macqwen/server.py` (L1473-L1476)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
do_POST now sends 404 for unknown routes before reading the request body. The handler is HTTP/1.1 keep-alive, so the unread body is parsed as the next request on the connection.
- **Reviewer Reasoning**:
protocol_version is "HTTP/1.1" and the 404 JSON reply sets Content-Length without closing the connection. A client that POSTs a JSON body to an unsupported path (e.g. /v1/completions or /v1/embeddings) and then reuses the pooled connection has its body bytes read as a request line, so the next real request fails with a 400 or garbled response. On the base branch `_payload()` consumed the body before the route check. Fix: read or discard Content-Length bytes (or set close_connection = True) before answering 404.

**Verification:** normal. Trigger: a client POSTs a body to a path not in `ROUTES` (for example /v1/completions or /v1/embeddings) and then reuses the same keep-alive connection, as pooled HTTP clients do.

How it happens:
- `MacqwenHandler` sets `protocol_version = "HTTP/1.1"` (server.py:1622), so the connection stays open unless the client sends `Connection: close`.
- The new `do_POST` checks `if path not in self.ROUTES: self._error("route not found", 404, "not_found"); return` before it calls `self._payload()`.
- **Current-code applicability**: Confirmed present in `macqwen/server.py:1473-1476` where `do_POST` rejected unknown routes with 404 before reading the request body on HTTP/1.1 keep-alive connections.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r2_04_do_post_drains_payload_on_404`.
- **Planned fix**: Call `self._drain_payload()` to consume `Content-Length` bytes before returning 404.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r2_04_do_post_drains_payload_on_404`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R2-05 — `macqwen/server.py` (L355-L357)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
`_coerce_argument` now handles `array`/`object` XML parameters itself and calls `json.loads(value)` on the raw text. That skips the transport unescape (`_unescape_argument`: `&lt;` `&gt;` `&amp;`) that `coerce_scalar` applies, which is the "escaping exactly once" rule described in the comment just above. Fix: drop the new branch and let `coerce_scalar` handle these types again (it already json-parses and type-checks array/object after unescaping), or unescape before `json.loads`.
- **Reviewer Reasoning**:
In the base, every XML tool parameter went through `coerce_scalar`, which unescapes and then json-parses array/object values (test_server.py:192 pins this rule for strings). After the change, an array/object parameter whose JSON holds an escaped delimiter is still valid JSON, so nothing raises. The escape sequences reach the client unchanged. Example: `<parameter=edits>[{"old":"a &lt; b"}]</parameter>` arrives as `"a &lt; b"`, while a string parameter in the same call decodes to `a < b`. An edit/patch tool then searches for or writes the literal `&lt;`, `&gt;` or `&amp;`, so code gets corrupted silently or the edit fails to match. The agent CLI path (`macqwen.tools.parse_tool_calls`) still decodes these sequences, so the same model output now means different things on the two paths. The `VEYSTRIX_INVALID_TOOL_CALLS=pass` fallback (`arguments[key] = raw.strip()`) also skips the unescape.

**Verification:** normal. Trigger: the model writes a transport-escaped delimiter (`&lt;`, `&gt;`, `&amp;`) inside the JSON of an XML tool parameter declared as `array` or `object`. The new branch in macqwen/server.py:354-364 (`if kind in ("array", "object"): parsed = json.loads(value) ... return parsed`) returns before it reaches `coerce_scalar`. In macqwen/tools/__init__.py, which this change does not touch, `coerce_scalar` starts with `text = _unescape_argument(raw)` (line 106) and only then runs `json.loads(text)` for object/array (lines 140-155).
- **Current-code applicability**: Confirmed present in `macqwen/server.py:355-357` where array/object tool call parameters bypassed XML entity unescaping.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r2_05_unescape_xml_in_tool_call_arguments`.
- **Planned fix**: Unescape XML entities (`&lt;`, `&gt;`, `&amp;`) before parsing JSON array/object arguments and in invalid tool calls.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r2_05_unescape_xml_in_tool_call_arguments`
- **Status**: verified
- **Associated commit**: 7f0da26

### R2-06 — `models/flashnext/prefix_cache.py` (L832-L841)
- **Phase**: Phase 3: Persistent Cache & Recovery
- **Severity**: normal
- **Reported Problem**:
shrink_ram and _evict_ram drop a RAM snapshot whenever an SSD tier is configured, even when `_persist` skipped the write (free disk below FLASHNEXT_DISK_RESERVE_GB, default 50 GB, or ProfiledPrefixCache._make_room refusing). The snapshot is lost, and RAM children whose parent left become unrestorable. [also at: models/flashnext/prefix_cache.py:855 - RAM eviction assumes every entry has spilled to SSD, but `_persist` silently skips the write when free disk is below `min_disk_free_bytes` (default `FLASHNEXT_DISK_RESERVE_GB=50`).]
- **Reviewer Reasoning**:
On a Mac with less than 50 GB free, every `_persist` returns early and only bumps writes_skipped. `_evict_ram` (spill=True because a directory is set) and `shrink_ram` still pop the entry, so it is discarded even though the docstring says nothing is lost. This runs from `_release_between_turns` while the controller is ELEVATED, from reclaim_owned_memory, and on every put over the RAM budget. A resident child whose parent was dropped still shows up in `peek` (the server's admission estimate), but `lookup` hits CorruptEntry, invalidates it and the backend prefills the whole prompt cold. That undoes the cache and the admission estimate exactly when memory and disk are tight. Fix: drop a RAM entry only after confirming `key in self._disk` once `_persist` returns, otherwise keep it (and its ancestors).

**Verification:** normal. Trigger: an SSD tier is configured and `_persist` declines to write, either because free disk is below `min_disk_free_bytes` (default `FLASHNEXT_DISK_RESERVE_GB`=50, prefix_cache.py:664) or because `ProfiledPrefixCache._make_room` returns False (prefix_profiles.py:460-462). In both cases a RAM entry that was never persisted is discarded. prefix_cache.py is a new file in this diff, so the base does not have this problem.
- **Current-code applicability**: Confirmed present in `models/flashnext/prefix_cache.py:795` and `shrink_ram()` where entries were popped from `_ram` before verifying disk write completion.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase3_remediations.py:test_r2_06_evict_ram_preserves_parent_with_resident_child_if_persist_fails` and `test_r2_06_shrink_ram_does_not_drop_unpersisted_entries`.
- **Planned fix**: Ensure `_persist(entry)` is attempted before popping from `_ram`. Only pop from `_ram` if the entry is confirmed present on SSD (`key in self._disk`) or if the entry has no resident children in RAM (`not self._has_resident_child(key)`).
- **Regression test**: `macqwen/tests/test_phase3_remediations.py:test_r2_06_evict_ram_preserves_parent_with_resident_child_if_persist_fails`, `test_r2_06_shrink_ram_does_not_drop_unpersisted_entries`
- **Status**: verified
- **Associated commit**: fe4682a

### R2-07 — `models/flashnext/prefix_profiles.py` (L382-L383)
- **Phase**: Phase 3: Persistent Cache & Recovery
- **Severity**: normal
- **Reported Problem**:
Snapshots stored under another fingerprint or format are never reclaimed. `ProfiledPrefixCache._disk_total` adds their bytes to the budget total, but `_evict_one` refuses to delete them, and `_foreign_victim` is dead code despite the module docstring saying they are evicted first. Once foreign bytes reach the budget, every `_make_room` evicts every unprotected local snapshot and then skips the write, and `_scan`'s `_enforce_budget` does the same at each startup. The base `PrefixCache._scan` (prefix_cache.py:695-696) also never deletes foreign entries, so they pile up on the SSD without limit. Fix: reclaim foreign leaves oldest-first before any local entry, as the docstring promises; this covers both sites (prefix_profiles.py:383/399 and prefix_cache.py:696).
- **Reviewer Reasoning**:
The fingerprint includes hashes of engine source files (sessions._ENGINE_FILES) plus routing profile, chunk policy, ring, KV dtype and layer-major settings, so any engine edit or settings change makes all existing snapshots in the shared `~/.flashnext/prefix-cache` foreign. With `--enable prefix-profiles` (10 GB budget by default), once 10 GB of old-fingerprint snapshots exist, each new snapshot write loops `_evict_one` until every unpinned, unleased current snapshot is deleted. It then returns False and the write is skipped (`writes_skipped`), and the next startup's `_enforce_budget` wipes whatever is left. The SSD tier is effectively disabled for good and only manual deletion recovers it. Without profiles, every fingerprint change adds up to the full disk budget (24-50 GB) of files that are never deleted, until the 50 GB free-space reserve makes every `_persist` skip.

**Verification:** normal. Trigger: `FLASHNEXT_PREFIX_PROFILES=1` is on, and the shared cache directory (default `~/.flashnext/prefix-cache`) already holds old-fingerprint snapshots close to `disk_budget` (default 10 GB). acknowledged in diff: the `_evict_one` docstring says "foreign configuration snapshots are never deleted", and the test `test_entries_of_another_fingerprint_are_not_loaded_or_evicted` pins `foreign_evicted == 0`. Neither one bounds the effect on the budget.
- **Current-code applicability**: Confirmed present in `ProfiledPrefixCache._evict_one()` where `_foreign_victim()` was dead code, and `PrefixCache._scan()` where foreign entries were ignored.
- **Reproduction method**: Demonstrated in `test_prefix_profiles.py:test_entries_of_another_fingerprint_are_not_loaded_or_evicted` and `macqwen/tests/test_phase3_remediations.py:test_r2_07_prefix_cache_evicts_foreign_leaves_first`.
- **Planned fix**: In `ProfiledPrefixCache._evict_one()`, call `self._foreign_victim()` first and evict foreign entries oldest-first before local entries; in base `PrefixCache._scan()`, track foreign entries in `self._foreign` and evict them in `_evict_disk()`.
- **Regression test**: `models/flashnext/tests/unit/test_prefix_profiles.py:SafetyTests.test_entries_of_another_fingerprint_are_not_loaded_or_evicted`, `macqwen/tests/test_phase3_remediations.py:test_r2_07_prefix_cache_evicts_foreign_leaves_first`
- **Status**: verified
- **Associated commit**: fe4682a

### R2-08 — `macqwen/server.py` (L1459-L1469)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
The signature-compat fallback around `backend.generate` catches every TypeError, including one raised inside a running generation, and calls `generate` again on the already-mutated backend. Both real backends (flashnext, frankenstein) and the base protocol already accept `should_cancel` and `on_prefill_progress`, so the fallback only ever fires on internal errors.
- **Reviewer Reasoning**:
Trigger: any TypeError raised from inside prefill or decode (MLX, a sampler, a callback such as the anthropic or responses `on_text`, a layer-gate probe). By then `_generate` has already moved `pending` into `tape` and may have streamed tokens. The retry runs a second generation on that state, so the client gets duplicated or garbage output instead of an error, and the conversation tape becomes inconsistent. The base called `generate` once and surfaced the error. Fix: pick the call signature up front (for example with `inspect.signature`) instead of catching TypeError around the whole call. The same applies to the `on_text` TypeError fallbacks at lines 1415-1491, which re-send a chunk if `on_text` raises TypeError after it has partly written.

**Verification:** normal. Trigger: any TypeError raised from inside a real generation (MLX/sampler/prefill code, the `streamed` callback's fallback `on_text(None if hold else complete)` at server.py:1423, a probe). server.py:1459-1469 wraps `backend.generate(**gen_kwargs, should_cancel=..., on_prefill_progress=...)` in a bare `except TypeError:` and re-calls `generate`, then calls it a third time. All real backends accept both kwargs, so for them the fallback only catches errors raised inside the call.
- **Current-code applicability**: Confirmed present in `macqwen/server.py:1459-1469` where broad `except TypeError:` retry loops retried failed calls on corrupted generator states.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r2_08_signature_inspection_prevents_re_execution_on_type_error`.
- **Planned fix**: Inspect `inspect.signature` upfront and invoke backend with supported kwargs without catch-all retry loops.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r2_08_signature_inspection_prevents_re_execution_on_type_error`
- **Status**: verified
- **Associated commit**: 7f0da26

### R2-09 — `macqwen/server.py` (L1245-L1270)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
_adopt sizes the prefill and enforces the 32K cold-prefill limit using prefix-cache `reusable_tokens()` even when the backend will not use the prefix cache (FLASHNEXT_PREFIX_CACHE off, or an MTP/fused decoder active). An oversized cold prefill is admitted and usage reports false cached tokens. [also at: macqwen/server.py:1251 - `_adopt` and `cache_diagnostics` read the SSD prefix store even when `FLASHNEXT_PREFIX_CACHE` is off. `FlashNextBackend._prefix_session` is always callable, and `PrefixSession.reusable_tokens()` never checks `prefix_session.enabled()`. It builds the `PrefixCache` (which claims and scans… Trigger: the default setup (prefix cache off) with compatible snapshots still on disk…]
- **Reviewer Reasoning**:
`backend._prefix_session` is always callable on FlashNextBackend, and `reusable_tokens` only checks FLASHNEXT_PREFIX_CACHE_RESTORE (default on), not `prefix_session.enabled()`. So with the cache off it still opens ~/.flashnext/prefix-cache, claims and scans it, and peeks for snapshots left by an earlier enabled run. If one matches, `needed_prefill = len(ids) - cached_tokens` falls under the limit and `_check_prefill` passes. But `_generate` only restores when `prefix_session.enabled() and decoder is None`, so it prefills the whole prompt cold, e.g. 60K tokens on a 16 GB Mac, past the limit the check exists to enforce (risking swap or a safety stop). `last_cached_tokens` also makes all three usage formats report cached tokens that were never reused. Fix: in _adopt, consult the prefix cache only under the same condition the backend uses to restore (enabled, restore on, no draft decoder).

**Verification:** Severity: normal. This happens when compatible snapshots sit in the SSD prefix store and the backend will not restore from them on that turn. Two cases trigger it. In the first, FLASHNEXT_PREFIX_CACHE is off (the default) and snapshots remain from an earlier run that had it on. In the second, the cache is on but a decoder is attached: the fused-quality routing profile, or native MTP with greedy sampling.
- **Current-code applicability**: Confirmed present in `macqwen/server.py:1245-1270` where `_adopt` looked up prefix cache even when disabled or draft decoder was active.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r2_09_prefix_session_enabled_guard_before_cache_peek`.
- **Planned fix**: Guard prefix cache token estimation with `prefix_session.enabled()` and `prefix_session.restore_enabled()` and ensure no active decoder.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r2_09_prefix_session_enabled_guard_before_cache_peek`
- **Status**: verified
- **Associated commit**: 7f0da26

### R2-10 — `models/flashnext/prefix_session.py` (L251-L268)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
The new cold-prefill cap (32K, or the memory tier admitted at startup) is only checked against a prediction. `_adopt` in server.py:1251-1261 sizes `needed_prefill` from `PrefixCache.peek()`, and peek's own docstring calls that an estimate, never a promise. When the real `store.lookup()` here misses (CorruptEntry or hash mismatch, UnsupportedCache, an entry evicted or dropped between peek and lookup), `hit` is None and the prefill silently runs the WHOLE prompt cold. Nothing re-checks the cap at execution. Fix: enforce the cold/incremental limit where the real prefill size is known (after lookup, before `prefill_target`), and refuse with recovery_required instead of falling through.
- **Reviewer Reasoning**:
Condition: the prefix cache is on and a stored snapshot covers most of a long prompt (e.g. 180K of a 200K-token conversation). `_adopt` computes needed_prefill = 20K, which passes `_check_prefill`. Then `lookup` fails to restore: the SSD file is corrupt or truncated, verify_hash fails, or the chain was evicted between peek and lookup (memory_manager.before_prefill shrink_ram, the purge monitor's reclaim hook, `_evict_disk`, or the spill skipped on low disk). `lookup` invalidates the entry and returns LookupResult(None, None, STATE_INVALID/RESTORE_FAILED), and `prefill_target` runs all 200K tokens cold. That is exactly what `_check_prefill`'s RecoveryRequiredError message says is 'prohibited on 16 GB hardware' and beyond what prefill_readiness admitted at startup. The likely result is swap growth past the 2.5 GiB hard stop, which ends the server (memory_critical_stop), or a prefill lasting many minutes. The base had no such cap at all.

**Verification:** normal: triggers when the prefix cache is on and a stored entry that `peek` reports cannot actually be restored by `lookup`. For example, an SSD file is truncated, replaced or fails its checksum or token-hash check, or a restore raises UnsupportedCache. The prompt is then prefilled cold at full length, and the new cap never sees that size.

How it happens:
- `server.py:1251` sets `cached_tokens = int(prefix_session().reusable_tokens(ids))`. At 1256, `needed_prefill = max(0, len(ids) - cached_tokens)`, and `_check_prefill` (1261-1270) judges only that predicted value.
- **Current-code applicability**: Confirmed present in `models/flashnext/prefix_session.py:251-268` where failed restore resulted in full cold prefill bypassing memory tier bounds.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r2_10_post_lookup_prefill_limit_enforcement`.
- **Planned fix**: Re-evaluate required cold prefill after `store.lookup()`, raising `RecoveryRequiredError` if cold work exceeds limits.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r2_10_post_lookup_prefill_limit_enforcement`
- **Status**: verified
- **Associated commit**: 7f0da26

### R3-01 — `macqwen/cli.py` (L37-L65)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
`./chat.sh --server` with no `--mode`, the server launch the README documents, now always exits with code 70. The CLI builds a CHAT plan, so `VEYSTRIX_MODE` becomes `chat`, but the new `supervision.check` only accepts `agentic`. Fix: when `--server` or `/server` is given without `--mode`, either use AGENTIC or reject the launch in cli.py with a message that names `--mode agentic`. [also at: macqwen/session.py:1018 - The new supervision.check turns away the documented `./chat.sh --server` and `/server` launches. With no --mode, cli.py builds a CHAT plan that sets VEYSTRIX_MODE=chat, and session.py then exits with code 70 before the model loads.]
- **Reviewer Reasoning**:
A user runs `./chat.sh --server` (README.md:169, 187, 193) or `./chat.sh /server`. In cli.py, `plan_mode(known.mode or modes.CHAT, serve=True)` builds a CHAT plan, and `apply_mode` sets `VEYSTRIX_MODE=chat` and `FLASHNEXT_PREFILL_CHUNK_TOKENS=0`. In session.py:1011-1014, `supervision.check` sees `model=flashnext` with `server=True` and refuses: 'VEYSTRIX_MODE is not agentic (the mode launcher did not run)'. The process exits with 70. On the base branch this command started the API server. The message is also wrong, because the launcher did run. AGENTIC adds `--server` automatically, but `--server` never selects AGENTIC, so the documented command can never succeed.

**Verification:** normal: triggered when a user runs `./chat.sh --server` or `./chat.sh /server` without `--mode`. This is the launch README.md:169/187/193 still documents, and the diff leaves README unchanged.

How it fails:
- macqwen/cli.py:259 rewrites `/server` to `--server`.
- When `model == "flashnext"` and the launch is not a benchmark, cli.py calls `modes.plan_mode(known.mode or modes.CHAT, ..., serve="--server" in remaining, ...)`. With no `--mode`, this builds a CHAT plan with serve=True.
- Only `plan.mode == modes.AGENTIC` adds `--server` and agentic settings. CHAT does not.
- **Current-code applicability**: Confirmed present in `macqwen/cli.py:37-65` where `--server` without `--mode` defaulted to CHAT mode which was refused by `supervision.check`.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r3_01_cli_defaults_server_to_agentic_mode`.
- **Planned fix**: Default `--server` launches to `modes.AGENTIC` when `--mode` is not specified.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r3_01_cli_defaults_server_to_agentic_mode`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R3-02 — `models/flashnext/chat_guard.py` (L121-L124)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
Conditional, on Macs without a battery (Mac mini, Studio, iMac): every default flashnext launch now turns on ChatGuard (modes.COMMON sets FLASHNEXT_CHAT_GUARD=1). ChatGuard treats a missing battery_percent or power_source reading as 'critical probe unavailable', so the process hard-exits before the model loads.
- **Reviewer Reasoning**:
On a desktop Mac, `pmset -g batt` prints no '<n>%;' line, so parse_battery returns percent=None. During ChatGuard.start() the first synchronous tracer.step() calls _reason, which returns 'critical probe unavailable: battery_percent'. _stop then calls os._exit(GUARD_EXIT_CODE), so `./chat.sh` can never load the model on these machines. Base: the default flashnext CHAT launch ran without a guard and worked. Fix: treat 'no battery present' (an AC-only machine) as a valid reading, not a missing probe, and keep fail-closed only for probes that actually fail.

**Verification:** normal. Trigger: any default (non-benchmark) flashnext launch on a Mac with no battery (Mac mini, Mac Studio, iMac, Mac Pro). On these machines `pmset -g batt` prints only "Now drawing from 'AC Power'" and no "<n>%;" line.

How it happens:
- macqwen/cli.py (new in this diff) now calls `modes.plan_mode(known.mode or modes.CHAT, ...)` for every flashnext launch that is not a benchmark, then runs `modes.apply_mode(environment, plan)`.
- The plan env starts with `{**COMMON, ...}` (modes.py:247), and COMMON contains `"FLASHNEXT_CHAT_GUARD": "1"` (modes.py:57-58).
- **Current-code applicability**: Confirmed present in `models/flashnext/chat_guard.py:121-124` where `battery_percent is None` on desktop Macs (Mac mini / Mac Studio / Mac Pro) was treated as missing probe.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r3_02_chat_guard_allows_desktop_mac_without_battery`.
- **Planned fix**: Treat AC-powered desktop Macs (`power_source == "ac"` and `battery_percent is None`) as valid power states rather than missing probes.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r3_02_chat_guard_allows_desktop_mac_without_battery`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R3-03 — `macqwen/cli.py` (L303-L304)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
Every flashnext launch now passes `--profile plan.profile` (CHAT means `plain`). session.main treats that as an explicit flag and saves it to preferences, so a saved `agent` profile is silently overridden and then overwritten. [also at: macqwen/cli.py:56 - Every flashnext launch without `--profile`, including the default `./chat.sh`, now gets the mode's profile (`plain` for CHAT) via `--profile`. session.py treats that as an explicit flag, so it overrides the user's saved profile and then saves `plain` as the new preference.]
- **Reviewer Reasoning**:
On the base, cli.py passed `--profile` only when the user gave one, so session.py used the saved `prefs["profile"]`. Now, for any flashnext launch that is not a benchmark, `plan` is always set (CHAT by default), so `--profile plain` is always passed. session.py:948 (`if args.profile: prefs["profile"] = args.profile`) then runs `preferences.save`. A user whose saved profile is `agent` runs `./chat.sh`, gets the plain profile, and finds the preference rewritten to `plain` for every later run. Likewise, `--mode agentic` saves `agent` permanently for later CHAT runs. A fix should apply the mode profile without persisting it, or only when the user has no saved profile, so the saved preference is left alone.

**Verification:** normal. Trigger: a user whose saved preference is `profile: agent` (set with `--profile agent` or `/profile agent`, session.py:283-286 and 226/260) runs any non-benchmark flashnext launch without `--profile`, for example plain `./chat.sh`.

How it happens:
- cli.py:286-295 builds a plan whenever `model == "flashnext" and (known.mode or not benchmarking)`. Without `--mode`, the mode is `modes.CHAT`.
- modes.py:34 has `PROFILE = {CHAT: "plain", AGENTIC: "agent"}`, and modes.py:295 builds the plan with `ModePlan(mode, PROFILE[mode], ...)`. So `plan.profile` is "plain".
- **Current-code applicability**: Confirmed present in `macqwen/cli.py:303-304` where `--profile` was appended even when not explicitly passed, overwriting user saved preferences.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r3_03_cli_does_not_override_saved_profile_when_unspecified`.
- **Planned fix**: Only pass `--profile` argument if explicitly provided by user.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r3_03_cli_does_not_override_saved_profile_when_unspecified`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R3-04 — `models/flashnext/request_admission.py` (L47-L53)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
With `./chat.sh --enable layer-major` (also `layer-major-4k` and `layer-major-hybrid-4k`) in the default CHAT mode, every request is refused. `request_admission.enabled()` turns the gate on from `FLASHNEXT_PREFILL_ORDER=layer` alone, but `check()` then requires `progress.memory_watch_ready()`, which needs `FLASHNEXT_WATCHDOG=1`. CHAT never sets that flag, and `modes.apply_mode` strips it from the environment because it is an OWNED key.
- **Reviewer Reasoning**:
`plan_mode(CHAT, enable=["layer-major"])` is accepted: only prefix-profiles and long-context are rejected for CHAT. It sets `FLASHNEXT_PREFILL_ORDER=layer`. Only AGENTIC sets `FLASHNEXT_WATCHDOG=1`, and `apply_mode` pops OWNED keys, so even an exported `FLASHNEXT_WATCHDOG=1` is removed for CHAT. On every turn, `backends/flashnext.generate` (lines 557-573) calls `request_admission.check`. `enabled()` is True. `progress.memory_watch_ready()` returns False because `enabled(env)` is False and no watchdog process exists. So the request raises `RequestMemoryRefused` with "independent memory watchdog is unavailable or stale; retry when monitoring resumes". Retrying never helps, and the chat cannot produce a single answer. Fix: gate admission on the condition it depends on (the AGENTIC mode, or a running watchdog), or have `plan_mode` reject the layer-major options for CHAT (or turn the watchdog on for them), so that no accepted mode plan can fail every request.

**Verification:** normal. Trigger: a flashnext CHAT launch (the cli default) with `--enable layer-major`, `layer-major-4k` or `layer-major-hybrid-4k`, and FLASHNEXT_REQUEST_ADMISSION not set to 0. Every request is then refused with RequestMemoryRefused.

How it happens, traced in the code:
- **The gate turns on from the prefill order alone.** In models/flashnext/request_admission.py:47-50, `enabled()` returns True when `FLASHNEXT_PREFILL_ORDER == "layer"` and `FLASHNEXT_REQUEST_ADMISSION != "0"` (default "1").
- **Current-code applicability**: Confirmed present in `models/flashnext/settings/modes.py` and `request_admission.py:47-53` where CHAT mode with `layer-major` prefill enabled was accepted but every request refused due to absent watchdog.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r3_04_chat_mode_refuses_layer_major`.
- **Planned fix**: Disallow `layer-major` in `modes.plan_mode(CHAT)`, and gate `request_admission.enabled()` on `VEYSTRIX_MODE == "agentic"`.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r3_04_chat_mode_refuses_layer_major`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R3-05 — `macqwen/sessions.py` (L154-L189)
- **Phase**: Phase 2: Correctness & Tool-Calling
- **Severity**: normal
- **Reported Problem**:
ReasoningLedger skips assistant turns that had no reasoning (`if not key or not reasoning_content: return`, and the server records only when `extracted_reasoning` is set). `restore_reasoning` then matches turns by position. After one turn without reasoning, the recorded list no longer lines up and restoration stops working for the rest of the conversation.
- **Reviewer Reasoning**:
Say a conversation has a turn sent with thinking off (`reasoning_effort: none`, or `enable_thinking: false`; reasoning is per request) or a turn whose output had no `<think>` text. That turn is never recorded, so the next request carries more assistant messages than `recorded_turns`. `restore_reasoning` returns 0 at line 189, or the content comparison at the shifted index fails. Every later turn in that conversation then has its stripped reasoning left out. The rendered tokens diverge from the cached prefix, and each turn re-prefills cold (minutes on this machine), which is the problem the ledger was added to solve. A correct fix records every generated assistant turn, including ones with empty reasoning, so turn positions stay aligned, or matches recorded turns by content or tool calls instead of by ordinal.

**Verification:** normal: the trigger is any conversation that has at least one assistant turn with no extracted reasoning. That happens when a request turns thinking off (macqwen/reasoning.py resolves `enable_thinking`, `chat_template_kwargs.enable_thinking` and `reasoning_effort: none` per request, lines 119-136), or when a generation produces no `<think>` text.

How it breaks:
- **Turns without reasoning are never recorded.** In macqwen/server.py:1498-1499 the server calls `record_turn` only under `if key is not None and extracted_reasoning:`.
- **Current-code applicability**: Confirmed present in `macqwen/sessions.py:154-189` and `macqwen/server.py` where turns without extracted reasoning were skipped, causing turn misalignment.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase2_remediations.py:test_r3_05_reasoning_ledger_records_empty_thinking_turns`.
- **Planned fix**: Record every assistant turn in `ReasoningLedger` to maintain turn alignment throughout multi-turn conversations.
- **Regression test**: `macqwen/tests/test_phase2_remediations.py:test_r3_05_reasoning_ledger_records_empty_thinking_turns`
- **Status**: verified
- **Associated commit**: 7f0da26

### R3-06 — `macqwen/cache_tools.py` (L114-L140)
- **Phase**: Phase 3: Persistent Cache & Recovery
- **Severity**: normal
- **Reported Problem**:
`clean_cache` works out the set of leaf snapshots once, before it deletes anything (`leaves = [k for k in manifests if k not in parents]`). A parent whose only child it just removed never becomes a candidate in the same run. Prefix snapshots are delta chains (every manifest has `parent`/`parent_tokens`, and AGENTIC sets FLASHNEXT_PREFIX_CACHE_BOUNDARIES=all, so there is a snapshot at every 2048-token edge). So `--target-gb` and `--max-age-days` can only drop the tip of each chain per run and stop far short of the target, while still reporting `status: ok`. Fix: recompute the leaves after each removal and loop until the target is met or no unleased leaf is left, the way `PrefixCache._evict_disk`/`_disk_victim` already does (prefix_cache.py:865-893). …
- **Reviewer Reasoning**:
…The budget total should also count the `.json` sizes.

An AGENTIC user's cache in ~/.flashnext/prefix-cache holds 20 conversations of about 16K tokens, roughly 8 chained snapshots each (160 entries, about 40 GB). They run `python -m macqwen.cache_tools --clean --target-gb 5`. Only the 20 chain tips are leaves, so the loop removes at most those 20, a few GB, and stops because the 140 parents were never in `leaves`. The tool prints `status: ok` with `removed_entries: 20`, and the cache stays around 35 GB, well over the 5 GB the user asked for. `--max-age-days 7` on a cache where everything has expired likewise removes only one layer of each chain per run, so expired snapshots stay on disk. The new tool's documented budget is never reached in one run, and nothing tells the user. The prefix cache's own `_evict_disk` loop does reach its budget.

**Verification:** normal: triggers whenever the prefix cache holds chained snapshots (any entry with a `parent`). That is the usual layout, and AGENTIC makes it denser because models/flashnext/settings/modes.py:78 sets `"FLASHNEXT_PREFIX_CACHE_BOUNDARIES": "all"`.

How it fails, in macqwen/cache_tools.py:
- Line 114 computes `parents` once, and line 119 computes `leaves = [k for k in manifests if k not in parents]` once, both before the loop.
- The loop at lines 128-140 only walks that fixed `leaves` list.
- **Current-code applicability**: Confirmed present in `macqwen/cache_tools.py:114-140` where `leaves` was computed once upfront.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_cache_tools.py:test_clean_cache_iterative_leaf_pruning_on_delta_chains`.
- **Planned fix**: Recompute eligible leaves iteratively inside a loop until `target_bytes` is reached or no unleased leaf remains; count `.json` manifest size in freed bytes.
- **Regression test**: `macqwen/tests/test_cache_tools.py:test_clean_cache_iterative_leaf_pruning_on_delta_chains`
- **Status**: verified
- **Associated commit**: fe4682a

### R3-07 — `macqwen/session.py` (L649-L661)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: normal
- **Reported Problem**:
In _apply_tier, the cold-prefill limit of the memory-admitted tier never takes effect for launcher-started AGENTIC servers. apply_mode always writes VEYSTRIX_MAX_COLD_PREFILL (default 32768), so `env_cold` is always set and wins over `tier.max_cold_prefill`.
- **Reviewer Reasoning**:
Condition: AGENTIC started through `./chat.sh --mode agentic` (the only launch supervision allows) on a machine where tiered admission picks a smaller tier, such as compact (cold prefill 1536) or minimal (1024). modes.apply_mode always sets `VEYSTRIX_MAX_COLD_PREFILL=str(plan.max_cold_prefill or DEFAULT_MAX_COLD_PREFILL)` for AGENTIC and always sets VEYSTRIX_CONTEXT_REQUIRED=1. So _apply_tier takes the `required and current` branch, `max_cold_prefill_from_environment()` returns 32768, and `min(tier.max_cold_prefill, ...)` never runs. serve() is then given max_cold_prefill=32768, not the 1–4K the memory verdict admitted. The server accepts cold prefills far larger than the size admission measured as safe. This goes against the docstring's 'enforce the admitted size' and the stated design that oversized requests are refused, and it leaves only the runtime swap guard to stop the run.

**Verification:** normal. Trigger: a launcher-started AGENTIC server run with the opt-in strict memory policy (FLASHNEXT_MEMORY_POLICY=strict-staged). The candidate's compact/minimal example is mostly wrong, but the same mechanism is reached through another path, given below.

**The mechanism holds.** `modes.apply_mode` (models/flashnext/settings/modes.py:316-317) always writes `VEYSTRIX_MAX_COLD_PREFILL = str(plan.max_cold_prefill or DEFAULT_MAX_COLD_PREFILL)` for AGENTIC.
- **Current-code applicability**: Confirmed present in `macqwen/session.py:649-661` where `_apply_tier` allowed `env_cold` to override the admitted tier's `max_cold_prefill`.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r3_07_apply_tier_caps_cold_prefill`.
- **Planned fix**: Enforce `min(tier.max_cold_prefill, env_cold)` in `_apply_tier()` so runtime cold prefill never exceeds admitted memory tier limits.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r3_07_apply_tier_caps_cold_prefill`
- **Status**: verified
- **Associated commit**: 9cac8f5

### R3-08 — `macqwen/cache_tools.py` (L101-L151)
- **Phase**: Phase 3: Persistent Cache & Recovery
- **Severity**: nit
- **Reported Problem**:
`clean_cache` deletes snapshot `.json` and `.safetensors` files without taking the prefix-cache directory ownership lock (`prefix_cache._claim_directory` and `_with_directory_owner`, which every other file helper uses). It also ignores leases, even though its docstring promises 'unleased leaf snapshots'. Fix: wrap the deletion in the directory-owner claim, or refuse when the lock is held.
- **Reviewer Reasoning**:
A user runs `python -m macqwen.cache_tools --clean --target-gb N` while the AGENTIC server owns `~/.flashnext/prefix-cache`. The tool unlinks leaf snapshots that the server's in-memory `_disk` index and `_leases` still reference, and that the server may be restoring at that moment. The server then fails that restore or falls back to a cold multi-minute prefill. Its disk accounting also drifts from what is actually on disk. `prefix_cache` uses the `.veystrix-owner.lock` flock specifically to stop a second process from changing the directory, and this tool skips it. The tool is new in this change.

**Verification:** nit. Triggering condition: a user runs `python -m macqwen.cache_tools --clean ...` while a flashnext server owns the same prefix-cache directory.

The mechanism holds. `clean_cache` (macqwen/cache_tools.py:91-149, a new file in this diff; the candidate's line 151 is right at the end of it) calls `(target_dir / f"{key}.safetensors").unlink(missing_ok=True)` and `(target_dir / f"{key}.json").unlink(missing_ok=True)` (lines 137-138) without ever calling `prefix_cache._claim_directory`. The prefix cache treats that lock as a hard rule.
- **Current-code applicability**: Confirmed present in `macqwen/cache_tools.py:101-151` where deletions lacked directory ownership locking.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_cache_tools.py:test_clean_cache_respects_directory_ownership_lock`.
- **Planned fix**: Claim directory ownership using `P._claim_directory(target_dir)` before deleting files, releasing it in a `finally` block, and returning error gracefully on `P.CacheOwnershipError`.
- **Regression test**: `macqwen/tests/test_cache_tools.py:test_clean_cache_respects_directory_ownership_lock`
- **Status**: verified
- **Associated commit**: fe4682a

### R3-09 — `macqwen/cache_tools.py` (L156-L158)
- **Phase**: Phase 3: Persistent Cache & Recovery
- **Severity**: nit
- **Reported Problem**:
The documented maintenance command does not match the new tool's parser. docs/flashnext/persistent_cache_design.md:96 says to run `python -m macqwen.cache_tools --cleanup --target-disk-gb 20 --max-age-days 7`, but the parser only defines `--clean` and `--target-gb`. argparse does not accept `--cleanup` as an abbreviation of `--clean`, so the documented cleanup exits with 'unrecognized arguments'.
- **Reviewer Reasoning**:
An operator whose SSD prefix cache has grown follows persistent_cache_design.md:96. argparse exits with status 2 ('unrecognized arguments: --cleanup --target-disk-gb 20') and nothing is cleaned, so the documented way to reclaim cache disk space never works. Fix: either accept the documented spellings (add `--cleanup`/`--target-disk-gb` as aliases) or correct the document, so that the documented command and the parser agree.

**Verification:** nit — the condition is an operator running the cleanup command exactly as written in the design doc. The doc already exists at base 4458d8e5 and is not touched by this diff. Line 96 of docs/flashnext/persistent_cache_design.md reads `.venv/bin/python -m macqwen.cache_tools --cleanup --target-disk-gb 20 --max-age-days 7`. macqwen/cache_tools.py is new in this diff (`git show <base>:macqwen/cache_tools.py` returns "exists on disk, but not in" the base commit). Its parser defines only `parser.add_argument("--clean", ...)` (line 156) and `parser.add_argument("--target-gb", ...)` (line 158).
- **Current-code applicability**: Confirmed present in `macqwen/cache_tools.py:156-158` where argparse lacked `--cleanup` and `--target-disk-gb` flags specified in documentation.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_cache_tools.py:test_cli_argument_aliases`.
- **Planned fix**: Add `--cleanup` as an alias to `--clean`, and `--target-disk-gb` as an alias to `--target-gb` in `argparse`.
- **Regression test**: `macqwen/tests/test_cache_tools.py:test_cli_argument_aliases`
- **Status**: verified
- **Associated commit**: fe4682a

### R3-10 — `macqwen/session.py` (L1153-L1157)
- **Phase**: Phase 4: Supervision & Failure Recovery
- **Severity**: pre_existing
- **Reported Problem**:
The in-chat `/server` path (`session.server_requested`) still calls `serve()` without `supervision.check` and without the new `context_limit`, `max_cold_prefill` and `max_incremental_prefill` arguments. A CHAT session can therefore start the unsupervised, unchunked, unlimited server that the new fail-closed check is meant to refuse. Fix: run the same `supervision.check` here and pass the same limits as the `--server` path at line 1094, or refuse `/server` outside AGENTIC.
- **Reviewer Reasoning**:
A user starts the default CHAT mode (`FLASHNEXT_PREFILL_CHUNK_TOKENS=0`, no context limit) and types `/server` (commands.py:412). On exit from the chat loop, session.py:1153-1157 calls `serve(session, host, port, key, allow_origin)` with no supervision check and `context_limit=None`. The resulting server accepts a large agent-harness prompt and prefills it in one call with no context, cold-prefill or incremental limits. This is the 2026-11-09 incident pattern that the new `supervision.check` at line 1011 was added to block, but only the `--server` path is guarded.

**Verification:** pre-existing. Trigger: a flashnext chat session started with no --mode (CHAT, the default) where the user types `/server` (commands.py:412, which sets `session.server_requested = True` at session.py:233). The mechanism is real. The new fail-closed guard (session.py:1013-1018, `supervision.check(os.environ, model=..., server=bool(args.server), ...)`) only looks at `args.server`. A CHAT launch has `args.server` false, so the check returns None at supervision.py:27 (`if model != "flashnext" or not server ...: return None`) and the model loads.
- **Current-code applicability**: Confirmed present in `macqwen/session.py:1153-1157` where in-chat `/server` invocation bypassed `supervision.check` and launched unlimited server.
- **Reproduction method**: Demonstrated in `macqwen/tests/test_phase4_remediations.py:test_r3_10_server_requested_runs_supervision_check`.
- **Planned fix**: Enforce `supervision.check` and pass bounded context and prefill limits in `/server` handler before invoking `serve()`.
- **Regression test**: `macqwen/tests/test_phase4_remediations.py:test_r3_10_server_requested_runs_supervision_check`
- **Status**: verified
- **Associated commit**: 9cac8f5
