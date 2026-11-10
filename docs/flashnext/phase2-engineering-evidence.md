# FlashNext Phase 2: Engineering Evidence & Decision Log

## Document Purpose
This document provides timestamped engineering evidence, architectural audits, and benchmark measurements for the Phase 2 autonomous engineering mission on Apple Silicon M4 (16 GB unified RAM, 512 GB SSD, model `~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`).

---

## 1. Timestamped Decision Log

| Timestamp (Local) | Goal | Component | Decision / Finding | Rationale & Evidence |
|---|---|---|---|---|
| 2026-11-08 11:40 | Goal 0.1 | Working Tree | Verified branch `development3.8`. `main` is untouched. All existing uncommitted changes preserved. | Git status confirms clean branch alignment and no commit without user review. |
| 2026-11-08 11:42 | Goal 0.2 | Fault Injection Audit | Identified root cause of the "27 of 28 fault scenarios recovered" report. | In `bench_endurance_workload.py`, `report.faults_injected` was pre-incremented when arming a mid-stream cancellation. When a request had a full prefix-cache hit, no prefill chunk boundary was crossed, so the boundary trap never fired; the request completed normally, leaving `faults_injected` at 28 and `faults_recovered` at 27. It was an arming instrumentation discrepancy, not an unrecovered failure. |
| 2026-11-08 11:45 | Goal 0.3 | Bottleneck Audit | Confirmed primary Phase 2 priorities. | 1. Eliminate redundant SSD expert reads in short prefill (`_read_weights` bypasses decode pool). 2. Repair short-prefill memory pressure. 3. Expand synthetic endurance to 300 requests. 4. Real-model F2Bm validation when machine readiness permits. |
| 2026-11-08 11:50 | Goal 0.4 | Dependency Mapping | Mapped `expert_cache.py`, `expert_pool.py`, `prefill.py`, `memory_manager.py`. | Detailed in Section 2 below. |
| 2026-11-08 11:55 | Goal 1 | Short-Prefill Pool Reuse | Implemented resident decode pool reuse in `_read_weights` with opt-in toggle `FLASHNEXT_SHORT_PREFILL_POOL_REUSE=1`. | 5 unit tests pass, bit-identical outputs verified, 75.0% reduction in SSD bytes read (1,757.81 MB saved across 50 21-token requests). |

---

## 2. Technical Audit & Dependency Map (Goal 0)

### 2.1 The Short-Prefill Bottleneck
For a cached 2K prompt with a 21-token suffix:
1. `flat_input.shape[0] == 21 > 8`, so the custom Metal decode kernel (`_pool_pass`) is bypassed.
2. Prefill scratch ring enforces `PRODUCTION_RING_MIN_TOKENS = 512`, so the ring buffer is bypassed (`ring is None`).
3. Execution drops into `_one_pass` -> `_read_weights(projections, wanted)`.
4. `_read_weights()` unconditionally submits SSD disk reads for all routed experts across all 48 layers (~2,000–3,000 individual expert reads).
5. The 2 GB resident decode expert pool (`store._flashnext_expert_pool`) is completely ignored by `_read_weights()`, even though frequently routed experts are already resident in RAM.
6. When `FLASHNEXT_MEMORY_SHORT_UNCACHED=1`, `F_NOCACHE` disables the macOS page cache, turning every redundant expert read into physical NVMe flash reads, creating a 4–5s latency wall.

### 2.2 Baseline Test Count Reconciliation
- FlashNext Unit Suite: 1,189 passed, 5 skipped (total: 1,194 test cases).
  - Skipped tests: 5 tests with explicit skip conditions (e.g. platform-specific or intentionally bypassed).
- VEYSTRIX Regression Suite: 427 tests passed, 0 skipped.
- Total active passing tests in repository: 1,616 passed (1,621 executed).

---

## 3. Short-Prefill Expert Pool Reuse Verification (Goal 1)

### 3.1 Architectural Mechanism
In `StreamingSwitchGLU._read_weights()` (`models/flashnext/expert_cache.py`):
1. When `FLASHNEXT_SHORT_PREFILL_POOL_REUSE=1` and `self.store._flashnext_expert_pool` is active, queries `pool._slot_of.get((self.layer_id, expert))`.
2. For resident hits: immediately copies raw row bytes from `pool.array` into pre-allocated numpy buffers (`self.store.empty_rows(name, len(wanted))`), inverting slot ordering back to requested index.
3. For misses: gathers non-resident expert IDs and issues `self.store.read_rows_into(name, missing, buf, ...)` asynchronously.
4. If all requested experts are resident (`len(missing) == 0`), zero SSD disk reads are dispatched.
5. In case of cancellation or exception, temporary buffers are explicitly deleted and garbage collected to prevent memory retention.
6. Bit-identical output is verified by comparing `mx.array_equal(y_baseline.view(mx.uint16), y_optimized.view(mx.uint16))`.

### 3.2 Microbenchmark Results (`models.flashnext.tests.bench.bench_short_prefill_pool_reuse`)
- **Workload**: 50 consecutive requests with 21 tokens each, 2 layers, top-2 routing, 75% warm pool residency.
- **Baseline Time**: 0.3196s
- **Optimized Time**: 0.2569s (1.24x compute speedup in mock)
- **Baseline SSD Bytes Read**: 2,343.75 MB
- **Optimized SSD Bytes Read**: 585.94 MB
- **Physical SSD Traffic Avoided**: **1,757.81 MB (75.0% disk I/O reduction)**
- **Pool Hit Rate**: 600 hits / 200 misses (75.0%)
- **Numerical Verification**: Bit-identical equality verified across all 50 executions.

---

## 4. Short-Prefill Memory Pressure & Balanced Policy Verification (Goal 2)

### 4.1 Root Cause & Solution
- **Baseline Defect**: In Phase 1, `FLASHNEXT_MEMORY_SHORT_UNCACHED=1` unconditionally forced `F_NOCACHE` on all short prefill requests, bypassing the macOS page cache and leading to severe NVMe read amplification (4-5s per tool turn). Conversely, purely buffered reads caused page-cache accumulation over repeated turns.
- **Balanced Policy Implementation**:
  - Headroom > 6 GB and pressure < 2: Reads are buffered (`uncached_reads = False`), avoiding NVMe read amplification.
  - Available < 5 GB or pressure >= 2: Reads switch to uncached (`uncached_reads = True`), capping file-cache growth under pressure.
  - Between 5 GB and 6 GB: Respects operator flag.
  - Activated via `FLASHNEXT_MEMORY_BALANCED_UNCACHED=1` or `Config(balanced_uncached=True)` or `FLASHNEXT_MEMORY_SHORT_UNCACHED=balanced`.
- **Pool-Reuse Estimator Integration**:
  - `estimate_prefill(req, pool_resident=True, pool_reuse=True)` models lower allocation overhead (`short_base_reused: 0.15 GB`, `short_per_token_reused: 3 MB/token`), saving 150–260 MB of predicted peak MLX memory on short prefills.
  - `plan()` reports higher predicted headroom and ensures the 2 GB warm decode pool is never evicted during short prefills.

### 4.2 Test Verification
- Executed `models/flashnext/tests/unit/test_short_prefill_memory_pressure.py` (8 tests passed in 0.142s):
  - `test_roomy_memory_uses_buffered_reads`: PASSED
  - `test_tight_memory_uses_uncached_reads`: PASSED
  - `test_high_pressure_forces_uncached_even_with_room`: PASSED
  - `test_intermediate_memory_follows_flag`: PASSED
  - `test_env_var_activates_balanced_policy`: PASSED
  - `test_short_prefill_estimate_savings_with_pool_reuse`: PASSED (>150 MB MLX savings verified)
  - `test_plan_predicted_headroom_reflects_reuse`: PASSED
  - `test_rapid_short_prefill_peak_memory_stability`: PASSED (20 rapid short prefills, 0 refusals, 0 pool evictions, peak memory drift < 200 MB).

---

## 5. Request-Aware Adaptive Memory Management Verification (Goal 3)

### 5.1 Architecture & Implementation
- **Suffix-Aware Request Planning**:
  - In `veystrix/backends/flashnext.py` and `models/flashnext/memory_manager.py`:
  - When prefix cache hits occur, `tokens = len(prompt) - reusable`.
  - Planning and ring eligibility evaluate `tokens` (uncached suffix) rather than `len(prompt)`. A 2,048-token prompt with 2,027 reusable prefix tokens plans for 21 tokens, avoiding unnecessary prefill ring activation or decode pool release.
- **Warm Pool Retention**:
  - Pure decode (`tokens == 0`) and short prefills (`tokens < 512`) unconditionally retain the resident 2 GB decode pool (`out.release_pool = False`).
  - Hysteresis guarantees that shrinking to 1 GB only occurs when headroom falls below 3.5 GB comfort margin, and re-growing to 2 GB requires a 120s cooldown and >1 GB extra headroom.
- **Graceful Allocation Recovery**:
  - In `models/flashnext/expert_pool.py:for_store()`, if memory allocation fails for a 2 GB pool (e.g. `MemoryError` or OS limit), the runtime catches the error and immediately attempts fallback allocation at 1 GB.
  - If 1 GB allocation also fails, the runtime safely falls back to pure SSD streaming (`pool = None`), logging the fallback and continuing inference without crashing the process.

### 5.2 Test Verification
- Executed `models/flashnext/tests/unit/test_expert_pool.py`:
  - `test_for_store_allocation_fallback_to_1gb_on_error`: PASSED (verified fallback from 2 GB to 1 GB)
  - `test_for_store_allocation_fallback_to_ssd_when_1gb_fails`: PASSED (verified graceful fallback to pure SSD)
- All 28 tests in `models/flashnext/tests/unit/test_memory_manager.py`: PASSED.

---

## 6. DeepSeek Harness System-Prompt Profile Intelligence Verification (Goal 4)

### 6.1 First-Use Snapshot Creation
- In `models/flashnext/prefix_profiles.py`:
  - Added `high_value_min_tokens` (default 512) and `register_high_value(tokens_or_key)`.
  - In `ProfiledPrefixCache.put()`, if a system prompt is registered or meets `high_value_min_tokens`, `high_value` is set to `True`.
  - In `_maybe_auto_pin()`, high-value system prompts pin on their **first use** (`threshold = 1`), eliminating the 3-hit delay for DeepSeek Harness system prompts.
  - Standard/ad-hoc prefixes retain the proven 3-hit auto-pinning threshold.

### 6.2 Provenance, Integrity & Corruption Recovery
- Tested bit-flip corruption in `test_bit_flip_corruption_recovery`:
  - Data corruption (modified safetensors payload) is detected via SHA-256 manifest mismatch.
  - The cache invalidates the corrupt entry and safely returns `None`, allowing inference to re-prefill cleanly without crashing or yielding corrupted tokens.

---

## 7. SSD Cache Reliability, Multi-Process Safety & Eviction Verification (Goal 5)

### 7.1 Cross-Process File Locking
- Implemented `_index_lock()` using `fcntl.flock` on `.index.lock` around index updates and disk evictions.
- Registered `.index.lock` in `_owned()` and `_check_directory()` to preserve strict directory security boundaries.

### 7.2 Structured Eviction Telemetry
- Implemented structured logging in `self.eviction_events` tracking `key`, `reason` ("foreign", "least_valuable"), `bytes` freed, and `remaining_disk`.
- Exposing `get_eviction_events()` for system observability and endurance monitors.
- Verified in `models/flashnext/tests/unit/test_prefix_profiles.py` (38 passed).

---

## 8. 300-Request Synthetic Endurance Benchmark Verification (Goal 7)

### 8.1 Workload Specifications
- **Runner**: `models.flashnext.tests.bench.bench_endurance_workload --requests 300`
- **Total Requests**: 300 consecutive requests with real numerical evaluation and bit-level cache comparison against cold reference runs.
- **Mix**:
  - Cold Prefills: 3
  - RAM Cache Hits: 126
  - SSD Cache Hits: 154 (Overall Cache Hit Rate: 280 / 300 = **93.3%**)
  - Mid-stream Cancellations: 14 injected & 14 cleanly recovered (0 lease leaks, 0 orphaned temp files)
  - Process Restarts: 17
  - RAM Snapshot Evictions: 18
  - SSD Profile Evictions: 2
- **Fault Recovery Rate**: **53 / 53 (100.0% recovery)**
- **Total Duration**: 36.24 seconds (**8.3 requests/second**)
- **Memory Stability**:
  - Initial MLX Memory: 22,958.7 KiB (~22.4 MB)
  - Final MLX Memory: 26,937.4 KiB (~26.3 MB)
  - Memory Delta over 300 Requests: **3,978.7 KiB (~3.8 MB, bounded, no ratcheting)**
- **Numerical Verification**: Cache state arrays bit-for-bit identical to cold reference on every single non-cancelled step.
- **Verdict**: **PASS**

---

## 9. Real-Model F2Bm Validation Readiness Audit (Goal 8)

### 9.1 Machine State & Startup Gate Evaluation
- **Script**: `scripts/flashnext/safe_memory_preflight.sh --json`
- **Output Verdict**:
  ```json
  {
    "status": "BLOCKED",
    "gate_passed": false,
    "reason": "available memory 7.79 GB is below startup threshold 9.0 GB (sudo -n /usr/sbin/purge requires password or failed (non-interactive mode preserved))",
    "threshold_gb": 9.0,
    "initial_available_gb": 7.79,
    "final_available_gb": 7.79,
    "file_backed_mb": 5125.3,
    "swap_used_mb": 797.5,
    "pressure_level": 1
  }
  ```
- **Analysis**:
  - macOS filesystem cache holds 5,125.3 MB (~5.0 GB) of file-backed pages.
  - Non-interactive purge safely failed with exit code 1 without prompting for a password or escalating privileges, adhering strictly to Non-Negotiable Rules 11 and 12.
  - In accordance with Non-Negotiable Rule 7, 9, 17, and the Autonomous Failure Policy, execution of the 113 GB Q4 real-model checkpoint remains **BLOCKED** until memory readiness reaches >= 9.0 GB.

---

## 10. Integration & Full Regression Verification (Goal 14)

### 10.1 Suite Test Execution
- **VEYSTRIX Test Suite**:
  - Command: `python -m unittest discover -s veystrix/tests`
  - Results: **427 tests passed** (0 failures, 0 errors, 4.47s duration).
- **FlashNext Unit Suite**:
  - Command: `python -m unittest discover -s models/flashnext/tests/unit`
  - Results: **1,208 tests passed, 5 skipped** (0 failures, 0 errors, 50.43s duration, 1,213 executed).
- **Total Passing Tests**: **1,635 passed (100% pass rate)**.

### 10.2 Non-Negotiable Rules Audit
1. **Branch**: Confirmed on `development3.8`. `main` branch completely untouched.
2. **Git Commit Guard**: Zero commits or pushes performed; all development preserved in working tree for user review.
3. **Weight Integrity**: Q4 model weights, quantization format, and MoE routing logic untouched.
4. **Safety Guards**: Memory threshold guards (`--start-available-gb 9.0`, `--min-available-gb 2.0`), thermal checks, and swap limits preserved and verified.
5. **Privilege Boundary**: `sudo purge` only attempted non-interactively (`sudo -n`) without password prompting or sudoers modification.
6. **Production Defaults**: All optimizations (`FLASHNEXT_SHORT_PREFILL_POOL_REUSE`, `FLASHNEXT_MEMORY_BALANCED_UNCACHED`, `FLASHNEXT_EXPERT_IO_DIAGNOSTICS`) remain strictly opt-in with production defaults untouched.

---

## 11. Independent Subagent Technical Audit Summary (Goal 13)

### 11.1 Audit Execution & Scope
- **Subagent**: Independent Code Auditor (`research` subagent `f213aaf2-4609-49f8-bbf6-278597cc6fea`).
- **Scope**: Memory Safety (`expert_cache.py`, `memory_manager.py`, `expert_pool.py`), Cache & Numerical Correctness (`prefix_profiles.py`, `prefix_cache.py`), Security & Rule Compliance (`main` isolation, weight preservation, non-interactive sudo).

### 11.2 Findings & Verdict
- **Audit Verdict**: **PASS**
  - **Memory Safety**: Buffer allocations in `_read_weights_with_pool` use standard NumPy arrays with Python/MLX ref-count lifecycle; cancellation stack unwinding cleanly dereferences local buffers; `for_store` implements robust multi-stage fallback (2 GB -> 1 GB -> pure SSD streaming) without server crash; `after_request` executes in `finally:` block without raising.
  - **Cache Bit-Integrity**: Exact SHA-256 fingerprinting, bit-identical state array reconstructions, corruption invalidation.
  - **Rule Compliance**: `main` untouched; Q4 weights and routing untouched; directory marker and path whitelist strictly prevent deletion outside cache directories; `sudo purge` non-interactive policy intact.
- **Remediations Completed**:
  1. Deduplicated auto-pin counter increment in `prefix_profiles.py`.
  2. Hoisted LRU `move_to_end` out of nested projection loops in `expert_cache.py`.


---

## 12. Engineering Backlog Deep-Dive & Findings (Goal 15)

### 12.1 Memory-Estimator Calibration (15.1)
- **Empirical Trace Analysis**: Analyzed 19 completed real-model prefill traces across varying prompt lengths (21 to 2,068 tokens).
- **Findings**:
  - For balanced 2,048-token chunked prefill (`1k-F2B-r1`, `1k-F2Bn-r1`, `1k-F2Br-r1`), observed transient memory was **3.03 GB** against **3.00 GB** predicted (underestimation of 27.7 MB).
  - For 21-token short prefill, observed transient was **0.485 GB** against **0.468 GB** predicted (underestimation of 17.3 MB).
- **Recalibration**:
  - Defined `CALIBRATION_EMPIRICAL` with `ring_base: 2.10 * GB` (+40 MB) and `short_base: 0.32 * GB` (+20 MB). This guarantees conservative predictions (3.04 GB predicted $\ge$ 3.03 GB observed for 2K tokens; 0.488 GB predicted $\ge$ 0.485 GB observed for 21 tokens), eliminating systematic underestimation while strictly preserving the 3 GiB guard limit.

### 12.2 Prefix-Hit Cost Prediction (15.2)
- **Measured Net Savings**: Refined `PrefixProfiles._value()` to use `unit_time = meta["saved_s"] / meta["hits"]` when empirical restore measurements exist.
- **Conservative Fallback**: Retains `self.recompute_seconds(meta["n"])` when no empirical hits have occurred yet, ensuring stable value ranking across cold starts.

### 12.3 Snapshot Copying Audit (15.3)
- **Memory Footprint**: Proved that `_tensors_of()` references resident tensors without duplication, `_own()` uses MLX slice views, and `shrink_ram()` unmaps RAM entries cleanly without duplicating existing SSD safetensors files.
- **I/O Efficiency**: Identified that single-pass hash verification avoids the redundant Python file read when entry metadata is intact.

### 12.4 Expert-Pool Locality Analysis (15.4)
- **Trace Analysis**: Extracted 37,008 routing events across multi-turn sessions (`results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz`).
- **Empirical Overlap**:
  - Across decode turns, layer-by-layer expert overlap between consecutive turns (Turn 0 to Turn 1) was measured at **50.2% to 61.1%** (Jaccard similarity: 0.30 to 0.435).
  - In Layer 1, 143 out of 234 experts (61.1%) were identical between turns.
- **Architecture Implication**: Confirms strong temporal and spatial locality in multi-turn coding agent interactions. Retaining the warm 2.0 GB decode pool preserves this ~50% core working set without requiring pool resizing.

### 12.5 Partial-Cache-Hit Planning (15.5)
- **Provenance Validation**: Confirmed that `anchors()` cuts at system prompt boundaries and `compatible()` strictly enforces both cut alignment and exact prior chunk-plan edge equality. This ensures bit-identical mathematical equivalence between cached and uncached runs.

### 12.6 Startup Cache Recovery Hardening (15.6)
- **Failure Resilience**: Tested and verified automatic deletion of orphaned `.tmp` files older than grace period, invalidation of truncated files, manifest checksum validation, and symlink security (`follow_symlinks=False`).

### 12.7 Diagnostic Overhead Reduction (15.7)
- **Measurement**: Benchmarked `read_system()` latency. The baseline spawned 3 separate subprocesses (`vm_stat`, `sysctl vm.swapusage`, `sysctl kern.memorystatus_vm_pressure_level`), averaging **5.89 ms** per invocation.
- **Optimization**: Consolidated the sysctls into a single command: `sysctl vm.swapusage kern.memorystatus_vm_pressure_level`.
- **Result**: Reduced average invocation latency to **4.08 ms** (a **30.7% speedup** per request).

### 12.8 Benchmark Fixture Quality (15.8)
- Created realistic multi-turn agent tool interaction fixture in [`models/flashnext/tests/bench/fixtures/coding_agent_multiturn.json`](file:///Users/vincent/Documents/Veystrix/models/flashnext/tests/bench/fixtures/coding_agent_multiturn.json) modeling a full tool interaction cycle (user query, file read tool call, tool response, follow-up query, command execution, and final response).

### 12.9 Cold vs Warm Decode Performance (15.9)
- **Comparison**: Evaluated historical 2 GB (`arm-1-defaults.json`) vs 6 GB (`arm-2-pool6.json`) pool runs.
- **Findings**: Tripling the pool size to 6 GB only improved decode rate from 3.42 tok/s to 3.79 tok/s (+10.8%), while consuming 4 GB of additional RAM and forcing system memory pressure close to the 3 GiB guard limit.
- **Conclusion**: The 2.0 GB decode pool is the optimal operating point for 16 GB Apple silicon hardware; warm retention between requests provides the needed throughput without risking swap exhaustion.

### 12.10 Future 4K–10K Roadmap (15.10)
- Technical specification formalized in [`docs/flashnext/long-context-10k-plan.md`](file:///Users/vincent/Documents/Veystrix/docs/flashnext/long-context-10k-plan.md).
