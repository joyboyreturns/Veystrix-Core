# FlashNext: Autonomous Memory Engineering & Continuous Operation

This document serves as the technical reference for autonomous memory management, startup recovery, prompt cache persistence, and continuous multi-request operation on Apple Silicon 16 GB configurations.

## System Model & Constraints
- Device: Apple M4 MacBook Air, 16 GB unified RAM, fanless.
- Checkpoint: `Qwen3.8-Flash-Next-MLX-4bit-MTP` (~113 GB unquantized / Q4 streamed MoE weights).
- Base startup memory requirement: 9.0 GB available RAM.
- Runtime safety floor: 2.0 GB available RAM.
- Compressor guard: Growth > 2.0 GiB accompanied by available memory < 3.0 GiB triggers controlled stop.

## 1. macOS Memory Recovery & Startup Readiness (Goal 1)

### 1.1 Root Cause of Startup Memory Blockers
On macOS Apple Silicon unified memory systems, `vm_stat` breaks down 16 GB into:
1. **Free + Inactive + Speculative (Available Memory)**:
   - Inactive pages are cached file data and inactive process pages.
   - The Veystrix launch gate defines available memory as `free + inactive + speculative` pages.
2. **File-Backed Cache Churn**:
   - Typically 3.5–5.0 GB of RAM is consumed by macOS unified buffer cache (`file_backed_mb`).
   - Although reclaimable by the kernel under active memory pressure, launch-time readiness checks require 9.0 GB available before initiating weight loading.
3. **MLX Non-Expert Footprint**:
   - Base weights consume 2.79 GB of MLX active memory.
   - When loading weights from disk, available memory temporarily drops by ~3.0 GB (from ~11.3 GB down to ~8.2 GB).
   - Hence, a launch gate of 9.0 GB ensures that after weight loading, the machine retains > 5.0 GB for prefill scratch buffers and decode pools.

### 1.2 Safe Purge Policy & Frequency Guard
The script `scripts/flashnext/safe_memory_preflight.sh` enforces:
- Non-interactive execution (`sudo -n /usr/sbin/purge`).
- Strict mission limit: At most **two** successful purges per mission session.
- Minimum 30-minute settling window between purges (`/tmp/flashnext_purge_state.json`).
- Prohibition during active prefill, decode, or competing inference workloads.
- Settle delay (3 seconds) followed by re-sampling of memory counters.
- Transparent reporting of cold-cache side effects (subsequent SSD reads incur physical I/O latency).
- Graceful non-interactive degradation: If `sudo -n` demands credentials, the limitation is reported in structured JSON without hanging or prompting.

## 2. Launch Gate vs. Runtime Safety Guard Semantics (Goal 2)

### 2.1 The Launch / Runtime Threshold Separation
During the prior run (`arm-2k-F2Bm-r1`), passing `--min-available-gb 9` to the orchestrator caused the child process to inherit 9.0 GB as its runtime stop floor. Because base weights loading naturally consumes ~2.79 GB of RAM (causing available memory to settle around 8.19 GB), the child stopped at phase `prompts` before any prefill occurred.

The correct architecture decouples these two distinct thresholds:
- **Startup Launch Gate (`--start-available-gb 9`)**:
  Evaluated exclusively at startup before child process invocation. Ensures the system has ample headroom before loading model weights.
- **Runtime Safety Floor (`--min-available-gb 2`)**:
  Enforced by the child process tracer at every 0.25 s / 1.0 s sample interval. Stops inference if available memory falls below 2.0 GB (preventing macOS freeze or OOM kill).
- **Argument Forwarding Protection**:
  `diagnostic_args()` explicitly filters `--start-available-gb` out of child process arguments, ensuring the child strictly receives `--min-available-gb 2.0`.
- **Preflight Log Transparency**:
  `check_ready()` explicitly lists both the launch gate and runtime stop floor in the preflight summary.

## 3. Short-Prefill Latency & Decode Optimization (Goal 6)

### 3.1 Investigation of the 4–5 Second TTFT Delay on 21-Token Cached Prompts
When a 2K prompt has an exact system-prefix cache hit, only the 21-token suffix needs prefilling before generating the first token. Despite only 21 tokens being processed, TTFT currently takes 4–5 seconds.

### 3.2 Diagnostic Root Causes
1. **Bypass of Custom Metal Decode Kernel**:
   - In `expert_cache.py`:
     ```python
     custom = (scores is not None and flat_input.shape[0] <= 8 and self.metal_combines_scores)
     ```
   - For 21 tokens, `flat_input.shape[0] == 21 > 8`, disabling `_pool_pass`.
2. **Ring Minimum Token Guard (`PRODUCTION_RING_MIN_TOKENS = 512`)**:
   - The prefill ring is gated on `tokens >= 512`. For 21 tokens, `ring` is `None`.
3. **Absence of Decode Expert Pool Reuse in `_read_weights`**:
   - Falling back to `_one_pass` calls `self._read_weights(projections, wanted)` for all 48 layers.
   - `_read_weights` issues disk reads for every distinct expert routed across the 21 tokens (~30–60 experts per layer).
   - Across 48 layers, this issues ~2,000–3,000 distinct expert read requests to SSD, completely bypassing the 2 GB decode pool resident in memory.
4. **Buffered vs. Uncached Reads (`FLASHNEXT_MEMORY_SHORT_UNCACHED=1`)**:
   - With `FLASHNEXT_MEMORY_SHORT_UNCACHED=1`, `F_NOCACHE` is enabled on the file descriptors to avoid polluting macOS filesystem cache.
   - While this protects RAM from page-cache inflation, uncached reads force every single read to hit physical NVMe flash with zero kernel readahead, resulting in ~4–5 seconds of cumulative SSD I/O latency.
5. **Temporary Host Buffer Overhead & MLX Synchronization**:
   - Each expert read allocates host buffers, reads row bytes, and converts them into MLX arrays via `to_mx`, creating transient allocation churn and GPU synchronizations across 48 layers.

### 3.3 Opt-in Optimization Opportunities
- **Decode Pool Lookup in `_read_weights`**: If an expert is already resident in `store._flashnext_expert_pool`, copy or slice its weights from `pool.array` / `pool.buffer_mx` instead of hitting disk.
- **Selective Caching**: When available memory is high (> 6 GB), keep buffered reads for short prefills; only enforce `short_uncached` under tight memory pressure.
- **Lowering Ring Minimum for Medium Suffixes**: Allowing the ring to stage batches down to 64 tokens when memory headroom permits.
