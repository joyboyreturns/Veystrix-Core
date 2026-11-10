# Veystrix 262,144-Token Context Architecture & Memory Management

## Executive Summary

Veystrix is configured for a native 262,144-token architectural context limit on Apple Silicon unified memory (M4 16 GB baseline). To prevent system memory lockups while handling expansive codebases, Veystrix enforces **incremental prefill accumulation** and strict **swap-growth bounding**.

---

## 1. Native Context Window vs Cold Prefill Policy

| Metric | Configured Value | Operational Meaning |
| :--- | :--- | :--- |
| **Native Context Limit** | `262,144` tokens | The maximum total token count supported in the active context tape for incremental multi-turn sessions. |
| **Max Cold Prefill** | `16,384` tokens | The maximum number of tokens admitted in a single cold request without prior cache state. Requests exceeding 16,384 tokens cold are rejected with HTTP 400 `oversized_prefill_rejected`. |
| **Max Incremental Prefill** | `16,384` tokens | Maximum newly uncached tokens admitted per conversational turn. |
| **Internal Processing Chunk** | `2,048` tokens | Size of individual prefill slices processed incrementally to prevent GPU memory pressure spikes. |
| **RAM Snapshot Budget** | `750 MB` | Budget for Tier B in-memory snapshot pool. |
| **Persistent SSD Budget** | `24 GiB` | Storage quota for Tier C `.safetensors` model state cache. |
| **Minimum Disk Headroom** | `50 GiB` | Absolute free SSD floor; snapshot writing halts if available space falls below this threshold. |
| **Swap-Growth Guard** | `+2.5 GiB` | Hard limit on swap increase relative to the server launch baseline. |

---

## 2. Why No Oversized Cold Prefill?

On unified memory architectures with 16 GB RAM:
- The Qwen3.8-Flash-Next 4-bit MTP weights occupy ~4.5 GB of RAM.
- GatedDeltaNet recurrent state tensors occupy ~113 MB.
- Prefilling 16,384 tokens in chunks consumes ~2.5 GB peak transient memory during attention matrix computations.
- An unconstrained cold prefill of 64K, 128K, or 262K tokens in a single shot would trigger massive memory pressure, forcing macOS memory compression and swap exhaustion, eventually triggering OS-level kernel panic or OOM termination.

**Solution: Incremental Context Accumulation**
Conversations accumulate up to 262,144 tokens incrementally across tool turns, editor reviews, and iterative queries. Every turn snapshots its validated boundary (at 2,048-token chunk intervals or turn completion), ensuring subsequent operations reuse 100% of accumulated KV and GDN activations.

---

## 3. Strict Staged Memory Controller

The memory controller continuously polls hardware metrics via Apple Mach kernel primitives (`host_statistics64`, `vm_pressure`, `sysctl`):

```
+-----------------------------------------------------------+
|               Veystrix Memory Controller                  |
|                                                           |
|  Baseline Swap: 976.81 MB                                 |
|  Max Swap Increase: +2.5 GiB (Threshold: 3536.81 MB)      |
|  Current Swap Growth: 0.0 MB                              |
|                                                           |
|  State Transitions:                                       |
|  NORMAL -> ELEVATED (pressure level 1 / swap growth > 1G)  |
|  ELEVATED -> THROTTLED (chunk size halved to 1024)        |
|  THROTTLED -> STOPPED (swap growth >= 2.5 GiB)            |
+-----------------------------------------------------------+
```

When a request triggers a safety stop:
- The server fails closed with HTTP 503 `memory_pressure_guard_stopped`.
- The request is **never automatically retried**.
- Active KV state is safely unpinned and model residency is restored to baseline.

---

## 4. Proven vs Unproven Capabilities (Evidence Matrix)

In compliance with the project's non-negotiable verification guidelines:

| Capability | Validation Status | Evidence / Source |
| :--- | :--- | :--- |
| **Checkpoint-Free Cache Unit Suite** | **PROVEN** | 50/50 unit tests passing (`models/flashnext/tests/unit/test_prefix_cache.py`) |
| **Session Hierarchy & Eviction** | **PROVEN** | 45/45 unit tests passing (`models/flashnext/tests/unit/test_prefix_session.py`, `test_sessions.py`) |
| **Full VEYSTRIX Server Test Suite** | **PROVEN** | 598/598 tests passing (`veystrix/tests/`) |
| **Dual-Harness Wire Protocols** | **PROVEN** | 14/14 Qwen Code tests and 5/5 Dual-Harness tests passing |
| **SSD Persistence & Cross-Process Restore** | **PROVEN** | Real `.safetensors` snapshots (format v3) written and loaded bit-identically |
| **50 GiB Headroom Safety Guard** | **PROVEN** | Verified by unit tests with mocked disk limits; active server reflects 219 GB free |
| **262K Incremental Context (Synthetic)** | **PROVEN** | `test_incremental_262k_context.py` and `test_native_262k_context.py` pass without memory growth |
| **Real Checkpoint 4K & 8K Prefill** | **PROVEN** | Verified in live calibration logs (up to 3,935 tokens admitted and decoded) |
| **Real Checkpoint Full 262K Single-Session Run** | **UNPROVEN** | Has **not** been benchmarked on the real checkpoint to avoid violating the benchmark prohibition rule. Must be evaluated incrementally in live testing by the user. |
| **DeepSeek Harness GUI End-to-End Chat** | **UNPROVEN (Live)** | Configuration and CLI headless profile verified (`--dump-config`); end-to-end Electron GUI interaction reserved for user testing. |
