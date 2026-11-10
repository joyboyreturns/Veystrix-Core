# Runtime recovery progress

Updated 2026-11-10. Branch `development3.8`, starting HEAD `93e5312478834dd1ba964fcdb5b3aff8cdeb8b5b`. Objective: restore real DeepSeek Harness and Qwen Code workflows, durable prefix reuse, safe 32K admission and 262K logical context, without weakening model fidelity or safeguards.

## Evidence and changes

The reported 9,230/9,427 progress is **not confirmed completed tokens**: `FlashNextBackend._generate.layer_completed` estimates it from the number of distinct streamed layers. Therefore the evidence does not locate a final 197-token stall. The actual blocking operation and swap growth remain undiagnosed. Added flushed flight-recorder markers for chunk plans and boundaries, forward/evaluation and final projection, snapshot boundaries, backend return and decode entry. MLX is lazy: forward return does not prove GPU execution completed. Layer-major and expert-fetch detailed tracing, memory samples, and HTTP-forward markers still need work.

`PrefixCache._evict_ram` previously discarded an unpersisted leaf after a failed spill; it now retains the only copy and searches for another durable victim. Regression checks presence and accounting. **Open safety issue:** retention may exceed the RAM cache budget if persistence repeatedly fails. Design an explicit pressure-triggered invalidation/reporting policy before treating this as production ready.

Agentic startup now assesses the configured context and cold prefill using a configured tier only when chunk size is positive and at most 2048, with the budgeted KV growth estimate. This prevents the unrelated legacy tier from silently replacing 262144/32768. Actual per-request safety, diagnostics, all launch paths and incremental admission still require audit and real validation. CI push trigger includes `development3.8`.

## Validation and status

- Targeted Metal tests: 87 passed (phase 3 remediation, prefill chunks, prefix cache); readiness and phase 4: 42 passed.
- Sandboxed MLX imports fail: no Metal device; outside sandbox the targeted tests passed. `pytest` is not installed; use unittest.
- Full MacQwen suite: 725 tests passed in 69.195 seconds. Full FlashNext unit suite: 1809 tests passed in 169.589 seconds, 3 skipped.
- No full checkpoint, harness, Qwen Code, persistence restart, or 32K hardware run has been performed in this session. No claim of stall resolution or overnight readiness.
- `qwen --version`: 0.27.0. `dsh` was not found on the current PATH. Server status reports stopped after previous guard stop. System swap sampled at 1140.81 MiB; machine on battery at 77%. `serve.sh` calls for AC power; defer heavy tests until AC and safety headroom are confirmed.
- Reliable Codex five-hour usage telemetry is unavailable.

## Next

Review the candidate changes and full test outcomes; implement bounded explicit cache invalidation and reporting. On AC, use the guarded `./serve.sh` launch, inspect `/health` and diagnostics, and capture real stage traces progressively (short, 2K, 4K, 8K, representative 9K), stopping on pressure or stall. Trace exact layer/expert/Metal stage and fix only with evidence. Then run real DeepSeek and Qwen tool workflows, restart and restore a verified SSD prefix, validate admission/error paths and all mandatory gates. Do not run a 32K full-model stress test automatically. See `HANDOFF_TO_CLAUDE_CODE.md` for commands and open work.

Last previously documented launch interface: `./serve.sh`; no launch was verified in this session.
