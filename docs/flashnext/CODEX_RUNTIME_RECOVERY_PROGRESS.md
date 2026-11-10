# Runtime recovery progress

Updated 2026-11-10. Branch `development3.8`, starting HEAD `93e5312478834dd1ba964fcdb5b3aff8cdeb8b5b`. Objective: restore real DeepSeek Harness and Qwen Code workflows, durable prefix reuse, safe 32K admission and 262K logical context, without weakening model fidelity or safeguards.

## Evidence and changes

The reported 9,230/9,427 progress is **not confirmed completed tokens**: `FlashNextBackend._generate.layer_completed` estimates it from the number of distinct streamed layers. Therefore the evidence does not locate a final 197-token stall. The actual blocking operation and swap growth remain undiagnosed. Added flushed flight-recorder markers for chunk plans and boundaries, forward/evaluation and final projection, snapshot boundaries, backend return and decode entry. MLX is lazy: forward return does not prove GPU execution completed. Layer-major and expert-fetch detailed tracing, memory samples, and HTTP-forward markers still need work.

`PrefixCache._evict_ram` previously discarded an unpersisted leaf after a failed spill; it now retains the only copy and searches for another durable victim. Regression checks presence and accounting. **Open safety issue:** retention may exceed the RAM cache budget if persistence repeatedly fails. Design an explicit pressure-triggered invalidation/reporting policy before treating this as production ready.

Agentic startup now assesses the configured context and cold prefill using a configured tier only when chunk size is positive and at most 2048, with the budgeted KV growth estimate. This prevents the unrelated legacy tier from silently replacing 262144/32768. Actual per-request safety, diagnostics, all launch paths and incremental admission still require audit and real validation. CI push trigger includes `development3.8`.

## Validation and status

Continuation: on AC, guarded real checkpoint run on 2026-11-10. Initial supported `./serve.sh --pool-gb 0` exposed `swap_ceiling_mib=0` on `/health`; it was stopped without an inference request. Agentic mode now sets a 3072 MiB ceiling, at least 256 MiB margin and independent memory watchdog; a stricter configured ceiling/margin is preserved. Restarted server PID 91731 reported 3072/256 in `/health`. Baseline swap 1060.81 MiB. Cold requests returned streamed real replies: 469 prompt tokens, `READY`, first token 26.836 s; 1974 tokens, `OK`, first token 49.093 s, peak system swap about 1541 MiB; 3974 tokens, `OK`, first token 79.836 s, swap around 1085 MiB at end. Both longer requests used a seven-token final chunk; 3974's final long-prompt projection and state evaluation completed in 2.431 s. These are synthetic prompts and do not diagnose or resolve the real 9K Harness stall. A 7974-token cold request was cancelled after 3984 evaluated tokens when pressure level 2 persisted, available memory fell to 3459 MiB and compressor growth reached about 5 GiB. Peak swap growth was 827 MiB. Cancellation and subsequent supervisor shutdown were clean. Flight log: `~/.flashnext/logs/server-91731.jsonl`; safety trace: `/var/folders/37/fn6fxqyd2rgfmvc4954dggbm0000gn/T/veystrix-chat-91731.jsonl`. No 8K reply or 9K test was achieved.

New candidate fixes: token progress now reports completed, evaluated chunk boundaries instead of layer-derived token estimates; layer callbacks still feed watchdog activity. A failed prefill marks partially advanced state for replay. SSD child snapshots cannot be declared durable unless their parent is durable. Targeted suites: 145 tests passed; mode/CLI tests 35 passed. The first combined watchdog run had test-code errors (missing mock import and missing known flags), subsequently corrected; watchdog tests themselves did not report a failure in that run. `git diff --check` passed before these latest edits.

Cache-chain correctness: the 7974 run's 1992-token snapshot was automatically parented to an older 1984-token snapshot computed with a different chunk plan. This is not safe to restore as exact state. PrefixSession now disables automatic parent selection for newly computed roots and skips chaining from a token-identical snapshot with conflicting plan. `snapshot_chain_semantics=2` changes the fingerprint so older snapshots cannot be silently trusted; their files have not been deleted. The failed original request left a 9024-token manifest and ancestors (e.g. fingerprint `95575b3c…`), but their state is not proven valid and the new fingerprint intentionally prevents reuse. Test suite covering prefix session/cache/profiles: 148 passed. Fingerprint migration and quota cleanup of foreign entries require further review before declaring user caches protected.

- Targeted Metal tests: 87 passed (phase 3 remediation, prefill chunks, prefix cache); readiness and phase 4: 42 passed.
- Sandboxed MLX imports fail: no Metal device; outside sandbox the targeted tests passed. `pytest` is not installed; use unittest.
- Full MacQwen suite: 725 tests passed in 69.195 seconds. Full FlashNext unit suite: 1809 tests passed in 169.589 seconds, 3 skipped.
- Full-checkpoint synthetic runs are described above. No real Harness or Qwen Code workflow, real SSD restart restoration, or 32K hardware run has passed. No claim of stall resolution or overnight readiness.
- `qwen --version`: 0.27.0. `dsh` was not found on the current PATH. AC was connected for the real run; the server is now stopped. Avoid further heavy tests until the pressure behavior is understood and headroom is confirmed.
- Reliable Codex five-hour usage telemetry is unavailable.

## Next

Review the candidate changes and full test outcomes; implement bounded explicit cache invalidation and reporting. On AC, use the guarded `./serve.sh` launch, inspect `/health` and diagnostics, and capture real stage traces progressively (short, 2K, 4K, 8K, representative 9K), stopping on pressure or stall. Trace exact layer/expert/Metal stage and fix only with evidence. Then run real DeepSeek and Qwen tool workflows, restart and restore a verified SSD prefix, validate admission/error paths and all mandatory gates. Do not run a 32K full-model stress test automatically. See `HANDOFF_TO_CLAUDE_CODE.md` for commands and open work.

Last verified launch command: `./serve.sh --pool-gb 0`, with `/health` confirming the absolute 3072/256 MiB guard before requests. The server was stopped cleanly after the pressure-triggered cancellation.
