# Qwen-to-Nemotron feature parity

Status legend: **Tested (synthetic)** = covered by model-free tests on a tiny synthetic checkpoint in the real tensor layout;
**unverified** = needs authorized real-checkpoint runs. Nothing below claims real-checkpoint correctness or speed.

| Qwen feature | Nemotron | Status |
| --- | --- | --- |
| Model selection / checkpoint validation | `--model nemotron35-lightning`, verified-checkpoint resolver (headers, inventory, extents; optional SHA-256); manual only | CLI/session: tested. Desktop picker implemented and builds; Swift tests unexecuted here |
| Power, battery floor, AC-only, Low Power Mode | Shared `power_gate` at start and shared `ChatGuard` during runs | Wired and unit-tested at launch-line level; guard behavior on a real run: unverified |
| Guard, swap/thermal pressure, watchdog, restart | Shared guard via the same launcher contract; supervisor `--model nemotron35-lightning`; no replay of tool actions | Wired; real stops unverified |
| Engine supervisor and diagnostic logs | `models.flashnext.overnight --model nemotron35-lightning` (auto-warm off) | Launch line tested |
| `/health`, activity, prefill progress, decode rate | Shared server reads backend progress callbacks, `last_decode_rate`; backend adds pool and expert-byte stats | Tested (HTTP, synthetic); values on a real run unverified |
| OpenAI/Anthropic APIs, streaming, usage | Shared server with the model's own chat template; stop tokens 2/11 | Tested (HTTP, synthetic backend; template parity with the real tokenizer) |
| Hermes reasoning alias, multi-turn tools | Template renders `<tool_call>` XML identical to the shared parser; tool results framed per template | Parser and framing tested; actual Nemotron-generated calls unverified |
| Prompt and persistent cache, session isolation | Named fingerprinted sessions; live tape extension; cold replay on divergence | Tested (synthetic); automatic persistent prefix cache **not implemented** |
| SSD experts, bounded pool | Exact 6-of-128 plus shared expert; bounded LRU; synchronous reads | Tested (equals in-memory quantized reference); speed unknown, no prefetch |
| Admission, 16 GB planning, single-model residency | `admission.py` plan and per-request gate; refuses before loading | Arithmetic tested; estimates unmeasured |
| Context and incremental prefill | Context bounded by plan; chunked prefill; server limits tightened to the plan | Tested (synthetic) |
| MTP | Not implemented | n/a |
| Packaging and signing | Bundled engine includes the shared modules; Nemotron-specific packaging checks not added | **Unverified** |

Do not advertise parity, auto-route, or make Nemotron the default until real-checkpoint correctness, reliability and performance
are established with explicit authorization.
