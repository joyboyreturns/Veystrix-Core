# Guarded baseline and candidate runs

The baseline was a clean source export of commit `b6310f0`. The candidate used
the working checkout. Each engine shut down before the next started. Artifacts
in this directory contain timing, generated output or token IDs, and supervisor
telemetry. The [completion report](../../docs/architecture/unified-cache-nemotron-1m-results.md)
states which figures are direct measurements.

- `qwen-baseline` and `qwen-after-retry` contain supervised
  `models.flashnext.tests.bench.bench_layer_gate` runs. Both used 430-token
  excerpts appended to the same agent system prompt, full routing, 0 GB expert
  pool and uncached expert reads. The second engine reused the first engine's
  private prefix directory and requested a different excerpt.
- `nemotron-bf16` and `nemotron-int8` contain two streaming OpenAI-compatible
  coding requests per engine. Both used the same checkpoint, a 1 GiB expert
  pool and a 32-token completion limit. The second request shared the first
  request's system text and changed the requested Python function.
- `nemotron-long-admission` contains the supervisor logs and request events
  from the stopped three-record retrieval attempt under the 1M configuration.
- `qwen-verified-file-read.json` records six alternating trials per reader
  on the same 2,107-token Qwen snapshot chain. This measures verified file
  loading without a model or cache-chain reconstruction.
