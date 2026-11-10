# Benchmark plan — not run

`python -m models.nemotron35.tests.bench.benchmark_plan` prints the target matrix without loading models or running a benchmark. Implement a measurement runner only after the loader, safe offload and instrumentation exist. Follow `docs/measurement-standard.md`: same machine, full-quality routing, fresh-process paired alternating order, identical prompts/sampling, explicit cold/warm storage and cache conditions, thermal/power observations, and no simultaneous model residency.

Measure 1K/4K/8K/16K/32K cold and warm prefill, TTFT, 128 and 512 decode tokens, context growth, physical SSD bytes/token, pool hit/miss, GPU versus I/O time, unified memory, swap and pressure, thermal throttling, Hermes tool latency and successful coding tasks/hour. Record successful-task semantics as well as speed.

Targets, not results: cold 8K prefill 120–150+ tokens/s, sustained decode 8–10+ tokens/s, stretch 12+, and 1.5× full-quality Flash-Next on both prefill and sustained decode. If measured results miss these goals, report bottlenecks; do not promote Nemotron by default.
