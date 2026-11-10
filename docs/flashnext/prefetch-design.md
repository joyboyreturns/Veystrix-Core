# Flash-Next prefill expert prefetch: design

Status: scheduler implemented and unit-tested with mocks
(`models/flashnext/prefetch.py`). It is **not wired into the runtime**, is off
by default (`FLASHNEXT_PREFILL_PREFETCH_GB` unset or 0), and has no
full-model evidence. Nothing here changes what the model computes.

## Premise

In a prefill layer the order is fixed: router sync (drains the GPU), read the
layer's experts (GPU idle), issue `gather_qmm`, and the next router sync runs
it. The recorded 1,007-token prefill
(`results/flashnext/20260922-224718-prefill-io-manual/`) spent 25.6 s waiting
for reads and 14.9 s draining GPU work, 95% of 49.3 s with the two never
overlapping. The drive reached 2.99 GB/s on the real read pattern in an
isolated test but averaged 1.0 GB/s during prefill. The scheduler's job is to
start layer L+1's reads while layer L's GPU work runs.

## Contract

- `submit(layer, experts)` reserves bytes, allocates staging buffers and
  queues reads. It returns None when refused (disabled, over budget, depth
  limit, duplicate, closed, allocation failure). Refusal is silent and means
  "read on demand", the existing path.
- `claim(layer, needed)` returns a `Claim` with `staged` (expert to row) and
  `missing` (experts to read on demand, in request order). It waits only for
  the chunks that cover `needed`, cancels unstarted chunks nobody needs, and
  treats a failed, cancelled or late chunk as missing. It never raises for an
  I/O problem. Staged rows are exposed per part (`claim.rows(name)`) in the
  same `[expert, row_bytes]` layout `SafeTensorStore.rows_into` writes.
- `cancel(layer=None)` drops queued reads. Reads already running finish into
  their buffer, and the bytes stay reserved until they do.
- `close()` cancels, drains and stops an owned executor.
- `select_experts(...)` picks what to stage. At or above `dense_min_tokens`
  (default 2048, where about 400 of 512 experts are routed) it stages every
  expert and needs no prediction. Below it, it stages the predicted experts,
  most-used first when the cap bites.

Correctness argument: staged bytes are the same file rows the on-demand path
reads. The consumer remaps `local` indices to staged plus on-demand rows, so
the `gather_qmm` operands are identical. A wrong prediction only changes how
many bytes are read.

## Memory

- The budget covers every staging buffer from allocation until `free`
  succeeds, so the accounted peak is the real peak. `peak_reserved_bytes` in
  `stats()` reports it.
- Buffers are anonymous `mmap`: pages commit as reads land and `close()`
  returns them. A buffer still pinned by an exported view stays reserved and
  is retried on the next call.
- `safe_budget(configured, free_bytes, floor_bytes)` clamps the budget to
  free memory at the moment of use. The integration must call it with a real
  free-memory probe; the scheduler does not read system memory itself.
- Overlap needs two layers' buffers live: the layer being computed (claimed)
  and the layer being read. A whole dense layer is 512 x 3.072 MB = 1.57 GB,
  so dense mode needs about 3.1 GB, and a sparse layer (about 334 experts at
  1,007 tokens) about 1.0 GB each, 2.1 GB together.
- That is a real problem on this machine. The recorded prefill peaked at 6.1
  GB of MLX memory with about 100 MB free, two 2,031-token arms were killed
  (exit 137), and the chat default now adds a 6 GB expert pool. The
  integration therefore must not allocate staging on top of a resident pool.
  The options to test are a lazily touched pool, shrinking or skipping the
  pool during prefill, and `F_NOCACHE` staging reads so the page cache does not
  hold a second copy. None is decided.

## Remaining integration work

1. A read adapter around `store.rows_into(name, rows, out, mode)` with the
   nine parts and row sizes from the checkpoint (`gate/up/down` weight,
   scales, biases).
2. In `StreamingSwitchGLU._one_pass`: claim, read only `missing` through the
   existing `_read_weights`, then build the nine MLX arrays over staged plus
   missing rows (zero-copy `to_mx`) and remap `local`. Dense mode can use the
   expert id as the index.
3. A submit point for layer L+1 after layer L's reads are issued, a
   predictor for prompts below the dense threshold (layer L+1's router on
   layer L's hidden state, recall unmeasured), and a cancel call from the
   prefill cancellation hook (`layer_completed` in
   `veystrix/backends/flashnext.py`) and on error.
4. Worker count and I/O priority: the default 4 prefetch workers are
   unmeasured, and prefetch reads must not starve demand reads. Either inject
   the existing `_POOL` or measure a separate pool.
5. Setting wiring (the registry entry exists, research-only, startup) and a
   free-memory probe feeding `safe_budget`.
6. Full-model gates, only with permission: bit-identical `gather_qmm`
   operands, identical token digest, paired reversed arms at 1K and 2K
   tokens, MLX peak, free memory, exit-137 watch, and a check that keep-warm
   still behaves with overlap.
