# Cache and decode changes, 2026-11-13

The table reports checkpoint-free measurements on the M4 Mac. Prefix timings
are medians from eight alternating pairs. The Flash-Next fixture held 20
indexed snapshots at 5,000-token intervals. The Nemotron fixture repeatedly
restored an 8 MiB synthetic state from a standalone cache.

| Operation | Baseline | With the change |
| --- | ---: | ---: |
| Flash-Next prefix miss at 100K tokens | 4.825 ms | 3.939 ms |
| Flash-Next lookup with a 25K match in a 100K prompt | 4.885 ms | 4.065 ms |
| Flash-Next longest-prefix hit | 3.944 ms | 3.946 ms |
| Nemotron warm state restore | 3.550 ms | 0.153 ms |
| Flash-Next expert fill tasks for ten misses with 100 KiB scale and bias rows | 45 | 21 |

Flash-Next's rolling index covers entries whose token tapes are in memory.
Cold, disk-only entries still use the existing cache key. Chained restores
retain only the leaf's recurrent state while rebuilding the cache. The loader
still reads and validates ancestor snapshots, so this reduces retained memory
without reducing the ancestor bytes read.

The repository's September routing trace gave the following decode hit rates.
Its route widths vary from two to nine, so these numbers describe that older
session and cannot establish a full top-10 decode gain.

| Expert pool | LRU | S3-FIFO |
| --- | ---: | ---: |
| 1 GB | 0.0% | 32.5% |
| 2 GB | 46.6% | 47.3% |
| 4 GB | 63.6% | 63.0% |

```sh
python -m models.flashnext.tests.bench.bench_pool_policy_trace \
  results/flashnext/20260922-221213-route-trace-manual/route-trace.json.gz \
  --gb 1 2 4 --policies lru s3fifo
```

LRU remains the default. Set `FLASHNEXT_EXPERT_POOL_POLICY=s3fifo` to try
the alternative. `VEYSTRIX_NEMOTRON_PREFIX_RAM_MB` reserves a bounded
RAM tier in Nemotron's memory admission plan and defaults to zero. Set
`FLASHNEXT_DECODE_OVERLAP=1` or `VEYSTRIX_NEMOTRON_DECODE_OVERLAP=1` to
try overlapping resident expert compute with miss reads. Synthetic tests
exercise output parity. Real-checkpoint tokens per second, physical disk I/O,
and peak memory are unmeasured for these changes.

`Scheduler.run` accepts priorities from 0 to 10. An older queued job gains
one point per second relative to new arrivals. HTTP handlers use priority 0,
and only one inference job runs at a time.
