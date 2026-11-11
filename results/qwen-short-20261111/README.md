# Qwen short throughput comparison

One supervised pair used the original Qwen Flash-Next checkpoint and
BF16 KV, exact routing, a 2 GB expert pool, and a 2,569-token repository
prompt. Both arms used greedy sampling and generated 32 tokens.
Each engine started with an empty private prefix cache.

The baseline used the standard prefill and decode paths. The candidate
used 4K hybrid layer-major windows, layer prefetch, uncached prefill
reads and buffered decode pool fills. The new `short-throughput` option
sets those same flags. The hybrid path now starts above 2,048 uncached
tokens, so the short request exercised the layer-major path.

| Metric | Baseline | Experimental profile |
| --- | ---: | ---: |
| Cold prefill time | 72.91 s | 33.22 s |
| Cold prefill speed | 35.23 tokens/s | 77.32 tokens/s |
| Decode speed | 2.22 tokens/s | 2.78 tokens/s |
| Physical reads during prefill | 154.11 GB | 85.57 GB |
| Physical reads through decode | 186.95 GB | 108.59 GB |
| Metal peak during prefill | 7.31 GB | 8.88 GB |
| Peak system swap | 1,820 MiB | 2,422 MiB |

All generated token IDs, last-row logits and captured cache-state
checksums matched. This is one paired workload. The faster path used
more Metal memory and swap, so it remains an experimental opt-in.
The run used the existing watchdog, power, thermal and swap guards.

Enable **Qwen faster prefill and decode (experimental)** in app Settings
and restart the engine, or launch with:

```sh
./serve.sh --enable short-throughput --pool-gb 2
```

The engine can admit the pool only when memory permits. Qwen requests
up to 2,048 uncached tokens continue to use chunk-major prefill under
this profile.
