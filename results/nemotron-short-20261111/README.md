# Nemotron short prefill and decode comparison

The baseline was a copy of the working checkout before the prefill and
decode changes. All arms used the same Nemotron checkpoint, affine INT8 attention KV, a 1 GiB expert pool, the 1,048,576-token logical context,
and a 65,536-token incremental prefill cap. Each supervisor shut down
before the next started. The cold request used 1,009 input tokens. The
second request changed the user message, reused the same 969-token
system prefix and used 1,010 input tokens. Each arm used an empty
private cache on startup. Benchmarks ran under the existing watchdog
and swap, power and thermal checks.

The sampled pairs used 48 completion tokens. The second pair reversed
the run order. Both showed a shorter cold prefill and faster decode
after the cold prompt, while decode after the warm prompt was flat.
The sampled runs used the engine's process sampling defaults, so their
generated sequences could differ. The greedy pair used the engine
preferences file with temperature zero and a 32-token completion cap.

The optimized cold path processes the system and user chunks together
one layer at a time when each fits a single internal chunk. It shares
routed expert reads across the two chunks and saves the system boundary
state after the segment. A cancellation before the segment completes may leave no new system snapshot. The decode path starts demand reads before
predicting the next layer's experts and assembles the single-token expert
outputs in router order.

No benchmark prompt exceeded 1,010 input tokens.


| Greedy measurement | Baseline | Optimized |
| --- | ---: | ---: |
| Cold prefill | 11.98 s | 9.12 s |
| Cold time to first token | 12.67 s | 9.84 s |
| Decode after cold prompt | 4.49 tokens/s | 5.02 tokens/s |
| Cold request expert bytes | 45.88 GB | 38.36 GB |
| Cold process Metal peak | 4.32 GB | 2.83 GB |
| Warm SSD-restored tokens | 969 | 969 |
| Decode after warm prompt | 5.33 tokens/s | 5.26 tokens/s |

Both greedy requests produced identical output text at the 32-token
cap. Across the two sampled pairs, cold prefill averaged 12.00 s before
and 8.82 s after. Decode after the cold prompt averaged 4.60 and 5.26
tokens/s, respectively. Warm decode stayed near 5.4 tokens/s.
