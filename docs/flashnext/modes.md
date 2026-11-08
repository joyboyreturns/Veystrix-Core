# CHAT and AGENTIC: the two launch modes (2026-11-08)

One runtime, two presets. Both modes start the same `macqwen/session.py` process, load the same checkpoint through the same
`FlashNextBackend`, and use the same Metal kernels, expert streaming, cache-integrity rules and safety supervisor
(`ChatGuard`, exit on a memory, swap, thermal or battery stop; refuses to start under 9 GiB available). A mode is
configuration: `models/flashnext/settings/modes.py` (pure Python, no dependencies) builds an environment preset, `macqwen/cli.py`
applies it, and `session.py` prints the startup banner from the environment the process actually received.
There is no live reconfiguration: the flags are read at import or first request, so switching mode means a new process.
`chat.sh` is the only launcher; `./chat.sh` with no `--mode` is CHAT.

## Commands

```sh
# CHAT: terminal chat, streamed, multi-turn, no tools. /new or /reset clears the conversation, /quit leaves.
./chat.sh --mode chat

# AGENTIC: the local model server (OpenAI /v1/chat/completions, /v1/responses; Anthropic /v1/messages) on 127.0.0.1:8080
./chat.sh --mode agentic
./chat.sh --mode agentic --port 8081 --context-limit 2048 --prefix-dir ~/.flashnext/prefix-cache
```

Options (both modes): `--pool-gb 0|1|2` (never above 2), `--enable NAME` (repeatable, see below), `--context-limit N`.
Max answer length, temperature and thinking are the existing chat commands: `/max-tokens N`, `/sampling ...`, `/thinking on|off`, `/status`.

## What each mode sets

| | CHAT | AGENTIC |
|---|---|---|
| Profile | plain (no tools) | agent (repository prompt, tools) |
| Prefill | one call per prompt: reference numerics | balanced 2,048-token chunks, 512-expert scratch ring, release of the decode pool before a fresh long prefill |
| Conversation state | live KV and recurrent state kept between turns | live state plus the exact prefix cache |
| Prefix cache | off | exact, RAM 0.75 GB, SSD 20 GB, 50 GB free-disk reserve (the cache turns itself off rather than use an unlimited budget), snapshot at the system boundary, chunk-plan provenance, model and tokenizer fingerprints |
| Decode pool | 2 GB, LRU | 2 GB, LRU |
| Context limit | none | 4,096 tokens; a longer prompt gets an explicit 400 error naming both numbers, nothing is truncated, live state is kept |
| Always on | `ChatGuard`, eager linear-state evaluation (reproduced the 1K digest, retained memory 0.99 to 0.24 GB), native MTP off (recorded rejection), no automatic `purge` | same |

Features the mode owns are written over anything inherited from the shell or `~/.frankenstein/env`; unrelated research flags still pass through.

## Decode pool: why 2 GB, and what is unknown

* Recorded: no pool 3.42-3.47 tok/s, 6 GB pool 3.79-3.93 tok/s (`results/flashnext/20260923-153742-expert-pool`). **No decode run at 2 GB exists.**
* Offline replay of one recorded route trace (not a speed measurement): LRU hits 0% at 1 GB, 46.6% at 2 GB, 63.6% at 4 GB.
* So the supported choices are 0 or 2 GB; 1 GB is allowed with a warning; above 2 GB is refused (the old 6 GB default is gone from CHAT and AGENTIC).
* 2 GB is the memory-safe default (it ran under every guarded prefill run). Its decode speed is what the first real session measures:
  compare `--pool-gb 2` with `--pool-gb 0` before deciding anything else.

## Capability status (nothing here is called production-ready)

| Capability | Status |
|---|---|
| CHAT: streaming, multi-turn, reset, cancel, clean exit, guard stop, fence failure poison | offline-tested (tiny real MLX model and fakes); **not yet run on the 113 GB checkpoint** |
| Exact system-prompt cache, RAM and SSD, restart, same chunk plan | real-model evidence at 2K (cached equals uncached bit for bit; 124 of 124 state tensors) |
| Balanced 2,048-token chunked prefill + scratch ring | real-model evidence at 2K (33.9 s against 65.4 s); chunked output is **not** bit-identical to a single call (bf16 shape drift), real-model quality suite not run |
| `--enable memory-manager` | offline only; F2Bm real run pending |
| `--enable prefix-profiles` | offline only |
| `--enable auto-reclaim` | offline only |
| `--enable short-pool-reuse` | offline only; replay predicts under 5% pool hits for a new request's prefill, so little to gain |
| Context above 4K, 8K, 10K | not validated; `--context-limit` can be raised, the banner warns |
| DeepSeek Harness protocol | **not verified**: the repository holds no harness spec. The server speaks OpenAI chat-completions and responses and Anthropic messages; which one the harness uses, and its tool-call dialect, must be confirmed against the harness itself |
| Tool schema in prefix cache | tools are rendered in the system block, so the system-boundary snapshot covers them (`long-context-engine.md` section 6) |

## Startup banner

Printed after the guard starts and before the model loads: mode and profile, checkpoint, decode pool and why, prefix-cache status and
budgets, chunk policy, memory readiness against the 9 GB gate, supervisor state with its trace file, context limit, every
`EXPERIMENTAL` feature, and warnings.
