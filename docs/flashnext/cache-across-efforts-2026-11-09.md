# One cache for every effort, thinking on or off, Qwen Code and DeepSeek Harness (2026-11-09)

## What was wrong

The checkpoint's chat template writes the reasoning-effort sentence as the first text of the system message, before the tool
schemas and the client's system prompt. A KV cache entry is only valid for an identical token prefix, so:

| effort | tokens shared with `medium` (DSH, 8.8K prompt, measured) |
|---|---|
| off (thinking disabled) | 8,852 of 8,853 (identical but the generation prompt) |
| medium | all |
| high | 7,719 (its extra sentence is prepended to the client system prompt) |
| low, xhigh, max | 3 |

Switching `low` to `medium` therefore re-prefilled the entire ~10.7K-token Qwen Code prompt cold (about 5 minutes) even though
the SSD cache was intact. That, not the server restart, was the slow first message.

## What changed (default `VEYSTRIX_EFFORT_PLACEMENT=late`)

The server renders the template exactly as `medium` (no sentence, nothing prepended) and inserts the effort sentence as its own
short **system turn right after the system message** (`macqwen/effort_slot.py`, `ModelService._with_effort_turn`):

```
<|im_start|>system \n [tools block] [client system prompt] <|im_end|> \n          <- identical for every effort
<|im_start|>system \n Reasoning effort is set to low. ... <|im_end|> \n            <- only low / high / xhigh
<|im_start|>user ...
```

* `medium` and thinking-off have no extra turn: their tokens are identical to before, so **every existing cache entry stays
  valid** (tested with the real tokenizer: `test_effort_slot.py`; and live: a no-thinking Qwen Code request still restored from
  entries written before the change).
* `low`, `high`, `xhigh` (and `max`, clamped to `xhigh`) add the turn. The state the cache already stores at the end of the
  system message (an anchor of the chunk plan in every cache mode) is therefore the shared prefix of **every** effort. The chunk
  plan of the shared part needs no change at all. Bit-exactness against cold runs is tested on the small model
  (`EffortTurnSharingTests`: low restores 302 of 302 system tokens from medium's snapshot and equals a cold low run exactly).
* The turn is delimited by special tokens on both ends, so no BPE merge can alter the shared tokens.
* `high` no longer prepends its sentence to the client system prompt; it uses the same turn.

Result: switching effort re-prefills only the turn (tens of tokens) and what follows it (the conversation). The tools and the
system prompt are not recomputed.

## The trade-off you should know about

This is **not the layout the checkpoint was trained with.** The instruction is the same sentence, read after the system
prompt as a second system turn instead of before it as the first line of the system message. Whether the model follows it as well from there has not been evaluated beyond the check in the
"Real runs" section. `VEYSTRIX_EFFORT_PLACEMENT=template` restores the checkpoint's own layout exactly (then efforts share
nothing again). Two limits remain whatever the placement:

* Changing the effort **in the middle of a conversation** re-prefills the history after the system message: attention state of
  every later token depends on the earlier ones.
* The first prefill of a new prompt (new client, changed system prompt, evicted entry) is cold once.

## Cache behaviour across conversations, sessions, harnesses and restarts

| situation | what happens |
|---|---|
| same conversation, next turn | live state extends (tens of tokens) |
| new session, same harness and effort | restore through the system message from SSD |
| new session, other effort | whole system message restored; the effort turn + new messages prefilled |
| other harness (Qwen Code vs DSH) | separate prompts, separate entries (nothing exact to share); both stay cached |
| server restart / crash | SSD entries and their index survive; live state is rebuilt from the latest message-end snapshot |
| cancelled or crashed cold prefill | resumes from the last stored chunk edge (`boundaries=all`) |
| entries you stop using | refreshed by the idle warm-up; budget 50 GB, oldest unleased leaf evicted first |

## Files

`macqwen/effort_slot.py`, `macqwen/server.py` (`_with_effort_turn`, `_adopt`), tests `macqwen/tests/test_effort_slot.py` and
`models/flashnext/tests/unit/test_prefix_session.py::EffortTurnSharingTests`. No change to the prefix cache itself.

## Real runs (live server, supervised, swap guard active, battery power 38-56%, Low Power Mode off)

DeepSeek Harness 0.2.2-rc.1, real `dsh` sessions, `say ok`, first request of each:

| effort | before this change | now |
|---|---|---|
| off | 5,593 of 5,600 restored, prefill 2.2 s | unchanged |
| low | miss, 193 s prefill | 5,385 restored (whole system message), 243 prefilled, 30 s |
| xhigh | only after `low` had just run | 5,385 restored, 255 prefilled, 33 s |
| high | never cached | 5,385 restored, 256 prefilled, 31 s |

Qwen Code 0.27.0, real `qwen -p` (git status and environment sit in its first user message, about 2K tokens, so they are
re-prefilled after any change before them):

| model entry / effort | restored | prefilled | prefill |
|---|---|---|---|
| no thinking (`reasoning_effort: none`) | 7,534 of 7,541 | 7 | 2.4 s |
| low | 5,473 (whole system message) | 2,096 | about 100 s |
| xhigh | 5,473 | 2,108 | 105 s |
| high | 5,473 | 2,109 | 112 s |

The earlier layout cost 193 s (DSH) and about 230 s (Qwen Code) cold for each new effort. What remains is the effort turn plus the
first user message, which comes after it and so must be recomputed.

**Behaviour check, inconclusive:** same short puzzle, two samples each, temperature 1.0: thinking was 185-234 characters at both
`low` and `xhigh` (xhigh not longer); the styles differed (`xhigh` terse "We need ..."). Too small and too easy to say whether the
instruction works as well from its new position as from the template's. To judge, give `xhigh` a hard task; if its thinking is
not deeper than `low`, set `VEYSTRIX_EFFORT_PLACEMENT=template` and restart (efforts then share nothing again).

Not done: a template-vs-late quality comparison on hard tasks; mid-conversation effort switch timing on a long real session.

## Usage reporting (cache hit shown by clients)

DeepSeek Harness showed "Cache hit 0%, Uncached input 9,459 tok" even when the SSD cache restored almost the whole prompt: the
server's `usage` carried no cache fields, and its `prompt_tokens` was only what the request had prefilled (e.g. 19 on a
continued turn), not the prompt length. Every API now reports, from `macqwen/server.py` (`usage_counts`, `chat_usage`,
`responses_usage`, `anthropic_usage`):

| API | fields |
| --- | --- |
| Chat Completions (stream and not) | `prompt_tokens` (full prompt), `prompt_tokens_details.cached_tokens`, `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` |
| Responses | `input_tokens` (full prompt), `input_tokens_details.cached_tokens` |
| Anthropic Messages | `input_tokens` (uncached), `cache_read_input_tokens`, `cache_creation_input_tokens: 0` |

`cached = prompt length - tokens prefilled by this request`, so it counts both the live conversation state and SSD snapshot
restores. Tests: `macqwen/tests/test_usage_cache_fields.py`. A server started before this change keeps the old numbers until
restarted; the cache itself is unaffected by a restart.
