# DeepSeek Harness with Veystrix AGENTIC mode (2026-11-08)

Status: protocol verified against the real `dsh` client and the real Veystrix HTTP server with a scripted backend and the real Qwen tokenizer and chat template. **Not yet run on the full checkpoint.** Nothing here is committed.

## What the harness sends (captured from `dsh headless`, v0.2.2-rc.1)

* Route: `POST {baseURL}/chat/completions`, OpenAI Chat Completions. A custom gateway configured with `api: openai-completions` goes through the `@deepseek-ai/dsh-llm-pi-ai` plugin (pi-ai 1.3.0). Not the Responses or Anthropic route.
* Headers: `Authorization: Bearer <value of apiKeyEnv>`, a `user-agent` of `deepseek-harness/...`. The harness does not call `/v1/models` in headless mode.
* Body fields, and only these: `model`, `messages`, `stream: true`, `stream_options: {include_usage: true}`, `store: false`, `max_completion_tokens` (= contextWindow minus prompt size), `tools`.
* Messages: `system`, then the user task, then a second `user` message ("Current runtime context", cwd and file policy), all append-only. An assistant turn with tool calls has `content: null` and `tool_calls[].function.arguments` as a JSON **string**. Tool results are `role: tool` with `tool_call_id` and string `content`.
* Tools are standard function schemas. Default set: 23 tools, **5,654-token request**. The default exceeds the AGENTIC 4,096 context limit, so a trimmed profile is required.
* A second, tool-less request generates a session title from the first prompt. It is a different conversation, so on one live KV session it evicts the main conversation and forces a full re-prefill of the follow-up. The patch turns it off (`session-title-llm`).

## Server changes this required (`macqwen/server.py`, all uncommitted)

1. **Bug fixed:** string `arguments` in assistant history crashed the Qwen template (`Can only get item pairs from a mapping`), so every turn after a tool call failed. Tool calls are now converted to the mapping the template iterates. Arguments that are not a JSON object are a 400 naming the message.
2. Silent drops removed. Non-text content parts (images), unknown message fields, `tool_calls` on a non-assistant role, and unknown tool-call fields are refused by name. `reasoning_content` is passed to the template. Empty harmless fields (`refusal: null`) and a tool message's `name` are accepted.
3. The model's own tool calls are validated against the schema the client declared: undeclared tool, unknown parameter, missing required parameter, enum violation, wrong type. Nothing is repaired or dropped; the request fails with the offender named, which the harness can feed back to the model. Array and object parameters must parse as JSON of that type.
4. Request fields: unsupported ones (`stop`, `n` other than 1, `response_format`, `logprobs`, forced `tool_choice`, legacy `functions`, unknown `stream_options` keys) are a 400 before any stream starts. Sampling controls the server fixes per process (`temperature`, `top_p`, penalties, `seed`, `parallel_tool_calls`) are accepted and listed in the `X-Veystrix-Ignored-Fields` response header.
5. `stream_options.include_usage` now produces the final usage chunk (`choices: []`, prompt, completion, total tokens), as the harness requests.

Tests: `macqwen/tests/test_server_harness_protocol.py` (19). It includes rendering the harness history through the real checkpoint template when present, tool-call streaming shape, usage chunk, field policy and a client disconnecting mid-stream. Full suites: MACQWEN 464 OK; FlashNext unit 1,335 OK (3 skipped; the earlier note said 5, not investigated).

## Verified with the real client (scripted backend, real tokenizer, no model)

Sequence against the real Veystrix server, in a network-sandboxed (localhost only) isolated `DSH_HOME`: task, streamed `glob` tool call, harness runs `glob` in an isolated directory, tool result, follow-up answer. Prompts were 1,697 and 1,745 tokens (under 2K, under the 4,096 limit). Correct bearer key works; a wrong key makes the harness exit 1. Live-state reuse across the tool turn, measured in that run: the follow-up reused 1,722 of 1,745 tokens and prefilled 23 (with the title call disabled). Works from a brand-new `DSH_HOME`. A client disconnect resets the live state and the next request succeeds (offline test).

## Setup

Patch: `docs/flashnext/dsh/veystrix-agentic.patch.yml` (provider `veystrix`, `api: openai-completions`, `baseURL http://127.0.0.1:8080/v1`, model `macqwen-flashnext`, `contextWindow: 4096`, file tools only: read, glob, grep, edit, write, read_image; no bash, goals, todo, web, subagents, workflows, LLM titles).

```sh
# terminal 1: server (needs your approval to run on the real model; not run yet)
export MACQWEN_SERVER_API_KEY=local-agentic-key
./chat.sh --mode agentic --pool-gb 0 --auto-purge

# terminal 2: a harmless isolated workspace and one task
mkdir -p ~/veystrix-agentic-smoke && cd ~/veystrix-agentic-smoke && touch alpha.txt beta.md
export DSH_HOME=~/.veystrix-dsh VEYSTRIX_API_KEY=local-agentic-key   # separate from ~/.dsh, your desktop profile
"/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh" headless \
  --patch ~/Documents/Veystrix/docs/flashnext/dsh/veystrix-agentic.patch.yml \
  "list the files in the current directory"
```

`edit` and `write` come with the file tool plugin and cannot be split off, so the set is not strictly read-only. The test task only needs `glob`. The harness asks for approval on writes and fails closed in headless mode.

## Limits

* The first full-model run was stopped by the memory guard in cold prefill (`first-agentic-smoke-guard-stop-2026-11-08.md`). AGENTIC now also refuses to load when the cold-prefill budget in `agentic-prefill-readiness.md` is not met.

* Full-checkpoint behaviour is unmeasured: decode speed with this prompt, first-request prefill time for about 1.7K tokens, memory and guard behaviour, and the real model's tool-call formatting. The server only accepts Qwen XML or JSON tool calls.
* Prefix-cache hits across requests and restarts are verified offline on a tiny real MLX model (`test_prefix_session.py`: cold versus restored equality, restart, cancel, tool-result conversations), not through the harness on the full model. Within one conversation the live KV state is reused without the cache.
* The default harness profile (5.6K) is refused by the 4,096 limit with an explicit 400. Raising the limit is unvalidated.
* One conversation at a time; a second concurrent conversation evicts the first.
