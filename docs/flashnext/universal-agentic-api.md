# Veystrix as one local backend for agent clients

Status 2026-11-08. Offline only: **no full-checkpoint AGENTIC run has completed.** Evidence in this document is labelled
**scripted** (a fake model behind the real HTTP server, or recorded client traffic replayed against it), **documented**
(taken from a public client's documentation, never run here) or **real** (the 113 GB checkpoint). There is no real-model
AGENTIC evidence yet.

## 1. The contract (OpenAI Chat Completions, `POST /v1/chat/completions`)

Start: `./chat.sh --mode agentic` (default `127.0.0.1:8080`; a non-local `--host` requires `--server-api-key`).

| Area | Behaviour |
| --- | --- |
| Auth | `Authorization: Bearer <key>` or `x-api-key`, constant-time compare, applies to every route including status routes. No key configured means no auth (localhost only). |
| `GET /v1/models`, `GET /v1/models/{id}` | One model. `context_window` is the limit this server **enforces** (4096 in the standard AGENTIC tier, smaller if memory required it), never the model's theoretical maximum; `context_limit_enforced`, `max_output_tokens` (window minus the output reserve) and `created` are included. An unknown id is a 404 `model_not_found`. The older `models` array is unchanged. |
| `GET /health` | Liveness. `status: ok`, `ready`, `busy`, queue state (active request, waiting, completed, rejected, dropped), `context_limit`, cache counters (reused, rebuilt, evictions, refused oversize prefills, warm tokens, the admitted `max_cold_prefill`) and the session registry (live session, switches, recent sessions with their rebuild counts). Answers while a request is generating. |
| `GET /ready` | 200 when the model is loaded and no GPU lifecycle fault is recorded; 503 with the fault text otherwise (the process must then be restarted). |
| Chat fields honoured | `model` (**not validated**: any id is accepted and the real one is returned), `messages`, `stream`, `stream_options.include_usage`, `max_tokens`, `max_completion_tokens`, `tools`. |
| Accepted, no effect | `store`, `user`, `metadata`. |
| Ignored, reported | `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `parallel_tool_calls`: sampling is fixed per process. Listed in the `X-Veystrix-Ignored-Fields` response header, never silently dropped. |
| Anything else | 400 naming the field, nothing run. `tool_choice` is accepted only as `auto`; `n` only as 1. |
| Messages | Roles system/user/assistant/tool. Assistant `content: null` with `tool_calls`, JSON-**string** arguments, `role: tool` with `tool_call_id`. Unknown message fields are refused by name. |
| Tool calls out | `message.tool_calls[]` with `id` (`call_<uuid>`, unique per call), `type: function`, JSON-string `arguments`; `content: null` when there is no text; `finish_reason: tool_calls`. Arguments are validated against the client's own schema before they are returned. Streaming sends each call in one delta with the complete arguments (valid per the spec, not token-by-token). |
| Usage | `prompt_tokens`, `completion_tokens`, `total_tokens` on the response, and as the final chunk with `include_usage`. |
| Finish reasons | `stop`, `length`, `tool_calls`. |
| Errors | `{"error": {"message", "type", "param", "code"}}` always. 400/404/409/413 `invalid_request_error`, 401 `authentication_error` (`invalid_api_key`), 403 `permission_error`, 5xx `server_error`. Codes in use: `context_length_exceeded`, `model_not_found`, `not_found`, `server_busy`, `worker_stalled`, `prefill_budget_exceeded`, `model_tool_call_invalid`, `internal_error`. After a stream has started, an error is a final `data: {"error": ...}` then `data: [DONE]`. |

Model tool-call **quality** failures (the model emits an undeclared tool, arguments that are not a JSON object, a missing
required parameter, or a wrong type) are answered `422` with `type: server_error`, `code: model_tool_call_invalid` and
the parameter named, never hidden as a successful empty answer and never labelled a client mistake. The status is a
non-retryable 4xx on purpose: a client that auto-retries 5xx would repeat a slow generation likely to fail the same way,
while a harness can feed the message back to the model. This is distinct from API shape failures (400
`invalid_request_error`), which the scripted protocol tests cover. Tool execution always belongs to the client.

### Context and output budget

* The server counts the exact tokens of the rendered prompt. Nothing is truncated, and system prompt, tool definitions and
  tool results are never dropped to fit.
* A prompt that leaves less than 256 tokens for the answer is refused with `context_length_exceeded` (400) before any
  work. Otherwise output is clamped to the room left. A client asking for 32,000 tokens gets what fits, with
  `finish_reason: length` if it runs out.
* AGENTIC enforces 4096 tokens when memory admits the standard tier, and a smaller **enforced** tier otherwise (compact:
  context 2048, single prefill up to 1536 tokens; minimal: context 1536, up to 1024), chosen at startup from measured
  memory under unchanged safety rules (`autonomous-memory-preparation.md`). A request above the admitted tier is refused
  with `400 prefill_budget_exceeded` before any state changes. `/v1/models` reports the window actually enforced. 2K to
  4K to 8K to 10K is raised only on evidence: nothing above 2K has run on the real model.

### One request at a time

The model has one live conversation and one set of GPU buffers.

* Exactly one inference runs; it runs on the main thread (so the guards and the watchdog's record are unchanged). HTTP
  connections are handled by threads.
* Up to 4 further requests wait, at most 20 s each, in arrival order. A fifth, or one whose wait expires, gets
  `503 server_busy` with `Retry-After`, and **never runs**.
* A waiting client that disconnects is dropped without running. A generating client that disconnects ends the generation
  (checked about once a second) and drops that request's live state; the server is immediately usable.
* A stalled worker is ended by the out-of-process watchdog (see `gpu-gil-deadlock-and-watchdog.md`), which first tells the
  client (`503 worker_stalled`, or a terminating error event if streaming had begun).

### Sessions, isolation and side requests

The model holds **one** live conversation state, which cannot be rewound. No request is refused for being short or
tool-less; a title or summary side call is served like any other. What happens instead is explicit and visible:

* Every request is assigned to a conversation: the client's own id when it sends one (`X-Veystrix-Session` header, or
  `metadata.session_id` / `metadata.conversation_id`), otherwise a digest of the first system message and first user
  message, which are identical on every turn of one conversation and differ between unrelated ones.
* A request that **extends** the live conversation reuses it and prefills only the new tokens.
* A request from **another conversation** replaces the live state. The replaced conversation is marked cold and its next
  request is rebuilt from its own full, exact token sequence (the prefix cache supplies any stored prefix). Isolation comes
  from that exact-token reconstruction: a conversation can never see another conversation's cache or tokens.
* Each switch and eviction is counted in `/health` (`cache.sessions`, `cache.evictions`).

The cost is real and unavoidable with one state: after a side request the next turn of the replaced conversation pays a cold
prefill (about 18 s for the 898-token fixture, roughly 49 tokens per second). Prefer disabling the client's side requests,
and give each agent its own `X-Veystrix-Session` id so switches are attributed. The admitted tier's `max_cold_prefill` also
applies to a rebuild, so a conversation longer than that cannot be rebuilt after being replaced: it is refused, not truncated.

## 2. Exact prefix caching: audit (read-only, 2026-11-08)

Source: `models/flashnext/prefix_cache.py`, `prefix_session.py`. Verified by reading the code; the existing unit suites
(`test_prefix_cache.py` 48, `test_prefix_session.py` 36, `test_prefix_endurance.py` 5) cover these properties. No
real-model run was made for this audit.

| Property | Finding |
| --- | --- |
| Exact token prefix | Entries are keyed by SHA-256 over the engine fingerprint, the token count and every token id, and the stored ids are compared again on lookup. No approximate matching. |
| Engine identity | The fingerprint covers checkpoint, tokenizer and template digests, engine and numeric settings, chunk size and policy, ring and QSA settings. A change in any is a miss. |
| Chunk-plan provenance | A stored state is reused only if the new prompt's cold plan cuts at that length **and** made exactly the stored chunk history (`compatible`). Vetoed entries stay for other plans. |
| Atomic writes | Temporary file, fsync, rename, manifest last, directory fsync; mode 0600 in a 0700 directory; leftover temporaries matching the exact pattern are cleaned at open; corrupt or truncated entries are rejected on load. |
| Bounded storage | RAM and SSD budgets; SSD eviction takes only a leaf that nothing chains to and that is not leased; a free-disk reserve applies. |
| Active-snapshot protection | Leases (`_Lease`) pin an entry and its ancestors while a restore or snapshot is in use; eviction skips leased keys. |
| Session isolation | Content-addressed: only an exact token-and-fingerprint match can be restored, so one conversation cannot receive another's state. One writer per directory (flock ownership lock). |

Gaps, none of them a defect found in the code: cross-process behaviour and long-session growth are not verified on the real
model; the live-conversation cache (separate from this snapshot cache) holds one conversation only (section 1).

## 3. Client setup

Common to all: base URL `http://127.0.0.1:8080/v1`, the model id from `GET /v1/models`, and an API key only if the
server was started with one. Set the client's context window to the value `/v1/models` reports (4096 or less) and its output cap well below it.

* **CHAT** (`./chat.sh --mode chat`): interactive terminal, no server. Unchanged by this work.
* **AGENTIC server** (`./chat.sh --mode agentic`): the settings above. Start it only after the readiness gate passes.
* **DeepSeek Harness**: run `dsh headless --patch <patch> --json "<task>"` (the `--patch` flag must come before
  `--json`). Its provider uses `openai-completions`, sends `stream_options.include_usage`, `store: false` and
  `max_completion_tokens`, assistant `content: null` with JSON-string arguments and `role: tool` results; its
  title request is disabled by disabling `session-title-llm`. Details and the patch used are in `agentic-harness.md`.
  **scripted:** the recorded wire protocol replays correctly (20 tests). **real:** none.
* **OpenClaw** (**documented**, not installed here): a custom provider under `models.providers` with `baseUrl`
  `http://127.0.0.1:8080/v1`, `api: "openai-completions"`, an `apiKey` (any non-empty text if the server has no key) and
  the model listed in the provider's `models` array with an `id` that matches `/v1/models` exactly. Check the current
  OpenClaw docs for the context-window field before relying on this. Untested.
* **Hermes Agent** (**documented**, not installed here): `provider: custom` with `base_url` and the model name, and a
  `context_length` that matches the server. Third-party guides state Hermes wants roughly **64K** of context for agent
  use. That is far above the 4096 this server enforces, so **expect Hermes to refuse or misbehave**; this is an
  unresolved incompatibility, not a configuration detail. Untested.
* **OpenCode** (installed at `~/.opencode/bin/opencode`, not exercised): an OpenAI-compatible provider pointing at the
  base URL with the model's limits set to the enforced window. Its exact config schema was not verified here. Untested.
* **Generic OpenAI client**: any client that sends only the honoured fields works; one that sends `tool_choice` other
  than `auto`, `n` above 1, or unknown fields gets a 400 naming the field. Disable client-side title, summary and
  auto-compaction requests, which are small tool-less calls that replace the live conversation and make its
  next turn pay a cold prefill. They are served, not refused; give each agent its own `X-Veystrix-Session` id.

## 4. Compatibility matrix

`S` scripted, `D` documented only, `R` real checkpoint, `-` none.

| Client | API connectivity | Streaming | Tool calls | Multi-turn | Cache | Full checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek Harness | S, **R** (ran against the real server) | S | S (tool call **not yet produced by the real model**) | S | R: SSD prefix hit of 643 tokens restored | Prefill and decode started, clean shutdown (preliminary, battery); the one-token output cap from the client patch ended the run before a tool call. Fixed offline, not re-run (`agentic-smoke-retry-4-2026-11-08.md`) |
| Hermes Agent | - | - | - | - | - | **blocked by context size (needs about 64K, server enforces 4K)** |
| OpenClaw | D | - | - | - | - | - |
| OpenCode | - | - | - | - | - | - |
| Generic OpenAI SDK | S | S | S | S | - | - |

Real evidence that exists for this backend: CHAT works on the real checkpoint (five turns, pool off); two AGENTIC
~900-token prefills completed on the real model when admitted (18.5 s and 20.0 s), the second followed by a decode start
and a clean shutdown, with autonomous memory preparation and a tiered admission (preliminary: battery). No valid real
tool call, no tool-result continuation and no final answer yet.

## 5. Review of draft PR #1 (read-only; not merged, not pushed)

Documentation only (`README.md`, `docs/README.md`, `docs/flashnext/agentic-validation.md`), base `development3.8`.
Checked against the local tree:

* Accurate and still correct: default port 8080, the `--mode`, `--pool-gb` and `--auto-purge` flags, the 4096 AGENTIC
  preset, the 9 GiB startup gate, CHAT measurements, "do not merge before the code is published".
* Stale against the local work: it says the harness protocol is unconfirmed (it is now verified offline, still not
  on the real model); lists cancellation and error handling as unbuilt gates (implemented offline: busy answers, disconnect
  handling, standard errors, watchdog); does not mention the readiness gate, the GPU/GIL deadlock and its repair, `/ready`,
  the single-request scheduler, or the 256-token output reserve; and the Stage A to E plan predates the two real attempts.
* Recommendation: leave the PR as it is until the code is committed, then fold in sections 1 to 4 of this document and the
  deadlock document, and keep the "unverified on the real model" labels.
