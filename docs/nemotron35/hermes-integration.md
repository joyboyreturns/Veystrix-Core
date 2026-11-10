# Hermes Agent integration — planned, not yet operational

[Hermes custom-provider documentation](https://hermes-agent.nousresearch.com/docs/integrations/providers) supports the OpenAI-compatible base URL below. This example is a target configuration only: the current branch does not register or serve this model ID yet. Do not change the user's live Hermes configuration for testing.

```yaml
model:
  provider: custom
  default: veystrix-nemotron35-lightning
  base_url: http://127.0.0.1:8080/v1
  key_env: VEYSTRIX_API_KEY
  context_length: 4096  # provisional; set only to the actual enforced validated server limit
```

The server must verify model ID, chat template, incremental Mamba and attention state, streamed tool calls, tool result ordering, local key authentication, disconnections, usage and memory admission before this configuration becomes usable. Hermes executes tools externally; the engine must never replay a tool action automatically. Actual Nemotron-generated calls and multi-hour work remain unverified.
