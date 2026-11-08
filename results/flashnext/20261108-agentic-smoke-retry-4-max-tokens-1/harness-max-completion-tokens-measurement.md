# Measured offline with the real DeepSeek Harness against a capturing mock server (no model)

The real run's request ended after one token (`finish: length`, harness reason `max-tokens`). The harness's request was
captured offline with the same patch, same task, same workspace:

| patch model entry | `max_completion_tokens` the harness sent |
| --- | --- |
| contextWindow 4096 (the failed run) | 1 |
| contextWindow 6144 | 1316 |
| contextWindow 8192 | 3364 |
| contextWindow 16384 | 11556 |
| contextWindow 8192, maxTokens 512 | 512 |
| contextWindow 6144, maxTokens 1024 (the fix) | 1024 |

The window minus the sent value is 4,828 in every row: a fixed prompt estimate plus headroom inside the harness (the
actual request is 2,803 characters, about 902 tokens). The request is floored at 1 token, so a 4096 window leaves no room to
answer.
