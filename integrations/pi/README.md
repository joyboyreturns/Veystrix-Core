# pi

The pi coding agent gets Veystrix through one TypeScript extension,
`~/.pi/agent/extensions/veystrix.ts`, which registers a `veystrix` provider
whose models are discovered from the running engine's `GET /v1/models`.

**The canonical source is `veystrix/pi_extension.py`** in the repository
(`EXTENSION_SOURCE`), not a file in this directory: the packaged app's engine
bundle ships only the `veystrix` and `models` packages, so the installer must
be importable from inside the bundle.

Install or refresh it (backs up an existing file, idempotent):

```sh
.venv/bin/python -m veystrix.harness_configure --apply
```

The macOS app's **Agents → Refresh Harness Models** button runs the same
installer against the bundled engine. The extension defaults to
`http://127.0.0.1:8080/v1` and overrides via `VEYSTRIX_BASE_URL` and
`VEYSTRIX_API_KEY`. `contextWindow` is the engine's window for the model and
`maxTokens` is the server's `max_output_tokens`, so pi never asks for more
output than fits and the engine clamps what remains to the room actually
left — uncapped generation in practice.

Start the Veystrix engine, then use `/model` to select a model under
`veystrix`. Use `/thinking` to change how it reasons.

| Model | Pi thinking levels |
| --- | --- |
| Qwen3.8 Flash-Next | Off, low, medium, high, xhigh |
| Nemotron 3.5 Lightning | Off, medium (the checkpoint's native thinking mode) |

Pi sends its session ID to Veystrix so follow-up turns can reuse live
state and a model switch within the same session keeps router ownership.
When changing the reasoning level alters the rendered prompt, Veystrix
reuses only a matching prefix.
