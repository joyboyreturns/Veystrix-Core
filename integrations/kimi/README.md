# Veystrix swarms from Kimi Code

Use the native Veystrix runner when Qwen and Nemotron share one local engine.
In a fresh Kimi Code session, type `!` to enter shell mode, then run the
workflow command below. Shell mode can launch the workflow before Kimi sends
any model requests to the local engine. To have Kimi interpret the request
and launch the workflow through `/skill:veystrix-swarm`, use a separate
provider for Kimi's main conversation. [Kimi shell mode](https://www.kimi.com/en/help/kimi-code/cli-interaction)

Install the skill under `${KIMI_CODE_HOME:-$HOME/.kimi-code}/skills/veystrix-swarm/`
and open a new Kimi session. Invoke
`/skill:veystrix-swarm Use Qwen to plan and review, and Nemotron to implement and check …`.
Pass the repository, model IDs and the Veystrix checkout or executable location
if Kimi needs them. Kimi's normal command approvals still apply. [Kimi skill locations](https://www.kimi.com/code/docs/en/kimi-code-cli/customization/skills.html)

With an idle Veystrix engine started fresh, the equivalent shell command is

```sh
workflow_root="$HOME/Library/Application Support/Veystrix/Workflows"
mkdir -p -m 700 "$workflow_root"
veystrix workflow start "$workflow_root/architecture-demo" \
  --repo /path/to/checkout --goal "Plan and propose the requested changes" \
  --mode swarm --model "$QWEN_INSTALLATION_ID" \
  --executor-model "$NEMOTRON_INSTALLATION_ID" --allow-edits --detach
veystrix workflow status "$workflow_root/architecture-demo"
veystrix workflow report "$workflow_root/architecture-demo"
```

The architect handles planning, architecture, review and the final report.
Coder and tester tasks use the executor. Ready tasks for the current model are
batched together. Veystrix unloads before switching and keeps distinct agent
sessions. Actual switch time depends on the installations and available memory.
Set `VEYSTRIX_SERVER_API_KEY` for authenticated loopback access and pass
`--port` for a non-default port. Mock runs use `--demo` with the model options omitted.
Runs created under this path also appear under **Agents → Refresh History**.

Kimi Code's built-in `AgentSwarm` can also use both local models. Use a fresh,
idle Veystrix engine. With Qwen set as the main model and Nemotron as the
secondary model in your normal Kimi configuration, launch this from the
project directory to create a separate profile and preserve your existing
Veystrix endpoint and credentials.

```sh
/path/to/Veystrix/.venv/bin/python /path/to/Veystrix/integrations/kimi/launch-swarm.py
```

To configure the profile manually, use a separate `KIMI_CODE_HOME`. In that
profile, point both model aliases at the same Veystrix provider. Set
`default_model` to the Qwen alias and `[secondary_model] default_model` to the
Nemotron alias. Under `[providers.veystrix.custom_headers]`, set
`X-Veystrix-Workflow` to a fresh 32-character lowercase hexadecimal ID for
the run. This lets the router recognize Qwen and Nemotron requests as part of
the same workflow while it switches checkpoints. Generate a new ID for each
unrelated swarm. [Kimi provider configuration](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/config-files)

Launch Kimi with `KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY=1` and enable
`/swarm` in its terminal UI. Ask Qwen to invoke `AgentSwarm` with at least
two items. Kimi collects the child results and returns control to Qwen.
[Kimi AgentSwarm controls](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/tools.html)
