# Native workflows

## Architecture decision

Kimi AgentSwarm provides task fan-out, while Tower adds worktrees and review
gates. OpenCode and Codex give children explicit assignments and separate
contexts. Claude Code documents isolated worktrees for parallel edits. Pi exposes
delegation through an extension. Veystrix uses small, typed task packets with
dependency results and one model assignment per workflow. The Mac's one resident
inference slot makes large simultaneous generation pools unsuitable.

- [Kimi AgentSwarm](https://www.kimi.com/en/help/agent/agent-swarm)
- [Kimi Tower](https://www.kimi.com/code/docs/en/kimi-code-cli/guides/tower.html)
- [OpenCode agents](https://opencode.ai/docs/agents/)
- [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)
- [Claude Code agent teams](https://code.claude.com/docs/en/agent-teams)
- [Pi subagent extension](https://pi.dev/packages/pi-subagents?name=pi-coding+-agent)

Temporal records progress and retries activities independently. Veystrix uses a
local SQLite journal with one runner per goal, durable task transitions, and
bounded retries. Interrupted tasks with permission to write require manual
reconciliation. Completed tasks retain their reports across recovery.

- [Temporal durable execution](https://assets.temporal.io/durable-execution.pdf)

## Run an overnight goal without a model

In the macOS app, open **Agents → Native workflows**, select a Git repository,
enter an objective and choose **Swarm** or **Overnight**. **Use mock backend**
runs the offline demonstration and requires no engine. The workflow stays
available in the history after the app closes. Its progress panel shows task
states and dependencies, with **Pause**, **Resume**, **Cancel**, and **View Report**
available as the runner changes state.

From a Veystrix checkout with its existing virtual environment, create a mock
fixture and an empty parent directory for workflow state. Select the root of a
Git checkout with at least one commit.

```sh
mkdir -p "$HOME/veystrix-workflows"
cat > /tmp/veystrix-demo.json <<'JSON'
{
  "plan": ["[{\"id\":\"inspect\",\"role\":\"explorer\",\"objective\":\"Inspect the repository root\",\"depends_on\":[]}]"],
  "inspect": [
    {
      "finish": "tool_calls",
      "message": {
        "tool_calls": [{"id":"list","type":"function","function":{"name":"list_dir","arguments":"{}"}}]
      }
    },
    "Inspected the committed root. No files were edited."
  ],
  "summary": ["Repository inspection is complete. No edits were proposed."]
}
JSON

.venv/bin/python -m veystrix.cli workflow start "$HOME/veystrix-workflows/demo" \
  --repo "$PWD" --goal "Inspect the repository root" --mode overnight \
  --mock /tmp/veystrix-demo.json --detach
.venv/bin/python -m veystrix.cli workflow status "$HOME/veystrix-workflows/demo"
.venv/bin/python -m veystrix.cli workflow report "$HOME/veystrix-workflows/demo"
```

Use `--mode single` with the mock task key `main` for one agent. In
`swarm` or `overnight` mode, pass `--plan path/to/tasks.json` to provide the
same task array directly. Otherwise the planner emits it from the goal. Each
item has an `id`, `role`, `objective` and optional `depends_on` list.
Ready tasks can run in parallel. Agents receive their own task and compact
reports from their declared dependencies. A coordinator receives the reports
once all tasks succeed. The journal still produces a report when a prerequisite
fails and its dependents are skipped.

Use `--model <installed-model-id> --port 8080` with a running Veystrix
server to use the loopback adapter. The runner leaves engine startup and model
admission to Veystrix. Set `VEYSTRIX_SERVER_API_KEY` in the runner environment
if the engine requires authentication. Each attempt supplies a distinct
session ID and a full independent transcript to the server. The runner sends
one request at a time. The server continues to govern admission and cache
reuse across all clients. This adapter was tested with a mock loopback server.

### Architect and executor

Select Qwen as the **Architect model** and Nemotron as the **Executor model**
in Agents, or pass the Nemotron installation ID with `--executor-model`.
Planning, review, and reporting use the primary model. Coder and tester
roles use the executor. The runner batches ready tasks for the same model.
Each model switch runs through the router's existing unload and readiness
checks, and agents keep distinct session identities across the switches.
The two-model workflow needs a freshly started, idle engine. An existing
engine owner blocks the workflow. Checkpoint loading and switching were
verified with mock models.

See the [Kimi Code integration](../../integrations/kimi/README.md) for the
skill and shell commands. Kimi's own AgentSwarm is a separate workflow.

## Pause, recover and inspect

- `workflow pause STATE` requests a cooperative pause. `workflow cancel STATE`
  requests termination. The runner stops scheduling, signals active requests,
  and records the resulting states. A remote engine may still be unwinding
  generation after the client closes its connection.
- `workflow resume STATE --detach` restarts a paused or interrupted run. Completed
  tasks are preserved. Interrupted read-only tasks may restart within the attempt
  budget. A write-capable task interrupted before its result was recorded is
  blocked and its workspace is retained for manual reconciliation.
- `workflow run STATE` runs a ready goal or reconciles a crashed runner with
  its recorded control state. `workflow events STATE --limit 100` shows the
  latest journal entries. `workflow report STATE` emits a Markdown report.
  Detached runners write to `runner.log`.
- The deadline runs from the first start across pauses and resumes. The
  runner bounds agent count, tool slots, turns, output tokens, context size,
  and attempts. The local adapter also bounds request time and response size.
  It retries only explicit pre-generation busy errors, through the read-only
  task retry policy. Custom Python backends must honor the cancellation callback.

The Python control surface consists of `WorkflowStore.create`, `status`,
`events`, `control`, and `Runner.run`. Inject a backend implementing
`complete(request, cancelled)` for tests or alternate model-independent
providers. Inject a shared scheduler when embedding multiple runners in
one process, and an `admit` callback for application-specific resource limits.

## Repository and permission boundaries

A workflow snapshots the selected checkout's committed HEAD. Dirty and
untracked files stay in the source checkout. The snapshot omits symlinks,
Git metadata and submodule contents. By default agents can list, find, search
and read files in that snapshot. `--allow-edits` grants only the coder role
text writes in a separate copy per attempt. Built-in syntax checks cover
Python, JSON and TOML. Shell commands, external search, Git mutation and
deployment are outside this workflow profile.

Proposal files stay under `STATE/workspaces/`. The report lists edited
paths and flags overlaps. Compare each proposal against `STATE/base/` before
applying it. Review agents receive bounded diff previews and verification
results from their dependencies. Workflow completion means the task graph
finished and generated reports. Acceptance of a proposal requires a human
review. The private state directory contains transcripts and file excerpts.

The SwiftUI screen, CLI and Python API share the same durable workflow
journals. Linked Git worktrees, automated review-gated integration, and
sandboxed project test execution remain future work. There are no real-model
switching, performance or overnight stability measurements yet.

## Offline tests

```sh
.venv/bin/python -m unittest veystrix.tests.test_workflows
```

The tests exercise automatic and explicit plans, isolation, concurrent
tools and serialized inference, process crash recovery, pause and resume,
cancellation, bounded retries, detached CLI execution and a mock loopback API.
