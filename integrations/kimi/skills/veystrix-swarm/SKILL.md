---
name: veystrix-swarm
description: Launch or inspect a user-requested native Veystrix swarm or overnight goal from Kimi Code.
disableModelInvocation: true
---

Use the installed `veystrix workflow` command, or the Veystrix checkout's
`.venv/bin/python -m veystrix.cli workflow` command if `veystrix` is absent.
Treat the user's objective as a CLI argument, with shell quoting that preserves
it literally. Use the user-selected Git repository and a new directory under
their local workflow state directory. The directory's parent must exist.

For a Qwen architect and Nemotron executor, use the exact installed model IDs
from the Veystrix Agents screen or its model catalog. Use a freshly started,
idle local engine for the two-model workflow. Pass the Qwen ID as `--model` and the Nemotron ID as
`--executor-model`. Launch with `--mode swarm` or `--mode overnight` and
`--detach`. Pass `--allow-edits` only when the user authorized proposal edits
in isolated copies. If the user requests an offline demonstration, pass
`--demo` and omit both model options.

Return the workflow directory and how to inspect it with
`veystrix workflow status STATE` or `veystrix workflow report STATE`.
Use `pause`, `resume STATE --detach`, or `cancel` only when requested.
The Veystrix runner owns the task graph and single-slot model sequencing.
