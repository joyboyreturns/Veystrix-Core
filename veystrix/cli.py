#!/usr/bin/env python3
"""Select one model runtime, then execute the shared chat in its environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from veystrix import preferences
from models.flashnext.settings import modes
from veystrix.checkpoints import installed_checkpoints


PYTHON_ENV = {
    "shared": "VEYSTRIX_PYTHON",
    "flashnext": "VEYSTRIX_FLASHNEXT_PYTHON",
    "nemotron35-lightning": "VEYSTRIX_NEMOTRON_PYTHON",
}
MANAGED_ENV = ROOT / ".venv"
MANAGED_PYTHON = MANAGED_ENV / "bin" / "python"
RUNTIME_VERSIONS = {
    "mlx": "0.32.2",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.6.17",
    "transformers": "5.16.1",
    "numpy": "2.5.2",
    "requests": "2.34.2",
    "huggingface-hub": "1.29.0",
}
RUNTIME_PROBES = {
    "flashnext": """
from mlx_vlm.utils import (apply_generation_config_defaults, get_model_and_args,
                           load_config, update_module_configs)
""",
    "nemotron35-lightning": """
from mlx_lm.models.nemotron_h import Model, ModelArgs
from mlx_lm.models.cache import ArraysCache, KVCache
from transformers import AutoTokenizer
""",
}


def branch_sync_warning(root: Path = ROOT) -> str:
    """Warn when a checkout lacks commits from its known origin/main."""
    if not (root / ".git").exists():
        return ""              # the packaged app is not a checkout (and must not poke at git)
    try:
        current = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor",
             "origin/main", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return ""
    if current.returncode != 1:
        return ""
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        capture_output=True,
        check=False,
        text=True,
    ).stdout.strip() or "current branch"
    return (
        f"warning: {branch} does not include known origin/main; "
        "fetch and merge origin/main before validating chat behavior"
    )


def _split_build(argv: list[str]) -> tuple[str | None, list[str]]:
    if argv and not argv[0].startswith("-"):
        return argv[0], argv[1:]
    return None, argv


def _runtime_probe(model: str) -> str:
    versions = repr(RUNTIME_VERSIONS)
    return f"""
from importlib.metadata import PackageNotFoundError, version
import sys

expected = {versions}
for package, wanted in expected.items():
    try:
        found = version(package)
    except PackageNotFoundError:
        raise SystemExit(f"missing package {{package}}")
    if found != wanted:
        raise SystemExit(f"{{package}} {{found}}, expected {{wanted}}")

import mlx.core as mx
import mlx.nn as nn
if not hasattr(mx, "array") or not hasattr(mx, "fast") or not hasattr(nn, "Module"):
    raise SystemExit("MLX backend API is incomplete")
{RUNTIME_PROBES[model]}
"""


def _python_runtime_error(path: Path, model: str) -> str | None:
    try:
        result = subprocess.run(
            [str(path), "-c", _runtime_probe(model)],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip().splitlines()
    return detail[-1][:240] if detail else f"probe exited with {result.returncode}"


def _supports_python(path: Path, model: str) -> bool:
    return _python_runtime_error(path, model) is None


def _setup_failure(operation: str, error: BaseException) -> SystemExit:
    detail = getattr(error, "stderr", None) or str(error)
    return SystemExit(
        f"Veystrix setup failed during {operation}: {detail}\n"
        "Retry with './chat.sh setup'."
    )


def _run_setup_step(operation: str, command: list[str]) -> None:
    print(f"Veystrix setup: {operation}", flush=True)
    try:
        subprocess.check_call(command)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _setup_failure(operation, exc) from exc


def _interpreter(model: str) -> Path:
    override_name = next(
        (name for name in (PYTHON_ENV[model], PYTHON_ENV["shared"])
         if os.environ.get(name)),
        None,
    )
    override = os.environ.get(override_name) if override_name else None
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"missing Python environment override {override_name}: {path}")
        if not _supports_python(path, model):
            reason = _python_runtime_error(path, model) or "runtime probe failed"
            raise SystemExit(
                f"Python environment override {override_name} is incompatible "
                f"with {model}: {reason}"
            )
        return path

    if not _supports_python(MANAGED_PYTHON, model):
        setup_environment(["--venv", str(MANAGED_ENV)])
    if not _supports_python(MANAGED_PYTHON, model):
        reason = _python_runtime_error(MANAGED_PYTHON, model) or "runtime probe failed"
        raise SystemExit(
            f"managed Veystrix environment is incompatible with {model}: {reason}\n"
            "Retry with './chat.sh setup'."
        )
    return MANAGED_PYTHON


def setup_environment(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="veystrix setup")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    args = parser.parse_args(argv)
    target = args.venv.expanduser().resolve()
    creator = sys.executable
    if sys.version_info < (3, 12):
        creator = shutil.which("python3.12") or ""
    if not creator:
        raise SystemExit("Python 3.12 is required; install it, then run setup again")
    _run_setup_step("create the managed Python environment", [creator, "-m", "venv", str(target)])
    python = target / "bin" / "python"
    _run_setup_step(
        "upgrade pip",
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
    )
    _run_setup_step(
        "install Veystrix and its shared runtime",
        [str(python), "-m", "pip", "install", "-e", str(ROOT)],
    )
    print(f"Veystrix environment ready: {target}")
    return 0


def _default_checkpoint_args(argv: list[str]) -> list[str]:
    """Select the only installed checkpoint, or ask before loading one."""
    if argv and (argv[0] == "setup" or not argv[0].startswith("-")):
        return argv
    if any(
        value in ("--model", "--model-path", "--checkpoint")
        or value.startswith(("--model=", "--model-path=", "--checkpoint="))
        for value in argv
    ):
        return argv

    choices = installed_checkpoints()
    if not choices:
        return argv
    if len(choices) == 1:
        model, path = choices[0]
        return [*argv, "--model", model, "--checkpoint", str(path)]

    print("Multiple compatible checkpoints found:", file=sys.stderr)
    for index, (model, path) in enumerate(choices, 1):
        print(f"  {index}. {model}: {path}", file=sys.stderr)
    if not sys.stdin.isatty():
        raise SystemExit(
            "choose a checkpoint with --model and --checkpoint before loading"
        )
    try:
        answer = input(f"Choose a checkpoint [1-{len(choices)}]: ").strip()
        index = int(answer) - 1
    except (EOFError, ValueError):
        raise SystemExit("invalid checkpoint choice") from None
    if not 0 <= index < len(choices):
        raise SystemExit("invalid checkpoint choice")
    model, path = choices[index]
    return [*argv, "--model", model, "--checkpoint", str(path)]


def command(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    argv = ["--server" if value == "/server" else value for value in argv]
    build, argv = _split_build(list(argv))
    if build:
        raise SystemExit(f"veystrix: unexpected argument {build!r}; model selection is --model flashnext|nemotron35-lightning")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=("flashnext", "nemotron35-lightning"))
    parser.add_argument("--profile", choices=("plain", "agent"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path")
    parser.add_argument("--preferences-file", default=preferences.DEFAULT_PATH)
    parser.add_argument("--v4", action="store_true")
    parser.add_argument("--mode", choices=modes.MODES)
    parser.add_argument("--pool-gb", help="auto (default), 0, or a fixed size up to 2")
    parser.add_argument("--enable", action="append", default=[], choices=tuple(modes.EXPERIMENTAL))
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--max-cold-prefill", type=int,
                        help="largest cold prefill in tokens (32768 default, 65536 with kv-tiered4, 2048-token internal chunks)")
    parser.add_argument("--max-incremental-prefill", type=int,
                        help="largest uncached increment per request in tokens (AGENTIC default 32768)")
    parser.add_argument("--prefix-dir")
    parser.add_argument("--auto-purge", action="store_true")
    parser.add_argument("--no-auto-purge", action="store_true",
                        help="turn off the automatic memory preparation purge (on by default for flashnext)")
    known, remaining = parser.parse_known_args(argv)

    model = known.model or "flashnext"
    if known.mode and model != "flashnext":
        raise SystemExit("--mode applies to the flashnext model only")
    # A benchmark launch states its own conditions and must not be re-configured by a mode.
    benchmarking = any(value.startswith("--benchmark") for value in remaining)
    plan = None
    if model == "flashnext" and (known.mode or not benchmarking):
        try:
            serve = "--server" in remaining
            default_mode = modes.AGENTIC if serve else modes.CHAT
            plan = modes.plan_mode(
                known.mode or default_mode, pool_gb=known.pool_gb, enable=known.enable,
                prefix_dir=known.prefix_dir, context_limit=known.context_limit,
                serve=serve, max_cold_prefill=known.max_cold_prefill,
                max_incremental_prefill=known.max_incremental_prefill)
        except modes.ModeError as exc:
            raise SystemExit(f"veystrix: {exc}") from exc
    interpreter = _interpreter(model)

    # A source checkout runs the session script; the packaged app ships compiled modules only, so it runs the module.
    session_script = ROOT / "veystrix" / "session.py"
    run_session = [str(interpreter), "-u", str(session_script)] if session_script.is_file() \
        else [str(interpreter), "-u", "-m", "veystrix.session"]
    chat_args = [
        *run_session,
        "--model", model,
        "--preferences-file", known.preferences_file,
    ]
    if known.profile or plan:
        chat_args += ["--profile", known.profile or plan.profile]
    if known.model_path:
        chat_args += ["--model-path", known.model_path]
    if os.environ.get("VEYSTRIX_WORKSPACE") and "--workspace" not in remaining:
        chat_args += ["--workspace", os.environ["VEYSTRIX_WORKSPACE"]]
    chat_args.extend(remaining)
    if plan and plan.mode == modes.AGENTIC and "--server" not in chat_args:
        chat_args.append("--server")

    environment = dict(os.environ)
    if known.prefix_dir:
        environment["VEYSTRIX_CACHE_ROOT"] = known.prefix_dir
    if plan:
        modes.apply_mode(environment, plan)
        if not known.profile:
            environment["VEYSTRIX_PROFILE_EPHEMERAL"] = "1"
    if known.no_auto_purge:
        environment["FLASHNEXT_AUTO_PURGE"] = "0"
    elif known.auto_purge:
        environment["FLASHNEXT_AUTO_PURGE"] = "1"
    elif model == "flashnext" and not known.no_auto_purge and "FLASHNEXT_AUTO_PURGE" not in environment:
        # Autonomous memory preparation: a conditional, authorized file-cache purge is part of normal startup. It never
        # runs during a load or a request, needs sudo -n permission, and backs off; FLASHNEXT_AUTO_PURGE=0 or
        # --no-auto-purge turns it off.
        environment["FLASHNEXT_AUTO_PURGE"] = "1"
    if model == "flashnext":
        from models.flashnext.settings.launch import apply_chat_environment

        apply_chat_environment(environment)
    if model == "nemotron35-lightning":
        # Same supervised-server contract as Flash-Next: guard on, chunked prefill, agentic mode for a server.
        # Real inference is allowed by default; VEYSTRIX_NEMOTRON_AUTHORIZED=0 is a kill switch.
        if "--server" in chat_args:
            environment.setdefault("VEYSTRIX_MODE", "agentic")
            environment.setdefault("FLASHNEXT_WATCHDOG", "1")
            environment.setdefault("FLASHNEXT_WATCHDOG_MEMORY", "1")
        environment.setdefault("FLASHNEXT_CHAT_GUARD", "1")
        environment.setdefault("FLASHNEXT_PREFILL_CHUNK_TOKENS", "2048")
        if known.pool_gb:
            environment["VEYSTRIX_NEMOTRON_POOL_GB"] = known.pool_gb
        if known.context_limit is not None:
            environment["VEYSTRIX_CONTEXT_LIMIT"] = str(known.context_limit)
    if not session_script.is_file():
        environment["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), environment.get("PYTHONPATH", "")) if p)
    return ["/usr/bin/caffeinate", "-i", *chat_args], environment


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "workflow":
        from veystrix.workflows.__main__ import main as workflow_main
        return workflow_main(sys.argv[2:])

    # Keep this check before runtime discovery or setup: it must never download
    # dependencies or load the user's model.
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--model")
    probe.add_argument("--model-path", "--checkpoint", dest="checkpoint")
    probe.add_argument("--check", action="store_true")
    options, _ = probe.parse_known_args(sys.argv[1:])
    if options.model == "nemotron35-lightning":
        if options.checkpoint and not Path(options.checkpoint).expanduser().is_dir():
            raise SystemExit(f"Nemotron checkpoint folder does not exist: {options.checkpoint}; no download was attempted.")
        if options.check and not options.checkpoint:
            raise SystemExit("Nemotron --check requires --checkpoint /path/to/model; no download was attempted.")
        if options.check:
            import json
            from models.nemotron35.config import ConfigurationError, inspect_checkpoint
            try:
                print(json.dumps(inspect_checkpoint(options.checkpoint, hashes="--hashes" in sys.argv), indent=2))
            except ConfigurationError as exc:
                raise SystemExit(f"Nemotron check: {exc}") from exc
            return 0
        if os.environ.get("VEYSTRIX_NEMOTRON_AUTHORIZED") == "0":
            raise SystemExit("Nemotron is disabled by VEYSTRIX_NEMOTRON_AUTHORIZED=0.")
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        return setup_environment(sys.argv[2:])
    warning = branch_sync_warning()
    if warning:
        print(warning, file=sys.stderr)
    executable, environment = command(_default_checkpoint_args(sys.argv[1:]))
    os.execvpe(executable[0], executable, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
