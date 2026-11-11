#!/usr/bin/env python3
"""The chat. One interface, any model, either profile.

    python -m veystrix.session                      plain profile, flashnext
    python -m veystrix.session --profile agent      tools and the repository prompt

Commands come from veystrix.commands, prompts from veystrix.profiles, and the
model from a backend. Nothing here knows which model is answering.
"""
from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import inspect
import json
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veystrix import commands, preferences
from veystrix.api_keys import (
    DEFAULT_PATH as DEFAULT_API_KEYS_PATH,
    KeyStore,
    sanitized_environment,
)
from veystrix.agent import Limits, run_agent
from veystrix.backends.base import GenerationCancelled
from veystrix.profiles import system_prompt, tools_for
from veystrix.reasoning_policy import (
    effective_reasoning_effort,
    effective_think_budget,
    reasoning_effort_status,
    think_budget_status,
)
from veystrix.sampling import Sampling
from veystrix.tools.repo import Repo
from veystrix.tools.toolbox import Toolbox
from veystrix.terminal import read_prompt
from veystrix.text import ThinkingStreamFilter, ToolCallStreamFilter
from veystrix.ui import (
    AgentUI,
    AsyncWordAnimator,
    C,
    IngestGlow,
    filter_thinking,
    rss_gb,
)


class _TurnTelemetry:
    """Small fallback report for a turn stopped before its backend returned."""

    def __init__(self, pending: int):
        self.started = time.perf_counter()
        self.prefilled_at = None
        self.prompt_done = 0
        self.prompt_total = pending
        self.tokens = 0
        self.decode_started = None

    def progress(self, done: int, total: int) -> None:
        self.prompt_done = max(self.prompt_done, int(done))
        self.prompt_total = int(total)
        if done >= total:
            self.prefilled()

    def prefilled(self) -> None:
        if self.prefilled_at is None:
            self.prefilled_at = time.perf_counter()
            self.decode_started = self.prefilled_at

    def token(self, _value: int, _piece: str) -> None:
        self.prefilled()
        self.tokens += 1

    def stats(self):
        now = time.perf_counter()
        prefill_seconds = (
            (self.prefilled_at or now) - self.started
        )
        decode_seconds = (
            now - self.decode_started if self.decode_started is not None else 0.0
        )
        prompt_tokens = self.prompt_done or self.prompt_total
        return SimpleNamespace(
            finish="interrupted",
            tokens=self.tokens,
            rate=self.tokens / decode_seconds if decode_seconds else 0.0,
            seconds=decode_seconds,
            prompt_tokens=prompt_tokens,
            prompt_rate=prompt_tokens / prefill_seconds if prefill_seconds else 0.0,
            prefill_seconds=prefill_seconds,
            tail_tokens=0,
            tail_seconds=0.0,
        )


_GENERATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="veystrix-generation"
)


def _run_generation(call):
    """Run model work away from the signal-handling terminal loop.

    Every turn shares one persistent worker thread. Model runtimes
    bind device state to the generating thread, so a fresh thread per
    turn would invalidate the live cache and force a full prefill
    replay on each continuation.
    """
    stop = Event()
    future = _GENERATION_EXECUTOR.submit(call, stop.is_set)
    presses = 0
    while not future.done():
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            presses += 1
            stop.set()
    error = future.exception()
    result = None if error is not None else future.result()
    if error is not None and not presses:
        raise error
    return result, error, presses


def _on_model_thread(call, *args):
    """Run model-state work on the generation thread and wait for it.

    Arrays left lazy by an interrupted answer belong to the generation
    thread's GPU stream; evaluating them from the terminal thread fails with
    "There is no Stream(gpu, N) in current thread". Saving and loading a
    session therefore run where generation runs.
    """
    return _GENERATION_EXECUTOR.submit(call, *args).result()


def _backend_generate(backend, kwargs, should_cancel):
    """Pass cancellation only to backends that implement the shared hook."""
    try:
        parameters = inspect.signature(backend.generate).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "should_cancel" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["should_cancel"] = should_cancel
    return backend.generate(**kwargs)


def ask_approval(name: str, args: dict, input_fn=input) -> bool:
    """Require a reply typed after the approval prompt becomes visible."""
    try:
        if sys.stdin.isatty():
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
    subject = args.get("path") or args.get("command") or str(args)
    prompts = (
        f"{C['y']}approve {name}: {str(subject)[:140]}? [y/s/N] {C['0']}",
        f"{C['y']}type y to approve or n to deny: {C['0']}",
    )
    for prompt in prompts:
        try:
            answer = input_fn(prompt)
        except EOFError:
            return False
        letters = re.sub(r"[^a-z]", "", answer.lower())
        if letters in ("y", "yes", "s", "sim"):
            return True
        if letters in ("n", "no", "nao"):
            return False
    return False


class Session:
    """What a command sees: the settings, the backend and the workspace."""

    def __init__(
        self,
        backend,
        profile: str,
        prefs: dict,
        path: str,
        api_keys_path: str = DEFAULT_API_KEYS_PATH,
        migrate_system_prompt: bool = True,
    ):
        self.backend = backend
        self.profile = profile
        self.preferences = prefs
        self.preferences_path = path
        self.running = True
        self.tools = None
        self.api_keys = KeyStore(api_keys_path)
        if migrate_system_prompt:
            self._migrate_system_prompt()
        self._build_tools()
        self.opened = False
        self.server_requested = False

    @property
    def repo(self):
        return self.tools.repo if self.tools is not None else None

    def _build_tools(self):
        if self.profile == "agent":
            self.tools = Toolbox.build(Repo(self.preferences["workspace"]))
        else:
            self.tools = None

    def save_preferences(self):
        preferences.save(self.preferences, self.preferences_path)
        mirror_preferences(self.backend, self.preferences, self.profile)

    def stop(self):
        self.running = False

    def start_server(self):
        self.server_requested = True
        self.running = False

    def current_system_prompt(self):
        try:
            saved = self.system_prompt_path().read_text().strip()
        except OSError:
            saved = ""
        return saved or system_prompt(
            self.profile, self.preferences["workspace"]
        )

    def system_prompt_path(self) -> Path:
        return Path(self.preferences_path).expanduser().resolve().with_name(
            f"system-prompt-{self.profile}.txt"
        )

    def _migrate_system_prompt(self):
        legacy = self.preferences.get("system_prompt", "").strip()
        if not legacy:
            return
        path = self.system_prompt_path()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_text(legacy + "\n")
            os.chmod(path, 0o600)
        self.preferences["system_prompt"] = ""
        preferences.save(self.preferences, self.preferences_path)

    def set_system_prompt(self, value: str):
        value = value.strip()
        path = self.system_prompt_path()
        if value:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_text(value + "\n")
            os.chmod(path, 0o600)
        else:
            path.unlink(missing_ok=True)
        self.preferences["system_prompt"] = ""
        self.save_preferences()

    def edit_system_prompt(self):
        path = self.system_prompt_path()
        if not path.is_file():
            self.set_system_prompt(self.current_system_prompt())
        subprocess.call(
            [os.environ.get("EDITOR", "nano"), str(path)],
            env=sanitized_environment(),
        )

    def set_profile(self, profile: str):
        changed = profile != self.profile
        self.profile = profile
        self.preferences["profile"] = profile
        self.save_preferences()
        if changed:
            self.reset()
        return changed

    def list_api_keys(self):
        return self.api_keys.status()

    def set_api_key(self, service: str):
        try:
            secret = getpass.getpass(f"{service} API key: ")
            self.api_keys.set(service, secret)
        except (EOFError, KeyboardInterrupt):
            return "API key input cancelled"
        except (OSError, ValueError) as exc:
            return f"could not save API key: {exc}"
        self._build_tools()
        return f"{service.lower()} API key saved in Application Support"

    def delete_api_key(self, service: str):
        try:
            deleted = self.api_keys.delete(service)
        except (OSError, ValueError) as exc:
            return f"could not delete API key: {exc}"
        self._build_tools()
        return f"{service.lower()} API key {'deleted' if deleted else 'not stored'}"

    def reset(self):
        self.backend.reset()
        self._build_tools()
        self.opened = False

    def save_session(self, name):
        return _on_model_thread(self.backend.save_session, name)

    def load_session(self, name):
        result = _on_model_thread(self.backend.load_session, name)
        self.opened = bool(self.backend.tape or self.backend.pending)
        return result

    def list_sessions(self):
        return self.backend.list_sessions()

    def delete_session(self, name):
        return self.backend.delete_session(name)

    def model_settings(self, argument: str):
        configure = getattr(self.backend, "configure", None)
        if configure is None:
            return "this model does not expose configurable settings"
        try:
            text = argument.strip()
            if not text or text == "all":
                prefs = self.preferences
                answer_budget, requested_think, effective_think, _ = _effective_turn_budgets(
                    self.backend, prefs, self.profile
                )
                budget = (
                    f"{answer_budget} answer + "
                    f"{effective_think} reasoning"
                    if prefs["thinking_enabled"] and requested_think is not None
                    else f"{answer_budget} answer (reasoning shares it)"
                )
                shared = (
                    "Chat settings\n"
                    f"  sampling            {Sampling.from_preferences(prefs).describe()}\n"
                    f"  effort              {prefs['effort']}\n"
                    f"  thinking            {'on' if prefs['thinking_enabled'] else 'off'}\n"
                    f"  token-budget        {budget}"
                )
                return shared + "\n\n" + configure(text)
            result = configure(argument)
            self._remember_backend_preferences(text)
            return result
        except ValueError as exc:
            return f"could not change settings: {exc}"

    def _remember_backend_preferences(self, argument: str):
        """Save the model settings that preferences own after a /config change."""
        name = argument.split(maxsplit=1)[0].lower() if argument else ""
        if name not in {"gpu-keepwarm", "keepwarm", "flashnext_gpu_keepwarm"}:
            return
        try:
            from models.flashnext.expert_cache import gpu_keepwarm
        except ImportError:
            return
        enabled = bool(gpu_keepwarm())
        if self.preferences.get("flashnext_gpu_keepwarm") != enabled:
            self.preferences["flashnext_gpu_keepwarm"] = enabled
            self.save_preferences()

    def status(self):
        prefs = self.preferences
        requested_effort = prefs["effort"]
        routing = getattr(self.backend, "routing_profile", None)
        answer_budget, requested_think, _, _ = _effective_turn_budgets(
            self.backend, prefs, self.profile
        )
        lines = [
            f"{C['b']}profile{C['0']}   {self.profile}"
            f"  model {prefs['model']}"
            f"{'  routing ' + routing if routing else ''}"
            f"{'  workspace ' + str(self.repo.root) if self.repo else ''}",
            f"{C['b']}options{C['0']}   "
            f"thinking={'on' if prefs['thinking_enabled'] else 'off'}  "
            f"display={'show' if prefs['show_thinking'] else 'hide'}  "
            f"animate={'on' if prefs['animate'] else 'off'}  "
            f"{reasoning_effort_status(requested_effort, getattr(self.backend, 'model_path', None))}  "
            f"{'greedy' if prefs['temperature'] <= 0 else 'sampled'}  "
            f"answer-tokens={answer_budget}  "
            f"{think_budget_status(requested_think, requested_effort, getattr(self.backend, 'model_path', None))}",
            f"{C['b']}context{C['0']}   {len(self.backend.tape)} tokens",
            f"{C['b']}memory{C['0']}    RSS {rss_gb():.2f} GB",
        ]
        try:
            import mlx.core as mx

            lines.append(f"{C['b']}mlx{C['0']}       "
                         f"active {mx.get_active_memory() / 1e9:.2f} GB, "
                         f"cache {mx.get_cache_memory() / 1e9:.2f} GB")
        except Exception:
            pass
        return "\n".join(lines)


def _effective_turn_budgets(
    backend, prefs: dict, profile: str,
) -> tuple[int, int | None, int | None, int]:
    """Resolve answer, reasoning, and total budgets for one interactive turn."""
    checkpoint = getattr(backend, "model_path", None)
    answer = preferences.answer_limit(
        prefs,
        preferences.DEFAULT_PLAIN_ANSWER_TOKENS
        if profile == "plain" else preferences.DEFAULT_ANSWER_TOKENS,
    )
    requested_think = preferences.separate_think_limit(prefs)
    think = effective_think_budget(
        requested_think, prefs["effort"], checkpoint,
    )
    if requested_think is None:
        # ``None`` is the historical shared-total sentinel: ``answer`` is the
        # one generation ceiling. Keep the effective cap in ``think`` so the
        # backend can close reasoning at that point, but do not add it to the
        # total or turn 2,048 total tokens into 2,048 + 4,096.
        total = answer
    else:
        total = (
            -1
            if answer < 0 or think is None or think < 0
            else answer + think
        )
    return answer, requested_think, think, total


def mirror_preferences(backend, prefs: dict, profile: str) -> None:
    """Copy preference state a backend needs onto it.

    `/settings` reads these to show what the next turn will do. The dedicated
    commands stay the writers, so each value has one owner.
    """
    backend.thinking_enabled = prefs["thinking_enabled"]
    if not hasattr(backend, "sampling"):
        return
    backend.sampling = Sampling.from_preferences(prefs)
    backend.reasoning_effort = effective_reasoning_effort(
        prefs["effort"], getattr(backend, "model_path", None)
    )
    answer, _, think, _ = _effective_turn_budgets(backend, prefs, profile)
    backend.think_budget = think
    backend.answer_budget = answer


def build_backend(name: str, args, prefs: dict):
    if name == "flashnext":
        from veystrix.backends.flashnext import FlashNextBackend
        from models.flashnext.settings import get_registry as get_flashnext_registry

        values = get_flashnext_registry().cli_values(args, "flashnext")
        values["model_path"] = args.model_path
        values["session_dir"] = args.session_dir or "~/.cache/flashnext/sessions"
        # `--resident-experts` defaults to None, so a value here was typed,
        # including an abbreviation argparse expanded (`--pinned 32`). Only
        # that beats the checkpoint policy. The checkpoint itself resolves
        # exactly the way the backend will resolve it (alias, env,
        # auto-select).
        explicit_resident = getattr(args, "resident_experts", None) is not None
        policy_source = None
        if not explicit_resident:
            from veystrix.checkpoints import resolve_flashnext
            from models.flashnext.checkpoint_policy import (
                resolve_resident_experts,
            )

            try:
                resolved_checkpoint = resolve_flashnext(args.model_path)
            except Exception:
                resolved_checkpoint = None
            try:
                policy_value = resolve_resident_experts(
                    None, resolved_checkpoint)
            except Exception:
                policy_value = None
            if policy_value is not None:
                values["resident_experts"] = policy_value
                policy_source = "checkpoint policy"
        backend = FlashNextBackend(**values)
        cli_args = set(sys.argv[1:])
        backend._setting_sources = {
            setting.name: "CLI"
            for setting in get_flashnext_registry().for_backend("flashnext")
            if setting.cli_dest
            and any(flag in cli_args for flag in setting.cli_flags)
        }
        if explicit_resident:
            backend._setting_sources["resident-experts"] = "CLI"
        elif policy_source is not None:
            backend._setting_sources["resident-experts"] = policy_source
        from models.flashnext.expert_cache import set_gpu_keepwarm

        set_gpu_keepwarm(bool(prefs.get("flashnext_gpu_keepwarm", True)))
        backend._setting_sources["gpu-keepwarm"] = "preferences"
        mirror_preferences(backend, prefs, prefs["profile"])
        return backend
    if name == NEMOTRON:
        if not args.model_path:
            raise SystemExit("--model-path is required for model 'nemotron35-lightning'")
        from veystrix.backends.nemotron35 import NemotronBackend
        from models.flashnext.settings.modes import context_limit_from_environment

        backend = NemotronBackend(
            model_path=args.model_path,
            session_dir=(args.session_dir or "~/.cache/veystrix/nemotron35/sessions"),
            context_tokens=context_limit_from_environment(),
        )
        mirror_preferences(backend, prefs, prefs["profile"])
        return backend
    raise SystemExit(
        f"unknown model {name!r}. Use 'flashnext' or 'nemotron35-lightning'.")


NEMOTRON = "nemotron35-lightning"
# Models that get the shared power gate, the swap/thermal/battery chat guard and the supervised-server rule.
GUARDED_MODELS = ("flashnext", NEMOTRON)


def _apply_backend_limits(backend) -> None:
    """A backend that admitted a smaller context than the environment asks for tightens the server limits."""
    context = getattr(backend, "context_limit", None)
    if not isinstance(context, int) or context <= 0:
        return
    current = os.environ.get("VEYSTRIX_CONTEXT_LIMIT")
    os.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(min(context, int(current)) if current and current.isdigit() else context)
    for name in ("VEYSTRIX_MAX_COLD_PREFILL", "VEYSTRIX_MAX_INCREMENTAL_PREFILL"):
        value = os.environ.get(name)
        os.environ[name] = str(min(context, int(value)) if value and value.isdigit() else context)


def _apply_seed(seed: int | None) -> None:
    if seed is not None:
        import mlx.core as mx

        mx.random.seed(seed)


def _enforce_power_gate(prefs, benchmarking, display) -> None:
    """Battery use is allowed; a battery below the floor (default 25%) refuses the start, before any model load."""
    if prefs["model"] not in GUARDED_MODELS or benchmarking:
        return
    from models.flashnext import power_gate

    verdict = power_gate.check()
    if verdict.ok:
        if verdict.source == "battery":
            print(f"power: {verdict.message}", file=display, flush=True)
        return
    print(f"power: {verdict.message}\nNothing was loaded.", file=sys.stderr, flush=True)
    raise SystemExit(power_gate.POWER_EXIT_CODE)


def _prepare_startup(prefs, benchmarking, display):
    """Autonomous memory preparation before any model load (CHAT and AGENTIC). Returns (recovery, verdict).

    Bounded rounds of safe recovery (Veystrix-owned memory, a conditional file-cache purge, optional graceful closing of
    owner-listed apps), a settle wait and a fresh reading, then admission. AGENTIC stops here with ONE actionable report
    (exit 69, nothing loaded) when no safe recovery can admit the launch. CHAT only prepares for the 9 GiB gate; the guard
    that follows keeps deciding. See models/flashnext/resource_prep.py."""
    if prefs["model"] != "flashnext" or benchmarking:
        return None, None
    from models.flashnext import prefill_readiness as readiness
    from models.flashnext import memory_policy, memory_waiver, purge_recovery, resource_prep
    from models.flashnext.app_closer import AppCloser

    agentic = _agentic_mode()
    recovery = purge_recovery.from_environment()
    if agentic and (memory_policy.is_swap_growth_policy() or memory_waiver.active()):
        # Even when the result is advisory, report and assess the configured
        # workload, not the unrelated 4K default of readiness.live().
        return recovery, readiness.live_tiered()
    if recovery is not None and agentic:
        recovery._blocker = readiness.purge_blocker_tiered      # no purge for a refusal that file cache cannot fix
    out = lambda text: print(text, file=display, flush=True)
    preparer = resource_prep.ResourcePreparer(agentic=agentic, recovery=recovery, closer=AppCloser(), out=out)
    report = preparer.prepare()
    if report.steps and any(step.action == "done" for step in report.steps):
        for step in report.steps:
            if step.action == "done":
                print(f"memory preparation: {step.name}: {step.detail} "
                      f"(available {step.available_before / 2 ** 30:.2f} -> {step.available_after / 2 ** 30:.2f} GiB)",
                      file=display, flush=True)
    if agentic:
        if not report.admitted:
            readiness.log(report.verdict)
            print(report.failure_report(), file=sys.stderr, flush=True)
            raise SystemExit(readiness.ADMISSION_EXIT_CODE)
        return recovery, report.verdict
    if recovery is not None:
        print(f"auto-purge: start gate {'READY' if report.admitted else 'NOT reached'} "
              f"({report.verdict.categories['gate_available'] / 2 ** 30:.2f} GiB available, {report.seconds:.0f}s)"
              + ("" if report.admitted else f"; the safety guard decides: {report.stop_reason}"), file=display, flush=True)
    return recovery, None


def _apply_tier(verdict, display) -> None:
    """Enforce the admitted size: a smaller context and cold-prefill limit than the default only when memory required it."""
    tier = getattr(verdict, "tier", None)
    if tier is None:
        return
    from models.flashnext.settings import modes

    current = modes.context_limit_from_environment()
    required = os.environ.get("VEYSTRIX_CONTEXT_REQUIRED") == "1"
    env_cold = modes.max_cold_prefill_from_environment()
    if required and current:
        limit = current
        cold_prefill = min(tier.max_cold_prefill, env_cold) if env_cold else tier.max_cold_prefill
    else:
        limit = min(tier.context_limit, current) if current else tier.context_limit
        cold_prefill = min(tier.max_cold_prefill, env_cold) if env_cold else tier.max_cold_prefill
    os.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(limit)
    os.environ["VEYSTRIX_MAX_COLD_PREFILL"] = str(cold_prefill)
    if tier.name != "standard" and not required:
        print(f"admitted at tier {tier.name}: context limit {limit}, cold prefill up to {cold_prefill} tokens "
              f"(memory does not admit the standard tier right now; larger requests are refused, never truncated)",
              file=display, flush=True)
    elif tier.max_cold_prefill < (env_cold or modes.DEFAULT_MAX_COLD_PREFILL):
        print(f"admitted at tier {tier.name}: cold prefill capped at {cold_prefill} tokens by memory tier",
              file=display, flush=True)


def _start_watchdog(chat_guard, display):
    """AGENTIC: spawn the out-of-process stall watchdog (FLASHNEXT_WATCHDOG=1). It shares no interpreter with this process."""
    from models.flashnext import progress

    if not progress.enabled():
        return None
    process = progress.start_watchdog(trace_path=str(chat_guard.path) if chat_guard is not None else None)
    if process is not None:
        import atexit

        atexit.register(progress.PROGRESS.stage, "stopping")
        print(f"watchdog       ON  (separate process {process.pid}; stage deadlines "
              f"{progress.status_dir()})", file=display, flush=True)
    return process


def _agentic_mode() -> bool:
    return os.environ.get("VEYSTRIX_MODE") == "agentic"


def _admit_agentic(prefs, benchmarking, display, verdict=None):
    """AGENTIC only: refuse to load when the cold-prefill budget is not met (the 9 GiB gate is part of the verdict).

    Runs before the guard and before any model load; returns the verdict, or exits with the admission code. ``verdict`` is
    the tiered one from ``_prepare_startup``; without it the standard-size verdict is read here."""
    if prefs["model"] != "flashnext" or benchmarking or not _agentic_mode():
        return None
    from models.flashnext import prefill_readiness as readiness
    from models.flashnext import memory_policy
    from models.flashnext import memory_waiver

    if verdict is None:
        verdict = readiness.live()
    readiness.log(verdict)
    if memory_policy.is_swap_growth_policy() or memory_waiver.active():
        print(memory_policy.banner(), file=display, flush=True)
        tier = getattr(verdict, "tier", None)
        if tier is None:
            tiers = readiness.active_tiers()
            if tiers:
                import dataclasses
                tier = tiers[0] if memory_policy.is_swap_growth_policy() else tiers[-1]
                verdict = dataclasses.replace(verdict, tier=tier)
        if not verdict.admitted and not memory_policy.is_absolute_swap_policy():
            print(f"memory admission note: former gates are telemetry-only under swap-growth-only policy; "
                  f"stopping condition is swap growth of 2.5 GiB over session baseline.",
                  file=display, flush=True)
        else:
            print("prefill readiness: " + verdict.checks[0].detail, file=display, flush=True)
        _apply_tier(verdict, display)
        return verdict
    if not verdict.admitted:
        print(readiness.render(verdict), file=sys.stderr, flush=True)
        print("Nothing was loaded. No guard was weakened; free the memory named above and start again.",
              file=sys.stderr, flush=True)
        raise SystemExit(readiness.ADMISSION_EXIT_CODE)
    print("prefill readiness: " + verdict.checks[0].detail, file=display, flush=True)
    _apply_tier(verdict, display)
    return verdict


def _purge_monitor(recovery, backend, display) -> None:
    """Watch every FLASHNEXT_PURGE_CHECK_SECONDS while idle (chat prompt or server), stop cleanly at exit."""
    if recovery is None:
        return
    import atexit
    from models.flashnext import purge_recovery

    if hasattr(backend, "reclaim_owned_memory"):
        recovery.add_reclaim(backend.reclaim_owned_memory)
    monitor = purge_recovery.Monitor(recovery).start()
    purge_recovery.register(recovery)             # also recovers at the request boundary (purge_recovery.prepare_for_request)
    atexit.register(monitor.stop)
    atexit.register(purge_recovery.register, None)
    print(f"auto-purge: idle monitor every {recovery.cfg.interval_s:g}s and before each request (purge only when "
          f"justified; log {recovery.log_path})", file=display, flush=True)


def _print_mode_banner(prefs, args, display, chat_guard) -> None:
    """Show the launch mode, checkpoint, pool, cache, chunk policy, memory readiness and supervisor state."""
    if prefs["model"] != "flashnext":
        return
    from models.flashnext.settings import modes

    plan = modes.plan_from_environment()
    if plan is None:
        return
    available = None
    try:
        from models.flashnext.memory_manager import read_system

        available = read_system().get("available", 0) / 1e9 or None
    except Exception:
        pass
    trace = getattr(getattr(chat_guard, "path", None), "name", None)
    print(modes.banner(plan, checkpoint=args.model_path, available_gb=available, guard_trace=trace,
                       auto_purge=os.environ.get("FLASHNEXT_AUTO_PURGE") == "1"),
          file=display, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    from veystrix.model_settings import FLASHNEXT_DEFAULTS
    parser.add_argument(
        "--model", default=None,
        choices=("flashnext", NEMOTRON),
    )
    parser.add_argument("--profile", default=None, choices=("plain", "agent"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", default=None)
    parser.add_argument(
        "--threshold", type=float, default=FLASHNEXT_DEFAULTS["threshold"]
    )
    parser.add_argument(
        "--resident-experts", "--pinned-experts", dest="resident_experts", type=int,
        # None means "not given", so a checkpoint policy can apply. The
        # backend falls back to FLASHNEXT_DEFAULTS when nothing supplies one.
        default=None,
    )
    parser.add_argument(
        "--pin-budget-gb", type=float,
        default=FLASHNEXT_DEFAULTS["pin_budget_gb"],
    )
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--standard", dest="routing_profile",
                         action="store_const", const="standard")
    routing.add_argument("--fast-quality", dest="routing_profile",
                         action="store_const", const="fast-quality")
    routing.add_argument("--exact-quality", dest="routing_profile",
                         action="store_const", const="exact-quality")
    routing.add_argument("--cache-aware", dest="routing_profile",
                         action="store_const", const="cache-aware")
    routing.add_argument("--fused-quality", dest="routing_profile",
                         action="store_const", const="fused-quality")
    parser.set_defaults(routing_profile=FLASHNEXT_DEFAULTS["routing"])
    parser.add_argument(
        "--swap-epsilon", type=float,
        default=FLASHNEXT_DEFAULTS["swap_epsilon"],
    )
    parser.add_argument(
        "--tail-experts", type=int, default=FLASHNEXT_DEFAULTS["tail_experts"]
    )
    parser.add_argument(
        "--tail-warmup", type=int, default=FLASHNEXT_DEFAULTS["tail_warmup"]
    )
    parser.add_argument(
        "--fusion-block", type=int, default=FLASHNEXT_DEFAULTS["fusion_block"]
    )
    parser.add_argument(
        "--fusion-min-margin", type=float,
        default=FLASHNEXT_DEFAULTS["fusion_min_margin"],
    )
    parser.add_argument(
        "--fusion-min-block", type=int,
        default=FLASHNEXT_DEFAULTS["fusion_min_block"],
    )
    parser.add_argument(
        "--fusion-margin-tokens", type=int,
        default=FLASHNEXT_DEFAULTS["fusion_margin_tokens"],
    )
    parser.add_argument(
        "--fusion-max-prompt", type=int,
        default=FLASHNEXT_DEFAULTS["fusion_max_prompt"],
    )
    parser.add_argument(
        "--fusion-model", default=FLASHNEXT_DEFAULTS["fusion_model"]
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--repetition-context-size", type=int, default=512)
    parser.add_argument("--backtrack-bias", type=float, default=0.0)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed sampling for reproducible chat runs",
    )
    parser.add_argument("--think-budget", type=int, default=None)
    parser.add_argument("--think", dest="thinking_enabled",
                        action="store_true", default=None)
    parser.add_argument("--no-think", dest="thinking_enabled",
                        action="store_false")
    parser.add_argument("--show-think", dest="show_thinking",
                        action="store_true", default=None)
    parser.add_argument("--hide-think", dest="show_thinking",
                        action="store_false")
    parser.add_argument("--preferences-file", default=preferences.DEFAULT_PATH)
    parser.add_argument("--api-keys-file", default=DEFAULT_API_KEYS_PATH)
    parser.add_argument("--server", action="store_true",
                        help="run a local API server without terminal chat")
    parser.add_argument("--allow-origin", action="append", default=[],
                        metavar="ORIGIN",
                        help="let a browser page from ORIGIN use the server; "
                             "repeatable, or '*' for any. Without this, "
                             "requests carrying an Origin header are refused, "
                             "so a site you visit cannot use your model")
    parser.add_argument("--host", default="127.0.0.1",
                        help="server bind address; default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080,
                        help="server port; default: 8080")
    parser.add_argument("--server-api-key",
                        default=os.environ.get("VEYSTRIX_SERVER_API_KEY"),
                        help="optional Bearer or x-api-key value")
    parser.add_argument("--benchmark-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-prompt", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-product-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--benchmark-chat-parity",
        choices=("raw", "formatted", "rendered", "sampled", "thinking", "thinking-sampled"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--benchmark-window", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-tokens", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-label", default="chat.sh", help=argparse.SUPPRESS)
    parser.add_argument(
        "--benchmark-product-sampling",
        choices=("greedy", "configured"), default="greedy",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    benchmarking = bool(
        args.benchmark_json or args.benchmark_product_json or args.benchmark_chat_parity
    )
    if benchmarking and not args.benchmark_prompt:
        parser.error("benchmark mode requires --benchmark-prompt")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.host not in ("127.0.0.1", "localhost", "::1") \
            and not args.server_api_key:
        parser.error("a non-local server requires --server-api-key")
    if args.max_tokens is not None and args.max_tokens != -1 and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive or -1")
    if args.think_budget is not None \
            and args.think_budget != -1 and args.think_budget <= 0:
        parser.error("--think-budget must be positive or -1")
    prefs = preferences.load(args.preferences_file)
    saved_profile = prefs.get("profile")
    if (
        args.tail_experts < 1
        or (args.resident_experts is not None and args.resident_experts < 1)
        or args.tail_warmup < 1
    ):
        parser.error("Flash-Next expert counts and warmup must be positive")
    if not 0.01 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.01 and 1.0")
    if not 0 <= args.swap_epsilon <= 1.0:
        parser.error("--swap-epsilon must be between 0 and 1.0")
    if not 0 <= args.pin_budget_gb <= 64:
        parser.error("--pin-budget-gb must be between 0 and 64")
    if (
        args.fusion_block < 1
        or args.fusion_min_block < 1
        or args.fusion_margin_tokens < 0
        or args.fusion_max_prompt < 1
        or args.fusion_min_margin < 0
    ):
        parser.error("Flash-Next fusion settings are outside their valid range")
    if args.fusion_min_block > args.fusion_block:
        parser.error("--fusion-min-block cannot exceed --fusion-block")
    # a flag beats the saved value, and becomes the saved value
    if args.model:
        prefs["model"] = args.model
    if args.profile:
        prefs["profile"] = args.profile
    if args.workspace:
        prefs["workspace"] = args.workspace
    if args.max_tokens is not None:
        prefs["max_tokens"] = args.max_tokens
    if args.think_budget is not None:
        prefs["think_budget"] = args.think_budget
    if args.thinking_enabled is not None:
        prefs["thinking_enabled"] = args.thinking_enabled
        if args.thinking_enabled and args.show_thinking is None:
            prefs["show_thinking"] = True
    if args.show_thinking is not None:
        prefs["show_thinking"] = args.show_thinking
    if prefs["model"] == "flashnext":
        from veystrix.checkpoints import resolve_flashnext

        explicit_checkpoint = args.model_path
        environment_checkpoint = os.environ.get("VEYSTRIX_FLASHNEXT_MODEL")
        saved_checkpoint = prefs["flashnext_checkpoint"]
        choice = (
            explicit_checkpoint
            or environment_checkpoint
            or saved_checkpoint
            or None
        )
        saved_only = bool(
            saved_checkpoint and not explicit_checkpoint and not environment_checkpoint
        )
        try:
            args.model_path = str(resolve_flashnext(
                choice, allow_stale_fallback=saved_only
            ))
        except ValueError as exc:
            parser.error(str(exc))
        if explicit_checkpoint or saved_only or not (environment_checkpoint or saved_checkpoint):
            prefs["flashnext_checkpoint"] = args.model_path
    elif prefs["model"] == NEMOTRON:
        from veystrix.checkpoints import resolve_nemotron35

        choice = args.model_path or os.environ.get("VEYSTRIX_NEMOTRON_MODEL") or prefs.get("nemotron35_checkpoint") or None
        try:
            args.model_path = str(resolve_nemotron35(choice))
        except ValueError as exc:
            parser.error(str(exc))
        prefs["nemotron35_checkpoint"] = args.model_path
    if benchmarking and prefs["profile"] != "plain":
        parser.error("benchmark mode requires --profile plain")
    if args.benchmark_chat_parity and prefs["model"] != "flashnext":
        parser.error("chat parity requires --model flashnext")
    # Fail closed: a flashnext server needs the safety configuration the launcher (./chat.sh --mode agentic) applies. A direct
    # `session.py --server` launch ran the 2026-11-09 DeepSeek Harness incident with no guard, no chunking, no cache.
    from models.flashnext import supervision

    refusal = supervision.check(os.environ, model=prefs["model"], server=bool(args.server), benchmarking=benchmarking)
    if refusal:
        print(refusal, file=sys.stderr, flush=True)
        return supervision.EXIT_CODE
    # A benchmark uses temporary command-line conditions. It must not change
    # the user's next interactive chat, as an earlier 20-token probe did.
    if not benchmarking:
        to_save = dict(prefs)
        if os.environ.get("VEYSTRIX_PROFILE_EPHEMERAL") == "1" and saved_profile is not None:
            to_save["profile"] = saved_profile
        preferences.save(to_save, args.preferences_file)

    signal.signal(signal.SIGTSTP, signal.SIG_IGN)

    began = time.time()
    product_json = args.benchmark_product_json
    display = sys.stderr if benchmarking else sys.stdout
    _enforce_power_gate(prefs, benchmarking, display)
    recovery, prepared = _prepare_startup(prefs, benchmarking, display)
    _admit_agentic(prefs, benchmarking, display, prepared)
    chat_guard = None
    if (prefs["model"] in GUARDED_MODELS and not benchmarking
            and os.environ.get("FLASHNEXT_CHAT_GUARD") == "1"):
        import tempfile
        from models.flashnext.chat_guard import ChatGuard

        trace = Path(tempfile.gettempdir()) / f"veystrix-chat-{os.getpid()}.jsonl"
        chat_guard = ChatGuard(trace).start()
        print(f"FlashNext safety trace: {trace}", file=display, flush=True)
    _start_watchdog(chat_guard, display)
    _print_mode_banner(prefs, args, display, chat_guard)
    print(f"{C['dim']}loading {prefs['model']}...{C['0']}", flush=True, file=display)
    from models.flashnext.purge_recovery import ACTIVITY

    from models.flashnext import progress as _progress

    _progress.PROGRESS.stage("loading")
    with ACTIVITY.loading_model():          # a cache purge never runs while the checkpoint loads
        if benchmarking:
            with contextlib.redirect_stdout(sys.stderr):
                backend = build_backend(prefs["model"], args, prefs)
        else:
            backend = build_backend(prefs["model"], args, prefs)
    _progress.PROGRESS.stage("idle")
    _apply_backend_limits(backend)
    if chat_guard is not None and hasattr(backend, "store"):
        chat_guard.tracer.store = backend.store
    _apply_seed(args.seed)
    session = Session(
        backend,
        prefs["profile"],
        prefs,
        args.preferences_file,
        args.api_keys_file,
        migrate_system_prompt=not benchmarking,
    )
    ready_seconds = time.time() - began
    _purge_monitor(recovery, backend, display)
    print(f"{C['dim']}ready in {time.time() - began:.1f}s  "
          f"{prefs['model']} / {prefs['profile']}  "
          f"use /help for commands{C['0']}\n", file=display)
    if args.benchmark_chat_parity:
        from models.flashnext.tests.bench.bench_chat_parity import run_chat_child

        run_chat_child(session, args.benchmark_chat_parity, args.benchmark_prompt)
        return 0
    if args.benchmark_json:
        print(json.dumps(run_benchmark(session, args.benchmark_prompt, ready_seconds)))
        return 0
    if product_json:
        run_product_benchmark(
            session, args.benchmark_prompt, args.benchmark_window, args.benchmark_label,
            args.benchmark_tokens, args.benchmark_product_sampling,
        )
        return 0
    if args.server:
        from veystrix.server import serve

        from models.flashnext.settings.modes import (
            context_limit_from_environment,
            max_cold_prefill_from_environment,
            max_incremental_prefill_from_environment,
        )

        serve(session, args.host, args.port, args.server_api_key,
              args.allow_origin, context_limit=context_limit_from_environment(),
              max_cold_prefill=max_cold_prefill_from_environment(),
              max_incremental_prefill=max_incremental_prefill_from_environment())
        return 0
    print(commands.render_help(session.profile) + "\n")

    glow = IngestGlow()
    interrupt_level = 0

    def handle_interrupt(level: int) -> bool:
        if level == 1:
            print(f"\n{C['dim']}answer stopped; Ctrl+C again closes the conversation, "
                  f"a third time quits{C['0']}\n")
            return False
        if level == 2:
            session.reset()
            print(f"\n{C['dim']}conversation closed; Ctrl+C again quits{C['0']}\n")
            return False
        print()
        return True

    while session.running:
        try:
            prompt = read_prompt(f"{C['b']}you>{C['0']} ")
        except KeyboardInterrupt:
            interrupt_level += 1
            if handle_interrupt(interrupt_level):
                return 0
            continue
        except EOFError:
            print()
            return 0
        if not prompt.strip():
            continue

        try:
            answered = commands.dispatch(session, prompt)
            if answered is not None:
                if answered:
                    print(answered + "\n")
                interrupt_level = 0
                continue

            if session.profile == "agent":
                presses = run_turn_agent(session, prompt)
            else:
                presses = run_turn_plain(session, prompt, glow)
            if presses:
                interrupt_level += presses
                if handle_interrupt(interrupt_level):
                    return 0
            else:
                interrupt_level = 0
        except KeyboardInterrupt:
            glow.finish()
            interrupt_level += 1
            if handle_interrupt(interrupt_level):
                return 0
    if session.server_requested:
        from models.flashnext import supervision

        refusal = supervision.check(os.environ, model=prefs["model"], server=True, benchmarking=benchmarking)
        if refusal:
            print(refusal, file=sys.stderr, flush=True)
            return supervision.EXIT_CODE

        from veystrix.server import serve
        from models.flashnext.settings.modes import (
            context_limit_from_environment,
            max_cold_prefill_from_environment,
            max_incremental_prefill_from_environment,
        )

        serve(session, args.host, args.port, args.server_api_key,
              args.allow_origin, context_limit=context_limit_from_environment(),
              max_cold_prefill=max_cold_prefill_from_environment(),
              max_incremental_prefill=max_incremental_prefill_from_environment())
        return 0
    return 0


def open_or_continue(session, prompt: str) -> None:
    """First turn carries the system prompt; later turns just add the user."""
    if not session.opened:
        session.backend.open_conversation(
            session.current_system_prompt(),
            prompt,
            tools=tools_for(session.profile),
            enable_thinking=session.preferences["thinking_enabled"],
            reasoning_effort=effective_reasoning_effort(
                session.preferences["effort"],
                getattr(session.backend, "model_path", None),
            ))
        session.opened = True
    else:
        session.backend.append_user(
            prompt, enable_thinking=session.preferences["thinking_enabled"])


def run_turn_plain(session, prompt: str, glow: IngestGlow) -> int:
    open_or_continue(session, prompt)
    prefs = session.preferences
    answer_budget, _, think_budget, limit = _effective_turn_budgets(
        session.backend, prefs, "plain"
    )
    thinking = ThinkingStreamFilter(
        prefs["thinking_enabled"], prefs["show_thinking"]
    )
    animator = AsyncWordAnimator(enabled=prefs["animate"])
    telemetry = _TurnTelemetry(len(session.backend.pending))

    def show(piece: str) -> None:
        visible = thinking.feed(piece)
        if visible:
            animator.feed(visible, C["gray"] if thinking.inside else "")

    glow.start(len(session.backend.pending))
    began = time.time()
    budget_state = getattr(session.backend, "_interactive_budgets", None)
    if hasattr(session.backend, "_interactive_budgets"):
        session.backend._interactive_budgets = (
            answer_budget,
            think_budget,
        )
    def progress(done, total):
        telemetry.progress(done, total)
        glow.update(done, total)

    def prefilled():
        telemetry.prefilled()
        glow.finish()

    generate_kwargs = {
        "max_tokens": limit,
        "out": show if prefs["stream_answers"] else None,
        "on_prefilled": prefilled,
        "on_prefill_progress": progress,
        "on_decode_token": telemetry.token,
    }
    try:
        result, error, presses = _run_generation(
            lambda should_cancel: _backend_generate(
                session.backend, generate_kwargs, should_cancel
            )
        )
    except KeyboardInterrupt:
        # Keep direct callers safe; terminal-driven calls are handled by the
        # worker above and use the staged Ctrl-C policy in main().
        animator.cancel()
        session.reset()
        print(f"\n{C['dim']}generation interrupted; conversation reset{C['0']}\n")
        return 0
    finally:
        if hasattr(session.backend, "_interactive_budgets"):
            session.backend._interactive_budgets = budget_state
        glow.finish()
    interrupted = bool(presses)
    if interrupted:
        animator.cancel()
        if error is not None and not isinstance(error, GenerationCancelled):
            raise error
        text, stats = result if result is not None else ("", telemetry.stats())
    else:
        text, stats = result
    if not interrupted:
        if prefs["stream_answers"]:
            tail = thinking.finish()
            if tail:
                animator.feed(tail, C["gray"] if thinking.inside else "")
            animator.finish(C["gray"] if thinking.inside else "")
        if not prefs["stream_answers"]:
            visible, _ = filter_thinking(
                text, prefs["thinking_enabled"], prefs["show_thinking"]
            )
            if visible:
                animator.feed(visible)
                animator.finish()
    elapsed = time.time() - began
    tail_tokens = getattr(stats, "tail_tokens", 0)
    tail_seconds = getattr(stats, "tail_seconds", 0.0)
    tail_text = (
        f" | tail {tail_tokens} @ {tail_tokens / tail_seconds:.1f} t/s"
        if tail_tokens and tail_seconds else ""
    )
    print(f"\n\n{C['dim']}{stats.prompt_tokens} new tok @ "
          f"{stats.prompt_rate:.1f} t/s | gen {stats.tokens} @ {stats.rate:.1f} t/s"
          f"{tail_text} | "
          f"ctx {len(session.backend.tape)} | {elapsed:.1f}s | "
          f"finish {getattr(stats, 'finish', 'unknown')}{C['0']}\n")
    if getattr(stats, "finish", None) == "length":
        print("Generation limit reached. Use /config tokens to inspect the answer limit.\n")
    return presses


def run_benchmark(session, prompt: str, ready_seconds: float) -> dict:
    # Benchmarks compare token IDs across arms to prove a change left the
    # trajectory alone. Sampling makes that impossible, so this path is greedy
    # whatever the preferences say.
    backend = session.backend
    if hasattr(backend, "sampling"):
        from veystrix.sampling import Sampling

        backend.sampling = Sampling.greedy_settings()
    open_or_continue(session, prompt)
    prefs = session.preferences
    # A JSON benchmark is a short validation run. Treat its answer limit as
    # the total decode ceiling, so a saved reasoning budget cannot expand a
    # `--max-tokens 32` check into thousands of tokens. Interactive turns use
    # generation_limit() and keep their separate answer and thinking budgets.
    limit = preferences.answer_limit(
        prefs, preferences.DEFAULT_PLAIN_ANSWER_TOKENS
    )
    # Physical reads make a rate comparable. Decode rate on this machine spans
    # about 21 percent with page-cache state, so a rate without its read
    # volume cannot be compared against another rate.
    try:
        from models.flashnext.diskio import disk_bytes_read
    except Exception:
        disk_bytes_read = None
    read_before = [-1]

    def mark_decode_start() -> None:
        read_before[0] = disk_bytes_read() if disk_bytes_read else -1

    began = time.time()
    try:
        generate_parameters = inspect.signature(backend.generate).parameters
        supports_prefilled = (
            "on_prefilled" in generate_parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in generate_parameters.values()
            )
        )
    except (TypeError, ValueError):
        supports_prefilled = True
    if supports_prefilled:
        text, stats = session.backend.generate(
            max_tokens=limit, on_prefilled=mark_decode_start
        )
    else:
        # Lightweight benchmark adapters may expose the old two-argument
        # method. They have no prefill callback, so physical telemetry is
        # unavailable instead of mixing prompt and decode reads.
        text, stats = session.backend.generate(max_tokens=limit)
    turn_seconds = time.time() - began
    read_after = disk_bytes_read() if disk_bytes_read else -1
    produced = session.backend.tape[-stats.tokens:] if stats.tokens else []
    token_bytes = b"".join(
        int(value).to_bytes(4, "little", signed=False) for value in produced
    )
    tail_seconds = getattr(stats, "tail_seconds", 0.0)
    result = {
        "profile": getattr(session.backend, "routing_profile", session.profile),
        "model": prefs["model"],
        "ready_seconds": ready_seconds,
        "prompt_tokens": stats.prompt_tokens,
        "prompt_tps": stats.prompt_rate,
        "generated_tokens": stats.tokens,
        "decode_tps": stats.rate,
        "tail_tokens": getattr(stats, "tail_tokens", 0),
        "tail_tps": (
            getattr(stats, "tail_tokens", 0) / tail_seconds if tail_seconds else 0.0
        ),
        "pinned_bytes": getattr(stats, "pinned_bytes", 0),
        "pinned_signature": getattr(stats, "pinned_signature", ""),
        "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "token_ids": produced,
        "output_text": text,
        "turn_seconds": turn_seconds,
        "complete_tps": stats.tokens / turn_seconds if turn_seconds else 0.0,
        "physical_bytes": (
            read_after - read_before[0]
            if read_before[0] >= 0 and read_after >= read_before[0] else -1
        ),
        "mb_per_token": (
            (read_after - read_before[0]) / 1e6 / stats.tokens
            if read_before[0] >= 0
            and read_after >= read_before[0]
            and stats.tokens else -1.0
        ),
    }
    try:
        import mlx.core as mx

        result.update({
            "mlx_active_gb": mx.get_active_memory() / 1e9,
            "mlx_cache_gb": mx.get_cache_memory() / 1e9,
            "mlx_peak_gb": mx.get_peak_memory() / 1e9,
        })
    except Exception:
        pass
    return result


def run_product_benchmark(session, prompt: str, window: int = 32,
                          label: str = "chat.sh", tokens: int = 256,
                          sampling_mode: str = "greedy") -> None:
    """Stream long-generation product metrics as one JSON object per window.

    This is a separate product probe. The established ``--benchmark-json``
    path stays unchanged and still reports one 32-token result.
    """
    if window <= 0 or tokens <= 0 or tokens % window:
        raise ValueError("benchmark tokens must be a positive multiple of its window")
    backend = session.backend
    if sampling_mode == "greedy" and hasattr(backend, "sampling"):
        from veystrix.sampling import Sampling

        backend.sampling = Sampling.greedy_settings()
    open_or_continue(session, prompt)
    try:
        from models.flashnext.diskio import disk_bytes_read
    except Exception:
        disk_bytes_read = None
    try:
        import mlx.core as mx
    except Exception:
        mx = None

    before = [-1]
    window_started = [0.0]
    window_index = [0]
    thinking_tokens = [0]
    answer_tokens = [0]
    phase = ["thinking" if getattr(backend, "thinking_enabled", False) else "answer"]
    produced = []
    physical_trace = [None]

    def slab_counts() -> tuple[int, int]:
        hits = misses = 0
        for layer in getattr(getattr(backend, "language", None), "layers", ()):
            switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
            if switch is not None:
                hits += int(getattr(switch, "hits", 0))
                misses += int(getattr(switch, "misses", 0))
        return hits, misses

    previous_hits = [0]
    previous_misses = [0]

    def on_prefilled() -> None:
        before[0] = disk_bytes_read() if disk_bytes_read else -1
        window_started[0] = time.perf_counter()
        previous_hits[0], previous_misses[0] = slab_counts()
        if os.environ.get("FLASHNEXT_PHYSICAL_MISS_TRACE") == "1":
            from models.flashnext.physical_miss import start_trace

            physical_trace[0] = start_trace(f"product-long:{label}")
            if label.startswith("core48-calibration"):
                physical_trace[0].profile["calibration"] = {
                    "canonical_core_slots": 48,
                    "equal_residency": True,
                }

    def emit_window(count: int) -> None:
        now = time.perf_counter()
        after = disk_bytes_read() if disk_bytes_read else -1
        physical = ((after - before[0]) / 1e6 / count
                    if before[0] >= 0 and after >= before[0] else -1.0)
        active = (mx.get_active_memory() / 1e6
                  if mx is not None else -1.0)
        hits, misses = slab_counts()
        window_hits = max(0, hits - previous_hits[0])
        window_misses = max(0, misses - previous_misses[0])
        hit_pct = (window_hits * 100.0 / (window_hits + window_misses)
                   if window_hits + window_misses else 0.0)
        record = {
            "type": "window",
            "path": label,
            "window": window_index[0] + 1,
            "tokens": count,
            "generated_tokens": len(produced),
            "generation_tps": count / (now - window_started[0])
            if now > window_started[0] else 0.0,
            "physical_mb_token": physical,
            "slab_hit_pct": hit_pct,
            "slab_hits": window_hits,
            "slab_misses": window_misses,
            "active_mb": active,
            "context_tokens": len(backend.tape),
            "phase": phase[0],
            "thinking_tokens": thinking_tokens[0],
            "answer_tokens": answer_tokens[0],
        }
        print(json.dumps(record), flush=True)
        before[0] = after
        window_started[0] = now
        previous_hits[0] = hits
        previous_misses[0] = misses
        window_index[0] += 1

    def on_token(value: int, piece: str) -> None:
        produced.append(value)
        if phase[0] == "thinking":
            thinking_tokens[0] += 1
        else:
            answer_tokens[0] += 1
        if "</think>" in piece:
            phase[0] = "answer"
        if len(produced) % window == 0:
            emit_window(window)

    try:
        _text, stats = backend.generate(
            max_tokens=tokens,
            on_prefilled=on_prefilled,
            on_decode_token=on_token,
        )
    finally:
        if physical_trace[0] is not None:
            from models.flashnext.physical_miss import stop_trace

            physical_trace[0].profile["tokens"] = len(produced)
            stop_trace(os.environ.get(
                "FLASHNEXT_PHYSICAL_MISS_PROFILE",
                "~/.cache/flashnext/physical-misses.json",
            ))
    remainder = len(produced) % window
    if remainder:
        emit_window(remainder)
    digest = hashlib.sha256(
        b"".join(int(value).to_bytes(4, "little", signed=False) for value in produced)
    ).hexdigest()
    print(json.dumps({
        "type": "complete", "path": label, "generated_tokens": len(produced),
        "digest": digest, "finish": stats.finish,
        "context_tokens": len(backend.tape),
        "thinking_tokens": thinking_tokens[0],
        "answer_tokens": answer_tokens[0],
        "sampling": sampling_mode,
        "effort": session.preferences.get("effort", ""),
    }), flush=True)


def token_stats_text(stats_items, context: int, elapsed: float) -> str:
    """Combine every model segment from one agent request."""
    prompt_tokens = sum(getattr(item, "prompt_tokens", 0) for item in stats_items)
    output_tokens = sum(getattr(item, "tokens", 0) for item in stats_items)
    prefill_seconds = sum(
        getattr(item, "prefill_seconds", 0.0) for item in stats_items
    )
    decode_seconds = sum(getattr(item, "seconds", 0.0) for item in stats_items)
    tail_tokens = sum(getattr(item, "tail_tokens", 0) for item in stats_items)
    tail_seconds = sum(getattr(item, "tail_seconds", 0.0) for item in stats_items)
    prompt_rate = prompt_tokens / prefill_seconds if prefill_seconds else 0.0
    output_rate = output_tokens / decode_seconds if decode_seconds else 0.0
    tail_text = (
        f" | tail {tail_tokens:,} @ {tail_tokens / tail_seconds:.1f} tok/s"
        if tail_tokens and tail_seconds else ""
    )
    return (
        f"{prompt_tokens:,} new tok @ {prompt_rate:.1f} tok/s | "
        f"gen {output_tokens:,} @ {output_rate:.1f} tok/s"
        f"{tail_text} | "
        f"ctx {context:,} | {elapsed:.1f}s"
    )


def run_turn_agent(session, prompt: str) -> int:
    open_or_continue(session, prompt)
    prefs = session.preferences
    answer_budget, requested_think, think_budget, limit = _effective_turn_budgets(
        session.backend, prefs, "agent"
    )

    def out(text=""):
        sys.stdout.write(str(text))
        sys.stdout.flush()

    thinking = [ThinkingStreamFilter(
        prefs["thinking_enabled"], prefs["show_thinking"]
    )]
    protocol = [ToolCallStreamFilter()]
    animator = [AsyncWordAnimator(enabled=prefs["animate"])]
    ui = AgentUI()
    turn_stats = []
    telemetry = _TurnTelemetry(len(session.backend.pending))

    def feed_model_text(piece=""):
        visible = protocol[0].feed(thinking[0].feed(str(piece)))
        if visible:
            if animator[0] is None:
                animator[0] = AsyncWordAnimator(enabled=prefs["animate"])
            animator[0].feed(
                visible, C["gray"] if thinking[0].inside else ""
            )
        started, name = protocol[0].take_events()
        if started or name:
            if animator[0] is not None:
                animator[0].finish(C["gray"] if thinking[0].inside else "")
                animator[0] = None
            ui.tool_pending(name)

    def model_out(piece=""):
        feed_model_text(piece)

    def model_done():
        tail = thinking[0].finish()
        if tail:
            feed_model_text(tail)
        visible = protocol[0].finish()
        if visible:
            if animator[0] is None:
                animator[0] = AsyncWordAnimator(enabled=prefs["animate"])
            animator[0].feed(
                visible, C["gray"] if thinking[0].inside else ""
            )
        if animator[0] is not None:
            animator[0].finish(C["gray"] if thinking[0].inside else "")
        thinking[0] = ThinkingStreamFilter(
            prefs["thinking_enabled"], prefs["show_thinking"]
        )
        protocol[0] = ToolCallStreamFilter()
        animator[0] = None

    began = time.time()
    budget_state = getattr(session.backend, "_interactive_budgets", None)
    if hasattr(session.backend, "_interactive_budgets"):
        session.backend._interactive_budgets = (
            answer_budget,
            think_budget,
        )
    try:
        result, error, presses = _run_generation(
            lambda should_cancel: run_agent(
                session.backend, session.tools, out,
                Limits(
                    max_tokens=limit if requested_think is None else answer_budget,
                    think_tokens=0 if requested_think is None else think_budget,
                ),
                approve=ask_approval if prefs["approval"] == "ask" else None,
                model_out=model_out,
                model_done=model_done,
                ui=ui,
                on_stats=turn_stats.append,
                on_prefill_progress=telemetry.progress,
                on_decode_token=telemetry.token,
                should_cancel=should_cancel,
            )
        )
    except KeyboardInterrupt:
        ui.finish()
        if animator[0] is not None:
            animator[0].cancel()
        raise
    finally:
        if hasattr(session.backend, "_interactive_budgets"):
            session.backend._interactive_budgets = budget_state
    if presses:
        ui.finish()
        if animator[0] is not None:
            animator[0].cancel()
        if error is not None and not isinstance(error, GenerationCancelled):
            raise error
        reason = "interrupted"
        turn_stats.append(telemetry.stats())
    else:
        reason = result
    elapsed = time.time() - began
    print(f"\n{C['dim']}{token_stats_text(turn_stats, len(session.backend.tape), elapsed)}"
          f"\nstopped: {reason}{C['0']}\n")
    return presses


if __name__ == "__main__":
    raise SystemExit(main())
