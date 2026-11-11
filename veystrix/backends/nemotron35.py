"""Synthetic-only Nemotron adapter; production model loading is not implemented.

This deliberately does not register a real model in the CLI or API. It exercises
real MLX hybrid state and SSD expert execution without pretending that the toy
fixture has the model's tokenizer, chat template, tool format, or weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from veystrix.backends.base import GenerationCancelled
from models.nemotron35.fixture import load_fixture
from models.nemotron35.runtime import SyntheticRuntime, Cancelled

@dataclass
class SyntheticStats:
    prompt_tokens: int
    tokens: int
    prefill_seconds: float
    seconds: float
    finish: str = 'length'
    @property
    def rate(self):
        return self.tokens / self.seconds if self.seconds else 0.0
    @property
    def prompt_rate(self):
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

class SyntheticBackend:
    """Token-level test adapter. No real checkpoint, model name, or chat API."""
    def __init__(self, fixture_dir):
        self.runtime = SyntheticRuntime(load_fixture(fixture_dir))
        self.pending = []
        self.tape = self.runtime.tokens
        self.thinking_enabled = False

    def append_tokens(self, tokens):
        tokens = list(tokens)
        if any(not isinstance(t, int) or not 0 <= t < 32 for t in tokens):
            raise ValueError('synthetic fixture token outside vocabulary')
        self.pending.extend(tokens)

    def common_prefix(self, tokens):
        return next((i for i, pair in enumerate(zip(self.tape, tokens)) if pair[0] != pair[1]),
                    min(len(self.tape), len(tokens)))

    def reset(self):
        self.runtime.reset()
        self.tape = self.runtime.tokens
        self.pending = []

    def generate(self, max_tokens, out=None, on_prefilled=None, on_prefill_progress=None,
                 on_decode_token=None, should_cancel=None):
        if not self.pending or max_tokens < 0:
            raise ValueError('synthetic generation requires pending input and a finite output limit')
        pending = list(self.pending)
        try:
            started = time.monotonic()
            if on_prefill_progress:
                on_prefill_progress(0, len(pending))
            logits = self.runtime.prefill(pending, cancelled=should_cancel or (lambda: False))
            self.pending.clear()
            prefill = time.monotonic() - started
            if on_prefill_progress:
                on_prefill_progress(len(pending), len(pending))
            if on_prefilled:
                on_prefilled()
            started = time.monotonic()
            ids, pieces = [], []
            for token in self.runtime.iter_greedy(logits, max_tokens, cancelled=should_cancel or (lambda: False)):
                piece = (" " if ids else "") + str(token)
                ids.append(token)
                pieces.append(piece)
                if on_decode_token:
                    on_decode_token(token, piece)
                if out:
                    out(piece)
            return ''.join(pieces), SyntheticStats(len(pending), len(ids), prefill, time.monotonic() - started)
        except Cancelled as exc:
            self.reset()
            raise GenerationCancelled() from exc
        except BaseException:
            self.reset()
            raise


# ---------------------------------------------------------------------------------------------------------------
# Real streamed backend. Real weights are only loaded when explicitly authorized; see NemotronBackend.
# ---------------------------------------------------------------------------------------------------------------
import hashlib
import json
import os
import re
from pathlib import Path

from veystrix.backends.base import DecodeTimer
from veystrix.conversation import Conversation, IM_END, IM_START
from veystrix.sampling import Sampler, Sampling
from veystrix.text import stream_decode

AUTHORIZE_ENV = "VEYSTRIX_NEMOTRON_AUTHORIZED"
RUNTIME_VERSION = "veystrix-nemotron35-stream-1"
_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
UNLIMITED = (1 << 31) - 1
LAYER_MAJOR_DEFAULT = 8192   # layer-major prefill segment in tokens (measured: 17% faster, 39% less SSD read, 40% lower peak than chunk-major); 0 = chunk-major


@dataclass
class LastPrefix:
    """What the last prefill reused and computed, in the shape the shared server and the warm-up client read."""
    prompt_tokens: int = 0
    hit_tokens: int = 0
    prefilled_tokens: int = 0
    prefill_seconds: float = 0.0
    snapshot_seconds: float = 0.0
    outcome: str = "cold"
    layer_major: bool = False

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "hit_tokens": self.hit_tokens,
                "reused_tokens": self.hit_tokens, "prefilled_tokens": self.prefilled_tokens,
                "prefill_seconds": round(self.prefill_seconds, 3), "snapshot_seconds": round(self.snapshot_seconds, 3),
                "outcome": self.outcome, "layer_major": self.layer_major}


@dataclass
class Stats:
    finish: str = "stop"
    tokens: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    tail_tokens: int = 0
    tail_seconds: float = 0.0
    ui_seconds: float = 0.0
    pinned_bytes: int = 0
    pinned_signature: str = ""
    host_free_gb: float | None = None
    swap_gb: float | None = None
    pool_hits: int = 0
    pool_misses: int = 0
    expert_bytes_read: int = 0
    prefetch_used: int = 0
    prefetch_issued: int = 0
    reused_tokens: int = 0

    @property
    def rate(self) -> float:
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def prompt_rate(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0


class NemotronBackend(Conversation):
    """Shared-protocol backend: SSD-streamed routed experts, resident dense core, Mamba + KV state.

    Allowed by default once the model is selected manually; ``VEYSTRIX_NEMOTRON_AUTHORIZED=0`` disables it.
    Qwen (Flash-Next) remains the default model and nothing routes between models automatically.
    """

    last_prefix = None

    def __init__(self, model_path=None, *, authorized: bool | None = None, pool_bytes: int | None = None,
                 context_tokens: int | None = None, chunk_tokens: int | None = None,
                 session_dir: str = "~/.cache/veystrix/nemotron35/sessions",
                 config_check=None, machine=None, tokenizer=None, verify: bool = True,
                 prefix_dir=None, prefix_cache: bool | None = None, prefix_quota_gb: float | None = None,
                 snapshot_tokens: int | None = None, layer_major_tokens: int | None = None):
        from models.nemotron35 import admission
        from models.nemotron35.backend_core import EngineCore
        from models.nemotron35.config import validate_config
        from models.nemotron35.inventory import estimate_bytes
        from models.nemotron35.streamed_model import load_streamed
        if not model_path:
            raise SystemExit("Nemotron requires --model-path/--checkpoint pointing at a complete local checkpoint")
        root = Path(model_path).expanduser()
        if authorized is None:
            # Owner approved real use on 2026-11-11. Selecting the model is the opt-in; AUTHORIZE_ENV=0 is a kill switch.
            authorized = os.environ.get(AUTHORIZE_ENV) != "0"
        if not authorized:
            raise SystemExit(f"Nemotron is disabled by {AUTHORIZE_ENV}=0; nothing was loaded.")
        check = config_check or validate_config
        config = json.loads((root / "config.json").read_text())
        check(config)
        machine = machine or admission.read_machine()
        chunk = chunk_tokens or int(os.environ.get("FLASHNEXT_PREFILL_CHUNK_TOKENS") or 2048)
        pool = admission.pool_bytes_from_env() if pool_bytes is None else pool_bytes   # None: sized from the plan
        # Plan from the exact dense-core size implied by the verified inventory, BEFORE reading any weight.
        plan = admission.make_plan(config, machine, dense_bytes=estimate_bytes(config)["dense_bytes"],
                                   context_tokens=context_tokens or admission.DEFAULT_CONTEXT,
                                   chunk_tokens=chunk, pool_bytes=pool)
        self.plan = plan
        if tokenizer is None:
            from veystrix.backends.flashnext import _load_transformers_tokenizer
            tokenizer = _load_transformers_tokenizer().from_pretrained(str(root))
        super().__init__(tokenizer)
        self.streamed = load_streamed(root, pool_bytes=plan.pool_bytes, config_check=check, verify=verify)
        if layer_major_tokens is None:
            layer_major_tokens = int(os.environ.get("VEYSTRIX_NEMOTRON_LAYER_MAJOR_TOKENS") or LAYER_MAJOR_DEFAULT)
        self.core = EngineCore(self.streamed, chunk_tokens=plan.chunk_tokens, layer_major_tokens=layer_major_tokens)
        self.model_path = str(root)
        self.config = config
        self.context_limit = plan.context_tokens
        self.session_dir = session_dir
        self.thinking_enabled = False
        self.sampling = Sampling.greedy_settings()
        self.reasoning_effort = "medium"
        self.think_budget = 0
        self.answer_budget = 0
        self._interactive_budgets = None
        self._setting_sources = {}
        self._replay_needed = False
        self.last_decode_rate = None
        eos = config.get("eos_token_id", [])
        eos = [eos] if isinstance(eos, int) else list(eos)
        self.stops = {int(t) for t in eos} | ({int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set())
        self._machine = machine
        self._identity = None
        self.snapshot_tokens = snapshot_tokens or int(os.environ.get("VEYSTRIX_NEMOTRON_SNAPSHOT_TOKENS") or 4096)
        self.prefix_cache = None
        if prefix_cache is None:
            prefix_cache = os.environ.get("VEYSTRIX_NEMOTRON_PREFIX_CACHE", "1") != "0"
        if prefix_cache:
            from models.nemotron35.prefix_cache import PrefixCache
            quota = prefix_quota_gb if prefix_quota_gb is not None else float(os.environ.get("VEYSTRIX_CACHE_GB") or 20)
            directory = prefix_dir or os.environ.get("VEYSTRIX_NEMOTRON_PREFIX_DIR") or "~/.cache/veystrix/nemotron35/prefix"
            self.prefix_cache = PrefixCache(directory, self._session_identity(), quota_bytes=int(quota * 2**30))
            self._prefix_session = lambda: self.prefix_cache      # what the shared server asks for (health, adoption)

    # -- framing: exactly the model's chat template (see chat_template.jinja) ----------------------------------
    @staticmethod
    def _assistant_prefix(enable_thinking: bool = True) -> str:
        return f"{IM_START}assistant\n<think>\n" if enable_thinking else f"{IM_START}assistant\n<think></think>"

    def open_conversation(self, system, user, tools=None, enable_thinking=True, reasoning_effort="medium") -> int:
        if self.tape or self.pending:
            raise RuntimeError("conversation already open")
        # The template has no effort control: effort is not injected into the prompt for this model.
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
        return self.append_text(text)

    def append_tool_results(self, results, enable_thinking: bool = True) -> int:
        head = f"{self._close()}{self._separator()}{IM_START}user\n"
        tail = f"{IM_END}\n{self._assistant_prefix(enable_thinking)}"
        coded = [self._content_ids(r) for r in results]
        if all(c is None for c in coded):
            body = "".join(f"<tool_response>\n{r}\n</tool_response>\n" for r in results)
            return self.append_text(f"{head}{body}{tail}")
        ids = self.encode(head)
        for result, content in zip(results, coded):
            ids.extend(self.encode("<tool_response>\n"))
            ids.extend(content if content is not None else self.encode(result))
            ids.extend(self.encode("\n</tool_response>\n"))
        ids.extend(self.encode(tail))
        self.pending.extend(ids)
        return len(ids)

    # -- state -------------------------------------------------------------------------------------------------
    @property
    def cache(self):
        return self.core.cache

    @property
    def cache_tokens(self) -> int:
        return self.core.cached_tokens

    def reset(self) -> None:
        self.core.reset()          # the expert pool stays warm: it holds weights, not conversation state
        self.tape, self.pending = [], []
        self.turn_closed = True
        self._replay_needed = False

    def reclaim_owned_memory(self, reading=None) -> int:
        """Release the expert pool and allocator cache (called between requests under pressure)."""
        freed = self.streamed.pool.resident_bytes
        self.streamed.pool.clear()
        import mlx.core as mx
        mx.clear_cache()
        return freed

    def _mark_replay_needed(self) -> None:
        self.core.reset()
        self._replay_needed = True

    def diagnostics(self) -> dict:
        return {"model": "nemotron35-lightning", "runtime": RUNTIME_VERSION, "plan": self.plan.summary(),
                "pool": self.streamed.pool.stats(), "cache_tokens": self.cache_tokens,
                "context_limit": self.context_limit, "expert_bytes_read": self.streamed.reader.bytes_read}

    # -- generation --------------------------------------------------------------------------------------------
    last_readiness = None
    _recovery = None

    def _purge(self, required: int):
        """Conditional file-cache purge (FLASHNEXT_AUTO_PURGE=1 and ``sudo -n`` permission); (action, detail)."""
        from models.flashnext import purge_recovery as P
        if not P.enabled():
            return "skipped", "auto purge is off"
        if self._recovery is None:
            self._recovery = P.PurgeRecovery(P.Config.from_environment())
        outcome = self._recovery.attempt(required, "request")
        return outcome.action, outcome.reason

    def generate(self, max_tokens: int, out=None, on_prefilled=None, on_prefill_progress=None,
                 on_decode_token=None, should_cancel=None):
        from models.flashnext import purge_recovery as P
        from models.nemotron35 import readiness
        if self.pending:
            self.last_readiness = readiness.ensure_ready(
                self._current_machine, plan=self.plan, resident_pool_bytes=self.streamed.pool.resident_bytes,
                reclaim=self.reclaim_owned_memory, purge=self._purge)
        with P.ACTIVITY.request():         # a purge never overlaps a request
            return self._generate(max_tokens, out, on_prefilled, on_prefill_progress, on_decode_token, should_cancel)

    def health_snapshot(self) -> dict:
        """What /health reports about this engine: pool, prefetch, prefix cache, last prefill, memory readiness."""
        pool = self.streamed.pool.stats()
        asked = pool["hits"] + pool["misses"]
        used, issued = pool["prefetch_used"], pool["prefetch_issued"]
        snapshot = {
            "model": "nemotron35-lightning", "runtime": RUNTIME_VERSION,
            "pool": {"budget_gib": round(pool["budget_bytes"] / 2**30, 2), "resident_gib": round(pool["resident_bytes"] / 2**30, 2),
                     "hit_rate": round(pool["hits"] / asked, 3) if asked else None,
                     "staged_mib": round(pool["staged_bytes"] / 2**20, 1)},
            "prefetch": {"issued": issued, "used": used, "wasted": pool["prefetch_wasted"],
                         "precision": round(used / issued, 3) if issued else None},
            "expert_gb_read": round(pool["bytes_read"] / 1e9, 2),
            "prefill": {"layer_major_tokens": self.core.layer_major_tokens, "chunk_tokens": self.core.chunk_tokens},
            "last_prefix": self.last_prefix.as_dict() if self.last_prefix is not None else None,
            "readiness": self.last_readiness.as_dict() if self.last_readiness is not None else None,
        }
        if self.prefix_cache is not None:
            st = self.prefix_cache.stats
            snapshot["prefix_cache"] = {"snapshots": len(self.prefix_cache._disk),
                                        "gb": round(sum(m["bytes"] for m in list(self.prefix_cache._disk.values())) / 1e9, 2),
                                        "ssd_hits": st.ssd_hits, "cold_lookups": st.cold_lookups,
                                        "reused_tokens": st.reused_tokens, "restore_failures": st.restore_failures,
                                        "evicted": st.evicted_ssd}
        return snapshot

    def _generate(self, max_tokens: int, out=None, on_prefilled=None, on_prefill_progress=None,
                  on_decode_token=None, should_cancel=None):
        import mlx.core as mx
        from models.nemotron35 import admission
        from models.nemotron35.runtime import Cancelled
        if not self.pending:
            return "", Stats(finish="stop")
        horizon = UNLIMITED if int(max_tokens) < 0 else int(max_tokens)
        replay = self._replay_needed
        fresh = replay or not self.tape
        pending = list(self.pending)
        feed = (list(self.tape) + pending) if replay else pending
        cached = 0 if fresh else self.cache_tokens
        admission.check_request(self.plan, self._current_machine(), new_tokens=len(feed), cached_tokens=cached)
        restored = 0
        if fresh:
            self.core.reset()
            if self.prefix_cache is not None and len(feed) > self.prefix_cache.min_tokens:
                cache, restored = self.prefix_cache.restore(feed)
                if restored:
                    self.core.cache = cache
        self.tape = self.tape + pending
        self.pending, self._replay_needed = [], False
        context_limit = getattr(self, "context_limit", None)
        if isinstance(context_limit, int) and not isinstance(context_limit, bool) and context_limit > 0:
            horizon = min(horizon, max(0, context_limit - len(self.tape)))
        total_prompt = len(feed)
        feed = feed[restored:]
        snapshots = []
        start = restored if fresh else cached          # tokens the cache already holds before this prefill
        mark = [start]

        def checkpoint(done):
            # A cancelled or restarted long prompt resumes from here instead of from zero.
            at = start + done
            if self.prefix_cache is not None and at < len(self.tape) and at - mark[0] >= self.snapshot_tokens:
                if self.prefix_cache.save(self.core.cache, self.tape[:at]):
                    snapshots.append(at)
                mark[0] = at
        cancel = should_cancel or (lambda: False)
        pool0 = self.streamed.pool.stats()
        began = DecodeTimer()
        prefill_began = began.mark()

        def interrupted():
            self._mark_replay_needed()

        def progress(done, total):
            self._tick()
            if on_prefill_progress:
                on_prefill_progress(restored + done, total_prompt)

        if on_prefill_progress:
            on_prefill_progress(restored, total_prompt)
        try:
            logits = self.core.prefill(feed, cancelled=cancel, progress=progress, boundary=checkpoint)
        except Cancelled as exc:
            interrupted()
            raise GenerationCancelled() from exc
        except BaseException:
            interrupted()
            raise
        prefill_seconds = began.since(prefill_began)
        outcome = ("extended" if not fresh else "ssd_hit" if restored else "cold")
        snapshot_began = time.perf_counter()
        if self.prefix_cache is not None and len(feed) >= self.prefix_cache.min_tokens:
            try:   # the prompt boundary: the next request usually shares this whole prefix
                if self.prefix_cache.save(self.core.cache, self.tape):
                    snapshots.append(len(self.tape))
                self.prefix_cache.prune(self.tape, snapshots)
            except Exception:
                pass   # a failed snapshot never fails the request
        self.last_prefix = LastPrefix(
            prompt_tokens=(restored if fresh else cached) + len(feed), hit_tokens=(restored if fresh else cached),
            prefilled_tokens=len(feed),
            prefill_seconds=prefill_seconds, snapshot_seconds=time.perf_counter() - snapshot_began, outcome=outcome,
            layer_major=bool(self.core.layer_major_tokens and len(feed) > self.core.chunk_tokens))
        if on_prefilled:
            on_prefilled()
        sampler = Sampler(self.sampling)
        budgets = self._interactive_budgets
        think_budget = answer_budget = None
        close_token = None
        phase = "thinking" if self.thinking_enabled else "answer"
        if budgets is not None:
            raw_answer, raw_think = budgets
            answer_budget = None if raw_answer is None or int(raw_answer) < 0 else int(raw_answer)
            think_budget = None if raw_think is None or int(raw_think) < 0 else int(raw_think)
            ids = self.encode("</think>")
            close_token = int(ids[0]) if len(ids) == 1 else None
        counts = {"thinking": 0, "answer": 0}
        produced, partial, pieces = [], [], []
        finish = "length"
        timer = DecodeTimer()
        try:
            while len(produced) < horizon:
                token = sampler(logits)
                mx.eval(token)
                value = int(token.item())
                forced = (budgets is not None and phase == "thinking" and think_budget and close_token is not None
                          and counts["thinking"] == think_budget - 1)
                if value in self.stops:
                    finish = "stop"
                    break
                if forced:
                    value = close_token
                if budgets is not None and phase == "answer" and answer_budget is not None \
                        and counts["answer"] >= answer_budget:
                    break
                if cancel():
                    raise GenerationCancelled()
                produced.append(value)
                self.tape.append(value)
                sampler.observe(value)
                if forced:
                    piece = (self.tokenizer.decode(partial) if partial else "") + "</think>"
                    partial.clear()
                else:
                    piece = stream_decode(self.tokenizer, partial, value)
                if budgets is not None:
                    counts[phase] += 1
                    if phase == "thinking" and (forced or "</think>" in piece):
                        phase = "answer"
                elif phase == "thinking" and "</think>" in piece:
                    phase = "answer"
                self._tick()
                if on_decode_token:
                    on_decode_token(value, piece)
                if piece:
                    pieces.append(piece)
                    if out is not None:
                        with timer.emitting():
                            out(piece)
                try:
                    logits = self.core.decode(value)   # one step is ~0.1 s: cancel is checked between tokens so state stays valid
                except Cancelled as exc:
                    raise GenerationCancelled() from exc
        except GenerationCancelled:
            # Between tokens the cache still equals the tape; mid-forward cancellation already reset the cache.
            if self.cache_tokens != len(self.tape):
                self._mark_replay_needed()
            self.turn_closed = False
            raise
        except BaseException:
            self._mark_replay_needed()
            raise
        finally:
            import mlx.core as mx2
            mx2.clear_cache()
        self.turn_closed = False
        seconds = timer.elapsed()
        pool1 = self.streamed.pool.stats()
        self.last_decode_rate = len(produced) / seconds if seconds and produced else None
        stats = Stats(finish=finish, tokens=len(produced), seconds=seconds, prompt_tokens=total_prompt,
                      prefill_seconds=prefill_seconds, ui_seconds=timer.emitted,
                      pool_hits=pool1["hits"] - pool0["hits"], pool_misses=pool1["misses"] - pool0["misses"],
                      expert_bytes_read=pool1["bytes_read"] - pool0["bytes_read"],
                      prefetch_used=pool1["prefetch_used"] - pool0["prefetch_used"],
                      prefetch_issued=pool1["prefetch_issued"] - pool0["prefetch_issued"], reused_tokens=restored)
        return "".join(pieces), stats

    def _tick(self) -> None:
        try:  # keeps the out-of-process watchdog informed when it is running
            from models.flashnext import progress
            progress.PROGRESS.tick()
        except Exception:
            pass

    def _current_machine(self):
        from models.nemotron35 import admission
        try:
            return admission.read_machine()
        except admission.AdmissionRefused:
            raise
        except Exception as exc:  # fail closed: no reading, no work
            raise admission.AdmissionRefused(f"memory probes failed: {exc}") from exc

    def check_invariant(self) -> bool:
        return self.cache_tokens == len(self.tape) if self.tape and not self._replay_needed else True

    # -- named sessions (fingerprinted, namespaced by model identity) ------------------------------------------
    def _session_identity(self) -> dict:
        if self._identity is None:
            root = Path(self.model_path)
            def digest(name):
                return hashlib.sha256((root / name).read_bytes()).hexdigest()
            index = json.loads((root / "model.safetensors.index.json").read_text())
            self._identity = {
                "runtime": RUNTIME_VERSION, "model": "nemotron35-lightning",
                "config": digest("config.json"), "tokenizer": digest("tokenizer.json"),
                "template": digest("chat_template.jinja"),
                "total_size": (index.get("metadata") or {}).get("total_size"),
                "quantization": self.config.get("quantization"), "state_dtypes": "bf16-kv/fp32-ssm"}
        return self._identity

    def _session_paths(self, name: str):
        if not isinstance(name, str) or not _SESSION_NAME.match(name):
            raise ValueError("session names use letters, digits, '.', '_' and '-' (max 64)")
        identity = hashlib.sha256(json.dumps(self._session_identity(), sort_keys=True).encode()).hexdigest()[:16]
        base = Path(self.session_dir).expanduser() / identity
        return base / f"{name}.safetensors", base / f"{name}.tokens.json"

    def save_session(self, name: str) -> str:
        from models.nemotron35 import state
        state_path, tokens_path = self._session_paths(name)
        if self.pending or self._replay_needed or not self.tape or self.cache_tokens != len(self.tape):
            return "nothing to save: no fully processed conversation"
        state.save(state_path, self.core.cache, self._session_identity(), list(self.tape))
        tokens_path.write_text(json.dumps({"tokens": list(self.tape), "turn_closed": self.turn_closed}))
        os.chmod(tokens_path, 0o600)
        return f"saved {name} ({len(self.tape)} tokens)"

    def load_session(self, name: str) -> str:
        from models.nemotron35 import state
        state_path, tokens_path = self._session_paths(name)
        try:
            meta = json.loads(tokens_path.read_text())
            tokens = meta["tokens"]
            if not isinstance(tokens, list) or len(tokens) > self.context_limit:
                raise state.SnapshotError("invalid token record")
            cache = state.restore(state_path, self._session_identity(), tokens)
        except (OSError, KeyError, ValueError) as exc:  # SnapshotError is a ValueError
            return f"cannot load {name}: {exc}"
        self.core.cache = cache
        self.tape, self.pending = list(tokens), []
        self.turn_closed = bool(meta.get("turn_closed", True))
        self._replay_needed = False
        return f"loaded {name} ({len(tokens)} tokens)"

    def list_sessions(self) -> str:
        state_path, _ = self._session_paths("x")
        names = sorted(p.stem for p in state_path.parent.glob("*.safetensors")) if state_path.parent.is_dir() else []
        return "\n".join(names) if names else "no saved sessions"

    def delete_session(self, name: str) -> str:
        state_path, tokens_path = self._session_paths(name)
        removed = False
        for path in (state_path, tokens_path):
            if path.exists():
                path.unlink()
                removed = True
        return f"deleted {name}" if removed else f"no session named {name}"

    def configure(self, argument: str) -> str:
        return "Nemotron has no runtime-tunable engine settings beyond the pool size set at launch."

    def settings_registry(self):
        return None

    def settings_state(self, include_research: bool = False) -> str:
        p = self.plan.summary()
        return (f"nemotron35-lightning: context {p['context_tokens']}, chunk {p['chunk_tokens']}, "
                f"pool {p['pool_bytes'] / 2**30:.2f} GiB (estimated budget {p['total_bytes'] / 2**30:.2f} of "
                f"{p['budget_bytes'] / 2**30:.2f} GiB)")

    def close(self) -> None:
        self.streamed.close()
