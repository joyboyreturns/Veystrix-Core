"""Qwen3.8-Flash-Next as an agent backend.

Supplies generation and cache accounting; the transcript comes from
Conversation and the loop from macqwen.agent. Streams experts from the
checkpoint, so a token costs an SSD read rather than resident memory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import time

import mlx.core as mx

from macqwen.backends.base import DecodeTimer, GenerationCancelled
from macqwen.checkpoints import resolve_flashnext
from macqwen.conversation import Conversation
from macqwen.model_settings import FLASHNEXT_DEFAULTS
from macqwen.sampling import Sampler, Sampling
from macqwen.text import stream_decode
from models.flashnext import gpu_sync, progress
from models.flashnext.settings import get_registry

DEFAULT_FUSION_MODEL = FLASHNEXT_DEFAULTS["fusion_model"]
_TRANSFORMERS_ADVISORY_ENV = "TRANSFORMERS_NO_ADVISORY_WARNINGS"

# Public convention is -1 = unlimited. Internal machinery needs a
# positive integer (range(), decoder loops, room math), so normalize
# to this effectively-unbounded horizon at the generate() boundary.
# Nothing is allocated proportional to it; range() is lazy and
# stop tokens/cancellation still terminate generation.
UNLIMITED_GENERATION_HORIZON = (1 << 31) - 1


@contextmanager
def _transformers_import_environment():
    """Hide Transformers' irrelevant PyTorch advisory during its import.

    Flash-Next uses MLX and only imports Transformers tokenizer utilities.
    Keep the environment change limited to the import, so later warnings and
    all real errors still reach the user.
    """
    previous = os.environ.get(_TRANSFORMERS_ADVISORY_ENV)
    os.environ[_TRANSFORMERS_ADVISORY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_TRANSFORMERS_ADVISORY_ENV, None)
        else:
            os.environ[_TRANSFORMERS_ADVISORY_ENV] = previous


def _load_transformers_tokenizer():
    with _transformers_import_environment():
        from transformers import AutoTokenizer
    return AutoTokenizer


def _resolve_startup_mtp(native_mtp=None, mtp_depth=None):
    """Resolve native MTP startup state with explicit > env > default.

    `build_backend()` passes no explicit value (no CLI flag), so the
    environment controls normal startup.
    """
    if native_mtp is None or str(native_mtp).lower() == "auto":
        native_mtp = os.environ.get("FLASHNEXT_NATIVE_MTP", "off")
    native_mtp = str(native_mtp).lower()
    if native_mtp not in ("off", "on"):
        raise ValueError("native-mtp must be off or on")
    if mtp_depth is None:
        mtp_depth = os.environ.get("FLASHNEXT_MTP_DEPTH", 3)
    try:
        mtp_depth = int(mtp_depth)
    except (TypeError, ValueError):
        mtp_depth = 3
    return native_mtp, max(1, min(8, int(mtp_depth)))


@dataclass
class Stats:
    """What the agent loop reads back from a turn.

    `host_free_gb` and `swap_gb` stay None: this backend does not measure
    them, and the loop skips those guards rather than treating None as zero.
    """

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

    @property
    def rate(self) -> float:
        """Tokens per second of model time.

        `seconds` excludes the streaming callback. The terminal fade and the
        writes behind it run inside the decode loop, and their cost varies
        per word, so counting them made the reported rate wander for reasons
        that had nothing to do with the model.
        """
        return self.tokens / self.seconds if self.seconds else 0.0

    @property
    def prompt_rate(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds else 0.0


RELEASE_CACHE_BYTES = 64 * 2 ** 20      # trim the MLX allocator cache between requests when it holds more than this


class FlashNextBackend(Conversation):
    # What the last prefill reused from the prefix cache (None when it was off); see prefix_session.Report.
    last_prefix = None
    # What the opt-in decode-pool release did before the last prefill (None when the flag is off or no fresh prefill
    # ran); see models/flashnext/pool_lifecycle.make_room.
    last_pool_release = None
    # What the opt-in memory manager planned and did before the last prefill, and its cleanup after the request.
    last_memory_report = None
    last_memory_after = None

    def settings_registry(self):
        return get_registry()

    def settings_state(self, include_research: bool = False) -> str:
        return self.settings_registry().render(self, "flashnext", include_research)

    def __init__(self, model_path: str | None = None,
                 threshold: float = FLASHNEXT_DEFAULTS["threshold"],
                 resident_experts: int = FLASHNEXT_DEFAULTS["resident_experts"],
                 pin_budget_gb: float = FLASHNEXT_DEFAULTS["pin_budget_gb"],
                 routing_profile: str = FLASHNEXT_DEFAULTS["routing"],
                 swap_epsilon: float = FLASHNEXT_DEFAULTS["swap_epsilon"],
                 tail_experts: int = FLASHNEXT_DEFAULTS["tail_experts"],
                 tail_warmup: int = FLASHNEXT_DEFAULTS["tail_warmup"],
                 fusion_block: int = FLASHNEXT_DEFAULTS["fusion_block"],
                 fusion_min_margin: float = FLASHNEXT_DEFAULTS["fusion_min_margin"],
                 fusion_min_block: int = FLASHNEXT_DEFAULTS["fusion_min_block"],
                 fusion_margin_tokens: int = FLASHNEXT_DEFAULTS["fusion_margin_tokens"],
                  fusion_max_prompt: int = FLASHNEXT_DEFAULTS["fusion_max_prompt"],
                  fusion_model: str = DEFAULT_FUSION_MODEL,
                  native_mtp: str | None = None,
                  mtp_depth: int | None = None,
                  session_dir: str = "~/.cache/flashnext/sessions"):
        gpu_sync.install_exit_drain()      # drain before mlx_vlm's GIL-holding stream cleanup at exit
        AutoTokenizer = _load_transformers_tokenizer()
        from models.flashnext.loader import load_streaming

        # Startup precedence: explicit constructor value, then the
        # environment, then off. build_backend() passes no explicit value
        # (no CLI flag), so the environment controls normal startup.
        native_mtp, mtp_depth = _resolve_startup_mtp(native_mtp, mtp_depth)
        if routing_profile == "fused-quality" and native_mtp == "on":
            raise ValueError("native MTP and fused-quality are mutually exclusive")
        from models.flashnext.qsa_chunk import apply as apply_qsa

        os.environ["FLASHNEXT_TOPK_THRESHOLD"] = str(threshold)
        try:
            path = str(resolve_flashnext(model_path))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        model, _, self.store = load_streaming(
            path, verbose=False, keep_vision=False,
            use_mtp=(native_mtp == "on"))
        apply_qsa()
        super().__init__(AutoTokenizer.from_pretrained(path))
        self.language = model.language_model
        self.model_path = path
        self.threshold = threshold
        self.resident_experts = resident_experts
        self.pin_budget_gb = pin_budget_gb
        self.tail_experts = tail_experts
        self.tail_warmup = tail_warmup
        self.routing_profile = routing_profile
        self.swap_epsilon = swap_epsilon
        self.fusion_block = fusion_block
        self.fusion_min_margin = fusion_min_margin
        self.fusion_min_block = fusion_min_block
        self.fusion_margin_tokens = fusion_margin_tokens
        self.fusion_max_prompt = fusion_max_prompt
        self.fusion_model = os.path.expanduser(fusion_model)
        self.native_mtp = native_mtp
        self.mtp_depth = mtp_depth
        self._mtp_active = bool(
            native_mtp == "on" and hasattr(model.language_model, "mtp")
        )
        # A restored target-only session populates target history with no
        # MTP history, and any target-only turn leaves a live MTP decoder
        # stale. Both cases block native MTP until reset starts both
        # histories together.
        self._mtp_blocked = False
        self._setting_sources = {}
        self.session_dir = session_dir
        self.thinking_enabled = False
        # Greedy by default so no benchmark can sample by accident. Every
        # comparison here proves a change left the trajectory alone by
        # matching token IDs across arms. The chat sets this from
        # preferences to Qwen's recommended thinking-mode sampler.
        self.sampling = Sampling.greedy_settings()
        # Mirrored from preferences for display, the way `thinking_enabled`
        # is. `/effort`, `/thinking` and `/think-budget` remain the writers,
        # so there is one owner per value and nothing to drift.
        self.reasoning_effort = "medium"
        self.think_budget = 0
        self.answer_budget = 0
        # Interactive chat enables separate reasoning and answer quotas for
        # this turn. Direct API and benchmark calls leave this unset, so their
        # max_tokens value remains one total generation ceiling.
        self._interactive_budgets = None
        self._store = None
        self._decoder = None
        self._fused_pending = routing_profile == "fused-quality"
        self._replay_needed = False
        from models.flashnext.routing import RoutingProfile, prewarm_enabled

        self.routing = RoutingProfile(
            routing_profile, self.store, self.language,
            threshold=threshold, resident_experts=resident_experts,
            tail_experts=tail_experts, warmup=tail_warmup,
            pin_budget_gb=pin_budget_gb, swap_epsilon_value=swap_epsilon,
        )
        if routing_profile == "cache-aware":
            self.store.set_residency_tracking(True)
        if prewarm_enabled():
            self.prewarmed_rows = self.routing.prewarm()
        self.cache = self.language.make_cache()
        self.stops = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        } - {None}

    def _settings_text(self) -> str:
        return get_registry().render(self, "flashnext")

    def _rebuild_routing(self) -> None:
        from models.flashnext.routing import RoutingProfile, prewarm_enabled

        decoder = self._decoder
        if decoder is not None and self._is_native_mtp_decoder(decoder):
            if self._turn_allows_mtp():
                # Changing draft depth does not invalidate MTP cache
                # history, so keep the synchronized decoder and sync depth.
                decoder.depth = max(1, min(8, int(self.mtp_depth)))
            else:
                # The new state cannot advance MTP history. Adopt the
                # target cache and block MTP until reset.
                self.cache = decoder.target_cache
                self._decoder = None
                self._mtp_blocked = True
        elif decoder is not None:
            self.cache = decoder.target_cache
            self._decoder = None
        self._fused_pending = self.routing_profile == "fused-quality" and not self.tape
        self.routing = RoutingProfile(
            self.routing_profile,
            self.store,
            self.language,
            threshold=self.threshold,
            resident_experts=self.resident_experts,
            tail_experts=self.tail_experts,
            warmup=self.tail_warmup,
            pin_budget_gb=self.pin_budget_gb,
            swap_epsilon_value=self.swap_epsilon,
        )
        self.store.set_residency_tracking(
            self.routing_profile == "cache-aware"
            or os.environ.get("FLASHNEXT_TRACK_RESIDENT") == "1"
        )
        self._store = None

    def configure(self, argument: str) -> str:
        return get_registry().configure(self, argument, "flashnext")

    # ---- session persistence -------------------------------------------
    # Exact snapshots: every DeltaNet state, QSA cache and indexer array is
    # written, so a restored session continues without replaying the prompt.
    # The 27B backend persists prompt-cache images instead; both satisfy the
    # same four methods, so the chat never learns the difference.

    def _session_profile(self):
        profile = self.routing.session_profile(self.stops)
        if self.routing_profile == "fused-quality":
            profile.update({
                "fused_quality": True,
                "speculative_fast": True,
                "draft_depth": self.fusion_block,
                "draft_model": os.path.basename(self.fusion_model),
                "fusion_block": self.fusion_block,
                "fusion_min_margin": self.fusion_min_margin,
                "fusion_min_block": self.fusion_min_block,
                "fusion_margin_tokens": self.fusion_margin_tokens,
                "fusion_max_prompt": self.fusion_max_prompt,
            })
        return profile

    def _prefix_session(self):
        """The prefix cache for this engine (built on first use; fingerprint = checkpoint, tokenizer,
        engine files and routing profile, the same compatibility record exact sessions use)."""
        from models.flashnext.prefix_session import PrefixSession

        if getattr(self, "_prefix", None) is None:
            self._prefix = PrefixSession(self.encode, lambda: self._sessions()._compatibility())
        return self._prefix

    def _sessions(self):
        from models.flashnext.sessions import SessionStore

        if getattr(self, "_store", None) is None:
            self._store = SessionStore(
                os.path.expanduser(self.session_dir),
                self.model_path,
                self._session_profile(),
                self.language,
            )
        return self._store

    def save_session(self, name: str) -> str:
        from models.flashnext.sessions import SessionError

        try:
            summary = self._sessions().save(
                name, self.cache, self.tape, not self.tape,
                self.thinking_enabled, self.turn_closed)
        except (OSError, SessionError, ValueError) as exc:
            return f"could not save session: {exc}"
        return (f"saved {summary.name}  {summary.cached_tokens} tokens, "
                f"thinking={'on' if summary.thinking else 'off'}")

    def load_session(self, name: str) -> str:
        from models.flashnext.sessions import SessionError, SessionStore

        try:
            sessions = self._sessions()
            saved_profile = sessions.saved_profile(name)
            loader = sessions
            if saved_profile != sessions.profile:
                loader = SessionStore(
                    os.path.expanduser(self.session_dir),
                    self.model_path,
                    saved_profile,
                    self.language,
                )
            loaded = loader.load(name)
        except (OSError, SessionError, ValueError) as exc:
            return f"could not load session: {exc}"
        self.cache = loaded.cache
        self.tape = list(loaded.token_ids)
        self.pending = []
        self.turn_closed = loaded.turn_closed
        # The restored cache already holds the whole tape. A replay flag left
        # by an earlier cancelled turn would feed the tape through it again.
        self._replay_needed = False
        # A loaded snapshot contains only the target cache. The MTP block
        # has no history, so native MTP stays off for this conversation
        # until reset starts both histories together.
        self._decoder = None
        if self._mtp_active:
            self._mtp_blocked = True
        self._fused_pending = self.routing_profile == "fused-quality" and not self.tape
        self.thinking_enabled = loaded.thinking
        self.language._position_ids = loaded.position_ids
        self.language._rope_deltas = loaded.rope_deltas
        saved_mode = saved_profile.get("mode", "unknown")
        mode_note = (
            "" if saved_mode == self.routing_profile
            else f", saved as {saved_mode}, continuing as {self.routing_profile}"
        )
        return (f"loaded {name}  {len(self.tape)} tokens, "
                f"thinking={'on' if loaded.thinking else 'off'}{mode_note}, "
                "no old prefill"
                + (", MTP disabled until reset" if self._mtp_active else ""))

    def list_sessions(self) -> str:
        from models.flashnext.sessions import SessionError

        try:
            saved = self._sessions().list()
        except (OSError, SessionError, ValueError) as exc:
            return f"could not list sessions: {exc}"
        if not saved:
            return "no saved sessions"
        rows = []
        for item in saved:
            if item.valid:
                rows.append(f"  {item.name:<24} {item.cached_tokens:>7} tok  "
                            f"thinking {'on' if item.thinking else 'off'}")
            else:
                rows.append(f"  {item.name:<24} invalid: {item.error}")
        return "\n".join(rows)

    def delete_session(self, name: str) -> str:
        from models.flashnext.sessions import SessionError

        try:
            removed = self._sessions().delete(name)
        except (OSError, SessionError, ValueError) as exc:
            return f"could not delete session: {exc}"
        return f"{name} {'deleted' if removed else 'not found'}"

    def reset(self) -> None:
        if self._decoder is not None:
            self.cache = self._decoder.target_cache
        self._decoder = None
        # A reset conversation starts target and MTP histories together,
        # so a previously blocked MTP decoder may start again.
        self._mtp_blocked = False
        self._fused_pending = self.routing_profile == "fused-quality"
        self.cache = self.language.make_cache()
        self.tape = []
        self.pending = []
        self.turn_closed = True
        self._replay_needed = False
        self.language._position_ids = None
        self.language._rope_deltas = None

    def _mark_replay_needed(self) -> None:
        """Discard a partial cache while keeping the transcript."""
        if self._decoder is not None:
            self.cache = self._decoder.target_cache
        self._decoder = None
        self.cache = self.language.make_cache()
        self._replay_needed = bool(self.tape)
        self.turn_closed = False
        self.language._position_ids = None
        self.language._rope_deltas = None

    def _is_native_mtp_decoder(self, decoder) -> bool:
        from models.flashnext.speculative import MTPGreedy

        return isinstance(decoder, MTPGreedy)

    def _turn_allows_mtp(self) -> bool:
        """Whether this turn's settings can advance MTP history."""
        if not self._mtp_active:
            return False
        if self.routing_profile == "fused-quality":
            return False
        if not hasattr(self.language, "mtp"):
            return False
        sampling = getattr(self, "sampling", None)
        return bool(getattr(sampling, "greedy", True))

    def _native_mtp_eligible(self) -> bool:
        """Whether this turn may use the native MTP decoder."""
        return self._turn_allows_mtp() and not self._mtp_blocked

    def _start_native_mtp_decoder(self):
        """Attach MTPGreedy for exact greedy turns. Fall back otherwise."""
        if not self._native_mtp_eligible():
            return None
        from models.flashnext.speculative import MTPGreedy

        decoder = MTPGreedy(self.language, depth=self.mtp_depth)
        decoder.target_cache = self.cache
        self._decoder = decoder
        return decoder

    def _start_fused_decoder(self):
        if (
            not self._fused_pending
            or self.tape
            or len(self.pending) > self.fusion_max_prompt
        ):
            self._fused_pending = False
            return None

        from mlx_vlm import load as load_mlx_vlm
        from models.flashnext.speculative import FastDraftGreedy
        from models.flashnext.transient import materialize_model
        AutoTokenizer = _load_transformers_tokenizer()

        if not self.fusion_model:
            raise RuntimeError(
                "fused-quality needs a draft model. Set one with "
                "/settings fusion-model <path>"
            )
        if not os.path.isdir(self.fusion_model):
            raise RuntimeError(f"fused draft model not found: {self.fusion_model}")
        draft_model, draft_processor = load_mlx_vlm(self.fusion_model)
        draft_language = draft_model.language_model
        if int(draft_language.args.vocab_size) != int(self.language.args.vocab_size):
            raise RuntimeError("fused draft and target vocabularies differ")
        draft_tokenizer = AutoTokenizer.from_pretrained(self.fusion_model)
        if draft_tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise RuntimeError("fused draft and target token IDs differ")
        draft_tokenizer = None
        draft_processor = None
        materialize_model(draft_language)
        draft_model = None
        decoder = FastDraftGreedy(
            self.language,
            self.store,
            depth=self.fusion_block,
            draft_language=draft_language,
            fallback_on_reject=True,
            release_draft_before_verify=True,
            draft_min_margin=self.fusion_min_margin,
            draft_min_block=self.fusion_min_block,
            draft_margin_tokens=self.fusion_margin_tokens,
        )
        decoder.target_cache = self.cache
        self._decoder = decoder
        self._fused_pending = False
        return decoder

    @property
    def cache_tokens(self) -> int:
        for entry in self.cache:
            offset = getattr(entry, "offset", None)
            if offset is not None:
                return int(offset)
        return len(self.tape)

    def generate(self, max_tokens: int, out=None, on_prefilled=None,
                 on_prefill_progress=None, on_decode_token=None,
                 should_cancel=None) -> tuple[str, Stats]:
        """Feed everything pending through the model, then decode a reply (see ``_generate``)."""
        from models.flashnext import pool_lifecycle, purge_recovery

        # The activity record covers the whole call (prefill, decode, GPU work, expert reads and the cleanup in
        # the finally below), so the optional cache-purge recovery never runs during any of it and a purge that
        # is already running finishes before this request starts.
        # Counts this call while FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=1 so a concurrent request is never
        # left without its decode pool; a no-op, and no other behaviour, when the flag is off.
        # Recovery BEFORE the request: the engine is idle here, the activity record below is not held yet, so a purge
        # (when RAM is short for this request and file cache can fix it) cannot overlap any GPU work or live state.
        try:
            purge_recovery.prepare_for_request(
                len(self.pending) + (len(self.tape) if getattr(self, "_replay_needed", False) else 0))
        except Exception:
            pass
        try:
            from models.flashnext import memory_controller
            memory_controller.begin_request()
            self.turn_memory_before = self._turn_memory()
        except Exception:
            self.turn_memory_before = None
        with purge_recovery.ACTIVITY.request(), pool_lifecycle.exclusive_request():
            from models.flashnext import request_admission
            if request_admission.enabled():
                try:
                    replay = getattr(self, "_replay_needed", False)
                    tokens = len(self.pending) + (len(self.tape) if replay else 0)
                    snapshot_count = 0
                    if (os.environ.get("FLASHNEXT_PREFIX_CACHE") == "1"
                            and os.environ.get("FLASHNEXT_PREFIX_CACHE_RESTORE", "1") != "0"
                            and (replay or not self.tape)):
                        prompt = (list(self.tape) if replay else []) + list(self.pending)
                        # Count the cold plan even if a prefix may restore: a miss must
                        # remain safe. One extra covers the final required boundary.
                        snapshot_count = len(self._prefix_session().boundaries(prompt, 0)) + 1
                except Exception as error:
                    raise request_admission.RequestMemoryRefused(
                        f"cannot estimate request memory: {error}; retry when memory admission is available") from error
                request_admission.check(tokens, max_output_tokens=max_tokens, snapshot_count=snapshot_count)
            fault = pool_lifecycle.lifecycle_fault() or getattr(self, "_memory_cleanup_error", None)
            if fault:
                raise RuntimeError("previous memory cleanup failed; restart the process: " + fault)
            with pool_lifecycle.in_flight():
                try:
                    from models.flashnext import expert_pool
                    expert_pool.reset_decline(pool_lifecycle.store_of(self.language))   # re-check headroom once per request
                except Exception:
                    pass
                from models.flashnext import trace_phase

                def prefilled():
                    trace_phase.set_phase("decode")      # the safety trace labels each stage of the request
                    try:
                        from models.flashnext import decode_profile
                        decode_profile.begin()           # opt-in (FLASHNEXT_DECODE_PROFILE_DIR); never raises
                    except Exception:
                        pass
                    if on_prefilled is not None:
                        on_prefilled()

                trace_phase.set_phase("prefill")
                try:
                    return self._generate(max_tokens, out, prefilled, on_prefill_progress,
                                          on_decode_token, should_cancel)
                finally:
                    try:
                        from models.flashnext import decode_profile
                        decode_profile.finish()          # writes the decode profile when one was started
                    except Exception:
                        pass
                    try:
                        try:
                            if os.environ.get("FLASHNEXT_CHAT_GUARD") == "1":
                                gpu_sync.synchronize()
                        finally:
                            self._after_request()
                    finally:
                        trace_phase.set_phase("idle")

    def reclaim_owned_memory(self, reading) -> int:
        """Free Veystrix-owned temporary memory before a cache purge is considered; returns bytes freed.

        Called only while the engine is idle. The allocator cache is always safe to trim. RAM prefix snapshots
        move to SSD (nothing is lost) only when the machine is actually under pressure or nearly out of room, and
        the decode pool is never touched here."""
        freed = 0
        try:
            cache = mx.get_cache_memory()
            if cache:
                mx.clear_cache()
                freed += int(cache)
        except Exception:
            pass
        store = getattr(self, "_prefix", None)
        stressed = reading.pressure >= 2 or reading.available < 3.5e9
        if store is not None and stressed:
            try:
                freed += int(self._prefix_session().store().shrink_ram(0))
            except Exception:
                pass
        return freed

    def _turn_memory(self) -> dict:
        """Where this process and the machine stand right now (never raises; values are bytes unless named)."""
        from models.flashnext import memory_controller, purge_recovery

        row = {"mlx_active": int(mx.get_active_memory()), "mlx_cache": int(mx.get_cache_memory()),
               "mlx_peak": int(mx.get_peak_memory())}
        try:
            import subprocess
            rss = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            row["rss"] = int(rss) * 1024 if rss else None
        except Exception:
            row["rss"] = None
        try:
            reading = purge_recovery.read_system()
            row["categories"] = memory_controller.categorize(
                reading, mlx_active=row["mlx_active"], mlx_cache=row["mlx_cache"], rss=row["rss"] or 0)
            row["compressor"] = reading.compressor * reading.page
            row["wired"] = reading.wired * reading.page
        except Exception:
            pass
        return row

    def _release_between_turns(self) -> None:
        """Between requests the engine is idle: the request has finished, its tokens are evaluated, and the activity record
        is still held, so no purge can overlap. Release what is safe and record before and after, so any memory that
        grows between tool turns is visible per request.

        Order: a GPU barrier that cannot deadlock against MLX's completion handler (``gpu_sync``, never ``mx.synchronize``),
        the prefill ring fence, Python garbage (numpy staging buffers), then the MLX allocator cache. The live KV and
        recurrent state, the decode pool and exact prefix snapshots stay. RAM snapshots move to SSD only while the memory
        controller is ELEVATED. Never raises."""
        from models.flashnext import memory_controller, prefill_ring

        released = {"cache_cleared": False, "snapshots_moved_bytes": 0}
        try:
            gpu_sync.synchronize()
            prefill_ring.fence()
            import gc
            gc.collect()
            if int(mx.get_cache_memory()) > RELEASE_CACHE_BYTES:
                mx.clear_cache()
                released["cache_cleared"] = True
            try:
                from models.flashnext import pool_lifecycle
                store = pool_lifecycle.store_of(self.language)
                if store is not None and getattr(store, "_uncached", False):
                    store.set_uncached_reads(False)          # decode keeps the file cache for hot experts
                    released["buffered_reads_restored"] = True
            except Exception:
                pass
            if memory_controller.elevated() and getattr(self, "_prefix", None) is not None:
                released["snapshots_moved_bytes"] = int(self._prefix_session().store().shrink_ram(0))
            self.last_turn_memory = {"before": getattr(self, "turn_memory_before", None),
                                     "after": self._turn_memory(), "released": released,
                                     "controller": memory_controller.ledger()}
        except Exception as error:
            self.last_turn_memory = {"error": f"{type(error).__name__}: {error}"}

    def _after_request(self) -> None:
        """Memory cleanup after every request (success, cancellation and error alike); never raises."""
        from models.flashnext import memory_manager, pool_lifecycle

        self._release_between_turns()
        if not memory_manager.enabled():
            return
        if pool_lifecycle.lifecycle_fault():
            self._memory_cleanup_error = pool_lifecycle.lifecycle_fault()
            self.last_memory_after = {"error": self._memory_cleanup_error}
            return
        try:
            import mlx.core as mx

            prefix_store = None
            try:
                from models.flashnext import prefix_session
                if prefix_session.enabled() or getattr(self, "_prefix", None) is not None:
                    prefix_store = self._prefix_session().store()
            except Exception:
                prefix_store = None

            self.last_memory_after = memory_manager.get().after_request(
                pool_lifecycle.store_of(self.language),
                prefix_store=prefix_store,
                clear_cache=mx.clear_cache,
                synchronize=gpu_sync.synchronize,
            )
            if self.last_memory_after.get("error"):
                self._memory_cleanup_error = self.last_memory_after["error"]
                pool_lifecycle.mark_lifecycle_fault(self._memory_cleanup_error)
        except Exception as error:
            self.last_memory_after = {"error": f"{type(error).__name__}: {error}"}
            self._memory_cleanup_error = self.last_memory_after["error"]
            pool_lifecycle.mark_lifecycle_fault(self._memory_cleanup_error)

    def _make_room_for_prefill(self, prompt, prefix_cache_runs: bool, fresh: bool = True) -> None:
        """Opt-in memory planning before a prefill.

        With FLASHNEXT_MEMORY_MANAGER=1 the manager plans every prefill (pool budget and release, RAM snapshots,
        uncached short reads, refusal); with only FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=1 the decode pool is dropped
        before a fresh ring-eligible prefill. What counts is the tokens the prefill will really run: a prompt whose
        long system prefix is restored from the prefix cache leaves a short suffix, which never uses the ring, so its
        warm pool is kept. The caller decides that the conversation is fresh (empty tape or replay, empty cache, no
        draft decoder) before any state changes. Never raises, except MemoryBudgetRefused from the manager.
        """
        from models.flashnext import memory_manager, pool_lifecycle

        managed = memory_manager.enabled()
        if not managed and not pool_lifecycle.enabled():
            self.last_pool_release = None
            return
        reusable = 0
        if prefix_cache_runs:
            try:
                reusable = int(self._prefix_session().reusable_tokens(prompt))
            except Exception:              # an unreadable cache means "assume nothing is reused"
                reusable = 0
        tokens = len(prompt) - reusable
        if managed:
            self.last_memory_report = self._plan_memory(tokens, len(prompt), prefix_cache_runs, fresh)
            return
        report = pool_lifecycle.make_room(self.language, tokens)
        if report is not None:
            report["prompt_tokens"], report["reusable_tokens"] = len(prompt), reusable
        self.last_pool_release = report

    def _plan_memory(self, tokens: int, prompt_tokens: int, prefix_cache_runs: bool, fresh: bool) -> dict:
        import mlx.core as mx

        from models.flashnext import expert_pool, memory_manager, pool_lifecycle, prefill, prefill_ring

        ring_eligible = bool(
            prefill_ring.enabled() and expert_pool.enabled() and prefill_ring.configured_experts() > 0
            and tokens >= prefill_ring.min_tokens() and prefill_ring.scratch_enabled())
        request = memory_manager.Request(
            tokens=tokens, prompt_tokens=prompt_tokens, chunk_tokens=prefill.chunk_tokens() or max(tokens, 1),
            fresh=fresh, ring_eligible=ring_eligible)
        prefix_store = None
        if prefix_cache_runs:
            try:
                prefix_store = self._prefix_session().store()
            except Exception:
                prefix_store = None
        report = memory_manager.get().before_prefill(
            pool_lifecycle.store_of(self.language), request, prefix_store,
            synchronize=gpu_sync.synchronize, clear_cache=mx.clear_cache)
        plan = report["plan"]
        return {"released_bytes": report["released_bytes"], "evicted_ram_bytes": report["evicted_ram_bytes"],
                "applied": report["applied"], "tokens": tokens, "prompt_tokens": prompt_tokens,
                "ring_eligible": ring_eligible, "pool_gb": plan.pool_gb, "release_pool": plan.release_pool,
                "uncached_reads": plan.uncached_reads, "predicted_headroom": plan.predicted_headroom,
                "predicted_decode_headroom": plan.predicted_decode_headroom, "reasons": list(plan.reasons),
                "available_before": report["snapshot"].available, "pool_before": report["snapshot"].pool_bytes}

    def _generate(self, max_tokens: int, out=None, on_prefilled=None,
                  on_prefill_progress=None, on_decode_token=None,
                  should_cancel=None) -> tuple[str, Stats]:
        """Feed everything pending through the model, then decode a reply.

        `on_prefilled` fires once the prompt is in the cache and before the
        first token is decoded, so a caller can stop a prefill animation
        rather than leaving it running over the answer.
        """
        from models.flashnext.expert_cache import set_prefill_progress
        from models.flashnext.prefill import prefill_language

        if not self.pending:
            return "", Stats(finish="stop")
        requested_max_tokens = int(max_tokens)
        generation_horizon = (
            UNLIMITED_GENERATION_HORIZON
            if requested_max_tokens < 0
            else requested_max_tokens
        )
        interactive_budgets = getattr(self, "_interactive_budgets", None)
        budget_answer = budget_think = 0
        close_token = None
        if interactive_budgets is not None:
            raw_answer, raw_think = interactive_budgets
            # Negative means unlimited; keep it as None so limits below
            # never trigger. max(0, ...) would turn unlimited into zero.
            budget_answer = (
                None if raw_answer is None or int(raw_answer) < 0
                else max(0, int(raw_answer))
            )
            budget_think = (
                None if raw_think is None or int(raw_think) < 0
                else max(0, int(raw_think))
            )
            if self.thinking_enabled and budget_think is not None:
                close_ids = self.encode("</think>")
                if len(close_ids) != 1:
                    raise ValueError(
                        "interactive reasoning closure must encode as one token"
                    )
                close_token = int(close_ids[0])
        separate_budgets = (
            interactive_budgets is not None
            and self.thinking_enabled
            and budget_think is not None
            and budget_think > 0
        )
        decoder = self._decoder
        if (
            decoder is not None
            and self._is_native_mtp_decoder(decoder)
            and (
                separate_budgets or self._mtp_blocked
                or not self._turn_allows_mtp()
            )
        ):
            # The target history is about to advance without MTP history,
            # so the draft state is stale. Adopt the target cache, drop the
            # decoder, and stay on target decoding until reset.
            self.cache = decoder.target_cache
            self._decoder = None
            decoder = None
            self._mtp_blocked = True
        if separate_budgets and decoder is not None:
            # The fused draft path has no phase callback. Use the standard
            # target decoder for interactive turns with separate quotas.
            self.cache = decoder.target_cache
            self._decoder = None
            decoder = None
        if (
            decoder is None
            and self.routing_profile == "fused-quality"
            and not separate_budgets
        ):
            decoder = self._start_fused_decoder()
        if decoder is None and not separate_budgets:
            # Native MTP is opt-in and greedy-only. Anything else falls
            # back to the normal target decoder without changing sampling.
            decoder = self._start_native_mtp_decoder()
        if (
            decoder is None
            and self._mtp_active
            and not self._mtp_blocked
            and (self.pending or self._replay_needed)
        ):
            # This turn advances target history through the target path
            # without MTP history, even with max_tokens=0. An eligible
            # full replay cannot reach here: it constructed a decoder
            # above. Block native MTP until reset.
            self._mtp_blocked = True
        prompt = list(self.tape) + self.pending if self._replay_needed else list(self.pending)
        from models.flashnext import prefix_session

        fresh_prefix_cache = (
            prefix_session.enabled() and decoder is None
            and (self._replay_needed or not self.tape) and self.cache_tokens == 0
        )
        ids = mx.array(prompt)[None]
        prompt_tokens = int(ids.shape[1])
        fresh_for_pool = decoder is None and (self._replay_needed or not self.tape) and self.cache_tokens == 0
        gate_probe = None
        if os.environ.get("VEYSTRIX_LAYER_GATE_PROBE"):
            from models.flashnext.tests.bench.layer_gate_probe import Probe

            if decoder is not None or not getattr(self.sampling, "greedy", False):
                raise RuntimeError("layer gate requires greedy target-model decoding")
            gate_probe = Probe(os.environ["VEYSTRIX_LAYER_GATE_PROBE"])
        self.tape.extend(self.pending)
        self.pending = []
        self._replay_needed = False
        cancelled = False

        def check_cancel():
            nonlocal cancelled
            if should_cancel is not None and should_cancel():
                cancelled = True
                raise GenerationCancelled

        def preserve_or_replay():
            if cancelled and self.check_invariant():
                self.turn_closed = False
            else:
                self._mark_replay_needed()

        prefill_began = time.perf_counter()
        self.routing.reset()
        streamed_layers = sum(
            hasattr(
                getattr(getattr(layer, "mlp", None), "switch_mlp", None),
                "layer_id",
            )
            for layer in self.language.model.layers
        )
        completed_layers = set()

        def layer_completed(layer_id):
            progress.PROGRESS.tick()             # real forward progress for the out-of-process watchdog
            check_cancel()
            completed_layers.add(layer_id)
            if on_prefill_progress is not None and streamed_layers:
                confirmed = max(0, len(completed_layers) - 1)
                done = prompt_tokens * confirmed // streamed_layers
                on_prefill_progress(done, prompt_tokens)

        if on_prefill_progress is not None:
            on_prefill_progress(0, prompt_tokens)
        set_prefill_progress(layer_completed)
        try:
            check_cancel()
            self.last_pool_release = None
            self.last_memory_report = None
            from models.flashnext import memory_manager

            if fresh_for_pool or memory_manager.enabled():
                try:
                    self._make_room_for_prefill(prompt, fresh_prefix_cache, fresh_for_pool)
                except memory_manager.MemoryBudgetRefused:
                    self._mark_replay_needed()        # nothing was allocated; the conversation can be retried
                    raise
            if decoder is None and fresh_prefix_cache:
                # Opt-in (FLASHNEXT_PREFIX_CACHE=1): restore the longest stored exact prefix, prefill only
                # the rest, snapshot at the system-prompt boundary. A fresh conversation only.
                token, self.cache, self.last_prefix = self._prefix_session().prefill(
                    self.language, prompt, self.cache, sampler=Sampler(self.sampling)
                )
            elif decoder is None:
                self.last_prefix = None
                _, token = prefill_language(
                    self.language, ids, self.cache, sampler=Sampler(self.sampling)
                )
            else:
                decoder.append(ids)
                self.cache = decoder.target_cache
        except GenerationCancelled:
            preserve_or_replay()
            raise
        except Exception:
            if gate_probe is not None:
                self._mark_replay_needed()
            raise
        finally:
            set_prefill_progress(None)
        prefill_seconds = time.perf_counter() - prefill_began
        if gate_probe is not None:
            try:
                gate_probe.prefilled(self, token, prefill_seconds)
            except Exception:
                self._mark_replay_needed()
                raise
        if on_prefill_progress is not None:
            on_prefill_progress(prompt_tokens, prompt_tokens)
        if on_prefilled is not None:
            on_prefilled()
        produced, partial, pieces = [], [], []
        budget_state = (
            {
                "phase": "thinking" if self.thinking_enabled else "answer",
                "thinking": 0,
                "answer": 0,
                "answer_limited": False,
                "stop_seen": False,
                "forced_close": False,
            }
            if interactive_budgets is not None else None
        )
        tail_began = None
        tail_index = 0
        timer = DecodeTimer()
        finish = "length"
        self.routing.begin_decode()
        if decoder is not None:
            decoder.set_route_observer(self.routing._observe)

        sampler = Sampler(self.sampling)

        def standard_tokens():
            nonlocal token
            for _index in range(generation_horizon):
                value = int(token.item())
                force_close = (
                    budget_state is not None
                    and budget_state["phase"] == "thinking"
                    and budget_think is not None
                    and budget_think > 0
                    and budget_state["thinking"] == budget_think - 1
                )
                if value in self.stops:
                    if budget_state is not None:
                        budget_state["stop_seen"] = True
                    return
                if force_close:
                    value = close_token
                    token = mx.array([value], dtype=mx.uint32)
                if budget_state is not None:
                    phase = budget_state["phase"]
                    if (
                        phase == "answer"
                        and budget_answer is not None
                        and budget_state["answer"] >= budget_answer
                    ):
                        budget_state["answer_limited"] = True
                        return
                if budget_state is not None:
                    budget_state["forced_close"] = force_close
                yield value
                sampler.observe(value)
                step = self.language(token[None], cache=self.cache)
                token = sampler(step.logits[:, -1, :])
                mx.eval(token)

        tokens = (
            standard_tokens()
            if decoder is None
            else decoder.generate(generation_horizon, self.stops)
        )
        decode_interrupted = False
        try:
            for index, value in enumerate(tokens):
                check_cancel()
                produced.append(value)
                self.tape.append(value)
                if self.routing.after_token(index + 1, generation_horizon):
                    tail_began = timer.mark()
                    # Token warmup+1 was generated before pinning. The next
                    # model call produces the first pinned-tail output.
                    tail_index = len(produced)
                if budget_state is not None and budget_state["forced_close"]:
                    prefix = self.tokenizer.decode(partial) if partial else ""
                    partial.clear()
                    piece = prefix + "</think>"
                else:
                    piece = stream_decode(self.tokenizer, partial, value)
                if budget_state is not None:
                    phase = budget_state["phase"]
                    budget_state[phase] += 1
                    if phase == "thinking" and (
                        budget_state["forced_close"] or "</think>" in piece
                    ):
                        budget_state["phase"] = "answer"
                progress.PROGRESS.tick(len(produced) + 1)
                if on_decode_token is not None:
                    on_decode_token(value, piece)
                if not piece:
                    continue
                pieces.append(piece)
                if out is not None:
                    with timer.emitting():
                        out(piece)
            if budget_state is not None and budget_state["answer_limited"]:
                finish = "length"
                self.turn_closed = False
            elif budget_state is not None and budget_state["stop_seen"]:
                # Stop IDs are protocol delimiters. We leave them out of the
                # tape and close the turn synthetically on the next append.
                finish = "stop"
                self.turn_closed = False
            elif len(produced) < generation_horizon:
                finish = "stop"
                self.turn_closed = False
            else:
                self.turn_closed = False
        except GenerationCancelled:
            decode_interrupted = True
            raise
        finally:
            if decoder is not None:
                decoder.set_route_observer(None)
                self.cache = decoder.target_cache
            self.routing.finish_decode()
            if decode_interrupted:
                preserve_or_replay()

        tail_tokens = len(produced) - tail_index if tail_began is not None else 0
        tail_seconds = timer.since(tail_began) if tail_began is not None else 0.0
        stats = Stats(
            finish=finish, tokens=len(produced), seconds=timer.elapsed(),
            prompt_tokens=prompt_tokens, prefill_seconds=prefill_seconds,
            tail_tokens=tail_tokens, tail_seconds=tail_seconds,
            ui_seconds=timer.emitted,
            pinned_bytes=self.routing.pinned_bytes,
            pinned_signature=self.routing.pinned_signature,
        )
        if gate_probe is not None:
            try:
                gate_probe.finished(produced, stats)
            except Exception:
                self._mark_replay_needed()
                raise
        text = "".join(pieces)
        return text, stats
