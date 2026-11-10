"""Pure logic for the end-to-end prefill-memory benchmark.

Nothing here imports MLX, loads a checkpoint or starts a benchmark. The
orchestrator and child in ``bench_prefill_memory`` use it, and the offline
unit tests exercise every function with fakes. The plan it implements is
section 14 of ``docs/flashnext/prefetch-integration.md``.

Contents: the condition table, the arm schedule, parsers for the macOS power,
thermal, swap and pressure readings, the safety policy and watchdog, the
child-process supervisor, and the exactness, gate and paired-effect analysis.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

SIZES = {"1k": 1024, "2k": 2048, "4k": 4096}                  # the planned matrix (section 14)
DIAGNOSTIC_SIZES = {"128": 128, "256": 256}                   # short prompts for memory diagnosis only
ALL_SIZES = {**DIAGNOSTIC_SIZES, **SIZES}
DECODE_TOKENS = 128
RING_EXPERTS = 512
FOLLOW_UP_OFFSET = 20000
TEMPLATE_OVERHEAD_FIRST = 36      # preface, system turn and markers around the first user text (measured with the real tokenizer)
TEMPLATE_OVERHEAD_LATER = 23      # preface and markers around a later user text (measured)
METAL_LIMIT_BYTES = 12_713_115_648   # recorded on this machine; the child reads the live value
HEADROOM_BYTES = 500_000_000      # G3
NEXT_SIZE_HEADROOM_BYTES = 2_000_000_000
MIN_AVAILABLE_GB = 2.0            # stop before macOS reaches a pressure level: free + inactive + speculative
GUARD_HEADROOM_BYTES = 500_000_000   # MLX active + cache must stay this far under the Metal limit
MAX_COMPRESSOR_GROWTH_GIB = 2.0   # per arm, relative to the arm's own starting compressor size
COMPRESSOR_STOP_AVAILABLE_GIB = 3.0   # growth over the limit only stops the arm below this much available memory
FAST_TRACE_SECONDS = 0.25         # child sampling interval during prefill (cheap readings only)
GUARD_EXIT_CODE = 75              # the child's own memory guard stopped it (EX_TEMPFAIL)
CLEANUP_TOLERANCE_BYTES = 250_000_000   # G4
MIN_ROUNDS_FOR_EFFECT = 3
TIME_LIMIT_SECONDS = {"128": 300, "256": 300, "1k": 600, "2k": 1200, "4k": 2400}

CACHE_WARNING = (
    "The first arm of a group reads the checkpoint cold from SSD and later arms find part of it in the page "
    "cache, so order alone changes the timing (a page-cache effect once inflated a smoke ratio to 1.55x). "
    "With fewer than 3 rounds in reversed order this is not a speed measurement: do not read an F0 versus "
    "F2 speed ratio from it. Free memory and page-in counters are recorded per arm to show the cache state."
)

SYSTEM_TEXT = "You are a careful assistant. Answer briefly."
FIRST_PREFACE = "Summarize the following text in one sentence.\n\n"
LATER_PREFACE = "Now summarize this further text in one sentence.\n\n"


# -- conditions -------------------------------------------------------------

def production_flags() -> dict[str, str]:
    """Environment names of the opt-in flags, read from the production modules."""
    from models.flashnext import linear_state, memory_manager, pool_lifecycle, prefill, prefill_memory, prefill_ring, prefix_session

    return {
        "ring": prefill_ring.FLAG_EXPERTS,
        "scratch": prefill_ring.FLAG_SCRATCH,
        "cache_mb": prefill_memory.FLAG_CACHE_MB,
        "min_tokens": prefill_ring.FLAG_MIN_TOKENS,
        "margin_gb": prefill_ring.FLAG_MARGIN_GB,
        "transient_gb": prefill_ring.FLAG_TRANSIENT_GB,
        "linear_state": linear_state.FLAG,
        "chunk_tokens": prefill.FLAG_CHUNK_TOKENS,
        "chunk_policy": prefill.FLAG_CHUNK_POLICY,
        "prefix_cache": prefix_session.FLAG,
        "prefix_restore": prefix_session.FLAG_RESTORE,
        "release_pool": pool_lifecycle.FLAG,
        "memory_manager": memory_manager.FLAG,
        "short_uncached": memory_manager.FLAG_SHORT_UNCACHED,
    }


@dataclass(frozen=True)
class Condition:
    id: str
    turn: str                 # "first": measure turn 1. "second": turn 1 is setup, measure turn 2.
    ring: int = 0             # experts per ring half; 0 = ring off
    scratch: bool = False
    cache_mb: int = 0         # 0 = no cap
    guarded: bool = False     # runs last, only at the sizes in ``only_sizes``
    only_sizes: tuple[str, ...] = ()
    summary: str = ""
    # Test-only control: the baseline with F_NOCACHE on the store's shard descriptors, set by the
    # benchmark child on its own store object. No environment flag, no production change.
    uncached_reads: bool = False
    # FLASHNEXT_EVAL_LINEAR_STATE=1: evaluate each GatedDeltaNet conv state as its layer finishes. Read at
    # import, so it applies to first-turn arms (the flag is in the child's launch environment).
    linear_state: bool = False
    chunk_tokens: int = 0         # FLASHNEXT_PREFILL_CHUNK_TOKENS; 0 = one call
    chunk_policy: str = ""        # FLASHNEXT_PREFILL_CHUNK_POLICY; "" = unset (the grid policy)
    prefix_cache: bool = False    # FLASHNEXT_PREFIX_CACHE=1 (directory and budgets come from the child options)
    release_pool: bool = False    # FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=1: drop the decode pool before a fresh ring-eligible prefill
    memory_manager: bool = False  # FLASHNEXT_MEMORY_MANAGER=1: the adaptive memory manager plans every prefill
    short_uncached: bool = False  # FLASHNEXT_MEMORY_SHORT_UNCACHED=1: short prefills read experts with F_NOCACHE
    prefix_restore: bool = True   # False: FLASHNEXT_PREFIX_CACHE_RESTORE=0, the session's chunk plan without restoring or snapshots

    @property
    def is_baseline(self) -> bool:
        return (self.ring == 0 and not self.scratch and self.cache_mb == 0 and not self.uncached_reads
                and not self.linear_state and not self.chunk_tokens and not self.prefix_cache
                and not self.chunk_policy)

    @property
    def ring_on(self) -> bool:
        return self.ring > 0


CONDITIONS: dict[str, Condition] = {c.id: c for c in (
    Condition("F0", "first", summary="baseline: no pool exists during prefill"),
    Condition("F1", "first", ring=RING_EXPERTS, guarded=True, only_sizes=("1k",),
              summary="ring in a fresh 6 GB pool (about 11.1 GB, estimate)"),
    Condition("F2", "first", ring=RING_EXPERTS, scratch=True, summary="scratch pool"),
    Condition("F3a", "first", ring=RING_EXPERTS, scratch=True, cache_mb=512,
              summary="scratch pool and 512 MB cache cap"),
    Condition("F3b", "first", ring=RING_EXPERTS, scratch=True, cache_mb=1024,
              summary="scratch pool and 1024 MB cache cap"),
    Condition("F4", "first", cache_mb=512, summary="512 MB cache cap alone"),
    Condition("S0", "second", summary="baseline: pool warm"),
    Condition("S1", "second", ring=RING_EXPERTS, summary="in-pool ring"),
    Condition("S2", "second", ring=RING_EXPERTS, cache_mb=512,
              summary="in-pool ring and 512 MB cache cap"),
)}
# Not in the planned matrix: diagnostic controls, run only when named.
SMALL_RING_EXPERTS = 256          # half the planned ring: a 0.84 GB scratch pool instead of 1.62 GB

DIAGNOSTIC_CONDITIONS: dict[str, Condition] = {c.id: c for c in (
    Condition("F2s", "first", ring=SMALL_RING_EXPERTS, scratch=True,
              summary="diagnostic: F2 with a 256-expert scratch ring; a layer that routes more experts "
                      "than 256 falls back to the normal read path"),
    Condition("F2L", "first", ring=RING_EXPERTS, scratch=True, linear_state=True,
              summary="F2 plus eager GatedDeltaNet state evaluation (FLASHNEXT_EVAL_LINEAR_STATE=1)"),
    Condition("F2P", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=512,
              prefix_cache=True,
              summary="F2L plus 512-token chunked prefill and the exact prefix cache (long-context validation)"),
    Condition("F2B", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=2048,
              chunk_policy="balanced", prefix_cache=True,
              summary="F2L plus boundary-aware 2,048-token chunked prefill and the exact prefix cache"),
    Condition("F2Pn", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=512,
              prefix_cache=True, prefix_restore=False,
              summary="F2P with restore and snapshots off: the identical 512-token chunk plan, always cold (same-plan reference)"),
    Condition("F2Bn", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=2048,
              chunk_policy="balanced", prefix_cache=True, prefix_restore=False,
              summary="F2B with restore and snapshots off: the identical balanced 2,048-token plan, always cold (same-plan reference)"),
    Condition("F2Bnr", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=2048,
              chunk_policy="balanced", prefix_cache=True, prefix_restore=False, release_pool=True,
              summary="F2Bn plus the decode-pool release before a fresh long prefill (sequential cold requests)"),
    Condition("F2Br", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=2048,
              chunk_policy="balanced", prefix_cache=True, release_pool=True,
              summary="F2B plus the decode-pool release: balanced 2,048 plan, exact prefix cache, sequential long requests"),
    Condition("F2Pnr", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=512,
              prefix_cache=True, prefix_restore=False, release_pool=True,
              summary="F2Pn plus the decode-pool release: the validated 512-token plan, always cold (quality reference)"),
    Condition("F2Bm", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=2048,
              chunk_policy="balanced", prefix_cache=True, memory_manager=True,
              summary="F2B under the adaptive memory manager (automatic pool budget, release, eviction)"),
    Condition("F2Bmu", "first", ring=RING_EXPERTS, scratch=True, linear_state=True, chunk_tokens=2048,
              chunk_policy="balanced", prefix_cache=True, memory_manager=True, short_uncached=True,
              summary="F2Bm plus uncached expert reads during short prefills"),
    Condition("F0u", "first", uncached_reads=True,
              summary="test-only control: baseline with uncached (F_NOCACHE) expert reads; no pool, no ring"),
)}
ALL_CONDITIONS = {**CONDITIONS, **DIAGNOSTIC_CONDITIONS}
BASELINES = {"first": "F0", "second": "S0"}


def condition_overlay(cond: Condition, flags: dict[str, str] | None = None) -> dict[str, str | None]:
    """The flag values a condition sets. ``None`` means the variable is unset.

    Every opt-in flag appears, so a variable inherited from the caller's shell
    can never leak into an arm. The baseline overlay unsets all of them, which
    leaves exactly the production chat environment.
    """
    names = flags or production_flags()
    overlay: dict[str, str | None] = {name: None for name in names.values()}
    if cond.ring:
        overlay[names["ring"]] = str(cond.ring)
    if cond.scratch:
        overlay[names["scratch"]] = "1"
    if cond.cache_mb:
        overlay[names["cache_mb"]] = str(cond.cache_mb)
    if cond.linear_state:
        overlay[names["linear_state"]] = "1"
    if cond.chunk_tokens:
        overlay[names["chunk_tokens"]] = str(cond.chunk_tokens)
    if cond.chunk_policy:
        overlay[names["chunk_policy"]] = cond.chunk_policy
    if cond.release_pool:
        overlay[names["release_pool"]] = "1"
    if cond.memory_manager:
        overlay[names["memory_manager"]] = "1"
    if cond.short_uncached:
        overlay[names["short_uncached"]] = "1"
    if cond.prefix_cache:
        overlay[names["prefix_cache"]] = "1"
        if not cond.prefix_restore:
            overlay[names["prefix_restore"]] = "0"
    return overlay


def launch_overlay(cond: Condition, flags: dict[str, str] | None = None) -> dict[str, str | None]:
    """Flags in the child's environment at launch.

    A second-turn arm launches with the baseline overlay: its first turn is
    setup and must be identical in every arm. The child applies the condition's
    flags only after that turn (flags are read at call time).
    """
    base = Condition("setup", cond.turn)
    return condition_overlay(cond if cond.turn == "first" else base, flags)


def apply_overlay(env: dict, overlay: dict[str, str | None]) -> dict:
    """Set or unset each variable in ``env`` in place and return it."""
    for name, value in overlay.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


# -- schedule ---------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    index: int
    size: str
    condition: str
    round: int                # 0-based
    reverse: bool

    @property
    def arm_id(self) -> str:
        return f"{self.size}-{self.condition}-r{self.round + 1}"


def _block(conditions: list[str], rounds: int) -> list[tuple[str, int, bool]]:
    out = []
    for round_index in range(rounds):
        reverse = round_index % 2 == 1
        order = list(reversed(conditions)) if reverse else list(conditions)
        out.extend((name, round_index, reverse) for name in order)
    return out


def build_schedule(sizes: list[str], conditions: list[str], rounds: int) -> list[Arm]:
    """Fresh-process arms: sizes ascending; per size the first-turn block, the
    second-turn block, then guarded conditions last. Even rounds run in listed
    order, odd rounds reversed."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    unknown = [name for name in conditions if name not in ALL_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    bad = [name for name in sizes if name not in ALL_SIZES]
    if bad:
        raise ValueError(f"unknown sizes: {bad}")
    arms: list[Arm] = []
    for size in sorted(sizes, key=lambda name: ALL_SIZES[name]):
        chosen = [ALL_CONDITIONS[name] for name in conditions]
        allowed = [c for c in chosen if not c.only_sizes or size in c.only_sizes]
        blocks = [
            [c.id for c in allowed if c.turn == "first" and not c.guarded],
            [c.id for c in allowed if c.turn == "second" and not c.guarded],
            [c.id for c in allowed if c.guarded],
        ]
        for block in blocks:
            if not block:
                continue
            for name, round_index, reverse in _block(block, rounds):
                arms.append(Arm(len(arms), size, name, round_index, reverse))
    return arms


# -- prompts ---------------------------------------------------------------

def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def digest_ids(ids) -> str:
    return hashlib.sha256(repr([int(i) for i in ids]).encode()).hexdigest()[:16]


def build_prompts(tokenizer, source: str, tokens: int, offset: int = FOLLOW_UP_OFFSET) -> dict:
    """Reproducible first and second user texts from one fixed source text.

    The first text is source tokens ``[0, n1)``, the second ``[offset, offset+n2)``,
    sized so the whole prompt is near ``tokens`` once the template is added.
    The actual prefilled count is measured, never assumed.
    """
    ids = tokenizer.encode(source)
    first_n = max(16, tokens - TEMPLATE_OVERHEAD_FIRST)
    later_n = max(16, tokens - TEMPLATE_OVERHEAD_LATER)
    if len(ids) < offset + later_n:
        raise ValueError("source text is too short for the requested size")
    first = FIRST_PREFACE + tokenizer.decode(ids[:first_n])
    later = LATER_PREFACE + tokenizer.decode(ids[offset:offset + later_n])
    return {
        "first": first, "later": later, "system": SYSTEM_TEXT,
        "source_hash": digest_text(source),
        "first_hash": digest_text(first), "later_hash": digest_text(later),
        "first_ids": first_n, "later_ids": later_n,
    }


LONG_SYSTEM_HEADER = "\n\nReference notes. Read them before answering:\n"
QUESTION_PREFACE = "Using the notes above, answer in one sentence: what does this passage say?\n\n"
LONG_NOTES_OFFSET = 1000          # source tokens used for the long system message
LONG_QUESTION_OFFSETS = {"A": 30000, "B": 31000}
LONG_QUESTION_TOKENS = 48


AGENT_PROMPT_FILE = Path(__file__).resolve().parent / "fixtures" / "agent_system_prompt.txt"
AGENT_QUESTIONS = {
    "A": "What does the ring buffer in this repository do?",
    "B": "List the tests that cover the prefix cache.",
}


def build_agent_prompts(tokenizer, path=AGENT_PROMPT_FILE) -> dict:
    """A realistic coding-agent system prompt (about 2K tokens) and two different short user questions.

    The token counts are measured with the real chat template, and the system-message count is where the
    first ``<|im_end|>\\n`` ends, which is where the prefix cache snapshots.
    """
    from veystrix.conversation import Conversation

    system = Path(path).read_text().strip()
    marker = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    counts = {}
    for name, question in AGENT_QUESTIONS.items():
        conversation = Conversation(tokenizer)
        conversation.open_conversation(system, question, enable_thinking=False)
        ids = list(conversation.pending)
        boundary = next(i + len(marker) for i in range(len(ids)) if ids[i:i + len(marker)] == marker)
        counts[name] = {"prompt_tokens": len(ids), "system_message_tokens": boundary,
                        "user_and_generation_tokens": len(ids) - boundary}
    return {
        "system": system, "questions": dict(AGENT_QUESTIONS), "system_hash": digest_text(system),
        "question_hashes": {k: digest_text(v) for k, v in AGENT_QUESTIONS.items()},
        "token_counts": counts, "system_words": len(system.split()),
    }


def build_long_prompts(tokenizer, source: str, tokens: int = 1025) -> dict:
    """A long system message (about ``tokens`` minus the question) and two different short questions.

    Lets one system prompt be reused across requests. The system text is the standard short sentence plus
    reference notes cut from the fixed source text, sized so that the first request is close to ``tokens``
    after the chat template. Token counts are measured with the real template, never assumed.
    """
    from veystrix.conversation import Conversation

    ids = tokenizer.encode(source)
    needed = max(LONG_QUESTION_OFFSETS.values()) + LONG_QUESTION_TOKENS
    if len(ids) < needed:
        raise ValueError("source text is too short for the long-system prompts")
    questions = {name: QUESTION_PREFACE + tokenizer.decode(ids[offset:offset + LONG_QUESTION_TOKENS])
                 for name, offset in LONG_QUESTION_OFFSETS.items()}

    def system_for(count: int) -> str:
        return SYSTEM_TEXT + LONG_SYSTEM_HEADER + tokenizer.decode(ids[LONG_NOTES_OFFSET:LONG_NOTES_OFFSET + count])

    def measure(system: str, question: str) -> int:
        conversation = Conversation(tokenizer)
        return int(conversation.open_conversation(system, question, enable_thinking=False))

    count = max(16, tokens - TEMPLATE_OVERHEAD_FIRST - LONG_QUESTION_TOKENS - 24)
    for _ in range(4):                                  # converge on the requested size
        got = measure(system_for(count), questions["A"])
        if got == tokens:
            break
        count += tokens - got
    system = system_for(count)
    return {
        "system": system, "questions": questions, "system_hash": digest_text(system),
        "question_hashes": {k: digest_text(v) for k, v in questions.items()},
        "tokens": {k: measure(system, v) for k, v in questions.items()},
        "notes_tokens": count,
    }


# -- parsers for the macOS readings ------------------------------------------

def parse_battery(text: str) -> dict:
    """``pmset -g batt``: charge, power source and charging state."""
    source = None
    match = re.search(r"Now drawing from '([^']+)'", text)
    if match:
        source = "ac" if "AC" in match.group(1) else "battery"
    percent = None
    state = None
    match = re.search(r"(\d+)%;\s*([A-Za-z ]+?)(?:;|$)", text, re.MULTILINE)
    if match:
        percent = int(match.group(1))
        state = match.group(2).strip().lower()
    return {"percent": percent, "source": source, "state": state}


def parse_low_power(text: str) -> bool | None:
    """``pmset -g``: Low Power Mode on or off, None if the line is absent."""
    match = re.search(r"^\s*lowpowermode\s+(\d)", text, re.MULTILINE)
    return None if match is None else match.group(1) == "1"


def parse_thermal(text: str) -> dict:
    """``pmset -g therm``: CPU speed limit and whether a warning was recorded."""
    limit = None
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text)
    if match:
        limit = int(match.group(1))
    warning = bool(re.search(r"warning level", text, re.IGNORECASE)) and \
        "No thermal warning level has been recorded" not in text
    return {"speed_limit": limit, "warning": warning}


def parse_swap_used_mb(text: str) -> float | None:
    """``sysctl vm.swapusage``: swap in use, in megabytes."""
    match = re.search(r"used\s*=\s*([\d.]+)([MGK])", text)
    if not match:
        return None
    scale = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[match.group(2)]
    return float(match.group(1)) * scale


def parse_pressure_level(text: str) -> int | None:
    """``sysctl -n kern.memorystatus_vm_pressure_level``: 1 normal, 2 warning, 4 critical."""
    match = re.search(r"(\d+)", text)
    return None if match is None else int(match.group(1))


def _run_text(command: list[str], timeout: float = 5) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    return result.stdout


class Sampler:
    """Reads the machine state. ``runner`` is injectable so tests need no macOS."""

    def __init__(self, runner: Callable[[list[str]], str] = _run_text,
                 vm: Callable[[], dict] | None = None,
                 clock: Callable[[], float] = time.time,
                 load: Callable[[], float] | None = None,
                 available: Callable[[], float | None] | None = None,
                 memory: Callable[[], dict] | None = None):
        self.runner = runner
        self.vm = vm
        self.clock = clock
        self.load = load
        self.available = available
        self.memory = memory

    def sample(self) -> dict:
        battery = parse_battery(self.runner(["pmset", "-g", "batt"]))
        thermal = parse_thermal(self.runner(["pmset", "-g", "therm"]))
        record = {
            "t": self.clock(),
            "battery_percent": battery["percent"],
            "power_source": battery["source"],
            "battery_state": battery["state"],
            "low_power": parse_low_power(self.runner(["pmset", "-g"])),
            "thermal_speed_limit": thermal["speed_limit"],
            "thermal_warning": thermal["warning"],
            "swap_used_mb": parse_swap_used_mb(self.runner(["sysctl", "vm.swapusage"])),
            "pressure_level": parse_pressure_level(
                self.runner(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
            ),
        }
        if self.load is not None:
            record["load"] = self.load()
        if self.available is not None:
            record["available_mb"] = self.available()
        if self.memory is not None:        # one vm_stat read: available, compressor, page cache
            reading = self.memory() or {}
            for key in ("available_mb", "compressor_mb", "file_backed_mb"):
                if reading.get(key) is not None:
                    record[key] = reading[key]
        if self.vm is not None:
            record["vm"] = self.vm()
        return record


def vm_delta(before: dict | None, after: dict | None) -> dict:
    """Counter movement between two ``vm`` readings (empty if either is missing)."""
    if not before or not after:
        return {}
    return {key: after[key] - before.get(key, 0) for key in after}


# -- safety -----------------------------------------------------------------

def _page_bytes() -> int:
    """The VM page size ``vm_stat`` counts in (16 KiB on Apple silicon)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 16384


def swapin_mib(vm: dict | None, page_bytes: int) -> float:
    """Swap-in growth in MiB from a counter movement (the counters are in pages)."""
    return max((vm or {}).get("swapin", 0), 0) * page_bytes / (1024 * 1024)


@dataclass
class SafetyPolicy:
    min_battery_percent: int = 25
    swap_growth_mb: float = 64.0
    # Cumulative swap-in growth allowed per arm, measured from that arm's own counter.
    # A few pages are touched back in when a process maps its memory while old swap
    # from other processes exists (a 4-page, 64 KiB swap-in stopped the first real run),
    # so a tiny historical swap-in is not swapping. Swap-out stays strict.
    swapin_limit_mib: float = 64.0
    # Stop on low available memory, not only on a macOS pressure level (level 4 was
    # reached within one 5 s sample on the first real run).
    min_available_gb: float = MIN_AVAILABLE_GB
    # The compressor holds memory macOS squeezed out of other pages. Growth of this much in
    # one arm means the arm is pushing the machine into compression (2.75 GB at 128 tokens
    # in F0), which "available memory" does not show because page cache counts as available.
    compressor_limit_mib: float = MAX_COMPRESSOR_GROWTH_GIB * 1024
    # Growth over that limit is a WARNING (recorded, never silent). It stops the arm only together
    # with low available memory: five arms were stopped by growth alone while pressure was 1,
    # 3.8 GB or more was available and nothing swapped, which answered nothing.
    compressor_stop_available_mib: float = COMPRESSOR_STOP_AVAILABLE_GIB * 1024
    page_bytes: int = field(default_factory=_page_bytes)
    pressure_stop_level: int = 2
    thermal_floor: int = 100
    thermal_samples: int = 2
    final: bool = False


@dataclass
class Watchdog:
    """Decides whether a sample means the run must stop. State: swap at the
    start of the run and the number of consecutive throttled samples."""

    policy: SafetyPolicy
    start: dict
    throttled: int = 0
    history: list = field(default_factory=list)
    last: dict = field(default_factory=dict)  # the figures behind the latest check
    warnings: list = field(default_factory=list)  # compressor growth over the limit, without a stop

    def check(self, sample: dict, vm: dict | None = None, arm_start: dict | None = None) -> str | None:
        """A stop reason, or None. ``vm`` is the counter movement since the arm began and
        ``arm_start`` the machine sample taken just before it (the compressor baseline).

        The figures behind the check are saved in ``last`` before any rule can
        return, so the sample that triggers a stop is always in the trace."""
        policy = self.policy
        self.history.append(sample)
        grown = swapin_mib(vm, policy.page_bytes)
        level = sample.get("pressure_level")
        compressor = sample.get("compressor_mb")
        compressor_base = (arm_start or {}).get("compressor_mb")
        growth = (compressor - compressor_base
                  if compressor is not None and compressor_base is not None else None)
        self.last = {"t": sample.get("t"), "pressure_level": level,
                     "swap_used_mb": sample.get("swap_used_mb"),
                     "available_mb": sample.get("available_mb"),
                     "compressor_mb": compressor,
                     "compressor_growth_mb": None if growth is None else round(growth, 1),
                     "compressor_warning": bool(growth is not None and growth > policy.compressor_limit_mib),
                     "swapin_mib": round(grown, 3),
                     "swapout_pages": max((vm or {}).get("swapout", 0), 0)}
        if sample.get("low_power"):
            return "Low Power Mode is on"
        percent = sample.get("battery_percent")
        source = sample.get("power_source")
        if source == "battery" and percent is not None and percent < policy.min_battery_percent:
            return f"battery {percent}% is below {policy.min_battery_percent}%"
        if policy.final:
            if source != "ac":
                return "a final run needs AC power"
            if sample.get("battery_state") == "discharging":
                return "AC is reported but the battery is discharging; final-run power is ambiguous"
            if sample.get("low_power") is None:
                return "Low Power Mode state is unreadable in a final run"
        if level is not None and level >= policy.pressure_stop_level:
            return f"memory pressure level {level}"
        available = sample.get("available_mb")
        if available is not None and available < policy.min_available_gb * 1000:
            return (f"available memory {available / 1000:.2f} GB is below "
                    f"{policy.min_available_gb:.1f} GB")
        if self.last["compressor_warning"]:
            self.warnings.append({"t": sample.get("t"), "growth_mb": round(growth, 1),
                                  "available_mb": available})
            if available is not None and available < policy.compressor_stop_available_mib:
                return (f"compressor grew {growth / 1024:.2f} GiB since the arm began "
                        f"(limit {policy.compressor_limit_mib / 1024:.0f} GiB) while only "
                        f"{available / 1024:.2f} GiB was available "
                        f"(floor {policy.compressor_stop_available_mib / 1024:.0f} GiB)")
        if (vm or {}).get("swapout", 0) > 0:
            return "swapping began (swapout)"
        if grown > policy.swapin_limit_mib:
            return (f"swap-in grew {grown:.1f} MiB since the arm began "
                    f"(limit {policy.swapin_limit_mib:.0f} MiB)")
        used = sample.get("swap_used_mb")
        base = self.start.get("swap_used_mb")
        if used is not None and base is not None and used - base > policy.swap_growth_mb:
            return f"swap grew {used - base:.0f} MB"
        limit = sample.get("thermal_speed_limit")
        if limit is not None and limit < policy.thermal_floor:
            self.throttled += 1
            if self.throttled >= policy.thermal_samples:
                return f"thermal throttling (CPU speed limit {limit})"
        else:
            self.throttled = 0
        return None


def classify_exit(returncode: int | None) -> tuple[str, str]:
    """(status, message) for a child exit status."""
    if returncode == 0:
        return "ok", "exited normally"
    if returncode == GUARD_EXIT_CODE:
        return "memory_guard", "the child's memory guard stopped it (see the arm's trace file)"
    if returncode in (137, -9):
        return "exit_137", "killed with SIGKILL (a memory kill is likely)"
    if returncode is not None and returncode < 0:
        return "signal", f"killed by signal {-returncode}"
    return "failed", f"exited with status {returncode}"


@dataclass
class ChildResult:
    status: str               # ok, exit_137, signal, failed, timeout, stopped
    returncode: int | None
    message: str
    samples: list = field(default_factory=list)
    stop_reason: str | None = None
    trace: list = field(default_factory=list)  # per-sample pressure, swap-in and swap-out


def run_child(command: list[str], env: dict, *, timeout: float, sampler: Sampler,
              watchdog: Watchdog, vm_start: dict | None = None, arm_start: dict | None = None,
              interval: float = 5.0,
              popen=subprocess.Popen, sleep=time.sleep, clock=time.monotonic,
              grace: float = 10.0) -> ChildResult:
    """Run one child; sample while it runs; kill it on a stop rule or timeout."""
    proc = popen(command, env=env)
    began = clock()
    samples: list = []
    trace: list = []

    def stop(reason: str, status: str) -> ChildResult:
        proc.terminate()
        deadline = clock() + grace
        while proc.poll() is None and clock() < deadline:
            sleep(0.2)
        if proc.poll() is None:
            proc.kill()
        return ChildResult(status, proc.poll(), reason, samples, reason, trace)

    while True:
        code = proc.poll()
        if code is not None:
            status, message = classify_exit(code)
            return ChildResult(status, code, message, samples, trace=trace)
        if clock() - began > timeout:
            return stop(f"exceeded {timeout:.0f} s", "timeout")
        reading = sampler.sample()
        samples.append(reading)
        reason = watchdog.check(reading, vm_delta(vm_start, reading.get("vm")), arm_start)
        if watchdog.last:
            trace.append(dict(watchdog.last))
        if reason:
            if trace:
                trace[-1]["stop"] = reason
            return stop(reason, "stopped")
        sleep(interval)


def window_rate(stamps: list[float], start: int, stop: int) -> float | None:
    """Tokens per second between decode tokens ``start`` and ``stop`` (1-based, inclusive)."""
    if len(stamps) < stop or stop <= start:
        return None
    span = stamps[stop - 1] - stamps[start - 1]
    return (stop - start) / span if span > 0 else None


# -- analysis -----------------------------------------------------------------

def measured_turn(arm: dict) -> dict:
    """The turn a result measures: the last one (turn 1 for first-turn arms)."""
    return arm["turns"][-1]


def group_of(arm: dict) -> tuple[str, str]:
    return arm["size"], arm["turn"]


def exactness(arms: list[dict]) -> dict:
    """Per (size, turn) group: pass when first token, cache hash and 128-token
    digest all equal the baseline's. A missing cache hash is noted and the
    digest still gates; a missing digest or first token is a mismatch."""
    groups: dict[tuple, list[dict]] = {}
    for arm in arms:
        groups.setdefault(group_of(arm), []).append(arm)
    report = {}
    for key, members in groups.items():
        base = next((a for a in members if a["condition"] == BASELINES[key[1]]), None)
        if base is None:
            report[key] = {"status": "no_baseline", "mismatches": [], "notes": []}
            continue
        ref = measured_turn(base)
        mismatches, notes = [], []
        for arm in members:
            turn = measured_turn(arm)
            for name in ("first_token", "kv_hash", "digest"):
                want, got = ref.get(name), turn.get(name)
                if name == "kv_hash" and "unavailable" in (want, got):
                    notes.append(f"{arm['arm_id']}: kv_hash unavailable")
                    continue
                if want is None or got is None or want != got:
                    mismatches.append((arm["arm_id"], name, want, got))
        report[key] = {
            "status": "pass" if not mismatches else "fail",
            "mismatches": mismatches, "notes": notes,
        }
    return report


def ring_health(arm: dict, cond: Condition) -> dict:
    """G2: in a ring condition the ring ran every layer with no fallback or error."""
    if not cond.ring_on:
        return {"status": "n/a", "reasons": []}
    turn = measured_turn(arm)
    ring = turn.get("ring") or {}
    reasons = []
    if not turn.get("ring_acquired"):
        reasons.append(f"ring not acquired ({turn.get('ring_reason')})")
    for name, value in ring.items():
        if name.startswith("fallback") and value:
            reasons.append(f"{name}={value}")
    if ring.get("last_error"):
        reasons.append(f"error: {ring['last_error']}")
    if ring.get("disabled"):
        reasons.append("ring disabled mid-prefill")
    return {"status": "pass" if not reasons else "invalid", "reasons": reasons}


def memory_gate(arm: dict, limit: int = METAL_LIMIT_BYTES) -> dict:
    """G3: MLX peak plus the largest cache reading leaves HEADROOM_BYTES."""
    turn = measured_turn(arm)
    limit = arm.get("metal_limit_bytes") or limit
    peak = max(turn["mem_end_prefill"]["peak"], turn["mem_end_decode"]["peak"])
    cache = max(turn["mem_end_prefill"]["cache"], turn["mem_end_decode"]["cache"],
                turn.get("cache_max", 0))
    headroom = limit - (peak + cache)
    return {
        "status": "pass" if headroom >= HEADROOM_BYTES else "fail",
        "peak": peak, "cache": cache, "headroom": headroom,
    }


def scratch_cleanup(arm: dict, cond: Condition, baseline: dict | None) -> dict:
    """G4: after a scratch prefill no pool exists, nothing is retained, and
    active memory is near the baseline's."""
    if not cond.scratch:
        return {"status": "n/a", "reasons": []}
    turn = measured_turn(arm)
    after = (turn.get("prefill_calls") or [{}])[-1].get("after", {})
    reasons = []
    if after.get("pool_exists"):
        reasons.append("a pool existed at the end of prefill")
    if after.get("retained"):
        reasons.append(f"{after['retained']} scratch pool(s) retained")
    if baseline is not None:
        base_active = measured_turn(baseline)["mem_end_prefill"]["active"]
        diff = turn["mem_end_prefill"]["active"] - base_active
        if diff > CLEANUP_TOLERANCE_BYTES:
            reasons.append(f"active memory {diff / 1e6:.0f} MB above baseline")
    return {"status": "pass" if not reasons else "fail", "reasons": reasons}


def resolution_band(values: list[float]) -> float:
    """Half the range of the baseline arms: the smallest effect distinguishable
    from the baseline's own run-to-run spread."""
    return (max(values) - min(values)) / 2 if len(values) > 1 else float("inf")


def paired_effect(baseline: dict[int, float], candidate: dict[int, float],
                  reverse_rounds: set[int]) -> dict:
    """Candidate minus baseline for each round both ran, with its resolution band.

    Resolved only with at least MIN_ROUNDS_FOR_EFFECT pairs, a median beyond the
    band, one sign in every pair, and the same sign in forward and reversed
    rounds. Anything else is reported as unresolved with the reason."""
    rounds = sorted(set(baseline) & set(candidate))
    diffs = {r: candidate[r] - baseline[r] for r in rounds}
    band = resolution_band([baseline[r] for r in rounds]) if rounds else float("inf")
    out = {"pairs": len(rounds), "differences": diffs, "band": band, "median": None,
           "verdict": "unresolved", "reason": ""}
    if len(rounds) < MIN_ROUNDS_FOR_EFFECT:
        out["reason"] = f"{len(rounds)} paired rounds, need {MIN_ROUNDS_FOR_EFFECT}"
        return out
    median = statistics.median(diffs.values())
    out["median"] = median
    signs = {d > 0 for d in diffs.values() if d != 0}
    forward = {d > 0 for r, d in diffs.items() if r not in reverse_rounds and d != 0}
    backward = {d > 0 for r, d in diffs.items() if r in reverse_rounds and d != 0}
    if abs(median) <= band:
        out["reason"] = "median effect is inside the resolution band"
    elif len(signs) > 1:
        out["reason"] = "the sign differs between rounds"
    elif forward and backward and forward != backward:
        out["reason"] = "the sign depends on run order"
    else:
        out["verdict"] = "resolved"
    return out


def metric_value(arm: dict, name: str) -> float | None:
    turn = measured_turn(arm)
    getters = {
        "prefill_tok_s": lambda: turn.get("prefill_tok_s"),
        "decode_tok_s_1_64": lambda: turn.get("decode", {}).get("tok_s_1_64"),
        "decode_tok_s_65_128": lambda: turn.get("decode", {}).get("tok_s_65_128"),
        "peak_gb": lambda: turn["mem_end_prefill"]["peak"] / 1e9,
        "ssd_prefill_gb": lambda: (turn.get("ssd_prefill_bytes") or 0) / 1e9,
    }
    return getters[name]()


def effects_by_group(arms: list[dict], metrics: tuple[str, ...]) -> dict:
    """Paired effects of every condition against its baseline, per size and turn."""
    out: dict = {}
    for key in sorted({group_of(a) for a in arms}):
        members = [a for a in arms if group_of(a) == key]
        base_id = BASELINES[key[1]]
        base_arms = {a["round"]: a for a in members if a["condition"] == base_id}
        reverse = {a["round"] for a in members if a.get("reverse")}
        for name in sorted({a["condition"] for a in members} - {base_id}):
            cand = {a["round"]: a for a in members if a["condition"] == name}
            for metric in metrics:
                def series(source, metric=metric):
                    values = {r: metric_value(a, metric) for r, a in source.items()}
                    return {r: v for r, v in values.items() if v is not None}

                out[(key[0], key[1], name, metric)] = paired_effect(
                    series(base_arms), series(cand), reverse
                )
    return out


def decode_regression(arms: list[dict]) -> dict:
    """G5: decode tok/s not slower than the baseline beyond the resolution band,
    and swap and page-out deltas not larger. One verdict per condition."""
    verdicts: dict = {}
    effects = effects_by_group(arms, ("decode_tok_s_1_64", "decode_tok_s_65_128"))
    for (size, turn, name, metric), effect in effects.items():
        entry = verdicts.setdefault((size, turn, name), {"status": "pass", "reasons": []})
        if effect["verdict"] == "resolved" and effect["median"] < 0:
            entry["status"] = "fail"
            entry["reasons"].append(f"{metric} slower by {-effect['median']:.2f}")
    for arm in arms:
        if arm["condition"] == BASELINES[arm["turn"]]:
            continue
        base = [a for a in arms if group_of(a) == group_of(arm)
                and a["condition"] == BASELINES[arm["turn"]] and a["round"] == arm["round"]]
        if not base:
            continue
        mine, ref = measured_turn(arm).get("vm", {}), measured_turn(base[0]).get("vm", {})
        for name in ("swapin", "swapout", "pageout"):
            if mine.get(name, 0) > ref.get(name, 0):
                entry = verdicts.setdefault(
                    (arm["size"], arm["turn"], arm["condition"]),
                    {"status": "pass", "reasons": []},
                )
                entry["status"] = "fail"
                entry["reasons"].append(f"{name} {mine[name]} above baseline {ref.get(name, 0)}")
    return verdicts


def worst_headroom(arms: list[dict], size: str) -> int | None:
    """Smallest G3 headroom among the arms of one size, or None without arms."""
    values = [memory_gate(a)["headroom"] for a in arms if a["size"] == size]
    return min(values) if values else None


def trace_summary(trace: list) -> dict:
    """What the watchdog saw during one arm: the worst swap-in, pressure and swap-out,
    plus the per-sample trace."""
    levels = [x["pressure_level"] for x in trace if x.get("pressure_level") is not None]
    return {"samples": len(trace),
            "max_swapin_mib": max((x["swapin_mib"] for x in trace), default=0.0),
            "max_swapout_pages": max((x["swapout_pages"] for x in trace), default=0),
            "max_pressure_level": max(levels, default=None),
            "min_available_mb": min((x["available_mb"] for x in trace
                                     if x.get("available_mb") is not None), default=None),
            "max_compressor_growth_mb": max((x["compressor_growth_mb"] for x in trace
                                             if x.get("compressor_growth_mb") is not None), default=None),
            "compressor_warnings": sum(1 for x in trace if x.get("compressor_warning")),
            "trace": trace}


# -- child-side memory tracing --------------------------------------------------

def parse_vm_stat(text: str) -> dict:
    """``vm_stat`` in one pass: page size, memory that could be reused, the compressor
    and the cumulative counters. Sizes are in MB, counters in pages."""
    match = re.search(r"page size of (\d+) bytes", text)
    page = int(match.group(1)) if match else 16384
    pages, counters = {}, {}
    names = {"Pages free": "free", "Pages inactive": "inactive", "Pages speculative": "speculative",
             "Pages purgeable": "purgeable", "Pages occupied by compressor": "compressor",
             "Pages wired down": "wired", "Pages active": "active",
             "File-backed pages": "file_backed", "Anonymous pages": "anonymous"}
    wanted = {"Swapins": "swapin", "Swapouts": "swapout", "Pageins": "pagein", "Pageouts": "pageout",
              "Compressions": "compress", "Decompressions": "decompress"}
    for line in text.splitlines():
        label, _, value = line.partition(":")
        label, value = label.strip().strip('"'), value.strip().rstrip(".")
        if not value.isdigit():
            continue
        if label in names:
            pages[names[label]] = int(value)
        elif label in wanted:
            counters[wanted[label]] = int(value)
    mb = lambda key: pages.get(key, 0) * page / 1048576  # noqa: E731
    available = mb("free") + mb("inactive") + mb("speculative")
    return {"page_bytes": page, "available_mb": round(available, 1),
            "free_mb": round(mb("free"), 1), "compressor_mb": round(mb("compressor"), 1),
            "wired_mb": round(mb("wired"), 1), "active_mb": round(mb("active"), 1),
            "file_backed_mb": round(mb("file_backed"), 1), "anonymous_mb": round(mb("anonymous"), 1),
            "counters": counters}


def guard_reason(reading: dict, *, min_available_gb: float = MIN_AVAILABLE_GB,
                 headroom_bytes: int = GUARD_HEADROOM_BYTES,
                 compressor_limit_mib: float = MAX_COMPRESSOR_GROWTH_GIB * 1024,
                 compressor_stop_available_mib: float = COMPRESSOR_STOP_AVAILABLE_GIB * 1024) -> str | None:
    """The child's own stop rule, checked at every sample. Three things stop it before
    macOS reaches a pressure level: little memory left to reuse; compressor growth over its
    limit while available memory is below the compressor floor (growth alone only warns);
    and MLX's own allocations (active plus cache) closing in on the Metal working-set limit."""
    available = reading.get("available_mb")
    if available is not None and available < min_available_gb * 1000:
        return (f"available memory {available / 1000:.2f} GB is below "
                f"{min_available_gb:.1f} GB")
    growth = reading.get("compressor_growth_mb")
    if (growth is not None and growth > compressor_limit_mib and available is not None
            and available < compressor_stop_available_mib):
        return (f"compressor grew {growth / 1024:.2f} GiB since the arm began "
                f"(limit {compressor_limit_mib / 1024:.0f} GiB) while only "
                f"{available / 1024:.2f} GiB was available "
                f"(floor {compressor_stop_available_mib / 1024:.0f} GiB)")
    limit = reading.get("metal_limit_bytes")
    if limit:
        held = reading.get("mlx_active", 0) + reading.get("mlx_cache", 0)
        if held > limit - headroom_bytes:
            return (f"MLX holds {held / 1e9:.2f} GB (active + cache), within "
                    f"{headroom_bytes / 1e9:.1f} GB of the {limit / 1e9:.2f} GB Metal limit")
    return None


def summarize_trace(rows: list) -> dict:
    """Per-phase extremes from a child's 1 s trace (rows without a ``phase`` are events)."""
    phases: dict = {}
    for row in rows:
        phase = row.get("phase")
        if phase is None:
            continue
        item = phases.setdefault(phase, {"samples": 0, "first_t": row["t"], "last_t": row["t"]})
        item["samples"] += 1
        item["last_t"] = row["t"]
        for key, pick in (("mlx_active", max), ("mlx_cache", max), ("mlx_peak", max), ("rss", max),
                          ("compressor_mb", max), ("compressor_growth_mb", max), ("file_backed_mb", max),
                          ("swap_used_mb", max),
                          ("pressure_level", max), ("available_mb", min)):
            value = row.get(key)
            if value is not None:
                item[key] = pick(item[key], value) if key in item else value
    return {"phases": phases,
            "events": [row for row in rows if "event" in row],
            "guard": next((row["guard"] for row in rows if "guard" in row), None),
            "last_sample": next((row for row in reversed(rows) if row.get("phase") is not None), None),
            "samples": sum(1 for row in rows if row.get("phase") is not None)}
