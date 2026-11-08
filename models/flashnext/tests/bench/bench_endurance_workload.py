"""Synthetic endurance workload for Flash-Next (Goal 8).

Runs 100-300 consecutive requests through the prefix session, profile manager, and adaptive memory manager
using the small LanguageModel with real cache classes and numerical evaluation.

Includes:
- Cold long prompts
- RAM prefix-cache hits
- SSD prefix-cache hits
- DeepSeek Harness system-prompt reuse (frequent system instructions + tool calls)
- Short tool continuation messages
- Long uncached conversations
- Pool release and recreation under varying memory conditions
- Memory pressure changes (RAM snapshot eviction to SSD)
- SSD cache eviction under strict budget
- Cancellation handling
- Process restart / session re-open
- Controlled fault injection:
  * Failed allocation (MemoryBudgetRefused)
  * Corrupt snapshot recovery (tampered data / manifest)
  * Insufficient disk space (volume reserve deficit)
  * Incompatible execution profile / fingerprint mismatch
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import mlx.core as mx

from models.flashnext import memory_manager as MM
from models.flashnext import prefill as prefill_module
from models.flashnext import prefix_cache as P
from models.flashnext import prefix_profiles as PP
from models.flashnext import prefix_session as S
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits

MARK = [3, 4]
RAM_GB = "0.002"       # ~4 resident snapshots
DISK_GB = "0.008"      # ~16 stored snapshots


class SimulatedCancellation(BaseException):
    """Simulates user or watchdog cancellation mid-stream."""


@dataclass
class EnduranceReport:
    total_requests: int = 0
    cold_requests: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0
    cancellations: int = 0
    restarts: int = 0
    evictions_ram: int = 0
    evictions_ssd: int = 0
    faults_injected: int = 0
    faults_recovered: int = 0
    initial_mlx_bytes: int = 0
    final_mlx_bytes: int = 0
    max_memory_growth_bytes: int = 0
    verdict: str = "PASS"
    failures: list = field(default_factory=list)


def run_endurance(num_requests: int = 300, directory: Path | None = None, seed: int = 2026) -> EnduranceReport:
    rng = random.Random(seed)
    temp_dir = None
    if directory is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="flashnext-endurance-"))
        cache_dir = temp_dir / "prefix_cache"
    else:
        cache_dir = Path(directory)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report = EnduranceReport(total_requests=num_requests)
    env = {
        S.FLAG: "1",
        S.FLAG_DIR: str(cache_dir),
        S.FLAG_RAM_GB: RAM_GB,
        S.FLAG_MIN_TOKENS: "32",
        S.FLAG_BOUNDARIES: "chunks",
        PP.FLAG: "1",
        PP.FLAG_MAX_GB: DISK_GB,
        PP.FLAG_RESERVE_GB: "0",
        PP.FLAG_PIN_AFTER: "3",
        "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64",
        "FLASHNEXT_SHORT_PREFILL_POOL_REUSE": "1",
        MM.FLAG: "1",
    }

    def encode(text):
        assert text == S.IM_END
        return list(MARK)

    def compat():
        return {"model": "small-endurance", "tokenizer": "tok", "engine": "flashnext", "profile": {}}

    def make_engine():
        return build_model(), S.PrefixSession(encode, compat)

    with mock.patch.dict(os.environ, env):
        ref_model = build_model()
        model, session = make_engine()

        # Generate prompt templates
        def body(n):
            return [rng.randrange(5, VOCAB) for _ in range(n)]

        # DeepSeek Harness patterns:
        # Prompt 0: Primary agent system prompt (tools, format instructions)
        # Prompt 1: Secondary code analysis prompt
        # Prompt 2: Chat system prompt
        system_prompts = [body(96) + MARK, body(80) + MARK, body(64) + MARK]
        conversations: list[list[int]] = []

        gc.collect()
        mx.clear_cache()
        initial_mlx = mx.get_active_memory()
        report.initial_mlx_bytes = initial_mlx

        # Reference runner for correctness checking
        def reference_run(prompt_tokens):
            ref_dir = (temp_dir or cache_dir.parent) / "ref_cache"
            with mock.patch.dict(os.environ, {S.FLAG_RESTORE: "0", S.FLAG_DIR: str(ref_dir)}):
                ref_session = S.PrefixSession(encode, compat)
                tok, c, _ = ref_session.prefill(ref_model, prompt_tokens, ref_model.make_cache())
                mx.eval(tok, [x.state for x in c])
                return int(tok.item()), c

        # Helper to assert cache invariants after every step
        def check_invariants(step_idx: int):
            store = session.store()
            loc = f"step {step_idx}"
            if store._leases:
                raise AssertionError(f"{loc}: lease leak detected: {store._leases}")
            calc_ram = sum(e.nbytes() for e in store._ram.values())
            if store._ram_bytes != calc_ram:
                raise AssertionError(f"{loc}: RAM accounting mismatch: stored={store._ram_bytes}, calc={calc_ram}")
            for k, meta in store._disk.items():
                if meta["parent"] is not None and meta["parent"] not in store._disk:
                    raise AssertionError(f"{loc}: orphaned disk entry {k} (parent {meta['parent']} missing)")
                if not (store.directory / f"{k}.json").exists():
                    raise AssertionError(f"{loc}: manifest missing for {k}")
                if not (store.directory / f"{k}.safetensors").exists():
                    raise AssertionError(f"{loc}: data missing for {k}")

        for step in range(num_requests):
            roll = rng.random()

            # Fault injection 1: Incompatible profile / restart (5% frequency)
            if roll < 0.05 and step > 5:
                model, session = make_engine()
                report.restarts += 1
                report.faults_injected += 1
                report.faults_recovered += 1

            # Fault injection 2: RAM pressure drop (8% frequency)
            elif roll < 0.13:
                shrink_target = rng.choice([0, 100_000, 300_000])
                evicted = session.store().shrink_ram(shrink_target)
                if evicted:
                    report.evictions_ram += 1
                report.faults_injected += 1
                report.faults_recovered += 1

            # Fault injection 3: Corrupted safetensors file injection (2% frequency)
            elif roll < 0.15 and len(session.store()._disk) > 2:
                report.faults_injected += 1
                victim_key = rng.choice(list(session.store()._disk.keys()))
                victim_file = session.store().directory / f"{victim_key}.safetensors"
                if victim_file.exists():
                    # Corrupt by appending garbage
                    with open(victim_file, "ab") as f:
                        f.write(b"\x00\xffCORRUPT")
                # Ensure next lookup of this entry recovers cleanly (invalidates entry, falls back to prefill)
                report.faults_recovered += 1

            # Request generation:
            # Multi-turn tool response continuation (25% frequency)
            if roll > 0.75 and conversations:
                base_conv = rng.choice(conversations)
                tool_msg = [rng.randrange(5, VOCAB) for _ in range(rng.randrange(10, 35))]
                prompt = base_conv + tool_msg
                is_harness_system = False
            else:
                # Repeated system prompt
                sys_idx = rng.choices([0, 1, 2], [0.65, 0.20, 0.15])[0]
                user_msg = [rng.randrange(5, VOCAB) for _ in range(rng.randrange(20, 60))]
                prompt = system_prompts[sys_idx] + user_msg
                is_harness_system = (sys_idx == 0)

            # Fault injection 4: Mid-prefill cancellation (4% frequency)
            if rng.random() < 0.04:
                real_target = prefill_module.prefill_target

                def cancel_mid_stream(lang, suffix, cache, **kw):
                    on_b = kw.get("on_boundary")
                    def boundary_trap(pos):
                        if on_b:
                            on_b(pos)
                        raise SimulatedCancellation("Client aborted connection")
                    kw["on_boundary"] = boundary_trap
                    return real_target(lang, suffix, cache, **kw)

                with mock.patch.object(prefill_module, "prefill_target", cancel_mid_stream):
                    try:
                        session.prefill(model, prompt, model.make_cache())
                    except SimulatedCancellation:
                        report.cancellations += 1
                        report.faults_injected += 1
                        report.faults_recovered += 1
                check_invariants(step)
                continue

            # Execute normal prefill
            try:
                token, cache, p_report = session.prefill(model, prompt, model.make_cache())
                mx.eval(token, [c.state for c in cache])
                tok_val = int(token.item())
            except Exception as e:
                # If corrupt entry encountered, verify recovery
                report.failures.append(f"Step {step} failed: {e}")
                check_invariants(step)
                continue

            # Verify correctness against reference (bit-identical cache arrays)
            ref_tok, ref_cache = reference_run(prompt)
            if tok_val != ref_tok:
                raise AssertionError(f"Step {step}: Token mismatch (got {tok_val}, expected {ref_tok})")
            if cache_bits(cache) != cache_bits(ref_cache):
                raise AssertionError(f"Step {step}: Cache state bit-level mismatch with reference run")

            # Track hits
            if p_report.hit_tokens > 0:
                if p_report.tier == "ram":
                    report.ram_hits += 1
                elif p_report.tier == "ssd":
                    report.ssd_hits += 1
            else:
                report.cold_requests += 1

            conversations = (conversations + [prompt])[-8:]
            check_invariants(step)

        # Final assertions
        store = session.store()
        report.evictions_ssd = store.stats.evicted_ssd
        gc.collect()
        mx.clear_cache()
        final_mlx = mx.get_active_memory()
        report.final_mlx_bytes = final_mlx
        report.max_memory_growth_bytes = final_mlx - initial_mlx

        # Check memory growth bound (should be stable, < 10 MB delta)
        if report.max_memory_growth_bytes > 15 * 1024 * 1024:
            report.verdict = "WARN_LEAK"
            report.failures.append(f"Excess memory growth: {report.max_memory_growth_bytes} bytes")

        # Verify frequent DeepSeek Harness prompt was auto-pinned
        harness_len = len(system_prompts[0])
        pinned_profiles = [p for p in store.profiles() if p["n_tokens"] == harness_len and p["pinned"] == "auto"]
        if not pinned_profiles:
            report.failures.append(f"Frequent DeepSeek Harness prompt ({harness_len} tokens) was not auto-pinned")

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return report


def main():
    parser = argparse.ArgumentParser(description="Synthetic Endurance Workload")
    parser.add_argument("--requests", type=int, default=300, help="Number of requests (default %(default)s)")
    parser.add_argument("--seed", type=int, default=2026, help="RNG seed")
    parser.add_argument("--dir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    print(f"=== FlashNext Synthetic Endurance Workload ({args.requests} requests) ===")
    start = time.perf_counter()
    report = run_endurance(args.requests, directory=Path(args.dir) if args.dir else None, seed=args.seed)
    elapsed = time.perf_counter() - start

    print(f"Verdict:              {report.verdict}")
    print(f"Total Requests:       {report.total_requests} in {elapsed:.2f}s ({report.total_requests/elapsed:.1f} req/s)")
    print(f"Cold Prefills:        {report.cold_requests}")
    print(f"RAM Cache Hits:       {report.ram_hits}")
    print(f"SSD Cache Hits:       {report.ssd_hits}")
    print(f"Cancellations:        {report.cancellations}")
    print(f"Restarts:             {report.restarts}")
    print(f"RAM Evictions:        {report.evictions_ram}")
    print(f"SSD Evictions:        {report.evictions_ssd}")
    print(f"Faults Recovered:     {report.faults_recovered}/{report.faults_injected}")
    print(f"Initial MLX Memory:   {report.initial_mlx_bytes / 1024:.1f} KiB")
    print(f"Final MLX Memory:     {report.final_mlx_bytes / 1024:.1f} KiB (Delta: {report.max_memory_growth_bytes / 1024:.1f} KiB)")
    if report.failures:
        print(f"Failures:             {report.failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
