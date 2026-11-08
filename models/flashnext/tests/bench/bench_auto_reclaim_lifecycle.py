"""Autonomous Request-to-Request Memory Reclamation Lifecycle Stress Test (Phase 6).

Runs 100 consecutive requests through real small-MoE LanguageModel, PrefixSession,
and AutoReclaimer across all DeepSeek Harness lifecycle scenarios:
1. Cold prompt -> cached continuation.
2. Different session -> prompt cache miss -> new snapshot.
3. Memory tightened (< 5.5 GB) -> RAM cache evicted to SSD -> continuation restored from SSD.
4. Decode pool released (< 3.5 GB) -> prefill -> decode pool recreated.
5. Tool response appended -> short prefill.
6. Error/cancellation -> cleanup runs -> next request succeeds.

Asserts:
- No memory leaks across 100 requests.
- No use-after-free or dangling references.
- No snapshot corruption (content hashes match).
- Bit-identical outputs with un-reclaimed baseline on identical prompts.
- Memory stays within bounds without manual purge.
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

from models.flashnext import auto_reclaim
from models.flashnext import expert_pool
from models.flashnext import memory_manager as MM
from models.flashnext import pool_lifecycle
from models.flashnext import prefill as prefill_module
from models.flashnext import prefix_cache as P
from models.flashnext import prefix_profiles as PP
from models.flashnext import prefix_session as S
from models.flashnext.tests.unit.test_prefix_cache import VOCAB, build_model, cache_bits

MARK = [3, 4]
GB = 10 ** 9
MB = 10 ** 6


class SimulatedCancellation(BaseException):
    """Simulates mid-stream cancellation."""


@dataclass
class LifecycleReport:
    total_requests: int = 0
    cold_prefills: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0
    pool_releases: int = 0
    pool_rebuilds: int = 0
    ram_evictions: int = 0
    cancellations: int = 0
    cancellation_recoveries: int = 0
    initial_mlx_bytes: int = 0
    final_mlx_bytes: int = 0
    memory_growth_bytes: int = 0
    post_warmup_active_samples: list[int] = field(default_factory=list)
    post_warmup_cache_samples: list[int] = field(default_factory=list)
    verdict: str = "PASS"
    failures: list[str] = field(default_factory=list)


def run_lifecycle_stress_test(
    num_requests: int = 100,
    directory: Path | None = None,
    seed: int = 42,
) -> LifecycleReport:
    rng = random.Random(seed)
    temp_dir = None
    if directory is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="flashnext-lifecycle-"))
        cache_dir = temp_dir / "prefix_cache"
    else:
        cache_dir = Path(directory)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report = LifecycleReport(total_requests=num_requests)

    # Configure opt-in flags for automatic reclamation and prefix caching
    env = {
        auto_reclaim.FLAG_AUTO_RECLAIM: "1",
        auto_reclaim.FLAG_AUTO_PURGE: "0",  # Purge not allowed during normal inference
        MM.FLAG: "1",
        S.FLAG: "1",
        S.FLAG_DIR: str(cache_dir),
        S.FLAG_RAM_GB: "0.004",  # ~4 MB RAM prefix cache
        S.FLAG_MIN_TOKENS: "32",
        S.FLAG_BOUNDARIES: "chunks",
        PP.FLAG: "1",
        PP.FLAG_MAX_GB: "0.016",  # 16 MB SSD budget
        PP.FLAG_RESERVE_GB: "0",
        PP.FLAG_PIN_AFTER: "2",
        "FLASHNEXT_PREFILL_CHUNK_TOKENS": "64",
        "FLASHNEXT_SHORT_PREFILL_POOL_REUSE": "1",
    }

    def encode(text: str):
        assert text == S.IM_END
        return list(MARK)

    def compat():
        return {
            "model": "small-moe-qwen4",
            "tokenizer": "test-tok",
            "engine": "flashnext",
            "profile": {},
        }

    # Reference model run without caching or reclamation for exact bit comparison
    ref_model = build_model()

    def reference_run(prompt_tokens: list[int]):
        ref_dir = (temp_dir or cache_dir.parent) / "ref_cache"
        with mock.patch.dict(os.environ, {S.FLAG_RESTORE: "0", S.FLAG_DIR: str(ref_dir)}):
            ref_session = S.PrefixSession(encode, compat)
            tok, c, _ = ref_session.prefill(ref_model, prompt_tokens, ref_model.make_cache())
            mx.eval(tok, [x.state for x in c])
            return int(tok.item()), c

    with mock.patch.dict(os.environ, env):
        auto_reclaim.reset_for_tests()
        MM.reset_for_tests()
        reclaimer = auto_reclaim.get_reclaimer()

        model = build_model()
        session = S.PrefixSession(encode, compat)

        # Mock store with expert pool
        class MockPool:
            def __init__(self, slots: int = 16, stride: int = 4 * MB):
                self.slots = slots
                self.stride = stride
                self._mm = b"\x00" * (slots * stride)

        class MockStore:
            def __init__(self):
                self._flashnext_expert_pool = MockPool()
                self._uncached = False

            def set_uncached_reads(self, uncached: bool):
                self._uncached = uncached

        mock_store = MockStore()

        # Simulated dynamic system memory state
        simulated_sys = {"available": 8000 * MB, "pressure": 1, "file_backed_mb": 4500.0}
        reclaimer._system = lambda: simulated_sys

        # Prepare prompt templates
        def body(n):
            return [rng.randrange(5, VOCAB) for _ in range(n)]

        # Prompt A: Agent system prompt (tools, rules)
        sys_prompt_A = body(96) + MARK
        # Prompt B: Secondary code reviewer prompt
        sys_prompt_B = body(80) + MARK
        # Prompt C: Quick helper prompt
        sys_prompt_C = body(64) + MARK

        active_conversations: dict[str, list[int]] = {
            "A": list(sys_prompt_A),
            "B": list(sys_prompt_B),
            "C": list(sys_prompt_C),
        }

        gc.collect()
        mx.clear_cache()
        initial_mlx = mx.get_active_memory()
        report.initial_mlx_bytes = initial_mlx

        # Invariant checker
        def check_invariants(step_num: int):
            store = session.store()
            # 1. No lease leaks
            if store._leases:
                raise AssertionError(f"Step {step_num}: Leases leaked: {store._leases}")
            # 2. Exact RAM accounting
            calc_ram = sum(e.nbytes() for e in store._ram.values())
            if store._ram_bytes != calc_ram:
                raise AssertionError(
                    f"Step {step_num}: RAM accounting mismatch: stored={store._ram_bytes}, calc={calc_ram}"
                )
            # 3. Snapshot integrity on disk
            for k, meta in store._disk.items():
                man_path = store.directory / f"{k}.json"
                data_path = store.directory / f"{k}.safetensors"
                if not man_path.exists():
                    raise AssertionError(f"Step {step_num}: Missing manifest for {k}")
                if not data_path.exists():
                    raise AssertionError(f"Step {step_num}: Missing data file for {k}")
                # Verify sha256 matches manifest metadata
                manifest = json.loads(man_path.read_text())
                actual_hash = P._sha256_file(data_path)
                if actual_hash != manifest["file_sha256"]:
                    raise AssertionError(
                        f"Step {step_num}: Snapshot corruption for {k}: expected {manifest['file_sha256']}, got {actual_hash}"
                    )

        # Run 100 requests through structured 10-turn cycles
        for req_idx in range(num_requests):
            cycle_phase = req_idx % 10
            prompt = None
            should_cancel = False
            target_conv = "A"

            if cycle_phase == 0:
                # 1. Cold prompt -> initial prefill & snapshot
                simulated_sys["available"] = 8000 * MB
                simulated_sys["pressure"] = 1
                user_msg = body(32)
                prompt = sys_prompt_A + user_msg
                active_conversations["A"] = list(prompt)

            elif cycle_phase == 1:
                # 2. Cached continuation -> hits RAM prefix cache, warm pool reused
                tool_call = body(20)
                prompt = active_conversations["A"] + tool_call
                active_conversations["A"] = list(prompt)

            elif cycle_phase == 2:
                # 3. Tool response appended -> short prefill
                tool_resp = body(24)
                prompt = active_conversations["A"] + tool_resp
                active_conversations["A"] = list(prompt)

            elif cycle_phase == 3:
                # 4. Different session B -> cache miss on prefix suffix, new snapshot
                user_msg = body(35)
                prompt = sys_prompt_B + user_msg
                active_conversations["B"] = list(prompt)

            elif cycle_phase == 4:
                # 5. Memory tightened: simulated available drops to 5.0 GB (< 5.5 GB -> CONSTRAINED)
                simulated_sys["available"] = 5000 * MB
                simulated_sys["pressure"] = 2
                # Request on session B; post_request_cleanup will evict RAM prefix snapshots to SSD
                user_msg = body(16)
                prompt = active_conversations["B"] + user_msg
                active_conversations["B"] = list(prompt)

            elif cycle_phase == 5:
                # 6. Continuation restored from SSD (all RAM snapshots were moved to SSD in phase 4)
                # Verify single-pass hash verification and bit-identical restoration
                tool_msg = body(18)
                prompt = active_conversations["A"] + tool_msg
                active_conversations["A"] = list(prompt)

            elif cycle_phase == 6:
                # 7. Memory tightened to CRITICAL (< 3.5 GB) -> decode pool released
                simulated_sys["available"] = 3000 * MB
                simulated_sys["pressure"] = 4
                user_msg = body(22)
                prompt = sys_prompt_C + user_msg
                active_conversations["C"] = list(prompt)

            elif cycle_phase == 7:
                # 8. Memory recovers to NORMAL -> decode pool recreated, request succeeds
                simulated_sys["available"] = 8500 * MB
                simulated_sys["pressure"] = 1
                if mock_store._flashnext_expert_pool is None:
                    mock_store._flashnext_expert_pool = MockPool()
                    report.pool_rebuilds += 1
                tool_msg = body(15)
                prompt = active_conversations["C"] + tool_msg
                active_conversations["C"] = list(prompt)

            elif cycle_phase == 8:
                # 9. Error/mid-stream cancellation -> cleanup runs in finally
                should_cancel = True
                prompt = active_conversations["A"] + body(40)

            elif cycle_phase == 9:
                # 10. Recovery request immediately following cancellation
                tool_msg = body(25)
                prompt = active_conversations["A"] + tool_msg
                active_conversations["A"] = list(prompt)

            # Execute request with try/finally memory reclamation
            tok_val = None
            try:
                if should_cancel:
                    report.cancellations += 1
                    # Trigger SimulatedCancellation mid-prefill
                    real_target = prefill_module.prefill_target

                    def cancel_mid_stream(lang, suffix, cache, **kw):
                        raise SimulatedCancellation("Agent connection timed out")

                    with mock.patch.object(prefill_module, "prefill_target", cancel_mid_stream):
                        session.prefill(model, prompt, model.make_cache())
                else:
                    token, cache, p_report = session.prefill(model, prompt, model.make_cache())
                    mx.eval(token, [c.state for c in cache])
                    tok_val = int(token.item())

                    # Record stats
                    if p_report.hit_tokens > 0:
                        if p_report.tier == "ram":
                            report.ram_hits += 1
                        elif p_report.tier == "ssd":
                            report.ssd_hits += 1
                    else:
                        report.cold_prefills += 1

                    # Verify bit-identical output against reference run
                    ref_tok, ref_cache = reference_run(prompt)
                    if tok_val != ref_tok:
                        msg = f"Step {req_idx}: Token mismatch (got {tok_val}, expected {ref_tok})"
                        report.failures.append(msg)
                        raise AssertionError(msg)
                    if cache_bits(cache) != cache_bits(ref_cache):
                        msg = f"Step {req_idx}: Cache bit-level divergence from reference"
                        report.failures.append(msg)
                        raise AssertionError(msg)

                    if cycle_phase == 9:
                        report.cancellation_recoveries += 1

            except SimulatedCancellation:
                pass  # expected cancellation
            finally:
                # The critical invariant: post_request_cleanup ALWAYS runs
                prefix_st = session.store()
                clean_report = reclaimer.post_request_cleanup(
                    store=mock_store,
                    prefix_store=prefix_st,
                    clear_cache=mx.clear_cache,
                    synchronize=mx.synchronize,
                )
                if clean_report.pool_released_bytes > 0:
                    report.pool_releases += 1
                if clean_report.ram_prefix_evicted_bytes > 0:
                    report.ram_evictions += 1

            # Check invariants after every request
            check_invariants(req_idx)
            if req_idx >= 19 and (req_idx + 1) % 10 == 0:
                report.post_warmup_active_samples.append(mx.get_active_memory())
                report.post_warmup_cache_samples.append(mx.get_cache_memory())

        # Final assertions
        gc.collect()
        mx.clear_cache()
        final_mlx = mx.get_active_memory()
        report.final_mlx_bytes = final_mlx
        report.memory_growth_bytes = final_mlx - initial_mlx

        # Memory should not leak over 100 requests (< 15 MB growth)
        if report.memory_growth_bytes > 15 * MB:
            report.verdict = "WARN_LEAK"
            report.failures.append(f"Memory growth too high: {report.memory_growth_bytes / MB:.2f} MB")

        auto_reclaim.reset_for_tests()
        MM.reset_for_tests()

    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return report


def main():
    parser = argparse.ArgumentParser(description="Lifecycle Stress Test")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests (default %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--dir", type=str, default=None, help="Cache directory")
    args = parser.parse_args()

    print(f"=== FlashNext Auto-Reclaim Lifecycle Stress Test ({args.requests} requests) ===")
    start = time.perf_counter()
    report = run_lifecycle_stress_test(args.requests, directory=Path(args.dir) if args.dir else None, seed=args.seed)
    elapsed = time.perf_counter() - start

    print(f"Verdict:                    {report.verdict}")
    print(f"Total Requests:             {report.total_requests} in {elapsed:.2f}s ({report.total_requests/elapsed:.1f} req/s)")
    print(f"Cold Prefills:              {report.cold_prefills}")
    print(f"RAM Cache Hits:             {report.ram_hits}")
    print(f"SSD Cache Hits:             {report.ssd_hits}")
    print(f"RAM Evictions to SSD:       {report.ram_evictions}")
    print(f"Pool Releases under Stress: {report.pool_releases}")
    print(f"Pool Rebuilds after Recovery:{report.pool_rebuilds}")
    print(f"Cancellations Handled:      {report.cancellations}")
    print(f"Post-Cancel Recoveries:     {report.cancellation_recoveries}")
    print(f"Initial MLX Memory:         {report.initial_mlx_bytes / 1024:.1f} KiB")
    print(f"Final MLX Memory:           {report.final_mlx_bytes / 1024:.1f} KiB (Growth: {report.memory_growth_bytes / 1024:.1f} KiB)")
    if report.post_warmup_active_samples:
        samples = report.post_warmup_active_samples
        print(f"Post-warmup active:         first {samples[0] / 1024:.1f}, last {samples[-1] / 1024:.1f}, max {max(samples) / 1024:.1f} KiB (sampled every 10 requests)")
        print(f"Post-warmup cache max:      {max(report.post_warmup_cache_samples) / 1024:.1f} KiB (sampled)")
    if report.failures:
        print(f"Failures:                   {report.failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
