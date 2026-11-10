"""Fail-closed memory admission for the streamed Nemotron backend (16 GB class Macs).

Everything is arithmetic on explicit inputs plus a fresh machine reading. The
per-token activation constant is a deliberately conservative *estimate*, not a
measurement; real peak memory must be measured with explicit authorization.
A refusal happens before any allocation and is retriable.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from .memory import MemoryEstimate, estimate

GIB = 2 ** 30
ACTIVATION_BYTES_PER_CHUNK_TOKEN = 256 * 1024     # estimate: projections, router, expert rows, SSM scan temporaries
MIN_OS_RESERVE = 3 * GIB                           # never plan into this, regardless of machine size
SWAP_REFUSE_BYTES = int(2.5 * GIB)                 # refuse new work at the project's swap-growth stop line
MAX_CONTEXT = 262144
DEFAULT_CONTEXT = 32768
DEFAULT_POOL_BYTES = 2 * GIB


try:  # the shared server maps this exception to HTTP 503 memory_admission_refused
    from models.flashnext.request_admission import RequestMemoryRefused as _Retriable
except ImportError:  # pragma: no cover - standalone use
    _Retriable = RuntimeError


class AdmissionRefused(_Retriable):
    """Retriable: nothing was loaded or allocated."""


@dataclass(frozen=True)
class Machine:
    total: int               # physical memory
    available: int           # free + inactive + speculative
    swap_used: int
    pressure: int            # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
    gpu_limit: int           # Metal recommended working set


def read_machine() -> Machine:
    """Strict reading; any missing probe raises (fail closed)."""
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
    vm = run('vm_stat')
    page = re.search(r'page size of (\d+) bytes', vm)
    pages = {}
    for label in ('Pages free', 'Pages inactive', 'Pages speculative'):
        m = re.search(rf'{label}:\s+(\d+)\.', vm)
        if not m:
            raise AdmissionRefused(f'memory probe unavailable: {label}')
        pages[label] = int(m.group(1))
    if not page:
        raise AdmissionRefused('memory probe unavailable: page size')
    sysctl = run('sysctl', 'hw.memsize', 'vm.swapusage', 'kern.memorystatus_vm_pressure_level')
    mem = re.search(r'hw\.memsize:\s*(\d+)', sysctl)
    swap = re.search(r'used = ([\d.]+)M', sysctl)
    pressure = re.search(r'kern\.memorystatus_vm_pressure_level:\s*(\d+)', sysctl)
    if not (mem and swap and pressure):
        raise AdmissionRefused('memory probe unavailable: sysctl')
    try:
        import mlx.core as mx
        info = mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info()
        gpu = int(info['max_recommended_working_set_size'])
    except Exception as exc:
        raise AdmissionRefused(f'GPU working-set limit unavailable: {exc}') from exc
    return Machine(total=int(mem.group(1)), swap_used=int(float(swap.group(1)) * 2 ** 20),
                   available=sum(pages.values()) * int(page.group(1)), pressure=int(pressure.group(1)),
                   gpu_limit=gpu)


@dataclass(frozen=True)
class Plan:
    context_tokens: int
    chunk_tokens: int
    pool_bytes: int
    estimate: MemoryEstimate
    budget_bytes: int

    def summary(self) -> dict:
        e = self.estimate
        return {'context_tokens': self.context_tokens, 'chunk_tokens': self.chunk_tokens,
                'pool_bytes': self.pool_bytes, 'dense_bytes': e.non_expert_bytes,
                'mamba_state_bytes': e.mamba_state_bytes, 'kv_bytes': e.attention_kv_bytes,
                'activation_bytes': e.activation_bytes, 'reserve_bytes': e.reserve_bytes,
                'total_bytes': e.total_bytes, 'budget_bytes': self.budget_bytes, 'estimated': True}


def budget(machine: Machine) -> int:
    """Bytes this engine may plan to hold: bounded by the GPU limit, free RAM and an OS reserve."""
    by_total = machine.total - max(MIN_OS_RESERVE, machine.total // 4)
    return max(0, min(machine.gpu_limit, by_total, machine.available))


def make_plan(config: dict, machine: Machine, *, dense_bytes: int, context_tokens: int = DEFAULT_CONTEXT,
              chunk_tokens: int = 2048, pool_bytes: int = DEFAULT_POOL_BYTES, allocator_slack: int = GIB // 2,
              shrink: bool = True) -> Plan:
    """Fit context, chunk and pool into the budget, shrinking pool, then chunk; refuse if the core cannot fit."""
    if machine.pressure >= 4:
        raise AdmissionRefused('system memory pressure is critical; retry when it falls')
    if machine.swap_used >= SWAP_REFUSE_BYTES:
        raise AdmissionRefused(f'swap in use ({machine.swap_used / GIB:.2f} GiB) is at the safety line; retry later')
    limit = budget(machine)
    context_tokens = min(context_tokens, MAX_CONTEXT, config.get('max_position_embeddings', MAX_CONTEXT))

    def build(pool, chunk):
        return estimate(config=config, context_tokens=context_tokens, non_expert_bytes=dense_bytes,
                        expert_pool_bytes=pool, activation_bytes=chunk * ACTIVATION_BYTES_PER_CHUNK_TOKEN,
                        reserve_bytes=allocator_slack)

    pool, chunk = pool_bytes, chunk_tokens
    while True:
        est = build(pool, chunk)
        if est.total_bytes <= limit:
            return Plan(context_tokens, chunk, pool, est, limit)
        if not shrink:
            break
        if pool > 0:
            pool = max(0, pool - 256 * 2 ** 20)
        elif chunk > 256:
            chunk //= 2
        else:
            break
    est = build(0, 256)
    raise AdmissionRefused(
        f'insufficient memory: even without an expert pool the model needs about {est.total_bytes / GIB:.2f} GiB '
        f'(dense core {dense_bytes / GIB:.2f} GiB, context {context_tokens}) but only {limit / GIB:.2f} GiB '
        f'may be planned (available {machine.available / GIB:.2f} GiB). Close apps or lower the context.')


def check_request(plan: Plan, machine: Machine, *, new_tokens: int, cached_tokens: int) -> None:
    """Per-request gate: the context must fit the plan and the machine must not be under stress."""
    if cached_tokens + new_tokens > plan.context_tokens:
        raise AdmissionRefused(
            f'request needs {cached_tokens + new_tokens} context tokens; this session was admitted for {plan.context_tokens}')
    if machine.pressure >= 4:
        raise AdmissionRefused('system memory pressure is critical; retry when it falls')
    if machine.swap_used >= SWAP_REFUSE_BYTES:
        raise AdmissionRefused('swap is at the safety line; retry when system memory pressure falls')


def pool_bytes_from_env(environ=None) -> int:
    env = os.environ if environ is None else environ
    raw = env.get('VEYSTRIX_NEMOTRON_POOL_GB', '')
    if raw in ('', 'auto'):
        return DEFAULT_POOL_BYTES
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError('VEYSTRIX_NEMOTRON_POOL_GB must be a number or auto') from exc
    if not 0 <= value <= 4:
        raise ValueError('VEYSTRIX_NEMOTRON_POOL_GB must be between 0 and 4')
    return int(value * GIB)
