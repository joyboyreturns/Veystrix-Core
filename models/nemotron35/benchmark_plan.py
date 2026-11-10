"""Emit a future real-checkpoint benchmark matrix; never load models or start runs."""
import json

def plan():
    return {
        'status': 'not run; requires user-provided checkpoints and separate authorization',
        'hardware': 'Apple Silicon M4, 16 GB unified memory',
        'arms': ['flashnext-full-routing', 'nemotron35-lightning'],
        'cold_prefill_tokens': [1024, 4096, 8192, 16384, 32768],
        'warm_prefill': True,
        'decode_tokens': [128, 512],
        'collect': ['ttft', 'prefill_tokens_per_second', 'decode_tokens_per_second',
                    'physical_ssd_bytes_per_token', 'expert_cache_hits', 'expert_cache_misses',
                    'gpu_time', 'io_wait', 'unified_memory', 'swap_growth', 'memory_pressure',
                    'thermal_throttling', 'hermes_tool_turn_latency', 'coding_tasks_per_hour'],
        'controls': ['fresh-process paired comparisons', 'alternating arm order', 'same context and sampler',
                     'record warm/cold cache state', 'thermal and power observations', 'no expert dropping'],
        'targets_not_promises': {'cold_8k_prefill_tokens_per_second': [120, 150],
                                 'sustained_decode_tokens_per_second': [8, 10],
                                 'decode_stretch_tokens_per_second': 12, 'relative_target': 1.5},
    }

if __name__ == '__main__':
    print(json.dumps(plan(), indent=2))
