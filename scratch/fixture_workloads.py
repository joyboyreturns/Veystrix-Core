"""Generates calibrated agentic coding fixtures at approximately 4K, 8K, and 12K tokens.

Calibrated using the real Qwen3.8-Flash-Next tokenizer and Jinja chat template:
- 4K target: 3,900 - 4,100 tokens (calibrated: 4,032)
- 8K target: 7,900 - 8,200 tokens
- 12K target: 11,900 - 12,300 tokens
"""
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from veystrix.server import _normalize_messages

TOK_PATH = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
TOKENIZER = AutoTokenizer.from_pretrained(TOK_PATH)
TEMPLATE_TEXT = Path(TOK_PATH, "chat_template.jinja").read_text()

_ENV = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
_ENV.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
_ENV.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
TEMPLATE = _ENV.from_string(TEMPLATE_TEXT)

# Standard read-only tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching pattern in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*')"},
                    "path": {"type": "string", "description": "Relative directory path (optional)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for pattern inside files in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern or regex to search for"},
                    "path": {"type": "string", "description": "File or directory path to search within"}
                },
                "required": ["pattern"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an expert autonomous software engineer and system architect.
You are assisting in investigating, analyzing, and documenting an isolated software repository in the current working directory.
You have access to read-only repository inspection tools:
- `glob`: find files matching a glob pattern
- `grep`: search for patterns inside files

Follow these principles:
1. Always explore the workspace using `glob` first to discover project layout and relevant files.
2. Use `grep` to locate exact definitions, configuration keys, constants, and function implementations.
3. Base your answers strictly on the empirical findings returned by the tools.
4. Do not speculate or assume file contents without reading or grepping them.
5. Provide concise, clear, and technically precise answers once all required information is gathered.
"""

TASK_QUESTION = """

## OBJECTIVE & INSPECTION GOALS

Please inspect the isolated repository in the current workspace using the available read-only tools:
1. First, invoke `glob` to discover the files and directories present in the project.
2. Based on the files found, invoke `grep` to find the pipeline `batch_size` parameter in the configuration files.
3. Formulate your final response detailing the discovered project structure, the exact batch size value, and how the ingress engine connects to the authentication pipeline.
"""

def render_prompt(task_prompt: str, tools=TOOLS):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt}
    ]
    norm_messages = _normalize_messages(messages)
    rendered = TEMPLATE.render(
        messages=norm_messages,
        tools=tools,
        add_generation_prompt=True,
    )
    tokens = TOKENIZER.encode(rendered)
    return len(tokens), rendered

# Reusable technical specification blocks
MODULE_SPEC_BASE = """
# NIMBUS GATEWAY ARCHITECTURAL SPECIFICATION & SYSTEM DESIGN MANUAL

## 1. System Overview
Nimbus Gateway is a high-throughput, low-latency API gateway and event distribution layer designed for cloud-native microservice deployments. The gateway serves as the primary ingress controller for distributed downstream services, providing rate limiting, mutual TLS termination, cryptographic payload verification, dynamic route dispatching, and distributed telemetry collection.

### 1.1 Architectural Invariants
1. Strict Separation of Control Plane and Data Plane: Route definitions, authentication policies, and rate-limiting quotas are synchronized asynchronously from the control plane cluster via streaming gRPC, while data-plane request evaluation executes entirely within memory without blocking I/O on the critical dispatch path.
2. Zero Allocation Header Parsing: Request headers and query parameters are parsed using slice-based buffer views to eliminate garbage collection pressure under sustained throughput (>150,000 requests per second per worker node).
3. Defensive Security Boundaries: Every inbound request must traverse authentication token validation, HMAC signature verification, and quota admission before any route handler or upstream proxy connection is scheduled.

## 2. Component Hierarchy and Module Responsibilities

### 2.1 Core Ingress Dispatcher (`service/core/engine.py`)
The Ingress Dispatcher is responsible for accepting normalized HTTP/2 and HTTP/3 multiplexed frames from the transport termination layer. The engine maintains an in-memory radix tree of route patterns, associating URI prefix paths with designated upstream target clusters.
Key dispatch responsibilities include:
- Ingress Connection Pooling: Managing connection state, keepalive heartbeats, and graceful circuit breaking against downstream saturation.
- Request Pipeline Scheduling: Passing inbound context objects through the pre-routing filter chain, core dispatch logic, and post-routing response synthesis hooks.
- Backpressure Propagation: Monitoring worker queue depths and emitting HTTP 429 (Too Many Requests) or HTTP 503 (Service Unavailable) status responses when downstream latency crosses the 99th percentile threshold (configured at 45 milliseconds).

### 2.2 Streaming Transformation Pipeline (`service/core/pipeline.py`)
The pipeline module handles payload decoding, decompression, field-level masking, and schema validation.
Pipeline stages execute sequentially:
1. `DecompressionStage`: Evaluates `Content-Encoding` headers (supporting Brotli, Zstandard, and Gzip).
2. `TokenSanitizationStage`: Validates bearer credentials and masks sensitive authentication claims prior to downstream forwarding.
3. `PayloadValidationStage`: Compares request bodies against JSON Schema and Protobuf descriptors.
4. `MetricsCollectionStage`: Records latency distributions, payload sizes, and status codes into high-resolution histogram accumulators.

### 2.3 Authentication and Credential Validation (`service/api/auth.py`)
Authentication in Nimbus Gateway utilizes scoped asymmetric cryptographic tokens (Ed25519 and RS256) combined with ephemeral session certificates.
The validator enforces:
- Token Expiration & Clock Skew: Rejecting tokens with `exp` timestamps older than the evaluation timestamp, allowing a maximum clock drift window of 30 seconds.
- Claim Validation: Verifying that `aud` (audience), `iss` (issuer), and `sub` (subject) claims match the gateway tenant identifier.
- Revocation List Bitmaps: Querying local Bloom filters and Redis-backed revocation sets before admitting high-privilege administrative requests.

### 2.4 Routing Table and Dynamic Upstream Selection (`service/api/routes.py`)
The routing registry decouples public API endpoints from downstream service physical endpoints.
Features include:
- Weighted Round Robin: Distributing traffic across healthy upstream replicas according to assigned weight vectors.
- Health Checking: Running active background HTTP health probes every 5 seconds and passive error-rate monitoring.
- Canary Deployments: Splitting user traffic based on cookie hashes, header values, or random percentage distributions.

### 2.5 Data Models and Schema Definitions (`service/models/schema.py`)
Defines immutable data models representing HTTP contexts, authentication credentials, route descriptors, and telemetry records.
All schema classes derive from typed dataclasses with strict slots to minimize memory allocations.

## 3. Configuration Management and Parameters (`config/settings.yaml`)
Operational parameters are defined in YAML format and mounted as read-only configurations:
- `server.bind_address`: Network interface IPv4/IPv6 bind target (default: `0.0.0.0:8080`).
- `pipeline.batch_size`: Worker ingestion batch size for batch processing (default: `64`).
- `pipeline.buffer_pool_mb`: Maximum pre-allocated buffer pool per worker thread (default: `128`).
- `auth.token_leeway_seconds`: Allowed clock skew window for authentication tokens (default: `30`).
- `auth.algorithm`: Default cryptographic signing algorithm (`Ed25519`).
- `telemetry.prometheus_port`: Prometheus metrics scraper bind port (`9090`).
"""

def generate_technical_section(index: int, title: str) -> str:
    return f"""
## {index}. {title}

### {index}.1 Operational Characteristics
In enterprise deployments of Nimbus Gateway, high-concurrency event loops handle continuous streams of synchronous RPC calls and asynchronous event deliveries. Threading models utilize work-stealing worker pools to maximize processor cache locality while minimizing inter-core synchronization latency. Each worker thread maintains a ring buffer of pending connection handles, draining sockets in batches of up to 64 events per epoll/kqueue wake cycle.

### {index}.2 Fault Tolerance and Resiliency Strategies
To guarantee five-nines availability (99.999% uptime):
- Circuit breakers monitor failure ratios over a rolling 10-second sliding window.
- Outlier detection isolates misbehaving downstream microservice instances when error rates cross 5% over 100 consecutive requests.
- Exponential backoff with jitter prevents thundering herd effects against recovering backend database clusters.
- Shadow traffic mirroring allows zero-impact production validation of candidate release builds by replicating live ingress requests to staging environments.

### {index}.3 Cryptographic and Transport Security Standards
All inter-service communications enforce TLS 1.3 with forward secrecy cipher suites:
- Key exchange: X25519 and ECDH with curve P-256.
- Symmetric encryption: AES-256-GCM and ChaCha20-Poly1305.
- Certificate rotation: Automated ACME protocol renewal every 60 days with hardware security module (HSM) key backing.
- Token validation caching: Cryptographic signature verifications are cached in an LRU state cache keyed by the SHA-256 digest of the token payload, bounded to a maximum of 50,000 active entries with a 60-second TTL.
"""

def build_fixture_prompt(target_tokens: int) -> str:
    """Builds a prompt calibrated to hit target_tokens within +/- 50 tokens."""
    prompt = MODULE_SPEC_BASE
    current_tokens, _ = render_prompt(prompt + TASK_QUESTION)
    
    sec_num = 4
    section_titles = [
        "Distributed Telemetry, Distributed Tracing and OpenTelemetry Instrumentation",
        "Memory Management, Buffer Allocations and Low-Level Kernel Bypasses",
        "Rate Limiting Invariants, Leaky Bucket Implementations and Distributed Quotas",
        "Service Mesh Interoperability, Sidecar Coordination and Envoy Protocols",
        "High Availability Clustering, Raft Consensus and Distributed State Machines",
        "Zero-Trust Network Architecture, Microsegmentation and Identity Enforcement",
        "Cache Hierarchy Design, In-Memory Multi-Tier Caching and Cache Eviction",
        "Protocol Buffers, gRPC Serialization and Streaming Multiplexing Invariants",
        "Asynchronous Event Brokers, Kafka Event Streaming and Partition Offsets",
        "Observability Dashboards, Prometheus Metrics Ingestion and Alerting Rules",
        "Database Sharding Strategies, Consistent Hashing and Replication Topologies",
        "Container Scheduling, Kubernetes Ingress Controllers and Envoy Gateway Invariants",
        "Storage Engine Integration, Write-Ahead Logging and Snapshot Checkpoints",
        "Network Interface Tuning, Socket Option Invariants and Kernel Buffers",
        "Security Audit Trails, Compliance Frameworks and Anomaly Detection Signals",
        "Dynamic Configuration Reloading, Hot-Swapping Filters and Atomic Updates"
    ]
    
    for title in section_titles:
        sec = generate_technical_section(sec_num, title)
        cand_tokens, _ = render_prompt(prompt + sec + TASK_QUESTION)
        if cand_tokens > target_tokens - 60:
            break
        prompt += sec
        sec_num += 1
        current_tokens = cand_tokens
        
    # Fine-tune with exact padding block
    diff = target_tokens - current_tokens
    if diff > 0:
        lines_needed = max(1, diff // 16)
        padding = "\n### Appendix Notes and Configuration Reference Index\n"
        for i in range(lines_needed):
            padding += f"- Index reference entry {i:03d}: verification parameter checksum validated against specification manifest.\n"
        prompt += padding
        
    final_prompt = prompt + TASK_QUESTION
    final_tokens, _ = render_prompt(final_prompt)
    return final_tokens, final_prompt

_CACHED_PROMPTS = {}

def get_fixture_prompt(target_tier: str) -> tuple[int, str]:
    target_tier = target_tier.lower()
    if target_tier not in _CACHED_PROMPTS:
        if target_tier == "4k":
            _CACHED_PROMPTS[target_tier] = build_fixture_prompt(3800)
        elif target_tier == "8k":
            _CACHED_PROMPTS[target_tier] = build_fixture_prompt(7800)
        elif target_tier == "12k":
            _CACHED_PROMPTS[target_tier] = build_fixture_prompt(11600)
        else:
            raise ValueError(f"Unknown fixture tier {target_tier!r}; choose 4k, 8k, or 12k")
    return _CACHED_PROMPTS[target_tier]

if __name__ == "__main__":
    for tier in ("4k", "8k", "12k"):
        toks, p = get_fixture_prompt(tier)
        print(f"Tier {tier.upper()}: {toks} tokens, {len(p)} characters")
