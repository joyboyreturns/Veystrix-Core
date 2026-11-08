import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from macqwen.server import _normalize_messages

# Load tokenizer and template
tok_path = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
tokenizer = AutoTokenizer.from_pretrained(tok_path)
template_text = Path(tok_path, "chat_template.jinja").read_text()

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
def raise_exception(msg): raise RuntimeError(msg)
template = env.from_string(template_text)

# Base captured tools and system prompt from real dsh
captured = json.load(open("/Users/vincent/.gemini/antigravity/brain/f9adb8b8-2aca-4265-99e4-6a15a70f5cc5/scratch/captured_dsh_payload.json"))
tools = captured["tools"]
system_msg = captured["messages"][0]
context_msg = captured["messages"][2]

# Project Architecture Specification (realistic, high-quality technical context)
SPEC_HEADER = """You are tasked with analyzing the architecture and source code of the Nimbus Gateway microservice project located in the current workspace.

Before inspecting the repository, review the following architectural specification and integration requirements for the project:

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
- Issuer Authority: Verifying that the `iss` claim matches the trusted authority cluster (`https://auth.nimbus.internal/v2`).
- Scope Matrix Enforcement: Matching granted scopes against the target resource route descriptor. Required scopes for administrative operations include `nimbus:admin:read` and `nimbus:admin:write`.
- Replay Protection: Tracking token nonces within an in-memory sliding bloom filter with a 5-minute eviction interval.

### 2.4 Data Models and Serialization Contracts (`service/models/schema.py`)
All internal events, telemetry spans, and route configurations are structured as immutable dataclasses with strict type annotations.
- `RouteDescriptor`: Defines path regex patterns, upstream cluster targets, timeout budgets, and retry parameters.
- `AuthContext`: Captures authenticated subject identifiers, tenant keys, roles, and cryptographic fingerprint signatures.
- `TelemetryEvent`: High-volume trace record containing span identifiers, timestamps, latency quantiles, and status flags.

### 2.5 Configuration and Runtime Parameters (`config/settings.yaml`)
Runtime behavior is driven by hierarchical configuration overlays:
- `server.port`: TCP listen port for public traffic.
- `server.worker_threads`: Concurrency execution pool size.
- `pipeline.batch_size`: Maximum number of stream frames aggregated prior to vector dispatch.
- `auth.token_leeway_seconds`: Permissible clock drift window.
- `telemetry.sample_rate`: Sampling ratio for distributed OpenTelemetry traces.

## 3. Operational Protocols and Read-Only Inspection Guidelines

When analyzing this repository:
1. Use only the read-only file discovery tools (`glob` and `grep`) available to you.
2. Do not attempt to modify, create, delete, or overwrite any files in the project workspace.
3. First inspect the directory structure using `glob` to verify which files are present.
4. Locate the configuration settings and source implementations relevant to the verification question.
5. Answer the technical question accurately using verified facts extracted from the files.

"""

TASK_QUESTION = """
TASK INSTRUCTION:
Inspect the files in this isolated test project using the available read-only tools (`glob` or `grep`).
Identify the relevant files, locate the definition of the pipeline batch size in `config/settings.yaml` and the authentication token validator in `service/api/auth.py`, explain the project structure, and answer what batch size is configured and what token leeway window is enforced.
Do not modify any files.
"""

# Let's see how many tokens this is, and adjust padding/context if needed to reach exactly ~4000 tokens
def evaluate_task(task_text):
    messages = [
        system_msg,
        {"role": "user", "content": task_text},
        context_msg
    ]
    rendered = template.render(
        messages=_normalize_messages(messages),
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
        raise_exception=raise_exception
    )
    tokens = tokenizer.encode(rendered)
    return len(tokens), rendered

tokens_count, _ = evaluate_task(SPEC_HEADER + TASK_QUESTION)
print("Initial token count:", tokens_count)

# Let's adjust technical sections to hit between 3950 and 4050 tokens
target = 4020
diff = target - tokens_count
print("Need to add approximately", diff, "tokens")

# Add detailed module specifications to reach target
EXTRA_SECTION = """
## 4. Extended Microservice Subsystem Specifications

### 4.1 Cluster State Management and Gossip Protocol
The gateway participates in an internal cluster state gossip protocol to synchronize active worker health across nodes. The gossip layer periodically broadcasts node heartbeat vectors and active connection counts. When a node experiences hardware failure or thermal throttling, sibling nodes detect missing heartbeats within 150 milliseconds and automatically adjust ingress traffic distribution weights.

### 4.2 Circuit Breaking and Fault Injection Safeguards
Downstream services protected by Nimbus Gateway are monitored by dynamic circuit breakers. A circuit transitions from Closed to Open when error rates exceed 5% over a 10-second rolling window. While Open, all inbound requests destined for that upstream service are immediately short-circuited with a fallback response or an HTTP 503 error, preventing cascading collapse across downstream dependencies. A Half-Open probing state allows controlled trial requests every 2 seconds to detect upstream service recovery.

### 4.3 High-Resolution Rate Limiting Engine
Rate limiting employs a distributed token bucket algorithm with sub-millisecond refill precision. Token buckets are stored in partitioned shared memory arrays synchronized across local CPU cores via atomic compare-and-swap (CAS) instructions. This architecture avoids lock contention on multi-socket NUMA servers, allowing continuous burst handling without degradation of packet dispatch throughput.

### 4.4 Upstream Connection Lifecycle and Keepalive Optimization
To maximize TCP socket reuse and minimize TLS handshake overhead against backend microservices, the connection manager implements:
- Persistent Connection Pools: Pre-warming a configurable baseline pool of warm HTTP/2 streams per backend endpoint.
- Aggressive Idle Reaping: Terminating idle connections older than 120 seconds to prevent TCP socket exhaustion on operating system network stacks.
- Half-Close Handling: Detecting TCP FIN packets from upstream peers and cleanly draining in-flight streams before socket disposal.

### 4.5 Memory Allocator Architecture and Buffer Lifecycle
Memory allocations in the data plane are governed by custom arena allocators. Each inbound connection receives a thread-local arena pool that grows dynamically in 64 KiB slabs. As request frames are decoded and forwarded, transient allocations are fulfilled directly from the arena. Upon completion of the request-response lifecycle, the entire arena is reset in O(1) time without triggering individual object deallocations or heap fragmentation.

### 4.6 Cryptographic Acceleration and Kernel Offload
Cryptographic token verification operations leverage hardware-accelerated AES-NI and SHA extensions. For asymmetric signature validation (Ed25519 and RSA), batch signature verification routines accumulate signatures across pipeline ticks, dispatching batched vector operations to hardware accelerators when available.

### 4.7 Distributed Tracing and W3C TraceContext Ingestion
Inbound requests carrying `traceparent` and `tracestate` headers per the W3C TraceContext recommendation have their context extracted and injected into downstream spans. The gateway generates unique 64-bit span identifiers for its own internal processing stages, measuring DNS lookup time, connection acquisition latency, TLS handshake duration, and time-to-first-byte (TTFB).

### 4.8 Zero-Downtime Hot Reloading and Configuration Drift Detection
Configuration updates pushed to `config/settings.yaml` are detected via kernel file-system event notifications (`kqueue` on macOS / BSD and `inotify` on Linux). When a file modification event fires, the gateway parses the candidate configuration into an isolated shadow state, runs structural validation tests, and atomically swaps the active pointer using an atomic pointer swap. Existing worker threads finish processing in-flight requests under the previous configuration while new requests immediately adopt the updated settings.

### 4.9 Error Handling Matrix and Diagnostic Telemetry
The following table documents error conditions and operational response codes:
- `ERR_AUTH_EXPIRED`: The provided bearer token expired prior to evaluation. HTTP 401 Unauthorized emitted.
- `ERR_AUTH_INVALID_SIGNATURE`: Cryptographic signature verification failed against trusted public keys. HTTP 401 Unauthorized emitted.
- `ERR_AUTH_INSUFFICIENT_SCOPE`: Token lacks required resource permissions. HTTP 403 Forbidden emitted.
- `ERR_ROUTE_NOT_FOUND`: URI prefix path does not match any registered route descriptor. HTTP 404 Not Found emitted.
- `ERR_UPSTREAM_TIMEOUT`: Backend service failed to deliver a response within the configured timeout budget. HTTP 504 Gateway Timeout emitted.
- `ERR_CIRCUIT_OPEN`: Target upstream cluster is currently isolated by circuit breaking policies. HTTP 503 Service Unavailable emitted.

### 4.10 Logging Invariants and Sensitive Data Scrubbing
Under no circumstances may bearer credentials, authorization headers, private keys, or unmasked user data appear in stdout, stderr, or persistent log streams. All log serializers utilize regex-free token lookaheads that redact values mapped to sensitive header keys before emitting structured JSON log lines to the log aggregation daemon.

## 5. API Route Definitions and Transport Protocols

### 5.1 Route Mapping Protocol
The routing registry maintains mapping tuples of the form `(verb, path_prefix, target_cluster, policy_chain)`.
Supported HTTP verbs include `GET`, `POST`, `PUT`, `DELETE`, `PATCH`, and `HEAD`.
- Public Routes (`/api/v1/public/*`): Unauthenticated endpoints with strict IP rate limiting (100 requests per minute per /24 subnet).
- Authenticated Business Endpoints (`/api/v1/services/*`): Require valid `AuthContext` with non-empty tenant scope.
- Internal Health and Management (`/management/*`): Restricted to local loopback and administrative overlay networks.

### 5.2 Transport Layer Security (TLS) and Cipher Negotiation
All non-loopback connections mandate TLS 1.3 or TLS 1.2 with forward secrecy cipher suites:
- `TLS_AES_256_GCM_SHA384`
- `TLS_CHACHA20_POLY1305_SHA256`
- `TLS_AES_128_GCM_SHA256`
- `ECDHE-ECDSA-AES256-GCM-SHA384`
- `ECDHE-RSA-AES256-GCM-SHA384`
Legacy cipher suites (CBC mode, RC4, 3DES) and TLS versions prior to 1.2 are rejected during the initial client hello handshake.

### 5.3 WebSocket and Streaming Bidirectional Upgrades
When an inbound request contains `Upgrade: websocket` headers:
1. The authentication filter performs an immediate upfront token check before connection promotion.
2. Connection handles are handed off to dedicated WebSocket worker loops utilizing epoll/kqueue event polling.
3. Keepalive ping/pong frames are exchanged every 30 seconds to maintain NAT firewall pinholes.

## 6. Verification and Integration Requirements

### 6.1 Isolated Project Structure
The repository structure under inspection contains the following organization:
```
nimbus-gateway/
├── README.md                  # Project overview, build instructions, and quickstart guide
├── config/
│   └── settings.yaml          # Operational configuration, port bindings, batch size, token leeway
├── service/
│   ├── __init__.py            # Package initialization marker
│   ├── api/
│   │   ├── __init__.py        # API subpackage marker
│   │   ├── auth.py            # Token validation, clock skew leeway, cryptographic checks
│   │   └── routes.py          # Route dispatcher and handler registrations
│   ├── core/
│   │   ├── __init__.py        # Core engine package marker
│   │   ├── engine.py          # Main ingress event dispatcher and worker loop
│   │   └── pipeline.py        # Stream transformation pipeline and frame processing
│   └── models/
│       ├── __init__.py        # Models package marker
│       └── schema.py          # Immutable dataclasses for route descriptors and auth context
└── tests/
    ├── __init__.py            # Test package marker
    └── test_engine.py         # Unit and integration test suite
```

### 6.2 Inspection Target Checklist
When completing this task:
1. Call `glob` with a pattern matching files in the project to verify all existing components.
2. Locate and inspect `config/settings.yaml` to extract the exact configured pipeline batch size (`pipeline.batch_size`).
3. Locate and inspect `service/api/auth.py` to extract the configured clock skew leeway window in seconds (`token_leeway_seconds`).
4. Detail the architectural relationship between `service/api/auth.py` and `service/core/engine.py`.
5. Present a clear, concise report answering the questions using the actual discovered values.

## 7. Deployment Topologies and Environment Matrices

### 7.1 Containerized Pod Topology
Nimbus Gateway instances run as stateless containers within Kubernetes worker nodes. Pods are deployed with horizontal pod autoscalers (HPA) targeting 70% CPU utilization and 65% memory working set. Each pod mounts local non-volatile scratch storage at `/var/run/nimbus/scratch` for ephemeral socket buffers and TLS session ticket caches.

### 7.2 Graceful Termination and Drain Protocols
Upon receiving a `SIGTERM` signal from the container runtime:
1. The gateway unregisters its health endpoint from the ingress service mesh to prevent new connections from being routed to the terminating pod.
2. Active client connections receive HTTP/2 `GOAWAY` frames with a graceful drain grace period of 25 seconds.
3. In-flight request pipelines are allowed to finish processing and transmit downstream responses.
4. After all active connections reach zero or the timeout expires, the process terminates with return code 0.

"""

tokens_count_2, _ = evaluate_task(SPEC_HEADER + EXTRA_SECTION + TASK_QUESTION)
print("Token count with extra sections:", tokens_count_2)

