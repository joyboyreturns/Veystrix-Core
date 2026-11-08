import os
from pathlib import Path

def create_workspace(root_dir: str):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    # config/settings.yaml
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text("""# Nimbus Gateway Operational Settings
server:
  host: "0.0.0.0"
  port: 8443
  worker_threads: 16
  keepalive_timeout_seconds: 120

pipeline:
  batch_size: 64
  max_payload_bytes: 10485760
  enable_compression: true

auth:
  issuer: "https://auth.nimbus.internal/v2"
  token_leeway_seconds: 30
  audience: "nimbus-gateway-api"
  cache_ttl_seconds: 300

telemetry:
  sample_rate: 0.05
  export_interval_seconds: 10
""")

    # README.md
    (root / "README.md").write_text("""# Nimbus Gateway

A high-performance API gateway and event distribution layer.

## Architecture
- `service/core/`: Core event loop, ingress dispatcher, and stream pipeline.
- `service/api/`: Routing handlers, endpoint definitions, and token authentication.
- `service/models/`: Immutable schema definitions and dataclass contracts.
- `config/`: Deployment configuration and runtime parameters.
""")

    # service/
    service_dir = root / "service"
    api_dir = service_dir / "api"
    core_dir = service_dir / "core"
    models_dir = service_dir / "models"
    tests_dir = root / "tests"

    for d in (service_dir, api_dir, core_dir, models_dir, tests_dir):
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text('"""Package marker."""\n')

    # service/api/auth.py
    (api_dir / "auth.py").write_text("""\"\"\"Authentication token validation and security policies.\"\"\"
from dataclasses import dataclass
import time

TOKEN_LEEWAY_SECONDS = 30

@dataclass(frozen=True)
class TokenValidator:
    issuer: str = "https://auth.nimbus.internal/v2"
    leeway_seconds: int = TOKEN_LEEWAY_SECONDS

    def validate_claims(self, claims: dict) -> bool:
        now = time.time()
        exp = claims.get("exp", 0)
        if now - exp > self.leeway_seconds:
            return False
        return claims.get("iss") == self.issuer
""")

    # service/api/routes.py
    (api_dir / "routes.py").write_text("""\"\"\"API route registration and handler dispatching.\"\"\"

ROUTES = [
    {"method": "GET", "path": "/health", "auth_required": False},
    {"method": "GET", "path": "/api/v1/services/status", "auth_required": True},
    {"method": "POST", "path": "/api/v1/services/dispatch", "auth_required": True},
]
""")

    # service/core/engine.py
    (core_dir / "engine.py").write_text("""\"\"\"Ingress dispatcher and core event processing engine.\"\"\"
from service.api.auth import TokenValidator
from service.core.pipeline import StreamPipeline

class IngressEngine:
    def __init__(self, batch_size: int = 64):
        self.validator = TokenValidator()
        self.pipeline = StreamPipeline(batch_size=batch_size)

    def process_request(self, token_claims: dict, payload: bytes) -> dict:
        if not self.validator.validate_claims(token_claims):
            return {"status": 401, "error": "unauthorized"}
        return self.pipeline.process(payload)
""")

    # service/core/pipeline.py
    (core_dir / "pipeline.py").write_text("""\"\"\"Streaming transformation pipeline.\"\"\"

class StreamPipeline:
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def process(self, payload: bytes) -> dict:
        return {"status": 200, "batch_size": self.batch_size, "bytes": len(payload)}
""")

    # service/models/schema.py
    (models_dir / "schema.py").write_text("""\"\"\"Immutable data contracts and schemas.\"\"\"
from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDescriptor:
    path: str
    target_cluster: str
    timeout_ms: int = 5000

@dataclass(frozen=True)
class AuthContext:
    subject: str
    tenant: str
    is_admin: bool = False
""")

    # tests/test_engine.py
    (tests_dir / "test_engine.py").write_text("""\"\"\"Unit tests for IngressEngine.\"\"\"
import unittest
from service.core.engine import IngressEngine

class TestEngine(unittest.TestCase):
    def test_pipeline_batch_size(self):
        engine = IngressEngine(batch_size=64)
        self.assertEqual(engine.pipeline.batch_size, 64)

if __name__ == "__main__":
    unittest.main()
""")
    print("Workspace created at:", root_dir)

if __name__ == "__main__":
    create_workspace("/tmp/veystrix_agentic_4k_fixture/workspace")
