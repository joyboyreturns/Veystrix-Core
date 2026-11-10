"""Smoke test for a built engine bundle. Imports engine modules and renders the engine's --help.

It never loads a model, opens a checkpoint, starts a server, or initialises Metal: it only proves the compiled
bundle is importable with the bundled interpreter and nothing reaches back to a source checkout.
Usage: <bundle>/python/bin/python3 smoke_import.py <bundle>
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
python = bundle / "python" / "bin" / "python3"
lib = bundle / "lib"
env = {"HOME": str(bundle), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(lib), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}

CODE = """
import importlib, os, sys
lib = os.environ["PYTHONPATH"]
for std in ("sqlite3", "ssl", "zlib", "json", "plistlib", "ctypes", "hashlib", "http.server"):
    importlib.import_module(std)      # stdlib pieces the engine needs must survive the size trimming
for name in ("veystrix", "veystrix.api_keys", "models.flashnext.settings.registry", "models.flashnext.memory_policy", "models.flashnext.chat_guard", "models.flashnext.tests.bench.layer_gate_probe", "models.flashnext.overnight", "veystrix.cli", "veystrix.session"):
    m = importlib.import_module(name)
    f = getattr(m, "__file__", None) or ""
    assert f.startswith(lib), f"{name} came from {f!r}, not the bundle"
from models.flashnext.overnight import server_command, bundled
from types import SimpleNamespace
assert bundled(), "bundle unexpectedly expects chat.sh"
cmd = server_command(SimpleNamespace(pool_gb="0", host="127.0.0.1", port=8080, checkpoint="/tmp/model with spaces"))
assert cmd[:3] == [sys.executable, "-m", "veystrix.cli"] and cmd[-1] == "/tmp/model with spaces", cmd
from veystrix.cli import command
os.environ["VEYSTRIX_PYTHON"] = sys.executable
planned, _ = command(cmd[3:])
assert planned[2:5] == [sys.executable, "-u", "-m"] and "veystrix.session" in planned, planned
assert "/tmp/model with spaces" in planned, planned
from models.flashnext.settings.registry import get_registry
assert len(get_registry().settings) > 0 if hasattr(get_registry(), "settings") else True
try:
    import cv2
    raise SystemExit("opencv (LGPL FFmpeg) must not be in the bundle")
except ImportError:
    pass
import mlx_vlm  # noqa: F401  (imports the package only; no model, no Metal session)
print("imports ok")
"""
for required in (python, lib / "models/flashnext/overnight.pyc", lib / "veystrix/cli.pyc", lib / "veystrix/session.pyc", lib / "models/flashnext/tests/bench/prefill_memory_harness.pyc", lib / "models/flashnext/tests/bench/bench_prefill_memory.pyc", lib / "models/flashnext/tests/bench/layer_gate_probe.pyc", lib / "models/bonsai2/settings.pyc", lib / "models/k2_horizon/settings.pyc"):
    if not required.is_file():
        sys.exit(f"mandatory bundle resource missing: {required}")
if not os.access(python, os.X_OK):
    sys.exit(f"bundled Python is not executable: {python}")
for args in (["-c", CODE], ["-m", "models.flashnext.overnight", "--help"]):
    r = subprocess.run([str(python), *args], env=env, cwd=str(bundle), capture_output=True, text=True, timeout=180)
    print(" ".join(args[:2]) if args[0] == "-m" else "import check", "->", r.returncode)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:], sep="\n")
        sys.exit(1)
allowed = {lib / "models/flashnext/tests"}
if any(path not in allowed for path in lib.rglob("tests")):
    sys.exit("unexpected tests directory leaked into the bundle")
sys.exit(0)
