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
for name in ("veystrix", "veystrix", "veystrix.env_compat", "models.flashnext.settings.registry", "models.flashnext.memory_policy"):
    m = importlib.import_module(name)
    f = getattr(m, "__file__", None) or ""
    assert f.startswith(lib), f"{name} came from {f!r}, not the bundle"
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
for args in (["-c", CODE], ["-m", "models.flashnext.overnight", "--help"]):
    r = subprocess.run([str(python), *args], env=env, cwd=str(bundle), capture_output=True, text=True, timeout=180)
    print(" ".join(args[:2]) if args[0] == "-m" else "import check", "->", r.returncode)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:], sep="\n")
        sys.exit(1)
if (bundle / "lib").exists() and list((bundle / "lib").rglob("tests")):
    sys.exit("tests directory leaked into the bundle")
sys.exit(0)
