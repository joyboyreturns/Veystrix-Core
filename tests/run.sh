#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${VEYSTRIX_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON" -m veystrix.testsuite "$@"
