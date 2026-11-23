#!/usr/bin/env bash
# Regenerate every figure of the PQ-PKS paper (Linux/macOS). See make_all_figures.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python3}"
if [ -x "$ROOT/.venv-confirm/bin/python" ]; then PY="$ROOT/.venv-confirm/bin/python"; fi
exec "$PY" "$HERE/make_all_figures.py" "$@"
