#!/usr/bin/env bash
set -euo pipefail

# Run tests against the running pipeline

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# docker compose reads .env itself for container env vars, but this script
# runs pytest directly on the host — it needs API_KEY exported too, so the
# e2e tests can send the same key the live services were started with.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Use the project-local .venv when present, so `make test` works whether or
# not the caller activated it first. Without this, a bare `make test` picks up
# whatever `pytest` happens to be on PATH — typically a system Python with none
# of this project's dependencies, which fails at import with a confusing
# ModuleNotFoundError rather than anything actionable.
PY_BIN="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
    PY_BIN="$(command -v python3 || true)"
fi
if [ -z "$PY_BIN" ]; then
    echo "ERROR: no Python interpreter found." >&2
    exit 1
fi

if ! "$PY_BIN" -c "import pytest, structlog, pymupdf" 2>/dev/null; then
    echo "ERROR: test dependencies missing from $PY_BIN" >&2
    echo "Create the project venv and install deps:" >&2
    echo "    python3 -m venv .venv" >&2
    echo "    .venv/bin/pip install -e '.[dev,tesseract]'" >&2
    exit 1
fi

# Export supervisor URL for tests
export SUPERVISOR_URL="${SUPERVISOR_URL:-http://localhost:8080/extract}"
export SUPERVISOR_HEALTH="${SUPERVISOR_HEALTH:-http://localhost:8080/health}"

echo "=== Checking supervisor health ==="
if ! curl -sf "$SUPERVISOR_HEALTH" >/dev/null 2>&1; then
    echo "ERROR: Supervisor not available at $SUPERVISOR_HEALTH"
    echo "Run scripts/start_services.sh first."
    exit 1
fi
echo "Supervisor is healthy."

echo ""
echo "=== Running tests ==="
PYTHONPATH=. "$PY_BIN" -m pytest tests/ -v --timeout=1100

echo ""
echo "=== Running local extraction test ==="
PYTHONPATH=. "$PY_BIN" scripts/local_test.py