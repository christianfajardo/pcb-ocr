#!/usr/bin/env bash
set -euo pipefail

# Run tests against the running pipeline

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

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
PYTHONPATH=. pytest tests/ -v --timeout=1100

echo ""
echo "=== Running local extraction test ==="
PYTHONPATH=. python scripts/local_test.py