#!/usr/bin/env bash
set -euo pipefail

# Start all services and wait for them to be healthy

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Starting PCB OCR pipeline ==="

# Build and start all services
docker compose -f "$COMPOSE_FILE" up -d --build

# Wait for each service to be healthy
wait_for() {
    local name=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=0

    echo -n "  Waiting for $name..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo " OK"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo " FAILED (after $((max_attempts * 2))s)"
    docker compose -f "$COMPOSE_FILE" logs "$name"
    return 1
}

# Start in dependency order
wait_for "vllm-glm-ocr"  "http://localhost:8010/health"  90    # ~180s, matches cold weight load
wait_for "vllm-qwen-vl"  "http://localhost:8011/health"  150   # ~300s, 8B model needs more headroom
wait_for "tesseract-ocr" "http://localhost:8001/health"  10
wait_for "glm-ocr-api"   "http://localhost:8002/health"  10
wait_for "qwen-vl-api"   "http://localhost:8003/health"  10
wait_for "supervisor"    "http://localhost:8080/health"  10

echo ""
echo "=== All services healthy ==="
echo "Supervisor API: http://localhost:8080/extract"