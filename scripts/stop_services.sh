#!/usr/bin/env bash
set -euo pipefail

# Stop all services and clean up

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Stopping PCB OCR pipeline ==="

docker compose -f "$COMPOSE_FILE" down

echo "=== All services stopped ==="