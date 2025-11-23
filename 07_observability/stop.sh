#!/bin/bash
# ==========================================
# Stop AI Performance Engineering Observability Stack
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping AI Performance Engineering Observability Stack..."

docker compose down

echo ""
echo "Stack stopped."
echo ""
echo "To also remove data volumes: docker compose down -v"
