#!/bin/bash

# AI Performance Engineering Observability Stack
# Stop script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Stopping AI Performance Observability Stack..."

# Stop host exporters
echo "Stopping host exporters..."
pkill -f "rocm_exporter.py" 2>/dev/null || true
pkill -f "ollama_exporter.py" 2>/dev/null || true
rm -f "$SCRIPT_DIR/.rocm_exporter.pid" "$SCRIPT_DIR/.ollama_exporter.pid"

# Stop Docker containers
echo "Stopping Docker containers..."
docker compose down

echo ""
echo "All services stopped."
