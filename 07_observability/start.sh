#!/bin/bash
# ==========================================
# Start AI Performance Engineering Observability Stack
# ==========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting AI Performance Engineering Observability Stack..."
echo "==========================================================="

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if ROCm is available (optional)
if [ -d "/opt/rocm" ]; then
    echo "ROCm detected at /opt/rocm"
else
    echo "WARNING: ROCm not found. GPU metrics may not be available."
fi

# Check if Ollama is running (optional)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama detected at localhost:11434"
else
    echo "WARNING: Ollama not running. LLM metrics will show ollama_up=0"
fi

echo ""
echo "Building custom exporters..."
docker compose build --quiet

echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 5

echo ""
echo "==========================================================="
echo "Observability Stack Started!"
echo "==========================================================="
echo ""
echo "Access Services:"
echo "  Grafana:    http://localhost:3000  (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Loki:       http://localhost:3100"
echo "  Tempo:      http://localhost:3200"
echo ""
echo "Exporters:"
echo "  Node:   http://localhost:9100/metrics"
echo "  ROCm:   http://localhost:9102/metrics"
echo "  Ollama: http://localhost:9103/metrics"
echo ""
echo "Pre-configured Dashboards:"
echo "  - AI Performance Engineering (LLM metrics, inference stats)"
echo "  - GPU Hardware Monitoring (temps, VRAM, power, clocks)"
echo ""
echo "To stop: ./stop.sh"
echo "To view logs: docker compose logs -f"
