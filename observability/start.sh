#!/bin/bash

# AI Performance Engineering Observability Stack
# Start script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting AI Performance Observability Stack..."
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

# Check if ROCm is available
if [ -f "/opt/rocm-6.3.0/bin/rocm-smi" ]; then
    echo "✓ ROCm detected"
else
    echo "⚠ ROCm not found at /opt/rocm-6.3.0 - GPU metrics may not work"
fi

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama is running"
else
    echo "⚠ Ollama not detected on port 11434 - LLM metrics may not work"
fi

# Install Python dependencies for exporters
echo ""
echo "Checking Python dependencies..."
pip install --quiet requests 2>/dev/null || pip3 install --quiet requests 2>/dev/null || true

# Stop any existing host exporters
echo "Stopping any existing exporters..."
pkill -f "rocm_exporter.py" 2>/dev/null || true
pkill -f "ollama_exporter.py" 2>/dev/null || true

echo ""
echo "Starting Docker containers..."
docker compose up -d

echo ""
echo "Starting host exporters..."

# Start ROCm exporter on host
python3 "$SCRIPT_DIR/exporters/rocm_exporter.py" &
ROCM_PID=$!
echo "ROCm exporter started (PID: $ROCM_PID)"

# Start Ollama exporter on host
python3 "$SCRIPT_DIR/exporters/ollama_exporter.py" &
OLLAMA_PID=$!
echo "Ollama exporter started (PID: $OLLAMA_PID)"

# Save PIDs for stop script
echo "$ROCM_PID" > "$SCRIPT_DIR/.rocm_exporter.pid"
echo "$OLLAMA_PID" > "$SCRIPT_DIR/.ollama_exporter.pid"

echo ""
echo "Waiting for services to start..."
sleep 8

# Check service health
echo ""
echo "Service Status:"
echo "──────────────────────────────────────────────"

check_service() {
    local name=$1
    local url=$2
    if curl -s "$url" > /dev/null 2>&1; then
        echo "✓ $name"
    else
        echo "✗ $name"
    fi
}

check_service "Prometheus" "http://localhost:9090/-/ready"
check_service "Grafana" "http://localhost:3000/api/health"
check_service "Loki" "http://localhost:3100/ready"
check_service "Node Exporter" "http://localhost:9100/metrics"
check_service "ROCm Exporter" "http://localhost:9102/metrics"
check_service "Ollama Exporter" "http://localhost:9103/metrics"

echo "──────────────────────────────────────────────"
echo ""
echo "Access Points:"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Loki:       http://localhost:3100"
echo ""
echo "To stop: ./stop.sh"
echo "To view logs: docker compose logs -f"
