#!/bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
VLLM_PORT=8000
TGI_PORT=8080

# Function to wait for service
wait_for_service() {
    local port=$1
    local service_name=$2
    echo "Waiting for $service_name to be ready on port $port..."
    for i in {1..60}; do
        if curl -s http://localhost:$port/health > /dev/null; then
            echo "$service_name is ready!"
            return 0
        fi
        sleep 5
    done
    echo "Timeout waiting for $service_name"
    return 1
}

echo "=================================================="
echo "Starting Accuracy vs Throughput Benchmark"
echo "=================================================="

# ------------------------------------------------
# 1. Test vLLM
# ------------------------------------------------
echo ""
echo "--- Stage 1: Testing vLLM ---"
echo "Starting vLLM with $MODEL..."

# Start vLLM in background
../venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype auto \
    > vllm_eval.log 2>&1 &

VLLM_PID=$!

# Wait for it to start
wait_for_service $VLLM_PORT "vLLM"

# Run Evaluation
echo "Running evaluation script..."
../venv/bin/python3 evaluate_efficiency.py --url http://localhost:$VLLM_PORT/v1 --model $MODEL --samples 20

# Stop vLLM
echo "Stopping vLLM..."
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

# ------------------------------------------------
# 2. Test TGI
# ------------------------------------------------
echo ""
echo "--- Stage 2: Testing TGI ---"
echo "Starting TGI with $MODEL..."

# Start TGI in background
docker run --gpus all --shm-size 1g -p $TGI_PORT:80 \
    -v $HOME/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id $MODEL \
    --max-total-tokens 8192 \
    > tgi_eval.log 2>&1 &

TGI_PID=$!

# Wait for it to start (TGI takes longer)
# Note: TGI health check path is /health
wait_for_service $TGI_PORT "TGI"

# Run Evaluation
echo "Running evaluation script..."
../venv/bin/python3 evaluate_efficiency.py --url http://localhost:$TGI_PORT/v1 --model $MODEL --samples 20

# Stop TGI
echo "Stopping TGI..."
docker stop $(docker ps -q --filter ancestor=ghcr.io/huggingface/text-generation-inference:latest)

echo ""
echo "=================================================="
echo "Benchmark Complete"
echo "=================================================="
