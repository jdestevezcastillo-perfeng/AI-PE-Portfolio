#!/bin/bash
# 4-Stage Benchmark Runner
# This script helps run all 12 benchmark tests across 4 configurations

set -e

echo "=================================================="
echo "  4-Stage Inference Engine Benchmark Runner"
echo "=================================================="
echo ""
echo "This will run 12 benchmark tests:"
echo "  - 4 configurations (2 models × 2 engines)"
echo "  - 3 load levels per configuration"
echo ""
echo "Total estimated time: ~2-3 hours"
echo "=================================================="
echo ""

# Configuration
VLLM_PORT=8000
TGI_PORT=8080
RESULTS_DIR="benchmark_results"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Models
LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

# Load levels
LOADS=(
  "50 1"
  "100 3"
  "200 5"
)

echo "=================================================="
echo "STAGE 1: Llama-3.1-8B on vLLM"
echo "=================================================="
echo ""
echo "Please start vLLM with:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model $LLAMA_MODEL \\"
echo "    --gpu-memory-utilization 0.9 \\"
echo "    --max-model-len 8192 \\"
echo "    --port $VLLM_PORT"
echo ""
read -p "Press Enter when vLLM is ready..."

for load in "${LOADS[@]}"; do
  read -r requests concurrency <<< "$load"
  echo ""
  echo "Running: $requests requests, concurrency $concurrency"
  python3 lab_05_benchmark_suite_enhanced.py \
    --engine vllm \
    --model "$LLAMA_MODEL" \
    --url "http://localhost:$VLLM_PORT/v1" \
    --requests "$requests" \
    --concurrency "$concurrency" \
    --output-dir "$RESULTS_DIR"
  echo "✓ Completed"
done

echo ""
echo "=================================================="
echo "STAGE 2: Llama-3.1-8B on TGI"
echo "=================================================="
echo ""
echo "Please:"
echo "  1. Stop vLLM (Ctrl+C)"
echo "  2. Start TGI with:"
echo "     docker run --gpus all --shm-size 1g -p $TGI_PORT:80 \\"
echo "       -v \$HOME/.cache/huggingface:/data \\"
echo "       ghcr.io/huggingface/text-generation-inference:latest \\"
echo "       --model-id $LLAMA_MODEL \\"
echo "       --max-total-tokens 8192"
echo ""
read -p "Press Enter when TGI is ready..."

for load in "${LOADS[@]}"; do
  read -r requests concurrency <<< "$load"
  echo ""
  echo "Running: $requests requests, concurrency $concurrency"
  python3 lab_05_benchmark_suite_enhanced.py \
    --engine tgi \
    --model "$LLAMA_MODEL" \
    --url "http://localhost:$TGI_PORT/v1" \
    --requests "$requests" \
    --concurrency "$concurrency" \
    --output-dir "$RESULTS_DIR"
  echo "✓ Completed"
done

echo ""
echo "=================================================="
echo "STAGE 3: Mistral-7B on vLLM"
echo "=================================================="
echo ""
echo "Please:"
echo "  1. Stop TGI (docker stop)"
echo "  2. Start vLLM with:"
echo "     python -m vllm.entrypoints.openai.api_server \\"
echo "       --model $MISTRAL_MODEL \\"
echo "       --gpu-memory-utilization 0.9 \\"
echo "       --max-model-len 8192 \\"
echo "       --port $VLLM_PORT"
echo ""
read -p "Press Enter when vLLM is ready..."

for load in "${LOADS[@]}"; do
  read -r requests concurrency <<< "$load"
  echo ""
  echo "Running: $requests requests, concurrency $concurrency"
  python3 lab_05_benchmark_suite_enhanced.py \
    --engine vllm \
    --model "$MISTRAL_MODEL" \
    --url "http://localhost:$VLLM_PORT/v1" \
    --requests "$requests" \
    --concurrency "$concurrency" \
    --output-dir "$RESULTS_DIR"
  echo "✓ Completed"
done

echo ""
echo "=================================================="
echo "STAGE 4: Mistral-7B on TGI"
echo "=================================================="
echo ""
echo "Please:"
echo "  1. Stop vLLM (Ctrl+C)"
echo "  2. Start TGI with:"
echo "     docker run --gpus all --shm-size 1g -p $TGI_PORT:80 \\"
echo "       -v \$HOME/.cache/huggingface:/data \\"
echo "       ghcr.io/huggingface/text-generation-inference:latest \\"
echo "       --model-id $MISTRAL_MODEL \\"
echo "       --max-total-tokens 8192"
echo ""
read -p "Press Enter when TGI is ready..."

for load in "${LOADS[@]}"; do
  read -r requests concurrency <<< "$load"
  echo ""
  echo "Running: $requests requests, concurrency $concurrency"
  python3 lab_05_benchmark_suite_enhanced.py \
    --engine tgi \
    --model "$MISTRAL_MODEL" \
    --url "http://localhost:$TGI_PORT/v1" \
    --requests "$requests" \
    --concurrency "$concurrency" \
    --output-dir "$RESULTS_DIR"
  echo "✓ Completed"
done

echo ""
echo "=================================================="
echo "  ALL BENCHMARKS COMPLETED! 🎉"
echo "=================================================="
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Open Grafana: http://localhost:3000"
echo "  2. View 'Inference Engine Comparison (Multi-Model)' dashboard"
echo "  3. Use model filter to analyze results"
echo "  4. Document findings in BENCHMARK_RESULTS.md"
echo ""
echo "=================================================="
