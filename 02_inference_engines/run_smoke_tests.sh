#!/bin/bash
# Comprehensive Smoke Test for All 4 Configurations
# This validates that all models work with both engines and metrics are flowing

set -e

echo "=========================================================="
echo "  COMPREHENSIVE SMOKE TEST - 4 Configurations"
echo "=========================================================="
echo ""
echo "This will validate:"
echo "  1. vLLM + Llama-3.1-8B"
echo "  2. TGI + Llama-3.1-8B"
echo "  3. vLLM + Mistral-7B"
echo "  4. TGI + Mistral-7B"
echo ""
echo "Each test validates:"
echo "  ✓ Engine health"
echo "  ✓ Basic inference"
echo "  ✓ Streaming inference"
echo "  ✓ Metrics exposure"
echo "  ✓ Prometheus scraping"
echo "  ✓ Grafana connectivity"
echo ""
echo "=========================================================="
echo ""

# Configuration
VLLM_PORT=8000
TGI_PORT=8080
LLAMA_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MISTRAL_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

# Track results
declare -a RESULTS

# Function to run smoke test
run_smoke_test() {
    local stage=$1
    local engine=$2
    local model=$3
    local url=$4
    local metrics_port=$5
    
    echo ""
    echo "=========================================================="
    echo "  STAGE $stage: ${engine^^} + ${model##*/}"
    echo "=========================================================="
    echo ""
    
    if python3 smoke_test.py \
        --engine "$engine" \
        --model "$model" \
        --url "$url" \
        --metrics-port "$metrics_port"; then
        RESULTS+=("PASS - Stage $stage: ${engine^^} + ${model##*/}")
        return 0
    else
        RESULTS+=("FAIL - Stage $stage: ${engine^^} + ${model##*/}")
        return 1
    fi
}

# Stage 1: vLLM + Llama
echo "=========================================================="
echo "STAGE 1: vLLM + Llama-3.1-8B"
echo "=========================================================="
echo ""
echo "Please start vLLM with:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model $LLAMA_MODEL \\"
echo "    --gpu-memory-utilization 0.9 \\"
echo "    --max-model-len 8192 \\"
echo "    --port $VLLM_PORT"
echo ""
read -p "Press Enter when vLLM is ready (or 's' to skip): " response

if [[ "$response" != "s" ]]; then
    run_smoke_test 1 "vllm" "$LLAMA_MODEL" "http://localhost:$VLLM_PORT/v1" "$VLLM_PORT"
else
    RESULTS+=("SKIP - Stage 1: vLLM + Llama-3.1-8B")
fi

# Stage 2: TGI + Llama
echo ""
echo "=========================================================="
echo "STAGE 2: TGI + Llama-3.1-8B"
echo "=========================================================="
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
read -p "Press Enter when TGI is ready (or 's' to skip): " response

if [[ "$response" != "s" ]]; then
    run_smoke_test 2 "tgi" "$LLAMA_MODEL" "http://localhost:$TGI_PORT/v1" "$TGI_PORT"
else
    RESULTS+=("SKIP - Stage 2: TGI + Llama-3.1-8B")
fi

# Stage 3: vLLM + Mistral
echo ""
echo "=========================================================="
echo "STAGE 3: vLLM + Mistral-7B"
echo "=========================================================="
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
read -p "Press Enter when vLLM is ready (or 's' to skip): " response

if [[ "$response" != "s" ]]; then
    run_smoke_test 3 "vllm" "$MISTRAL_MODEL" "http://localhost:$VLLM_PORT/v1" "$VLLM_PORT"
else
    RESULTS+=("SKIP - Stage 3: vLLM + Mistral-7B")
fi

# Stage 4: TGI + Mistral
echo ""
echo "=========================================================="
echo "STAGE 4: TGI + Mistral-7B"
echo "=========================================================="
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
read -p "Press Enter when TGI is ready (or 's' to skip): " response

if [[ "$response" != "s" ]]; then
    run_smoke_test 4 "tgi" "$MISTRAL_MODEL" "http://localhost:$TGI_PORT/v1" "$TGI_PORT"
else
    RESULTS+=("SKIP - Stage 4: TGI + Mistral-7B")
fi

# Final Summary
echo ""
echo "=========================================================="
echo "  COMPREHENSIVE SMOKE TEST SUMMARY"
echo "=========================================================="
echo ""

passed=0
failed=0
skipped=0

for result in "${RESULTS[@]}"; do
    if [[ $result == PASS* ]]; then
        echo "✓ $result"
        ((passed++))
    elif [[ $result == FAIL* ]]; then
        echo "✗ $result"
        ((failed++))
    else
        echo "⊘ $result"
        ((skipped++))
    fi
done

echo ""
echo "=========================================================="
echo "Results: $passed passed, $failed failed, $skipped skipped"
echo "=========================================================="
echo ""

if [ $failed -eq 0 ] && [ $passed -gt 0 ]; then
    echo "🎉 All tested configurations passed!"
    echo ""
    echo "You are ready to run the full benchmark suite:"
    echo "  ./run_4_stage_benchmark.sh"
    echo ""
    exit 0
elif [ $failed -gt 0 ]; then
    echo "❌ Some configurations failed."
    echo ""
    echo "Please fix the issues before running the full benchmark."
    echo ""
    exit 1
else
    echo "⚠️  All tests were skipped."
    echo ""
    exit 0
fi
