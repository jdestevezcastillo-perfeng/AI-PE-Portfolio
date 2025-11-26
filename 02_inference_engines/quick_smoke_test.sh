#!/bin/bash
# Quick Smoke Test - Single Configuration
# Use this to quickly validate one engine+model combination

set -e

# Default values
ENGINE=""
MODEL=""
URL=""
PORT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --url)
            URL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Quick Smoke Test - Single Configuration"
            echo ""
            echo "Usage:"
            echo "  ./quick_smoke_test.sh --engine vllm --model meta-llama/Llama-3.1-8B-Instruct"
            echo "  ./quick_smoke_test.sh --engine tgi --model mistralai/Mistral-7B-Instruct-v0.2"
            echo ""
            echo "Options:"
            echo "  --engine    Engine to test (vllm or tgi)"
            echo "  --model     Model to test (full model ID)"
            echo "  --url       Optional: API URL (default: auto-detected)"
            echo "  --port      Optional: Metrics port (default: auto-detected)"
            echo ""
            echo "Examples:"
            echo "  # Test vLLM with Llama"
            echo "  ./quick_smoke_test.sh --engine vllm --model meta-llama/Llama-3.1-8B-Instruct"
            echo ""
            echo "  # Test TGI with Mistral"
            echo "  ./quick_smoke_test.sh --engine tgi --model mistralai/Mistral-7B-Instruct-v0.2"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ENGINE" ] || [ -z "$MODEL" ]; then
    echo "Error: --engine and --model are required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate engine
if [ "$ENGINE" != "vllm" ] && [ "$ENGINE" != "tgi" ]; then
    echo "Error: --engine must be 'vllm' or 'tgi'"
    exit 1
fi

# Set defaults based on engine
if [ -z "$URL" ]; then
    if [ "$ENGINE" = "vllm" ]; then
        URL="http://localhost:8000/v1"
    else
        URL="http://localhost:8080/v1"
    fi
fi

if [ -z "$PORT" ]; then
    if [ "$ENGINE" = "vllm" ]; then
        PORT="8000"
    else
        PORT="8080"
    fi
fi

# Extract model name for display
MODEL_NAME="${MODEL##*/}"

echo "=========================================================="
echo "  QUICK SMOKE TEST"
echo "=========================================================="
echo ""
echo "Configuration:"
echo "  Engine:       ${ENGINE^^}"
echo "  Model:        $MODEL_NAME"
echo "  URL:          $URL"
echo "  Metrics Port: $PORT"
echo ""
echo "=========================================================="
echo ""

# Run the smoke test
if python3 smoke_test.py \
    --engine "$ENGINE" \
    --model "$MODEL" \
    --url "$URL" \
    --metrics-port "$PORT"; then
    echo ""
    echo "=========================================================="
    echo "  ✓ SMOKE TEST PASSED"
    echo "=========================================================="
    echo ""
    echo "This configuration is ready for benchmarking!"
    echo ""
    echo "Next steps:"
    echo "  1. Run a quick benchmark:"
    echo "     python3 lab_05_benchmark_suite_enhanced.py \\"
    echo "       --engine $ENGINE \\"
    echo "       --model $MODEL \\"
    echo "       --url $URL \\"
    echo "       --requests 10 --concurrency 1"
    echo ""
    echo "  2. Or run the full benchmark suite:"
    echo "     ./run_4_stage_benchmark.sh"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================================="
    echo "  ✗ SMOKE TEST FAILED"
    echo "=========================================================="
    echo ""
    echo "Please fix the issues before proceeding."
    echo ""
    exit 1
fi
