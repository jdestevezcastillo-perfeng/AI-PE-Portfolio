# vLLM vs TGI Benchmark Plan - 4-Stage Comparison

## Setup Complete ✅

### Dashboards Created
1. **vLLM Inference Metrics** - Detailed vLLM monitoring (multi-model support)
2. **TGI Inference Metrics** - Detailed TGI monitoring (multi-model support)
3. **Inference Engine Comparison** - Side-by-side comparison (multi-model support)

### Prometheus Configuration
- vLLM: `http://host.docker.internal:8000/metrics`
- TGI: `http://host.docker.internal:8080/metrics`
- Scrape interval: 2s for both

### Models Under Test
- **meta-llama/Llama-3.1-8B-Instruct** (8B parameters, gated)
- **mistralai/Mistral-7B-Instruct-v0.2** (7B parameters, open)

## 4-Stage Test Matrix

This benchmark compares **2 models × 2 engines = 4 configurations**:

| Stage | Engine | Model | Port | Purpose |
|-------|--------|-------|------|---------|
| **1** | vLLM | Llama-3.1-8B-Instruct | 8000 | Baseline vLLM with Llama |
| **2** | TGI | Llama-3.1-8B-Instruct | 8080 | Baseline TGI with Llama |
| **3** | vLLM | Mistral-7B-Instruct-v0.2 | 8000 | vLLM with Mistral |
| **4** | TGI | Mistral-7B-Instruct-v0.2 | 8080 | TGI with Mistral |

### Load Test Parameters (Applied to Each Stage)

| Requests | Concurrency | Purpose |
|----------|-------------|---------|
| 50       | 1           | Baseline single-request performance |
| 100      | 3           | Moderate concurrent load |
| 200      | 5           | High concurrent load |

**Total Tests**: 4 stages × 3 load levels = **12 benchmark runs**

### Metrics to Compare

#### Latency (Lower is Better)

- **TTFT (Time To First Token)** - How quickly the first token arrives
- **TPOT (Time Per Output Token)** - Smoothness of generation
- **ITL (Inter-Token Latency)** - Average time between tokens
- **P99 Latency** - Tail latency for reliability

#### Throughput (Higher is Better)

- **Tokens/sec** - Overall generation speed
- **Requests/sec** - Request processing rate

#### Resource Efficiency

- **GPU Utilization** - How well the GPU is used
- **VRAM Usage** - Memory efficiency
- **Queue Depth** - Request backlog

## Execution Plan

### Stage 1: Llama-3.1-8B on vLLM

```bash
# Start vLLM with Llama-3.1-8B
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# Run benchmarks
python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1 \
  --requests 50 --concurrency 1

python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1 \
  --requests 100 --concurrency 3

python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1 \
  --requests 200 --concurrency 5
```

### Stage 2: Llama-3.1-8B on TGI

```bash
# Stop vLLM, start TGI with Llama-3.1-8B
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --max-total-tokens 8192

# Wait for "Connected" message, then run benchmarks
python3 lab_05_benchmark_suite_enhanced.py \
  --engine tgi \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8080/v1 \
  --requests 50 --concurrency 1

python3 lab_05_benchmark_suite_enhanced.py \
  --engine tgi \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8080/v1 \
  --requests 100 --concurrency 3

python3 lab_05_benchmark_suite_enhanced.py \
  --engine tgi \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8080/v1 \
  --requests 200 --concurrency 5
```

### Stage 3: Mistral-7B on vLLM
```bash
# Stop TGI, start vLLM with Mistral
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# Run benchmarks
python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --url http://localhost:8000/v1 \
  --requests 50 --concurrency 1

python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --url http://localhost:8000/v1 \
  --requests 100 --concurrency 3

python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --url http://localhost:8000/v1 \
  --requests 200 --concurrency 5
```

### Stage 4: Mistral-7B on TGI
```bash
# Stop vLLM, start TGI with Mistral
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.2 \
  --max-total-tokens 8192

# Wait for "Connected" message, then run benchmarks
python3 lab_05_benchmark_suite_enhanced.py \
  --engine tgi \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --url http://localhost:8080/v1 \
  --requests 50 --concurrency 1

python3 lab_05_benchmark_suite_enhanced.py \
  --engine tgi \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --url http://localhost:8080/v1 \
  --requests 100 --concurrency 3

python3 lab_05_benchmark_suite_enhanced.py \
  --engine tgi \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --url http://localhost:8080/v1 \
  --requests 200 --concurrency 5
```

### Analysis Phase

- Open Grafana at `http://localhost:3000`
- Navigate to "Inference Engine Comparison" dashboard
- Use model filter to compare:
  - **Same model, different engines** (e.g., Llama on vLLM vs TGI)
  - **Same engine, different models** (e.g., vLLM with Llama vs Mistral)
  - **Cross-comparison** (all 4 configurations)
- Document findings in `BENCHMARK_RESULTS.md`

## Expected Insights

### Engine Comparison (Same Model)

- **vLLM Strengths**: PagedAttention, continuous batching, lower latency for concurrent requests
- **TGI Strengths**: Production-ready monitoring, robust error handling, resource management

### Model Comparison (Same Engine)

- **Llama-3.1-8B**: Slightly larger, may have different token generation patterns
- **Mistral-7B**: Smaller, potentially faster inference, different architecture optimizations

### Cross-Analysis

- Which engine handles which model better?
- Does model size impact engine performance differently?
- Are there engine-specific optimizations for certain architectures?

## Current Status

- ✅ Dashboards created and enhanced for multi-model support
- ✅ Prometheus configured
- ✅ Enhanced benchmark script created (`lab_05_benchmark_suite_enhanced.py`)
- ⏳ 4-stage benchmarks pending
- ⏳ Cross-analysis pending
