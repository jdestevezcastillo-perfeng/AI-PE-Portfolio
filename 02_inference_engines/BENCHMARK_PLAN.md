# vLLM vs TGI Benchmark Plan

## Setup Complete ✅

### Dashboards Created
1. **vLLM Inference Metrics** - Detailed vLLM monitoring
2. **TGI Inference Metrics** - Detailed TGI monitoring  
3. **Inference Engine Comparison** - Side-by-side comparison

### Prometheus Configuration
- vLLM: `http://host.docker.internal:8000/metrics`
- TGI: `http://host.docker.internal:8080/metrics`
- Scrape interval: 2s for both

### Model
- **meta-llama/Llama-3.1-8B-Instruct** (same for both engines)

## Benchmark Parameters

### Test Matrix
We'll run the following combinations:

| Requests | Concurrency | Purpose |
|----------|-------------|---------|
| 50       | 1           | Baseline single-request performance |
| 100      | 3           | Moderate concurrent load |
| 200      | 5           | High concurrent load |

### Metrics to Compare

#### Latency (Lower is Better)
- **TTFT (Time To First Token)** - How quickly the first token arrives
- **TPOT (Time Per Output Token)** - Smoothness of generation
- **P99 Latency** - Tail latency for reliability

#### Throughput (Higher is Better)
- **Tokens/sec** - Overall generation speed
- **Requests/sec** - Request processing rate

#### Resource Efficiency
- **GPU Utilization** - How well the GPU is used
- **VRAM Usage** - Memory efficiency
- **Queue Depth** - Request backlog

## Execution Plan

### Phase 1: vLLM Benchmarks
```bash
# Stop TGI, start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192

# Run benchmarks
python3 lab_04_benchmark_suite.py --model meta-llama/Llama-3.1-8B-Instruct --requests 50 --concurrency 1
python3 lab_04_benchmark_suite.py --model meta-llama/Llama-3.1-8B-Instruct --requests 100 --concurrency 3
python3 lab_04_benchmark_suite.py --model meta-llama/Llama-3.1-8B-Instruct --requests 200 --concurrency 5
```

### Phase 2: TGI Benchmarks
```bash
# Stop vLLM, TGI already running
# Wait for "Connected" message

# Run same benchmarks (TGI uses port 8080)
python3 lab_04_benchmark_suite.py --url http://localhost:8080/v1 --model meta-llama/Llama-3.1-8B-Instruct --requests 50 --concurrency 1
python3 lab_04_benchmark_suite.py --url http://localhost:8080/v1 --model meta-llama/Llama-3.1-8B-Instruct --requests 100 --concurrency 3
python3 lab_04_benchmark_suite.py --url http://localhost:8080/v1 --model meta-llama/Llama-3.1-8B-Instruct --requests 200 --concurrency 5
```

### Phase 3: Analysis
- Open Grafana at `http://localhost:3000`
- Navigate to "Inference Engine Comparison" dashboard
- Compare metrics side-by-side
- Document findings

## Expected Insights

### vLLM Strengths
- Optimized for high throughput with PagedAttention
- Better batching with continuous batching
- Lower latency for concurrent requests

### TGI Strengths
- Production-ready with built-in monitoring
- Robust error handling
- Better resource management for long-running deployments

## Current Status

- ✅ Dashboards created
- ✅ Prometheus configured
- ✅ vLLM tested (200 requests completed)
- ⏳ TGI downloading model (ETA: ~9 minutes)
- ⏳ Benchmarks pending
