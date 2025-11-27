# Module 02: Inference Engines - Complete Guide

## 🎯 Objective

Master high-performance inference engines (vLLM and TGI) through comprehensive benchmarking across multiple models. Learn how specialized engines optimize memory management (PagedAttention), batching (Continuous Batching), and kernel execution to deliver production-grade performance.

## 📚 Key Concepts

1. **PagedAttention (vLLM)** - Solving memory fragmentation in KV caches
2. **Continuous Batching** - Dynamic request batching without waiting for completion
3. **Tensor Parallelism** - Splitting models across multiple GPUs
4. **Production Metrics** - TTFT, TPOT, ITL, throughput, and resource utilization

## 🛠️ Tools

- **vLLM** - State-of-the-art high-throughput serving
- **TGI (Text Generation Inference)** - Hugging Face's production-ready server
- **Prometheus + Grafana** - Metrics collection and visualization

---

## 📋 Quick Start

### 1. Validate Setup (IMPORTANT - Do This First!)

Before running benchmarks, validate your setup with smoke tests:

```bash
# Quick test (30 seconds) - test current configuration
./quick_smoke_test.sh --engine vllm --model meta-llama/Llama-3.1-8B-Instruct

# Comprehensive test (5-10 min) - test all 4 configurations
./run_smoke_tests.sh
```

**Why?** A 30-second smoke test can save hours of debugging failed benchmarks.

### 2. Run Benchmarks

After smoke tests pass:

```bash
# Automated - runs all 12 benchmark tests
./run_4_stage_benchmark.sh

# Manual - see BENCHMARK_PLAN.md for detailed commands
```

### 3. Analyze Results

- Open Grafana: `http://localhost:3000`
- View dashboards with model filtering
- Document findings using `4_STAGE_RESULTS_TEMPLATE.md`

---

## 🧪 Lab Exercises

### Lab 01: Baseline Inference (`lab_01_baseline_hf.py`)

Establish baseline using standard Hugging Face transformers. Measures what "slow" looks like.

**Run**:

```bash
python lab_01_baseline_hf.py
```

### Lab 02: vLLM Integration (`lab_02_vllm_inference.py`)

Set up vLLM and measure TTFT and ITL.

**Run**:

```bash
# Start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Test
python lab_02_vllm_inference.py
```

### Lab 03: TGI Setup (`lab_03_tgi_setup.md`)

Deploy TGI via Docker and interact with its API.

### Lab 04: Basic Benchmark Suite (`lab_04_benchmark_suite.py`)

Original benchmark script for single model comparison.

### Lab 05: Enhanced Benchmark Suite (`lab_05_benchmark_suite_enhanced.py`)

**NEW!** Multi-model, multi-engine benchmark with JSON output.

**Features**:

- Engine tagging (vllm/tgi)
- Multi-model support
- Enhanced metrics (TTFT, ITL p50/p99, throughput)
- JSON output for analysis

**Usage**:

```bash
python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1 \
  --requests 50 \
  --concurrency 1
```

---

## 🔬 4-Stage Benchmark System

### Test Matrix

Compare **2 models × 2 engines = 4 configurations** across **3 load levels**:

| Stage | Engine | Model | Port | Tests |
|-------|--------|-------|------|-------|
| **1** | vLLM | Llama-3.1-8B-Instruct | 8000 | 50/1, 100/3, 200/5 |
| **2** | TGI | Llama-3.1-8B-Instruct | 8080 | 50/1, 100/3, 200/5 |
| **3** | vLLM | Mistral-7B-Instruct-v0.2 | 8000 | 50/1, 100/3, 200/5 |
| **4** | TGI | Mistral-7B-Instruct-v0.2 | 8080 | 50/1, 100/3, 200/5 |

**Total**: 12 benchmark runs

### Analysis Capabilities

1. **Engine Comparison** (Same Model) - Which engine is better for Llama or Mistral?
2. **Model Comparison** (Same Engine) - How does vLLM/TGI perform with different models?
3. **Cross-Analysis** - What's the optimal engine-model combination?

### Metrics Tracked

**Latency** (Lower is Better):

- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- ITL (Inter-Token Latency) - mean, p50, p99

**Throughput** (Higher is Better):

- Tokens/sec (per-request and overall)
- Requests/sec

**Resource Efficiency**:

- GPU utilization
- VRAM usage
- Queue depth

---

## 🧪 Smoke Testing

### Purpose

Validate that all components work before running long benchmarks:

- ✅ Engine health
- ✅ Basic and streaming inference
- ✅ Metrics exposure
- ✅ Prometheus scraping
- ✅ Grafana connectivity

### Quick Test (30 seconds)

```bash
./quick_smoke_test.sh --engine vllm --model meta-llama/Llama-3.1-8B-Instruct
```

### Comprehensive Test (5-10 minutes)

Tests all 4 configurations:

```bash
./run_smoke_tests.sh
```

### What Gets Tested

1. **Health Check** - Engine responding
2. **Basic Inference** - Non-streaming requests work
3. **Streaming Inference** - Streaming with TTFT measurement
4. **Metrics Endpoint** - Prometheus metrics exposed
5. **Prometheus Scraping** - Metrics being collected
6. **Grafana Connectivity** - Dashboards accessible

### Common Issues

**"Could not access metrics endpoint"**:

```bash
# Check metrics are exposed
curl http://localhost:8000/metrics  # vLLM
curl http://localhost:8080/metrics  # TGI
```

**"Prometheus is running but no data yet"**:
Wait 5-10 seconds for first scrape, then re-run test.

**"Could not query Prometheus"**:

```bash
# Start Prometheus
cd ../07_observability
docker-compose up -d prometheus
```

---

## 📊 Grafana Dashboards

All dashboards now support **multi-model filtering**:

### 1. Inference Engine Comparison (Multi-Model)

- Side-by-side comparison of vLLM vs TGI
- Model filter dropdown (All, Llama-3.1-8B, Mistral-7B)
- Latency, throughput, and resource metrics

### 2. vLLM Inference Metrics (Multi-Model)

- Deep dive into vLLM performance
- KV cache metrics
- Request queue visualization

### 3. TGI Inference Metrics (Multi-Model)

- Deep dive into TGI performance
- Batch processing metrics
- Request pipeline breakdown

**Access**: `http://localhost:3000`

---

## 📁 File Reference

### Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `smoke_test.py` | Core smoke test logic | Called by shell scripts |
| `quick_smoke_test.sh` | Quick single config test | `./quick_smoke_test.sh --engine vllm --model <model>` |
| `run_smoke_tests.sh` | Test all 4 configs | `./run_smoke_tests.sh` |
| `lab_05_benchmark_suite_enhanced.py` | Enhanced benchmark script | See usage above |
| `run_4_stage_benchmark.sh` | Automated benchmark runner | `./run_4_stage_benchmark.sh` |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | **This file** - Complete guide |
| `BENCHMARK_PLAN.md` | Detailed execution plan with all commands |
| `BENCHMARK_RESULTS.md` | Previous benchmark results (2-engine comparison) |
| `FILE_STRUCTURE.md` | Directory organization guide |

### Templates

| File | Purpose |
|------|---------|
| `4_STAGE_RESULTS_TEMPLATE.md` | Template for documenting your new findings |

---

## 🚀 Complete Workflow

### Step 1: Start Engine

```bash
# vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# TGI
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --max-total-tokens 8192
```

### Step 2: Smoke Test

```bash
./quick_smoke_test.sh --engine vllm --model meta-llama/Llama-3.1-8B-Instruct
```

### Step 3: Quick Benchmark (10 requests)

```bash
python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1 \
  --requests 10 --concurrency 1
```

### Step 4: Check Grafana

Open `http://localhost:3000` and verify metrics are flowing.

### Step 5: Full Benchmark

```bash
# Test all 4 configurations first
./run_smoke_tests.sh

# If all pass, run full benchmark
./run_4_stage_benchmark.sh
```

### Step 6: Analyze Results

- View Grafana dashboards
- Use model filter to compare configurations
- Create a new results file using `4_STAGE_RESULTS_TEMPLATE.md` as a guide
- See `BENCHMARK_RESULTS.md` for example of previous results

---

## 🎯 Expected Insights

### Engine Strengths

**vLLM**:

- PagedAttention for memory efficiency
- Continuous batching for throughput
- Lower latency under concurrent load
- Best for: Custom optimizations, cost-sensitive environments

**TGI**:

- Production-ready monitoring
- Robust error handling
- Better resource management
- Best for: Enterprise deployments, Hugging Face ecosystem

### Model Characteristics

**Llama-3.1-8B**:

- Larger model (8B parameters)
- Different token generation patterns
- Potentially higher quality outputs

**Mistral-7B**:

- Smaller model (7B parameters)
- Potentially faster inference
- Different architecture optimizations

---

## 📈 Results

Results are saved as JSON files in `benchmark_results/`:

```json
benchmark_results/
├── vllm_Llama-3.1-8B-Instruct_c1_r50_TIMESTAMP.json
├── tgi_Mistral-7B-Instruct-v0.2_c5_r200_TIMESTAMP.json
└── ...
```

Each file contains:

- Configuration details
- Latency metrics (TTFT, ITL, percentiles)
- Throughput metrics
- Raw per-request data

---

## 🔧 Troubleshooting

### Smoke Test Fails

1. Check engine is running: `curl http://localhost:8000/health`
2. Check Prometheus: `curl http://localhost:9090/api/v1/query?query=up`
3. Check Grafana: `curl http://localhost:3000/api/health`
4. Review error messages in smoke test output

### Benchmark Fails

1. Run smoke test first
2. Check GPU memory: `nvidia-smi`
3. Check logs from engine
4. Reduce concurrency or request count

### Metrics Not Showing

1. Wait 5-10 seconds for Prometheus to scrape
2. Check Prometheus targets: `http://localhost:9090/targets`
3. Verify metrics endpoint: `curl http://localhost:8000/metrics`

---

## 📚 Additional Resources

- **vLLM Documentation**: <https://docs.vllm.ai/>
- **TGI Documentation**: <https://huggingface.co/docs/text-generation-inference/>
- **Prometheus**: <https://prometheus.io/docs/>
- **Grafana**: <https://grafana.com/docs/>

---

## ✅ Checklist

Before running full benchmarks:

- [ ] Prometheus and Grafana running
- [ ] Smoke tests pass for all configurations
- [ ] Grafana dashboards accessible
- [ ] Sufficient disk space for results
- [ ] GPU memory available

---

**Ready to benchmark!** 🚀

For detailed execution commands, see `BENCHMARK_PLAN.md`.
