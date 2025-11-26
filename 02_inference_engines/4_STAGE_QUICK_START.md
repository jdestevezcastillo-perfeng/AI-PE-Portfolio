# 4-Stage Inference Engine Benchmark - Quick Start

## What's This?

An expanded benchmark comparing **vLLM** and **TGI** inference engines across **2 models** (Llama-3.1-8B and Mistral-7B) with **3 load levels** each.

**Total**: 12 benchmark runs across 4 configurations

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
cd /home/lostborion/AI-PE-Portfolio/02_inference_engines
./run_4_stage_benchmark.sh
```

The script will guide you through:

1. Starting each engine/model combination
2. Running 3 load tests per configuration
3. Saving results automatically

### Option 2: Manual Execution

See `BENCHMARK_PLAN.md` for detailed commands for each stage.

## Files

| File | Purpose |
|------|---------|
| `lab_05_benchmark_suite_enhanced.py` | Enhanced benchmark script with multi-model support |
| `run_4_stage_benchmark.sh` | Automated runner for all 12 tests |
| `BENCHMARK_PLAN.md` | Detailed execution plan and commands |
| `4_STAGE_BENCHMARK_SUMMARY.md` | Complete overview and analysis guide |
| `benchmark_results/` | Directory where JSON results are saved |

## Dashboards

All dashboards now support model filtering:

1. **Inference Engine Comparison (Multi-Model)** - Side-by-side comparison
2. **vLLM Inference Metrics (Multi-Model)** - vLLM deep dive
3. **TGI Inference Metrics (Multi-Model)** - TGI deep dive

Access: `http://localhost:3000`

## Test Matrix

| Stage | Engine | Model | Load Levels |
|-------|--------|-------|-------------|
| 1 | vLLM | Llama-3.1-8B | 50/1, 100/3, 200/5 |
| 2 | TGI | Llama-3.1-8B | 50/1, 100/3, 200/5 |
| 3 | vLLM | Mistral-7B | 50/1, 100/3, 200/5 |
| 4 | TGI | Mistral-7B | 50/1, 100/3, 200/5 |

## Analysis Questions

### 1. Engine Comparison

**Question**: Which engine is better for Llama-3.1-8B?

**Compare**: Stage 1 vs Stage 2

### 2. Model Comparison

**Question**: How does vLLM perform with different models?

**Compare**: Stage 1 vs Stage 3

### 3. Cross-Analysis

**Question**: What's the optimal engine-model combination?

**Compare**: All 4 stages

## Metrics Tracked

- **Latency**: TTFT, TPOT, ITL (mean/p50/p99)
- **Throughput**: Tokens/sec, Requests/sec
- **Resource**: GPU utilization, queue depth, batch size

## Example: Running a Single Test

```bash
# Start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Run benchmark
python3 lab_05_benchmark_suite_enhanced.py \
  --engine vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --url http://localhost:8000/v1 \
  --requests 50 \
  --concurrency 1
```

## Results

Results are saved as JSON files:

```
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

## Next Steps

1. ✅ Run benchmarks with `./run_4_stage_benchmark.sh`
2. 📊 Analyze in Grafana dashboards
3. 📝 Document findings in `BENCHMARK_RESULTS.md`
4. 🎯 Identify optimal configurations

## Need Help?

- **Detailed Plan**: See `BENCHMARK_PLAN.md`
- **Complete Guide**: See `4_STAGE_BENCHMARK_SUMMARY.md`
- **Previous Results**: See `BENCHMARK_RESULTS.md` (from 2-engine comparison)

---

**Ready to benchmark!** 🚀
