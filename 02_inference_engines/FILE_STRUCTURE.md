# Module 02: File Structure

## 📁 Directory Organization

```
02_inference_engines/
├── README.md                              # 👈 START HERE - Complete guide
├── BENCHMARK_PLAN.md                      # Detailed execution commands
├── BENCHMARK_RESULTS.md                   # Previous benchmark results
├── 4_STAGE_RESULTS_TEMPLATE.md           # Template for documenting new results
│
├── lab_01_baseline_hf.py                 # Lab 1: Baseline inference
├── lab_02_vllm_inference.py              # Lab 2: vLLM integration
├── lab_03_tgi_setup.md                   # Lab 3: TGI setup guide
├── lab_04_benchmark_suite.py             # Lab 4: Basic benchmark
├── lab_05_benchmark_suite_enhanced.py    # Lab 5: Enhanced multi-model benchmark
│
├── smoke_test.py                         # Core smoke test logic
├── quick_smoke_test.sh                   # Quick smoke test (30s)
├── run_smoke_tests.sh                    # Comprehensive smoke test (5-10min)
└── run_4_stage_benchmark.sh              # Full benchmark runner (2-3hrs)
```

## 📄 File Descriptions

### Documentation

| File | Purpose | When to Use |
|------|---------|-------------|
| **README.md** | Complete guide with quick start, smoke testing, benchmarking | **Start here** |
| **BENCHMARK_PLAN.md** | Detailed commands for each stage | Reference for manual execution |
| **BENCHMARK_RESULTS.md** | Previous results (vLLM vs TGI) | See example results |
| **4_STAGE_RESULTS_TEMPLATE.md** | Template for documenting findings | After running benchmarks |

### Lab Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `lab_01_baseline_hf.py` | Baseline with Hugging Face | `python lab_01_baseline_hf.py` |
| `lab_02_vllm_inference.py` | vLLM integration | `python lab_02_vllm_inference.py` |
| `lab_03_tgi_setup.md` | TGI setup instructions | Read for TGI setup |
| `lab_04_benchmark_suite.py` | Basic benchmark (original) | Legacy - use lab_05 instead |
| `lab_05_benchmark_suite_enhanced.py` | **Enhanced benchmark** | **Use this for benchmarking** |

### Testing & Automation

| File | Purpose | Usage |
|------|---------|-------|
| `smoke_test.py` | Core smoke test logic | Called by shell scripts |
| `quick_smoke_test.sh` | Quick validation (30s) | `./quick_smoke_test.sh --engine vllm --model <model>` |
| `run_smoke_tests.sh` | Test all 4 configs (5-10min) | `./run_smoke_tests.sh` |
| `run_4_stage_benchmark.sh` | Full benchmark (2-3hrs) | `./run_4_stage_benchmark.sh` |

## 🎯 Typical Workflow

### 1. First Time Setup

```bash
# Read the main guide
cat README.md

# Review detailed commands
cat BENCHMARK_PLAN.md
```

### 2. Validate Setup

```bash
# Quick test (30 seconds)
./quick_smoke_test.sh --engine vllm --model meta-llama/Llama-3.1-8B-Instruct

# OR comprehensive test (5-10 minutes)
./run_smoke_tests.sh
```

### 3. Run Benchmarks

```bash
# Automated (recommended)
./run_4_stage_benchmark.sh

# OR manual (see BENCHMARK_PLAN.md)
python3 lab_05_benchmark_suite_enhanced.py --engine vllm --model <model> ...
```

### 4. Document Results

```bash
# Use template as guide
cp 4_STAGE_RESULTS_TEMPLATE.md MY_RESULTS.md
# Fill in your findings
```

## 🗂️ Generated Files

After running benchmarks, you'll see:

```
02_inference_engines/
└── benchmark_results/                    # Auto-created
    ├── vllm_Llama-3.1-8B-Instruct_c1_r50_TIMESTAMP.json
    ├── tgi_Mistral-7B-Instruct-v0.2_c5_r200_TIMESTAMP.json
    └── ...
```

## 📊 Related Files (Other Modules)

```
07_observability/
└── grafana/
    └── dashboards/
        ├── inference-comparison.json      # Multi-model comparison
        ├── vllm-inference.json           # vLLM metrics
        └── tgi-inference.json            # TGI metrics
```

## 🧹 Cleaned Up Files

The following redundant files were removed:

- ❌ `4_STAGE_QUICK_START.md` → Merged into `README.md`
- ❌ `4_STAGE_BENCHMARK_SUMMARY.md` → Merged into `README.md`
- ❌ `SMOKE_TEST_GUIDE.md` → Merged into `README.md`
- ❌ `SMOKE_TEST_SUMMARY.md` → Merged into `README.md`
- ❌ `FINAL_COMPARISON.md` → Renamed to `BENCHMARK_RESULTS.md`

All important information is now in **README.md** (complete guide) and **BENCHMARK_PLAN.md** (detailed commands).

---

**Start with README.md for the complete guide!** 📖
