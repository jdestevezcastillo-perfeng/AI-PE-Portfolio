# Benchmark Scripts

This directory contains Python scripts for benchmarking LLM performance.

## Scripts Overview

### 1. basic_benchmark.py
**Purpose**: Sequential request benchmarking with basic metrics

**What it does**:
- Sends 100 sequential requests to Ollama
- Tests 3 different prompt lengths (short, medium, long)
- Measures end-to-end latency
- Calculates tokens per second (TPS)
- Measures time to first token (TTFT)

**Usage**:
```bash
python basic_benchmark.py
```

**Output**:
- Console output with real-time progress
- JSON file: `benchmark_results_TIMESTAMP.json`

**Runtime**: ~15-30 minutes depending on hardware

**Metrics Collected**:
- Min/Max/Mean/Median/P90/P95/P99 latency
- Tokens per second (decode phase)
- Time to first token (prefill phase)
- Total tokens generated

---

### 2. advanced_benchmark.py
**Purpose**: Comprehensive multi-scenario testing

**Experiments included**:

#### Experiment 1: Prompt Length Impact
- Tests 5 prompt categories (tiny, short, medium, long, very_long)
- Measures how input length affects latency and throughput

#### Experiment 2: Temperature Impact
- Tests temperature values: 0.0, 0.3, 0.7, 1.0, 1.5
- Analyzes generation quality vs performance tradeoff

#### Experiment 3: Concurrent Requests
- Tests 1, 2, and 4 concurrent workers
- Measures queueing behavior and throughput under load

#### Experiment 4: Resource Usage Monitoring
- Monitors CPU and memory usage during sustained inference
- Tracks system resource consumption patterns

**Usage**:
```bash
python advanced_benchmark.py
```

**Output**:
- Console output with experiment results
- JSON file: `advanced_benchmark_TIMESTAMP.json`

**Runtime**: ~20-40 minutes

---

### 3. visualize_results.py
**Purpose**: Generate charts and reports from benchmark data

**Features**:
- Latency distribution plots
- Tokens per second comparisons
- TTFT analysis
- Prompt length impact visualization
- Temperature impact charts
- Concurrency analysis
- Text summary report

**Usage**:
```bash
python visualize_results.py benchmark_results.json

# Or let it auto-detect the most recent results file
python visualize_results.py
```

**Output**:
- `plots/` directory with PNG images
- `benchmark_summary.txt` text report

**Dependencies**: matplotlib, seaborn, numpy

---

## Quick Start

### Step 1: Setup
```bash
# Ensure Ollama is running
ollama serve

# In another terminal, pull the model
ollama pull llama3.1:8b

# Install dependencies
pip install ollama openai requests numpy pandas matplotlib seaborn psutil
```

### Step 2: Run Basic Benchmark
```bash
cd scripts/
python basic_benchmark.py
```

Wait for completion (~20 minutes)

### Step 3: Visualize Results
```bash
python visualize_results.py
```

### Step 4: Run Advanced Benchmarks (Optional)
```bash
python advanced_benchmark.py
python visualize_results.py advanced_benchmark_*.json
```

---

## Understanding the Output

### Key Metrics Explained

**Latency (End-to-End)**:
- Total time from request start to completion
- Includes network, prefill, and decode time
- Lower is better

**TTFT (Time to First Token)**:
- Time until first token is generated
- Measures prefill phase latency
- Critical for user experience (perceived responsiveness)

**TPS (Tokens Per Second)**:
- Rate of token generation during decode phase
- Measures throughput
- Higher is better

**P95/P99 Latency**:
- 95th/99th percentile latency
- Important for understanding tail latency
- SLA targets often use P95 or P99

---

## Performance Expectations

### Typical Results (8B model on CPU)

**Latency**:
- Short prompts: 2-5s
- Medium prompts: 5-10s
- Long prompts: 10-20s

**TTFT**:
- Short prompts: 0.1-0.5s
- Long prompts: 0.5-2s

**TPS**:
- CPU: 5-20 tok/s
- GPU (NVIDIA): 30-100+ tok/s

---

## Customization

### Modify Number of Requests
Edit the script constants:
```python
NUM_REQUESTS = 100  # Change to 50, 200, etc.
```

### Add Custom Prompts
Add to `TEST_PROMPTS` list in basic_benchmark.py:
```python
TEST_PROMPTS = [
    "Your custom prompt here",
    # ...
]
```

### Adjust Model
Change `MODEL_NAME`:
```python
MODEL_NAME = "llama3.1:70b"  # For larger models
MODEL_NAME = "phi3:mini"     # For smaller models
```

---

## Troubleshooting

### Issue: "Connection refused"
**Solution**: Start Ollama service
```bash
ollama serve
```

### Issue: Benchmark runs very slowly
**Solutions**:
- Reduce `NUM_REQUESTS` to 20-50
- Use a smaller model: `phi3:mini`
- Check system resources with `htop`

### Issue: Out of memory
**Solutions**:
- Close other applications
- Use a smaller model
- Reduce concurrent workers in advanced benchmark

### Issue: Visualization fails
**Solution**: Install missing dependencies
```bash
pip install matplotlib seaborn numpy
```

---

## Next Steps

After running benchmarks:

1. **Analyze Results**:
   - Review plots in `plots/` directory
   - Read `benchmark_summary.txt`
   - Identify performance bottlenecks

2. **Document Findings**:
   - Add results to your GitHub repo
   - Write observations in README
   - Note interesting patterns

3. **Experiment Further**:
   - Try different models
   - Test with GPU if available
   - Modify prompts and parameters

4. **Prepare for Day 2**:
   - Tomorrow: Quantization and model optimization
   - You'll compare these baseline results with optimized models
