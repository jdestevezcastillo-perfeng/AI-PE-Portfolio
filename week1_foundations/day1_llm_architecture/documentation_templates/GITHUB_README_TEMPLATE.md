# Week 1 Day 1: LLM Architecture & First Benchmarks

**Date**: [YYYY-MM-DD]
**Model**: llama3.1:8b
**Platform**: [CPU/GPU] - [Your Hardware Details]

---

## Overview

First day of my AI infrastructure and performance learning journey. Today's focus:
- Understanding transformer architecture fundamentals
- Learning about KV-cache mechanisms
- Running first LLM performance benchmarks
- Establishing baseline metrics

---

## Learning Objectives

✅ Understand transformer self-attention mechanism
✅ Explain KV-cache and its role in inference
✅ Measure end-to-end latency and throughput
✅ Identify performance bottlenecks
✅ Document findings and visualizations

---

## Environment Setup

### Hardware
- **CPU**: [e.g., Intel i7-12700K]
- **RAM**: [e.g., 32GB DDR4]
- **GPU**: [e.g., None / NVIDIA RTX 3090]
- **Storage**: [e.g., 1TB NVMe SSD]

### Software
- **OS**: [e.g., Ubuntu 22.04 / macOS 14.2]
- **Ollama**: [version]
- **Python**: [version]
- **Model**: llama3.1:8b (4.7GB)

### Installation
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.1:8b

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install ollama requests numpy pandas matplotlib seaborn psutil
```

---

## Benchmark Results

### Basic Sequential Benchmark

Ran 100 sequential requests across 3 prompt lengths:

#### Short Prompt: "What is AI?"

| Metric | Value |
|--------|-------|
| Mean Latency | [X.XX]s |
| Median Latency | [X.XX]s |
| P95 Latency | [X.XX]s |
| P99 Latency | [X.XX]s |
| Mean TPS | [XX] tok/s |
| Mean TTFT | [X.XX]s |
| Total Tokens | [XXXX] |

#### Medium Prompt: "Explain the concept of machine learning..."

| Metric | Value |
|--------|-------|
| Mean Latency | [X.XX]s |
| Median Latency | [X.XX]s |
| P95 Latency | [X.XX]s |
| P99 Latency | [X.XX]s |
| Mean TPS | [XX] tok/s |
| Mean TTFT | [X.XX]s |
| Total Tokens | [XXXX] |

#### Long Prompt: "Describe the architecture of a transformer..."

| Metric | Value |
|--------|-------|
| Mean Latency | [X.XX]s |
| Median Latency | [X.XX]s |
| P95 Latency | [X.XX]s |
| P99 Latency | [X.XX]s |
| Mean TPS | [XX] tok/s |
| Mean TTFT | [X.XX]s |
| Total Tokens | [XXXX] |

---

### Advanced Experiments

#### Experiment 1: Prompt Length Impact

![Prompt Length Impact](plots/prompt_length_impact.png)

**Key Findings**:
- Latency scales [linearly/sub-linearly/super-linearly] with prompt length
- TTFT increases significantly for longer prompts ([X]s → [Y]s)
- TPS remains relatively stable at [XX] tok/s regardless of prompt length

#### Experiment 2: Temperature Impact

![Temperature Impact](plots/temperature_impact.png)

**Key Findings**:
- Temperature has [minimal/moderate/significant] impact on latency
- Higher temperatures ([1.5]) resulted in [X]% [slower/faster] generation
- Potential reason: [Your analysis]

#### Experiment 3: Concurrent Requests

![Concurrent Impact](plots/concurrent_impact.png)

**Key Findings**:
- Ollama processes requests [sequentially/in parallel]
- Throughput [increased/did not increase] with concurrent workers
- Queueing behavior observed: [Your observations]

#### Experiment 4: Resource Usage

**System Metrics During Sustained Load**:
- Average CPU: [XX]%
- Average Memory: [XX]GB
- Peak Memory: [XX]GB
- Throughput: [XX] req/s

**Bottleneck Analysis**:
- Primary bottleneck: [CPU/Memory/Disk]
- Evidence: [Your observations]

---

## Visualizations

### Latency Distribution
![Latency Distribution](plots/latency_distribution.png)

### Tokens Per Second
![Tokens Per Second](plots/tokens_per_second.png)

### Time to First Token
![TTFT Comparison](plots/ttft_comparison.png)

---

## Key Learnings

### Technical Insights

1. **Prefill vs Decode**:
   - TTFT (prefill) accounts for ~[XX]% of total latency for short prompts
   - Decode phase generates ~[XX] tokens/second
   - [Additional observations]

2. **Memory Characteristics**:
   - Base memory usage: ~[XX]GB
   - Memory growth with sequence length: [Your analysis]
   - KV-cache impact: [Estimated based on theory]

3. **Performance Patterns**:
   - [Observation 1]
   - [Observation 2]
   - [Observation 3]

### Theoretical Understanding

1. **Transformer Architecture**:
   - Self-attention creates context-aware representations
   - Multi-head attention captures different relationship types
   - Parallel processing during training vs sequential during generation

2. **KV-Cache Mechanism**:
   - Prevents O(n²) redundant computation
   - Trades memory for speed (5-10x faster)
   - Linear memory growth with sequence length

3. **Performance Metrics**:
   - TTFT = Time to First Token (user-perceived latency)
   - TPOT = Time Per Output Token (generation speed)
   - TPS = Tokens Per Second (throughput metric)

---

## Performance Analysis

### Bottleneck Identification

**Primary Bottleneck**: [CPU/Memory/Bandwidth]

**Evidence**:
- [Metric or observation 1]
- [Metric or observation 2]
- [Metric or observation 3]

**Optimization Opportunities**:
1. [Potential optimization 1]
2. [Potential optimization 2]
3. [Potential optimization 3]

### Comparison to Theoretical Expectations

| Metric | Expected (Theory) | Actual | Analysis |
|--------|-------------------|--------|----------|
| TPS (CPU) | 10-20 tok/s | [XX] tok/s | [Your analysis] |
| TTFT | 0.1-0.5s (short) | [X.XX]s | [Your analysis] |
| Memory | ~1GB per 2K ctx | [XX]GB | [Your analysis] |

---

## Challenges & Solutions

### Challenge 1: [Description]
- **Issue**: [What went wrong]
- **Time Lost**: [Duration]
- **Solution**: [How you fixed it]
- **Learning**: [What you learned]

### Challenge 2: [Description]
- **Issue**: [What went wrong]
- **Time Lost**: [Duration]
- **Solution**: [How you fixed it]
- **Learning**: [What you learned]

---

## Code & Scripts

All benchmark scripts available in [`scripts/`](scripts/) directory:
- `basic_benchmark.py` - Sequential request benchmarking
- `advanced_benchmark.py` - Multi-scenario testing
- `visualize_results.py` - Visualization generation

Raw results in JSON format:
- [`benchmark_results_TIMESTAMP.json`](benchmark_results_TIMESTAMP.json)
- [`advanced_benchmark_TIMESTAMP.json`](advanced_benchmark_TIMESTAMP.json)

---

## Connection to Performance Engineering

As a performance engineer transitioning to AI infrastructure, Day 1 highlighted:

1. **Familiar Concepts**:
   - Latency percentiles (P50, P95, P99)
   - Throughput optimization
   - Resource monitoring and bottleneck analysis
   - [Your observations]

2. **New Concepts**:
   - Prefill vs decode phases
   - KV-cache management
   - Token-level metrics (TPS, TPOT)
   - [Your observations]

3. **Transferable Skills**:
   - Systematic benchmarking methodology
   - Metrics-driven optimization
   - Understanding system bottlenecks
   - [Your observations]

---

## Questions for Further Investigation

1. [Question 1]
2. [Question 2]
3. [Question 3]

---

## Next Steps

Tomorrow (Day 2): **Quantization & Model Optimization**
- Explore INT8, INT4 quantization
- Compare quantized vs full-precision performance
- Measure memory reduction benefits
- Analyze accuracy vs speed trade-offs

---

## Resources Used

### Primary Resources
- [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face KV-Cache Docs](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)

### Additional Resources
- [List additional resources you found helpful]

---

## Appendix

### Raw Benchmark Data

<details>
<summary>Click to expand full benchmark results</summary>

```json
{
  // Paste your benchmark JSON here for reference
}
```

</details>

### System Information

<details>
<summary>Click to expand system details</summary>

```bash
$ ollama --version
[version]

$ python --version
[version]

$ uname -a
[output]
```

</details>

---

**Total Time Invested**: [XX] hours
**Status**: ✅ Completed

---

*This is part of my 4-week AI performance engineering crash course. Follow along at [YOUR_REPO_URL]*
