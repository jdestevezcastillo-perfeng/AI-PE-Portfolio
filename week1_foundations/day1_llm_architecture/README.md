# Day 1: LLM Architecture + First Benchmarks

**Learning Path**: AI Performance Engineering Crash Course
**Week**: 1 - Foundations
**Day**: 1 of 28
**Time Commitment**: 16 hours (10h hands-on, 3h study, 2h documentation, 1h breaks)

---

## 📋 Overview

Welcome to Day 1 of your AI performance engineering journey! Today you'll:

1. **Install and run** your first local LLM (Llama 3.1 8B)
2. **Benchmark** LLM inference performance with Python scripts
3. **Learn** transformer architecture and KV-cache fundamentals
4. **Measure** latency, throughput, and identify bottlenecks
5. **Document** your findings in a professional GitHub repository

By end of day, you'll have real performance data and deep understanding of how LLMs work under the hood.

---

## 🎯 Learning Objectives

### Technical Skills
- [ ] Run local LLM inference with Ollama
- [ ] Write Python benchmarking scripts from scratch
- [ ] Measure and analyze performance metrics
- [ ] Create data visualizations
- [ ] Identify system bottlenecks

### Theoretical Knowledge
- [ ] Understand transformer self-attention mechanism
- [ ] Explain KV-cache purpose and trade-offs
- [ ] Differentiate between prefill and decode phases
- [ ] Know key performance metrics (TTFT, TPOT, TPS)
- [ ] Understand why LLM inference is memory-bound

### Deliverables
- [ ] Benchmark results (JSON format)
- [ ] Performance visualizations (charts/graphs)
- [ ] GitHub repository with comprehensive documentation
- [ ] Personal learning notes and insights

---

## 📁 Repository Structure

```
day1_llm_architecture/
├── README.md                          # This file - start here
├── QUICK_START.md                     # Hour-by-hour schedule
├── setup_instructions.md              # Detailed setup guide
├── troubleshooting.md                 # Common issues & solutions
├── resources.md                       # All learning resources
│
├── scripts/                           # Benchmarking code
│   ├── README.md                      # Script documentation
│   ├── basic_benchmark.py             # Sequential benchmarks
│   ├── advanced_benchmark.py          # Multi-scenario tests
│   └── visualize_results.py           # Create plots
│
├── study_materials/                   # Learning guides
│   ├── study_guide.md                 # 3-hour study plan
│   └── quiz_answers.md                # Self-assessment answers
│
├── documentation_templates/           # Templates for your work
│   ├── DAILY_LOG_TEMPLATE.md          # Track your progress
│   └── GITHUB_README_TEMPLATE.md      # Document your results
│
├── plots/                             # Generated visualizations
│   └── (created after running benchmarks)
│
└── results/                           # Benchmark outputs
    └── (created after running benchmarks)
```

---

## 🚀 Quick Start

### Option 1: Guided Step-by-Step
**Best for**: First-time learners, want detailed guidance

1. Read [`QUICK_START.md`](QUICK_START.md) - Hour-by-hour schedule
2. Follow [`setup_instructions.md`](setup_instructions.md) - Get everything installed
3. Run benchmarks using [`scripts/README.md`](scripts/README.md)
4. Study using [`study_materials/study_guide.md`](study_materials/study_guide.md)
5. Document using [`documentation_templates/`](documentation_templates/)

### Option 2: Express Path
**Best for**: Experienced engineers, want minimum viable learning

```bash
# 1. Setup (30 min)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.1:8b

# 2. Python environment
python3 -m venv venv
source venv/bin/activate
pip install ollama requests numpy pandas matplotlib seaborn psutil

# 3. Run benchmark (20 min runtime)
cd scripts/
python basic_benchmark.py

# 4. Visualize
python visualize_results.py

# 5. Study (2-3 hours)
# Read: https://jalammar.github.io/illustrated-transformer/
# Read: https://jalammar.github.io/illustrated-gpt2/

# 6. Document your results
# Use templates in documentation_templates/
```

---

## 📚 What You'll Learn

### Hour-by-Hour Breakdown

#### Morning: Hands-on (6 hours)
- **Hour 1-2**: Setup Ollama, install dependencies, test inference
- **Hour 3-6**: Run basic benchmarks, collect performance data

#### Midday: Advanced Practice (4 hours)
- **Hour 7-8**: Advanced experiments (prompt length, concurrency, etc.)
- **Hour 9-10**: Data analysis and visualization

#### Afternoon: Study (3 hours)
- **Hour 11**: Transformer architecture deep dive
- **Hour 12**: KV-cache mechanism
- **Hour 13**: LLM inference optimization techniques

#### Evening: Documentation (3 hours)
- **Hour 14-15**: Analyze results, write findings
- **Hour 16**: Review, consolidate learning, prepare for Day 2

*Detailed schedule in [`QUICK_START.md`](QUICK_START.md)*

---

## 🔧 Prerequisites

### Required
- **Basic Python**: Can write scripts, use libraries
- **Command line**: Comfortable with terminal
- **Git basics**: Clone, commit, push
- **Performance engineering**: Helps but not required

### System Requirements
- **OS**: Linux, macOS, or WSL2 on Windows
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **CPU**: Any modern multi-core CPU (GPU optional)

### Recommended Background Reading
- None! We start from fundamentals

---

## 📊 Expected Results

### Performance Metrics (CPU with llama3.1:8b)

Typical results you should see:

| Metric | Expected Range | What It Means |
|--------|----------------|---------------|
| **Latency** | 2-15 seconds | End-to-end request time |
| **TPS** (CPU) | 5-20 tok/s | Token generation speed |
| **TTFT** | 0.1-2 seconds | Time to first token |
| **P95 Latency** | 1.2-1.5x mean | Tail latency |

*Your results may vary based on hardware. GPU will be significantly faster (50-200+ tok/s)*

### Deliverables You'll Create

1. **Benchmark Data**:
   - 100+ latency measurements per prompt type
   - Throughput metrics across scenarios
   - Resource usage data (CPU, memory)

2. **Visualizations**:
   - Latency distribution plots
   - Tokens per second comparisons
   - TTFT analysis charts
   - Prompt length impact graphs

3. **Documentation**:
   - Professional GitHub README
   - Learning notes and insights
   - Bottleneck analysis
   - Connection to performance engineering principles

---

## 🎓 Key Concepts

### Transformer Architecture
**What**: Neural network architecture that powers modern LLMs
**Key Innovation**: Self-attention mechanism for parallel processing
**Components**: Multi-head attention, feed-forward networks, position encodings
**Why It Matters**: Foundation of GPT, LLaMA, and all modern LLMs

### KV-Cache
**What**: Caching mechanism for Key and Value matrices during inference
**Problem Solved**: Prevents O(n²) redundant computation
**Trade-off**: Uses more memory for 5-10x speed improvement
**Impact**: Industry-standard technique, critical for efficient inference

### Performance Metrics

- **TTFT** (Time to First Token): User-perceived latency, prefill phase duration
- **TPOT** (Time Per Output Token): Decode phase per-token time
- **TPS** (Tokens Per Second): Throughput metric for generation speed
- **Latency**: End-to-end request time (TTFT + decode time)
- **Throughput**: Requests or tokens processed per second

---

## 🎯 Success Criteria

You've successfully completed Day 1 if you can:

### Demonstrate
- ✅ Run LLM inference locally
- ✅ Measure latency and throughput
- ✅ Create performance visualizations
- ✅ Identify system bottlenecks

### Explain
- ✅ How self-attention works in transformers
- ✅ Why KV-cache is necessary for efficient inference
- ✅ The difference between prefill and decode phases
- ✅ Why LLM inference is memory-bound, not compute-bound

### Deliver
- ✅ GitHub repository with benchmark results
- ✅ Performance analysis and insights
- ✅ Learning documentation
- ✅ Visualizations and data

---

## 🆘 Getting Help

### Self-Service Resources (Check First)
1. [`troubleshooting.md`](troubleshooting.md) - Common issues and solutions
2. [`resources.md`](resources.md) - All reference materials
3. [`study_materials/study_guide.md`](study_materials/study_guide.md) - Detailed learning content
4. Script documentation in [`scripts/README.md`](scripts/README.md)

### Community Support
- **Ollama Discord**: https://discord.gg/ollama
- **Reddit r/LocalLLaMA**: https://reddit.com/r/LocalLLaMA
- **Hugging Face Forums**: https://discuss.huggingface.co/

### Common Issues
- **"ollama: command not found"** → Restart terminal or add to PATH
- **"Connection refused"** → Run `ollama serve` in separate terminal
- **"Out of memory"** → Close apps or use smaller model (phi3:mini)
- **Slow performance** → Expected on CPU, verify no background tasks

*Full troubleshooting guide in [`troubleshooting.md`](troubleshooting.md)*

---

## 🔗 Essential Resources

### Must-Read (Included in 3-hour study time)
1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
2. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Jay Alammar
3. [KV-Cache Documentation](https://huggingface.co/docs/transformers/main/en/kv_cache) - Hugging Face
4. [LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) - Lilian Weng

### Recommended (Optional but valuable)
- Video: [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy
- Interactive: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- Paper: [vLLM - PagedAttention](https://arxiv.org/abs/2309.06180)

*Complete resource list in [`resources.md`](resources.md)*

---

## 🔄 Connection to Performance Engineering

As a performance engineer, Day 1 builds on your existing skills:

### Familiar Concepts
- **Latency percentiles** (P50, P95, P99) - Same as web services
- **Throughput optimization** - Maximizing requests/tokens per second
- **Bottleneck analysis** - CPU vs memory vs I/O bound
- **Benchmarking methodology** - Systematic measurement and analysis

### New Concepts (AI-Specific)
- **Prefill vs Decode** - Two distinct phases in LLM inference
- **KV-cache management** - Trading memory for compute efficiency
- **Token-level metrics** - TPS, TPOT instead of request-level only
- **Model architecture impact** - How transformer design affects performance

### Transferable Skills
- Your experience with **latency analysis** directly applies
- **Resource monitoring** skills transfer to GPU/memory profiling
- **Optimization mindset** is crucial for LLM inference tuning
- **Metrics-driven approach** is the same methodology

---

## 📈 Next Steps

### After Completing Day 1

1. **Consolidate Learning**:
   - Review your notes
   - Complete self-assessment quiz
   - Identify knowledge gaps

2. **Document & Share**:
   - Push to GitHub with comprehensive README
   - Share insights on LinkedIn/Twitter
   - Write a blog post (optional)

3. **Prepare for Day 2**:
   - **Topic**: Quantization & Model Optimization
   - **Preview**: INT8, INT4, memory reduction
   - **Prep**: Ensure 10GB+ free disk space

### Day 2 Preview: Quantization
- Learn about INT8 and INT4 quantization techniques
- Compare quantized vs full-precision performance
- Measure memory reduction benefits
- Analyze accuracy vs speed trade-offs
- Benchmark 4-bit vs 8-bit vs 16-bit models

---

## 💡 Tips for Success

### Time Management
- **Follow the schedule** in QUICK_START.md but adjust as needed
- **Don't rush** - understanding > completion
- **Take breaks** - 16 hours is intense, pace yourself
- **Prioritize** - If behind, skip advanced benchmarks (can do later)

### Learning Strategy
- **Hands-on first** - Run code before deep study
- **Visual learning** - Use interactive tools and diagrams
- **Active recall** - Explain concepts out loud
- **Connect to prior knowledge** - Relate to your perf eng experience

### Documentation
- **Take notes** throughout the day
- **Screenshot interesting results**
- **Document surprises** - What didn't match expectations?
- **Write for future you** - You'll reference this later

### Debugging
- **Read error messages** carefully
- **Check troubleshooting.md** before googling
- **Test components separately** - Isolate issues
- **Ask for help** when truly stuck (after trying solutions)

---

## 🎊 Celebrate Your Progress

Completing Day 1 is a significant achievement! You've:
- ✅ Set up a complete LLM inference environment
- ✅ Written production-quality benchmarking code
- ✅ Learned cutting-edge transformer architecture
- ✅ Collected real performance data
- ✅ Created a professional portfolio piece

**This is just the beginning. You're on your way to becoming an AI performance engineer!**

---

## 📝 License & Attribution

This learning material is created for educational purposes. All external resources and tools belong to their respective owners:
- Ollama: https://ollama.ai/
- LLaMA models: Meta AI
- Python libraries: Various open-source projects

---

## 🤝 Contribution

Found an issue? Have suggestions?
- Open an issue on your fork
- Improve the materials for future learners
- Share your enhanced version

---

**Ready to begin?** → Start with [`QUICK_START.md`](QUICK_START.md)

**Questions?** → Check [`troubleshooting.md`](troubleshooting.md) and [`resources.md`](resources.md)

**Let's learn!** 🚀

---

*Part of the 4-Week AI Performance Engineering Crash Course*
*Day 1 of 28 | Week 1: Foundations*
