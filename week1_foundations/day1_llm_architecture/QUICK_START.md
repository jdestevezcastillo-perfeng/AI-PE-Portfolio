# Day 1 Quick Start Guide

**Time commitment**: 16 hours
**Difficulty**: Beginner-friendly
**Prerequisites**: Basic Python, basic command line

---

## TL;DR - What You'll Do Today

1. Install Ollama and run your first LLM
2. Write Python scripts to benchmark LLM performance
3. Learn about transformer architecture and KV-cache
4. Collect and visualize performance metrics
5. Document your findings on GitHub

---

## Morning Session (8 hours)

### Hour 1-2: Setup ⚙️

**Goal**: Get everything installed and working

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama service (in separate terminal)
ollama serve

# 3. Pull model
ollama pull llama3.1:8b

# 4. Test it works
ollama run llama3.1:8b "What is AI?"

# 5. Setup Python environment
cd ~/AI-PE-Portfolio/week1_foundations/day1_llm_architecture
python3 -m venv venv
source venv/bin/activate
pip install ollama requests numpy pandas matplotlib seaborn psutil
```

**Success check**: Can you chat with the model interactively?

---

### Hour 3-6: Basic Benchmarking 📊

**Goal**: Measure how fast the LLM generates text

```bash
# Run basic benchmark
cd scripts/
python basic_benchmark.py

# This will:
# - Send 100 requests with different prompt lengths
# - Measure latency, throughput, TTFT
# - Save results to JSON
# - Take ~15-25 minutes
```

**While it runs**:
- Start reading "The Illustrated Transformer" (link in resources.md)
- Take notes on self-attention mechanism
- Sketch out the transformer architecture diagram

**After completion**:
```bash
# Visualize results
python visualize_results.py

# Check plots/ directory for graphs
ls plots/
```

**Key questions to answer**:
- What's the average latency?
- How many tokens/second does it generate?
- How does latency change with prompt length?

---

### Hour 7-8: Advanced Experiments 🔬

**Goal**: Test different scenarios

```bash
# Run advanced benchmark suite
python advanced_benchmark.py

# This will test:
# 1. Different prompt lengths
# 2. Different temperatures
# 3. Concurrent requests
# 4. Resource usage
# Takes ~30-40 minutes
```

**While it runs**:
- Continue reading about KV-cache
- Watch StatQuest attention mechanism video (15min)
- Review the quiz questions in study_guide.md

**After completion**:
```bash
# Visualize advanced results
python visualize_results.py advanced_benchmark_*.json
```

---

## Afternoon Session (5 hours)

### Hour 9-11: Deep Study 📚

**Goal**: Understand the theory behind what you just measured

#### Hour 9: Transformer Architecture
- Read: "The Illustrated Transformer" by Jay Alammar (full read)
- Focus: Self-attention, multi-head attention, position encodings
- Activity: Draw transformer architecture from memory
- Self-test: Explain self-attention to a rubber duck

#### Hour 10: KV-Cache Deep Dive
- Read: "The Illustrated GPT-2" by Jay Alammar
- Read: Hugging Face KV-cache documentation
- Focus: Why KV-cache exists, prefill vs decode
- Activity: Calculate KV-cache size for llama3.1:8b with 2048 tokens

#### Hour 11: Inference Optimization
- Read: vLLM paper (introduction + Section 3)
- Read: Lilian Weng's inference optimization blog
- Focus: TTFT, TPOT, memory bandwidth bottlenecks
- Activity: Take the self-assessment quiz

---

### Hour 12-13: Analysis & Documentation 📝

**Goal**: Make sense of your results

#### Analyze Results (1 hour)
1. Compare your metrics to expected values
   - Expected TPS on CPU: 5-20 tok/s
   - Your TPS: _____
   - Analysis: _____

2. Identify bottleneck
   - Check CPU usage in results
   - Check memory usage in results
   - Conclusion: CPU-bound or memory-bound?

3. Interesting findings
   - How much does prompt length affect latency?
   - Does temperature impact speed?
   - What's the relationship between TTFT and total latency?

#### Document (1 hour)
1. Copy GITHUB_README_TEMPLATE.md to README.md
2. Fill in your results
3. Add your analysis and insights
4. Include your best visualizations

---

### Hour 14: GitHub & Sharing 🌐

**Goal**: Create your portfolio piece

```bash
# Initialize git repo (if not already)
git init
git add .
git commit -m "Day 1: LLM architecture basics and first benchmarks

- Benchmarked llama3.1:8b on CPU
- Measured latency, throughput, TTFT across different scenarios
- Key finding: [your main insight]
- Learned about transformer architecture and KV-cache mechanism
"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/AI-PE-Portfolio.git
git push -u origin main
```

**Create a professional README**:
- Clear results tables
- Embedded visualizations
- Key learnings section
- Connection to performance engineering

---

## Evening Session (3 hours)

### Hour 15: Review & Consolidate 🔄

**Goal**: Solidify understanding

1. **Review your notes** (30min)
   - What did you learn about transformers?
   - What did you learn about KV-cache?
   - What surprised you?

2. **Answer quiz questions** (30min)
   - Check study_materials/quiz_answers.md after attempting
   - Review any topics you got wrong

3. **Write learning summary** (30min)
   - In your own words, explain:
     - How transformers work
     - Why KV-cache is necessary
     - What bottlenecks LLM inference
   - Post to your blog/LinkedIn/Twitter

4. **Identify gaps** (30min)
   - What concepts are still unclear?
   - What experiments would you like to try?
   - What questions do you have for Day 2?

---

### Hour 16: Preview & Plan 🗓️

**Goal**: Prepare for Day 2

1. **Review Day 1 accomplishments** (15min)
   - [ ] Completed all benchmarks
   - [ ] Understood transformer architecture
   - [ ] Documented findings on GitHub
   - [ ] Can explain KV-cache

2. **Preview Day 2 content** (30min)
   - Topic: Quantization & Model Optimization
   - Will learn: INT8, INT4, quantization techniques
   - Will measure: Memory reduction, speed improvement
   - Will compare: Full precision vs quantized models

3. **Set up for tomorrow** (15min)
   - Ensure you have 10GB+ free disk space
   - Bookmark quantization resources
   - Prepare questions about today's learning

4. **Rest & reflect** (remaining time)
   - You've done 15 hours of intense learning!
   - Get good sleep for Day 2
   - Celebrate your progress

---

## Success Criteria Checklist

By end of Day 1, you should have:

### Technical Skills
- [ ] Can run local LLM inference
- [ ] Can write Python benchmarking scripts
- [ ] Can measure latency and throughput
- [ ] Can create performance visualizations
- [ ] Can identify system bottlenecks

### Theoretical Knowledge
- [ ] Understand transformer self-attention
- [ ] Explain KV-cache purpose and trade-offs
- [ ] Know the difference between prefill and decode
- [ ] Understand TTFT, TPOT, TPS metrics
- [ ] Can explain why inference is memory-bound

### Deliverables
- [ ] Benchmark results (JSON files)
- [ ] Visualization plots (PNG files)
- [ ] GitHub repository with documentation
- [ ] Learning notes and insights
- [ ] Completed self-assessment quiz

### Meta Skills
- [ ] Know how to troubleshoot LLM inference issues
- [ ] Can find and use relevant resources
- [ ] Developed systematic benchmarking methodology
- [ ] Connected concepts to performance engineering background

---

## If You're Running Behind

**Don't panic!** This is an aggressive schedule. Adjust as needed:

### Minimum Viable Day 1 (10 hours):
1. Setup + Basic benchmark (3h)
2. Study transformer basics (2h)
3. Study KV-cache basics (1h)
4. Analyze and document results (2h)
5. GitHub push (1h)
6. Review and quiz (1h)

Skip advanced benchmarks - can do later or Day 2.

### If You Have Extra Time:
1. Watch Andrej Karpathy's "Let's Build GPT" video
2. Try the interactive transformer visualizers
3. Run benchmarks with different models (phi3:mini, tinyllama)
4. Read FlashAttention paper
5. Experiment with different prompt types

---

## Common Questions

**Q: Do I need a GPU?**
A: No, CPU is fine for Day 1. GPU makes things faster but not required.

**Q: How fast should inference be?**
A: On CPU with llama3.1:8b, expect 5-20 tokens/second. Faster is fine!

**Q: My results differ from expected values. Is that OK?**
A: Yes! Every system is different. Focus on understanding patterns.

**Q: I don't understand transformers yet. Should I continue?**
A: Yes, understanding deepens with practice. Re-read resources on Day 2.

**Q: Can I use a different model?**
A: Yes, but llama3.1:8b is recommended for consistency.

**Q: How much should I understand on Day 1?**
A: You don't need to master everything. Basic understanding is enough.

---

## Emergency Contacts

**Stuck?** Check these:

1. **troubleshooting.md** - Common issues and solutions
2. **resources.md** - All reference materials
3. **study_materials/study_guide.md** - Detailed learning guide

**Still stuck?**
- Ollama Discord: https://discord.gg/ollama
- r/LocalLLaMA: https://reddit.com/r/LocalLLaMA
- GitHub Issues: Open an issue on your repo

---

## Motivation

You're about to spend 16 hours learning. Here's why it's worth it:

- **Career pivot**: LLM performance engineering is in high demand
- **Transferable skills**: Your perf eng background gives you an edge
- **Hands-on**: By end of day, you'll have real benchmark data
- **Portfolio**: GitHub repo proves your skills to employers
- **Foundation**: Today builds the base for advanced topics

**You've got this!** 🚀

---

## Final Checklist

Before you start:
- [ ] Read this entire guide
- [ ] Ensure you have 16 hours available (or adjust schedule)
- [ ] Have 10GB+ free disk space
- [ ] Have stable internet connection
- [ ] Have notebook for handwritten notes
- [ ] Minimized distractions
- [ ] Mindset: Curiosity > Perfection

**Ready? Let's begin!**

Start with setup_instructions.md →
