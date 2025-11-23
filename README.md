# AI Performance Engineering Portfolio

## About Me

I am an **Expert Performance Engineer** with **15 years of experience** optimizing high-scale financial systems at **Reuters (7 years)** and **Nordea (8 years)**.

Throughout my career, I have specialized in squeezing every millisecond of latency out of critical trading and banking infrastructure. Now, I am pivoting that deep expertise in systems engineering, concurrency, and observability to the frontier of **AI and Large Language Models (LLMs)**.

## Project Objective

This repository documents my journey mastering the **AI Performance Engineering** stack. It serves as a living lab where I apply rigorous engineering principles to modern ML systems.

This is not just a collection of tutorials; it is a **comprehensive engineering study** covering:

- **Inference Optimization:** Benchmarking vLLM vs. TGI vs. Ollama under high-concurrency loads
- **Quantization:** Measuring the precise trade-offs between model size (GGUF, AWQ, EXL2) and accuracy/perplexity
- **Distributed Systems:** Simulating multi-GPU training and serving architectures (FSDP, DeepSpeed)
- **Observability:** Building production-grade dashboards (Prometheus/Grafana) to track LLM-specific metrics like TTFT (Time To First Token) and TPOT (Time Per Output Token)

## Current Progress

| Module | Status | Description |
|--------|--------|-------------|
| 00 - Setup Guide | Done | Linux Mint, ROCm, Docker, Python uv |
| 01 - LLM Architecture | **Active** | Transformer fundamentals, quantization benchmarking |
| 07 - Observability | **Active** | Full PLG stack (Prometheus, Loki, Grafana, Tempo) |
| 02-06, 08-09 | Planned | Inference engines, load testing, fine-tuning, etc. |

## Environment

- **OS:** Linux Mint (Kernel 6.14+)
- **GPU:** AMD Radeon RX 6700 XT (12GB VRAM) with ROCm 6.3
- **Inference Server:** Ollama (local models)
- **Python:** 3.12 with **uv** package manager

## Module Details

### [Module 01: LLM Architecture & Quantization](./01_llm_architecture/README.md)

Hands-on exploration of transformer internals and quantization trade-offs.

**Labs Completed:**

- **lab_01_inspect_transformer.py** - Visualize GPT-2 architecture, attention patterns, parameter counts
- **lab_02_train_gpt_demo.py** - Train a mini-GPT from scratch on Shakespeare text
- **lab_03_train_gpt_parquet.py** - Train on structured parquet datasets
- **lab_04_quantization_benchmark.py** - Benchmark Ollama models with GPU telemetry and OpenTelemetry tracing

**Key Files:**

- **shared_gpt_model.py** - Reusable GPT model implementation
- **shared_model_config.py** - Centralized hyperparameter configs

### [Module 07: Observability](./07_observability/README.md)

Production-grade monitoring stack for AI/ML workloads.

**Stack:**

```bash
Grafana (3000) ─┬─ Prometheus (9090) ← node-exporter, rocm-exporter, ollama-exporter
                ├─ Loki (3100) ← promtail (logs)
                └─ Tempo (3200) ← OTLP traces (4317)
```

**Dashboards:**

| Dashboard | Metrics |
|-----------|---------|
| AI Performance | TPS, TTFT, TPOT, model VRAM, request counts |
| GPU Hardware | Temperature (edge/junction/memory), power, VRAM, clocks, fan speed |
| System Hardware | CPU, memory, disk I/O, network, system temperatures |
| LLM Traces | OpenTelemetry spans for inference requests |

**Custom Exporters:**

- **rocm_exporter.py** - AMD GPU metrics via rocm-smi
- **ollama_exporter.py** - LLM inference metrics with **/record** endpoint for benchmarks

**Quick Start:**

```bash
cd 07_observability
./start.sh
# Grafana: http://localhost:3000 (admin/admin)
```

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **TTFT** | Time To First Token | < 500ms |
| **TPOT** | Time Per Output Token | < 50ms |
| **TPS** | Tokens Per Second | > 30 |
| **VRAM** | GPU memory usage | Monitor for OOM |
| **GPU Temp** | Junction temperature | < 90°C |

## Repository Structure

```bash
AI-PE-Portfolio/
├── 00_setup_guide/          # Environment setup docs
├── 01_llm_architecture/     # Transformer labs, quantization benchmarks
│   ├── lab_*.py             # Hands-on exercises
│   ├── shared_*.py          # Reusable modules
│   └── doc_*.md             # Architecture diagrams, references
├── 02_inference_engines/    # (Planned) vLLM, TGI comparisons
├── 03_load_testing/         # (Planned) Locust, k6, GenAI-Perf
├── 04_model_optimization/   # (Planned) TensorRT-LLM, torch.compile
├── 05_peft_finetuning/      # (Planned) LoRA, QLoRA with Unsloth
├── 06_distributed_training/ # (Planned) DDP, FSDP, DeepSpeed
├── 07_observability/        # Prometheus/Grafana/Loki/Tempo stack
│   ├── docker-compose.yml   # Full observability stack
│   ├── exporters/           # Custom Prometheus exporters
│   ├── grafana/dashboards/  # Pre-configured dashboards
│   └── prometheus/          # Scrape configs
├── 08_evaluation/           # (Planned) LLM-as-Judge, RAG eval
├── 09_capstone/             # (Planned) Enterprise RAG pipeline
├── CLAUDE.md                # AI assistant instructions
├── GLOSSARY.md              # AI/ML terminology
└── CODE_GUIDELINES.md       # Script structure standards
```

## Running Benchmarks

```bash
# Activate environment
source .venv/bin/activate

# Run quantization benchmark with tracing
python 01_llm_architecture/lab_04_quantization_benchmark.py \
    --model llama3.1:8b \
    --requests 5

# Results saved to: quantization_benchmark_*.json
# Traces visible in: Grafana -> Explore -> Tempo
```

## Curriculum Roadmap

### Phase 1: Foundations (Current)

- [x] Module 00: Environment Setup
- [x] Module 01: Architecture & Quantization
- [ ] Module 02: Inference Engines (vLLM, TGI, Ollama)

### Phase 2: High-Performance Serving

- [ ] Module 03: Load Testing
- [ ] Module 04: Model Optimization

### Phase 3: Training & Fine-Tuning

- [ ] Module 05: PEFT & Fine-Tuning (LoRA, QLoRA)
- [ ] Module 06: Distributed Training (DDP, FSDP)

### Phase 4: MLOps & Observability

- [x] Module 07: Observability Stack
- [ ] Module 08: Evaluation Pipelines

### Phase 5: Capstone

- [ ] Module 09: Enterprise RAG Pipeline

---

*This repository is constantly evolving as I run new experiments and benchmarks.*
