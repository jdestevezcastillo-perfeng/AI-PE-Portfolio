# AI Performance Engineering Portfolio

## 👋 About Me

I am an **Expert Performance Engineer** with **15 years of experience** optimizing high-scale financial systems at **Reuters (7 years)** and **Nordea (8 years)**.

Throughout my career, I have specialized in squeezing every millisecond of latency out of critical trading and banking infrastructure. Now, I am pivoting that deep expertise in systems engineering, concurrency, and observability to the frontier of **AI and Large Language Models (LLMs)**.

## 🚀 Project Objective

The goal of this repository is to document my journey mastering the **AI Performance Engineering** stack. It serves as a living lab where I apply rigorous engineering principles to modern ML systems.

This is not just a collection of tutorials; it is a **comprehensive engineering study** covering:

- **Inference Optimization:** Benchmarking vLLM vs. TGI vs. Ollama under high-concurrency loads.
- **Quantization:** Measuring the precise trade-offs between model size (GGUF, AWQ, EXL2) and accuracy/perplexity.
- **Distributed Systems:** Simulating multi-GPU training and serving architectures (FSDP, DeepSpeed).
- **Observability:** Building production-grade dashboards (Prometheus/Grafana) to track LLM-specific metrics like TTFT (Time To First Token) and TPOT (Time Per Output Token).

## 📚 Curriculum & Roadmap

I have designed a structured path to master these technologies, broken down into the following modules:

### [Phase 0: Environment Setup](./00_setup_guide/README.md)

**Goal:** Establish a reproducible, high-performance Linux baseline with ROCm/CUDA and Docker.

- **Module 00:** Setup Guide (Linux Mint, Kernel, Drivers, Python `uv`, Docker).

### Phase 1: Foundations of Inference

**Goal:** Understand how LLMs run and where the latency comes from.

- **[Module 01: Architecture & Quantization](./01_llm_architecture/README.md)**
  - **Concept:** Transformer blocks, KV Cache, Attention mechanisms.
  - **Optimization:** Quantization types (GGUF, AWQ, GPTQ, EXL2).
  - **Project:** "The Quantization Benchmark" (Compare FP16 vs Q4 vs Q8 latency & perplexity).
- **[Module 02: Inference Engines](./02_inference_engines/README.md)**
  - **Concept:** Continuous Batching, PagedAttention.
  - **Tools:** `vLLM`, `TGI`, `Ollama`.
  - **Project:** "Engine Shootout" (Benchmark vLLM vs Ollama on high-concurrency loads).

### Phase 2: High-Performance Serving

**Goal:** Scale from one user to thousands.

- **[Module 03: Load Testing](./03_load_testing/README.md)**
  - **Concept:** Throughput (Tokens/sec) vs Latency (TTFT).
  - **Tools:** `Locust`, `k6`, `GenAI-Perf`.
  - **Project:** Build a load generator that simulates realistic chat traffic patterns.
- **[Module 04: Model Optimization](./04_model_optimization/README.md)**
  - **Concept:** Graph capture, Kernel fusion.
  - **Tools:** `TensorRT-LLM`, `Torch.compile`.
  - **Project:** Compile a Llama 3 model for maximum throughput.

### Phase 3: Training & Fine-Tuning

**Goal:** Optimize the training loop for speed and memory.

- **[Module 05: PEFT & Fine-Tuning](./05_peft_finetuning/README.md)**
  - **Concept:** LoRA, QLoRA, DoRA.
  - **Tools:** `Unsloth` (Critical for speed), `HuggingFace PEFT`.
  - **Project:** Fine-tune Llama 3 on a custom dataset and measure training time vs memory usage.
- **[Module 06: Distributed Training](./06_distributed_training/README.md)**
  - **Concept:** Data Parallel (DDP), FSDP.
  - **Tools:** `DeepSpeed`, `FSDP`, `Ray Train`.
  - **Project:** Simulate a multi-GPU training run.

### Phase 4: MLOps & Observability

**Goal:** Monitor and debug AI in production.

- **[Module 07: Observability](./07_observability/README.md)**
  - **Concept:** LLM-specific metrics (Hallucination rate, Token usage).
  - **Tools:** `Prometheus`, `Grafana`, `OpenTelemetry`.
  - **Project:** Build the "Ultimate AI Dashboard".
- **[Module 08: Evaluation](./08_evaluation/README.md)**
  - **Concept:** LLM-as-a-Judge, RAG Evaluation.
  - **Tools:** `LangSmith`, `Arize Phoenix`, `Ragas`.
  - **Project:** Build an automated eval pipeline.

### [Phase 5: Capstone Project](./09_capstone/README.md)

**Enterprise Doc Search:** An end-to-end RAG pipeline demonstrating optimized serving, custom fine-tuning, and full observability.

---
*This repository is constantly evolving as I run new experiments and benchmarks. Feel free to explore the modules to see the code and results.*
