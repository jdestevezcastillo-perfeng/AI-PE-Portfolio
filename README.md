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

### [Phase 0: Environment Setup](./00_setup_guide/SETUP.md)
Establishing a reproducible, high-performance Linux baseline with ROCm/CUDA and Docker.

### Phase 1: Foundations of Inference
- **[Module 01: Architecture & Quantization](./01_llm_architecture/README.md)** - Understanding the Transformer block and memory hierarchy.
- **[Module 02: Inference Engines](./02_inference_engines/README.md)** - Comparative analysis of serving engines.

### Phase 2: High-Performance Serving
- **[Module 03: Load Testing](./03_load_testing/README.md)** - Stress testing endpoints to find saturation points.
- **[Module 04: Model Optimization](./04_model_optimization/README.md)** - Kernel fusion and graph compilation (TensorRT-LLM).

### Phase 3: Training & Fine-Tuning
- **[Module 05: PEFT & Fine-Tuning](./05_peft_finetuning/README.md)** - Efficient training with LoRA/QLoRA.
- **[Module 06: Distributed Training](./06_distributed_training/README.md)** - Scaling across GPUs with FSDP.

### Phase 4: MLOps & Observability
- **[Module 07: Observability](./07_observability/README.md)** - Full stack instrumentation.
- **[Module 08: Evaluation](./08_evaluation/README.md)** - Automated model quality gates (LLM-as-a-Judge).

### [Phase 5: Capstone Project](./09_capstone/README.md)
**Enterprise Doc Search:** An end-to-end RAG pipeline demonstrating optimized serving, custom fine-tuning, and full observability.

---
*This repository is constantly evolving as I run new experiments and benchmarks. Feel free to explore the modules to see the code and results.*
