# AI Performance Engineering Curriculum
**Target Audience:** Senior Performance Engineer (15y exp) pivoting to AI/ML.
**Goal:** Master the tools, concepts, and methodologies for optimizing Large Language Model (LLM) systems.

## Phase 0: Environment & Setup
**Goal:** Establish a reproducible, high-performance baseline environment.
- **Module 00:** [Setup Guide](./00_setup_guide/README.md)
    - Linux Mint Optimization (Kernel, Drivers).
    - ROCm/CUDA Installation & Verification.
    - Python Environment Strategy (`uv` vs `conda` vs `venv`).
    - Docker & NVIDIA Container Toolkit / ROCm Container setup.

## Phase 1: Foundations of Inference
**Goal:** Understand how LLMs run and where the latency comes from.
- **Module 01:** [LLM Architecture & Quantization](./01_llm_architecture/README.md)
    - **Concept:** Transformer blocks, KV Cache, Attention mechanisms.
    - **Optimization:** Quantization types (GGUF, AWQ, GPTQ, EXL2).
    - **Tool:** `llama.cpp`, `AutoGPTQ`, `ExLlamaV2`.
    - **Project:** "The Quantization Benchmark" (Compare FP16 vs Q4 vs Q8 latency & perplexity).

- **Module 02:** [Inference Engines](./02_inference_engines/README.md)
    - **Concept:** Continuous Batching, PagedAttention.
    - **Tools:** `vLLM`, `TGI` (Text Generation Inference), `Ollama`.
    - **Project:** "Engine Shootout" (Benchmark vLLM vs Ollama on high-concurrency loads).

## Phase 2: High-Performance Serving
**Goal:** Scale from one user to thousands.
- **Module 03:** [Load Testing & Concurrency](./03_load_testing/README.md)
    - **Concept:** Throughput (Tokens/sec) vs Latency (TTFT).
    - **Tools:** `Locust`, `k6`, `GenAI-Perf` (Triton).
    - **Project:** Build a load generator that simulates realistic chat traffic patterns.

- **Module 04:** [Model Compilation & Optimization](./04_model_optimization/README.md)
    - **Concept:** Graph capture, Kernel fusion.
    - **Tools:** `TensorRT-LLM` (NVIDIA) / `MIGraphX` (AMD), `Torch.compile`.
    - **Project:** Compile a Llama 3 model for maximum throughput.

## Phase 3: Training & Fine-Tuning Performance
**Goal:** Optimize the training loop for speed and memory.
- **Module 05:** [Parameter Efficient Fine-Tuning (PEFT)](./05_peft_finetuning/README.md)
    - **Concept:** LoRA, QLoRA, DoRA.
    - **Tools:** `Unsloth` (Critical for speed), `HuggingFace PEFT`.
    - **Project:** Fine-tune Llama 3 on a custom dataset and measure training time vs memory usage.

- **Module 06:** [Distributed Training](./06_distributed_training/README.md)
    - **Concept:** Data Parallel (DDP), Model Parallel (Pipeline/Tensor), FSDP.
    - **Tools:** `DeepSpeed`, `FSDP`, `Ray Train`.
    - **Project:** Simulate a multi-GPU training run (even on single GPU via gradient accumulation) to understand the config.

## Phase 4: MLOps & Observability
**Goal:** Monitor and debug AI in production.
- **Module 07:** [AI Observability](./07_observability/README.md)
    - **Concept:** LLM-specific metrics (Hallucination rate, Token usage).
    - **Tools:** `Prometheus` + `Grafana` (Custom Exporters), `OpenTelemetry`.
    - **Project:** Build the "Ultimate AI Dashboard" (System metrics + Model metrics).

- **Module 08:** [Evaluation & Tracing](./08_evaluation/README.md)
    - **Concept:** LLM-as-a-Judge, RAG Evaluation.
    - **Tools:** `LangSmith`, `Arize Phoenix`, `Ragas`.
    - **Project:** Build an automated eval pipeline for your fine-tuned model.

## Phase 5: Capstone
- **Module 09:** [The Final Project](./09_capstone/README.md)
    - **Goal:** End-to-end RAG pipeline with optimized serving, custom fine-tuning, and full observability.

---
## Glossary
[Link to Glossary](./GLOSSARY.md)
