# Module 01: LLM Architecture & Quantization

## 🎯 Objective
Understand the fundamental building blocks of LLMs (Transformer architecture) and how reducing precision (Quantization) impacts performance vs. accuracy.

## 📚 Concepts
1.  **The Transformer Block:** Self-Attention, Feed-Forward Networks, and Layer Norm.
2.  **The KV Cache:** Why memory usage grows with context length.
3.  **Quantization Formats:**
    - **GGUF:** CPU/Apple Silicon optimized (llama.cpp).
    - **AWQ (Activation-aware Weight Quantization):** GPU optimized.
    - **GPTQ:** Older but widely supported GPU format.
    - **EXL2:** Fastest inference on modern GPUs (ExLlamaV2).

## 🛠️ Tools to Master
- **llama.cpp:** The "Swiss Army Knife" of local inference.
- **AutoGPTQ:** For running GPTQ models.
- **ExLlamaV2:** The speed king for consumer GPUs.

## 🧪 Lab: The Quantization Trade-off
**Goal:** Measure the exact latency and perplexity cost of quantization.

### Steps:
1.  Download `Llama-3-8B` in FP16 (Unquantized).
2.  Download `Llama-3-8B` in Q4_K_M (GGUF) and 4.0bpw (EXL2).
3.  Run the `quantization_benchmark.py` script (from Day 1) against all three.
4.  **Challenge:** Calculate the "Tokens per Watt" efficiency for each.

## 📝 Deliverable
A report comparing:
- VRAM Usage (GB)
- Throughput (Tokens/sec)
- Perplexity (Accuracy proxy)
