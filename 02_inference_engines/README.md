# Module 02: Inference Engines

## 🎯 Objective

Move beyond simple PyTorch execution and master high-performance inference engines designed for production. We will explore how specialized engines optimize memory management (PagedAttention), batching (Continuous Batching), and kernel execution to deliver orders of magnitude faster inference.

## 📚 Concepts

1. **PagedAttention (vLLM):** Solving the memory fragmentation problem in KV caches.
2. **Continuous Batching:** How to batch requests dynamically without waiting for all to finish.
3. **Tensor Parallelism:** Splitting large models across multiple GPUs (introductory).
4. **Speculative Decoding:** Using a small model to draft tokens for a large model.

## 🛠️ Tools to Master

- **vLLM:** The current state-of-the-art for high-throughput serving.
- **Text Generation Inference (TGI):** Hugging Face's production-ready inference server.
- **TensorRT-LLM:** NVIDIA's highly optimized inference library (optional/advanced).

## 🧪 Labs

### Lab 01: Baseline Inference

Establish a baseline using standard Hugging Face `transformers` pipelines. We need to know what "slow" looks like to appreciate "fast".

### Lab 02: vLLM Integration

Set up vLLM and run inference. We will measure Time to First Token (TTFT) and Inter-Token Latency (ITL).

### Lab 03: TGI Setup

Deploy TGI (via Docker) and interact with its API.

### Lab 04: Benchmarking Suite

A unified script to hammer our endpoints and generate a comparative report on throughput and latency.
