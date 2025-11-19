# Module 02: Inference Engines

## 🎯 Objective
Compare the leading inference engines to understand their architecture, strengths, and weaknesses. Not all engines are created equal.

## 📚 Concepts
1.  **Continuous Batching:** How engines process multiple requests at different stages of generation simultaneously.
2.  **PagedAttention (vLLM):** Managing KV Cache like virtual memory pages to eliminate fragmentation.
3.  **Tensor Parallelism:** Splitting a model across multiple GPUs (if you had them).

## 🛠️ Tools to Master
- **vLLM:** The industry standard for high-throughput serving.
- **TGI (Text Generation Inference):** HuggingFace's production server (Rust-based).
- **Ollama:** The developer-friendly local wrapper (built on llama.cpp).

## 🧪 Lab: Engine Shootout
**Goal:** Benchmark vLLM vs Ollama.

### Steps:
1.  Set up vLLM to serve `Llama-3-8B`.
2.  Set up Ollama to serve `Llama-3-8B`.
3.  Send a **single** request to both. Record latency.
4.  Send **10 concurrent** requests to both. Record latency and throughput.
5.  **Observation:** Watch how vLLM maintains higher throughput under load compared to Ollama (which may queue requests depending on config).

## 📝 Deliverable
A "Battle Card" table comparing vLLM and Ollama on:
- Ease of Setup
- Single-stream Latency
- Max Concurrent Throughput
- VRAM Overhead
