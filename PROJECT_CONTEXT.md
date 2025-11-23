# Project Context: AI Performance Engineering Portfolio

## Overview
This project is a portfolio for AI Performance Engineering (AI-PE), focusing on optimizing, benchmarking, and serving LLMs.

## Current Active Task: Quantization Benchmarking
We are currently working on **Module 01: LLM Architecture & Quantization**.
We have created a script to benchmark local Ollama models to measure the impact of quantization on performance.

### Key Files
*   `01_llm_architecture/quantization_benchmark.py`: The main Python script.
    *   **Features:** Measures Latency, Tokens/Sec, Time-to-First-Token (TTFT).
    *   **Telemetry:** Captures GPU VRAM usage and Utilization (via `rocm-smi` or `nvidia-smi`).
    *   **Target:** Currently configured for `llama3.1:8b`.
*   `quantization_benchmark_llama3_1_8b_*.json`: Output results from previous runs.

### Recent Progress
1.  We successfully ran the benchmark against `llama3.1:8b`.
2.  We attempted to explore the Hugging Face `transformers` repo for inspiration but decided to focus on our own script for now.

### Immediate Goals
1.  Analyze the JSON results to see how the model performed.
2.  Run the benchmark against different quantization levels (e.g., `q4_0`, `q8_0`, `fp16`) to compare metrics.
3.  Visualize the trade-offs between **VRAM usage** vs. **Tokens/Sec**.

## Environment
*   **OS:** Linux
*   **Hardware:** GPU available (likely AMD based on `rocm-smi` checks in code, or NVIDIA).
*   **Tools:** `ollama`, `python3`, `rocm-smi`/`nvidia-smi`.
