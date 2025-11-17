AI Performance Engineering Portfolio (Single-GPU 3090/4090)
==========================================================

This repo is a 6-week crash-course playground for AI performance engineering on a single GPU (3090/4090). All code is Python 3.11, CPU-first with safe fallbacks; GPU hooks are marked with TODO blocks. Projects:
- project01_llm_benchmarks: synthetic + model-driven inference benchmarks (latency/throughput).
- project02_rag_perf: end-to-end RAG pipeline with eval scaffolding and performance hooks.
- project03_llm_observability: logging, tracing, metrics, and Grafana dashboard examples.
- project04_multigpu_inference: scaffolds for multi-GPU (or queued) inference; CPU fallbacks.
- common: shared utilities (logging, tracing, perf timers, synthetic data, exporters).

Quickstart
----------
1) Python 3.11 virtualenv: `python3.11 -m venv .venv && source .venv/bin/activate`
2) Install deps: `pip install -r requirements.txt` (CPU-safe; GPU extras TODO once hardware arrives).
3) Dry-run synthetic perf: `python common/data_gen.py --mode latency` or project scripts (see each README).
4) Metrics/obs demo: `python project03_llm_observability/observability_demo.py`.

TODO (GPU bring-up)
-------------------
- Enable CUDA builds of torch/transformers/flash-attn; pin to GPU architecture once detected.
- Swap CPU vector search to GPU (e.g., FAISS-GPU) in project02.
- Enable tensor parallel/multi-GPU in project04 with NCCL configs.
- Optional: integrate Triton Inference Server / vLLM for project01.

Repo conventions
----------------
- All GPU-specific code wrapped in try/except with CPU fallback and warnings.
- Configs live under each project’s `configs/`; adjust batch sizes/tokens there.
- Metrics via Prometheus (pushgateway/exporter) with example Grafana dashboards.
- Synthetic load generators allow testing before GPU is available.
