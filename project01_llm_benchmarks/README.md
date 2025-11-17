Project 01: LLM Benchmarks
=========================

Goal: measure latency/throughput for LLM-style workloads on CPU/GPU. Includes synthetic benchmarks so you can start before GPU arrival. Swap configs to real models once ready.

Quickstart
----------
- Synthetic dry run: `python bench_runner.py --config configs/synthetic.yaml`
- Prometheus-enabled run: `python bench_runner.py --config configs/synthetic.yaml --exporter-port 9000`
- Replace model config under `configs/` to point to real HF model once GPU is available.

Files
-----
- `bench_runner.py`: orchestrates benchmarks, timing, and metrics.
- `configs/synthetic.yaml`: CPU-safe synthetic benchmark defaults.
- `scripts/`: place additional benchmark scripts (seeded with placeholder).

GPU TODOs
---------
- Enable CUDA/flash-attn paths in `load_model` for 3090/4090 (currently CPU).
- Consider batching + attention kernels; add tensor parallel for larger models.
