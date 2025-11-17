Project 03: LLM Observability
============================

Goal: Demonstrate logging, tracing, and Prometheus metrics for LLM/RAG workloads with a ready-to-import Grafana dashboard.

Quickstart
----------
- Run demo server: `python observability_demo.py --port 8001`
- Scrape metrics at `/metrics`, view traces in console or OTLP endpoint if configured.
- Import `dashboards/llm_obs.json` into Grafana.

Files
-----
- `observability_demo.py`: FastAPI app with sample endpoints and instrumented spans/metrics.
- `dashboards/llm_obs.json`: Example Grafana panels for latency/throughput.

GPU TODOs
---------
- Add GPU utilization gauges once NVIDIA APIs are available.
- Instrument CUDA kernels in pipeline code paths for per-stage latency.
