Project 02: RAG Performance
==========================

Goal: Implement an end-to-end RAG pipeline with performance and evaluation scaffolding. Defaults to CPU-safe FAISS; GPU TODOs marked.

Quickstart
----------
- Index synthetic docs: `python rag_pipeline.py index --config configs/default.yaml`
- Query: `python rag_pipeline.py query --config configs/default.yaml --q "What is chunk 0 about?"`
- Evaluate (synthetic): `python evals/eval_runner.py --config configs/default.yaml`

Files
-----
- `rag_pipeline.py`: indexing + retrieval + generation pipeline with metrics hooks.
- `configs/default.yaml`: CPU-safe defaults.
- `evals/eval_runner.py`: evaluation scaffold (latency + simple relevance proxy).

GPU TODOs
---------
- Swap FAISS CPU for FAISS GPU or other GPU vector DB.
- Replace CPU generator with GPU-accelerated LLM (e.g., vLLM, transformers w/ CUDA).
- Add batching and async querying for higher throughput.
