Project 04: Multi-GPU Inference (Scaffold)
==========================================

Goal: Provide a scaffold for multi-GPU or queued inference with telemetry. Uses CPU fallback; TODOs mark GPU-specific sections.

Quickstart
----------
- Dry run (CPU): `python multi_infer.py --config configs/default.yaml`
- Add multi-GPU configs once hardware present (NCCL, tensor parallel, model sharding).

Files
-----
- `multi_infer.py`: orchestrates batch inference across devices (CPU fallback).
- `configs/default.yaml`: starter config.

GPU TODOs
---------
- Implement real device pooling and tensor parallel (vLLM/DeepSpeed/FasterTransformers).
- Add CUDA stream management, pinned memory, and NCCL env wiring.
