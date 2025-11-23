# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Performance Engineering Portfolio - A comprehensive study of LLM optimization, benchmarking, and serving. Focus areas: inference optimization, quantization trade-offs, distributed training, and observability.

## Environment

- **OS:** Linux (Mint, kernel 6.5+)
- **GPU:** AMD Radeon (ROCm) - use `rocm-smi` for GPU telemetry
- **Python:** 3.12 with `uv` package manager
- **Inference Server:** Ollama (local models)

## Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies with uv
uv pip install <package>

# Run quantization benchmark (Module 01)
python 01_llm_architecture/lab_04_quantization_benchmark.py --model llama3.1:8b --requests 5

# GPU monitoring
rocm-smi                    # AMD GPU status
rocm-smi --showmeminfo vram # VRAM usage
```

## Code Architecture

### Module Structure
- `00_setup_guide/` - Environment setup (ROCm, Docker, uv)
- `01_llm_architecture/` - Transformer fundamentals, quantization benchmarking
- `02_inference_engines/` - vLLM, TGI, Ollama comparisons
- `03_load_testing/` - Locust, k6, GenAI-Perf
- `04_model_optimization/` - TensorRT-LLM, torch.compile
- `05_peft_finetuning/` - LoRA, QLoRA with Unsloth
- `06_distributed_training/` - DDP, FSDP, DeepSpeed
- `07_observability/` - Prometheus/Grafana dashboards
- `08_evaluation/` - LLM-as-Judge, RAG evaluation
- `09_capstone/` - Enterprise RAG pipeline

### Key Files in Module 01
- `shared_model_config.py` - Centralized model/training configs (DEMO_CONFIG, GPT2_SMALL_CONFIG)
- `shared_gpt_model.py` - Reusable GPT model implementation
- `lab_*.py` - Hands-on exercises (numbered sequentially)

## Code Guidelines

### File Naming Convention
Format: `[Category]_[Number]_[Name].[Extension]` (number optional for doc/shared)

- `lab_XX_name.py` - Hands-on exercises
- `setup_XX_name.sh` - Setup scripts
- `doc_name.md` - Documentation
- `shared_name.py` - Reusable modules

### Script Structure Requirements

1. **ASCII diagram header** - Every script starts with a data flow/architecture diagram in docstring
2. **Numbered section headers** - Use block separators:
```python
# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
```

3. **Standard sections for training scripts:**
   1. Configuration & Setup
   2. Data Preparation (Loading/Tokenization)
   3. Data Loader (Batching)
   4. Model Initialization
   5. Training Loop
   6. Inference / Evaluation

### AIPE Performance Notes

Flag optimization opportunities with `>>> AIPE NOTE:`. Explain **what** is happening, **why** it matters, and **how** to optimize.

```python
# >>> AIPE NOTE: Mixed Precision (AMP)
# Standard training uses FP32. Use torch.amp.autocast for 2x speedup on Tensor Cores.
logits, loss = model(xb, yb)
```

**Keywords to flag:**
- **Hardware:** `device = 'cuda'` (Check for CPU fallback)
- **Data Loading:** `.to(device)` (Blocking transfers, memory pinning)
- **Memory:** `@torch.no_grad()` (Gradient overhead), `model.to(device)` (VRAM usage)
- **Compute:** `optimizer.step()` (Precision), `autocast` (Mixed Precision)
- **Quantization:** `int8` vs `fp16` trade-offs

### Coding Standards

- **Imports:** Group standard libs, third-party libs, then local modules
- **Type Hinting:** Use `typing` for function signatures
- **Config:** Use `shared_model_config.py` for hyperparameters - do not hardcode
- **Comments:** Focus on "Why", not "What" - avoid obvious inline comments

## Key Metrics

- **TTFT** - Time To First Token (latency)
- **TPOT** - Time Per Output Token (decoding speed)
- **TPS** - Tokens Per Second (throughput)
- **VRAM** - GPU memory usage
