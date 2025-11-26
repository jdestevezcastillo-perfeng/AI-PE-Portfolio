# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Performance Engineering Portfolio - A comprehensive study of LLM optimization, benchmarking, and serving. Focus areas: inference optimization, quantization trade-offs, distributed training, and observability.

**Context:** Owner is an Expert Performance Engineer (15 years at Reuters/Nordea) pivoting deep systems engineering expertise to AI/ML performance optimization.

## Environment

- **OS:** Linux Mint (Kernel 6.14+)
- **GPU:** AMD Radeon RX 6700 XT (12GB VRAM) with ROCm 6.3
- **Python:** 3.12 with `uv` package manager
- **Inference Server:** Ollama (local models on port 11434)

## Common Commands

```bash
# Virtual environment
source .venv/bin/activate

# Package management (use uv, not pip)
uv pip install <package>

# Run quantization benchmark (Module 01)
python 01_llm_architecture/lab_04_quantization_benchmark.py --model llama3.1:8b --requests 5

# GPU monitoring
rocm-smi                    # AMD GPU status
rocm-smi --showmeminfo vram # VRAM usage

# Observability stack (Module 07)
cd 07_observability
./start.sh                  # Start full PLG stack
./stop.sh                   # Stop all services
docker compose logs -f      # View logs

# Access monitoring
# Grafana:    http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Loki:       http://localhost:3100
# Tempo:      http://localhost:3200
```

## Code Architecture

### Module Structure
- `00_setup_guide/` - Environment setup (ROCm, Docker, uv)
- `01_llm_architecture/` - Transformer fundamentals, quantization benchmarking
- `02_inference_engines/` - (Planned) vLLM, TGI, Ollama comparisons
- `03_load_testing/` - (Planned) Locust, k6, GenAI-Perf
- `04_model_optimization/` - (Planned) TensorRT-LLM, torch.compile
- `05_peft_finetuning/` - (Planned) LoRA, QLoRA with Unsloth
- `06_distributed_training/` - (Planned) DDP, FSDP, DeepSpeed
- `07_observability/` - Production-grade Prometheus/Grafana/Loki/Tempo stack
- `08_evaluation/` - (Planned) LLM-as-Judge, RAG evaluation
- `09_capstone/` - (Planned) Enterprise RAG pipeline

### Key Files in Module 01
- `shared_model_config.py` - Centralized model/training configs (DEMO_CONFIG, GPT2_SMALL_CONFIG)
- `shared_gpt_model.py` - Reusable GPT model implementation
- `lab_*.py` - Hands-on exercises (numbered sequentially)
  - `lab_01_inspect_transformer.py` - Visualize GPT-2 architecture
  - `lab_02_train_gpt_demo.py` - Train mini-GPT on Shakespeare
  - `lab_03_train_gpt_parquet.py` - Train on structured datasets
  - `lab_04_quantization_benchmark.py` - Benchmark Ollama models with GPU telemetry + OpenTelemetry

### Observability Stack (Module 07)

**Architecture:**
```
Grafana (3000) ─┬─ Prometheus (9090) ← node-exporter, rocm-exporter, ollama-exporter
                ├─ Loki (3100) ← promtail (logs)
                └─ Tempo (3200) ← OTLP traces (4317)
```

**Custom Exporters:**
- `rocm_exporter.py` - AMD GPU metrics via rocm-smi (port 9102)
  - Temperature (edge/junction/memory), power, VRAM, clocks, fan speed
- `ollama_exporter.py` - LLM inference metrics (port 9103)
  - TPS, TTFT, TPOT, model VRAM, request counts
  - **Important:** Has `/record` endpoint for benchmark scripts to POST metrics

**Pre-configured Dashboards:**
- AI Performance - LLM inference metrics
- GPU Hardware - Temperature, VRAM, power, utilization
- System Hardware - CPU, memory, disk, network
- LLM Traces - OpenTelemetry spans

**Recording Metrics from Benchmarks:**
```python
import requests

# After each Ollama inference, POST to ollama-exporter
requests.post('http://localhost:9103/record', json={
    'eval_count': result['eval_count'],
    'eval_duration': result['eval_duration'],
    'prompt_eval_count': result.get('prompt_eval_count', 0),
    'prompt_eval_duration': result.get('prompt_eval_duration', 0)
})
```

**OpenTelemetry Tracing:**
```python
# Tempo accepts OTLP traces on localhost:4317
# Use in lab_04_quantization_benchmark.py and future scripts
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# endpoint="localhost:4317", insecure=True
```

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

| Metric | Description | Target |
|--------|-------------|--------|
| **TTFT** | Time To First Token (latency) | < 500ms |
| **TPOT** | Time Per Output Token (decoding speed) | < 50ms |
| **TPS** | Tokens Per Second (throughput) | > 30 |
| **VRAM** | GPU memory usage | Monitor for OOM |
| **GPU Junction Temp** | Hotspot temperature (critical) | < 90°C |

## Architecture Decisions

### ROCm GPU Paths
The system uses ROCm 6.3 at `/opt/rocm-6.3.0/bin/rocm-smi`. Exporters check multiple fallback paths:
- `/opt/rocm-6.3.0/bin/rocm-smi` (current)
- `/opt/rocm/bin/rocm-smi`
- `/usr/bin/rocm-smi`

### Docker Networking
Observability stack uses `host.docker.internal` to reach Ollama on the host machine (port 11434). Exporters must be in the `observability` network.

### Benchmark Output Format
Quantization benchmarks save results as JSON with timestamp: `quantization_benchmark_<model>_<timestamp>.json`

Traces are sent to Tempo and viewable in Grafana → Explore → Tempo.

## Dependencies

**Core ML Stack:**
- torch (PyTorch)
- transformers (Hugging Face)
- datasets (Hugging Face)
- tiktoken (tokenization)
- numpy

**Observability:**
- opentelemetry-api, opentelemetry-sdk
- opentelemetry-exporter-otlp-proto-grpc
- requests (for posting metrics to ollama-exporter)

**Monitoring Tools (Docker):**
- Prometheus, Grafana, Loki, Tempo, Promtail
- node-exporter (system metrics)
- Custom exporters: rocm-exporter, ollama-exporter

## Troubleshooting

### Ollama Not Accessible
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check if exporters can reach it
docker exec ollama-exporter curl http://host.docker.internal:11434/api/tags
```

### ROCm Exporter Errors
```bash
# Verify ROCm access from container
docker exec rocm-exporter /opt/rocm-6.3.0/bin/rocm-smi

# Check user permissions
sudo usermod -aG render,video $USER

# Check device access
ls -l /dev/kfd /dev/dri
```

### Grafana Shows "No Data"
1. Check Prometheus targets: http://localhost:9090/targets
2. Verify exporters: `docker compose ps`
3. Check datasource config in Grafana provisioning

### Virtual Environment
- Always use `.venv/bin/activate` before running Python scripts
- Use `uv` for package installation, not standard pip
