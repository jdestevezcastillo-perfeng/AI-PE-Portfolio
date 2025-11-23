# Module 07: AI Observability

## Objective

You can't optimize what you can't measure. This module provides a complete observability stack for monitoring AI/ML workloads, GPU performance, and LLM inference.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Grafana (port 3000)                              │
│                 Dashboards, Alerts, Visualization                       │
│    ┌──────────────────┬──────────────────┬──────────────────┐          │
│    │ AI Performance   │ GPU Hardware     │ Logs Explorer    │          │
│    │ Dashboard        │ Dashboard        │ (Loki)           │          │
│    └──────────────────┴──────────────────┴──────────────────┘          │
└────────────┬─────────────────────┬─────────────────────┬───────────────┘
             │                     │                     │
┌────────────▼─────────┐ ┌────────▼─────────┐ ┌────────▼─────────┐
│  Prometheus (9090)   │ │   Loki (3100)    │ │  Tempo (3200)    │
│  Metrics storage     │ │  Log aggregation │ │  Trace storage   │
└──────────┬───────────┘ └────────┬─────────┘ └────────┬─────────┘
           │                      │                    │
┌──────────▼──────────────────────▼────────────────────▼─────────────────┐
│                           Exporters                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ node-exporter (9100)  │ CPU, RAM, disk, network                        │
│ rocm-exporter (9102)  │ AMD GPU: temps, VRAM, power, clocks, fan       │
│ ollama-exporter (9103)│ LLM: TPS, TTFT, TPOT, model VRAM               │
│ promtail              │ Log shipping to Loki                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- ROCm installed (for AMD GPU metrics)
- Ollama running on port 11434 (optional)

### Start the Stack

```bash
cd 07_observability

# Start all services
./start.sh

# Or manually:
docker compose up -d
```

### Access Services

| Service    | URL                    | Credentials |
|------------|------------------------|-------------|
| Grafana    | http://localhost:3000  | admin/admin |
| Prometheus | http://localhost:9090  | -           |
| Loki       | http://localhost:3100  | -           |
| Tempo      | http://localhost:3200  | -           |

### Stop the Stack

```bash
./stop.sh

# To also remove data volumes:
docker compose down -v
```

## Pre-configured Dashboards

### 1. AI Performance Engineering

Monitors LLM inference performance:
- **Service Status**: Ollama up/down, available/running models
- **Throughput**: Tokens/sec (avg, min, max)
- **Latency**: TTFT (Time to First Token), TPOT (Time per Output Token)
- **Token Counts**: Total requests, generated tokens, prompt tokens
- **Model VRAM**: Per-model GPU memory usage
- **System Resources**: CPU, RAM, GPU utilization

### 2. GPU Hardware Monitoring

Comprehensive AMD GPU monitoring:
- **Temperature**: Edge, Junction/Hotspot, Memory temps with thresholds
- **Utilization**: GPU compute %, Memory bandwidth %
- **Power**: Current draw in watts
- **VRAM**: Used/Total in GB and percentage
- **Clocks**: Graphics and memory clock frequencies
- **Fan & Voltage**: Fan speed %, voltage

## Custom Exporters

### ROCm Exporter (rocm_exporter.py)

Exports AMD GPU metrics from `rocm-smi`:

| Metric | Description |
|--------|-------------|
| `rocm_gpu_temperature_edge_celsius` | GPU edge temperature |
| `rocm_gpu_temperature_junction_celsius` | Junction/hotspot temp (critical) |
| `rocm_gpu_temperature_memory_celsius` | Memory temperature |
| `rocm_gpu_utilization_percent` | GPU compute utilization |
| `rocm_gpu_memory_utilization_percent` | Memory bandwidth utilization |
| `rocm_gpu_power_watts` | Power consumption |
| `rocm_gpu_vram_used_bytes` | VRAM used |
| `rocm_gpu_vram_used_percent` | VRAM usage % |
| `rocm_gpu_clock_graphics_mhz` | Graphics clock |
| `rocm_gpu_clock_memory_mhz` | Memory clock |
| `rocm_gpu_fan_speed_percent` | Fan speed |
| `rocm_gpu_voltage_mv` | GPU voltage |

### Ollama Exporter (ollama_exporter.py)

Exports LLM inference metrics:

| Metric | Description |
|--------|-------------|
| `ollama_up` | Server status (1=up) |
| `ollama_models_available` | Number of downloaded models |
| `ollama_models_running` | Currently loaded models |
| `ollama_model_vram_bytes` | Per-model VRAM usage |
| `ollama_inference_tokens_per_second_avg` | Average throughput |
| `ollama_inference_ttft_ms_avg` | Average time to first token |
| `ollama_inference_tpot_ms_avg` | Average time per output token |

## Recording Inference Metrics

The Ollama exporter has a `/record` endpoint for your benchmark scripts:

```python
import requests

# After each Ollama inference call, record the metrics
response = requests.post('http://localhost:9103/record', json={
    'eval_count': 100,              # tokens generated
    'eval_duration': 2000000000,    # nanoseconds
    'prompt_eval_count': 50,        # prompt tokens
    'prompt_eval_duration': 500000000  # nanoseconds (for TTFT)
})
```

Integration with `lab_04_quantization_benchmark.py`:

```python
# Add after each inference request
if 'eval_count' in result:
    requests.post('http://localhost:9103/record', json={
        'eval_count': result['eval_count'],
        'eval_duration': result['eval_duration'],
        'prompt_eval_count': result.get('prompt_eval_count', 0),
        'prompt_eval_duration': result.get('prompt_eval_duration', 0)
    })
```

## Sending Traces (OpenTelemetry)

Tempo accepts traces via OTLP:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("ai-benchmark")

# Use in your code
with tracer.start_as_current_span("inference") as span:
    span.set_attribute("llm.model", "llama3.1:8b")
    span.set_attribute("llm.prompt_tokens", 50)
    # ... do inference ...
    span.set_attribute("llm.completion_tokens", 100)
```

## PromQL Examples

```promql
# Average GPU utilization over 5 minutes
avg_over_time(rocm_gpu_utilization_percent[5m])

# VRAM usage percentage
rocm_gpu_vram_used_percent

# Tokens per second rate of change
rate(ollama_inference_total_tokens_generated[1m])

# CPU usage
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory used in GB
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1024^3
```

## Troubleshooting

### ROCm exporter shows errors

```bash
# Verify ROCm is accessible
docker exec rocm-exporter /opt/rocm/bin/rocm-smi

# Check exporter logs
docker compose logs rocm-exporter
```

### Ollama exporter shows ollama_up=0

```bash
# Ensure Ollama is running
curl http://localhost:11434/api/tags

# Check if the container can reach the host
docker exec ollama-exporter curl http://host.docker.internal:11434/api/tags
```

### Grafana shows "No Data"

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify exporters are healthy: `docker compose ps`
3. Check datasource configuration in Grafana

### Permission issues

```bash
# For promtail log access
sudo chmod -R a+r /var/log

# For ROCm access
sudo usermod -aG render,video $USER
```

## Resource Usage

Approximate memory:
- Prometheus: ~200MB
- Grafana: ~100MB
- Loki: ~100MB
- Tempo: ~100MB
- Exporters: ~30MB each

**Total: ~600MB RAM**

## The 3 Pillars of Observability

1. **Metrics (Prometheus)**: Numeric measurements over time - GPU temps, TPS, latency percentiles
2. **Logs (Loki)**: Event records - inference requests, errors, model loading
3. **Traces (Tempo)**: Request flow across services - end-to-end latency breakdown

## LLM-Specific Metrics to Watch

| Metric | Why It Matters |
|--------|----------------|
| **TTFT** | User-perceived responsiveness |
| **TPOT** | Streaming speed, inverse of TPS |
| **Tokens/sec** | Throughput capacity |
| **VRAM Usage** | Model size limits, batching capacity |
| **GPU Hotspot Temp** | Thermal throttling indicator |
| **Queue Depth** | Concurrency bottleneck |
