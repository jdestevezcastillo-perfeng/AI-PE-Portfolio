# AI Performance Engineering Observability Stack

Open-source observability stack for monitoring AI/ML workloads, GPU performance, and LLM inference.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Grafana (port 3000)                     │
│              Dashboards, Alerts, Visualization              │
└─────────────────┬───────────────────┬───────────────────────┘
                  │                   │
┌─────────────────▼─────┐   ┌─────────▼───────────────────────┐
│   Prometheus (9090)   │   │        Loki (3100)              │
│   Metrics storage     │   │      Log aggregation            │
└─────────────────┬─────┘   └─────────┬───────────────────────┘
                  │                   │
┌─────────────────▼─────────────────  ▼───────────────────────┐
│                    Exporters                                │
├─────────────────────────────────────────────────────────────┤
│ • node-exporter (9100)   - CPU, RAM, disk, network          │
│ • rocm-exporter (9102)   - AMD GPU metrics                  │
│ • ollama-exporter (9103) - LLM inference metrics            │
│ • promtail               - Log shipping                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- ROCm installed (for AMD GPU metrics)
- Ollama running on port 11434

### Start the Stack

```bash
cd /home/lostborion/AI-PE-Portfolio/observability

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### Access Services

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Loki**: http://localhost:3100

### Stop the Stack

```bash
docker compose down

# To also remove data volumes:
docker compose down -v
```

## Components

### Prometheus (Metrics)

Time-series database collecting metrics every 5 seconds:
- System metrics (CPU, RAM, disk, network)
- GPU metrics (temperature, utilization, VRAM, power)
- Ollama metrics (model status, inference stats)

### Grafana (Visualization)

Pre-configured dashboard "AI Performance Engineering" with:
- GPU temperature, utilization, power, VRAM gauges
- CPU and memory time series
- Ollama status and inference metrics

### Loki (Logs)

Log aggregation from:
- System logs (/var/log)
- Journald (systemd services)
- Ollama service logs

### Custom Exporters

#### ROCm Exporter (rocm_exporter.py)

Exports AMD GPU metrics from `rocm-smi`:
- `rocm_gpu_temperature_celsius`
- `rocm_gpu_utilization_percent`
- `rocm_gpu_power_watts`
- `rocm_gpu_vram_used_bytes`
- `rocm_gpu_sclk_mhz` / `rocm_gpu_mclk_mhz`

#### Ollama Exporter (ollama_exporter.py)

Exports LLM inference metrics:
- `ollama_up`
- `ollama_models_count`
- `ollama_running_models_count`
- `ollama_model_vram_bytes`
- `ollama_inference_tokens_per_second_avg`

## Usage During Benchmarks

### Viewing Metrics

1. Open Grafana at http://localhost:3000
2. Go to Dashboards → AI Performance Engineering
3. Set time range to "Last 15 minutes"
4. Enable auto-refresh (5s)

### Recording Inference Metrics

The Ollama exporter has a `/record` endpoint to track inference metrics from your benchmarks:

```python
import requests

# After each inference call, record the metrics
response = requests.post('http://localhost:9103/record', json={
    'eval_count': 100,  # tokens generated
    'eval_duration': 2000000000,  # nanoseconds
    'prompt_eval_count': 50
})
```

### Querying Prometheus

Example PromQL queries:

```promql
# Average GPU utilization over 5 minutes
avg_over_time(rocm_gpu_utilization_percent[5m])

# VRAM usage percentage
(rocm_gpu_vram_used_bytes / rocm_gpu_vram_total_bytes) * 100

# CPU usage
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

### Viewing Logs

In Grafana:
1. Go to Explore
2. Select Loki datasource
3. Query: `{job="ollama"}`

## Troubleshooting

### ROCm exporter not getting GPU metrics

Check that ROCm is accessible in the container:

```bash
docker exec rocm-exporter /opt/rocm-6.3.0/bin/rocm-smi
```

### Ollama exporter shows "ollama_up 0"

Ensure Ollama is running and accessible:

```bash
curl http://localhost:11434/api/tags
```

### Grafana dashboard shows "No Data"

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify exporters are running: `docker compose ps`
3. Check exporter logs: `docker compose logs rocm-exporter`

### Permission issues with Promtail

Promtail needs access to /var/log. If you see permission errors:

```bash
# Add read permissions
sudo chmod -R a+r /var/log
```

## Customization

### Adding New Metrics

1. Modify `exporters/rocm_exporter.py` or `exporters/ollama_exporter.py`
2. Rebuild: `docker compose build rocm-exporter ollama-exporter`
3. Restart: `docker compose up -d`

### Creating New Dashboards

1. Create dashboard in Grafana UI
2. Export as JSON
3. Save to `grafana/dashboards/`
4. Dashboard will auto-load on next restart

### Adjusting Scrape Interval

Edit `prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 1s  # More frequent for detailed analysis
```

## Resource Usage

Approximate memory usage:
- Prometheus: ~200MB
- Grafana: ~100MB
- Loki: ~100MB
- node-exporter: ~20MB
- Custom exporters: ~30MB each

Total: ~500MB RAM

## License

MIT - Use freely for your AI performance engineering work.
