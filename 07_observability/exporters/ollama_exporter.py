#!/usr/bin/env python3
"""
Ollama LLM Metrics Exporter for Prometheus
==========================================
Exports LLM inference metrics from Ollama API including:
- Server status
- Loaded models
- VRAM usage per model
- Inference performance (tokens/sec, TTFT, TPOT)
"""

import os
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

PORT = int(os.environ.get('PORT', 9103))
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# Store recent inference metrics (rolling window)
MAX_INFERENCE_SAMPLES = 100


@dataclass
class InferenceMetrics:
    """Stores inference performance metrics."""
    tokens_per_second: deque = field(default_factory=lambda: deque(maxlen=MAX_INFERENCE_SAMPLES))
    ttft_ms: deque = field(default_factory=lambda: deque(maxlen=MAX_INFERENCE_SAMPLES))  # Time to first token
    tpot_ms: deque = field(default_factory=lambda: deque(maxlen=MAX_INFERENCE_SAMPLES))  # Time per output token
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_tokens_prompted: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


inference_metrics = InferenceMetrics()


def fetch_json(url: str, timeout: int = 5) -> Optional[dict]:
    """Fetch JSON from URL."""
    try:
        req = Request(url, headers={'Accept': 'application/json'})
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except (URLError, json.JSONDecodeError, Exception):
        return None


def get_ollama_status() -> tuple[bool, list, list]:
    """Get Ollama server status and model info."""
    # Check if server is up
    tags = fetch_json(f'{OLLAMA_HOST}/api/tags')
    if tags is None:
        return False, [], []

    available_models = tags.get('models', [])

    # Get running models
    ps = fetch_json(f'{OLLAMA_HOST}/api/ps')
    running_models = ps.get('models', []) if ps else []

    return True, available_models, running_models


def get_prometheus_metrics() -> str:
    """Generate Prometheus metrics."""
    metrics = []

    # Get Ollama status
    is_up, available_models, running_models = get_ollama_status()

    # Server status
    metrics.append(f'ollama_up {1 if is_up else 0}')

    if is_up:
        # Model counts
        metrics.append(f'ollama_models_available {len(available_models)}')
        metrics.append(f'ollama_models_running {len(running_models)}')

        # Per-model metrics for available models
        for model in available_models:
            name = model.get('name', 'unknown')
            size = model.get('size', 0)
            # Sanitize model name for Prometheus label
            safe_name = name.replace(':', '_').replace('.', '_')
            metrics.append(f'ollama_model_size_bytes{{model="{safe_name}"}} {size}')

        # Per-model metrics for running models
        for model in running_models:
            name = model.get('name', 'unknown')
            safe_name = name.replace(':', '_').replace('.', '_')

            # VRAM usage
            vram = model.get('size_vram', 0)
            metrics.append(f'ollama_model_vram_bytes{{model="{safe_name}"}} {vram}')
            metrics.append(f'ollama_model_vram_gb{{model="{safe_name}"}} {vram / (1024**3):.3f}')

            # Model details
            details = model.get('details', {})
            params = details.get('parameter_size', 'unknown')
            quant = details.get('quantization_level', 'unknown')
            metrics.append(f'ollama_model_info{{model="{safe_name}",parameters="{params}",quantization="{quant}"}} 1')

    # Inference metrics from recorded samples
    with inference_metrics.lock:
        metrics.append(f'ollama_inference_total_requests {inference_metrics.total_requests}')
        metrics.append(f'ollama_inference_total_tokens_generated {inference_metrics.total_tokens_generated}')
        metrics.append(f'ollama_inference_total_tokens_prompted {inference_metrics.total_tokens_prompted}')

        if inference_metrics.tokens_per_second:
            tps_list = list(inference_metrics.tokens_per_second)
            metrics.append(f'ollama_inference_tokens_per_second_avg {sum(tps_list) / len(tps_list):.2f}')
            metrics.append(f'ollama_inference_tokens_per_second_max {max(tps_list):.2f}')
            metrics.append(f'ollama_inference_tokens_per_second_min {min(tps_list):.2f}')

        if inference_metrics.ttft_ms:
            ttft_list = list(inference_metrics.ttft_ms)
            metrics.append(f'ollama_inference_ttft_ms_avg {sum(ttft_list) / len(ttft_list):.2f}')
            metrics.append(f'ollama_inference_ttft_ms_p99 {sorted(ttft_list)[int(len(ttft_list) * 0.99)]:.2f}')

        if inference_metrics.tpot_ms:
            tpot_list = list(inference_metrics.tpot_ms)
            metrics.append(f'ollama_inference_tpot_ms_avg {sum(tpot_list) / len(tpot_list):.2f}')

    # Build output with HELP and TYPE
    output = [
        '# HELP ollama_up Ollama server status (1=up, 0=down)',
        '# TYPE ollama_up gauge',
        '# HELP ollama_models_available Number of available models',
        '# TYPE ollama_models_available gauge',
        '# HELP ollama_models_running Number of currently loaded models',
        '# TYPE ollama_models_running gauge',
        '# HELP ollama_model_size_bytes Model file size in bytes',
        '# TYPE ollama_model_size_bytes gauge',
        '# HELP ollama_model_vram_bytes Model VRAM usage in bytes',
        '# TYPE ollama_model_vram_bytes gauge',
        '# HELP ollama_model_vram_gb Model VRAM usage in GB',
        '# TYPE ollama_model_vram_gb gauge',
        '# HELP ollama_model_info Model metadata',
        '# TYPE ollama_model_info gauge',
        '# HELP ollama_inference_total_requests Total inference requests recorded',
        '# TYPE ollama_inference_total_requests counter',
        '# HELP ollama_inference_total_tokens_generated Total tokens generated',
        '# TYPE ollama_inference_total_tokens_generated counter',
        '# HELP ollama_inference_total_tokens_prompted Total prompt tokens processed',
        '# TYPE ollama_inference_total_tokens_prompted counter',
        '# HELP ollama_inference_tokens_per_second_avg Average tokens per second',
        '# TYPE ollama_inference_tokens_per_second_avg gauge',
        '# HELP ollama_inference_tokens_per_second_max Maximum tokens per second',
        '# TYPE ollama_inference_tokens_per_second_max gauge',
        '# HELP ollama_inference_tokens_per_second_min Minimum tokens per second',
        '# TYPE ollama_inference_tokens_per_second_min gauge',
        '# HELP ollama_inference_ttft_ms_avg Average time to first token in ms',
        '# TYPE ollama_inference_ttft_ms_avg gauge',
        '# HELP ollama_inference_ttft_ms_p99 P99 time to first token in ms',
        '# TYPE ollama_inference_ttft_ms_p99 gauge',
        '# HELP ollama_inference_tpot_ms_avg Average time per output token in ms',
        '# TYPE ollama_inference_tpot_ms_avg gauge',
        '',
    ]

    return '\n'.join(output + metrics) + '\n'


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = get_prometheus_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(metrics.encode('utf-8'))
        elif self.path == '/health' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Record inference metrics from benchmark scripts."""
        if self.path == '/record':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode('utf-8'))

                with inference_metrics.lock:
                    inference_metrics.total_requests += 1

                    # Tokens generated
                    eval_count = data.get('eval_count', 0)
                    inference_metrics.total_tokens_generated += eval_count

                    # Prompt tokens
                    prompt_count = data.get('prompt_eval_count', 0)
                    inference_metrics.total_tokens_prompted += prompt_count

                    # Tokens per second (from eval_duration in nanoseconds)
                    eval_duration_ns = data.get('eval_duration', 0)
                    if eval_duration_ns > 0 and eval_count > 0:
                        tps = eval_count / (eval_duration_ns / 1e9)
                        inference_metrics.tokens_per_second.append(tps)

                        # TPOT (time per output token)
                        tpot = (eval_duration_ns / 1e6) / eval_count  # ms per token
                        inference_metrics.tpot_ms.append(tpot)

                    # TTFT (time to first token, from prompt_eval_duration)
                    prompt_duration_ns = data.get('prompt_eval_duration', 0)
                    if prompt_duration_ns > 0:
                        ttft = prompt_duration_ns / 1e6  # Convert to ms
                        inference_metrics.ttft_ms.append(ttft)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')

            except (json.JSONDecodeError, Exception) as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    print(f'Ollama Exporter starting on port {PORT}')
    print(f'Ollama host: {OLLAMA_HOST}')
    print(f'POST /record to submit inference metrics from benchmarks')

    server = HTTPServer(('0.0.0.0', PORT), MetricsHandler)
    print(f'Ollama Exporter running on http://0.0.0.0:{PORT}/metrics')
    server.serve_forever()
