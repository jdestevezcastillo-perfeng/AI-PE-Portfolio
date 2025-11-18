#!/usr/bin/env python3
"""
Ollama Metrics Exporter for Prometheus
Exports LLM inference metrics from Ollama API
"""

import os
import time
import requests
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque
import threading

PORT = 9103
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
PROBE_INTERVAL = 30  # seconds between probe requests

# Store recent inference metrics
inference_history = deque(maxlen=100)
lock = threading.Lock()

# Track cumulative stats
cumulative_stats = {
    'total_requests': 0,
    'total_tokens': 0,
    'total_prompt_tokens': 0,
    'total_duration_ns': 0,
    'last_tps': 0,
    'last_ttft_ms': 0,
}


def probe_ollama():
    """Send a small test request to Ollama to measure current performance."""
    global cumulative_stats

    while True:
        try:
            # Get the currently loaded model
            ps_response = requests.get(f'{OLLAMA_HOST}/api/ps', timeout=5)
            if ps_response.status_code != 200:
                time.sleep(PROBE_INTERVAL)
                continue

            ps_data = ps_response.json()
            running_models = ps_data.get('models', [])

            if not running_models:
                time.sleep(PROBE_INTERVAL)
                continue

            model_name = running_models[0].get('name', 'llama3.1:8b')

            # Send a small probe request
            probe_start = time.time()
            response = requests.post(
                f'{OLLAMA_HOST}/api/generate',
                json={
                    'model': model_name,
                    'prompt': 'Hi',
                    'stream': False,
                    'options': {
                        'num_predict': 10  # Very short response
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                eval_count = data.get('eval_count', 0)
                eval_duration = data.get('eval_duration', 1)
                prompt_eval_duration = data.get('prompt_eval_duration', 0)
                prompt_eval_count = data.get('prompt_eval_count', 0)

                # Calculate metrics
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1e9)
                else:
                    tps = 0

                ttft_ms = prompt_eval_duration / 1e6  # Convert ns to ms

                # Update cumulative stats
                with lock:
                    cumulative_stats['total_requests'] += 1
                    cumulative_stats['total_tokens'] += eval_count
                    cumulative_stats['total_prompt_tokens'] += prompt_eval_count
                    cumulative_stats['total_duration_ns'] += eval_duration
                    cumulative_stats['last_tps'] = tps
                    cumulative_stats['last_ttft_ms'] = ttft_ms

                    # Also add to history for averaging
                    inference_history.append({
                        'eval_count': eval_count,
                        'eval_duration': eval_duration,
                        'prompt_eval_count': prompt_eval_count,
                        'prompt_eval_duration': prompt_eval_duration,
                        'timestamp': time.time()
                    })

        except Exception as e:
            pass  # Silently continue on errors

        time.sleep(PROBE_INTERVAL)


def get_ollama_metrics():
    """Fetch metrics from Ollama and return Prometheus-formatted output."""
    metrics = []

    try:
        # Check if Ollama is running
        response = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
        if response.status_code == 200:
            metrics.append('ollama_up 1')

            # Get loaded models
            data = response.json()
            models = data.get('models', [])
            metrics.append(f'ollama_models_count {len(models)}')

            for model in models:
                name = model.get('name', 'unknown').replace('"', '\\"')
                size_bytes = model.get('size', 0)
                size_gb = size_bytes / (1024**3)
                metrics.append(f'ollama_model_size_bytes{{model="{name}"}} {size_bytes}')
                metrics.append(f'ollama_model_size_gb{{model="{name}"}} {size_gb:.2f}')
        else:
            metrics.append('ollama_up 0')

        # Get running models
        try:
            ps_response = requests.get(f'{OLLAMA_HOST}/api/ps', timeout=5)
            if ps_response.status_code == 200:
                ps_data = ps_response.json()
                running_models = ps_data.get('models', [])
                metrics.append(f'ollama_running_models_count {len(running_models)}')

                for model in running_models:
                    name = model.get('name', 'unknown').replace('"', '\\"')
                    size_vram = model.get('size_vram', 0)
                    size_vram_gb = size_vram / (1024**3)
                    metrics.append(f'ollama_model_vram_bytes{{model="{name}"}} {size_vram}')
                    metrics.append(f'ollama_model_vram_gb{{model="{name}"}} {size_vram_gb:.2f}')
        except:
            pass

        # Calculate inference stats
        with lock:
            # Current/last measurements
            if cumulative_stats['last_tps'] > 0:
                metrics.append(f'ollama_inference_tokens_per_second {cumulative_stats["last_tps"]:.2f}')

            if cumulative_stats['last_ttft_ms'] > 0:
                metrics.append(f'ollama_inference_ttft_ms {cumulative_stats["last_ttft_ms"]:.2f}')

            # Cumulative totals
            metrics.append(f'ollama_inference_requests_total {cumulative_stats["total_requests"]}')
            metrics.append(f'ollama_inference_tokens_total {cumulative_stats["total_tokens"]}')
            metrics.append(f'ollama_inference_prompt_tokens_total {cumulative_stats["total_prompt_tokens"]}')

            # Average from history
            if inference_history:
                total_tokens = sum(h.get('eval_count', 0) for h in inference_history)
                total_time = sum(h.get('eval_duration', 0) for h in inference_history)

                if total_time > 0:
                    avg_tps = total_tokens / (total_time / 1e9)
                    metrics.append(f'ollama_inference_tokens_per_second_avg {avg_tps:.2f}')

    except requests.exceptions.ConnectionError:
        metrics.append('ollama_up 0')
    except requests.exceptions.Timeout:
        metrics.append('ollama_up 0')
        metrics.append('ollama_exporter_error{type="timeout"} 1')
    except Exception as e:
        metrics.append(f'ollama_exporter_error{{type="unknown"}} 1')

    # Add help text
    help_text = """# HELP ollama_up Whether Ollama is running
# TYPE ollama_up gauge
# HELP ollama_models_count Number of available models
# TYPE ollama_models_count gauge
# HELP ollama_running_models_count Number of currently loaded models
# TYPE ollama_running_models_count gauge
# HELP ollama_model_size_bytes Model size in bytes
# TYPE ollama_model_size_bytes gauge
# HELP ollama_model_vram_bytes Model VRAM usage in bytes
# TYPE ollama_model_vram_bytes gauge
# HELP ollama_inference_tokens_per_second Current tokens per second
# TYPE ollama_inference_tokens_per_second gauge
# HELP ollama_inference_tokens_per_second_avg Average tokens per second
# TYPE ollama_inference_tokens_per_second_avg gauge
# HELP ollama_inference_ttft_ms Time to first token in milliseconds
# TYPE ollama_inference_ttft_ms gauge
# HELP ollama_inference_requests_total Total inference requests tracked
# TYPE ollama_inference_requests_total counter
# HELP ollama_inference_tokens_total Total tokens generated
# TYPE ollama_inference_tokens_total counter
# HELP ollama_inference_prompt_tokens_total Total prompt tokens processed
# TYPE ollama_inference_prompt_tokens_total counter
"""

    return help_text + '\n'.join(metrics)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = get_ollama_metrics()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # Endpoint to record inference metrics from your benchmarks
        if self.path == '/record':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data)

                eval_count = data.get('eval_count', 0)
                eval_duration = data.get('eval_duration', 1)
                prompt_eval_count = data.get('prompt_eval_count', 0)
                prompt_eval_duration = data.get('prompt_eval_duration', 0)

                # Calculate TPS
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1e9)
                else:
                    tps = 0

                ttft_ms = prompt_eval_duration / 1e6

                with lock:
                    cumulative_stats['total_requests'] += 1
                    cumulative_stats['total_tokens'] += eval_count
                    cumulative_stats['total_prompt_tokens'] += prompt_eval_count
                    cumulative_stats['total_duration_ns'] += eval_duration
                    cumulative_stats['last_tps'] = tps
                    cumulative_stats['last_ttft_ms'] = ttft_ms

                    inference_history.append({
                        'eval_count': eval_count,
                        'eval_duration': eval_duration,
                        'prompt_eval_count': prompt_eval_count,
                        'prompt_eval_duration': prompt_eval_duration,
                        'timestamp': time.time()
                    })

                self.send_response(200)
                self.end_headers()
            except:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


if __name__ == '__main__':
    # Start probe thread
    probe_thread = threading.Thread(target=probe_ollama, daemon=True)
    probe_thread.start()
    print(f'Ollama Exporter running on port {PORT}')
    print(f'Connecting to Ollama at {OLLAMA_HOST}')
    print(f'Auto-probing every {PROBE_INTERVAL} seconds')

    server = HTTPServer(('0.0.0.0', PORT), MetricsHandler)
    server.serve_forever()
