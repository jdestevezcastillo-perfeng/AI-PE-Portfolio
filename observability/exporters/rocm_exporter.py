#!/usr/bin/env python3
"""
ROCm GPU Metrics Exporter for Prometheus
Exports AMD GPU metrics from rocm-smi
"""

import subprocess
import time
import re
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 9102
SCRAPE_INTERVAL = 2


def get_rocm_metrics():
    """Parse rocm-smi output and return Prometheus-formatted metrics."""
    metrics = []

    try:
        # Get GPU info
        result = subprocess.run(
            ['/opt/rocm-6.3.0/bin/rocm-smi', '--showtemp', '--showuse', '--showpower',
             '--showmeminfo', 'vram', '--showclkfrq'],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout

        # Parse temperature
        temp_match = re.search(r'Temperature \(Sensor edge\) \(C\):\s*(\d+\.?\d*)', output)
        if temp_match:
            metrics.append(f'rocm_gpu_temperature_celsius{{gpu="0"}} {temp_match.group(1)}')

        # Parse GPU usage
        gpu_use_match = re.search(r'GPU use \(%\):\s*(\d+)', output)
        if gpu_use_match:
            metrics.append(f'rocm_gpu_utilization_percent{{gpu="0"}} {gpu_use_match.group(1)}')

        # Parse power
        power_match = re.search(r'Average Graphics Package Power \(W\):\s*(\d+\.?\d*)', output)
        if power_match:
            metrics.append(f'rocm_gpu_power_watts{{gpu="0"}} {power_match.group(1)}')

        # Parse VRAM
        vram_used_match = re.search(r'VRAM Total Used Memory \(B\):\s*(\d+)', output)
        vram_total_match = re.search(r'VRAM Total Memory \(B\):\s*(\d+)', output)
        if vram_used_match:
            vram_used_gb = int(vram_used_match.group(1)) / (1024**3)
            metrics.append(f'rocm_gpu_vram_used_bytes{{gpu="0"}} {vram_used_match.group(1)}')
            metrics.append(f'rocm_gpu_vram_used_gb{{gpu="0"}} {vram_used_gb:.2f}')
        if vram_total_match:
            vram_total_gb = int(vram_total_match.group(1)) / (1024**3)
            metrics.append(f'rocm_gpu_vram_total_bytes{{gpu="0"}} {vram_total_match.group(1)}')
            metrics.append(f'rocm_gpu_vram_total_gb{{gpu="0"}} {vram_total_gb:.2f}')

        # Parse clock frequencies
        sclk_match = re.search(r'sclk clock level:\s*\d+:\s*(\d+)Mhz', output)
        mclk_match = re.search(r'mclk clock level:\s*\d+:\s*(\d+)Mhz', output)
        if sclk_match:
            metrics.append(f'rocm_gpu_sclk_mhz{{gpu="0"}} {sclk_match.group(1)}')
        if mclk_match:
            metrics.append(f'rocm_gpu_mclk_mhz{{gpu="0"}} {mclk_match.group(1)}')

    except subprocess.TimeoutExpired:
        metrics.append('rocm_exporter_error{type="timeout"} 1')
    except FileNotFoundError:
        metrics.append('rocm_exporter_error{type="rocm_smi_not_found"} 1')
    except Exception as e:
        metrics.append(f'rocm_exporter_error{{type="unknown"}} 1')

    # Add help text
    help_text = """# HELP rocm_gpu_temperature_celsius GPU temperature in Celsius
# TYPE rocm_gpu_temperature_celsius gauge
# HELP rocm_gpu_utilization_percent GPU utilization percentage
# TYPE rocm_gpu_utilization_percent gauge
# HELP rocm_gpu_power_watts GPU power consumption in watts
# TYPE rocm_gpu_power_watts gauge
# HELP rocm_gpu_vram_used_bytes GPU VRAM used in bytes
# TYPE rocm_gpu_vram_used_bytes gauge
# HELP rocm_gpu_vram_total_bytes GPU VRAM total in bytes
# TYPE rocm_gpu_vram_total_bytes gauge
# HELP rocm_gpu_sclk_mhz GPU shader clock in MHz
# TYPE rocm_gpu_sclk_mhz gauge
# HELP rocm_gpu_mclk_mhz GPU memory clock in MHz
# TYPE rocm_gpu_mclk_mhz gauge
"""

    return help_text + '\n'.join(metrics)


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = get_rocm_metrics()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(metrics.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), MetricsHandler)
    print(f'ROCm Exporter running on port {PORT}')
    server.serve_forever()
