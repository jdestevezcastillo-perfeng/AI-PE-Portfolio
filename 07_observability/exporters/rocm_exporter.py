#!/usr/bin/env python3
"""
ROCm GPU Metrics Exporter for Prometheus
=========================================
Exports comprehensive AMD GPU metrics from rocm-smi including:
- Temperature (edge, junction/hotspot, memory)
- Utilization (GPU, memory bandwidth)
- Power consumption
- VRAM usage
- Clock frequencies
- Fan speed
"""

import subprocess
import re
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(os.environ.get('PORT', 9102))
ROCM_SMI_PATH = os.environ.get('ROCM_SMI_PATH', '/opt/rocm/bin/rocm-smi')

# Try alternative paths if default doesn't exist
ROCM_SMI_PATHS = [
    ROCM_SMI_PATH,
    '/opt/rocm-6.3.0/bin/rocm-smi',
    '/opt/rocm-6.2.0/bin/rocm-smi',
    '/opt/rocm-6.1.0/bin/rocm-smi',
    '/usr/bin/rocm-smi',
]


def find_rocm_smi():
    """Find the rocm-smi binary."""
    for path in ROCM_SMI_PATHS:
        if os.path.exists(path):
            return path
    return None


def run_rocm_smi(args):
    """Run rocm-smi with given arguments and return output."""
    rocm_smi = find_rocm_smi()
    if not rocm_smi:
        return None
    try:
        result = subprocess.run(
            [rocm_smi] + args,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def get_rocm_metrics():
    """Parse rocm-smi output and return Prometheus-formatted metrics."""
    metrics = []
    errors = []

    # Get comprehensive GPU info
    output = run_rocm_smi([
        '--showtemp',
        '--showuse',
        '--showpower',
        '--showmeminfo', 'vram',
        '--showclkfrq',
        '--showfan',
        '--showmemuse',
        '--showvoltage',
    ])

    if output is None:
        errors.append('rocm_exporter_error{type="rocm_smi_not_found"} 1')
    else:
        # ==========================================
        # TEMPERATURE METRICS
        # ==========================================

        # Edge temperature (standard GPU temp)
        temp_edge = re.search(r'Temperature \(Sensor edge\) \(C\):\s*(\d+\.?\d*)', output)
        if temp_edge:
            metrics.append(f'rocm_gpu_temperature_edge_celsius{{gpu="0",sensor="edge"}} {temp_edge.group(1)}')

        # Junction/Hotspot temperature (critical for thermal throttling)
        temp_junction = re.search(r'Temperature \(Sensor junction\) \(C\):\s*(\d+\.?\d*)', output)
        if temp_junction:
            metrics.append(f'rocm_gpu_temperature_junction_celsius{{gpu="0",sensor="junction"}} {temp_junction.group(1)}')
            metrics.append(f'rocm_gpu_temperature_hotspot_celsius{{gpu="0",sensor="hotspot"}} {temp_junction.group(1)}')

        # Memory temperature
        temp_mem = re.search(r'Temperature \(Sensor memory\) \(C\):\s*(\d+\.?\d*)', output)
        if temp_mem:
            metrics.append(f'rocm_gpu_temperature_memory_celsius{{gpu="0",sensor="memory"}} {temp_mem.group(1)}')

        # HBM temperature (for cards with HBM memory)
        temp_hbm = re.search(r'Temperature \(Sensor HBM 0\) \(C\):\s*(\d+\.?\d*)', output)
        if temp_hbm:
            metrics.append(f'rocm_gpu_temperature_hbm_celsius{{gpu="0",sensor="hbm"}} {temp_hbm.group(1)}')

        # ==========================================
        # UTILIZATION METRICS
        # ==========================================

        # GPU compute utilization
        gpu_use = re.search(r'GPU use \(%\):\s*(\d+)', output)
        if gpu_use:
            metrics.append(f'rocm_gpu_utilization_percent{{gpu="0"}} {gpu_use.group(1)}')

        # GPU memory utilization (bandwidth)
        gpu_mem_use = re.search(r'GPU memory use \(%\):\s*(\d+)', output)
        if gpu_mem_use:
            metrics.append(f'rocm_gpu_memory_utilization_percent{{gpu="0"}} {gpu_mem_use.group(1)}')

        # ==========================================
        # POWER METRICS
        # ==========================================

        # Current power draw
        power_avg = re.search(r'Average Graphics Package Power \(W\):\s*(\d+\.?\d*)', output)
        if power_avg:
            metrics.append(f'rocm_gpu_power_watts{{gpu="0"}} {power_avg.group(1)}')

        # Socket power (total board power)
        power_socket = re.search(r'Current Socket Graphics Package Power \(W\):\s*(\d+\.?\d*)', output)
        if power_socket:
            metrics.append(f'rocm_gpu_power_socket_watts{{gpu="0"}} {power_socket.group(1)}')

        # ==========================================
        # VRAM METRICS
        # ==========================================

        vram_used = re.search(r'VRAM Total Used Memory \(B\):\s*(\d+)', output)
        vram_total = re.search(r'VRAM Total Memory \(B\):\s*(\d+)', output)

        if vram_used:
            used_bytes = int(vram_used.group(1))
            metrics.append(f'rocm_gpu_vram_used_bytes{{gpu="0"}} {used_bytes}')
            metrics.append(f'rocm_gpu_vram_used_gb{{gpu="0"}} {used_bytes / (1024**3):.3f}')

        if vram_total:
            total_bytes = int(vram_total.group(1))
            metrics.append(f'rocm_gpu_vram_total_bytes{{gpu="0"}} {total_bytes}')
            metrics.append(f'rocm_gpu_vram_total_gb{{gpu="0"}} {total_bytes / (1024**3):.3f}')

        if vram_used and vram_total:
            used = int(vram_used.group(1))
            total = int(vram_total.group(1))
            pct = (used / total) * 100 if total > 0 else 0
            metrics.append(f'rocm_gpu_vram_used_percent{{gpu="0"}} {pct:.2f}')

        # ==========================================
        # CLOCK FREQUENCY METRICS
        # ==========================================

        # Shader/Graphics clock
        sclk = re.search(r'sclk clock level:\s*\d+:\s*(\d+)Mhz', output)
        if sclk:
            metrics.append(f'rocm_gpu_clock_graphics_mhz{{gpu="0"}} {sclk.group(1)}')

        # Memory clock
        mclk = re.search(r'mclk clock level:\s*\d+:\s*(\d+)Mhz', output)
        if mclk:
            metrics.append(f'rocm_gpu_clock_memory_mhz{{gpu="0"}} {mclk.group(1)}')

        # ==========================================
        # FAN METRICS
        # ==========================================

        fan_speed = re.search(r'Fan speed \(%\):\s*(\d+)', output)
        if fan_speed:
            metrics.append(f'rocm_gpu_fan_speed_percent{{gpu="0"}} {fan_speed.group(1)}')

        fan_rpm = re.search(r'Fan RPM:\s*(\d+)', output)
        if fan_rpm:
            metrics.append(f'rocm_gpu_fan_rpm{{gpu="0"}} {fan_rpm.group(1)}')

        # ==========================================
        # VOLTAGE METRICS
        # ==========================================

        voltage = re.search(r'Voltage \(mV\):\s*(\d+)', output)
        if voltage:
            mv = int(voltage.group(1))
            metrics.append(f'rocm_gpu_voltage_mv{{gpu="0"}} {mv}')
            metrics.append(f'rocm_gpu_voltage_v{{gpu="0"}} {mv / 1000:.3f}')

    # Add exporter status
    if errors:
        metrics.extend(errors)
    else:
        metrics.append('rocm_exporter_up 1')

    # Build output with HELP and TYPE annotations
    output_lines = [
        '# HELP rocm_gpu_temperature_edge_celsius GPU edge temperature in Celsius',
        '# TYPE rocm_gpu_temperature_edge_celsius gauge',
        '# HELP rocm_gpu_temperature_junction_celsius GPU junction (hotspot) temperature in Celsius',
        '# TYPE rocm_gpu_temperature_junction_celsius gauge',
        '# HELP rocm_gpu_temperature_hotspot_celsius GPU hotspot temperature in Celsius (alias for junction)',
        '# TYPE rocm_gpu_temperature_hotspot_celsius gauge',
        '# HELP rocm_gpu_temperature_memory_celsius GPU memory temperature in Celsius',
        '# TYPE rocm_gpu_temperature_memory_celsius gauge',
        '# HELP rocm_gpu_temperature_hbm_celsius GPU HBM temperature in Celsius',
        '# TYPE rocm_gpu_temperature_hbm_celsius gauge',
        '# HELP rocm_gpu_utilization_percent GPU compute utilization percentage',
        '# TYPE rocm_gpu_utilization_percent gauge',
        '# HELP rocm_gpu_memory_utilization_percent GPU memory bandwidth utilization percentage',
        '# TYPE rocm_gpu_memory_utilization_percent gauge',
        '# HELP rocm_gpu_power_watts GPU power consumption in watts',
        '# TYPE rocm_gpu_power_watts gauge',
        '# HELP rocm_gpu_power_socket_watts GPU socket/board power in watts',
        '# TYPE rocm_gpu_power_socket_watts gauge',
        '# HELP rocm_gpu_vram_used_bytes GPU VRAM used in bytes',
        '# TYPE rocm_gpu_vram_used_bytes gauge',
        '# HELP rocm_gpu_vram_total_bytes GPU VRAM total in bytes',
        '# TYPE rocm_gpu_vram_total_bytes gauge',
        '# HELP rocm_gpu_vram_used_gb GPU VRAM used in GB',
        '# TYPE rocm_gpu_vram_used_gb gauge',
        '# HELP rocm_gpu_vram_total_gb GPU VRAM total in GB',
        '# TYPE rocm_gpu_vram_total_gb gauge',
        '# HELP rocm_gpu_vram_used_percent GPU VRAM usage percentage',
        '# TYPE rocm_gpu_vram_used_percent gauge',
        '# HELP rocm_gpu_clock_graphics_mhz GPU graphics/shader clock in MHz',
        '# TYPE rocm_gpu_clock_graphics_mhz gauge',
        '# HELP rocm_gpu_clock_memory_mhz GPU memory clock in MHz',
        '# TYPE rocm_gpu_clock_memory_mhz gauge',
        '# HELP rocm_gpu_fan_speed_percent GPU fan speed percentage',
        '# TYPE rocm_gpu_fan_speed_percent gauge',
        '# HELP rocm_gpu_fan_rpm GPU fan speed in RPM',
        '# TYPE rocm_gpu_fan_rpm gauge',
        '# HELP rocm_gpu_voltage_mv GPU voltage in millivolts',
        '# TYPE rocm_gpu_voltage_mv gauge',
        '# HELP rocm_gpu_voltage_v GPU voltage in volts',
        '# TYPE rocm_gpu_voltage_v gauge',
        '# HELP rocm_exporter_up ROCm exporter status (1=up, 0=down)',
        '# TYPE rocm_exporter_up gauge',
        '',
    ]

    return '\n'.join(output_lines + metrics) + '\n'


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            metrics = get_rocm_metrics()
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

    def log_message(self, format, *args):
        # Suppress request logging for cleaner output
        pass


if __name__ == '__main__':
    print(f'ROCm GPU Exporter starting on port {PORT}')
    rocm_smi = find_rocm_smi()
    if rocm_smi:
        print(f'Using rocm-smi at: {rocm_smi}')
    else:
        print('WARNING: rocm-smi not found, metrics will show errors')

    server = HTTPServer(('0.0.0.0', PORT), MetricsHandler)
    print(f'ROCm GPU Exporter running on http://0.0.0.0:{PORT}/metrics')
    server.serve_forever()
