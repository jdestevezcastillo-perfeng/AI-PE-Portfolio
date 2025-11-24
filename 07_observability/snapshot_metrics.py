#!/usr/bin/env python3
"""
Observability Snapshot Tool
============================
Captures metrics, traces, and logs from the observability stack
for a given time range and saves them locally for later analysis.

Usage:
    # Capture last 5 minutes
    python snapshot_metrics.py --duration 5m --output benchmark_snapshot.json

    # Capture specific time range
    python snapshot_metrics.py --start "2024-01-15T10:00:00Z" --end "2024-01-15T10:05:00Z"

    # Capture with trace ID from benchmark
    python snapshot_metrics.py --duration 5m --trace-id abc123 --output snapshot.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode, quote
from typing import Optional, Any

# Default endpoints
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')
LOKI_URL = os.environ.get('LOKI_URL', 'http://localhost:3100')
TEMPO_URL = os.environ.get('TEMPO_URL', 'http://localhost:3200')


def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '5m', '1h', '30s' into timedelta."""
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])

    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f"Unknown duration unit: {unit}")


def fetch_json(url: str, timeout: int = 30) -> Optional[dict]:
    """Fetch JSON from URL."""
    try:
        req = Request(url, headers={'Accept': 'application/json'})
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except URLError as e:
        print(f"  Warning: Failed to fetch {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Warning: Invalid JSON from {url}: {e}")
        return None


def query_prometheus_range(query: str, start: datetime, end: datetime, step: str = '15s') -> Optional[dict]:
    """Query Prometheus for a time range."""
    params = {
        'query': query,
        'start': start.isoformat(),
        'end': end.isoformat(),
        'step': step,
    }
    url = f"{PROMETHEUS_URL}/api/v1/query_range?{urlencode(params)}"
    return fetch_json(url)


def query_prometheus_instant(query: str) -> Optional[dict]:
    """Query Prometheus for instant value."""
    params = {'query': query}
    url = f"{PROMETHEUS_URL}/api/v1/query?{urlencode(params)}"
    return fetch_json(url)


def query_loki(query: str, start: datetime, end: datetime, limit: int = 1000) -> Optional[dict]:
    """Query Loki for logs in time range."""
    params = {
        'query': query,
        'start': str(int(start.timestamp() * 1e9)),  # nanoseconds
        'end': str(int(end.timestamp() * 1e9)),
        'limit': str(limit),
    }
    url = f"{LOKI_URL}/loki/api/v1/query_range?{urlencode(params)}"
    return fetch_json(url)


def query_tempo_trace(trace_id: str) -> Optional[dict]:
    """Fetch a specific trace from Tempo."""
    url = f"{TEMPO_URL}/api/traces/{trace_id}"
    return fetch_json(url)


def search_tempo_traces(service_name: str, start: datetime, end: datetime, limit: int = 20) -> Optional[dict]:
    """Search for traces in Tempo."""
    params = {
        'start': str(int(start.timestamp())),
        'end': str(int(end.timestamp())),
        'limit': str(limit),
    }
    if service_name:
        params['tags'] = f'service.name={service_name}'

    url = f"{TEMPO_URL}/api/search?{urlencode(params)}"
    return fetch_json(url)


def capture_gpu_metrics(start: datetime, end: datetime) -> dict:
    """Capture GPU metrics from Prometheus."""
    print("  Capturing GPU metrics...")

    queries = {
        'gpu_temperature_edge': 'rocm_gpu_temperature_edge_celsius',
        'gpu_temperature_junction': 'rocm_gpu_temperature_junction_celsius',
        'gpu_temperature_memory': 'rocm_gpu_temperature_memory_celsius',
        'gpu_utilization': 'rocm_gpu_utilization_percent',
        'gpu_memory_utilization': 'rocm_gpu_memory_utilization_percent',
        'gpu_power_watts': 'rocm_gpu_power_watts',
        'gpu_vram_used_gb': 'rocm_gpu_vram_used_gb',
        'gpu_vram_used_percent': 'rocm_gpu_vram_used_percent',
        'gpu_clock_graphics': 'rocm_gpu_clock_graphics_mhz',
        'gpu_clock_memory': 'rocm_gpu_clock_memory_mhz',
        'gpu_fan_speed': 'rocm_gpu_fan_speed_percent',
    }

    results = {}
    for name, query in queries.items():
        data = query_prometheus_range(query, start, end)
        if data and data.get('status') == 'success':
            results[name] = data.get('data', {})

    return results


def capture_llm_metrics(start: datetime, end: datetime) -> dict:
    """Capture LLM inference metrics from Prometheus."""
    print("  Capturing LLM metrics...")

    queries = {
        'ollama_up': 'ollama_up',
        'ollama_models_running': 'ollama_models_running',
        'inference_tps_avg': 'ollama_inference_tokens_per_second_avg',
        'inference_tps_max': 'ollama_inference_tokens_per_second_max',
        'inference_ttft_avg': 'ollama_inference_ttft_ms_avg',
        'inference_tpot_avg': 'ollama_inference_tpot_ms_avg',
        'inference_total_requests': 'ollama_inference_total_requests',
        'inference_total_tokens': 'ollama_inference_total_tokens_generated',
    }

    results = {}
    for name, query in queries.items():
        data = query_prometheus_range(query, start, end)
        if data and data.get('status') == 'success':
            results[name] = data.get('data', {})

    return results


def capture_system_metrics(start: datetime, end: datetime) -> dict:
    """Capture system metrics from Prometheus."""
    print("  Capturing system metrics...")

    queries = {
        'cpu_usage': '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)',
        'memory_used_percent': '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
        'load_1m': 'node_load1',
        'load_5m': 'node_load5',
    }

    results = {}
    for name, query in queries.items():
        data = query_prometheus_range(query, start, end)
        if data and data.get('status') == 'success':
            results[name] = data.get('data', {})

    return results


def capture_logs(start: datetime, end: datetime) -> dict:
    """Capture logs from Loki."""
    print("  Capturing logs...")

    queries = {
        'ollama_logs': '{job="ollama"} |= ""',
        'system_logs': '{job="varlogs"} |~ "ollama|gpu|rocm"',
    }

    results = {}
    for name, query in queries.items():
        data = query_loki(query, start, end)
        if data and data.get('status') == 'success':
            results[name] = data.get('data', {})

    return results


def capture_traces(start: datetime, end: datetime, trace_ids: list = None) -> dict:
    """Capture traces from Tempo."""
    print("  Capturing traces...")

    results = {
        'trace_search': None,
        'traces': []
    }

    # Search for traces
    search_result = search_tempo_traces('llm-benchmark', start, end)
    if search_result:
        results['trace_search'] = search_result

        # Fetch individual traces
        traces = search_result.get('traces', [])
        for trace_info in traces[:10]:  # Limit to 10 traces
            trace_id = trace_info.get('traceID')
            if trace_id:
                trace_data = query_tempo_trace(trace_id)
                if trace_data:
                    results['traces'].append(trace_data)

    # Also fetch specific trace IDs if provided
    if trace_ids:
        for trace_id in trace_ids:
            trace_data = query_tempo_trace(trace_id)
            if trace_data:
                results['traces'].append(trace_data)

    return results


def calculate_summary(snapshot: dict) -> dict:
    """Calculate summary statistics from the snapshot."""
    summary = {}

    # GPU summary
    gpu_metrics = snapshot.get('gpu_metrics', {})
    for metric_name, metric_data in gpu_metrics.items():
        if metric_data and 'result' in metric_data:
            values = []
            for result in metric_data['result']:
                for point in result.get('values', []):
                    try:
                        values.append(float(point[1]))
                    except (ValueError, IndexError):
                        pass

            if values:
                summary[f'{metric_name}_avg'] = sum(values) / len(values)
                summary[f'{metric_name}_max'] = max(values)
                summary[f'{metric_name}_min'] = min(values)

    # LLM summary
    llm_metrics = snapshot.get('llm_metrics', {})
    for metric_name, metric_data in llm_metrics.items():
        if metric_data and 'result' in metric_data:
            values = []
            for result in metric_data['result']:
                for point in result.get('values', []):
                    try:
                        values.append(float(point[1]))
                    except (ValueError, IndexError):
                        pass

            if values:
                summary[f'{metric_name}_last'] = values[-1] if values else None

    return summary


def main():
    parser = argparse.ArgumentParser(description='Capture observability snapshot')
    parser.add_argument('--duration', '-d', default='5m',
                        help='Duration to capture (e.g., 5m, 1h, 30s)')
    parser.add_argument('--start', help='Start time (ISO format)')
    parser.add_argument('--end', help='End time (ISO format)')
    parser.add_argument('--output', '-o', default='observability_snapshot.json',
                        help='Output file path')
    parser.add_argument('--trace-id', '-t', action='append',
                        help='Specific trace ID(s) to capture')
    parser.add_argument('--no-logs', action='store_true',
                        help='Skip log capture')
    parser.add_argument('--no-traces', action='store_true',
                        help='Skip trace capture')

    args = parser.parse_args()

    # Determine time range
    if args.start and args.end:
        start = datetime.fromisoformat(args.start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(args.end.replace('Z', '+00:00'))
    else:
        end = datetime.now(timezone.utc)
        duration = parse_duration(args.duration)
        start = end - duration

    print(f"Capturing observability snapshot")
    print(f"  Time range: {start.isoformat()} to {end.isoformat()}")
    print(f"  Duration: {end - start}")
    print()

    # Capture all data
    snapshot = {
        'metadata': {
            'captured_at': datetime.now(timezone.utc).isoformat(),
            'start_time': start.isoformat(),
            'end_time': end.isoformat(),
            'duration_seconds': (end - start).total_seconds(),
        },
        'gpu_metrics': capture_gpu_metrics(start, end),
        'llm_metrics': capture_llm_metrics(start, end),
        'system_metrics': capture_system_metrics(start, end),
    }

    if not args.no_logs:
        snapshot['logs'] = capture_logs(start, end)

    if not args.no_traces:
        snapshot['traces'] = capture_traces(start, end, args.trace_id)

    # Calculate summary
    snapshot['summary'] = calculate_summary(snapshot)

    # Save to file
    with open(args.output, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print()
    print(f"Snapshot saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")

    # Print summary
    if snapshot['summary']:
        print()
        print("Summary:")
        for key, value in snapshot['summary'].items():
            if value is not None:
                print(f"  {key}: {value:.2f}")


if __name__ == '__main__':
    main()
