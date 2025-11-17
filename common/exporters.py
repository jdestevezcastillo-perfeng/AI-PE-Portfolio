import time
from typing import Callable, Optional

from prometheus_client import CollectorRegistry, Gauge, Histogram, start_http_server

from .perf_timers import PerfStats


class PerfExporter:
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        registry = registry or CollectorRegistry()
        self.latency = Histogram("perf_latency_ms", "Latency in milliseconds")
        self.throughput = Gauge("perf_throughput_qps", "Throughput (queries per second)")
        self.registry = registry

    def observe_latency(self, duration_ms: float):
        self.latency.observe(duration_ms)

    def set_throughput(self, qps: float):
        self.throughput.set(qps)


def run_with_exporter(fn: Callable, port: int = 8000, runs: int = 20):
    """Expose Prometheus metrics while running a callable multiple times."""
    start_http_server(port)
    stats = PerfStats()
    exporter = PerfExporter()
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        duration_ms = (end - start) * 1000
        stats.add(duration_ms)
        exporter.observe_latency(duration_ms)
    # crude throughput estimate
    total_s = sum(stats.records) / 1000
    if total_s > 0:
        exporter.set_throughput(runs / total_s)
    return stats.summary()
