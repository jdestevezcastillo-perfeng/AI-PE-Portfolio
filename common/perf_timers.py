import time
from contextlib import contextmanager
from typing import Callable, Dict, List


class PerfStats:
    def __init__(self):
        self.records: List[float] = []

    def add(self, duration_ms: float):
        self.records.append(duration_ms)

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {}
        sorted_durations = sorted(self.records)
        return {
            "count": len(self.records),
            "mean_ms": sum(self.records) / len(self.records),
            "p50_ms": sorted_durations[int(0.5 * (len(self.records) - 1))],
            "p95_ms": sorted_durations[int(0.95 * (len(self.records) - 1))],
            "p99_ms": sorted_durations[int(0.99 * (len(self.records) - 1))],
        }


@contextmanager
def time_block(stats: PerfStats):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    stats.add((end - start) * 1000)


def benchmark(func: Callable, runs: int = 10) -> Dict[str, float]:
    stats = PerfStats()
    for _ in range(runs):
        with time_block(stats):
            func()
    return stats.summary()
