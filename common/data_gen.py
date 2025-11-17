import random
import time
from typing import List

import numpy as np
import typer

from .perf_timers import PerfStats, time_block

app = typer.Typer(help="Synthetic latency/throughput generators to dry-run without GPU.")


def synthetic_latency_work(duration_ms: float):
    target = duration_ms / 1000.0
    start = time.perf_counter()
    while time.perf_counter() - start < target:
        np.exp(np.random.randn(256)).sum()  # Small CPU-bound op


def synthetic_throughput_work(batch: int, dims: int):
    data = np.random.randn(batch, dims)
    _ = data @ data.T  # matrix multiply for CPU load


@app.command()
def latency(mode: str = "p50", runs: int = 20):
    stats = PerfStats()
    target = {"p50": 30, "p95": 80, "p99": 150}.get(mode, 50)
    for _ in range(runs):
        with time_block(stats):
            synthetic_latency_work(target)
    typer.echo(stats.summary())


@app.command()
def throughput(batches: int = 10, batch_size: int = 32, dims: int = 512):
    stats = PerfStats()
    for _ in range(batches):
        with time_block(stats):
            synthetic_throughput_work(batch_size, dims)
    typer.echo(stats.summary())


@app.command()
def mixed(runs: int = 20):
    stats = PerfStats()
    for _ in range(runs):
        mode = random.choice(["latency", "throughput"])
        with time_block(stats):
            if mode == "latency":
                synthetic_latency_work(random.choice([20, 40, 80]))
            else:
                synthetic_throughput_work(random.choice([8, 16, 32]), 256)
    typer.echo(stats.summary())


if __name__ == "__main__":
    app()
