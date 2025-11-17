import random
import time
from typing import Optional

import typer
from fastapi import FastAPI
from prometheus_client import Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from common.logging_utils import get_logger, setup_logging
from common.tracing import init_tracer, traced_span

app = typer.Typer(help="Observability demo with FastAPI, Prometheus, tracing")
log = get_logger("obs_demo")


def build_app(enable_tracing: bool, otlp_endpoint: Optional[str]):
    if enable_tracing:
        init_tracer("obs-demo", use_console=True, otlp_endpoint=otlp_endpoint)
    fastapi_app = FastAPI()
    registry = CollectorRegistry()
    latency_hist = Histogram("demo_request_latency_ms", "Request latency", registry=registry)
    throughput = Gauge("demo_throughput_qps", "Requests per second", registry=registry)

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok"}

    @fastapi_app.get("/metrics")
    async def metrics():
        data = generate_latest(registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    @fastapi_app.get("/infer")
    async def infer(delay: float = 0.05):
        start = time.perf_counter()
        with traced_span("infer") if enable_tracing else noop_context():
            # Synthetic latency; replace with real model call
            time.sleep(delay)
            result = {"output": f"synthetic-result-{random.randint(0,1000)}"}
        duration_ms = (time.perf_counter() - start) * 1000
        latency_hist.observe(duration_ms)
        throughput.set(1 / (duration_ms / 1000))
        log.info("infer_complete", duration_ms=duration_ms)
        return result

    return fastapi_app


class noop_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


@app.command()
def main(port: int = 8001, log_level: str = "INFO", enable_tracing: bool = True, otlp_endpoint: Optional[str] = None):
    setup_logging(log_level)
    fastapi_app = build_app(enable_tracing, otlp_endpoint)
    import uvicorn

    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    app()
