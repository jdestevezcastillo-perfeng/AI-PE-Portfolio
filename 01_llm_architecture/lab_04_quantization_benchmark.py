
"""
BENCHMARK PIPELINE DIAGRAM (with OpenTelemetry)
------------------------------------------------

   [Benchmark Config]  (Model, Prompt, Requests)
          |
          v
   [OTel Tracer]       (Distributed Tracing)
      |   |            - Exports to Tempo via OTLP
      |   |            - Trace ID propagation
      v   v
   [Telemetry Sampler] (Background Thread)
      |   |   |        - Polls nvidia-smi / rocm-smi
      |   |   |        - Tracks VRAM & GPU Util
      v   v   v
   [Inference Loop]    (Sequential Requests)
          |
          v
      [Ollama API]     (HTTP POST /api/generate)
          |            - TTFT (Time To First Token)
          |            - TPS (Tokens Per Second)
          v
   [Results Aggregator] (JSON Report)
          |
          v
      [Output File]    (quantization_benchmark_*.json)

   >>> AIPE NOTE: Observability with OpenTelemetry
   This script exports traces to Tempo (via OTLP) for end-to-end visibility.
   Each inference request is a span with LLM-specific attributes.
   View traces in Grafana -> Explore -> Tempo.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import shutil

# ==========================================
# OpenTelemetry Imports (Graceful Degradation)
# ==========================================
# >>> AIPE NOTE: Optional Tracing
# If OTel packages aren't installed, tracing is disabled but the script still works.
# Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

TRACING_ENABLED = False
tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode
    TRACING_ENABLED = True
except ImportError:
    pass

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
METRICS_EXPORTER_URL = os.environ.get("METRICS_EXPORTER_URL", "http://localhost:9103/record")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an Ollama model with GPU telemetry.")
    parser.add_argument("--model", default="llama3.1:8b", help="Model name registered with Ollama.")
    parser.add_argument("--prompt", default="Explain how quantization affects LLM inference performance.", help="Prompt to benchmark.")
    parser.add_argument("--requests", type=int, default=5, help="Number of sequential requests to run.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout seconds.")
    parser.add_argument("--output", help="Optional output JSON path.")
    parser.add_argument("--no-tracing", action="store_true", help="Disable OpenTelemetry tracing.")
    return parser.parse_args()


def init_tracing(service_name: str = "llm-benchmark") -> None:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    >>> AIPE NOTE: Distributed Tracing Setup
    Traces are exported to Tempo via gRPC on port 4317.
    Each span includes LLM-specific semantic conventions.
    """
    global tracer, TRACING_ENABLED

    if not TRACING_ENABLED:
        print("OpenTelemetry not installed. Tracing disabled.")
        print("Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc")
        return

    try:
        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)

        otlp_exporter = OTLPSpanExporter(
            endpoint=OTEL_ENDPOINT,
            insecure=True  # No TLS for local development
        )

        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer(__name__)
        print(f"OpenTelemetry tracing enabled -> {OTEL_ENDPOINT}")
    except Exception as e:
        print(f"Failed to initialize tracing: {e}")
        TRACING_ENABLED = False

# ==========================================
# 2. UTILITIES (Quantization Detection)
# ==========================================

def detect_quantization(model: str) -> Optional[str]:
    """Best-effort parse of the quantization level from `ollama show` output."""
    try:
        result = subprocess.run(
            ["ollama", "show", model],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("quantization"):
            parts = line.split()
            if parts:
                return parts[-1]
    return None


def calc_stats(values: List[float]) -> Dict[str, float]:
    """Return mean/min/max/median/stdev for a list of values."""
    if not values:
        return {}

    summary: Dict[str, float] = {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }

    if len(values) > 1:
        summary["stdev"] = statistics.stdev(values)

    return summary

def sanitize_filename(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)
    return safe.strip("_") or "model"

def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024 ** 3)

# ==========================================
# 3. TELEMETRY (GPU Monitoring)
# ==========================================

class NullSampler:
    """Fallback sampler when rocm-smi is unavailable."""
    def start(self) -> None: pass
    def stop(self) -> None: pass
    def activate(self) -> None: pass
    def deactivate(self) -> None: pass
    def summary(self) -> Dict[str, Any]: return {"available": False}

class GPUTelemetrySampler:
    """
    Samples GPU utilization and VRAM usage via rocm-smi or nvidia-smi.
    
    >>> AIPE NOTE: Observability
    This class implements a sidecar pattern for monitoring. It runs in a separate thread
    to avoid blocking the main inference loop, providing real-time visibility into
    hardware saturation (Compute vs Memory Bound).
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.rocm_available = shutil.which("rocm-smi") is not None
        self.nvidia_available = shutil.which("nvidia-smi") is not None
        self.available = self.rocm_available or self.nvidia_available

        self.baseline_vram: Optional[int] = None
        self.total_vram: Optional[int] = None

        self.max_gpu_util: float = 0.0
        self.min_vram_seen: Optional[int] = None
        self.max_vram_seen: Optional[int] = None
        self._gpu_util_sum: float = 0.0
        self._sample_count: int = 0

        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.available:
            return

        baseline = self._query_once()
        if baseline:
            self.baseline_vram = baseline["vram_used"]
            self.total_vram = baseline["vram_total"]

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.available or not self._thread:
            return

        self._stop.set()
        self._thread.join(timeout=2)

    def activate(self) -> None:
        if self.available:
            self._active.set()

    def deactivate(self) -> None:
        if self.available:
            self._active.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._active.is_set():
                time.sleep(self.interval)
                continue

            sample = self._query_once()
            if sample:
                gpu_util = sample["gpu_pct"]
                vram_used = sample["vram_used"]

                self._gpu_util_sum += gpu_util
                self._sample_count += 1

                self.max_gpu_util = max(self.max_gpu_util, gpu_util)
                if self.min_vram_seen is None or vram_used < self.min_vram_seen:
                    self.min_vram_seen = vram_used
                if self.max_vram_seen is None or vram_used > self.max_vram_seen:
                    self.max_vram_seen = vram_used

            time.sleep(self.interval)

    def _query_once(self) -> Optional[Dict[str, float]]:
        if self.nvidia_available:
            return self._query_nvidia()
        elif self.rocm_available:
            return self._query_rocm()
        return None

    def _query_nvidia(self) -> Optional[Dict[str, float]]:
        try:
            # Query utilization.gpu, memory.used, memory.total
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            line = result.stdout.strip()
            if not line: return None
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 3: return None
            
            return {
                "gpu_pct": float(parts[0]),
                "vram_used": int(parts[1]) * 1024 * 1024, # MB to Bytes
                "vram_total": int(parts[2]) * 1024 * 1024, # MB to Bytes
            }
        except (subprocess.CalledProcessError, ValueError, IndexError):
            return None

    def _query_rocm(self) -> Optional[Dict[str, float]]:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        output = result.stdout.strip()
        if not output: return None

        try:
            data = json.loads(output)
            card = next(iter(data.values()), None)
            if not card: return None
            
            return {
                "gpu_pct": float(card.get("GPU use (%)", "0").strip("%")),
                "vram_used": int(card.get("VRAM Total Used Memory (B)", "0")),
                "vram_total": int(card.get("VRAM Total Memory (B)", "0")),
            }
        except (json.JSONDecodeError, ValueError, AttributeError):
            return None

    def summary(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}

        avg_gpu = None
        if self._sample_count:
            avg_gpu = self._gpu_util_sum / self._sample_count

        additional_vram = None
        if self.max_vram_seen is not None and self.baseline_vram is not None:
            additional_vram = max(0, self.max_vram_seen - self.baseline_vram)

        return {
            "available": True,
            "max_gpu_percent": self.max_gpu_util if self.max_gpu_util else None,
            "avg_gpu_percent": avg_gpu,
            "peak_vram_used_bytes": self.max_vram_seen,
            "peak_vram_used_gib": bytes_to_gib(self.max_vram_seen) if self.max_vram_seen is not None else None,
            "baseline_vram_bytes": self.baseline_vram,
            "additional_vram_used_bytes": additional_vram,
            "additional_vram_used_gib": bytes_to_gib(additional_vram) if additional_vram is not None else None,
            "total_vram_bytes": self.total_vram,
        }

# ==========================================
# 4. INFERENCE LOOP
# ==========================================

def export_metrics_to_prometheus(result: Dict[str, Any]) -> None:
    """
    Export inference metrics to the Ollama exporter for Prometheus.

    >>> AIPE NOTE: Metrics Pipeline
    This pushes metrics to our custom exporter which aggregates them for Prometheus.
    Enables real-time dashboards in Grafana showing TPS, TTFT, TPOT trends.
    """
    try:
        metrics_payload = {
            "eval_count": result.get("eval_count", 0),
            "eval_duration": result.get("eval_duration", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "prompt_eval_duration": result.get("prompt_eval_duration", 0),
        }
        requests.post(METRICS_EXPORTER_URL, json=metrics_payload, timeout=2)
    except Exception:
        pass  # Don't fail benchmark if metrics export fails


def run_single_request(
    model: str,
    prompt: str,
    sampler: Any,
    temperature: float,
    timeout: int,
    quantization: Optional[str] = None,
    request_num: int = 0,
    total_requests: int = 0,
) -> Dict[str, Any]:
    """
    Run a single inference request with OpenTelemetry tracing.

    >>> AIPE NOTE: Span Attributes
    We use semantic conventions for LLM observability:
    - llm.model: Model identifier
    - llm.request_type: "completion" or "chat"
    - llm.prompt_tokens: Input token count
    - llm.completion_tokens: Output token count
    - llm.tokens_per_second: Throughput metric
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
        },
    }

    # Create span context if tracing is enabled
    span_context = None
    if TRACING_ENABLED and tracer:
        span_context = tracer.start_as_current_span(
            "llm.inference",
            attributes={
                "llm.model": model,
                "llm.request_type": "completion",
                "llm.quantization": quantization or "unknown",
                "llm.temperature": temperature,
                "benchmark.request_num": request_num,
                "benchmark.total_requests": total_requests,
            }
        )
        span_context.__enter__()

    sampler.activate()
    start = time.time()

    try:
        # >>> AIPE NOTE: Latency Measurement
        # We measure "Client-Side Latency" here. This includes Network RTT + Server Queue + Inference Time.
        # For pure model benchmarking, server-side metrics (TTFT, TPOT) are more accurate.
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        total_time = time.time() - start

        eval_count = result.get("eval_count", 0)
        eval_duration = result.get("eval_duration", 0)
        prompt_eval_count = result.get("prompt_eval_count", 0)
        prompt_eval_duration = result.get("prompt_eval_duration", 0)

        tokens_per_second = None
        if eval_count and eval_duration:
            tokens_per_second = eval_count / (eval_duration / 1e9)

        ttft = None
        if prompt_eval_duration:
            ttft = prompt_eval_duration / 1e9

        tpot = None
        if eval_count and eval_duration:
            tpot = (eval_duration / 1e6) / eval_count  # ms per token

        # Add span attributes for successful request
        if span_context:
            span = trace.get_current_span()
            span.set_attribute("llm.prompt_tokens", prompt_eval_count)
            span.set_attribute("llm.completion_tokens", eval_count)
            span.set_attribute("llm.tokens_per_second", tokens_per_second or 0)
            span.set_attribute("llm.ttft_seconds", ttft or 0)
            span.set_attribute("llm.tpot_ms", tpot or 0)
            span.set_attribute("llm.total_latency_seconds", total_time)
            span.set_status(Status(StatusCode.OK))

        # Export to Prometheus metrics
        export_metrics_to_prometheus({
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
        })

        return {
            "success": True,
            "latency_seconds": total_time,
            "tokens_generated": eval_count,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token": ttft,
            "time_per_output_token_ms": tpot,
            "prompt_tokens": prompt_eval_count,
            "response_length": len(result.get("response", "")),
        }
    except Exception as exc:
        total_time = time.time() - start

        # Mark span as error
        if span_context:
            span = trace.get_current_span()
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.set_attribute("error.message", str(exc))

        return {
            "success": False,
            "error": str(exc),
            "latency_seconds": total_time,
        }
    finally:
        sampler.deactivate()
        if span_context:
            span_context.__exit__(None, None, None)

def aggregate_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [r for r in records if r.get("success")]
    latencies = [r["latency_seconds"] for r in successful]
    tps_values = [r["tokens_per_second"] for r in successful if r.get("tokens_per_second")]
    ttft_values = [r["time_to_first_token"] for r in successful if r.get("time_to_first_token")]

    return {
        "total_requests": len(records),
        "successful_requests": len(successful),
        "latency_seconds": calc_stats(latencies),
        "tokens_per_second": calc_stats(tps_values),
        "time_to_first_token_seconds": calc_stats(ttft_values),
        "avg_tokens_generated": statistics.mean([r["tokens_generated"] for r in successful]) if successful else 0,
    }

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main() -> None:
    args = parse_args()

    # Initialize OpenTelemetry tracing
    if not args.no_tracing:
        init_tracing(service_name="llm-benchmark")

    quantization = detect_quantization(args.model)

    telemetry: Any
    if shutil.which("rocm-smi") or shutil.which("nvidia-smi"):
        telemetry = GPUTelemetrySampler(interval=0.5)
    else:
        telemetry = NullSampler()

    telemetry.start()

    records: List[Dict[str, Any]] = []

    # Create parent span for entire benchmark run
    benchmark_span = None
    if TRACING_ENABLED and tracer:
        benchmark_span = tracer.start_as_current_span(
            "benchmark.run",
            attributes={
                "benchmark.model": args.model,
                "benchmark.quantization": quantization or "unknown",
                "benchmark.num_requests": args.requests,
                "benchmark.temperature": args.temperature,
            }
        )
        benchmark_span.__enter__()

    print(f"Running {args.requests} requests against {args.model} (quantization={quantization or 'unknown'})")
    for idx in range(1, args.requests + 1):
        print(f"Request {idx}/{args.requests}...", end=" ", flush=True)
        result = run_single_request(
            model=args.model,
            prompt=args.prompt,
            sampler=telemetry,
            temperature=args.temperature,
            timeout=args.timeout,
            quantization=quantization,
            request_num=idx,
            total_requests=args.requests,
        )
        records.append(result)
        if result.get("success"):
            tps = result.get("tokens_per_second")
            ttft = result.get("time_to_first_token")
            print(f"done ({result['latency_seconds']:.2f}s, TPS={tps:.1f} TTFT={ttft:.2f}s)" if (tps and ttft) else f"done ({result['latency_seconds']:.2f}s)")
        else:
            print("failed")

    telemetry.stop()

    aggregated = aggregate_results(records)
    telemetry_summary = telemetry.summary()

    # Add aggregated results to benchmark span
    if benchmark_span:
        span = trace.get_current_span()
        if aggregated.get("tokens_per_second"):
            span.set_attribute("benchmark.avg_tps", aggregated["tokens_per_second"].get("mean", 0))
        if aggregated.get("time_to_first_token_seconds"):
            span.set_attribute("benchmark.avg_ttft", aggregated["time_to_first_token_seconds"].get("mean", 0))
        span.set_attribute("benchmark.successful_requests", aggregated.get("successful_requests", 0))
        benchmark_span.__exit__(None, None, None)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "quantization": quantization,
        "prompt": args.prompt,
        "num_requests": args.requests,
        "records": records,
        "aggregated": aggregated,
        "telemetry": telemetry_summary,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"quantization_benchmark_{sanitize_filename(args.model)}_{timestamp}.json"
        output_path = Path(fname)

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {output_path}")

    if TRACING_ENABLED:
        print("Traces exported to Tempo. View in Grafana -> Explore -> Tempo")


if __name__ == "__main__":
    main()

