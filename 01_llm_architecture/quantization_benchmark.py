"""
Quantization Benchmark Script

Runs a short benchmark against an Ollama-served model while capturing:
- End-to-end latency
- Tokens/sec (decode)
- Time-to-first-token (TTFT)
- GPU telemetry via rocm-smi (VRAM + utilization)

Use this to compare multiple quantization variants of the same base model.
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

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")


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


class NullSampler:
    """Fallback sampler when rocm-smi is unavailable."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def summary(self) -> Dict[str, Any]:
        return {
            "available": False,
        }


class GPUTelemetrySampler:
    """
    Samples GPU utilization and VRAM usage via rocm-smi.

    Only records samples while `activate()` has been called so we avoid idle data.
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.available = shutil.which("rocm-smi") is not None

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
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

        output = result.stdout.strip()
        if not output:
            return None

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None

        card = next(iter(data.values()), None)
        if not card:
            return None

        try:
            gpu_pct = float(card.get("GPU use (%)", "0").strip("%"))
            vram_used = int(card.get("VRAM Total Used Memory (B)", "0"))
            total_vram = int(card.get("VRAM Total Memory (B)", "0"))
        except (ValueError, AttributeError):
            return None

        return {
            "gpu_pct": gpu_pct,
            "vram_used": vram_used,
            "vram_total": total_vram,
        }

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


def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024 ** 3)


def run_single_request(model: str, prompt: str, sampler: Any, temperature: float, timeout: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
        },
    }

    sampler.activate()
    start = time.time()

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        total_time = time.time() - start

        eval_count = result.get("eval_count", 0)
        eval_duration = result.get("eval_duration", 0)
        prompt_eval_duration = result.get("prompt_eval_duration", 0)

        tokens_per_second = None
        if eval_count and eval_duration:
            tokens_per_second = eval_count / (eval_duration / 1e9)

        ttft = None
        if prompt_eval_duration:
            ttft = prompt_eval_duration / 1e9

        return {
            "success": True,
            "latency_seconds": total_time,
            "tokens_generated": eval_count,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token": ttft,
            "response_length": len(result.get("response", "")),
        }
    except Exception as exc:
        total_time = time.time() - start
        return {
            "success": False,
            "error": str(exc),
            "latency_seconds": total_time,
        }
    finally:
        sampler.deactivate()


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


def sanitize_filename(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)
    return safe.strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an Ollama model with GPU telemetry.")
    parser.add_argument("--model", default="llama3.1:8b", help="Model name registered with Ollama.")
    parser.add_argument("--prompt", default="Explain how quantization affects LLM inference performance.", help="Prompt to benchmark.")
    parser.add_argument("--requests", type=int, default=5, help="Number of sequential requests to run.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout seconds.")
    parser.add_argument("--output", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    quantization = detect_quantization(args.model)

    telemetry: Any
    if shutil.which("rocm-smi"):
        telemetry = GPUTelemetrySampler(interval=0.5)
    else:
        telemetry = NullSampler()

    telemetry.start()

    records: List[Dict[str, Any]] = []

    print(f"Running {args.requests} requests against {args.model} (quantization={quantization or 'unknown'})")
    for idx in range(1, args.requests + 1):
        print(f"Request {idx}/{args.requests}...", end=" ", flush=True)
        result = run_single_request(
            model=args.model,
            prompt=args.prompt,
            sampler=telemetry,
            temperature=args.temperature,
            timeout=args.timeout,
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


if __name__ == "__main__":
    main()
