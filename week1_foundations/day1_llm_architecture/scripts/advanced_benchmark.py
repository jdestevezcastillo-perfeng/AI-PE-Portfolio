"""
Advanced LLM Benchmarking Script
Tests various scenarios: concurrent requests, different prompt lengths, temperature variations
"""

import time
import statistics
import json
from datetime import datetime
import requests
from typing import List, Dict
import concurrent.futures
import psutil
import os

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


# Prompt categories with varying token lengths
PROMPT_CATEGORIES = {
    "tiny": [
        "Hi",
        "Hello",
        "Test",
    ],
    "short": [
        "What is AI?",
        "Explain Python.",
        "Define recursion.",
    ],
    "medium": [
        "Explain the concept of machine learning in simple terms.",
        "What are the key differences between supervised and unsupervised learning?",
        "Describe how neural networks work.",
    ],
    "long": [
        "Describe the architecture of a transformer model and explain how self-attention works in detail.",
        "Explain the training process of large language models, including pre-training and fine-tuning phases.",
        "What are the main performance bottlenecks in LLM inference and how can they be addressed?",
    ],
    "very_long": [
        "Provide a comprehensive explanation of the transformer architecture, including multi-head attention, "
        "position encodings, feed-forward layers, and residual connections. Then explain how this architecture "
        "enables parallel processing during training and the challenges it faces during inference.",
    ]
}


def generate_text(prompt: str, model: str = MODEL_NAME, temperature: float = 0.7) -> Dict:
    """Send generation request to Ollama."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
        }
    }

    start_time = time.time()

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()

        end_time = time.time()
        total_time = end_time - start_time

        result = response.json()

        return {
            "success": True,
            "response_text": result.get("response", ""),
            "total_time": total_time,
            "eval_count": result.get("eval_count", 0),
            "eval_duration": result.get("eval_duration", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "prompt_eval_duration": result.get("prompt_eval_duration", 0),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "total_time": time.time() - start_time,
        }


def get_system_stats() -> Dict:
    """Capture current system resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }


def benchmark_prompt_length_impact(num_requests: int = 20) -> Dict:
    """
    Test how prompt length affects latency and throughput.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Prompt Length Impact")
    print("="*60)

    results = {}

    for category, prompts in PROMPT_CATEGORIES.items():
        print(f"\nTesting {category} prompts...")

        latencies = []
        ttfts = []
        tokens_per_second = []

        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            result = generate_text(prompt)

            if result["success"]:
                latencies.append(result["total_time"])

                if result["prompt_eval_duration"] > 0:
                    ttfts.append(result["prompt_eval_duration"] / 1e9)

                if result["eval_count"] > 0 and result["eval_duration"] > 0:
                    tps = result["eval_count"] / (result["eval_duration"] / 1e9)
                    tokens_per_second.append(tps)

        results[category] = {
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg_ttft": statistics.mean(ttfts) if ttfts else 0,
            "avg_tps": statistics.mean(tokens_per_second) if tokens_per_second else 0,
            "num_samples": len(latencies),
        }

        print(f"  Avg Latency: {results[category]['avg_latency']:.3f}s")
        print(f"  Avg TTFT: {results[category]['avg_ttft']:.3f}s")
        print(f"  Avg TPS: {results[category]['avg_tps']:.1f} tok/s")

    return results


def benchmark_temperature_impact(num_requests: int = 15) -> Dict:
    """
    Test how temperature setting affects generation time.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Temperature Impact")
    print("="*60)

    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    test_prompt = "Explain the concept of temperature in language model generation."

    results = {}

    for temp in temperatures:
        print(f"\nTesting temperature={temp}...")

        latencies = []
        tokens_per_second = []

        for i in range(num_requests):
            result = generate_text(test_prompt, temperature=temp)

            if result["success"]:
                latencies.append(result["total_time"])

                if result["eval_count"] > 0 and result["eval_duration"] > 0:
                    tps = result["eval_count"] / (result["eval_duration"] / 1e9)
                    tokens_per_second.append(tps)

        results[f"temp_{temp}"] = {
            "temperature": temp,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "avg_tps": statistics.mean(tokens_per_second) if tokens_per_second else 0,
        }

        print(f"  Avg Latency: {results[f'temp_{temp}']['avg_latency']:.3f}s")
        print(f"  Avg TPS: {results[f'temp_{temp}']['avg_tps']:.1f} tok/s")

    return results


def benchmark_concurrent_requests(max_workers: int = 4, requests_per_batch: int = 10) -> Dict:
    """
    Test concurrent request handling (simulates multiple users).
    Note: Ollama by default processes requests sequentially, but this tests queueing behavior.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Concurrent Request Handling")
    print("="*60)

    test_prompt = "What is machine learning?"

    results = {}

    for num_workers in [1, 2, 4]:
        if num_workers > max_workers:
            continue

        print(f"\nTesting with {num_workers} concurrent worker(s)...")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(generate_text, test_prompt)
                for _ in range(requests_per_batch)
            ]

            results_list = [f.result() for f in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time

        successful = [r for r in results_list if r.get("success")]
        latencies = [r["total_time"] for r in successful]

        results[f"{num_workers}_workers"] = {
            "num_workers": num_workers,
            "total_time": total_time,
            "requests_per_second": len(successful) / total_time,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "successful_requests": len(successful),
            "total_requests": requests_per_batch,
        }

        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {results[f'{num_workers}_workers']['requests_per_second']:.2f} req/s")
        print(f"  Avg latency: {results[f'{num_workers}_workers']['avg_latency']:.3f}s")

    return results


def benchmark_resource_usage(duration_seconds: int = 60) -> Dict:
    """
    Monitor system resource usage during sustained inference.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Resource Usage Monitoring")
    print(f"Duration: {duration_seconds}s")
    print("="*60)

    test_prompt = "Explain machine learning."

    cpu_samples = []
    memory_samples = []
    request_count = 0

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        # Capture system stats before request
        stats = get_system_stats()
        cpu_samples.append(stats["cpu_percent"])
        memory_samples.append(stats["memory_used_gb"])

        # Make request
        result = generate_text(test_prompt)
        if result["success"]:
            request_count += 1

        print(f"\rRequests: {request_count} | CPU: {stats['cpu_percent']:.1f}% | "
              f"Memory: {stats['memory_used_gb']:.2f}GB", end="", flush=True)

    print()  # New line after progress

    results = {
        "duration_seconds": duration_seconds,
        "total_requests": request_count,
        "requests_per_second": request_count / duration_seconds,
        "cpu_usage": {
            "min": min(cpu_samples),
            "max": max(cpu_samples),
            "avg": statistics.mean(cpu_samples),
        },
        "memory_usage_gb": {
            "min": min(memory_samples),
            "max": max(memory_samples),
            "avg": statistics.mean(memory_samples),
        },
    }

    print(f"\nAvg CPU usage: {results['cpu_usage']['avg']:.1f}%")
    print(f"Avg Memory usage: {results['memory_usage_gb']['avg']:.2f}GB")
    print(f"Throughput: {results['requests_per_second']:.2f} req/s")

    return results


def main():
    """Run all advanced benchmarks."""
    print("="*60)
    print("LLM ADVANCED PERFORMANCE BENCHMARK SUITE")
    print("="*60)
    print(f"Model: {MODEL_NAME}")

    # Test connectivity
    print("\nTesting API connectivity...")
    test_result = generate_text("Hi")
    if not test_result or not test_result.get("success"):
        print("ERROR: Cannot connect to Ollama API.")
        return
    print("✓ API connection successful\n")

    # Run experiments
    all_results = {
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "system_info": get_system_stats(),
    }

    # Experiment 1: Prompt length
    all_results["prompt_length_experiment"] = benchmark_prompt_length_impact(num_requests=20)

    # Experiment 2: Temperature
    all_results["temperature_experiment"] = benchmark_temperature_impact(num_requests=15)

    # Experiment 3: Concurrent requests
    all_results["concurrent_experiment"] = benchmark_concurrent_requests(max_workers=4, requests_per_batch=10)

    # Experiment 4: Resource usage (shorter duration for demo)
    all_results["resource_usage_experiment"] = benchmark_resource_usage(duration_seconds=30)

    # Save results
    output_file = f"advanced_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_file}")
    print(f"\nNext: python visualize_results.py {output_file}")


if __name__ == "__main__":
    main()
