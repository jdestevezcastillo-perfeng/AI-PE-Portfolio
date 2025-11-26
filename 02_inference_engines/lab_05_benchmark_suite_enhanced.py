"""
LAB 05: ENHANCED BENCHMARK SUITE - MULTI-MODEL SUPPORT
-------------------------------------------------------
Goal: Compare inference engines (vLLM vs TGI) with multiple models.
This enhanced version supports:
1. Multiple models (Llama-3.1-8B, Mistral-7B, etc.)
2. Engine tagging for better metrics organization
3. JSON output for automated analysis
4. Enhanced error handling and reporting

Metrics measured:
- Time To First Token (TTFT) - Latency
- Inter-Token Latency (ITL) - Smoothness
- End-to-End Latency
- Throughput (Tokens/second)

Usage:
    python lab_05_benchmark_suite_enhanced.py \
        --engine vllm \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --url http://localhost:8000/v1 \
        --requests 50 \
        --concurrency 1
"""

import time
import json
import requests
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


def benchmark_request(url, model, prompt, max_tokens=100):
    """
    Sends a single request and measures timing metrics.
    Returns detailed timing information for analysis.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True  # Essential for measuring TTFT
    }
    
    start_time = time.time()
    ttft = 0
    first_token_time = 0
    token_times = []
    last_token_time = 0
    
    try:
        endpoint = f"{url}/chat/completions"
        
        with requests.post(endpoint, headers=headers, json=data, stream=True, timeout=60) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return None

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        
                        current_time = time.time()
                        
                        # First token logic
                        if first_token_time == 0:
                            first_token_time = current_time
                            ttft = first_token_time - start_time
                            last_token_time = current_time
                        else:
                            token_times.append(current_time - last_token_time)
                            last_token_time = current_time
                        
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    end_time = time.time()
    total_duration = end_time - start_time
    num_tokens = len(token_times) + 1  # +1 for the first token
    
    return {
        "ttft": ttft,
        "itl_mean": np.mean(token_times) if token_times else 0,
        "itl_p50": np.percentile(token_times, 50) if token_times else 0,
        "itl_p99": np.percentile(token_times, 99) if token_times else 0,
        "throughput": num_tokens / total_duration if total_duration > 0 else 0,
        "total_time": total_duration,
        "tokens": num_tokens
    }


def run_benchmark(engine, url, model, concurrency=1, num_requests=10, output_dir="benchmark_results"):
    """
    Run a complete benchmark suite and save results.
    
    Args:
        engine: Engine name (vllm, tgi)
        url: Base URL of the API
        model: Model ID to request
        concurrency: Number of concurrent requests
        num_requests: Total number of requests
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {engine.upper()} - {model}")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Concurrency: {concurrency}, Total Requests: {num_requests}")
    print(f"{'='*60}\n")
    
    # Prepare diverse prompts
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to love.",
        "What are the benefits of eating vegetables?",
        "Describe the architecture of a Transformer model.",
        "How do I reverse a linked list in Python?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of gradient descent.",
        "What are the key features of Python programming language?",
        "Describe how neural networks work.",
        "What is the purpose of attention mechanisms in transformers?"
    ] * (num_requests // 10 + 1)
    prompts = prompts[:num_requests]
    
    results = []
    start_benchmark = time.time()
    
    # Run concurrent requests
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(benchmark_request, url, model, p) for p in prompts]
        for i, future in enumerate(futures, 1):
            res = future.result()
            if res:
                results.append(res)
                print(f"✓ Request {i}/{num_requests} completed ({res['tokens']} tokens, {res['ttft']*1000:.2f}ms TTFT)")
            else:
                print(f"✗ Request {i}/{num_requests} failed")

    end_benchmark = time.time()
    benchmark_duration = end_benchmark - start_benchmark

    if not results:
        print("\n❌ No successful requests. Benchmark failed.")
        return None

    # Calculate aggregated metrics
    avg_ttft = np.mean([r["ttft"] for r in results])
    p50_ttft = np.percentile([r["ttft"] for r in results], 50)
    p99_ttft = np.percentile([r["ttft"] for r in results], 99)
    
    avg_itl = np.mean([r["itl_mean"] for r in results])
    p50_itl = np.percentile([r["itl_mean"] for r in results], 50)
    p99_itl = np.percentile([r["itl_mean"] for r in results], 99)
    
    avg_throughput = np.mean([r["throughput"] for r in results])
    total_tokens = sum([r["tokens"] for r in results])
    overall_throughput = total_tokens / benchmark_duration
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {engine.upper()} - {model}")
    print(f"{'='*60}")
    print(f"Successful Requests:     {len(results)}/{num_requests}")
    print(f"Total Duration:          {benchmark_duration:.2f}s")
    print(f"Total Tokens Generated:  {total_tokens}")
    print(f"\n--- LATENCY METRICS (Lower is Better) ---")
    print(f"Avg TTFT:                {avg_ttft*1000:.2f} ms")
    print(f"P50 TTFT:                {p50_ttft*1000:.2f} ms")
    print(f"P99 TTFT:                {p99_ttft*1000:.2f} ms")
    print(f"\nAvg ITL:                 {avg_itl*1000:.2f} ms")
    print(f"P50 ITL:                 {p50_itl*1000:.2f} ms")
    print(f"P99 ITL:                 {p99_itl*1000:.2f} ms")
    print(f"\n--- THROUGHPUT METRICS (Higher is Better) ---")
    print(f"Avg Per-Request:         {avg_throughput:.2f} tokens/sec")
    print(f"Overall System:          {overall_throughput:.2f} tokens/sec")
    print(f"Request Rate:            {len(results)/benchmark_duration:.2f} req/sec")
    print(f"{'='*60}\n")
    
    # Prepare result summary
    result_summary = {
        "timestamp": datetime.now().isoformat(),
        "engine": engine,
        "model": model,
        "url": url,
        "config": {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "successful_requests": len(results)
        },
        "metrics": {
            "latency": {
                "ttft_avg_ms": avg_ttft * 1000,
                "ttft_p50_ms": p50_ttft * 1000,
                "ttft_p99_ms": p99_ttft * 1000,
                "itl_avg_ms": avg_itl * 1000,
                "itl_p50_ms": p50_itl * 1000,
                "itl_p99_ms": p99_itl * 1000
            },
            "throughput": {
                "per_request_tokens_per_sec": avg_throughput,
                "overall_tokens_per_sec": overall_throughput,
                "requests_per_sec": len(results) / benchmark_duration
            },
            "totals": {
                "total_tokens": total_tokens,
                "total_duration_sec": benchmark_duration
            }
        },
        "raw_results": results
    }
    
    # Save results to file
    Path(output_dir).mkdir(exist_ok=True)
    model_short = model.split('/')[-1]
    filename = f"{output_dir}/{engine}_{model_short}_c{concurrency}_r{num_requests}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    print(f"📊 Results saved to: {filename}\n")
    
    return result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced benchmark suite for inference engines")
    parser.add_argument("--engine", type=str, required=True, choices=["vllm", "tgi"],
                        help="Inference engine being tested")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1",
                        help="Base URL of the API")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID to request")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests")
    parser.add_argument("--requests", type=int, default=50,
                        help="Total number of requests")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    run_benchmark(
        engine=args.engine,
        url=args.url,
        model=args.model,
        concurrency=args.concurrency,
        num_requests=args.requests,
        output_dir=args.output_dir
    )
