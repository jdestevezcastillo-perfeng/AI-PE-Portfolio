"""
LAB 04: BENCHMARK SUITE
-----------------------
Goal: Compare inference engines (vLLM vs TGI) using a unified benchmark.
We will measure:
1. Time To First Token (TTFT) - Latency
2. Inter-Token Latency (ITL) - Smoothness
3. End-to-End Latency
4. Throughput (Tokens/second)

Usage:
    python lab_04_benchmark_suite.py --url http://localhost:8000/v1 --model <model_id>
"""

import time
import json
import requests
import argparse
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

def benchmark_request(url, model, prompt, max_tokens=100):
    """
    Sends a single request and measures timing metrics.
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
    
    try:
        # Adjust endpoint if needed (TGI might be /generate_stream or /v1/chat/completions)
        # We assume OpenAI compatible /v1/chat/completions for both vLLM and newer TGI
        endpoint = f"{url}/chat/completions"
        
        with requests.post(endpoint, headers=headers, json=data, stream=True) as response:
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
                        else:
                            token_times.append(current_time - last_token_time)
                        
                        last_token_time = current_time
                        
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    end_time = time.time()
    total_duration = end_time - start_time
    num_tokens = len(token_times) + 1 # +1 for the first token
    
    return {
        "ttft": ttft,
        "itl_mean": np.mean(token_times) if token_times else 0,
        "throughput": num_tokens / total_duration,
        "total_time": total_duration,
        "tokens": num_tokens
    }

def run_benchmark(url, model, concurrency=1, num_requests=10):
    print(f"--- Benchmarking {model} at {url} ---")
    print(f"Concurrency: {concurrency}, Total Requests: {num_requests}")
    
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to love.",
        "What are the benefits of eating vegetables?",
        "Describe the architecture of a Transformer model.",
        "How do I reverse a linked list in Python?"
    ] * (num_requests // 5 + 1)
    prompts = prompts[:num_requests]
    
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(benchmark_request, url, model, p) for p in prompts]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)

    if not results:
        print("No successful requests.")
        return

    # Aggregated Metrics
    avg_ttft = np.mean([r["ttft"] for r in results])
    avg_itl = np.mean([r["itl_mean"] for r in results])
    avg_throughput = np.mean([r["throughput"] for r in results])
    total_tokens = sum([r["tokens"] for r in results])
    
    print("\n==========================================")
    print(f"RESULTS: {model}")
    print("==========================================")
    print(f"Avg TTFT (Latency):      {avg_ttft*1000:.2f} ms")
    print(f"Avg ITL (Smoothness):    {avg_itl*1000:.2f} ms")
    print(f"Avg Throughput (User):   {avg_throughput:.2f} tokens/sec")
    print(f"Total Tokens Generated:  {total_tokens}")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="Base URL of the API")
    parser.add_argument("--model", type=str, required=True, help="Model ID to request")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--requests", type=int, default=5, help="Total number of requests")
    
    args = parser.parse_args()
    
    run_benchmark(args.url, args.model, args.concurrency, args.requests)
