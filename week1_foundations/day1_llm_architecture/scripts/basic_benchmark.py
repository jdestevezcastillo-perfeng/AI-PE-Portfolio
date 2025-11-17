"""
Basic LLM Benchmarking Script
Measures latency for sequential requests to Ollama
"""

import time
import statistics
import json
from datetime import datetime
import requests
from typing import List, Dict

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
NUM_REQUESTS = 100

# Test prompts of varying lengths
TEST_PROMPTS = [
    "What is AI?",  # Short
    "Explain the concept of machine learning in simple terms.",  # Medium
    "Describe the architecture of a transformer model and explain how self-attention works.",  # Long
]


def generate_text(prompt: str, model: str = MODEL_NAME) -> Dict:
    """
    Send a generation request to Ollama and measure timing.

    Returns dict with:
        - response_text: generated text
        - total_time: end-to-end latency
        - eval_count: number of tokens generated
        - eval_duration: time spent in evaluation (nanoseconds)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }

    start_time = time.time()

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()

        end_time = time.time()
        total_time = end_time - start_time

        result = response.json()

        return {
            "response_text": result.get("response", ""),
            "total_time": total_time,
            "eval_count": result.get("eval_count", 0),
            "eval_duration": result.get("eval_duration", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "prompt_eval_duration": result.get("prompt_eval_duration", 0),
        }
    except Exception as e:
        print(f"Error during generation: {e}")
        return None


def calculate_metrics(latencies: List[float]) -> Dict:
    """Calculate statistical metrics from latency measurements."""
    if not latencies:
        return {}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "count": n,
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies) if n > 1 else 0,
        "p50": sorted_latencies[int(n * 0.50)],
        "p90": sorted_latencies[int(n * 0.90)],
        "p95": sorted_latencies[int(n * 0.95)],
        "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
    }


def run_benchmark(prompt: str, num_requests: int = NUM_REQUESTS) -> Dict:
    """
    Run benchmark for a given prompt.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking with prompt: '{prompt[:50]}...'")
    print(f"Number of requests: {num_requests}")
    print(f"{'='*60}\n")

    latencies = []
    tokens_generated = []
    tokens_per_second = []
    time_to_first_token = []  # TTFT

    for i in range(num_requests):
        print(f"Request {i+1}/{num_requests}...", end=" ", flush=True)

        result = generate_text(prompt)

        if result:
            latency = result["total_time"]
            latencies.append(latency)

            eval_count = result["eval_count"]
            tokens_generated.append(eval_count)

            # Calculate tokens per second (decode phase)
            if eval_count > 0 and result["eval_duration"] > 0:
                tps = eval_count / (result["eval_duration"] / 1e9)  # Convert ns to seconds
                tokens_per_second.append(tps)

            # Time to first token (TTFT) = prefill time
            if result["prompt_eval_duration"] > 0:
                ttft = result["prompt_eval_duration"] / 1e9
                time_to_first_token.append(ttft)

            print(f"✓ ({latency:.2f}s, {eval_count} tokens)")
        else:
            print("✗ Failed")

    # Calculate metrics
    latency_metrics = calculate_metrics(latencies)
    tps_metrics = calculate_metrics(tokens_per_second)
    ttft_metrics = calculate_metrics(time_to_first_token)

    results = {
        "prompt": prompt,
        "num_requests": num_requests,
        "timestamp": datetime.now().isoformat(),
        "latency_seconds": latency_metrics,
        "tokens_per_second": tps_metrics,
        "time_to_first_token_seconds": ttft_metrics,
        "total_tokens_generated": sum(tokens_generated),
        "avg_tokens_per_request": statistics.mean(tokens_generated) if tokens_generated else 0,
    }

    return results


def print_results(results: Dict):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Prompt: {results['prompt'][:50]}...")
    print(f"Total requests: {results['num_requests']}")
    print(f"Total tokens generated: {results['total_tokens_generated']}")
    print(f"Avg tokens per request: {results['avg_tokens_per_request']:.1f}")

    print(f"\n--- End-to-End Latency (seconds) ---")
    lat = results['latency_seconds']
    print(f"  Min:    {lat['min']:.3f}s")
    print(f"  Mean:   {lat['mean']:.3f}s")
    print(f"  Median: {lat['median']:.3f}s")
    print(f"  P90:    {lat['p90']:.3f}s")
    print(f"  P95:    {lat['p95']:.3f}s")
    print(f"  P99:    {lat['p99']:.3f}s")
    print(f"  Max:    {lat['max']:.3f}s")

    print(f"\n--- Tokens Per Second (Decode) ---")
    tps = results['tokens_per_second']
    if tps:
        print(f"  Min:    {tps['min']:.1f} tok/s")
        print(f"  Mean:   {tps['mean']:.1f} tok/s")
        print(f"  Median: {tps['median']:.1f} tok/s")
        print(f"  Max:    {tps['max']:.1f} tok/s")

    print(f"\n--- Time to First Token / TTFT (seconds) ---")
    ttft = results['time_to_first_token_seconds']
    if ttft:
        print(f"  Min:    {ttft['min']:.3f}s")
        print(f"  Mean:   {ttft['mean']:.3f}s")
        print(f"  Median: {ttft['median']:.3f}s")
        print(f"  Max:    {ttft['max']:.3f}s")

    print(f"{'='*60}\n")


def save_results(results: Dict, filename: str = "benchmark_results.json"):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


def main():
    """Main benchmark execution."""
    print("="*60)
    print("LLM PERFORMANCE BENCHMARK - Basic Sequential Test")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"API Endpoint: {OLLAMA_API_URL}")

    # Test API connectivity
    print("\nTesting API connectivity...")
    test_result = generate_text("Hi")
    if not test_result:
        print("ERROR: Cannot connect to Ollama API. Is the service running?")
        print("Run: ollama serve")
        return
    print("✓ API connection successful\n")

    # Run benchmarks for each prompt type
    all_results = []

    for prompt in TEST_PROMPTS:
        results = run_benchmark(prompt, num_requests=NUM_REQUESTS)
        print_results(results)
        all_results.append(results)

        # Brief pause between batches
        time.sleep(2)

    # Save all results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(all_results, output_file)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review results in {output_file}")
    print(f"2. Run: python visualize_results.py {output_file}")
    print(f"3. Analyze latency patterns and bottlenecks")


if __name__ == "__main__":
    main()
