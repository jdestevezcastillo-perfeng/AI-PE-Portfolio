"""
Visualization Script for LLM Benchmark Results
Creates charts and graphs from benchmark JSON files
"""

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(filename: str):
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_latency_distribution(results, output_dir="plots"):
    """
    Plot latency distribution from basic benchmark results.
    """
    Path(output_dir).mkdir(exist_ok=True)

    if isinstance(results, list):
        # Basic benchmark format
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))

        if len(results) == 1:
            axes = [axes]

        for idx, result in enumerate(results):
            metrics = result['latency_seconds']

            # Create bar chart of percentiles
            percentiles = ['min', 'p50', 'p90', 'p95', 'p99', 'max']
            values = [metrics[p] for p in percentiles]

            axes[idx].bar(percentiles, values, color='steelblue', alpha=0.7)
            axes[idx].set_ylabel('Latency (seconds)')
            axes[idx].set_title(f'Latency Distribution - "{result["prompt"][:40]}..."')
            axes[idx].grid(True, alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.3f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/latency_distribution.png")
        plt.close()


def plot_tokens_per_second(results, output_dir="plots"):
    """
    Plot tokens per second metrics.
    """
    Path(output_dir).mkdir(exist_ok=True)

    if isinstance(results, list):
        fig, ax = plt.subplots(figsize=(12, 6))

        prompt_labels = [f"Prompt {i+1}" for i in range(len(results))]
        tps_means = [r['tokens_per_second'].get('mean', 0) for r in results]
        tps_medians = [r['tokens_per_second'].get('median', 0) for r in results]

        x = np.arange(len(prompt_labels))
        width = 0.35

        ax.bar(x - width/2, tps_means, width, label='Mean', color='lightcoral', alpha=0.7)
        ax.bar(x + width/2, tps_medians, width, label='Median', color='skyblue', alpha=0.7)

        ax.set_ylabel('Tokens per Second')
        ax.set_title('Token Generation Throughput (Decode Phase)')
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/tokens_per_second.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/tokens_per_second.png")
        plt.close()


def plot_ttft_comparison(results, output_dir="plots"):
    """
    Plot Time to First Token (TTFT) comparison.
    """
    Path(output_dir).mkdir(exist_ok=True)

    if isinstance(results, list):
        fig, ax = plt.subplots(figsize=(10, 6))

        prompt_labels = [f"Prompt {i+1}\n({r['prompt'][:20]}...)" for i, r in enumerate(results)]
        ttft_means = [r['time_to_first_token_seconds'].get('mean', 0) for r in results]

        ax.barh(prompt_labels, ttft_means, color='mediumseagreen', alpha=0.7)
        ax.set_xlabel('Time to First Token (seconds)')
        ax.set_title('TTFT - Prefill Phase Latency')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(ttft_means):
            ax.text(v, i, f' {v:.3f}s', va='center')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/ttft_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/ttft_comparison.png")
        plt.close()


def plot_advanced_results(results, output_dir="plots"):
    """
    Plot results from advanced benchmark experiments.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Experiment 1: Prompt Length Impact
    if 'prompt_length_experiment' in results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        exp = results['prompt_length_experiment']
        categories = list(exp.keys())

        # Latency by prompt length
        latencies = [exp[cat]['avg_latency'] for cat in categories]
        axes[0, 0].bar(categories, latencies, color='steelblue', alpha=0.7)
        axes[0, 0].set_ylabel('Avg Latency (s)')
        axes[0, 0].set_title('Latency vs Prompt Length')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # TTFT by prompt length
        ttfts = [exp[cat]['avg_ttft'] for cat in categories]
        axes[0, 1].bar(categories, ttfts, color='coral', alpha=0.7)
        axes[0, 1].set_ylabel('Avg TTFT (s)')
        axes[0, 1].set_title('Time to First Token vs Prompt Length')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # TPS by prompt length
        tps = [exp[cat]['avg_tps'] for cat in categories]
        axes[1, 0].bar(categories, tps, color='lightgreen', alpha=0.7)
        axes[1, 0].set_ylabel('Avg TPS (tokens/s)')
        axes[1, 0].set_title('Throughput vs Prompt Length')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # P95 latency
        p95s = [exp[cat]['p95_latency'] for cat in categories]
        axes[1, 1].bar(categories, p95s, color='mediumpurple', alpha=0.7)
        axes[1, 1].set_ylabel('P95 Latency (s)')
        axes[1, 1].set_title('P95 Latency vs Prompt Length')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/prompt_length_impact.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/prompt_length_impact.png")
        plt.close()

    # Experiment 2: Temperature Impact
    if 'temperature_experiment' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        exp = results['temperature_experiment']
        temps = [exp[key]['temperature'] for key in exp.keys()]
        latencies = [exp[key]['avg_latency'] for key in exp.keys()]
        tps = [exp[key]['avg_tps'] for key in exp.keys()]

        axes[0].plot(temps, latencies, marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel('Temperature')
        axes[0].set_ylabel('Avg Latency (s)')
        axes[0].set_title('Latency vs Temperature')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(temps, tps, marker='s', linewidth=2, markersize=8, color='coral')
        axes[1].set_xlabel('Temperature')
        axes[1].set_ylabel('Avg TPS (tokens/s)')
        axes[1].set_title('Throughput vs Temperature')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/temperature_impact.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/temperature_impact.png")
        plt.close()

    # Experiment 3: Concurrent Requests
    if 'concurrent_experiment' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        exp = results['concurrent_experiment']
        workers = [exp[key]['num_workers'] for key in exp.keys()]
        throughputs = [exp[key]['requests_per_second'] for key in exp.keys()]
        latencies = [exp[key]['avg_latency'] for key in exp.keys()]

        axes[0].bar(range(len(workers)), throughputs, tick_label=workers, color='lightgreen', alpha=0.7)
        axes[0].set_xlabel('Number of Workers')
        axes[0].set_ylabel('Requests per Second')
        axes[0].set_title('Throughput vs Concurrency')
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(range(len(workers)), latencies, tick_label=workers, color='salmon', alpha=0.7)
        axes[1].set_xlabel('Number of Workers')
        axes[1].set_ylabel('Avg Latency (s)')
        axes[1].set_title('Latency vs Concurrency')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/concurrent_impact.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/concurrent_impact.png")
        plt.close()


def create_summary_report(results, output_file="benchmark_summary.txt"):
    """
    Create a text summary report of benchmark results.
    """
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LLM BENCHMARK SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        if isinstance(results, list):
            # Basic benchmark
            for idx, result in enumerate(results, 1):
                f.write(f"Test {idx}: {result['prompt'][:50]}...\n")
                f.write(f"  Requests: {result['num_requests']}\n")
                f.write(f"  Total tokens: {result['total_tokens_generated']}\n")

                lat = result['latency_seconds']
                f.write(f"  Latency (mean): {lat['mean']:.3f}s\n")
                f.write(f"  Latency (p95): {lat['p95']:.3f}s\n")

                tps = result['tokens_per_second']
                f.write(f"  TPS (mean): {tps.get('mean', 0):.1f} tok/s\n")

                ttft = result['time_to_first_token_seconds']
                f.write(f"  TTFT (mean): {ttft.get('mean', 0):.3f}s\n")
                f.write("\n")

        elif isinstance(results, dict):
            # Advanced benchmark
            f.write(f"Model: {results.get('model', 'Unknown')}\n")
            f.write(f"Timestamp: {results.get('timestamp', 'Unknown')}\n\n")

            if 'prompt_length_experiment' in results:
                f.write("-" * 70 + "\n")
                f.write("PROMPT LENGTH EXPERIMENT\n")
                f.write("-" * 70 + "\n")
                for cat, data in results['prompt_length_experiment'].items():
                    f.write(f"\n{cat.upper()}:\n")
                    f.write(f"  Avg Latency: {data['avg_latency']:.3f}s\n")
                    f.write(f"  Avg TTFT: {data['avg_ttft']:.3f}s\n")
                    f.write(f"  Avg TPS: {data['avg_tps']:.1f} tok/s\n")

            if 'temperature_experiment' in results:
                f.write("\n" + "-" * 70 + "\n")
                f.write("TEMPERATURE EXPERIMENT\n")
                f.write("-" * 70 + "\n")
                for key, data in results['temperature_experiment'].items():
                    f.write(f"\nTemperature {data['temperature']}:\n")
                    f.write(f"  Avg Latency: {data['avg_latency']:.3f}s\n")
                    f.write(f"  Avg TPS: {data['avg_tps']:.1f} tok/s\n")

            if 'concurrent_experiment' in results:
                f.write("\n" + "-" * 70 + "\n")
                f.write("CONCURRENT REQUEST EXPERIMENT\n")
                f.write("-" * 70 + "\n")
                for key, data in results['concurrent_experiment'].items():
                    f.write(f"\n{data['num_workers']} worker(s):\n")
                    f.write(f"  Throughput: {data['requests_per_second']:.2f} req/s\n")
                    f.write(f"  Avg Latency: {data['avg_latency']:.3f}s\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Summary report saved to: {output_file}")


def main():
    """Main visualization function."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <benchmark_results.json>")
        print("\nLooking for JSON files in current directory...")

        json_files = list(Path(".").glob("*benchmark*.json"))
        if json_files:
            print("Found:")
            for f in json_files:
                print(f"  - {f}")
            print(f"\nUsing most recent: {json_files[-1]}")
            results_file = str(json_files[-1])
        else:
            print("No benchmark JSON files found.")
            return
    else:
        results_file = sys.argv[1]

    print(f"Loading results from: {results_file}")
    results = load_results(results_file)

    print("\nGenerating visualizations...")

    # Determine result type and plot accordingly
    if isinstance(results, list):
        # Basic benchmark
        plot_latency_distribution(results)
        plot_tokens_per_second(results)
        plot_ttft_comparison(results)
    elif isinstance(results, dict):
        # Advanced benchmark
        plot_advanced_results(results)

    # Create summary report
    create_summary_report(results)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("Check the 'plots/' directory for generated charts")
    print("Check 'benchmark_summary.txt' for text report")


if __name__ == "__main__":
    main()
