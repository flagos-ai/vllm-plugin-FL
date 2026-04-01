#!/usr/bin/env python3
"""Collect and summarize vLLM benchmark results"""

import csv
import glob
import json
import os
from pathlib import Path


def load_json(path):
    """Load JSON file"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_throughput_metrics(data):
    """Extract throughput metrics from JSON data"""
    metrics = {
        "num_prompts": data.get("num_requests") or data.get("num_prompts"),
        "total_tokens": data.get("total_num_tokens") or data.get("total_output_tokens"),
        "elapsed_time": data.get("elapsed_time"),
        "tokens_per_sec": data.get("tokens_per_second")
        or data.get("output_throughput"),
        "requests_per_sec": data.get("requests_per_second")
        or data.get("request_throughput"),
    }
    return metrics


def extract_latency_metrics(data):
    """Extract latency metrics from JSON data"""
    # Get percentiles if available
    percentiles = data.get("percentiles", {})

    metrics = {
        "num_iters": len(data.get("latencies", [])) or data.get("num_iters"),
        "mean_latency": data.get("avg_latency") or data.get("mean_latency"),
        "median_latency": percentiles.get("50")
        or data.get("median_latency")
        or data.get("p50_latency"),
        "p90_latency": percentiles.get("90") or data.get("p90_latency"),
        "p99_latency": percentiles.get("99") or data.get("p99_latency"),
        "ttft": data.get("time_to_first_token") or data.get("avg_ttft"),
    }

    # Convert seconds to milliseconds if values are in seconds (< 100)
    for key in ["mean_latency", "median_latency", "p90_latency", "p99_latency", "ttft"]:
        if metrics[key] is not None and metrics[key] < 100:
            metrics[key] *= 1000

    return metrics


def print_throughput_results(results_dir):
    """Print throughput test results"""
    print("\n" + "=" * 80)
    print("THROUGHPUT TEST RESULTS")
    print("=" * 80)

    files = sorted(glob.glob(os.path.join(results_dir, "throughput_*.json")))
    if not files:
        print("No throughput results found")
        return []

    print(
        f"\n{'Scenario':<15} {'Prompts':<8} {'Tokens':<10} {'Time(s)':<9} {'Tokens/s':<12} {'Req/s':<10}"
    )
    print("-" * 80)

    results = []
    for path in files:
        data = load_json(path)
        if not data:
            continue

        scenario = Path(path).stem.replace("throughput_", "")
        metrics = extract_throughput_metrics(data)
        results.append({"scenario": scenario, "type": "throughput", **metrics})

        print(
            f"{scenario:<15} "
            f"{metrics['num_prompts'] or 'N/A':<8} "
            f"{metrics['total_tokens'] or 'N/A':<10} "
            f"{metrics['elapsed_time'] or 0:>7.1f}s "
            f"{metrics['tokens_per_sec'] or 0:>10.1f} "
            f"{metrics['requests_per_sec'] or 0:>9.2f}"
        )

    print("-" * 80)
    return results


def print_latency_results(results_dir):
    """Print latency test results"""
    print("\n" + "=" * 80)
    print("LATENCY TEST RESULTS")
    print("=" * 80)

    files = sorted(glob.glob(os.path.join(results_dir, "latency_*.json")))
    if not files:
        print("No latency results found")
        return []

    print(
        f"\n{'Scenario':<15} {'Iters':<6} {'Mean(ms)':<12} {'P50(ms)':<12} {'P90(ms)':<12} {'P99(ms)':<12}"
    )
    print("-" * 80)

    results = []
    for path in files:
        data = load_json(path)
        if not data:
            continue

        scenario = Path(path).stem.replace("latency_", "")
        metrics = extract_latency_metrics(data)
        results.append({"scenario": scenario, "type": "latency", **metrics})

        def fmt(v):
            return f"{v:.2f}" if v else "N/A"

        print(
            f"{scenario:<15} "
            f"{metrics['num_iters'] or 'N/A':<6} "
            f"{fmt(metrics['mean_latency']):<12} "
            f"{fmt(metrics['median_latency']):<12} "
            f"{fmt(metrics['p90_latency']):<12} "
            f"{fmt(metrics['p99_latency']):<12}"
        )

    print("-" * 80)
    return results


def print_summary(throughput_results, latency_results):
    """Print key metrics summary"""
    print("\n" + "=" * 80)
    print("KEY METRICS SUMMARY")
    print("=" * 80)

    # Best throughput
    if throughput_results:
        best_tp = max(throughput_results, key=lambda x: x.get("tokens_per_sec") or 0)
        print(
            f"\nBest Throughput:  {best_tp.get('tokens_per_sec', 0):.1f} tokens/s ({best_tp['scenario']})"
        )

    # Latency results
    if latency_results:
        print("\nLatency Results:")
        for r in latency_results:
            mean = r.get("mean_latency") or 0
            p99 = r.get("p99_latency") or 0
            print(f"  {r['scenario']:<15} Mean={mean:.1f}ms, P99={p99:.1f}ms")

    print("-" * 80)


def export_csv(throughput_results, latency_results, output_file):
    """Export all results to CSV"""
    rows = []

    for r in throughput_results:
        rows.append(
            {
                "type": "throughput",
                "scenario": r["scenario"],
                "num_prompts": r.get("num_prompts"),
                "total_tokens": r.get("total_tokens"),
                "elapsed_time_s": r.get("elapsed_time"),
                "tokens_per_sec": r.get("tokens_per_sec"),
                "requests_per_sec": r.get("requests_per_sec"),
                "mean_latency_ms": "",
                "p50_latency_ms": "",
                "p90_latency_ms": "",
                "p99_latency_ms": "",
            }
        )

    for r in latency_results:
        rows.append(
            {
                "type": "latency",
                "scenario": r["scenario"],
                "num_prompts": r.get("num_iters"),
                "total_tokens": "",
                "elapsed_time_s": "",
                "tokens_per_sec": "",
                "requests_per_sec": "",
                "mean_latency_ms": r.get("mean_latency"),
                "p50_latency_ms": r.get("median_latency"),
                "p90_latency_ms": r.get("p90_latency"),
                "p99_latency_ms": r.get("p99_latency"),
            }
        )

    if rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults exported to: {output_file}")


def export_json(throughput_results, latency_results, output_file):
    """Export all results to JSON"""
    data = {
        "throughput": throughput_results,
        "latency": latency_results,
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to: {output_file}")


def main():
    import sys

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "bench_results"

    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        print(f"Usage: python {sys.argv[0]} [results_dir]")
        return 1

    print(f"Collecting results from: {results_dir}/")

    # Collect and print results
    throughput_results = print_throughput_results(results_dir)
    latency_results = print_latency_results(results_dir)

    # Print summary
    print_summary(throughput_results, latency_results)

    # Export to files
    export_csv(throughput_results, latency_results, "bench_summary.csv")
    # export_json(throughput_results, latency_results, "bench_summary.json")

    return 0


if __name__ == "__main__":
    exit(main())
