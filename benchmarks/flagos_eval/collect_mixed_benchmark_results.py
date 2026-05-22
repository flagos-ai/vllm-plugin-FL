#!/usr/bin/env python3
"""
Collect and analyze mixed-length benchmark results for vendor acceptance testing.
"""

import csv
import glob
import json
import os
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    """Load JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_metrics(data: dict) -> dict:
    """Extract throughput metrics from benchmark result."""
    return {
        "num_prompts": data.get("num_requests") or data.get("num_prompts"),
        "total_tokens": data.get("total_num_tokens") or data.get("total_output_tokens"),
        "elapsed_time": data.get("elapsed_time"),
        "tokens_per_sec": data.get("tokens_per_second")
        or data.get("output_throughput"),
        "requests_per_sec": data.get("requests_per_second")
        or data.get("request_throughput"),
    }


def parse_scenario_name(filename: str) -> dict:
    """
    Parse scenario name to extract input/output lengths.
    Example: throughput_1k_1k.json -> {input: 1024, output: 1024}
    """
    name = Path(filename).stem.replace("throughput_", "")
    parts = name.split("_")

    def parse_size(s: str) -> int:
        s = s.lower()
        if s.endswith("k"):
            return int(s[:-1]) * 1024
        return int(s)

    if len(parts) >= 2:
        return {
            "name": name,
            "input_len": parse_size(parts[0]),
            "output_len": parse_size(parts[1]),
        }
    return {"name": name, "input_len": 0, "output_len": 0}


def collect_results(results_dir: str) -> list[dict]:
    """Collect all benchmark results from directory."""
    files = sorted(glob.glob(os.path.join(results_dir, "throughput_*.json")))

    if not files:
        print(f"No results found in {results_dir}/")
        return []

    results = []
    for path in files:
        data = load_json(path)
        if not data:
            continue

        scenario_info = parse_scenario_name(path)
        metrics = extract_metrics(data)

        results.append(
            {
                **scenario_info,
                **metrics,
            }
        )

    return results


def print_detailed_results(results: list[dict], seed: str):
    """Print detailed results for each test configuration."""
    print("\n" + "=" * 100)
    print(f"MIXED-LENGTH BENCHMARK RESULTS (Seed: {seed})")
    print("=" * 100)

    if not results:
        print("No results to display")
        return

    print(
        f"\n{'Scenario':<12} {'Input':<8} {'Output':<8} {'Samples':<8} "
        f"{'Tokens':<10} {'Time(s)':<9} {'Tokens/s':<12} {'Req/s':<10}"
    )
    print("-" * 100)

    total_prompts = 0
    total_tokens = 0
    total_time = 0

    for r in results:
        print(
            f"{r['name']:<12} "
            f"{r['input_len']:<8} "
            f"{r['output_len']:<8} "
            f"{r['num_prompts'] or 0:<8} "
            f"{r['total_tokens'] or 0:<10} "
            f"{r['elapsed_time'] or 0:>7.1f}s "
            f"{r['tokens_per_sec'] or 0:>10.1f} "
            f"{r['requests_per_sec'] or 0:>9.2f}"
        )

        total_prompts += r["num_prompts"] or 0
        total_tokens += r["total_tokens"] or 0
        total_time += r["elapsed_time"] or 0

    print("-" * 100)

    overall_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    overall_req_per_sec = total_prompts / total_time if total_time > 0 else 0

    print(
        f"{'TOTAL':<12} "
        f"{'':<8} "
        f"{'':<8} "
        f"{total_prompts:<8} "
        f"{total_tokens:<10} "
        f"{total_time:>7.1f}s "
        f"{overall_tokens_per_sec:>10.1f} "
        f"{overall_req_per_sec:>9.2f}"
    )

    print("=" * 100)


def print_summary(results: list[dict], seed: str):
    """Print summary statistics."""
    if not results:
        return

    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    total_samples = sum(r["num_prompts"] or 0 for r in results)
    total_tokens = sum(r["total_tokens"] or 0 for r in results)
    total_time = sum(r["elapsed_time"] or 0 for r in results)

    print("\nDataset Information:")
    print(f"  Seed: {seed}")
    print(f"  Total samples: {total_samples}")
    print(
        f"  Input length range: {min(r['input_len'] for r in results)} - {max(r['input_len'] for r in results)} tokens"
    )

    print("\nLength Distribution:")
    for r in sorted(results, key=lambda x: x["input_len"]):
        print(
            f"  {r['input_len']:>6}/{r['output_len']:<6} tokens: {r['num_prompts']:>4} samples"
        )

    overall_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    overall_req_per_sec = total_samples / total_time if total_time > 0 else 0

    print("\nOverall Performance:")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Total elapsed time: {total_time:.2f}s")
    print(f"  Overall throughput: {overall_tokens_per_sec:.2f} tokens/s")
    print(f"  Overall request rate: {overall_req_per_sec:.2f} req/s")

    print("\nPer-Length Performance:")
    for r in sorted(results, key=lambda x: x["input_len"]):
        print(f"  {r['name']:<12} {r['tokens_per_sec'] or 0:>10.2f} tokens/s")

    best = max(results, key=lambda x: x["tokens_per_sec"] or 0)
    worst = min(results, key=lambda x: x["tokens_per_sec"] or 0)

    print("\nPerformance Range:")
    print(f"  Best:  {best['name']:<12} {best['tokens_per_sec']:>10.2f} tokens/s")
    print(f"  Worst: {worst['name']:<12} {worst['tokens_per_sec']:>10.2f} tokens/s")

    print("=" * 100)


def export_csv(results: list[dict], seed: str, output_file: str):
    """Export results to CSV file."""
    if not results:
        return

    rows = []
    for r in sorted(results, key=lambda x: x["input_len"]):
        rows.append(
            {
                "seed": seed,
                "scenario": r["name"],
                "input_len": r["input_len"],
                "output_len": r["output_len"],
                "num_samples": r["num_prompts"],
                "total_tokens": r["total_tokens"],
                "elapsed_time_s": r["elapsed_time"],
                "tokens_per_sec": r["tokens_per_sec"],
                "requests_per_sec": r["requests_per_sec"],
            }
        )

    # Summary row
    total_samples = sum(r["num_prompts"] or 0 for r in results)
    total_tokens = sum(r["total_tokens"] or 0 for r in results)
    total_time = sum(r["elapsed_time"] or 0 for r in results)
    overall_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    overall_req_per_sec = total_samples / total_time if total_time > 0 else 0

    rows.append(
        {
            "seed": seed,
            "scenario": "OVERALL",
            "input_len": "",
            "output_len": "",
            "num_samples": total_samples,
            "total_tokens": total_tokens,
            "elapsed_time_s": total_time,
            "tokens_per_sec": overall_tokens_per_sec,
            "requests_per_sec": overall_req_per_sec,
        }
    )

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults exported to: {output_file}")


def export_json(results: list[dict], seed: str, output_file: str):
    """Export results to JSON file."""
    if not results:
        return

    total_samples = sum(r["num_prompts"] or 0 for r in results)
    total_tokens = sum(r["total_tokens"] or 0 for r in results)
    total_time = sum(r["elapsed_time"] or 0 for r in results)
    overall_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    overall_req_per_sec = total_samples / total_time if total_time > 0 else 0

    data = {
        "seed": seed,
        "summary": {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "overall_tokens_per_sec": overall_tokens_per_sec,
            "overall_req_per_sec": overall_req_per_sec,
        },
        "results": sorted(results, key=lambda x: x["input_len"]),
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results exported to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <results_dir> [seed]")
        return 1

    results_dir = sys.argv[1]
    seed = sys.argv[2] if len(sys.argv) > 2 else "42"

    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        return 1

    print(f"Collecting results from: {results_dir}/")

    results = collect_results(results_dir)

    if not results:
        print("No results found")
        return 1

    print_detailed_results(results, seed)
    print_summary(results, seed)

    export_csv(results, seed, "mixed_bench_summary.csv")
    export_json(results, seed, "mixed_bench_summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
