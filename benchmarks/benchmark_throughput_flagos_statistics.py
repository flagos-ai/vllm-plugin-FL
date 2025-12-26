#!/usr/bin/env python3
import re
import os
import statistics

SCENARIOS = [
    "p128d128",
    "p6144d128",
    "p30720d128",
    "p128d6144",
    "p6144d6144",
    "p30720d6144",
    "p4096d2048"
]

LOG_DIR = "./vllm_bench_logs"

def extract_throughputs(log_path):
    output_throughput = None
    total_throughput = None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None, None

    out_match = re.search(r'Output token throughput \(tok/s\):\s*([\d.]+)', content)
    tot_match = re.search(r'Total Token throughput \(tok/s\):\s*([\d.]+)', content)

    if out_match:
        output_throughput = float(out_match.group(1))
    if tot_match:
        total_throughput = float(tot_match.group(1))

    return output_throughput, total_throughput

def compute_extended_stats(values):
    valid_vals = [v for v in values if v is not None]
    if not valid_vals:
        return ("N/A", "N/A", "N/A", "N/A", "N/A")

    mean_val = statistics.mean(valid_vals)
    median_val = statistics.median(valid_vals)
    max_val = max(valid_vals)

    stdev_val = 0.0 if len(valid_vals) == 1 else statistics.stdev(valid_vals)

    max_abs_dev = max(abs(x - mean_val) for x in valid_vals)
    if stdev_val > 0:
        sigma_str = f"{max_abs_dev / stdev_val:.2f}σ"
    else:
        sigma_str = "0.00σ" if max_abs_dev == 0 else "N/A"

    return (
        f"{mean_val:.2f}",
        f"{median_val:.2f}",
        f"{max_val:.2f}",
        f"{stdev_val:.2f}",
        sigma_str
    )

def main():
    # aligned output
    scene_width = 14
    num_width = 10
    sigma_width = 8

    # headers
    header = (
        f"{'Scenario':<{scene_width}} | "
        f"{'Out Mean':>{num_width}} | {'Out Med':>{num_width}} | {'Out Max':>{num_width}} | {'Out σ':>{num_width}} | {'Out Dev':>{sigma_width}} | "
        f"{'Tot Mean':>{num_width}} | {'Tot Med':>{num_width}} | {'Tot Max':>{num_width}} | {'Tot σ':>{num_width}} | {'Tot Dev':>{sigma_width}}"
    )
    print(header)
    print("-" * len(header))

    # only run2, run3, run4, ignore run1
    RUN_IDS = [2, 3, 4]

    for scene in SCENARIOS:
        output_vals = []
        total_vals = []

        for run_id in RUN_IDS:
            log_file = os.path.join(LOG_DIR, f"{scene}_run{run_id}.log")
            out_tok_s, total_tok_s = extract_throughputs(log_file)
            output_vals.append(out_tok_s)
            total_vals.append(total_tok_s)

        out_stats = compute_extended_stats(output_vals)
        tot_stats = compute_extended_stats(total_vals)

        line = (
            f"{scene:<{scene_width}} | "
            f"{out_stats[0]:>{num_width}} | {out_stats[1]:>{num_width}} | {out_stats[2]:>{num_width}} | {out_stats[3]:>{num_width}} | {out_stats[4]:>{sigma_width}} | "
            f"{tot_stats[0]:>{num_width}} | {tot_stats[1]:>{num_width}} | {tot_stats[2]:>{num_width}} | {tot_stats[3]:>{num_width}} | {tot_stats[4]:>{sigma_width}}"
        )
        print(line)

if __name__ == "__main__":
    main()
