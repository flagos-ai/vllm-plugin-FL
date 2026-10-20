#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage:
#  1. Start the server as follows (adjust model path and args as needed):
# vllm serve /models/Qwen3.6-35B-A3B --tensor-parallel-size 2 --max-model-len 262144 --no-enable-log-requests --no-enable-prefix-caching

#  2. Run this benchmark script (default: 4 test cases):
# python benchmarks/benchmark_throughput_serve.py --model /models/Qwen3.6-35B-A3B
#
# [Optional] Run all 10 test cases:
# python benchmarks/benchmark_throughput_serve.py \
#   --model /models/Qwen3.6-35B-A3B --served-model-name qwen \
#   --port 8000 --enable-all
#
# [Optional] Run custom test cases or port or served model name:
# Each test case: [input_len, output_len, concurrency, num_prompts]
# python benchmarks/benchmark_throughput_serve.py \
#   --model /models/Qwen3.6-35B-A3B --served-model-name qwen --port 8000 \
#   --test-cases '[[1024,1024,64,256],[4096,1024,64,256]]'


import argparse
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime
from statistics import mean

# total runs for each case
RUNS = 4

# skip first N runs
SKIP_FIRST = 1

# Baseline cases used when --enable-all is not set.
# Each case is a tuple:
# (random_input_len, random_output_len, max_concurrency, num_prompts)
DEFAULT_TEST_CASES = [
    (1024, 1024, 64, 256),
    (4096, 1024, 64, 256),
    (16384, 1024, 64, 256),
    (65536, 1024, 64, 256),
]

ALL_TEST_CASES = [
    *DEFAULT_TEST_CASES,
    (4096, 1024, 1, 256),
    (4096, 1024, 4, 256),
    (4096, 1024, 16, 256),
    (4096, 1024, 256, 256),
    (131072, 1024, 64, 64),
    (262144, 1024, 64, 64),
]


def parse_test_cases(value):
    try:
        raw_cases = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(raw_cases, list) or not raw_cases:
        raise argparse.ArgumentTypeError("test cases must be a non-empty JSON list")

    for index, raw_case in enumerate(raw_cases):
        case_label = f"test case at index {index}"

        if not isinstance(raw_case, list) or len(raw_case) != 4:
            raise argparse.ArgumentTypeError(
                f"{case_label} must be a list of exactly 4 values: "
                "input_len, output_len, concurrency, num_prompts"
            )

        if any(type(item) is not int or item <= 0 for item in raw_case):
            raise argparse.ArgumentTypeError(
                f"{case_label} values must be positive integers"
            )

    return [tuple(case) for case in raw_cases]


def existing_path(value):
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"model path does not exist: {path}")
    return path


def valid_port(value):
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port: {value}") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=existing_path,
        required=True,
        help="Local model path used to load the tokenizer.",
    )
    parser.add_argument(
        "--port",
        type=valid_port,
        default=8000,
        help="Server port (default: 8000).",
    )
    parser.add_argument(
        "--served-model-name",
        help="Model name exposed by the server. If omitted, use the model path.",
    )

    test_case_group = parser.add_mutually_exclusive_group()

    test_case_group.add_argument(
        "--enable-all",
        action="store_true",
        help="Enable all 10 test cases. If not set, run default 4 cases.",
    )
    test_case_group.add_argument(
        "--test-cases",
        type=parse_test_cases,
        metavar="JSON",
        help=(
            "Run custom test cases from a JSON list. Each case is "
            "[input_len, output_len, concurrency, num_prompts]."
        ),
    )

    return parser.parse_args(argv)


def build_common_args(model, port, served_model_name=None):
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        served_model_name or model,
        "--tokenizer",
        model,
        "--endpoint",
        "/v1/completions",
        "--host",
        "localhost",
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--ignore-eos",
    ]


PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "benchmark_duration": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "total_input_tokens": r"Total input tokens:\s+([0-9.]+)",
    "total_output_tokens": r"Total generated tokens:\s+([0-9.]+)",
    "request_throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "peak_output_throughput": r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
}

RAW_CSV_COLUMNS = [
    "Prefill",
    "Decode",
    "Conc",
    "Num Prompts",
    "Successful Requests",
    "Run Status",
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Output tok/s",
    "Peak Output tok/s",
    "Total tok/s",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
]

SUMMARY_CSV_COLUMNS = [
    "Prefill",
    "Decode",
    "Conc",
    "Num Prompts",
    "Benchmark Duration (s)",
    "Total Input Tokens",
    "Total Output Tokens",
    "Req/s",
    "Output tok/s",
    "Peak Output tok/s",
    "Total tok/s",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
]


def extract_metrics(output_text):
    result = {}

    for key, pattern in PATTERNS.items():
        match = re.search(
            pattern,
            output_text,
            re.IGNORECASE,
        )

        result[key] = float(match.group(1)) if match else None

    return result


def format_result(case, metrics, include_successful_requests=True):
    input_len, output_len, concurrency, num_prompts = case

    result = {
        "Prefill": input_len,
        "Decode": output_len,
        "Conc": concurrency,
        "Num Prompts": num_prompts,
        "Benchmark Duration (s)": metrics.get("benchmark_duration"),
        "Total Input Tokens": metrics.get("total_input_tokens"),
        "Total Output Tokens": metrics.get("total_output_tokens"),
        "Req/s": metrics.get("request_throughput"),
        "Output tok/s": metrics.get("output_throughput"),
        "Peak Output tok/s": metrics.get("peak_output_throughput"),
        "Total tok/s": metrics.get("total_token_throughput"),
        "Mean TTFT (ms)": metrics.get("mean_ttft_ms"),
        "Median TTFT (ms)": metrics.get("median_ttft_ms"),
        "P99 TTFT (ms)": metrics.get("p99_ttft_ms"),
        "Mean TPOT (ms)": metrics.get("mean_tpot_ms"),
        "Median TPOT (ms)": metrics.get("median_tpot_ms"),
        "P99 TPOT (ms)": metrics.get("p99_tpot_ms"),
        "Mean ITL (ms)": metrics.get("mean_itl_ms"),
        "Median ITL (ms)": metrics.get("median_itl_ms"),
        "P99 ITL (ms)": metrics.get("p99_itl_ms"),
    }

    if include_successful_requests:
        result["Successful Requests"] = metrics.get("successful_requests")

    return result


def append_csv(row, filename, columns):
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def run_once(case, run_id, common_args):
    input_len, output_len, concurrency, num_prompts = case

    name = f"{input_len}_{output_len}_c{concurrency}"

    print("=" * 80)
    print(f"Running: {name} | Run {run_id}/{RUNS}")
    print("=" * 80)

    cmd = common_args + [
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
    ]

    print(" ".join(cmd))
    print()

    start_time = time.time()

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    elapsed = time.time() - start_time

    output = process.stdout

    print(output)

    metrics = extract_metrics(output)

    metrics["elapsed_sec"] = round(elapsed, 2)

    return metrics


def average_metrics(results):
    avg_result = {}

    keys = results[0].keys()

    for key in keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]

        if values:
            avg_result[key] = round(mean(values), 2)

    return avg_result


def run_test_case(case, raw_csv, common_args):
    all_runs = []

    for run_id in range(1, RUNS + 1):
        metrics = run_once(case, run_id, common_args)

        raw_row = format_result(case, metrics, include_successful_requests=True)
        expected_successful_requests = case[3]
        raw_row["Run Status"] = (
            "SUCCESS"
            if metrics.get("successful_requests") == expected_successful_requests
            else "FAILED"
        )

        append_csv(raw_row, raw_csv, RAW_CSV_COLUMNS)

        all_runs.append(metrics)

    valid_runs = all_runs[SKIP_FIRST:]

    expected_successful_requests = case[3]
    has_failed_run = any(
        run.get("successful_requests") != expected_successful_requests
        for run in valid_runs
    )

    avg_metrics = average_metrics(valid_runs)

    summary_row = format_result(
        case,
        avg_metrics,
        include_successful_requests=False,
    )

    return summary_row, has_failed_run


def print_summary(results):
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    for r in results:
        print(
            f"Prefill={r['Prefill']} "
            f"Decode={r['Decode']} "
            f"Conc={r['Conc']} "
            f"NumPrompts={r['Num Prompts']} "
            f"Req/s={r['Req/s']} "
            f"Total tok/s={r['Total tok/s']} "
            f"TTFT={r['Mean TTFT (ms)']}ms"
        )


def main():
    args = parse_args()
    common_args = build_common_args(args.model, args.port, args.served_model_name)

    if args.test_cases:
        test_cases = args.test_cases
    elif args.enable_all:
        test_cases = ALL_TEST_CASES
    else:
        test_cases = DEFAULT_TEST_CASES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    raw_csv = os.path.join(output_dir, f"raw_runs_{timestamp}.csv")
    summary_csv = os.path.join(output_dir, f"summary_{timestamp}.csv")

    all_summary = []

    print()
    print(f"RUNS={RUNS}")
    print(f"SKIP_FIRST={SKIP_FIRST}")
    print(f"MODEL={args.model}")
    print(f"SERVED_MODEL_NAME={args.served_model_name or args.model}")
    print(f"PORT={args.port}")
    print(f"ENABLE_ALL={args.enable_all}")
    print(f"TOTAL_CASES={len(test_cases)}")
    print(f"TEST_CASES={test_cases}")
    print()

    for case in test_cases:
        try:
            summary_row, has_failed_run = run_test_case(
                case,
                raw_csv,
                common_args,
            )

            if has_failed_run:
                print(f"SKIP SUMMARY ROW (failed case): {case}")
                continue

            append_csv(summary_row, summary_csv, SUMMARY_CSV_COLUMNS)

            all_summary.append(summary_row)

        except Exception as e:
            print(f"ERROR: {e}")

    print_summary(all_summary)

    print()
    print(f"Raw CSV: {raw_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
