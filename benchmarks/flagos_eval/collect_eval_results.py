#!/usr/bin/env python3
"""Collect lm_eval evaluation results and output summary"""

import csv
import glob
import json
import os

# Define preferred metrics for each dataset
DATASET_METRICS = {
    # Current evaluation tasks
    "mmlu_pro": "exact_match,custom-extract",
    "bbh": "exact_match,get-answer",
    "gsm8k": "exact_match,strict-match",
    "hendrycks_math": "exact_match,none",
    "humaneval": "pass@1,create_test",
    "mbpp": "pass_at_1,none",
    "mmlu_redux": "exact_match,default",
    "gpqa": "exact_match,flexible-extract",
    "mgsm": "exact_match,flexible-extract",
    # Other common tasks
    "hellaswag": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
    "commonsense_qa": "acc,none",
    "piqa": "acc,none",
    "openbookqa": "acc,none",
    "boolq": "acc,none",
    "arc_easy": "acc,none",
    "arc_challenge": "acc,none",
    "mmlu": "acc,none",
    "ceval": "acc,none",
    "cmmlu": "acc,none",
    "minerva_math": "exact_match,none",
}


def get_preferred_metric(task_name, metrics_dict):
    """Get preferred metric value for a task"""
    task_lower = task_name.lower()

    # Try exact match first
    for dataset, metric in DATASET_METRICS.items():
        if dataset in task_lower and metric in metrics_dict:
            return metric, metrics_dict[metric]

    # Fallback to generic matching
    for key in ["exact_match", "acc", "pass@1", "pass_at_1"]:
        for metric_name, value in metrics_dict.items():
            if key in metric_name and "stderr" not in metric_name:
                return metric_name, value

    # Return first non-stderr metric
    for metric_name, value in metrics_dict.items():
        if "stderr" not in metric_name and isinstance(value, (int, float)):
            return metric_name, value

    return None, None


def collect_eval_results(output_dir="output"):
    """Collect all evaluation results"""
    results = {}

    # Find all results*.json files
    pattern = os.path.join(output_dir, "**", "results*.json")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No result files found, please check {output_dir} directory")
        return None

    for path in files:
        with open(path, "r") as f:
            data = json.load(f)

        # Extract task results
        for task, metrics in data.get("results", {}).items():
            if task not in results:
                metric_name, value = get_preferred_metric(task, metrics)
                if metric_name and value is not None:
                    results[task] = {"metric": metric_name, "value": value}

    return results


def print_results(results):
    """Format and print results"""
    if not results:
        return

    print("\n" + "=" * 75)
    print("Evaluation Results Summary")
    print("=" * 75)

    # Print table
    print(f"\n{'Task':<45} {'Metric':<20} {'Score':<10}")
    print("-" * 75)

    for task in sorted(results.keys()):
        data = results[task]
        metric = data["metric"].split(",")[0]  # Simplify metric name
        value = data["value"]
        print(f"{task:<45} {metric:<20} {value:.4f}")

    print("-" * 75)

    # Print main task summary
    print("\nMain Task Summary:")
    print("-" * 45)
    print(f"{'Task':<30} {'Score':<10}")
    print("-" * 45)

    main_tasks = [
        "mmlu_pro",
        "bbh",
        "gsm8k",
        "hendrycks_math",
        "humaneval",
        "mbpp",
        "mmlu_redux",
        "gpqa",
        "mgsm",
    ]

    for task in main_tasks:
        found = False
        # Try to find exact match for main task first
        for result_task, data in results.items():
            if result_task.lower() == task:
                print(f"{result_task:<30} {data['value']:.4f}")
                found = True
                break

        if not found:
            # For gpqa and mgsm, calculate average of subtasks
            if task == "gpqa":
                gpqa_scores = [
                    data["value"]
                    for t, data in results.items()
                    if t.startswith("gpqa_") and "cot_zeroshot" in t
                ]
                if gpqa_scores:
                    avg = sum(gpqa_scores) / len(gpqa_scores)
                    print(f"{'gpqa (avg)':<30} {avg:.4f}")
                    found = True
            elif task == "mgsm":
                # Average of mgsm_direct subtasks
                mgsm_direct = [
                    data["value"]
                    for t, data in results.items()
                    if t.startswith("mgsm_direct_") and "spanish_bench" not in t
                ]
                if mgsm_direct:
                    avg = sum(mgsm_direct) / len(mgsm_direct)
                    print(f"{'mgsm_direct (avg)':<30} {avg:.4f}")
                    found = True
            else:
                # Try to match results containing task name
                for result_task, data in results.items():
                    if (
                        task in result_task.lower()
                        and result_task.count("_") <= task.count("_") + 1
                        and "cot_fewshot" not in result_task
                        and "direct" not in result_task
                    ):
                        print(f"{result_task:<30} {data['value']:.4f}")
                        found = True
                        break

    print("-" * 45)


def save_csv(results, output_file="results_summary.csv"):
    """Save as CSV format"""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "metric", "value"])
        for task in sorted(results.keys()):
            data = results[task]
            writer.writerow([task, data["metric"], data["value"]])
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"

    results = collect_eval_results(output_dir)
    if results:
        print_results(results)
        save_csv(results)
