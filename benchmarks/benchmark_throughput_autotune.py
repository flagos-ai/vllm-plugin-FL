# Copyright (c) 2026 BAAI. All rights reserved.

"""
Auto-tune FlagGems operators based on throughput.

Workflow:
1. Round 1: Run baseline throughput without FlagGems operators.
2. Round 2: For each provided FlagGems operator, enable only that operator
   (whitelist) and measure throughput. Save all results into history.csv,
   sorted by throughput (high to low).
3. Round 3: Enable only the operators that improved throughput over baseline
   and measure final throughput.

This script assumes:
- vllm CLI is available (`vllm bench throughput`).
- Worker uses USE_FLAGGEMS and VLLM_FL_FLAGOS_WHITELIST to control FlagGems.
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import vllm_fl.envs as fl_envs
from vllm_fl.utils import OOT_OP_NAMES


# ====== default configs (aligned with benchmark_throughput_flagos.py) ======
# Per requirement: run twice per configuration and take the 2nd run (skip warmup)
NUM_RUNS = 2

LOG_DIR = "autotune_logs"
CSV_PATH = "history.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _str_to_bool(value: str) -> bool:
    return value.lower() in ("1", "true")


def _strip_background_args(argv: List[str]) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--background="):
            continue
        if arg == "--background":
            if i + 1 < len(argv) and argv[i + 1] != "--":
                skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def extract_gems_ops_from_enable_log(log_path: str) -> List[str]:
    """
    Extract FlagGems op names from enable log. One pattern: last dotted segment
    before ': GEMS'. E.g. flag_gems.ops.fill.fill_tensor_: GEMS xxx ->
    fill_tensor_; flag_gems.fused...fused_recurrent_gated_delta_rule_fwd: GEMS xxx ->
    fused_recurrent_gated_delta_rule_fwd.
    """
    pattern = re.compile(r"flag_gems\.(?:ops|fused)\..*\.([^.\s:]+):\s*GEMS")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    except OSError:
        return []

    ops: set[str] = set()
    for line in lines:
        match = pattern.search(line)
        if match:
            ops.add(match.group(1).lower())
    return sorted(ops)


def get_registered_ops_by_backend() -> Dict[str, List[str]]:
    """
    Get registered operator lists for FlagOS (DEFAULT), vendor, and reference.

    Returns:
        {
            "flagos": [...],
            "vendor": [...],
            "reference": [...],
        }
    """
    from vllm_fl.dispatch import get_default_manager, BackendImplKind

    manager = get_default_manager()
    manager.ensure_initialized()
    snapshot = manager.registry.snapshot()

    flagos_ops: set[str] = set()
    vendor_ops: set[str] = set()
    reference_ops: set[str] = set()

    for op_name, impls in snapshot.impls_by_op.items():
        for impl in impls:
            if impl.kind == BackendImplKind.DEFAULT:
                flagos_ops.add(op_name)
            elif impl.kind == BackendImplKind.VENDOR:
                vendor_ops.add(op_name)
            elif impl.kind == BackendImplKind.REFERENCE:
                reference_ops.add(op_name)

    return {
        "flagos": sorted(flagos_ops),
        "vendor": sorted(vendor_ops),
        "reference": sorted(reference_ops),
    }


def get_all_tunable_ops() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Get all tunable operators and their categories.

    If --ops is not provided, collect operators from:
    1. FlagGems operators: get_flaggems_all_ops() + flagos backend registered ops
    2. Vendor operators: vendor backend registered ops
    3. Reference operators: reference backend registered ops

    Returns:
        Tuple of (list of all operator names, dict mapping op_name -> backends)
        Backends: ["flagos"], ["vendor"], ["reference"] or combinations
    """
    from vllm_fl.utils import get_flaggems_all_ops

    all_ops: List[str] = []
    op_categories: Dict[str, List[str]] = {}

    # Get FlagGems all_ops
    try:
        gems_all_ops = get_flaggems_all_ops()
    except Exception:
        gems_all_ops = []

    # Get registered ops by backend
    try:
        registered = get_registered_ops_by_backend()
    except Exception:
        registered = {"flagos": [], "vendor": [], "reference": []}

    # Combine gems ops (from flag_gems.all_ops + flagos backend)
    flagos_registered = set(registered.get("flagos", []))
    vendor_registered = set(registered.get("vendor", []))
    reference_registered = set(registered.get("reference", []))
    all_registered_ops = flagos_registered | vendor_registered | reference_registered
    gems_ops_set = set(gems_all_ops) | flagos_registered
    for op in sorted(gems_ops_set):
        if op not in op_categories:
            all_ops.append(op)
            op_categories[op] = ["flagos"]
        elif "flagos" not in op_categories[op]:
            op_categories[op].append("flagos")

    # Add OOT operators only if registered in backends (categorize as flagos)
    for op in OOT_OP_NAMES:
        if op not in all_registered_ops:
            continue
        if op not in op_categories:
            all_ops.append(op)
            op_categories[op] = ["flagos"]
        elif "flagos" not in op_categories[op]:
            op_categories[op].append("flagos")

    # Add vendor operators
    for op in vendor_registered:
        if op not in op_categories:
            all_ops.append(op)
            op_categories[op] = ["vendor"]
        elif "vendor" not in op_categories[op]:
            op_categories[op].append("vendor")

    # Add reference operators
    for op in reference_registered:
        if op not in op_categories:
            all_ops.append(op)
            op_categories[op] = ["reference"]
        elif "reference" not in op_categories[op]:
            op_categories[op].append("reference")

    return all_ops, op_categories


def is_oot_op(op_name: str) -> bool:
    """Check if an operator is an OOT (out-of-tree) operator."""
    return op_name in OOT_OP_NAMES


def load_auto_tune_config() -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    config_path = os.environ.get("VLLM_FL_CONFIG", "").strip()
    if not config_path:
        return False, None, None
    if not os.path.isfile(config_path):
        logger.info("Auto-tune config not found at %s", config_path)
        return False, None, config_path
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.info("Failed to load auto-tune config: %s", exc)
        return False, None, config_path

    action = str(config.get("action", "")).strip().lower()
    return action == "auto_tune", config, config_path


def write_auto_tune_ops_file(
    file_path: str,
    ops: List[str],
    op_categories: Dict[str, List[str]],
) -> None:
    default_order = ["flagos", "vendor", "reference"]
    op_backends: Dict[str, List[str]] = {}
    for op in ops:
        backends = op_categories.get(op)
        if backends:
            op_backends[op] = sorted(set(backends), key=default_order.index)
        else:
            op_backends[op] = default_order.copy()

    payload: Dict[str, Any] = {
        "action": "auto_tune",
        "op_backends": op_backends,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    logger.info("Auto-tune op list file written to: %s", file_path)


def write_round_config(
    file_path: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    payload = payload or {}
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def write_best_config(file_path: str, op_backends: Dict[str, List[str]]) -> None:
    payload = {"op_backends": op_backends}
    write_round_config(file_path, payload)
    logger.info("Best config written to: %s", file_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-tune FlagGems operators based on throughput."
    )
    parser.add_argument(
        "--background",
        type=str,
        default="False",
        help="Run in background (True/False). Default: False",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default="",
        help=(
            "Comma-separated list of operator names to tune. "
            "If not provided, auto-discover all tunable operators from: "
            "gems (flag_gems.all_ops + flagos backend), OOT ops, vendor backend, reference backend."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of runs per configuration (default: {NUM_RUNS})",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=CSV_PATH,
        help=f"Path to output CSV file (default: {CSV_PATH})",
    )

    args, passthrough = parser.parse_known_args()
    # Passthrough args go directly to `vllm bench throughput`
    args.passthrough_args = passthrough
    args.background = _str_to_bool(args.background)
    return args


def ensure_log_dir(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)


def build_bench_cmd(throughput_args: List[str]) -> List[str]:
    cmd = [
        "vllm",
        "bench",
        "throughput",
    ]
    cmd.extend(throughput_args)
    return cmd


def _run_and_stream_to_file(cmd: List[str], log_file: str, env: dict) -> int:
    """
    Run a command and stream stdout/stderr to both console and log_file in real
    time (append).
    Returns the process exit code.
    """
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", buffering=1, encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            # Tee to console + file
            logger.info(line.rstrip("\n"))
            f.write(line)
            f.flush()
        return proc.wait()


def extract_throughput(log_text: str) -> Optional[Tuple[float, float]]:
    """
    Extract total and output token throughput (tok/s) from vllm bench log text.
    Returns (total_throughput, output_throughput) or None if not found.
    """
    # vllm bench throughput prints variants like:
    # - "Total Token throughput (tok/s): 12345.67"
    # - "Throughput: 3.66 requests/s, 26237.36 total tokens/s, 3748.19 output tokens/s"

    # Try new format first: "Throughput: ..., <total> total tokens/s, <output> output tokens/s"
    new_format_match = re.search(
        r"Throughput:[^,]*,\s*([\d.]+)\s*total tokens/s[^,]*,\s*([\d.]+)\s*output tokens/s",
        log_text,
    )
    if new_format_match:
        try:
            total = float(new_format_match.group(1))
            output = float(new_format_match.group(2))
            return (total, output)
        except ValueError:
            pass

    # Fallback to old format: "Total Token throughput (tok/s): <num>"
    old_format_match = re.search(
        r"Total Token throughput \(tok/s\):\s*([\d.]+)", log_text
    )
    if old_format_match:
        try:
            total = float(old_format_match.group(1))
            # If only total is found, set output to None (will be empty in CSV)
            return (
                total,
                0.0,
            )  # Use 0.0 as placeholder, will be handled in CSV writing
        except ValueError:
            pass

    return None


def run_benchmark_once(
    label: str,
    throughput_args: List[str],
    log_file: str,
    use_flaggems: bool = True,
    gems_whitelist: Optional[List[str]] = None,
    oot_whitelist: Optional[List[str]] = None,
    oot_enabled: bool = True,
    config_path: Optional[str] = None,
) -> Optional[Tuple[float, float]]:
    """
    Run a single benchmark invocation and return (total_throughput, output_throughput).

    Args:
        label: Label for logging
        throughput_args: Arguments passed to vllm bench throughput
        log_file: Path to log file
        use_flaggems: Whether to enable FlagGems (USE_FLAGGEMS=1)
        gems_whitelist: FlagGems operator whitelist (VLLM_FL_FLAGOS_WHITELIST)
        oot_whitelist: OOT operator whitelist (VLLM_FL_OOT_WHITELIST)
        oot_enabled: Whether to enable OOT registration (VLLM_FL_OOT_ENABLED)
        config_path: Optional YAML config path (VLLM_FL_CONFIG)
    """
    ensure_log_dir(log_file)
    cmd = build_bench_cmd(throughput_args)

    env = os.environ.copy()
    if config_path:
        env["VLLM_FL_CONFIG"] = config_path
    else:
        env.pop("VLLM_FL_CONFIG", None)

    # OOT control
    env["VLLM_FL_OOT_ENABLED"] = "1" if oot_enabled else "0"
    if oot_whitelist:
        env["VLLM_FL_OOT_WHITELIST"] = ",".join(oot_whitelist)
    else:
        env.pop("VLLM_FL_OOT_WHITELIST", None)

    # FlagGems control
    env["USE_FLAGGEMS"] = "1" if use_flaggems else "0"
    if use_flaggems and gems_whitelist:
        env["VLLM_FL_FLAGOS_WHITELIST"] = ",".join(gems_whitelist)
        env.pop("VLLM_FL_FLAGOS_BLACKLIST", None)
    else:
        env.pop("VLLM_FL_FLAGOS_WHITELIST", None)
        env.pop("VLLM_FL_FLAGOS_BLACKLIST", None)

    logger.info(
        "[%s] Running benchmark: %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        label,
    )
    logger.info(
        "  USE_FLAGGEMS=%s, gems_whitelist=%s",
        "1" if use_flaggems else "0",
        gems_whitelist or "None",
    )
    logger.info(
        "  OOT_ENABLED=%s, oot_whitelist=%s",
        "1" if oot_enabled else "0",
        oot_whitelist or "None",
    )
    logger.info("  VLLM_FL_CONFIG=%s", config_path or "None")
    logger.info("  Log: %s", log_file)

    exit_code = _run_and_stream_to_file(cmd, log_file, env)
    if exit_code != 0:
        logger.info(
            "  Benchmark failed for %s with exit code %s. See log for details.",
            label,
            exit_code,
        )
        return None

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""

    result = extract_throughput(content)
    if result is None:
        logger.info("  Failed to parse throughput for %s.", label)
    else:
        total, output = result
        if output > 0:
            logger.info(
                "  Total Token throughput (tok/s): %.2f, Output: %.2f",
                total,
                output,
            )
        else:
            logger.info("  Total Token throughput (tok/s): %.2f", total)

    return result


def run_benchmark_multi(
    label: str,
    throughput_args: List[str],
    num_runs: int,
    run_dir: str,
    log_counter: List[int],
    use_flaggems: bool = True,
    gems_whitelist: Optional[List[str]] = None,
    oot_whitelist: Optional[List[str]] = None,
    oot_enabled: bool = True,
    config_dir: Optional[str] = None,
    config_payload: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    """
    Run multiple benchmark runs and return the metric value from the second
    successful run (to skip warmup). If fewer than two successful runs exist,
    return the last successful value.
    Returns (total_throughput, output_throughput) or None.
    When config_dir is provided, writes per-run config files with the same
    prefix as the log filename.
    """
    if label == "baseline":
        base_name = "baseline"
    elif label == "baseline_fake_gems_op":
        base_name = "baseline_fake_gems_op"
    elif label == "baseline_gems_enable":
        base_name = "baseline_gems_enable"
    elif label == "combined_best_ops":
        base_name = "combined_best"
    elif label.startswith("op_"):
        base_name = f"op_{label[len('op_') :]}"
    else:
        base_name = f"op_{label}"

    # Baseline runs should not register any OOT ops.
    if label in {"baseline", "baseline_fake_gems_op", "baseline_gems_enable"}:
        oot_enabled = False
        oot_whitelist = None

    name_prefix = base_name
    if label.startswith("op_") and config_payload:
        op_name = label[len("op_") :]
        op_backends = config_payload.get("op_backends", {}).get(op_name, [])
        if op_backends and all(
            backend in {"flagos", "reference", "vendor"} for backend in op_backends
        ):
            name_prefix = f"{base_name}_{'_'.join(op_backends)}"

    results: List[Optional[Tuple[float, float]]] = []
    for i in range(1, num_runs + 1):
        counter = log_counter[0]
        log_counter[0] += 1
        counter_str = f"{counter:04d}"
        if i < num_runs:
            file_stem = f"{counter_str}_{name_prefix}_warmup_{i}"
            log_file = os.path.join(run_dir, f"{file_stem}.log")
        else:
            file_stem = f"{counter_str}_{name_prefix}_benchmark"
            log_file = os.path.join(run_dir, f"{file_stem}.log")
        config_path = None
        if config_dir:
            config_path = os.path.join(config_dir, f"{file_stem}.yaml")
            write_round_config(config_path, config_payload)

        val = run_benchmark_once(
            label,
            throughput_args,
            log_file=log_file,
            use_flaggems=use_flaggems,
            gems_whitelist=gems_whitelist,
            oot_whitelist=oot_whitelist,
            oot_enabled=oot_enabled,
            config_path=config_path,
        )
        results.append(val)

    if not any(results):
        logger.info("No successful runs for %s.", label)
        return None

    # Prefer the benchmark run (last run); fall back to last successful
    chosen = results[-1]
    chosen_idx = num_runs
    if chosen is None:
        for idx in range(len(results) - 1, -1, -1):
            if results[idx] is not None:
                chosen = results[idx]
                chosen_idx = idx + 1
                break

    if chosen is None:
        logger.info("No successful runs for %s.", label)
        return None

    total, output = chosen
    if output > 0:
        logger.info(
            "Selected run #%s for %s: total=%.2f tok/s, output=%.2f tok/s, "
            "successful runs: %s/%s",
            chosen_idx,
            label,
            total,
            output,
            sum(1 for r in results if r is not None),
            num_runs,
        )
    else:
        logger.info(
            "Selected run #%s for %s: total=%.2f tok/s, successful runs: %s/%s",
            chosen_idx,
            label,
            total,
            sum(1 for r in results if r is not None),
            num_runs,
        )
    return chosen


def write_results_csv(
    csv_path: str,
    baseline_result: Optional[Tuple[float, float]],
    per_op_results: Dict[str, Optional[Tuple[float, float]]],
    combined_result: Optional[Tuple[float, float]] = None,
    baseline_fake_result: Optional[Tuple[float, float]] = None,
    baseline_enable_result: Optional[Tuple[float, float]] = None,
    op_backends: Optional[Dict[str, List[str]]] = None,
    per_op_backend_results: Optional[
        Dict[str, Dict[str, Optional[Tuple[float, float]]]]
    ] = None,
) -> None:
    """
    Write per-operator results into a CSV file, sorted by total throughput desc.
    Includes baseline row at the top.
    """
    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float]]] = []

    # Add baseline rows first (baseline + baseline_gems_fake_op + baseline_gems_enable)
    if baseline_result is not None:
        baseline_total, baseline_output = baseline_result
        rows.append(("baseline", "", baseline_total, baseline_output, 1.0))
    else:
        rows.append(("baseline", "", None, None, None))

    if baseline_fake_result is not None:
        fake_total, fake_output = baseline_fake_result
        if baseline_result is not None and baseline_total > 0:
            fake_ratio = fake_total / baseline_total
        else:
            fake_ratio = None
        rows.append(("baseline_gems_fake_op", "", fake_total, fake_output, fake_ratio))

    if baseline_enable_result is not None:
        enable_total, enable_output = baseline_enable_result
        if baseline_result is not None and baseline_total > 0:
            enable_ratio = enable_total / baseline_total
        else:
            enable_ratio = None
        rows.append(
            ("baseline_gems_enable", "", enable_total, enable_output, enable_ratio)
        )

    # Add per-operator rows
    baseline_total = baseline_result[0] if baseline_result is not None else None
    ops = set(per_op_results.keys())
    if per_op_backend_results:
        ops |= set(per_op_backend_results.keys())
    for op in sorted(ops):
        backend_results = (
            per_op_backend_results.get(op, {}) if per_op_backend_results else {}
        )
        if op_backends:
            backend_order = [
                backend
                for backend in op_backends.get(op, [])
                if backend in backend_results
            ]
        else:
            backend_order = list(backend_results.keys())
        if backend_order:
            for backend in backend_order:
                result = backend_results.get(backend)
                if result is None or baseline_total is None:
                    speedup = None
                    total_threshold = None
                    output_threshold = None
                else:
                    total_threshold, output_threshold = result
                    speedup = (
                        total_threshold / baseline_total if baseline_total > 0 else None
                    )
                rows.append((op, backend, total_threshold, output_threshold, speedup))
        else:
            result = per_op_results.get(op)
            if result is None:
                continue
            if baseline_total is None:
                speedup = None
                total_threshold = None
                output_threshold = None
            else:
                total_threshold, output_threshold = result
                speedup = (
                    total_threshold / baseline_total if baseline_total > 0 else None
                )
            rows.append((op, "", total_threshold, output_threshold, speedup))

    # Add combined row (if available)
    if combined_result is not None:
        if baseline_total is None:
            speedup = None
        else:
            speedup = (
                combined_result[0] / baseline_total if baseline_total > 0 else None
            )
        rows.append(
            ("combined_best", "", combined_result[0], combined_result[1], speedup)
        )

    # Sort by total throughput (desc), None last, but keep baseline rows first
    baseline_rows = [row for row in rows if row[0].startswith("baseline")]
    op_rows = [row for row in rows if not row[0].startswith("baseline")]
    op_rows.sort(key=lambda x: (x[2] is not None, x[2] or 0.0), reverse=True)
    rows = baseline_rows + op_rows

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "op",
                "backend",
                "total_throughput(tokens/s)",
                "output_throughput(tokens/s)",
                "performance_ratio",
            ]
        )
        for op, backend, total, output, rel in rows:
            writer.writerow(
                [
                    op,
                    backend,
                    f"{total:.2f}" if total is not None else "",
                    f"{output:.2f}" if output is not None and output > 0 else "",
                    f"{rel:.4f}" if rel is not None else "",
                ]
            )

    logger.info("Per-operator results written to: %s", csv_path)


def write_op_config_json(
    json_path: str,
    baseline_result: Optional[Tuple[float, float]],
    per_op_results: Dict[str, Optional[Tuple[float, float]]],
) -> None:
    baseline_total = baseline_result[0] if baseline_result is not None else None
    ops_data: List[Dict[str, Optional[object]]] = []

    for op_name, result in per_op_results.items():
        # if "fake" in op_name:
        #     continue
        if baseline_total is None or result is None or baseline_total <= 0:
            performance_ratio = None
            backend = "native"
        else:
            total_threshold, _output_threshold = result
            performance_ratio = total_threshold / baseline_total
            backend = "flagos" if performance_ratio > 1 else "native"

        ops_data.append(
            {
                "op_name": op_name,
                "backend": backend,
                "vendor": None,
                "performance_ratio": performance_ratio,
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ops_data, f, ensure_ascii=True, indent=2)

    logger.info("Op config written to: %s", json_path)


def main() -> None:
    args = parse_args()
    throughput_args = args.passthrough_args
    if not throughput_args:
        raise ValueError(
            "No throughput arguments provided. "
            "Pass them after the script args, e.g.: "
            'python benchmark_gems_autotune.py --ops "silu_and_mul" -- '
            "--model /models/Qwen3-Next-80B-A3B-Instruct --tensor-parallel-size 4 ..."
        )

    if args.background and not os.environ.get("FLAGGEMS_AUTOTUNE_BACKGROUND"):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(LOG_DIR, f"autotune_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "autotune.log")
        env = os.environ.copy()
        env["FLAGGEMS_AUTOTUNE_BACKGROUND"] = "1"
        env["FLAGGEMS_AUTOTUNE_RUN_DIR"] = run_dir
        child_args = _strip_background_args(sys.argv[1:])
        cmd = [sys.executable, os.path.abspath(__file__)] + child_args
        with open(log_path, "a", buffering=1, encoding="utf-8") as f:
            subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
        logger.info("Background run started. Log: %s", log_path)
        return

    run_dir = os.environ.get("FLAGGEMS_AUTOTUNE_RUN_DIR")
    if run_dir is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(LOG_DIR, f"autotune_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, os.path.basename(args.csv_path))

    auto_tune_action, auto_tune_config, auto_tune_path = load_auto_tune_config()

    # Determine operators to tune
    if args.ops:
        ops = [op.strip() for op in args.ops.split(",") if op.strip()]
        # Build op_categories for provided ops using discovered backends
        _, all_categories = get_all_tunable_ops()
        op_categories: Dict[str, List[str]] = {}
        for op in ops:
            backends = all_categories.get(op, []).copy()
            if not backends:
                backends = ["flagos"]
            elif is_oot_op(op) and "flagos" not in backends:
                backends.append("flagos")
            op_categories[op] = backends
    elif auto_tune_action:
        logger.info("Auto-tune action detected in config: %s", auto_tune_path)
        per_op_raw = auto_tune_config.get("op_backends", {}) if auto_tune_config else {}
        if isinstance(per_op_raw, dict) and per_op_raw:
            ops = list(per_op_raw.keys())
            _, all_categories = get_all_tunable_ops()
            op_categories = {op: all_categories.get(op, []).copy() for op in ops}
        else:
            # Auto-discover all tunable operators
            ops, op_categories = get_all_tunable_ops()
    else:
        # Auto-discover all tunable operators
        ops, op_categories = get_all_tunable_ops()

    round_config_dir: Optional[str] = None
    default_order = ["flagos", "vendor", "reference"]
    auto_tune_ops_path = os.path.join(run_dir, "autotune_ops.yaml")
    auto_tune_ops_initial_path = os.path.join(run_dir, "autotune_ops.initial.yaml")
    round_config_dir = os.path.join(run_dir, "autotune_configs")
    os.makedirs(round_config_dir, exist_ok=True)
    best_config_temp_path = os.path.join(run_dir, "best_config_temp.yaml")
    best_op_backends: Dict[str, List[str]] = {}
    tuned_op_backends: Dict[str, List[str]] = {}
    write_best_config(best_config_temp_path, best_op_backends)

    if not ops:
        raise ValueError(
            "No operators to tune. Provide --ops or ensure backends are available."
        )

    logger.info("=== FlagGems Auto-tuning ===")
    logger.info("Throughput args: %s", " ".join(throughput_args))
    logger.info("Run dir: %s", run_dir)
    logger.info("Operators to tune (%d): %s", len(ops), ", ".join(ops))
    logger.info(
        "Operator backends: %s",
        ", ".join(
            f"{op}=[{','.join(sorted(backends))}]"
            for op, backends in op_categories.items()
        ),
    )
    logger.info("Number of runs per configuration: %s", args.num_runs)

    # Counter for log file numbering (starts at 1, formatted as 0001, 0002, ...)
    log_counter = [1]

    # Round 1: baseline (no FlagGems, no OOT)
    logger.info("=== Round 1a: Baseline (no FlagGems, no OOT) ===")
    baseline_result = run_benchmark_multi(
        label="baseline",
        throughput_args=throughput_args,
        num_runs=args.num_runs,
        run_dir=run_dir,
        log_counter=log_counter,
        use_flaggems=False,
        gems_whitelist=None,
        oot_whitelist=None,
        oot_enabled=False,
        config_dir=round_config_dir,
        config_payload=None,
    )
    write_results_csv(
        csv_path,
        baseline_result,
        per_op_results={},
        baseline_fake_result=None,
        baseline_enable_result=None,
        op_backends=tuned_op_backends,
    )

    # Baseline with placeholder whitelist (gems_fake_op) - for overhead measurement
    logger.info("=== Round 1b: Baseline (gems_fake_op whitelist) ===")
    baseline_fake_result = run_benchmark_multi(
        label="baseline_fake_gems_op",
        throughput_args=throughput_args,
        num_runs=args.num_runs,
        run_dir=run_dir,
        log_counter=log_counter,
        use_flaggems=True,
        gems_whitelist=["gems_fake_op"],
        oot_whitelist=None,
        oot_enabled=False,
        config_dir=round_config_dir,
        config_payload=None,
    )

    write_results_csv(
        csv_path,
        baseline_result,
        per_op_results={},
        baseline_fake_result=baseline_fake_result,
        baseline_enable_result=None,
        op_backends=tuned_op_backends,
    )

    # Round 1c: enable all FlagGems to capture actual executed ops
    logger.info("=== Round 1c: FlagGems enable-all (capture ops) ===")
    baseline_enable_result = run_benchmark_multi(
        label="baseline_gems_enable",
        throughput_args=throughput_args,
        num_runs=args.num_runs,
        run_dir=run_dir,
        log_counter=log_counter,
        use_flaggems=True,
        gems_whitelist=None,
        oot_whitelist=None,
        oot_enabled=True,
        config_dir=round_config_dir,
        config_payload=None,
    )

    write_results_csv(
        csv_path,
        baseline_result,
        per_op_results={},
        baseline_fake_result=baseline_fake_result,
        baseline_enable_result=baseline_enable_result,
        op_backends=tuned_op_backends,
    )

    # Ensure initial list includes all discovered ops + FlagGems all ops.
    try:
        from vllm_fl.utils import get_flaggems_all_ops

        gems_all_ops = get_flaggems_all_ops()
    except Exception:
        gems_all_ops = []
    initial_ops = sorted(set(ops) | set(gems_all_ops))
    initial_categories = {op: op_categories.get(op, ["flagos"]) for op in initial_ops}
    for op in gems_all_ops:
        backends = initial_categories.setdefault(op, ["flagos"])
        if "flagos" not in backends:
            backends.append("flagos")

    write_auto_tune_ops_file(
        auto_tune_ops_initial_path, initial_ops, initial_categories
    )
    logger.info("Generated initial auto-tune ops file: %s", auto_tune_ops_initial_path)

    if not args.ops:
        enabled_ops = extract_gems_ops_from_enable_log(
            fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH
        )
        if enabled_ops:
            try:
                from vllm_fl.utils import get_flaggems_all_ops

                registered_ops = set(get_flaggems_all_ops())
            except Exception:
                registered_ops = set()

            enabled_set = set(enabled_ops)
            if registered_ops:
                enabled_set &= registered_ops

            flagos_registered_ops = {
                op for op, backends in op_categories.items() if "flagos" in backends
            }
            backend_ops = {
                op
                for op, backends in op_categories.items()
                if any(b in ("vendor", "reference") for b in backends)
            }

            if enabled_set:
                # Final list: (enabled FlagGems ops ∩ all FlagGems ops) + all registered backend ops.
                # Always include ops that already have flagos backend registered.
                merged_ops = sorted(backend_ops | flagos_registered_ops | enabled_set)
                op_categories = {
                    op: op_categories.get(op, ["flagos"]) for op in merged_ops
                }
                for op in enabled_set:
                    backends = op_categories.setdefault(op, ["flagos"])
                    if "flagos" not in backends:
                        backends.append("flagos")
                ops = merged_ops
                logger.info(
                    "Merged ops with enable log (%d): %s",
                    len(ops),
                    ", ".join(ops),
                )
            else:
                ops = sorted(backend_ops | flagos_registered_ops)
                op_categories = {op: op_categories.get(op, ["flagos"]) for op in ops}
                logger.info(
                    "Enable log ops did not overlap registered FlagGems ops; "
                    "using backend-registered operator list."
                )
        else:
            logger.info(
                "No ops found in FlagGems enable log; keeping existing operator list."
            )

    write_auto_tune_ops_file(auto_tune_ops_path, ops, op_categories)
    logger.info("Generated auto-tune ops file: %s", auto_tune_ops_path)

    # Round 2: per-operator tuning
    # For each operator:
    # - If NOT in OOT: disable OOT, enable FlagGems with whitelist=[op]
    # - If IN OOT: enable only this OOT op, disable FlagGems
    logger.info("=== Round 2: Per-operator tuning ===")
    per_op_results: Dict[str, Optional[Tuple[float, float]]] = {}
    per_op_backend_results: Dict[str, Dict[str, Optional[Tuple[float, float]]]] = {}
    write_results_csv(
        csv_path,
        baseline_result,
        per_op_results,
        baseline_fake_result=baseline_fake_result,
        baseline_enable_result=baseline_enable_result,
        op_backends=tuned_op_backends,
        per_op_backend_results=per_op_backend_results,
    )

    for op in ops:
        label = f"op_{op}"
        backends = op_categories.get(op, [])
        ordered_backends = (
            sorted(set(backends), key=default_order.index) if backends else ["flagos"]
        )
        tuned_op_backends[op] = ordered_backends
        best_result: Optional[Tuple[float, float]] = None
        best_backend: Optional[str] = None
        baseline_total = baseline_result[0] if baseline_result is not None else None

        for backend in ordered_backends:
            op_config_payload = None
            if round_config_dir:
                op_config_payload = {"op_backends": {op: [backend]}}

            if is_oot_op(op):
                if auto_tune_action:
                    logger.info(
                        "Auto-tune: OOT op '%s' -> register only this custom op; backend=%s",
                        op,
                        backend,
                    )
                logger.info("Tuning OOT operator: %s (backend=%s)", op, backend)
                use_flaggems = backend == "flagos"
                gems_whitelist = [op] if use_flaggems else None
                oot_enabled = True
                oot_whitelist = [op]
            else:
                if auto_tune_action:
                    logger.info(
                        "Auto-tune: non-OOT op '%s' -> backend=%s",
                        op,
                        backend,
                    )
                logger.info(
                    "Tuning operator: %s (backend=%s)",
                    op,
                    backend,
                )
                use_flaggems = backend == "flagos"
                gems_whitelist = [op] if use_flaggems else None
                oot_enabled = False
                oot_whitelist = None

            val = run_benchmark_multi(
                label=label,
                throughput_args=throughput_args,
                num_runs=args.num_runs,
                run_dir=run_dir,
                log_counter=log_counter,
                use_flaggems=use_flaggems,
                gems_whitelist=gems_whitelist,
                oot_whitelist=oot_whitelist,
                oot_enabled=oot_enabled,
                config_dir=round_config_dir,
                config_payload=op_config_payload,
            )

            per_op_backend_results.setdefault(op, {})[backend] = val
            write_results_csv(
                csv_path,
                baseline_result,
                per_op_results,
                baseline_fake_result=baseline_fake_result,
                baseline_enable_result=baseline_enable_result,
                op_backends=tuned_op_backends,
                per_op_backend_results=per_op_backend_results,
            )
            if val is not None and baseline_total is not None:
                if best_result is None or val[0] > best_result[0]:
                    best_result = val
                    best_backend = backend

        per_op_results[op] = best_result
        if (
            baseline_total is not None
            and best_result is not None
            and best_backend is not None
            and best_result[0] > baseline_total
        ):
            best_op_backends[op] = [best_backend]
        else:
            best_op_backends.pop(op, None)

        write_best_config(best_config_temp_path, best_op_backends)
        # Write CSV after each op
        write_results_csv(
            csv_path,
            baseline_result,
            per_op_results,
            baseline_fake_result=baseline_fake_result,
            baseline_enable_result=baseline_enable_result,
            op_backends=tuned_op_backends,
            per_op_backend_results=per_op_backend_results,
        )

    # Round 3: combined best operators (throughput > baseline)
    logger.info("=== Round 3: Combined best operators ===")
    if baseline_result is None:
        logger.info("Baseline throughput is None, skipping Round 3.")
        write_best_config(os.path.join(run_dir, "best_config.yaml"), best_op_backends)
        return

    baseline_total = baseline_result[0]
    improved_ops = [
        op
        for op, result in per_op_results.items()
        if result is not None and result[0] > baseline_total
    ]

    if not improved_ops:
        logger.info("No operators improved over baseline. Skipping combined run.")
        op_config_path = os.path.join(run_dir, "op_config.json")
        write_op_config_json(op_config_path, baseline_result, per_op_results)
        write_best_config(os.path.join(run_dir, "best_config.yaml"), best_op_backends)
        _print_summary(baseline_result, per_op_results, None)
        return

    # Separate improved ops into OOT, flagos, and other backends
    improved_oot_ops: List[str] = []
    improved_gems_ops: List[str] = []
    improved_other_ops: List[str] = []
    for op in improved_ops:
        backend = best_op_backends.get(op, [None])[0]
        if is_oot_op(op):
            improved_oot_ops.append(op)
        elif backend == "flagos":
            improved_gems_ops.append(op)
        else:
            improved_other_ops.append(op)

    logger.info(
        "Operators with throughput > baseline: %s",
        ", ".join(f"{op}({best_op_backends[op][0]})" for op in improved_ops),
    )
    logger.info(
        "  OOT ops: %s", ", ".join(improved_oot_ops) if improved_oot_ops else "None"
    )
    logger.info(
        "  FlagGems ops: %s",
        ", ".join(improved_gems_ops) if improved_gems_ops else "None",
    )
    if improved_other_ops:
        logger.info(
            "  Other backends: %s",
            ", ".join(f"{op}({best_op_backends[op][0]})" for op in improved_other_ops),
        )

    # Combined run: enable improved OOT ops + FlagGems whitelist for improved gems ops
    combined_config_payload = None
    if round_config_dir:
        combined_config_payload = {"op_backends": best_op_backends}
    combined_result = run_benchmark_multi(
        label="combined_best_ops",
        throughput_args=throughput_args,
        num_runs=args.num_runs,
        run_dir=run_dir,
        log_counter=log_counter,
        use_flaggems=bool(
            improved_gems_ops
            or any(
                best_op_backends.get(op, [None])[0] == "flagos"
                for op in improved_oot_ops
            )
        ),
        gems_whitelist=improved_gems_ops if improved_gems_ops else None,
        oot_whitelist=improved_oot_ops if improved_oot_ops else None,
        oot_enabled=bool(improved_oot_ops),
        config_dir=round_config_dir,
        config_payload=combined_config_payload,
    )
    write_results_csv(
        csv_path,
        baseline_result,
        per_op_results,
        combined_result=combined_result,
        baseline_fake_result=baseline_fake_result,
        baseline_enable_result=baseline_enable_result,
        op_backends=tuned_op_backends,
        per_op_backend_results=per_op_backend_results,
    )

    op_config_path = os.path.join(run_dir, "op_config.json")
    write_op_config_json(op_config_path, baseline_result, per_op_results)

    write_best_config(os.path.join(run_dir, "best_config.yaml"), best_op_backends)
    _print_summary(baseline_result, per_op_results, combined_result)


def _print_summary(
    baseline_result: Optional[Tuple[float, float]],
    per_op_results: Dict[str, Optional[Tuple[float, float]]],
    combined_result: Optional[Tuple[float, float]],
) -> None:
    """Print summary of tuning results."""
    logger.info("=== Summary ===")
    if baseline_result is not None:
        baseline_total, baseline_output = baseline_result
        if baseline_output > 0:
            logger.info(
                "Baseline: total=%.2f tok/s, output=%.2f tok/s",
                baseline_total,
                baseline_output,
            )
        else:
            logger.info("Baseline: total=%.2f tok/s", baseline_total)
    else:
        baseline_total = 0.0
        logger.info("Baseline throughput: N/A")

    for op, result in per_op_results.items():
        if result is None or baseline_result is None:
            logger.info("  %s: throughput=N/A, speedup=N/A", op)
        else:
            total, output = result
            rel = total / baseline_total if baseline_total > 0 else None
            if output > 0:
                logger.info(
                    "  %s: total=%.2f tok/s, output=%.2f tok/s, speedup=%s vs baseline",
                    op,
                    total,
                    output,
                    f"{rel:.4f}x" if rel is not None else "N/A",
                )
            else:
                logger.info(
                    "  %s: total=%.2f tok/s, speedup=%s vs baseline",
                    op,
                    total,
                    f"{rel:.4f}x" if rel is not None else "N/A",
                )

    if combined_result is not None:
        combined_total, combined_output = combined_result
        if baseline_result is not None:
            combined_rel = (
                combined_total / baseline_total if baseline_total > 0 else None
            )
            if combined_output > 0:
                logger.info(
                    "Combined best ops: total=%.2f tok/s, output=%.2f tok/s "
                    "(%s vs baseline)",
                    combined_total,
                    combined_output,
                    f"{combined_rel:.4f}x" if combined_rel is not None else "N/A",
                )
            else:
                logger.info(
                    "Combined best ops: total=%.2f tok/s (%s vs baseline)",
                    combined_total,
                    f"{combined_rel:.4f}x" if combined_rel is not None else "N/A",
                )
        else:
            if combined_output > 0:
                logger.info(
                    "Combined best ops: total=%.2f tok/s, output=%.2f tok/s",
                    combined_total,
                    combined_output,
                )
            else:
                logger.info("Combined best ops: total=%.2f tok/s", combined_total)
    else:
        logger.warning("Combined best ops throughput: N/A")


if __name__ == "__main__":
    # res = get_all_tunable_ops()
    # print(res)
    main()
