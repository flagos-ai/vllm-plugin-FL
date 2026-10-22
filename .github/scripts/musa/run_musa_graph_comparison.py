#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Run owned MUSA synthetic policies serially; any failure blocks later GPU work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

POLICIES = ("current", "admission", "public_drop", "native")
IMAGE = "harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:8f865d2e2ee21e62b1c0c56f06079c072f9c2023d82fe7e240af4919fc75ce06"


def sealed_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def execution_safe(result):
    return (
        type(result.get("actual_exit")) is int
        and result.get("actual_exit") == 0
        and all(
            result.get(key) is True
            for key in (
                "process_started",
                "direct_process_reaped",
                "cleanup_confirmed",
                "safe_to_continue",
            )
        )
        and type(result.get("remaining_owned_children_count")) is int
        and result.get("remaining_owned_children_count") == 0
        and result.get("deadline_exceeded") is False
        and result.get("deadline_or_signal_exit") is False
    )


def worker_contract(worker):
    spec = importlib.util.spec_from_file_location("_comparison_contract", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_passed(report, policy, contract):
    try:
        expected_capture = 1900 if policy == "admission" else 6171
        expected_runtime = (6171, 0) if policy == "native" else (1900, 4271)
        source = report["source_identity"]
        identity = {
            key: os.environ.get(key)
            for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        }
        return (
            report["status"] == "PASS"
            and report["policy"] == policy
            and report["capture_sizes"] == list(contract.SIZES)
            and report["pieces_per_size"] == 121
            and report["exec_cap"] == 1900
            and report["source_baseline_commit"]
            == "9ea227b705c3bcf77e94ce70e2502a2fc2f2e8f6"
            and set(source) == set(contract.EXPECTED_SOURCE_SHA256)
            and all(
                source[name]["sha256"] == digest
                for name, digest in contract.EXPECTED_SOURCE_SHA256.items()
            )
            and report["github_identity"] == identity
            and report["runtime"]["declared_container_image"] == IMAGE
            and report["runtime"]["visible_device_count"] == 4
            and report["runtime"]["device_ordinal"] == 0
            and report["capture_requested"] == 6171
            and report["native_capture_completed"] == expected_capture
            and report["capture_exact_checks"] == 6171
            and (
                report["postcapture_validation_outcomes"]["native_replay"],
                report["postcapture_validation_outcomes"]["eager"],
            )
            == expected_runtime
            and len(report["runtime_cycles"]) == 2
            and all(
                row["cycle"] == index
                and row["exact_checks"] == 6171
                and (row["native_replay"], row["eager"]) == expected_runtime
                for index, row in enumerate(report["runtime_cycles"], 1)
            )
            and (
                report["explicit_python_operations"]["ops_loader"] > 0
                if policy == "current"
                else report["explicit_python_operations"]["ops_loader"] == 0
            )
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def run_comparison(
    root,
    timeout_seconds,
    run,
    worker=None,
    executable=None,
    total_seconds=3000,
    clock=time.monotonic,
):
    worker = worker or Path(__file__).with_name("musa_graph_compare_worker.py")
    executable = executable or sys.executable
    contract = worker_contract(worker)
    rows = []
    deadline = clock() + total_seconds
    for policy in POLICIES:
        destination = root / policy
        budget = min(timeout_seconds, deadline - clock())
        try:
            execution = run(
                [
                    executable,
                    "-I",
                    "-B",
                    str(worker),
                    "--policy",
                    policy,
                    "--output-dir",
                    str(destination),
                ],
                budget,
            )
        except BaseException as exc:
            execution = {
                "actual_exit": None,
                "safe_to_continue": False,
                "cleanup_confirmed": False,
                "controller_exception": type(exc).__name__ + ": " + str(exc),
            }
        report = None
        error = None
        try:
            path = destination / "result.json"
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Worker report exceeds finite size limit")
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            error = str(exc)
        passed = (
            execution_safe(execution)
            and isinstance(report, dict)
            and report_passed(report, policy, contract)
        )
        row = {
            "policy": policy,
            "execution": execution,
            "worker_report": report,
            "report_read_error": error,
            "passed": passed,
        }
        sealed_json(root / (policy + "-execution.json"), row)
        rows.append(row)
        if not passed:
            break
    summary = {
        "status": "PASS"
        if len(rows) == len(POLICIES) and all(row["passed"] for row in rows)
        else "FAIL",
        "requested_policies": list(POLICIES),
        "completed_policies": [row["policy"] for row in rows],
        "blocked_policies": list(POLICIES[len(rows) :]),
        "results": rows,
        "no_GPU_queries_after_failure": True,
        "owned_process_reaping_is_not_GPU_idle_certification": True,
        "synthetic_only": True,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "budget_seconds": total_seconds,
    }
    sealed_json(root / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant-timeout-seconds", type=int, default=1500)
    parser.add_argument("--total-timeout-seconds", type=int, default=3000)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if os.environ.get("MUSA_CI_SUITE_IMAGE") != IMAGE:
        sealed_json(
            root / "comparison.json",
            {
                "status": "FAIL",
                "reason": "fixed_container_image_mismatch",
                "GPU_children_started": 0,
            },
        )
        return 1
    if (
        not 1 <= args.variant_timeout_seconds <= 1500
        or not 1 <= args.total_timeout_seconds <= 3000
    ):
        raise ValueError("Comparison bounds require variant<=1500s and total<=3000s")
    identity = {
        key: os.environ.get(key)
        for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    }
    if not re.fullmatch(r"[0-9a-f]{40}", identity["GITHUB_SHA"] or "") or any(
        not re.fullmatch(r"[1-9][0-9]*", identity[key] or "")
        for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
    ):
        sealed_json(
            root / "comparison.json",
            {
                "status": "FAIL",
                "reason": "missing_official_run_identity",
                "GPU_children_started": 0,
            },
        )
        return 1
    if shutil.which("timeout") is None:
        raise RuntimeError("GNU timeout is required")
    source = Path(__file__).with_name("run_ci_stage.py")
    spec = importlib.util.spec_from_file_location("_comparison_owned_runner", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.prepare_owned_reaping()
    result = run_comparison(
        root,
        args.variant_timeout_seconds,
        helper.bounded_run,
        total_seconds=args.total_timeout_seconds,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
