#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Run the existing MUSA test commands with owned-child deadline/reap evidence."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate E2E matrix key")
        result[key] = value
    return result


def matrix_groups(raw, suite_image):
    if len(raw.encode()) > 2 * 1024 * 1024:
        raise ValueError("E2E matrix exceeds the size limit")
    groups = json.loads(raw, object_pairs_hook=unique_keys)
    if not isinstance(groups, list) or len(groups) > 128:
        raise ValueError("E2E matrix must be a bounded list")
    for group in groups:
        if not isinstance(group, dict) or set(group) - {
            "task",
            "device",
            "cases",
            "timeout",
            "image",
        }:
            raise ValueError("Unexpected E2E group fields")
        if group.get("task") not in {"inference", "serving"}:
            raise ValueError("Unknown E2E task")
        device = group.get("device")
        if not isinstance(device, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", device
        ):
            raise ValueError("Invalid E2E device")
        timeout = group.get("timeout")
        if type(timeout) is not int or not 1 <= timeout <= 500:
            raise ValueError("Invalid E2E group timeout")
        if group.get("image") not in {None, "", suite_image}:
            raise ValueError("A single MUSA container cannot change image per group")
        raw_cases = group.get("cases")
        if not isinstance(raw_cases, str):
            raise ValueError("E2E cases must be the existing encoded JSON list")
        cases = json.loads(raw_cases, object_pairs_hook=unique_keys)
        if not isinstance(cases, list) or not cases:
            raise ValueError("E2E group must contain cases")
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"model", "case"}:
                raise ValueError("Invalid E2E case fields")
            if any(
                not isinstance(value, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
                for value in case.values()
            ):
                raise ValueError("Invalid E2E model or case")
    return groups


def prepare_owned_reaping():
    if sys.platform != "linux":
        raise RuntimeError("Owned descendant reaping requires the Linux CI container")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Cannot establish this helper as a child subreaper")
    value = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
        raise RuntimeError("Cannot verify this helper's child subreaper state")


def reap_owned_children():
    """Reap only children assigned to this helper by the kernel, never host PIDs."""
    deadline = time.monotonic() + 2
    reaped = 0
    while True:
        try:
            while True:
                try:
                    child, _status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if child == 0:
                    break
                reaped += 1
            # This kernel interface contains only this helper's direct/adopted
            # children. No process arguments, environment or account data is read.
            raw = Path(f"/proc/self/task/{os.getpid()}/children").read_text()
            children = raw.split()
            if any(not child.isdecimal() for child in children):
                raise ValueError("Invalid own-child kernel report")
        except (OSError, ValueError):
            return {
                "adopted_children_reaped": reaped,
                "remaining_owned_children_count": None,
                "cleanup_confirmed": False,
            }
        if not children:
            return {
                "adopted_children_reaped": reaped,
                "remaining_owned_children_count": 0,
                "cleanup_confirmed": True,
            }
        if time.monotonic() >= deadline:
            return {
                "adopted_children_reaped": reaped,
                "remaining_owned_children_count": len(children),
                "cleanup_confirmed": False,
            }
        time.sleep(0.02)


def bounded_run(argv, remaining_seconds):
    if remaining_seconds <= 0:
        return {
            "actual_exit": 124,
            "process_started": False,
            "direct_process_reaped": False,
            "deadline_exceeded": True,
            "cleanup_confirmed": True,
            "safe_to_continue": False,
        }
    # A new session and GNU timeout restrict deadline signals to this command's
    # process group. The subreaper also detects surviving orphan descendants,
    # including descendants that changed session/group. Uncertainty stops GPU work.
    process = subprocess.Popen(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            f"{remaining_seconds:.3f}s",
            *argv,
        ],
        start_new_session=True,
    )
    actual_exit = process.wait()
    cleanup = reap_owned_children()
    timed_out = actual_exit == 124
    stop_required = actual_exit in {124, 137} or actual_exit < 0
    return {
        "actual_exit": actual_exit,
        "process_started": True,
        "direct_process_reaped": True,
        "deadline_exceeded": timed_out,
        "deadline_or_signal_exit": stop_required,
        **cleanup,
        "safe_to_continue": not stop_required and cleanup["cleanup_confirmed"],
    }


def run_groups(groups, output_root, run=bounded_run, clock=time.monotonic):
    results = []
    for index, group in enumerate(groups):
        output = output_root / f"e2e-{index}-{group['task']}-{group['device']}"
        output.mkdir(parents=True, exist_ok=False)
        started = clock()
        budget = group["timeout"] * 60
        shared = [
            "--platform",
            "musa",
            "--device",
            group["device"],
            "--cases",
            group["cases"],
        ]
        model_result = run(
            [
                sys.executable,
                ".github/scripts/generate_matrix.py",
                *shared,
                "--check-models",
            ],
            budget,
        )
        test_result = None
        if model_result["actual_exit"] == 0 and model_result["safe_to_continue"]:
            test_result = run(
                [
                    sys.executable,
                    "tests/run.py",
                    *shared,
                    "--scope",
                    "e2e",
                    "--task",
                    group["task"],
                    "--output-dir",
                    str(output),
                ],
                budget - (clock() - started),
            )
        safe = model_result["safe_to_continue"] and (
            test_result is None or test_result["safe_to_continue"]
        )
        row = {
            "group_index": index,
            "task": group["task"],
            "device": group["device"],
            "cases": json.loads(group["cases"]),
            "timeout_minutes": group["timeout"],
            "model_check": model_result,
            "test_execution": test_result,
            "model_check_exit": model_result["actual_exit"],
            "test_exit": test_result["actual_exit"] if test_result else None,
            "passed": model_result["actual_exit"] == 0
            and test_result is not None
            and test_result["actual_exit"] == 0
            and safe,
            "safe_to_continue": safe,
        }
        (output / "execution.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        results.append(row)
        if not safe:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope", choices=["unit", "functional", "e2e", "benchmark"], required=True
    )
    args = parser.parse_args()
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as output:
            output.write("gpu_stages_safe=false\n")
    if shutil.which("timeout") is None:
        raise ValueError("GNU timeout is required to retain the existing E2E bounds")
    output_root = Path(os.environ["MUSA_SUITE_ARTIFACT_DIR"]).resolve(strict=True)
    prepare_owned_reaping()
    if args.scope == "e2e":
        groups = matrix_groups(
            os.environ["MUSA_CI_E2E_MATRIX"], os.environ["MUSA_CI_SUITE_IMAGE"]
        )
        results = run_groups(groups, output_root)
        requested_count = len(groups)
    else:
        minutes = {"unit": 360, "functional": 15, "benchmark": 30}[args.scope]
        report_root = output_root / args.scope
        report_root.mkdir(exist_ok=True)
        execution = bounded_run(
            [
                sys.executable,
                "tests/run.py",
                "--platform",
                "musa",
                "--scope",
                args.scope,
                "--output-dir",
                str(report_root),
            ],
            minutes * 60,
        )
        row = {
            "scope": args.scope,
            "timeout_minutes": minutes,
            "execution": execution,
            "passed": execution["actual_exit"] == 0 and execution["safe_to_continue"],
            "safe_to_continue": execution["safe_to_continue"],
        }
        (report_root / "execution.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        results = [row]
        requested_count = 1
    summary = {
        "schema_version": 1,
        "scope": args.scope,
        "original_adaptation_gate_included": False,
        "groups": results,
        "group_count": len(results),
        "requested_group_count": requested_count,
        "gpu_stages_safe": all(row["safe_to_continue"] for row in results),
        "status": "SKIPPED"
        if not results
        else "PASS"
        if all(row["passed"] for row in results)
        else "FAIL",
        "passed": all(row["passed"] for row in results) if results else None,
    }
    (output_root / f"{args.scope}-execution.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary))
    if output_file and summary["gpu_stages_safe"]:
        with open(output_file, "a") as output:
            output.write("gpu_stages_safe=true\n")
    return 0 if not results or summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
