# Copyright 2026 FlagOS Contributors
"""Run actual THead pytest suites, rejecting missing collection and skips."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# -I removes the script directory from sys.path. Load this exact sibling
# helper without changing business-package search paths.
_common_spec = importlib.util.spec_from_file_location(
    "_thead_ci_common", Path(__file__).resolve().with_name("common.py")
)
_common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)
run_command = _common.run_command
save_json = _common.save_json
validate_junit = _common.validate_junit

REPO = Path(__file__).resolve().parents[3]
SUITES = (
    ("cpu-backend", 202, ["tests/unit_tests/dispatch/test_thead_backend_cpu.py"]),
    (
        "cpu-cache",
        5,
        [
            "tests/unit_tests/dispatch/test_thead_cache.py",
            "-m",
            "not gpu",
        ],
    ),
    (
        "gpu",
        45,
        [
            "tests/unit_tests/dispatch/test_thead_cache.py",
            "tests/unit_tests/dispatch/test_thead_attention.py",
            "-m",
            "gpu",
        ],
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for label, count, selection in SUITES:
        env = dict(os.environ)
        env.pop("FL_BACKEND", None)
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
        # The full-class CPU fixture deliberately isolates package parents. It
        # runs in a separate process and is not evidence of normal registration.
        if label == "cpu-backend":
            env.update(CUDA_VISIBLE_DEVICES="", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        junit = args.output_dir / (label + ".xml")
        argv = [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "pytest",
            *selection,
            "-o",
            "addopts=",
            "-q",
            "--junitxml=" + str(junit.resolve()),
        ]
        receipt = run_command(
            argv,
            args.output_dir / label,
            timeout=900,
            env=env,
            cwd=REPO,
        )
        row = {"scope": label, "expected": count, "receipt": receipt, "passed": False}
        try:
            row["junit"] = validate_junit(junit, count)
            row["passed"] = receipt["clean"]
        except Exception as exc:
            row["validation_error"] = repr(exc)
        results.append(row)
        save_json(args.output_dir / "summary.json", {"suites": results})
        if not receipt["cleanup_complete"]:
            break
    cleanup_complete = bool(results) and all(
        row["receipt"]["cleanup_complete"] for row in results
    )
    passed = len(results) == len(SUITES) and all(row["passed"] for row in results)
    save_json(
        args.output_dir / "summary.json",
        {
            "passed": passed,
            "only_owned_cleanup_complete": cleanup_complete,
            "suites": results,
            "gpu_pytest_expected": 45,
            "cpu_cache_expected": 5,
            "isolated_backend_cpu_expected": 202,
            "tensor_aot_metadata_required": False,
            "external_fa3_oracle_24_and_gdn_72_included": False,
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
