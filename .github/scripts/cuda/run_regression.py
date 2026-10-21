#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
"""Run one explicitly selected NVIDIA regression on reserved, visible GPUs.

No runner is scheduled and no checkpoint is downloaded. Hy4 TP8/PP2 requires
separate two-node orchestration; it is deliberately not a single-host case.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.utils.nvidia_validation import (
    MANUAL_CASES,
    checkpoint_facts,
    load_manual_case,
    validate_resources,
    verify_execution,
)
from tests.utils.report import TestReport, TestResult


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--case")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", default="nvidia-regression-results")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate checkpoint, idle GPU capacity and installed stack without loading weights",
    )
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(sorted(f"{m}/{c}" for m, c in MANUAL_CASES), indent=2))
        return 0
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference.json").write_text('{"status": "not_run"}\n', encoding="utf-8")
    report = TestReport(platform="cuda", device="explicit-reservation")
    started = time.monotonic()
    error = None
    try:
        if not args.model or not args.case or not args.model_path:
            raise ValueError(
                "--model, --case and --model-path are required; no implicit all-model run"
            )
        cfg, checks = load_manual_case(args.model, args.case, args.model_path)
        checkpoint = checkpoint_facts(cfg.model, checks["expected_architectures"])
        import torch

        if torch.version.cuda is None:
            raise RuntimeError("This entry point requires NVIDIA CUDA")
        devices = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_bytes": torch.cuda.mem_get_info(i)[0],
                "total_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        validate_resources(devices, cfg, checks["minimum_memory_gib"])
        (out / "preflight.json").write_text(
            json.dumps({"checkpoint": checkpoint, "devices": devices}, indent=2) + "\n"
        )
        env = {
            **os.environ,
            "VLLM_PLUGINS": "fl",
            "USE_FLAGGEMS": "1",
            "GEMS_VENDOR": "nvidia",
            "USE_FLAGTUNE": "0",
            "VLLM_USE_V2_MODEL_RUNNER": str(int(checks["runner"] == "v2")),
            "FL_TEST_PLATFORM": "cuda",
            "FL_TEST_MODEL": args.model,
            "FL_TEST_CASE": args.case,
            "FL_NVIDIA_MODEL_PATH": cfg.model,
            "FL_NVIDIA_RUNNER": checks["runner"],
            "FL_NVIDIA_RESULT_PATH": str(out / "inference.json"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        with (out / "stack.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-I", "-B", ".github/scripts/cuda/stack.py"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not args.check_only:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "tests/e2e_tests/inference/test_nvidia_regression.py",
                "--platform=cuda",
                "-v",
                "-s",
                "--tb=short",
                "--json-report",
                f"--json-report-file={out / 'pytest.json'}",
                f"--junitxml={out / 'pytest.xml'}",
            ]
            with (out / "inference.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            verify_execution(out / "pytest.json", out / "inference.json")
    except Exception as exc:
        error = str(exc)
        print(f"NVIDIA regression failed: {exc}", file=sys.stderr)
    report.results.append(
        TestResult(
            name=f"{args.model}/{args.case}",
            passed=error is None,
            task="preflight" if args.check_only else "inference",
            duration=time.monotonic() - started,
            message=error
            or (
                "Preflight only; inference was not executed" if args.check_only else ""
            ),
        )
    )
    report.finalize()
    report.save_json(out / "test-results-cuda.json")
    report.save_junit_xml(out / "test-results-cuda.xml")
    report.print_summary()
    print(f"Artifacts: {out}")
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
