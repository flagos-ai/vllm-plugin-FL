#!/usr/bin/env python3
# Copyright (c) 2025 BAAI. All rights reserved.

"""
Unified Python test entry point — replaces shell-script orchestration.

Wraps ``pytest`` with platform-aware configuration: tolerance injection,
gold-value comparison, environment setup, and structured reporting.

Usage::

    # Run all tests for a platform/device
    python tests/run.py --platform cuda --device a100

    # Run only functional tests (ops, compilation, distributed)
    python tests/run.py --platform cuda --device a100 --scope functional

    # Run only E2E tests (inference, serving — require model files)
    python tests/run.py --platform cuda --device a100 --scope e2e

    # Run only unit tests
    python tests/run.py --platform ascend --device 910b --scope unit

    # Run a specific E2E test case
    python tests/run.py --platform cuda --device a100 \\
        --scope e2e --task inference --model qwen3 --case 4b_tp2

    # Dry-run — show what would be executed
    python tests/run.py --platform cuda --device a100 --dry-run

    # Save current outputs as gold values
    python tests/run.py --platform cuda --device a100 --save-gold
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure repo root is on sys.path so ``tests.*`` imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests.utils.cleanup import device_cleanup
from tests.utils.platform_config import PlatformConfig
from tests.utils.report import TestReport, TestResult

# ---------------------------------------------------------------------------
# Test case descriptor
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A single test invocation to be run via pytest."""

    name: str
    pytest_path: str
    task: str = ""
    model: str = ""
    case: str = ""
    extra_args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Platform-aware test runner that delegates to pytest.

    Responsibilities:
    - Load platform config and apply env defaults
    - Discover test cases from platform YAML (functional) or filesystem (unit)
    - Build and execute pytest commands
    - Collect results into a structured report
    """

    def __init__(
        self,
        platform: str,
        device: str | None = None,
        scope: str = "all",
        task: str | None = None,
        model: str | None = None,
        case: str | None = None,
        cases: list[dict] | None = None,
        dry_run: bool = False,
        save_gold: bool = False,
        output_dir: str = ".",
        extra_pytest_args: list[str] | None = None,
    ):
        self.config = PlatformConfig.load(platform, device)
        self.scope = scope
        self.task = task
        self.model = model
        self.case = case
        self.cases = cases  # explicit [{model, case}] allow-list from CI matrix
        self.dry_run = dry_run
        self.save_gold = save_gold
        self.output_dir = Path(output_dir)
        self.extra_pytest_args = extra_pytest_args or []

        self.report = TestReport(
            platform=self.config.platform,
            device=self.config.device,
        )

    def run(self) -> int:
        """Discover and run tests, return 0 if all passed."""
        # Apply platform environment defaults
        self.config.apply_env_defaults()

        # Inject platform info as env vars for conftest.py to read
        os.environ["FL_TEST_PLATFORM"] = self.config.platform
        os.environ["FL_TEST_DEVICE"] = self.config.device

        test_cases = self.discover_tests()

        if not test_cases:
            print("[run] No test cases found for the given filters.")
            return 0

        print(f"[run] Platform: {self.config.platform}")
        print(f"[run] Device:   {self.config.device}")
        print(f"[run] Scope:    {self.scope}")
        print(f"[run] Cases:    {len(test_cases)}")
        print()

        for tc in test_cases:
            result = self._run_single(tc)
            self.report.results.append(result)

            # Clean up device resources between e2e tests to prevent
            # GPU/NPU memory leaks from cascading into subsequent cases
            if tc.task in ("inference", "serving"):
                device_cleanup(self.config.platform)

        self.report.finalize()
        self.report.print_summary()

        # Save reports
        xml_path = self.output_dir / f"test-results-{self.config.platform}.xml"
        json_path = self.output_dir / f"test-results-{self.config.platform}.json"
        self.report.save_junit_xml(xml_path)
        self.report.save_json(json_path)
        print(f"[run] JUnit XML: {xml_path}")
        print(f"[run] JSON:      {json_path}")

        return 0 if self.report.all_passed else 1

    # --- Test discovery ------------------------------------------------------

    def discover_tests(self) -> list[TestCase]:
        """Discover test cases based on scope, task, model, case filters.

        Scopes:
        - ``unit``: unit tests
        - ``functional``: component-level GPU tests (ops, compilation, distributed)
        - ``e2e``: end-to-end model tests (inference, serving)
        - ``all``: all of the above
        """
        cases: list[TestCase] = []

        if self.scope in ("all", "unit"):
            cases.extend(self._discover_unit_tests())

        if self.scope in ("all", "functional"):
            cases.extend(self._discover_functional_tests())

        if self.scope in ("all", "e2e"):
            cases.extend(self._discover_e2e_tests())

        return cases

    def _discover_unit_tests(self) -> list[TestCase]:
        """Build unit test case from platform config."""
        unit_filter = self.config.get_unit_filter()
        test_path = "tests/unit_tests/"

        extra_args = [
            "--tb=short",
            "--cov=vllm_fl",
            "--cov-report=term-missing",
            f"--cov-report=json:coverage-{self.config.platform}.json",
            "--json-report",
            f"--json-report-file=report-{self.config.platform}.json",
            "-q",
        ]

        # Apply exclude patterns
        for pattern in unit_filter.exclude:
            extra_args.extend(["--ignore", f"tests/unit_tests/{pattern}"])

        # Apply include filter (if not wildcard, use -k)
        if unit_filter.include != "*" and isinstance(unit_filter.include, list):
            include_expr = " or ".join(unit_filter.include)
            extra_args.extend(["-k", include_expr])

        return [
            TestCase(
                name=f"unit ({self.config.platform})",
                pytest_path=test_path,
                task="unit",
                extra_args=extra_args,
            )
        ]

    def _discover_from_yaml(
        self,
        base_dir: str,
    ) -> list[TestCase]:
        """Build test cases from platform YAML config for e2e tests.

        For inference tasks, routes to the unified ``test_inference_smoke.py``
        and injects ``FL_TEST_MODEL``/``FL_TEST_CASE`` env vars so the smoke
        test can load the correct model YAML config.

        For other tasks (serving), falls back to ``-k model`` filtering.

        Args:
            base_dir: Root directory (e.g. ``tests/e2e_tests``).
        """
        func = self.config.get_e2e_tests()
        raw_cases = func.get_cases(task=self.task, model=self.model)

        # Filter by case name if specified
        if self.case:
            raw_cases = [c for c in raw_cases if c["case"] == self.case]

        # Apply explicit cases allow-list from CI matrix (PR smart-skip)
        if self.cases is not None:
            allowed = {(c["model"], c["case"]) for c in self.cases}
            raw_cases = [c for c in raw_cases if (c["model"], c["case"]) in allowed]

        cases: list[TestCase] = []
        for c in raw_cases:
            task = c["task"]
            model = c["model"]
            case = c["case"]

            # Skip unsupported features
            if self.config.should_skip_model(model):
                print(f"[run] Skipping {task}/{model}/{case} (unsupported feature)")
                continue

            test_dir = f"{base_dir}/{task}"
            if not Path(test_dir).exists():
                print(f"[run] Warning: test dir not found: {test_dir}")
                continue

            extra_args = ["-v", "--tb=short", "-s"]
            extra_env: dict[str, str] = {}

            if task == "inference":
                # Route to unified smoke test with env-based config
                pytest_path = f"{test_dir}/test_inference_smoke.py"
                extra_env = {
                    "FL_TEST_MODEL": model,
                    "FL_TEST_CASE": case,
                }
            elif task == "serving":
                # Route to unified serving smoke test with env-based config
                pytest_path = f"{test_dir}/test_serving_smoke.py"
                extra_env = {
                    "FL_TEST_MODEL": model,
                    "FL_TEST_CASE": case,
                }
            else:
                # Other tasks: use directory with -k filter
                pytest_path = test_dir
                if model:
                    extra_args.extend(["-k", model])

            name = f"{task}/{model}/{case}"
            cases.append(
                TestCase(
                    name=name,
                    pytest_path=pytest_path,
                    task=task,
                    model=model,
                    case=case,
                    extra_args=extra_args,
                    extra_env=extra_env,
                )
            )

        return cases

    def _discover_functional_tests(self) -> list[TestCase]:
        """Component-level GPU tests (ops, compilation, distributed).

        Runs all tests under tests/functional_tests/ with include/exclude
        filtering, similar to unit tests.
        """
        func_filter = self.config.get_functional_filter()
        test_path = "tests/functional_tests/"

        extra_args = ["-v", "--tb=short", "-s"]

        # Apply exclude patterns
        for pattern in func_filter.exclude:
            extra_args.extend(["--ignore", f"tests/functional_tests/{pattern}"])

        # Apply include filter (if not wildcard, use -k)
        if func_filter.include != "*" and isinstance(func_filter.include, list):
            include_expr = " or ".join(func_filter.include)
            extra_args.extend(["-k", include_expr])

        return [
            TestCase(
                name=f"functional ({self.config.platform})",
                pytest_path=test_path,
                task="functional",
                extra_args=extra_args,
            )
        ]

    def _discover_e2e_tests(self) -> list[TestCase]:
        """End-to-end model tests (inference, serving)."""
        return self._discover_from_yaml("tests/e2e_tests")

    # --- Test execution ------------------------------------------------------

    def _run_single(self, tc: TestCase) -> TestResult:
        """Run a single test case via pytest subprocess."""
        cmd = self._build_pytest_cmd(tc)

        print(f"[run] --- {tc.name} ---")
        if tc.extra_env:
            env_str = " ".join(f"{k}={v}" for k, v in tc.extra_env.items())
            print(f"[run] Env:     {env_str}")
        print(f"[run] Command: {' '.join(cmd)}")

        if self.dry_run:
            print("[run] (dry-run, skipping)")
            return TestResult(
                name=tc.name,
                passed=True,
                task=tc.task,
                model=tc.model,
                case=tc.case,
                message="dry-run",
            )

        # Merge extra env vars (e.g. FL_TEST_MODEL/FL_TEST_CASE for inference)
        env = None
        if tc.extra_env:
            env = {**os.environ, **tc.extra_env}

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=False,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        duration = time.time() - start

        passed = result.returncode == 0
        message = "" if passed else f"pytest exited with code {result.returncode}"

        return TestResult(
            name=tc.name,
            passed=passed,
            duration=duration,
            message=message,
            task=tc.task,
            model=tc.model,
            case=tc.case,
        )

    def _build_pytest_cmd(self, tc: TestCase) -> list[str]:
        """Build the full pytest command for a test case."""
        cmd = [sys.executable, "-m", "pytest", tc.pytest_path]

        # Inject platform/device as pytest options
        cmd.extend(
            [
                f"--platform={self.config.platform}",
                f"--device={self.config.device}",
            ]
        )

        # Save-gold mode
        if self.save_gold:
            cmd.append("--save-gold")

        cmd.extend(tc.extra_args)
        cmd.extend(self.extra_pytest_args)

        return cmd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified test runner with platform-aware configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name (e.g., cuda, ascend)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device type (e.g., a100, 910b). Defaults to first in platform config.",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "unit", "functional", "e2e"],
        default="all",
        help="Which test scope to run: unit, functional (ops/compilation/"
        "distributed), e2e (inference/serving), or all (default: all)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Functional test task filter (e.g., inference, serve)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name filter (e.g., qwen3)",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Test case variant filter (e.g., 4b_tp2)",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help='JSON array of {model, case} dicts to run, e.g. \'[{"model":"qwen3","case":"4b_tp2"}]\'. '
        "When set, only the listed model/case combinations are executed. "
        "Takes precedence over --model/--case when filtering e2e tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--save-gold",
        action="store_true",
        help="Save test outputs as gold values instead of comparing",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for report output files (default: cwd)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases_filter: list[dict] | None = None
    if args.cases:
        cases_filter = json.loads(args.cases)

    runner = TestRunner(
        platform=args.platform,
        device=args.device,
        scope=args.scope,
        task=args.task,
        model=args.model,
        case=args.case,
        cases=cases_filter,
        dry_run=args.dry_run,
        save_gold=args.save_gold,
        output_dir=args.output_dir,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
