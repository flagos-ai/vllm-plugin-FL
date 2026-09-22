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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Ensure repo root is on sys.path so ``tests.*`` imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests.utils.cleanup import device_cleanup, wait_for_memory
from tests.utils.device_scheduler import DeviceScheduler
from tests.utils.device_utils import get_device_count, get_visible_device_env_var
from tests.utils.model_config import ModelConfig
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
    # Total devices this case occupies (tensor_parallel_size * pipeline_parallel_size).
    # 0 = does not use DeviceScheduler (unit tests, functional tests via xdist).
    # >0 = DeviceScheduler allocates this many contiguous slots before running.
    num_devices: int = 0


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

        # Split cases: e2e cases go through the device scheduler (parallel);
        # everything else (unit, functional, benchmark) runs serially as before.
        e2e_cases = [tc for tc in test_cases if tc.num_devices > 0]
        other_cases = [tc for tc in test_cases if tc.num_devices == 0]

        for tc in other_cases:
            result = self._run_single(tc)
            self.report.results.append(result)

        if e2e_cases:
            self._run_e2e_parallel(e2e_cases)

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

    def _run_e2e_parallel(self, cases: list[TestCase]) -> None:
        """Run e2e cases in parallel, limited by available device slots.

        Uses a DeviceScheduler to hand out non-overlapping device index sets.
        Each case gets exactly ``tc.num_devices`` contiguous slots; the
        VISIBLE_DEVICES env var for that subprocess is set to those indices so
        the test process only sees its own cards.

        Degree of parallelism is determined automatically:
          - total_devices=8, tp=2  → up to 4 cases run at once
          - total_devices=8, tp=4  → up to 2 cases run at once
          - total_devices=8, tp=8  → 1 case at a time (serial)
        """
        total_devices = get_device_count()
        if total_devices == 0:
            # No accelerators detected — run serially without device pinning
            print("[run] E2E parallel: no accelerators detected, running serially")
            for tc in cases:
                result = self._run_single(tc)
                self.report.results.append(result)
                device_cleanup(self.config.platform)
            return

        scheduler = DeviceScheduler(total_devices)
        visible_env_var = get_visible_device_env_var()

        # Max workers = total_devices (worst case: all tp=1 cases)
        max_workers = total_devices

        # --- 调度总览 ---
        max_concurrent = max(total_devices // tc.num_devices for tc in cases)
        print()
        print(f"[run] {'=' * 60}")
        print("[run] E2E parallel schedule")
        print(f"[run]   Total devices : {total_devices}  ({visible_env_var})")
        print(f"[run]   Cases         : {len(cases)}")
        print(f"[run]   Max concurrent: {max_concurrent}  (limited by largest tp)")
        print(f"[run] {'─' * 60}")
        print(f"[run]   {'Case':<40} {'devices':>7}")
        print(f"[run]   {'─' * 40} {'─' * 7}")
        for tc in cases:
            print(f"[run]   {tc.name:<40} {tc.num_devices:>7}")
        print(f"[run] {'=' * 60}")
        print()

        results: list[TestResult] = []
        results_lock = threading.Lock()

        def _ts() -> str:
            return time.strftime("%H:%M:%S")

        def _run_one(tc: TestCase) -> None:
            slots = scheduler.acquire(tc.num_devices)
            visible = ",".join(str(s) for s in slots)
            print(
                f"[run] [{_ts()}] START  {tc.name}  "
                f"{visible_env_var}={visible}  (devices={tc.num_devices}, stdout buffered until done)",
                flush=True,
            )
            status = "FAIL"
            try:
                device_env = {visible_env_var: visible}
                result = self._run_single(
                    tc, extra_env_override=device_env, _quiet=True
                )
                status = "PASS" if result.passed else "FAIL"
                # Flush captured subprocess output with per-case prefix so
                # parallel outputs don't interleave in the terminal/CI log.
                if result.stdout:
                    for line in result.stdout.splitlines():
                        print(f"[{tc.name}] {line}", flush=True)
                device_cleanup(self.config.platform, slots=slots)
            finally:
                scheduler.release(slots)
                print(
                    f"[run] [{_ts()}] {status:<6} {tc.name}  "
                    f"released {visible_env_var}={visible}",
                    flush=True,
                )
            with results_lock:
                results.append(result)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one, tc) for tc in cases]
            for fut in as_completed(futures):
                # Propagate exceptions from worker threads
                fut.result()

        # Preserve original case ordering in the report
        order = {tc.name: i for i, tc in enumerate(cases)}
        results.sort(key=lambda r: order.get(r.name, 0))
        self.report.results.extend(results)

    # --- Test discovery ------------------------------------------------------

    def discover_tests(self) -> list[TestCase]:
        """Discover test cases based on scope, task, model, case filters.

        Scopes:
        - ``unit``: unit tests
        - ``functional``: component-level GPU tests (ops, compilation, distributed)
        - ``e2e``: end-to-end model tests (inference, serving)
        - ``benchmark``: benchmark smoke tests
        - ``all``: all of the above
        """
        cases: list[TestCase] = []

        if self.scope in ("all", "unit"):
            cases.extend(self._discover_unit_tests())

        if self.scope in ("all", "functional"):
            cases.extend(self._discover_functional_tests())

        if self.scope in ("all", "e2e"):
            cases.extend(self._discover_e2e_tests())

        if self.scope in ("all", "benchmark"):
            cases.extend(self._discover_benchmark_tests())

        return cases

    def _discover_unit_tests(self) -> list[TestCase]:
        """Build unit test case from platform config."""
        unit_filter = self.config.get_unit_filter()
        test_path = "tests/unit_tests/"

        # Use explicit CPU count so the log shows the exact parallelism level.
        # os.cpu_count() can return None on exotic systems; fallback to 1.
        # Cap at 16: unit tests are short and pure-CPU; beyond 16 workers the
        # per-process import overhead (torch, vllm_fl) outweighs the parallelism
        # benefit and can cause memory pressure on large CI machines.
        cpu_workers = min(os.cpu_count() or 1, 16)

        extra_args = [
            "--tb=short",
            "-q",
            "-n",
            str(cpu_workers),
            "--cov=vllm_fl",
            "--cov-report=term-missing",
            f"--cov-report=json:coverage-{self.config.platform}.json",
            "--json-report",
            f"--json-report-file=report-{self.config.platform}.json",
        ]

        # Apply exclude patterns
        for pattern in unit_filter.exclude:
            extra_args.extend(["--ignore", f"tests/unit_tests/{pattern}"])

        # Apply include filter (if not wildcard, use -k)
        if unit_filter.include != "*" and isinstance(unit_filter.include, list):
            include_expr = " or ".join(unit_filter.include)
            extra_args.extend(["-k", include_expr])

        print(
            f"[run] Unit tests parallel: {cpu_workers} workers (CPU-only, no device pinning)"
        )

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

            # Compute device footprint for the scheduler.
            # Load the model YAML to read tp/pp sizes so the scheduler can
            # allocate the right number of contiguous device slots.
            num_devices = 1
            try:
                cfg = ModelConfig.load(
                    model,
                    case,
                    platform=self.config.platform,
                    device=self.config.device,
                )
                tp = cfg.engine.get("tensor_parallel_size", 1)
                pp = cfg.engine.get("pipeline_parallel_size", 1)
                num_devices = int(tp) * int(pp)
            except Exception:
                pass

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
                    num_devices=num_devices,
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

        # Parallelize across available devices: each xdist worker gets one device.
        # device_count() returns the number of accelerators visible inside the
        # container, so we never exceed available hardware.
        from tests.utils.device_utils import get_device_count

        num_devices = get_device_count()
        workers = max(num_devices, 1)

        visible_env_var = get_visible_device_env_var()
        print(
            f"[run] Functional tests parallel: {workers} workers, "
            f"device isolation via {visible_env_var}"
        )
        for i in range(workers):
            print(f"[run]   worker gw{i} → {visible_env_var}={i}")

        extra_args = [
            "-v",
            "--tb=short",
            "-s",
            "-n",
            str(workers),
            "--cov=vllm_fl",
            "--cov-append",
            "--cov-report=term-missing",
            f"--cov-report=json:coverage-{self.config.platform}.json",
        ]

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

    def _discover_benchmark_tests(self) -> list[TestCase]:
        """Benchmark smoke tests selected by platform YAML."""
        benchmark = self.config.get_benchmark_tests()
        if not benchmark.get("enabled", False):
            return []
        selected_smoke = benchmark.get("smoke", [])

        if not selected_smoke:
            return []

        if isinstance(selected_smoke, str):
            selected_types = {selected_smoke}
        else:
            selected_types = set(selected_smoke)

        config_path = Path(
            benchmark.get(
                "config_path",
                benchmark.get("config", "tests/benchmarks/configs/smoke.yaml"),
            )
        )
        if not config_path.exists():
            print(f"[run] Warning: benchmark config not found: {config_path}")
            return []

        with open(config_path) as f:
            smoke_config = yaml.safe_load(f) or {}

        cases: list[TestCase] = []
        for bench_type, case_list in smoke_config.items():
            if bench_type not in selected_types:
                continue
            if not isinstance(case_list, list):
                continue

            pytest_path = f"tests/benchmarks/test_benchmark_{bench_type}.py"
            pytest_abspath = _REPO_ROOT / pytest_path
            if not pytest_abspath.exists():
                print(f"[run] Warning: benchmark test file not found: {pytest_path}")
                continue

            for case_cfg in case_list:
                runtime_case = dict(case_cfg)
                model_name = str(runtime_case.pop("model"))
                model_case = str(runtime_case.pop("case"))
                model_cfg = ModelConfig.load(
                    model_name,
                    model_case,
                    platform=self.config.platform,
                    device=self.config.device,
                )

                if "parameters" in runtime_case:
                    params = {
                        "model": model_cfg.model,
                        "tokenizer": model_cfg.model,
                        **model_cfg.engine,
                        **runtime_case.get("parameters", {}),
                    }
                    runtime_case["parameters"] = params

                if "server_parameters" in runtime_case:
                    server_params = {
                        "model": model_cfg.model,
                        "tokenizer": model_cfg.model,
                        **model_cfg.engine,
                        **runtime_case.get("server_parameters", {}),
                    }
                    runtime_case["server_parameters"] = server_params

                if "client_parameters" in runtime_case:
                    client_params = {
                        "tokenizer": model_cfg.model,
                        **runtime_case.get("client_parameters", {}),
                    }
                    if model_cfg.engine.get("trust_remote_code"):
                        client_params.setdefault("trust_remote_code", True)
                    runtime_case["client_parameters"] = client_params

                name = str(runtime_case.get("name", f"{bench_type}_unnamed"))
                cases.append(
                    TestCase(
                        name=f"benchmark/{bench_type}/{name}",
                        pytest_path=pytest_path,
                        task="benchmark",
                        model=bench_type,
                        case=name,
                        extra_args=["-v", "--tb=short", "-s"],
                        extra_env={
                            "FL_BENCHMARK_TYPE": bench_type,
                            "FL_BENCHMARK_CASE": json.dumps(runtime_case),
                        },
                    )
                )

        return cases

    # --- Test execution ------------------------------------------------------

    def _run_single(
        self,
        tc: TestCase,
        extra_env_override: dict[str, str] | None = None,
        _quiet: bool = False,
    ) -> TestResult:
        """Run a single test case via pytest subprocess.

        Args:
            tc: The test case descriptor.
            extra_env_override: Additional env vars merged last (highest priority).
                Used by the device scheduler to inject VISIBLE_DEVICES.
            _quiet: If True, skip the header/command log lines (used by the
                parallel e2e path, which already emits a timestamped START line).
        """
        cmd = self._build_pytest_cmd(tc)

        if not _quiet:
            print(f"[run] --- {tc.name} ---")
            if tc.extra_env:
                env_str = " ".join(f"{k}={v}" for k, v in tc.extra_env.items())
                print(f"[run] Env:     {env_str}")
            if extra_env_override:
                dev_str = " ".join(f"{k}={v}" for k, v in extra_env_override.items())
                print(f"[run] Devices: {dev_str}")
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

        # Wait for sufficient device memory before e2e tests.
        # When running in parallel the memory check runs under the already-acquired
        # device slots, so we only look at the assigned cards.
        if tc.task in ("inference", "serving") and tc.model and tc.case:
            gpu_util = ModelConfig.load(
                tc.model,
                tc.case,
                platform=self.config.platform,
                device=self.config.device,
            ).engine.get("gpu_memory_utilization", 0.9)
            # Extract device indices from VISIBLE_DEVICES override so the
            # memory check only inspects this case's assigned slots and does
            # not block on memory held by other concurrently running cases.
            device_indices: list[int] | None = None
            if extra_env_override:
                for val in extra_env_override.values():
                    parsed = [
                        int(x)
                        for x in val.split(",")
                        if x.strip().lstrip("-").isdigit()
                    ]
                    if parsed:
                        device_indices = parsed
                        break
            ok, info = wait_for_memory(
                self.config.platform, gpu_util, device_indices=device_indices
            )
            if not ok:
                print("[run] FAILED: timed out waiting for device memory")
                return TestResult(
                    name=tc.name,
                    passed=False,
                    duration=0.0,
                    message=f"OOM: timed out waiting for device memory\n{info}",
                    task=tc.task,
                    model=tc.model,
                    case=tc.case,
                )

        # Build subprocess env: process env → tc.extra_env → extra_env_override
        env = {**os.environ}
        if tc.extra_env:
            env.update(tc.extra_env)
        if extra_env_override:
            env.update(extra_env_override)

        start = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=_quiet,
            text=_quiet,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        duration = time.time() - start

        passed = proc.returncode == 0
        message = "" if passed else f"pytest exited with code {proc.returncode}"

        return TestResult(
            name=tc.name,
            passed=passed,
            duration=duration,
            message=message,
            task=tc.task,
            model=tc.model,
            case=tc.case,
            stdout=proc.stdout if _quiet else "",
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
        choices=["all", "unit", "functional", "e2e", "benchmark"],
        default="all",
        help="Which test scope to run: unit, functional (ops/compilation/"
        "distributed), e2e (inference/serving), benchmark, or all (default: all)",
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
