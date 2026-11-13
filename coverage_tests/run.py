# Copyright (c) 2026 BAAI. All rights reserved.

"""Append isolated supplementary test coverage to the core CI run."""

import argparse
from dataclasses import replace
from pathlib import Path

from tests.run import TestCase, TestRunner

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Environment drift in the CI image, isolated at this entry point rather than
# inside the tests: vllm_fl/models/qwen3_8_flash_next/gpu/mtp.py imports
# vllm.model_executor.models.utils.maybe_fuse_shared_experts, which only
# exists in vllm >= 0.33, so the module (168 statements) cannot be imported
# on the vllm 0.24 image. Restore when the image ships vllm >= 0.33.
# An ignore is applied only while its file still exists; if the test is
# renamed or removed, the entry is skipped instead of failing collection.
ENV_DRIFT_IGNORES = [
    "coverage_tests/functional/qwen3_8/test_cov_mtp.py",
]


class CoverageRunner(TestRunner):
    def discover_tests(self) -> list[TestCase]:
        selected = []
        if self.scope in ("unit", "all"):
            selected.extend(self._discover_unit_tests())
        if self.scope in ("functional", "all"):
            selected.extend(self._discover_functional_tests())
        cases = []
        for case in selected:
            scope = case.task
            suite = "unit_tests" if scope == "unit" else "functional_tests"
            args = []
            for arg in case.extra_args:
                if arg.startswith("--json-report-file="):
                    filename = Path(arg.split("=", 1)[1]).name
                    arg = f"--json-report-file={self.output_dir / filename}"
                args.append(arg)
            if "--cov-append" not in args:
                args.append("--cov-append")
            args.extend(
                [
                    "--timeout=180",
                    "-p",
                    "tests.conftest",
                    "-p",
                    f"tests.{suite}.conftest",
                ]
            )
            if scope == "functional":
                for ignore_path in ENV_DRIFT_IGNORES:
                    if (_REPO_ROOT / ignore_path).exists():
                        args.append(f"--ignore={ignore_path}")
                    else:
                        print(
                            "[cov] skipping stale env-drift ignore "
                            f"(file not found): {ignore_path}"
                        )
            cases.append(
                replace(case, pytest_path=f"coverage_tests/{scope}/", extra_args=args)
            )
        cases.extend(self._qwen3_8_cases())
        return cases

    def _qwen3_8_cases(self) -> list[TestCase]:
        """Qwen3.8 research tests that live outside the default collection tree.

        They belong to the model author (#455) and are kept in place; this
        runner is the explicit collection entry that runs them under the
        supplementary coverage data, split by hardware marker.
        """
        base_args = [
            "--tb=short",
            "-q",
            "--cov=vllm_fl",
            "--cov-append",
            "--cov-report=term-missing",
            "--cov-report=json:coverage-nvidia.json",
            "--json-report",
            f"--json-report-file={self.output_dir / 'report-nvidia-qwen3_8.json'}",
            "--timeout=180",
        ]
        return [
            TestCase(
                name="qwen3_8 flash next (cpu)",
                pytest_path="tests/qwen3_8_flash_next/",
                task="unit",
                extra_args=base_args + ["-m", "not gpu"],
            ),
            TestCase(
                name="qwen3_8 flash next (gpu)",
                pytest_path="tests/qwen3_8_flash_next/",
                task="functional",
                extra_args=base_args + ["-m", "gpu"],
            ),
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="nvidia", choices=["nvidia"])
    parser.add_argument("--device", default="a100", choices=["a100"])
    parser.add_argument("--scope", choices=["unit", "functional", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return CoverageRunner(
        platform=args.platform,
        device=args.device,
        scope=args.scope,
        output_dir=str(args.output_dir.resolve()),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
