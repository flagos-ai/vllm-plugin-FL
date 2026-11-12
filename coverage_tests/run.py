# Copyright (c) 2026 BAAI. All rights reserved.

"""Append isolated supplementary test coverage to the core CI run."""

import argparse
from dataclasses import replace
from pathlib import Path

from tests.run import TestCase, TestRunner


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
            cases.append(
                replace(case, pytest_path=f"coverage_tests/{scope}/", extra_args=args)
            )
        return cases


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
