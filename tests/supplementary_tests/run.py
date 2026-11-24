#!/usr/bin/env python3
# Copyright 2025 BAAI. All rights reserved.

"""Multi-platform supplementary coverage test runner — a thin wrapper over tests/run.py.

Subclasses the core ``TestRunner`` and only replaces test discovery so the
supplementary suite runs through the same subprocess/report machinery while
the core entry points (``tests/run.py`` scopes) stay untouched:

* parallel pass: ``-m "supplementary and not multi_gpu and not serial"``
  with xdist workers, writing ``coverage-<platform>-supp.json``;
* serial pass:   ``-m "supplementary and (multi_gpu or serial)"`` with
  ``--cov-append`` into the same JSON.

Coverage data is additionally isolated per platform via a dedicated
``COVERAGE_FILE`` (``.coverage-<platform>-supp``) so that supplementary runs
for different platforms sharing one checkout directory can never append into
each other's data files.

Usage (mirrors the core runner, one invocation per platform)::

    python tests/supplementary_tests/run.py --platform kunlunxin --device p800
    python tests/supplementary_tests/run.py --platform hygon --device bw1000
    python tests/supplementary_tests/run.py --platform iluvatar --device bi_v150
    python tests/supplementary_tests/run.py --platform thead --device zw810e

The platform YAML ``unit.include/exclude`` filters are intentionally NOT
applied here: they gate core unit tests, not supplementary ones.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests.run import TestCase, TestRunner  # noqa: E402

_SUPP_DIR = "tests/supplementary_tests/"


class SupplementaryRunner(TestRunner):
    """Run the supplementary suite with coverage isolated to its own JSON."""

    def _run_single(
        self,
        tc: TestCase,
        extra_env_override: dict[str, str] | None = None,
        _quiet: bool = False,
        device_slots: list[int] | None = None,
        device_ids: list[int] | None = None,
    ):
        """Extend the core exit-code handling to the supplementary serial pass.

        The core runner treats pytest exit 5 (no tests collected) as a pass
        only for ``unit-serial``. The supplementary serial pass selects
        ``multi_gpu``/``serial``-marked tests and may legitimately collect
        none, so the same vacuous-collection exemption is applied here. Real
        test failures still exit 1 and are reported as failures.
        """
        result = super()._run_single(
            tc,
            extra_env_override=extra_env_override,
            _quiet=_quiet,
            device_slots=device_slots,
            device_ids=device_ids,
        )
        if (
            not result.passed
            and tc.name.startswith("supplementary-serial")
            and "pytest exited with code 5" in (result.message or "")
        ):
            result.passed = True
            result.message = "no serial-marked supplementary tests collected"
        return result

    def discover_tests(self) -> list[TestCase]:
        """Build supplementary TestCase descriptors (parallel + serial)."""
        cpu_workers = min(os.cpu_count() or 1, 16)
        platform = self.config.platform
        coverage_file = f".coverage-{platform}-supp"

        parallel_args = [
            "--tb=short",
            "-q",
            "-n",
            str(cpu_workers),
            "-m",
            "supplementary and not multi_gpu and not serial",
            "--cov=vllm_fl",
            "--cov-report=term-missing",
            f"--cov-report=json:coverage-{platform}-supp.json",
            "--json-report",
            f"--json-report-file=report-{platform}-supp.json",
        ]

        serial_args = [
            "--tb=short",
            "-q",
            "-m",
            "supplementary and (multi_gpu or serial)",
            "--cov=vllm_fl",
            "--cov-append",
            "--cov-report=term-missing",
            f"--cov-report=json:coverage-{platform}-supp.json",
            "--json-report",
            f"--json-report-file=report-{platform}-supp-serial.json",
        ]

        print(
            f"[supp] Supplementary tests parallel: {cpu_workers} workers; "
            "multi_gpu and serial-marked tests run serially afterwards"
        )

        return [
            TestCase(
                name=f"supplementary ({platform})",
                pytest_path=_SUPP_DIR,
                task="unit",
                extra_args=parallel_args,
                extra_env={"COVERAGE_FILE": coverage_file},
            ),
            TestCase(
                name=f"supplementary-serial ({platform})",
                pytest_path=_SUPP_DIR,
                task="unit",
                extra_args=serial_args,
                extra_env={"COVERAGE_FILE": coverage_file},
            ),
        ]


def main() -> int:
    """Parse core-runner-compatible args and run the supplementary suite."""
    from tests.run import parse_args

    args = parse_args(sys.argv[1:])

    runner = SupplementaryRunner(
        platform=args.platform,
        device=args.device,
        scope="supplementary",
        stage=args.stage,
        task=args.task,
        model=args.model,
        case=args.case,
        dry_run=args.dry_run,
        save_gold=args.save_gold,
        output_dir=args.output_dir,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
