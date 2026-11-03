# Copyright (c) 2025 BAAI. All rights reserved.

"""
Structured test report generation.

Produces:
- **JUnit XML** for CI systems (GitHub Actions natively parses this).
- **JSON** for trend analysis and dashboards.
- **Console summary** for human readers.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Result of a single test case execution."""

    name: str
    passed: bool
    duration: float = 0.0  # seconds
    message: str = ""
    task: str = ""
    model: str = ""
    case: str = ""
    stdout: str = ""

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"


@dataclass
class TestReport:
    """Aggregate report over a collection of test results."""

    results: list[TestResult] = field(default_factory=list)
    platform: str = ""
    device: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return sum(r.duration for r in self.results)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def finalize(self) -> None:
        """Mark the end time."""
        self.end_time = time.time()

    # --- JUnit XML -----------------------------------------------------------

    def save_junit_xml(self, path: str | Path = "test-results.xml") -> Path:
        """Write JUnit XML report."""
        path = Path(path)
        testsuite = ET.Element(
            "testsuite",
            name=f"{self.platform}-{self.device}",
            tests=str(self.total),
            failures=str(self.failed),
            time=f"{self.duration:.2f}",
        )

        for r in self.results:
            tc = ET.SubElement(
                testsuite,
                "testcase",
                name=r.name,
                classname=f"{r.task}.{r.model}" if r.task else r.name,
                time=f"{r.duration:.2f}",
            )
            if not r.passed:
                failure = ET.SubElement(tc, "failure", message=r.message)
                failure.text = r.stdout[-4000:] if r.stdout else r.message
            if r.stdout:
                out = ET.SubElement(tc, "system-out")
                out.text = r.stdout[-4000:]

        tree = ET.ElementTree(testsuite)
        # ET.indent requires Python 3.9+
        if hasattr(ET, "indent"):
            ET.indent(tree, space="  ")
        path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(path, encoding="unicode", xml_declaration=True)
        return path

    # --- JSON ----------------------------------------------------------------

    def save_json(self, path: str | Path = "test-results.json") -> Path:
        """Write JSON report for trend analysis."""
        path = Path(path)
        data: dict[str, Any] = {
            "platform": self.platform,
            "device": self.device,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "duration": round(self.duration, 2),
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration": round(r.duration, 2),
                    "task": r.task,
                    "model": r.model,
                    "case": r.case,
                    "message": r.message,
                }
                for r in self.results
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    # --- Console summary -----------------------------------------------------

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Test Report: {self.platform}/{self.device}")
        print(f"{'=' * 60}")
        print(f"Total: {self.total}  Passed: {self.passed}  Failed: {self.failed}")
        print(f"Duration: {self.duration:.1f}s")

        if self.failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    label = f"  FAIL  {r.name}"
                    if r.message:
                        label += f" — {r.message}"
                    print(label)

        status = "PASSED" if self.all_passed else "FAILED"
        print(f"\nOverall: {status}")
        print(f"{'=' * 60}\n")
