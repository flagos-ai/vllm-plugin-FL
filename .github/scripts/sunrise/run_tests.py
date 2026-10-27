# Copyright 2026 FlagOS Contributors
"""Run finite Sunrise regression suites, rejecting empty or skipped coverage."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

UNIT_PATHS = [
    "tests/unit_tests/dispatch/test_sunrise_attention.py",
    "tests/unit_tests/dispatch/test_sunrise_runtime.py",
    "tests/unit_tests/dispatch/test_sunrise_graph_capture.py",
    "tests/unit_tests/test_sunrise_device_utils.py",
    "tests/unit_tests/test_runner_selection.py",
    "tests/unit_tests/worker/test_worker.py",
]


def junit_counts(path):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    counts = {
        key: sum(int(s.get(key, "0")) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] == 0 or any(
        counts[key] for key in ("failures", "errors", "skipped")
    ):
        raise ValueError(
            "Sunrise requires executed, passing tests with zero skips: " + repr(counts)
        )
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *UNIT_PATHS,
        "-k",
        "sunrise or accelerator_defaults_to_fl_runner or accelerator_rejects_explicit_upstream_v2_runner",
        "--junitxml=" + str(output / "junit.xml"),
        "-v",
    ]
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(
            command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    facts = {"returncode": result.returncode, "passed": False}
    try:
        facts["counts"] = junit_counts(output / "junit.xml")
        facts["passed"] = result.returncode == 0
    except (ValueError, OSError, ET.ParseError) as error:
        facts["error"] = str(error)
    (output / "summary.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))
    return 0 if facts["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
