# SPDX-License-Identifier: Apache-2.0
"""Check diagnostics survive failed tests and delayed worker completion."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("workers", [0, 2])
def test_live_failure_and_active_test_are_preserved(tmp_path, workers):
    test_file = tmp_path / "test_repro.py"
    test_file.write_text(
        "import ctypes\n"
        "def test_failure():\n"
        "    assert 1 == 2, 'intentional diagnostic regression'\n"
        "def test_slow():\n"
        "    ctypes.PyDLL(None).usleep(500000)\n"
    )
    diagnostics = tmp_path / "diagnostics"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "tests.utils.pytest_diagnostics",
        f"--test-diagnostics-dir={diagnostics}",
        "--test-stack-timeout=0.1",
        "-q",
        str(test_file),
    ]
    if workers:
        command.extend(["-n", str(workers)])
    repo = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")]),
    )
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 1, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "FAILED NOW:" in output
    assert output.index("FAILED NOW:") < output.index("short test summary info")
    records = [
        json.loads(line)
        for path in diagnostics.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    failure = next(record for record in records if record.get("outcome") == "failed")
    assert failure["nodeid"].endswith("test_failure")
    assert "intentional diagnostic regression" in failure["traceback"]
    assert any(record.get("nodeid", "").endswith("test_slow") for record in records)
    stacks = "\n".join(path.read_text() for path in diagnostics.glob("*-stacks.log"))
    assert "Timeout (" in stacks
    assert "test_slow" in stacks
    assert "1 failed, 1 passed" in output
