# SPDX-License-Identifier: Apache-2.0
"""Exercise functional shutdown hooks in real serial and xdist subprocesses."""

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# These tests spawn their own xdist workers; avoid nested parallel fan-out.
pytestmark = pytest.mark.serial
ROOT = Path(__file__).resolve().parents[2]


def _run_probe(tmp_path, *, workers, fails=False, late_failure=False):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copyfile(
        ROOT / "tests/functional_tests/conftest.py", project / "conftest.py"
    )
    (project / "test_probe.py").write_text(
        f"def test_first(): assert True\ndef test_second(): assert {not fails!r}\n",
        encoding="utf-8",
    )
    (project / "finish_probe.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "import pytest\n"
        "@pytest.hookimpl(wrapper=True, trylast=True)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    result = yield\n"
        "    if os.environ.get('PROBE_LATE_FAILURE') == '1':\n"
        "        session.exitstatus = pytest.ExitCode.TESTS_FAILED\n"
        "    worker = os.environ.get('PYTEST_XDIST_WORKER', 'master')\n"
        "    (Path(os.environ['PROBE_MARKERS']) / (worker + '.json')).write_text(\n"
        "        json.dumps({'status': int(session.exitstatus)}))\n"
        "    return result\n",
        encoding="utf-8",
    )
    markers = project / "markers"
    markers.mkdir()
    report = project / "report.json"
    junit = project / "junit.xml"
    env = os.environ.copy()
    env.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PROBE_MARKERS": str(markers),
            "PROBE_LATE_FAILURE": "1" if late_failure else "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    env.pop("PYTEST_ADDOPTS", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        os.devnull,
        "--rootdir",
        str(project),
        "-p",
        "xdist.plugin",
        "-p",
        "pytest_jsonreport.plugin",
        "-p",
        "finish_probe",
        "-q",
        "--junitxml",
        str(junit),
        "--json-report",
        "--json-report-file",
        str(report),
    ]
    if workers:
        command.extend(["-n", str(workers)])
    result = subprocess.run(
        command,
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == int(fails or late_failure), output
    assert "node down" not in output.lower(), output
    assert "crashed worker" not in output.lower(), output
    assert "Not properly terminated" not in output, output
    expected_workers = {"master"} | {f"gw{i}" for i in range(workers)}
    assert {p.stem for p in markers.glob("*.json")} == expected_workers, output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["summary"]["total"] == 2, output
    assert data["summary"].get("failed", 0) == int(fails), output
    assert data["summary"]["passed"] == 2 - int(fails), output
    testcases = ET.parse(junit).getroot().findall(".//testcase")
    assert len(testcases) == 2, output
    assert sum(t.find("failure") is not None for t in testcases) == int(fails), output
    return markers, output


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("fails", [False, True])
def test_finish_preserves_reports_exit_status_and_worker_handshake(
    tmp_path, workers, fails
):
    _, output = _run_probe(tmp_path, workers=workers, fails=fails)
    assert "1 failed, 1 passed" in output if fails else "2 passed" in output


def test_finish_preserves_status_changed_by_an_inner_hook(tmp_path):
    markers, _ = _run_probe(tmp_path, workers=0, late_failure=True)
    assert json.loads((markers / "master.json").read_text())["status"] == 1
