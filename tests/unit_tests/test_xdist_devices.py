# SPDX-License-Identifier: Apache-2.0
"""Worker visibility must be set before importing the test conftest."""

import os
import subprocess
import sys
from pathlib import Path


def test_worker_visibility_isolated_before_conftest_import(tmp_path):
    (tmp_path / "conftest.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "worker = os.environ.get('PYTEST_XDIST_WORKER', 'controller')\n"
        "Path(worker + '.mask').write_text(os.environ['MACA_VISIBLE_DEVICES'])\n"
    )
    (tmp_path / "test_visibility.py").write_text(
        "import os\nimport pytest\n"
        "@pytest.mark.parametrize('case', range(4))\n"
        "def test_mask(case):\n"
        "    expected = {'gw0': '3', 'gw1': '6'}\n"
        "    assert os.environ['MACA_VISIBLE_DEVICES'] == expected[os.environ['PYTEST_XDIST_WORKER']]\n"
    )
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, MACA_VISIBLE_DEVICES="3,6", PYTHONPATH=str(repo))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "2",
            "-p",
            "tests.utils.xdist_devices",
            "--test-device-env=MACA_VISIBLE_DEVICES",
            str(tmp_path / "test_visibility.py"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "gw0.mask").read_text() == "3"
    assert (tmp_path / "gw1.mask").read_text() == "6"
    assert env["MACA_VISIBLE_DEVICES"] == "3,6"
