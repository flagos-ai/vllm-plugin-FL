# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend's hard exit must preserve xdist results and failure status."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.serial
@pytest.mark.parametrize("fails", [False, True])
def test_ascend_functional_exit_preserves_xdist_result(tmp_path, fails):
    pytest.importorskip("xdist")
    source = Path(__file__).resolve().parents[1] / "functional_tests/conftest.py"
    shutil.copyfile(source, tmp_path / "conftest.py")
    (tmp_path / "test_smoke.py").write_text(
        f"def test_smoke():\n    assert {not fails}\n"
    )
    env = dict(
        os.environ,
        FL_TEST_PLATFORM="ascend",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-n",
            "1",
            "-q",
            "--color=no",
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
    assert result.returncode == int(fails), result.stdout
    assert f"1 {'failed' if fails else 'passed'}" in result.stdout, result.stdout
    assert "crashed" not in result.stdout.lower(), result.stdout
