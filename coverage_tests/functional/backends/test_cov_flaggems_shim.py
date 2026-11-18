# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify the FlagGems collection shims in fresh interpreters."""

import builtins
import importlib
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIM_PATHS = [
    f"coverage_tests/{area}/_flaggems_shim.py"
    for area in (
        "functional/backends",
        "functional/compilation",
        "unit/compilation",
        "unit/dispatch",
    )
]


def _flag_gems_modules():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "flag_gems" or name.startswith("flag_gems.")
    }


def _check_real_flag_gems(shim_path):
    real = importlib.import_module("flag_gems")
    before = _flag_gems_modules()
    assert before["flag_gems"] is real

    runpy.run_path(shim_path)

    after = _flag_gems_modules()
    assert after.keys() == before.keys()
    for name, module in before.items():
        assert after[name] is module, f"{shim_path} replaced {name}"


def _check_missing_flag_gems(shim_path):
    real_import = builtins.__import__

    def import_without_flag_gems(name, *args, **kwargs):
        if name == "flag_gems" or name.startswith("flag_gems."):
            raise ModuleNotFoundError(name=name)
        return real_import(name, *args, **kwargs)

    assert not _flag_gems_modules()
    with patch("builtins.__import__", import_without_flag_gems):
        runpy.run_path(shim_path)

    placeholder = sys.modules["flag_gems"]
    assert placeholder._FULL_CONFIG == ()
    assert placeholder.runtime is sys.modules["flag_gems.runtime"]
    assert placeholder.runtime.backend is sys.modules["flag_gems.runtime.backend"]
    assert (
        placeholder.runtime.backend.device
        is sys.modules["flag_gems.runtime.backend.device"]
    )


@pytest.mark.parametrize("shim_path", _SHIM_PATHS)
@pytest.mark.parametrize(
    "installed", [pytest.param(True, marks=pytest.mark.gpu), False]
)
def test_flaggems_shim(shim_path, installed):
    """Preserve installed FlagGems and supply the expected missing-package fallback."""
    result = subprocess.run(
        [
            sys.executable,
            __file__,
            shim_path,
            "installed" if installed else "missing",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _, path, mode = sys.argv
    if mode == "installed":
        _check_real_flag_gems(path)
    else:
        _check_missing_flag_gems(path)
