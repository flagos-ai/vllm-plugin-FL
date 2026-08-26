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

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import setuptools


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_setup(monkeypatch: pytest.MonkeyPatch, soc_version: str | None):
    monkeypatch.setenv("VLLM_VENDOR", "ascend")
    monkeypatch.delenv("SOC_VERSION", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    if soc_version is not None:
        monkeypatch.setenv("SOC_VERSION", soc_version)

    setup_calls = []
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: setup_calls.append(kwargs))

    setup_globals = runpy.run_path(str(ROOT_DIR / "setup.py"))
    assert len(setup_calls) == 1
    return setup_globals, setup_calls[0]


@pytest.mark.parametrize(
    ("soc_version", "expected_prebuilt_dir"),
    [
        (None, "ascend910_93"),
        ("ascend910b", "ascend910b1"),
    ],
)
def test_ascend_build_selects_prebuilt_dir(
    monkeypatch: pytest.MonkeyPatch,
    soc_version: str | None,
    expected_prebuilt_dir: str,
):
    setup_globals, setup_kwargs = _load_setup(monkeypatch, soc_version)

    assert setup_globals["_ascend_prebuilt_dir"]() == expected_prebuilt_dir
    assert [extension.name for extension in setup_kwargs["ext_modules"]] == [
        "vllm_fl.dispatch.backends.vendor.ascend.prebuilt."
        f"{expected_prebuilt_dir}.lib._C_ascend"
    ]


def test_ascend_opp_build_defaults_to_a3():
    build_script = (ROOT_DIR / "csrc" / "ascend" / "build_opp.sh").read_text()

    assert "soc_version=${SOC_VERSION:-ascend910_93}" in build_script
