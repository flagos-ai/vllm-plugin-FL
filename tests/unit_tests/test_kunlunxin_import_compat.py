# Copyright (c) 2026 BAAI. All rights reserved.

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import vllm_fl

_VENDOR_ENV = ("VLLM_FL_PLATFORM", "GEMS_VENDOR", "VLLM_VENDOR")


def _clear_vendor_env(monkeypatch):
    for name in _VENDOR_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("VLLM_FL_PLATFORM", "ascend"),
        ("GEMS_VENDOR", "metax"),
        ("VLLM_VENDOR", "cuda"),
    ],
)
def test_explicit_non_kunlunxin_vendor_skips_import_compat(
    monkeypatch, variable, value
):
    _clear_vendor_env(monkeypatch)
    monkeypatch.setenv(variable, value)

    with patch("vllm_fl.importlib.import_module") as import_module:
        assert vllm_fl._init_early_vendor_compat() is False

    import_module.assert_not_called()


def test_platform_override_has_priority_over_other_vendor_variables(monkeypatch):
    _clear_vendor_env(monkeypatch)
    monkeypatch.setenv("VLLM_FL_PLATFORM", "ascend")
    monkeypatch.setenv("GEMS_VENDOR", "kunlunxin")
    monkeypatch.setenv("VLLM_VENDOR", "kunlunxin")

    with patch("vllm_fl.importlib.import_module") as import_module:
        assert vllm_fl._init_early_vendor_compat() is False

    import_module.assert_not_called()


@pytest.mark.parametrize(
    "environment",
    [
        {"VLLM_FL_PLATFORM": " Kunlunxin "},
        {"GEMS_VENDOR": "KUNLUNXIN"},
        {"VLLM_VENDOR": "kunlunxin"},
    ],
)
def test_explicit_kunlunxin_vendor_applies_import_compat(monkeypatch, environment):
    _clear_vendor_env(monkeypatch)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    compat = SimpleNamespace(apply_import_compat=Mock())
    with patch(
        "vllm_fl.importlib.import_module",
        return_value=compat,
    ) as import_module:
        assert vllm_fl._init_early_vendor_compat() is True

    import_module.assert_called_once_with("vllm_fl.patches.kunlunxin.import_compat")
    compat.apply_import_compat.assert_called_once_with()


def test_unspecified_vendor_auto_detects_kunlunxin_runtime(monkeypatch):
    _clear_vendor_env(monkeypatch)
    compat = SimpleNamespace(apply_import_compat=Mock())

    with (
        patch("vllm_fl.importlib.util.find_spec", return_value=object()),
        patch(
            "vllm_fl.importlib.import_module",
            return_value=compat,
        ) as import_module,
    ):
        assert vllm_fl._init_early_vendor_compat() is True

    import_module.assert_called_once_with("vllm_fl.patches.kunlunxin.import_compat")
    compat.apply_import_compat.assert_called_once_with()


def test_unspecified_non_kunlunxin_runtime_skips_import_compat(monkeypatch):
    _clear_vendor_env(monkeypatch)

    with (
        patch("vllm_fl.importlib.util.find_spec", return_value=None),
        patch("vllm_fl.importlib.import_module") as import_module,
    ):
        assert vllm_fl._init_early_vendor_compat() is False

    import_module.assert_not_called()


@pytest.mark.parametrize("use_flaggems", ["0", "1"])
def test_ascend_package_import_never_touches_triton(use_flaggems):
    script = r"""
import importlib.abc
import sys
import types

class RejectTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise AssertionError(f"unexpected Triton import: {fullname}")
        return None

sys.meta_path.insert(0, RejectTriton())

version = types.ModuleType("vllm_fl.version")
utils = types.ModuleType("vllm_fl.utils")
utils.get_op_config = lambda: None
sys.modules[version.__name__] = version
sys.modules[utils.__name__] = utils

import vllm_fl

assert vllm_fl._get_explicit_runtime_vendor() == "ascend"
"""
    environment = os.environ.copy()
    environment.update(
        {
            "GEMS_VENDOR": "kunlunxin",
            "USE_FLAGGEMS": use_flaggems,
            "VLLM_FL_PLATFORM": "ascend",
            "VLLM_VENDOR": "cuda",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
