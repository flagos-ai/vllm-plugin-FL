# Copyright (c) 2026 BAAI. All rights reserved.

"""Check empty-wheel import handling without loading the CUDA runtime."""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_SOURCE = Path(__file__).parents[2] / "vllm_fl" / "nvidia_compat.py"
_SPEC = importlib.util.spec_from_file_location("_nvidia_compat_test", _SOURCE)
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)


def test_empty_import_placeholder_does_not_claim_native_kernels():
    modules = {"vllm": ModuleType("vllm")}
    with (
        patch.dict(sys.modules, modules, clear=True),
        patch.object(compat.importlib.util, "find_spec", return_value=None),
    ):
        compat.prepare_nvidia_empty_imports()
        stub = sys.modules["vllm._C_stable_libtorch"]
        assert getattr(stub, compat._EMPTY_MARKER)
        assert not compat.has_native_vllm_kernels()
        compat.prepare_nvidia_empty_imports()
        assert sys.modules["vllm._C_stable_libtorch"] is stub


def test_native_extension_load_error_is_not_hidden_as_empty():
    name = "vllm._C_stable_libtorch"
    spec = importlib.machinery.ModuleSpec(
        name,
        loader=None,
        origin="/site-packages/vllm/_C_stable_libtorch"
        + importlib.machinery.EXTENSION_SUFFIXES[0],
    )
    with (
        patch.dict(sys.modules, {}, clear=True),
        patch.object(
            compat.importlib.util,
            "find_spec",
            side_effect=lambda module: spec if module == name else None,
        ),
        patch.object(
            compat.importlib, "import_module", side_effect=OSError("incompatible ABI")
        ),
        pytest.raises(OSError, match="incompatible ABI"),
    ):
        compat.prepare_nvidia_empty_imports()


def test_python_module_is_not_a_native_extension():
    spec = importlib.machinery.ModuleSpec(
        "vllm._C", loader=None, origin="/site-packages/vllm/_C.py"
    )
    with (
        patch.dict(sys.modules, {}, clear=True),
        patch.object(compat.importlib.util, "find_spec", return_value=spec),
    ):
        assert not compat.has_native_vllm_kernels()
