# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from vllm_fl.patches.kunlunxin import import_compat


def _fake_triton_modules(monkeypatch):
    triton = ModuleType("triton")
    triton.__path__ = []
    language = ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.delitem(sys.modules, "triton.knobs", raising=False)
    return triton, language


def test_missing_kunlunxin_triton_apis_receive_import_only_sentinels(monkeypatch):
    triton, language = _fake_triton_modules(monkeypatch)
    missing_knobs = ModuleNotFoundError(
        "No module named 'triton.knobs'", name="triton.knobs"
    )

    with patch.object(
        import_compat.importlib,
        "import_module",
        side_effect=missing_knobs,
    ):
        import_compat._patch_flag_gems_triton_import_compat()

    assert language.map_elementwise.__triton_builtin__ is True
    with pytest.raises(NotImplementedError, match="must remain blacklisted"):
        language.map_elementwise(None)
    assert triton.knobs.autotuning.adjust_block_size is True
    assert sys.modules["triton.knobs"] is triton.knobs


def test_triton_knobs_internal_import_error_is_not_masked(monkeypatch):
    _fake_triton_modules(monkeypatch)
    incompatible_runtime = ImportError(
        "cannot import name 'getenv' from 'triton._C.libtriton'"
    )

    with (
        patch.object(
            import_compat.importlib,
            "import_module",
            side_effect=incompatible_runtime,
        ),
        pytest.raises(ImportError, match="getenv"),
    ):
        import_compat._patch_flag_gems_triton_import_compat()


def test_missing_torch_float4_dtype_is_patched_only_in_vendor_module(monkeypatch):
    torch = ModuleType("torch")
    torch.uint8 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

    import_compat._patch_torch_float4_import_compat()

    assert torch.float4_e2m1fn_x2 is torch.uint8
