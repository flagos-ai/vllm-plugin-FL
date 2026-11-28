# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Unit tests for the vl_compat shim (bilinear pos-embed native fallback)."""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock

import pytest
import torch

_MOD_NAME = "vllm.model_executor.models.qwen3_vl"
_PARENTS = [
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
]


def _setup_fake_qwen3_vl(
    monkeypatch, triton_present: bool, native_present: bool = True
):
    """Install a fresh fake qwen3_vl and all parent stubs into sys.modules."""
    for p in _PARENTS:
        if p not in sys.modules:
            monkeypatch.setitem(sys.modules, p, types.ModuleType(p))

    mod = types.ModuleType(_MOD_NAME)

    if native_present:

        def pos_embed_interpolate_native(*args, **kwargs):
            return "native"

        mod.pos_embed_interpolate_native = pos_embed_interpolate_native

    if triton_present:

        def triton_pos_embed_interpolate(*args, **kwargs):
            return "triton"

        mod.triton_pos_embed_interpolate = triton_pos_embed_interpolate

    monkeypatch.setitem(sys.modules, _MOD_NAME, mod)
    monkeypatch.setattr(sys.modules[_PARENTS[-1]], "qwen3_vl", mod, raising=False)
    return mod


@pytest.fixture(autouse=True)
def _fresh_patch_flag(monkeypatch):
    """Isolate fake modules and restore any model imported during collection."""
    import vllm_fl.dispatch.backends.vendor.kunlunxin.patches.vl_compat  # noqa: F401

    monkeypatch.delitem(sys.modules, _MOD_NAME, raising=False)
    parent = sys.modules.get(_PARENTS[-1])
    if parent is not None:
        monkeypatch.delattr(parent, "qwen3_vl", raising=False)


def _get_vc():
    import importlib

    import vllm_fl.dispatch.backends.vendor.kunlunxin.patches.vl_compat as vc

    # Force a fresh module-level state by removing the patch flag from module
    # and from any previously patched qwen3_vl module
    sys.modules.pop(_MOD_NAME, None)
    importlib.reload(vc)
    return vc


def test_triton_replaced_with_native(monkeypatch):
    """When Triton is present, triton_pos_embed_interpolate → native."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(monkeypatch, triton_present=True)
    vc.apply_vl_compat()
    assert mod.triton_pos_embed_interpolate("x") == "native"
    assert getattr(mod, vc._PATCH_FLAG, False)


def test_no_triton_no_error(monkeypatch):
    """When Triton is absent the patch marks done without raising."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(monkeypatch, triton_present=False)
    vc.apply_vl_compat()
    assert getattr(mod, vc._PATCH_FLAG, False)


def test_idempotent(monkeypatch):
    """Calling apply_vl_compat twice does not raise or undo the patch."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(monkeypatch, triton_present=True)
    vc.apply_vl_compat()
    vc.apply_vl_compat()
    assert mod.triton_pos_embed_interpolate("x") == "native"


def test_missing_native_fn_no_crash(monkeypatch):
    """A missing native fallback emits a warning without installing the patch."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(monkeypatch, triton_present=True, native_present=False)
    original = mod.triton_pos_embed_interpolate
    warning = Mock()
    monkeypatch.setattr(vc.logger, "warning", warning)

    vc.apply_vl_compat()

    warning.assert_called_once()
    assert "pos_embed_interpolate_native not found" in warning.call_args.args[0]
    assert mod.triton_pos_embed_interpolate is original
    assert not hasattr(mod, vc._PATCH_FLAG)


def test_module_not_installed_no_crash(monkeypatch):
    """If qwen3_vl is not installed, apply_vl_compat returns silently."""
    vc = _get_vc()
    # Set to None so the import raises ImportError (Python's behavior with None in sys.modules)
    monkeypatch.setitem(sys.modules, _MOD_NAME, None)
    vc.apply_vl_compat()  # Must not raise


def test_ordered_vision_reduce_stays_in_weight_sharded_scope(monkeypatch):
    vc = _get_vc()
    patches = types.ModuleType("vllm.model_executor.models.qwen3_5")
    tensor = torch.ones((2, 3), dtype=torch.bfloat16)
    communicator = types.SimpleNamespace(
        world_size=4,
        unique_name="tp:0",
        pyflagcx_comm=types.SimpleNamespace(disabled=False),
        all_gather=Mock(
            return_value=torch.cat(
                [torch.full_like(tensor, value) for value in (256, 1, -256, 1)]
            )
        ),
    )

    class QwenModel:
        use_data_parallel = False

        def _process_image_input(self, image_input):
            if image_input is None:
                raise RuntimeError("encoder failure")
            return vc.maybe_reduce_vision_tp(communicator, image_input)

    patches.Qwen3_5MoeForConditionalGeneration = QwenModel
    monkeypatch.setitem(sys.modules, patches.__name__, patches)

    vc.apply_vision_tp_scope()
    installed = QwenModel._process_image_input
    vc.apply_vision_tp_scope()
    assert QwenModel._process_image_input is installed
    assert vc.maybe_reduce_vision_tp(communicator, tensor) is None

    model = QwenModel()
    torch.testing.assert_close(model._process_image_input(tensor), tensor * 2)
    communicator.all_gather.assert_called_once_with(tensor, dim=0)

    communicator.unique_name = "dp:0"
    assert model._process_image_input(tensor) is None
    communicator.unique_name = "tp:0"
    with pytest.raises(RuntimeError, match="encoder failure"):
        model._process_image_input(None)
    assert vc.maybe_reduce_vision_tp(communicator, tensor) is None

    model.use_data_parallel = True
    assert model._process_image_input(tensor) is None
    communicator.all_gather.assert_called_once()
