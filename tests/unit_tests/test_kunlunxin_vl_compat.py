# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Unit tests for the vl_compat shim (bilinear pos-embed native fallback)."""

from __future__ import annotations

import logging
import sys
import types

import pytest

_MOD_NAME = "vllm.model_executor.models.qwen3_vl"
_PARENTS = [
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
]


def _setup_fake_qwen3_vl(triton_present: bool, native_present: bool = True):
    """Install a fresh fake qwen3_vl and all parent stubs into sys.modules."""
    for p in _PARENTS:
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)

    mod = types.ModuleType(_MOD_NAME)

    if native_present:
        def pos_embed_interpolate_native(*args, **kwargs):
            return "native"
        mod.pos_embed_interpolate_native = pos_embed_interpolate_native

    if triton_present:
        def triton_pos_embed_interpolate(*args, **kwargs):
            return "triton"
        mod.triton_pos_embed_interpolate = triton_pos_embed_interpolate

    sys.modules[_MOD_NAME] = mod
    return mod


@pytest.fixture(autouse=True)
def _fresh_patch_flag():
    """Remove the patch flag from vl_compat before each test."""
    import vllm_fl.dispatch.backends.vendor.kunlunxin.patches.vl_compat as vc
    # Remove flag so each test starts with an un-patched state
    for attr in list(vars(vc).keys()):
        if attr.startswith("_vllm_fl_kunlunxin_vl"):
            pass  # We'll reset the qwen3_vl module itself
    yield
    # Clean up the installed fake module
    sys.modules.pop(_MOD_NAME, None)


def _get_vc():
    import importlib

    import vllm_fl.dispatch.backends.vendor.kunlunxin.patches.vl_compat as vc
    # Force a fresh module-level state by removing the patch flag from module
    # and from any previously patched qwen3_vl module
    sys.modules.pop(_MOD_NAME, None)
    importlib.reload(vc)
    return vc


def test_triton_replaced_with_native():
    """When Triton is present, triton_pos_embed_interpolate → native."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(triton_present=True)
    vc.apply_vl_compat()
    assert mod.triton_pos_embed_interpolate("x") == "native"
    assert getattr(mod, vc._PATCH_FLAG, False)


def test_no_triton_no_error():
    """When Triton is absent the patch marks done without raising."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(triton_present=False)
    vc.apply_vl_compat()
    assert getattr(mod, vc._PATCH_FLAG, False)


def test_idempotent():
    """Calling apply_vl_compat twice does not raise or undo the patch."""
    vc = _get_vc()
    mod = _setup_fake_qwen3_vl(triton_present=True)
    vc.apply_vl_compat()
    vc.apply_vl_compat()
    assert mod.triton_pos_embed_interpolate("x") == "native"


def test_missing_native_fn_no_crash(caplog):
    """If pos_embed_interpolate_native is absent, logs a warning."""
    vc = _get_vc()
    _setup_fake_qwen3_vl(triton_present=True, native_present=False)
    # Ensure the vc logger propagates to root so caplog can catch it
    vc_logger = logging.getLogger(vc.__name__)
    vc_logger.propagate = True
    with caplog.at_level(logging.WARNING, logger=vc.__name__):
        vc.apply_vl_compat()
    assert any("pos_embed_interpolate_native not found" in m for m in caplog.messages)


def test_module_not_installed_no_crash():
    """If qwen3_vl is not installed, apply_vl_compat returns silently."""
    vc = _get_vc()
    # Set to None so the import raises ImportError (Python's behavior with None in sys.modules)
    sys.modules[_MOD_NAME] = None  # type: ignore[assignment]
    vc.apply_vl_compat()  # Must not raise
