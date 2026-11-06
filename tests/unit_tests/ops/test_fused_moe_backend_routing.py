# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing tests for the FlagOS unquantized MoE provider."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _import_fused_moe_utils():
    from vllm_fl.ops.fused_moe import fused_moe_utils

    return fused_moe_utils


def _oot_platform(*, rocm: bool = False):
    return SimpleNamespace(
        is_cpu=lambda: False,
        is_tpu=lambda: False,
        is_rocm=lambda: rocm,
        is_cuda=lambda: False,
        is_xpu=lambda: False,
        is_out_of_tree=lambda: True,
    )


def _moe_config(backend: str):
    return SimpleNamespace(
        moe_backend=backend,
        is_lora_enabled=False,
        moe_parallel_config=SimpleNamespace(
            dp_size=1,
            use_batched_activation_format=False,
        ),
    )


class _SupportedNativeExperts:
    @staticmethod
    def is_supported_config(*_args, **_kwargs):
        return True, None


@pytest.mark.parametrize(
    "vendor,use_flaggems_vllm,execute_error",
    [
        ("nvidia", False, False),
        ("nvidia", False, True),
        ("nvidia", True, False),
        ("nvidia", True, True),
        ("kunlunxin", False, False),
        ("kunlunxin", False, True),
        ("kunlunxin", True, False),
        ("kunlunxin", True, True),
        ("thead", False, False),
        ("thead", True, False),
        ("thead", True, True),
    ],
)
def test_experts_apply_resolves_platform_before_vendor_dispatch(
    monkeypatch, vendor, use_flaggems_vllm, execute_error
):
    """Exercise the execution entry, including a vendor added after import."""
    import flag_gems
    import torch

    module = _import_fused_moe_utils()
    # Patch the environment so teardown does not shadow the lazy module attribute.
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1" if use_flaggems_vllm else "0")
    monkeypatch.setattr(
        module,
        "_get_current_platform",
        lambda: SimpleNamespace(vendor_name=vendor, is_cuda=lambda: vendor == "nvidia"),
    )
    result = torch.full((2, 4), 7.0)
    error = RuntimeError("fused experts failed") if execute_error else None
    flaggems = Mock(return_value=result, side_effect=error)
    flaggems_vllm = Mock(return_value=result, side_effect=error)
    kunlunxin = Mock(return_value=result, side_effect=error)
    monkeypatch.setattr(flag_gems, "fused_experts_impl", flaggems)
    fused_backend = types.ModuleType("flaggems_vllm")
    fused_backend.fused_experts_impl = flaggems_vllm
    monkeypatch.setitem(sys.modules, "flaggems_vllm", fused_backend)
    name = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.fused_moe"
    backend = types.ModuleType(name)
    backend.fused_experts_impl = kunlunxin
    monkeypatch.setitem(sys.modules, name, backend)
    experts = SimpleNamespace(
        _lora_context=None,
        gemm1_clamp_limit=None,
        quant_config=SimpleNamespace(
            gemm1_clamp_limit=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        ),
        per_act_token_quant=False,
        w1_scale=None,
        w2_scale=None,
        block_shape=None,
        w1_bias=None,
        w2_bias=None,
    )
    output = torch.zeros_like(result)
    if vendor == "thead" and not use_flaggems_vllm:
        # Without the dedicated library, retain the step-by-step pipeline.
        # Stop at its first shape check; no kernel execution is needed here.
        monkeypatch.setattr(experts.quant_config, "use_int4_w4a16", True)
        fallback_w1 = torch.full((2, 4, 4), 7.0)
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            module.TritonExpertsFL.apply(
                experts,
                output,
                result,
                fallback_w1,
                result,
                result,
                result,
                module.MoEActivation.SILU,
                2,
                None,
                None,
                None,
                result,
                result,
                None,
                False,
            )
        flaggems.assert_not_called()
        flaggems_vllm.assert_not_called()
        kunlunxin.assert_not_called()
        return

    def apply_experts():
        module.TritonExpertsFL.apply(
            experts,
            output,
            result,
            result,
            result,
            result,
            result,
            module.MoEActivation.SILU,
            2,
            None,
            None,
            None,
            result,
            result,
            None,
            False,
        )

    if execute_error:
        with pytest.raises(RuntimeError, match="fused experts failed"):
            apply_experts()
        torch.testing.assert_close(output, torch.zeros_like(result))
    else:
        apply_experts()
        torch.testing.assert_close(output, result)
    selected = (
        kunlunxin
        if vendor == "kunlunxin"
        else flaggems_vllm
        if use_flaggems_vllm
        else flaggems
    )
    selected.assert_called_once()
    assert selected.call_args.kwargs["activation"] == "silu"
    for other in (flaggems, flaggems_vllm, kunlunxin):
        if other is not selected:
            other.assert_not_called()


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_fl_provider_uses_flaggems_experts_for_auto_and_triton(monkeypatch, backend):
    fused_moe_utils = _import_fused_moe_utils()
    monkeypatch.setattr(fused_moe_utils, "_get_current_platform", _oot_platform)
    monkeypatch.setattr(fused_moe_utils, "use_flaggems", lambda: True)

    selected_backend, experts_cls = fused_moe_utils.select_unquantized_moe_backend_oot(
        _moe_config(backend),
        prefer_flaggems_experts=True,
    )

    assert selected_backend is fused_moe_utils.UnquantizedMoeBackend.TRITON
    assert experts_cls is fused_moe_utils.TritonExpertsFL


def test_explicit_non_triton_backend_remains_authoritative(monkeypatch):
    fused_moe_utils = _import_fused_moe_utils()
    monkeypatch.setattr(fused_moe_utils, "_get_current_platform", _oot_platform)
    monkeypatch.setattr(fused_moe_utils, "use_flaggems", lambda: True)
    monkeypatch.setattr(
        fused_moe_utils,
        "map_unquantized_backend",
        lambda _name: fused_moe_utils.UnquantizedMoeBackend.AITER,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "backend_to_kernel_cls",
        lambda _backend: _SupportedNativeExperts,
    )

    selected_backend, experts_cls = fused_moe_utils.select_unquantized_moe_backend_oot(
        _moe_config("aiter"),
        prefer_flaggems_experts=True,
    )

    assert selected_backend is fused_moe_utils.UnquantizedMoeBackend.AITER
    assert experts_cls is _SupportedNativeExperts


def test_native_oracle_does_not_reenable_flaggems_experts(monkeypatch):
    fused_moe_utils = _import_fused_moe_utils()
    monkeypatch.setattr(fused_moe_utils, "_get_current_platform", _oot_platform)
    monkeypatch.setattr(fused_moe_utils, "use_flaggems", lambda: True)
    monkeypatch.setattr(
        fused_moe_utils,
        "map_unquantized_backend",
        lambda _name: fused_moe_utils.UnquantizedMoeBackend.TRITON,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "backend_to_kernel_cls",
        lambda _backend: _SupportedNativeExperts,
    )

    selected_backend, experts_cls = fused_moe_utils.select_unquantized_moe_backend_oot(
        _moe_config("triton")
    )

    assert selected_backend is fused_moe_utils.UnquantizedMoeBackend.TRITON
    assert experts_cls is _SupportedNativeExperts


def test_factory_patch_rewrites_already_bound_model_symbol(monkeypatch):
    import vllm.model_executor.layers.fused_moe as fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    from vllm_fl.ops import custom_ops

    native_factory = object()
    fake_model = types.ModuleType("vllm.model_executor.models._fl_moe_test")
    fake_model.FusedMoE = native_factory
    monkeypatch.setitem(sys.modules, fake_model.__name__, fake_model)
    monkeypatch.setattr(fused_moe_layer, "FusedMoE", native_factory)
    monkeypatch.setattr(fused_moe_pkg, "FusedMoE", native_factory)

    custom_ops._patch_fused_moe_factory()

    assert fake_model.FusedMoE is custom_ops.FusedMoEFL
    assert fused_moe_layer.FusedMoE is custom_ops.FusedMoEFL
    assert fused_moe_pkg.FusedMoE is custom_ops.FusedMoEFL


@pytest.mark.parametrize(
    ("whitelist", "blacklist", "expect_fl_factory"),
    [
        (None, [], True),
        (["fused_moe"], [], True),
        (["rms_norm"], [], False),
        (None, ["fused_moe"], False),
    ],
)
def test_fused_moe_factory_obeys_oot_policy(
    monkeypatch, whitelist, blacklist, expect_fl_factory
):
    try:
        from vllm_fl import utils
        from vllm_fl.ops import custom_ops
    except (ImportError, ModuleNotFoundError) as exc:
        return pytest.skip(f"vLLM OOT dependencies unavailable: {exc}")

    patch_factory = Mock()
    patch_oracle = Mock()
    monkeypatch.setattr(custom_ops, "_patch_fused_moe_factory", patch_factory)
    monkeypatch.setattr(custom_ops, "_patch_unquantized_moe_oracle", patch_oracle)
    monkeypatch.setattr(custom_ops, "OOT_OPS", {})
    monkeypatch.setattr(utils, "is_oot_enabled", lambda: True)
    monkeypatch.setattr(utils, "get_oot_whitelist", lambda: whitelist)
    monkeypatch.setattr(utils, "get_oot_blacklist", lambda: blacklist)

    custom_ops.register_oot_ops()

    if expect_fl_factory:
        patch_factory.assert_called_once_with()
        patch_oracle.assert_called_once_with(prefer_flaggems_experts=True)
    else:
        patch_factory.assert_not_called()
        patch_oracle.assert_called_once_with(prefer_flaggems_experts=False)
    return None
