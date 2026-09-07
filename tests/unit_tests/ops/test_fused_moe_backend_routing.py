"""Routing tests for the FlagOS unquantized MoE provider."""

from types import SimpleNamespace
import sys
import types
from unittest.mock import Mock

import pytest


def _import_fused_moe_utils():
    try:
        from vllm_fl.ops.fused_moe import fused_moe_utils
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"vLLM MoE dependencies unavailable: {exc}")
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


@pytest.mark.parametrize("backend", ["auto", "triton"])
def test_fl_provider_uses_flaggems_experts_for_auto_and_triton(
    monkeypatch, backend
):
    fused_moe_utils = _import_fused_moe_utils()
    monkeypatch.setattr(
        fused_moe_utils, "_get_current_platform", lambda: _oot_platform()
    )
    monkeypatch.setattr(fused_moe_utils, "use_flaggems", lambda: True)

    selected_backend, experts_cls = (
        fused_moe_utils.select_unquantized_moe_backend_oot(
            _moe_config(backend),
            prefer_flaggems_experts=True,
        )
    )

    assert selected_backend is fused_moe_utils.UnquantizedMoeBackend.TRITON
    assert experts_cls is fused_moe_utils.TritonExpertsFL


def test_explicit_non_triton_backend_remains_authoritative(monkeypatch):
    fused_moe_utils = _import_fused_moe_utils()
    monkeypatch.setattr(
        fused_moe_utils, "_get_current_platform", lambda: _oot_platform()
    )
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

    selected_backend, experts_cls = (
        fused_moe_utils.select_unquantized_moe_backend_oot(
            _moe_config("aiter"),
            prefer_flaggems_experts=True,
        )
    )

    assert selected_backend is fused_moe_utils.UnquantizedMoeBackend.AITER
    assert experts_cls is _SupportedNativeExperts


def test_native_oracle_does_not_reenable_flaggems_experts(monkeypatch):
    fused_moe_utils = _import_fused_moe_utils()
    monkeypatch.setattr(
        fused_moe_utils, "_get_current_platform", lambda: _oot_platform()
    )
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

    selected_backend, experts_cls = (
        fused_moe_utils.select_unquantized_moe_backend_oot(
            _moe_config("triton")
        )
    )

    assert selected_backend is fused_moe_utils.UnquantizedMoeBackend.TRITON
    assert experts_cls is _SupportedNativeExperts


def test_factory_patch_rewrites_already_bound_model_symbol(monkeypatch):
    from vllm_fl.ops import custom_ops
    import vllm.model_executor.layers.fused_moe as fused_moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

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
        pytest.skip(f"vLLM OOT dependencies unavailable: {exc}")

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
