# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for dispatch builtin operator registration."""

from unittest.mock import MagicMock, patch

from vllm_fl.dispatch.builtin_ops import (
    _find_vendor_backend_dir,
    _register_vendor_backends,
)
from vllm_fl.dispatch.registry import OpRegistry


class TestFindVendorBackendDir:
    def test_maps_vendor_names(self):
        available = {"cuda", "ascend", "metax", "iluvatar", "supa"}
        assert _find_vendor_backend_dir("nvidia", available) == "cuda"
        assert _find_vendor_backend_dir("ascend", available) == "ascend"
        assert _find_vendor_backend_dir("metax", available) == "metax"
        assert _find_vendor_backend_dir("iluvatar", available) == "iluvatar"
        assert _find_vendor_backend_dir("biren", available) == "supa"

    def test_maca_alias_resolves_to_metax(self):
        assert _find_vendor_backend_dir("maca", {"metax"}) == "metax"

    def test_unknown_vendor_returns_none(self):
        assert _find_vendor_backend_dir("unknown", {"cuda", "ascend"}) is None


class TestRegisterVendorBackends:
    def test_registers_only_current_vendor_backend(self):
        registry = MagicMock()
        module = MagicMock()

        with (
            patch("vllm_fl.dispatch.builtin_ops._VENDOR_BACKENDS_DIR", "/fake/vendor"),
            patch(
                "vllm_fl.dispatch.builtin_ops._get_current_vendor_backend_dirs",
                return_value="metax",
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isdir", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.os.listdir",
                return_value=["metax", "cuda", "ascend"],
            ),
            patch("vllm_fl.dispatch.builtin_ops.os.path.isfile", return_value=True),
            patch(
                "vllm_fl.dispatch.builtin_ops.importlib.import_module",
                return_value=module,
            ) as import_module,
        ):
            _register_vendor_backends(registry)

        import_module.assert_called_once_with(
            ".backends.vendor.metax.register_ops",
            package="vllm_fl.dispatch",
        )
        module.register_builtins.assert_called_once_with(registry)


def test_reference_registration_keeps_implemented_fallbacks():
    """A missing optional method must not drop every reference fallback."""
    from vllm_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )

    registry = OpRegistry()
    register_reference(registry)

    implemented = {
        "dynamic_per_token_quant_int8",
        "silu_and_mul",
        "gelu_and_mul",
        "rms_norm",
        "rotary_embedding",
        "apply_rotary_emb",
        "attention_backend",
        "invoke_fused_moe_triton_kernel",
        "topk_sigmoid",
    }
    missing = {
        "moe_align_block_size",
        "moe_sum",
        "topk_softmax",
        "grouped_topk",
    }
    for op_name in implemented:
        assert registry.get_implementation(op_name, "reference.torch") is not None
    for op_name in missing:
        assert registry.get_implementation(op_name, "reference.torch") is None


def test_topk_sigmoid_optional_registration_and_policy(monkeypatch):
    import sys
    from types import ModuleType

    from vllm.platforms import current_platform

    from vllm_fl.dispatch.backends.flaggems import register_ops
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    # Registration imports unrelated vLLM kernels; keep this policy test CPU-only.
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda *a, **k: False
    )
    package = ModuleType("flaggems_vllm")
    monkeypatch.setitem(sys.modules, "flaggems_vllm", package)
    monkeypatch.setattr(FlagGemsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(
        register_ops, "use_flaggems_op", lambda name: name == "topk_sigmoid"
    )
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    registry = OpRegistry()
    register_ops.register_builtins(registry)
    assert not registry.get_implementation(
        "topk_sigmoid", "default.flagos"
    ).is_available()

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    registry = OpRegistry()
    register_ops.register_builtins(registry)
    impl = registry.get_implementation("topk_sigmoid", "default.flagos")
    assert not impl.is_available()  # An older library may lack this optional export.
    package.topk_sigmoid = lambda *args: None
    assert impl.is_available()
    monkeypatch.setattr(register_ops, "use_flaggems_op", lambda name: False)
    registry = OpRegistry()
    register_ops.register_builtins(registry)
    assert registry.get_implementation("topk_sigmoid", "default.flagos") is None


def test_topk_sigmoid_native_fallback_preserves_arguments(monkeypatch):
    import torch

    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router

    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend

    native = MagicMock(return_value=("weights", "ids"))
    monkeypatch.setattr(fused_topk_bias_router, "fused_topk_bias", native)
    inputs = dict(
        hidden_states=torch.zeros(2, 64),
        gating_output=torch.zeros(2, 128),
        topk=4,
        renormalize=True,
        e_score_correction_bias=torch.zeros(128),
        indices_type=torch.int64,
    )
    for backend in (FlagGemsBackend(), ReferenceBackend()):
        assert backend.topk_sigmoid(**inputs) == ("weights", "ids")
        assert native.call_args.kwargs == dict(inputs, scoring_func="sigmoid")


def test_topk_sigmoid_library_output_buffer_contract(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytest
    import torch

    from vllm_fl.dispatch.backends.flaggems import flaggems as backend_module

    # Metadata-only CUDA inputs prove the adapter ABI without allocating a device.
    logits = SimpleNamespace(
        ndim=2,
        shape=(3, 128),
        dtype=torch.float32,
        device=torch.device("cuda"),
        is_contiguous=lambda: True,
        size=lambda dim: (3, 128)[dim],
    )
    bias = SimpleNamespace(
        shape=(128,),
        dtype=torch.float32,
        device=logits.device,
        is_contiguous=lambda: True,
    )
    allocate = torch.empty
    monkeypatch.setattr(
        backend_module.torch,
        "empty",
        lambda shape, *, device, dtype: allocate(shape, dtype=dtype),
    )
    kernel = MagicMock()
    monkeypatch.setitem(
        sys.modules, "flaggems_vllm", SimpleNamespace(topk_sigmoid=kernel)
    )
    backend = backend_module.FlagGemsBackend()
    weights, ids = backend.topk_sigmoid(
        torch.zeros(3, 64), logits, 4, True, bias, torch.int64
    )
    args = kernel.call_args.args
    assert args[0] is weights and args[1] is ids
    assert weights.shape == ids.shape == args[2].shape == (3, 4)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int64
    assert args[2].dtype == torch.int32
    assert args[3] is logits and args[4] is True and args[5] is bias
    kernel.side_effect = RuntimeError("selected kernel failed")
    with pytest.raises(RuntimeError, match="selected kernel failed"):
        backend.topk_sigmoid(torch.zeros(3, 64), logits, 4, False, bias)
