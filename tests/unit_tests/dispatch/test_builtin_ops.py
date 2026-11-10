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
        available = {"cuda", "ascend", "metax", "iluvatar"}
        assert _find_vendor_backend_dir("nvidia", available) == "cuda"
        assert _find_vendor_backend_dir("ascend", available) == "ascend"
        assert _find_vendor_backend_dir("metax", available) == "metax"
        assert _find_vendor_backend_dir("iluvatar", available) == "iluvatar"

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
        "attention_backend",
        "invoke_fused_moe_triton_kernel",
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
