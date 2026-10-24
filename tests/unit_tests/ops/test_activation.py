# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for activation ops.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.ops.fused_moe.activation import apply_moe_activation


class TestSiluAndMulFL:
    """Test SiluAndMulFL class behavior."""

    @pytest.fixture
    def mock_cached_op(self):
        with patch("vllm_fl.ops.activation._silu_and_mul") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch("vllm_fl.ops.activation.SiluAndMul.__init__", return_value=None):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_cached_op):
        """Test forward_oot calls dispatch system with correct op name and input."""
        from vllm_fl.ops.activation import SiluAndMulFL

        mock_cached_op.return_value = torch.randn(2, 4)
        layer = SiluAndMulFL()
        x = torch.randn(2, 8)

        result = layer.forward_oot(x)

        mock_cached_op.assert_called_once_with(layer, x)
        assert result.shape == (2, 4)


def test_fused_moe_activation_tolerates_missing_optional_enum(monkeypatch):
    import vllm_fl.ops.fused_moe.activation as activation_module

    enum_without_uninterleave = SimpleNamespace(
        SILU=object(),
        GELU=object(),
        SWIGLUOAI=object(),
        SWIGLUSTEP=object(),
        SILU_NO_MUL=object(),
        GELU_NO_MUL=object(),
        RELU2_NO_MUL=object(),
    )
    unknown_activation = SimpleNamespace(is_gated=False, value="unknown")
    input_tensor = torch.randn(2, 4)
    output_tensor = torch.empty_like(input_tensor)
    monkeypatch.setattr(activation_module, "MoEActivation", enum_without_uninterleave)

    with pytest.raises(ValueError, match="Unsupported FusedMoe activation"):
        activation_module.apply_moe_activation(
            unknown_activation,
            output_tensor,
            input_tensor,
        )


def test_fused_moe_clamped_silu_uses_upstream_math(monkeypatch):
    import vllm_fl.ops.fused_moe.activation as activation_module

    input_tensor = torch.randn(2, 4)
    output_tensor = torch.empty(2, 2)
    with (
        patch.object(
            activation_module,
            "upstream_apply_moe_activation",
            return_value=output_tensor,
        ) as upstream,
        patch.object(activation_module, "_silu_and_mul") as ordinary_silu,
    ):
        result = activation_module.apply_moe_activation(
            activation_module.MoEActivation.SILU,
            output_tensor,
            input_tensor,
            clamp_limit=7.0,
        )

    assert result is output_tensor
    upstream.assert_called_once_with(
        activation_module.MoEActivation.SILU,
        output_tensor,
        input_tensor,
        clamp_limit=7.0,
    )
    ordinary_silu.assert_not_called()


_SPLIT_ACTIVATION = MoEActivation.SWIGLUOAI_UNINTERLEAVE


_SPLIT_LIB_MODULE = "flaggems_vllm.ops.fused_moe"


_SPLIT_NATIVE_MODULE = "vllm.models.minimax_m3.amd.ops.swiglu_oai"


_SPLIT_PARAMS = {"clamp_limit": 7.0, "alpha": 1.702, "beta": 1.0}


def _split_tensors():
    return (torch.zeros(2, 4), torch.empty(2, 2))


class TestSplitSwiGluOAI:
    @pytest.fixture(autouse=True)
    def clean_policy(self, monkeypatch):
        for name in (
            "VLLM_FL_USE_FLAGGEMS_VLLM",
            "VLLM_FL_PREFER_ENABLED",
            "VLLM_FL_PREFER",
            "USE_FLAGGEMS",
            "VLLM_FL_FLAGOS_WHITELIST",
            "VLLM_FL_FLAGOS_BLACKLIST",
            "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr("vllm_fl.dispatch.config.get_flagos_blacklist", lambda: [])

    @pytest.mark.parametrize(
        "params", [_SPLIT_PARAMS, {"clamp_limit": 3.5, "alpha": 0.7, "beta": -2.0}]
    )
    def test_direct_fv_api_preserves_output_and_gate_parameters(
        self, monkeypatch, params
    ):
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        kind = object()
        operator = Mock(side_effect=lambda activation, out, x, **kw: out.fill_(3))
        monkeypatch.setitem(
            sys.modules,
            _SPLIT_LIB_MODULE,
            SimpleNamespace(
                MoEActivation=SimpleNamespace(SWIGLUOAI_UNINTERLEAVE=kind),
                apply_moe_activation=operator,
            ),
        )
        monkeypatch.setitem(sys.modules, _SPLIT_NATIVE_MODULE, None)
        x, out = _split_tensors()
        assert apply_moe_activation(_SPLIT_ACTIVATION, out, x, **params) is out
        args, kwargs = operator.call_args
        assert args[0] is kind and args[1] is out and (args[2] is x)
        assert kwargs == {
            "gemm1_alpha": params["alpha"],
            "gemm1_beta": params["beta"],
            "gemm1_clamp_limit": params["clamp_limit"],
        }
        assert torch.equal(out, torch.full_like(out, 3))

    @pytest.mark.parametrize(
        "settings",
        [
            {},
            {"VLLM_FL_USE_FLAGGEMS_VLLM": "0"},
            {"VLLM_FL_USE_FLAGGEMS_VLLM": "1", "USE_FLAGGEMS": "0"},
            {"VLLM_FL_USE_FLAGGEMS_VLLM": "1", "VLLM_FL_PREFER_ENABLED": "false"},
            {"VLLM_FL_USE_FLAGGEMS_VLLM": "1", "VLLM_FL_PREFER": "reference"},
            {
                "VLLM_FL_USE_FLAGGEMS_VLLM": "1",
                "VLLM_FL_FLAGOS_WHITELIST": "silu_and_mul",
            },
            {
                "VLLM_FL_USE_FLAGGEMS_VLLM": "1",
                "VLLM_FL_FLAGOS_BLACKLIST": "swigluoai_uninterleave",
            },
            {
                "VLLM_FL_USE_FLAGGEMS_VLLM": "1",
                "VLLM_FL_FLAGOS_BLACKLIST_APPEND": "swigluoai_uninterleave",
            },
        ],
    )
    def test_original_native_path_needs_no_optional_fv_import(
        self, monkeypatch, settings
    ):
        for name, value in settings.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setitem(sys.modules, _SPLIT_LIB_MODULE, None)
        expected = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        native = Mock(return_value=expected)
        monkeypatch.setitem(
            sys.modules, _SPLIT_NATIVE_MODULE, SimpleNamespace(swiglu_oai_split=native)
        )
        x, out = _split_tensors()
        assert apply_moe_activation(_SPLIT_ACTIVATION, out, x, **_SPLIT_PARAMS) is out
        args = native.call_args.args
        assert args[0] is x and args[1:] == (1.702, 1.0, 7.0)
        assert torch.equal(out, expected)

    def test_default_gate_parameters_are_preserved(self, monkeypatch):
        native = Mock(return_value=torch.ones(2, 2))
        monkeypatch.setitem(
            sys.modules, _SPLIT_NATIVE_MODULE, SimpleNamespace(swiglu_oai_split=native)
        )
        x, out = _split_tensors()
        apply_moe_activation(_SPLIT_ACTIVATION, out, x, clamp_limit=6.0)
        assert native.call_args.args[1:] == (1.0, 0.0, 6.0)

    def test_explicit_whitelist_still_selects_fv(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "swigluoai_uninterleave")
        operator = Mock()
        monkeypatch.setitem(
            sys.modules,
            _SPLIT_LIB_MODULE,
            SimpleNamespace(MoEActivation=MoEActivation, apply_moe_activation=operator),
        )
        x, out = _split_tensors()
        apply_moe_activation(_SPLIT_ACTIVATION, out, x, **_SPLIT_PARAMS)
        operator.assert_called_once()

    def test_missing_clamp_is_rejected_before_optional_import(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        monkeypatch.setitem(sys.modules, _SPLIT_LIB_MODULE, None)
        x, out = _split_tensors()
        with pytest.raises(ValueError, match="requires clamp_limit"):
            apply_moe_activation(_SPLIT_ACTIVATION, out, x)

    def test_explicitly_selected_fv_failure_is_visible(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        operator = Mock(side_effect=RuntimeError("selected kernel failed"))
        monkeypatch.setitem(
            sys.modules,
            _SPLIT_LIB_MODULE,
            SimpleNamespace(MoEActivation=MoEActivation, apply_moe_activation=operator),
        )
        x, out = _split_tensors()
        with pytest.raises(RuntimeError, match="selected kernel failed"):
            apply_moe_activation(_SPLIT_ACTIVATION, out, x, **_SPLIT_PARAMS)

    def test_selected_missing_dependency_is_visible(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        monkeypatch.setitem(sys.modules, _SPLIT_LIB_MODULE, None)
        x, out = _split_tensors()
        with pytest.raises(ModuleNotFoundError):
            apply_moe_activation(_SPLIT_ACTIVATION, out, x, **_SPLIT_PARAMS)


def _packed_swiglu_reference(input, limit, alpha, beta):
    gate, up = input.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(input.dtype)


def _invoke_packed_swiglu(input, limit=7.0, alpha=1.702, beta=1.0):
    output = torch.empty(
        input.shape[0], input.shape[1] // 2, device=input.device, dtype=input.dtype
    )
    return apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output,
        input,
        clamp_limit=limit,
        alpha=alpha,
        beta=beta,
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)
class TestTheadPackedMoEActivation:
    def test_packed_activation_matches_reference(self):
        torch.manual_seed(31)
        input = (
            torch.randn(7, 6144, device="cuda", dtype=torch.bfloat16) * 9
        ).contiguous()
        actual = _invoke_packed_swiglu(input)
        expected = _packed_swiglu_reference(input, 7.0, 1.702, 1.0)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)

    @pytest.mark.parametrize(
        "width,quantize", [(6144, False), (768, True), (766, True)]
    )
    def test_minimax_packed_activation_cuda_graph_changed_input(self, width, quantize):
        from flaggems_vllm.ops import fused_moe as generic_moe

        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

        from vllm_fl.ops.fused_moe.activation import (
            _apply_swigluoai_with_optional_quantization,
        )

        if not hasattr(torch.cuda, "CUDAGraph"):
            pytest.skip("CUDA-compatible graph capture is unavailable")
        torch.manual_seed(32)
        input = torch.randn(5, width, device="cuda", dtype=torch.bfloat16).contiguous()
        output = torch.empty(5, width // 2, device="cuda", dtype=torch.bfloat16)
        reference = torch.empty_like(output)

        def invoke():
            return _apply_swigluoai_with_optional_quantization(
                output, input, quantize=quantize, **_SPLIT_PARAMS
            )

        # The public library may omit the optional fusion, or reject this
        # shape. Freeze its actual warmup choice instead of assuming a
        # particular vendor implementation is installed.
        for _ in range(3):
            warmup = invoke()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused = invoke()
        assert (fused is not None) == (warmup is not None)
        if not quantize or width == 766:
            assert fused is None
        for _ in range(2):
            input.copy_(torch.randn_like(input) * 9)
            generic_moe.apply_moe_activation(
                generic_moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                reference,
                input,
                gemm1_clamp_limit=7.0,
                gemm1_alpha=1.702,
                gemm1_beta=1.0,
            )
            if fused is not None:
                qref, sref = moe_kernel_quantize_input(
                    reference, None, torch.int8, True, None
                )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
            if fused is not None:
                assert torch.equal(fused[0], qref)
                assert torch.equal(fused[1], sref)


@pytest.mark.parametrize(
    "activation",
    [
        MoEActivation.SWIGLUOAI,
        MoEActivation.SWIGLUSTEP,
        MoEActivation.SILU_NO_MUL,
        MoEActivation.GELU_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
    ],
)
def test_native_moe_activations_delegate_without_duplicate_implementations(
    monkeypatch, activation
):
    import vllm_fl.ops.fused_moe.activation as module

    x = torch.zeros(2, 8)
    out = torch.empty(2, 4 if activation.is_gated else 8)
    native = Mock(return_value=out)
    monkeypatch.setattr(module, "upstream_apply_moe_activation", native)
    assert module.apply_moe_activation(activation, out, x) is out
    native.assert_called_once_with(
        activation, out, x, clamp_limit=None, alpha=1.0, beta=0.0
    )
