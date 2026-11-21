# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for ``vllm_fl.quantization.w8a8.moe_experts`` guards.

The core suite (``tests/unit_tests/quantization/test_w8a8_moe_experts.py``)
covers the numeric experts paths (native/Ascend/grouped kernels, NPU
routing, FlagGems contract forwarding) plus a handful of contract
violations.  Still uncovered on the combined baseline:

* ``_validate_w8a8_contract`` branch bodies that the core tests never
  trip: non-contiguous activations, non-3D weights, non-int8 weights,
  expert-count / hidden-size incompatibilities, a wrong gate/up width,
  weights non-contiguous in K, mismatched routing shapes, a routing/row
  mismatch, routing non-contiguous in top_k, scale device mismatch, and
  the two bias-shape guards;
* both experts classes' ``apply`` quant-config guards (non-INT8 config,
  non-per-token / blocked quant config), the LoRA rejections, and the
  non-floating-point hidden-state ``TypeError``;
* ``_native_w8a8_fused_experts`` skipping nonlocal (negative) local
  expert ids through an ``expert_map``.

The module imports the vllm fused-MoE graph at module level; on hosts
without vLLM a minimal stub tree is installed at file scope before the
import (the same pattern the core suite's stub-less imports rely on when
real vLLM is present).  The directory conftest's ``fl_sys_modules``
snapshot keeps the environment order-independent.
"""

import enum
import importlib.util
import sys
import types
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.supplementary


def _make_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class _StubMoEActivation(enum.Enum):
    """Stand-in for vLLM's MoEActivation with the is_gated contract."""

    SILU = "silu"
    GELU = "gelu"
    GELU_NO_MUL = "gelu_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.name.endswith("NO_MUL")


def _install_stub_vllm_graph() -> dict:
    """Install the minimal vllm graph moe_experts imports at module level.

    Mirrors the chain the core suite relies on real vLLM for:
    ``vllm.model_executor.layers.fused_moe.modular_kernel`` (aliased
    ``mk``), ``...activation.MoEActivation``, ``...experts.triton_moe.
    TritonExperts`` and ``...fused_moe.fused_experts``.
    """
    vllm = _make_module("vllm")
    me = _make_module("vllm.model_executor")
    layers = _make_module("vllm.model_executor.layers")
    fm = _make_module("vllm.model_executor.layers.fused_moe")
    mk = _make_module("vllm.model_executor.layers.fused_moe.modular_kernel")
    mk.FusedMoEExperts = object
    mk.ExpertTokensMetadata = object
    act = _make_module("vllm.model_executor.layers.fused_moe.activation")
    act.MoEActivation = _StubMoEActivation
    triton_moe = _make_module("vllm.model_executor.layers.fused_moe.experts.triton_moe")

    class _TritonExperts:
        """Upstream TritonExperts stand-in (never instantiated here)."""

        def __init__(self, *args, **kwargs):
            raise AssertionError("TritonExperts constructor must not run")

    triton_moe.TritonExperts = _TritonExperts
    fused = _make_module("vllm.model_executor.layers.fused_moe.fused_moe")
    fused.fused_experts = lambda *a, **k: None
    experts_dir = _make_module("vllm.model_executor.layers.fused_moe.experts")

    vllm.model_executor = me
    me.layers = layers
    layers.fused_moe = fm
    fm.modular_kernel = mk
    fm.activation = act
    fm.experts = experts_dir
    experts_dir.triton_moe = triton_moe
    fm.fused = fused

    entries = {
        "vllm": vllm,
        "vllm.model_executor": me,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.layers.fused_moe": fm,
        "vllm.model_executor.layers.fused_moe.modular_kernel": mk,
        "vllm.model_executor.layers.fused_moe.activation": act,
        "vllm.model_executor.layers.fused_moe.experts": experts_dir,
        "vllm.model_executor.layers.fused_moe.experts.triton_moe": triton_moe,
        "vllm.model_executor.layers.fused_moe.fused_moe": fused,
    }
    previous: dict[str, Any] = {}
    for name, module in entries.items():
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module
    return previous


def _restore_stub_vllm_graph(previous: dict) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _has_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


if _has_vllm():  # pragma: no cover - real-vllm hosts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    import vllm_fl.quantization.w8a8.moe_experts as moe_experts
else:
    _stubs_previous = _install_stub_vllm_graph()
    import vllm_fl.quantization.w8a8.moe_experts as moe_experts  # noqa: E402

    _restore_stub_vllm_graph(_stubs_previous)
    MoEActivation = _StubMoEActivation


# ---------------------------------------------------------------------------
# Shared argument builders.
# ---------------------------------------------------------------------------


def _quant_config(**overrides):
    """Return a quant config double with the INT8 per-token contract."""
    base = {
        "use_int8_w8a8": True,
        "per_act_token_quant": True,
        "block_shape": None,
        "w1_scale": torch.ones((2, 8, 1), dtype=torch.float32),
        "w2_scale": torch.ones((2, 4, 1), dtype=torch.float32),
        "w1_bias": None,
        "w2_bias": None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _experts_instance(quant_config=None, *, lora_context=None):
    return types.SimpleNamespace(
        quant_config=(_quant_config() if quant_config is None else quant_config),
        activation_config=types.SimpleNamespace(clamp_limit=None),
        _lora_context=lora_context,
    )


def _apply_arguments(**overrides):
    """Return the exact FlagGems W8A8 ``apply`` contract arguments."""
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    arguments = {
        "output": torch.empty_like(hidden_states),
        "hidden_states": hidden_states,
        "w1": torch.ones((2, 8, 4), dtype=torch.int8),
        "w2": torch.ones((2, 4, 4), dtype=torch.int8),
        "topk_weights": torch.ones((2, 1), dtype=torch.float32),
        "topk_ids": torch.zeros((2, 1), dtype=torch.int64),
        "activation": MoEActivation.SILU,
        "global_num_experts": 2,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": torch.empty(0),
        "workspace2": torch.empty(0),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": False,
    }
    arguments.update(overrides)
    return arguments


def _forbid_implementations(monkeypatch):
    """Fail loudly if any experts implementation is reached."""
    monkeypatch.setattr(
        moe_experts,
        "_flaggems_fused_experts_impl",
        lambda **kwargs: pytest.fail("FlagGems fused_experts must not run"),
    )
    monkeypatch.setattr(
        moe_experts,
        "_vllm_fused_experts",
        lambda **kwargs: pytest.fail("vLLM fused_experts must not run"),
    )


# ---------------------------------------------------------------------------
# _validate_w8a8_contract branch bodies.
# ---------------------------------------------------------------------------


def _valid_contract_arguments():
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    return {
        "hidden_states": hidden_states,
        "w1": torch.ones((2, 8, 4), dtype=torch.int8),
        "w2": torch.ones((2, 4, 4), dtype=torch.int8),
        "topk_weights": torch.ones((2, 1), dtype=torch.float32),
        "topk_ids": torch.zeros((2, 1), dtype=torch.int64),
        "activation": MoEActivation.SILU,
        "w1_scale": torch.ones((2, 8, 1), dtype=torch.float32),
        "w2_scale": torch.ones((2, 4, 1), dtype=torch.float32),
        "w1_bias": None,
        "w2_bias": None,
    }


class TestValidateW8A8ContractBranches:
    """Each contract violation raises its specific guard."""

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            pytest.param(
                lambda a: a.__setitem__(
                    "hidden_states",
                    torch.ones((2, 4, 8), dtype=torch.bfloat16)[:, :, ::2],
                ),
                "contiguous \\[M, K\\] activations",
                id="non-contiguous-activations",
            ),
            pytest.param(
                lambda a: a.__setitem__(
                    "hidden_states", torch.ones((2, 4, 1), dtype=torch.bfloat16)
                ),
                "contiguous \\[M, K\\] activations",
                id="three-dimensional-activations",
            ),
            pytest.param(
                lambda a: a.__setitem__("w1", torch.ones((8, 4), dtype=torch.int8)),
                "3D expert weights",
                id="two-dimensional-w1",
            ),
            pytest.param(
                lambda a: a.__setitem__("w2", torch.ones((4, 4), dtype=torch.int8)),
                "3D expert weights",
                id="two-dimensional-w2",
            ),
            pytest.param(
                lambda a: (
                    a.__setitem__("w1", torch.ones((3, 8, 4), dtype=torch.int8)),
                    a.__setitem__(
                        "w1_scale", torch.ones((3, 8, 1), dtype=torch.float32)
                    ),
                ),
                "weight shapes are incompatible",
                id="expert-count-mismatch",
            ),
            pytest.param(
                lambda a: a.__setitem__(
                    "hidden_states", torch.ones((2, 5), dtype=torch.bfloat16)
                ),
                "weight shapes are incompatible",
                id="hidden-size-mismatch",
            ),
            pytest.param(
                lambda a: a.__setitem__("w1", torch.ones((2, 4, 4), dtype=torch.int8)),
                "gate/up and down-projection shapes",
                id="wrong-w1-width",
            ),
            pytest.param(
                lambda a: a.__setitem__(
                    "w1", torch.ones((2, 8, 8), dtype=torch.int8)[:, :, ::2]
                ),
                "contiguous in K",
                id="w1-non-contiguous-in-k",
            ),
            pytest.param(
                lambda a: a.__setitem__(
                    "w2", torch.ones((2, 4, 8), dtype=torch.int8)[:, :, ::2]
                ),
                "contiguous in K",
                id="w2-non-contiguous-in-k",
            ),
            pytest.param(
                lambda a: a.__setitem__(
                    "topk_weights", torch.ones((2, 2), dtype=torch.float32)
                ),
                "matching \\[M, top_k\\] routing",
                id="routing-shape-mismatch",
            ),
            pytest.param(
                lambda a: (
                    a.__setitem__("topk_ids", torch.zeros((3, 1), dtype=torch.int64)),
                    a.__setitem__(
                        "topk_weights", torch.ones((3, 1), dtype=torch.float32)
                    ),
                ),
                "routing and activation rows must match",
                id="routing-row-mismatch",
            ),
        ],
    )
    def test_contract_violations_raise(self, mutate, match):
        """Each pre-scale contract breach raises its own ValueError."""
        arguments = _valid_contract_arguments()
        mutate(arguments)
        with pytest.raises(ValueError, match=match):
            moe_experts._validate_w8a8_contract(**arguments)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda a: a.__setitem__(
                    "w1", torch.ones((2, 8, 4), dtype=torch.float16)
                ),
                id="float-w1",
            ),
            pytest.param(
                lambda a: a.__setitem__("w2", torch.ones((2, 4, 4), dtype=torch.int16)),
                id="int16-w2",
            ),
        ],
    )
    def test_non_int8_weights_raise_type_error(self, mutate):
        """Non-int8 expert weights raise the TypeError guard."""
        arguments = _valid_contract_arguments()
        mutate(arguments)
        with pytest.raises(TypeError, match="expert weights must be int8"):
            moe_experts._validate_w8a8_contract(**arguments)

    def test_non_silu_activation_raises_not_implemented(self):
        """A gated non-silu activation is rejected before shape checks."""
        arguments = _valid_contract_arguments()
        arguments["activation"] = MoEActivation.GELU
        with pytest.raises(NotImplementedError, match="only silu"):
            moe_experts._validate_w8a8_contract(**arguments)

    def test_routing_non_contiguous_in_top_k_raises(self):
        """A routing tensor with an inner stride is rejected."""
        arguments = _valid_contract_arguments()
        arguments["topk_ids"] = torch.zeros((2, 2), dtype=torch.int64)[:, ::2]
        arguments["topk_weights"] = torch.ones((2, 2), dtype=torch.float32)[:, ::2]
        with pytest.raises(ValueError, match="contiguous in top_k"):
            moe_experts._validate_w8a8_contract(**arguments)

    @pytest.mark.parametrize(
        ("scale_name", "bad_scale"),
        [
            pytest.param("w1_scale", None, id="w1-scale-missing"),
            pytest.param("w2_scale", None, id="w2-scale-missing"),
            pytest.param(
                "w1_scale",
                torch.ones((2, 8, 2), dtype=torch.float32),
                id="w1-scale-wrong-depth",
            ),
            pytest.param(
                "w2_scale",
                torch.ones((2, 4, 2), dtype=torch.float32),
                id="w2-scale-wrong-depth",
            ),
        ],
    )
    def test_bad_scale_shapes_raise(self, scale_name, bad_scale):
        """A scale with the wrong shape (or None) is rejected."""
        arguments = _valid_contract_arguments()
        arguments[scale_name] = bad_scale
        with pytest.raises(ValueError, match="scale must have shape"):
            moe_experts._validate_w8a8_contract(**arguments)

    def test_scale_dtype_must_be_float32(self):
        """A float16 scale is rejected with the TypeError guard."""
        arguments = _valid_contract_arguments()
        arguments["w2_scale"] = torch.ones((2, 4, 1), dtype=torch.float16)
        with pytest.raises(TypeError, match="weight scales must be float32"):
            moe_experts._validate_w8a8_contract(**arguments)

    def test_scale_contiguity_required(self):
        """A non-contiguous scale is rejected."""
        arguments = _valid_contract_arguments()
        # A [2, 8, 1] tensor strided so the last dim steps by 2.
        arguments["w1_scale"] = torch.ones((2, 8, 2), dtype=torch.float32)[:, :, ::2]
        with pytest.raises(ValueError, match="weight scales must be contiguous"):
            moe_experts._validate_w8a8_contract(**arguments)

    def test_scale_device_mismatch_raises(self):
        """A scale on a different device than its weights is rejected."""
        arguments = _valid_contract_arguments()
        # No second device is needed: a meta-device weight pair keeps every
        # earlier contract check green (shape/dtype/stride are meta-safe)
        # while the CPU scale fails the shared-device guard.
        arguments["w1"] = torch.ones((2, 8, 4), dtype=torch.int8, device="meta")
        arguments["w2"] = torch.ones((2, 4, 4), dtype=torch.int8, device="meta")
        with pytest.raises(ValueError, match="must share a device"):
            moe_experts._validate_w8a8_contract(**arguments)

    @pytest.mark.parametrize(
        ("bias_name", "bad_bias"),
        [
            pytest.param(
                "w1_bias", torch.zeros((2, 7), dtype=torch.float32), id="w1-bias"
            ),
            pytest.param(
                "w2_bias", torch.zeros((2, 3), dtype=torch.float32), id="w2-bias"
            ),
        ],
    )
    def test_bias_shape_mismatch_raises(self, bias_name, bad_bias):
        """A bias whose shape does not match its weights is rejected."""
        arguments = _valid_contract_arguments()
        arguments[bias_name] = bad_bias
        with pytest.raises(ValueError, match="bias must have shape"):
            moe_experts._validate_w8a8_contract(**arguments)

    def test_valid_contract_passes_silently(self):
        """The canonical contract arguments raise nothing."""
        moe_experts._validate_w8a8_contract(**_valid_contract_arguments())


# ---------------------------------------------------------------------------
# VllmFunctionalW8A8Experts.apply guards (missing 314, 318, 322, 326, 335).
# ---------------------------------------------------------------------------


class TestVllmFunctionalApplyGuards:
    """``VllmFunctionalW8A8Experts.apply`` validation branches."""

    def test_non_int8_config_rejected(self, monkeypatch):
        """A config without use_int8_w8a8 raises before dispatch."""
        _forbid_implementations(monkeypatch)
        instance = _experts_instance(_quant_config(use_int8_w8a8=False))
        with pytest.raises(ValueError, match="INT8 W8A8 quant config"):
            moe_experts.VllmFunctionalW8A8Experts.apply(instance, **_apply_arguments())

    @pytest.mark.parametrize(
        "config_override",
        [
            pytest.param({"per_act_token_quant": False}, id="static-activation"),
            pytest.param({"block_shape": [128, 128]}, id="blocked-weights"),
        ],
    )
    def test_non_per_token_config_rejected(self, monkeypatch, config_override):
        """Static or blocked quant configs are rejected."""
        _forbid_implementations(monkeypatch)
        instance = _experts_instance(_quant_config(**config_override))
        with pytest.raises(ValueError, match="per-token/channel-wise"):
            moe_experts.VllmFunctionalW8A8Experts.apply(instance, **_apply_arguments())

    def test_lora_context_rejected(self, monkeypatch):
        """A LoRA context raises NotImplementedError."""
        _forbid_implementations(monkeypatch)
        instance = _experts_instance(lora_context=object())
        with pytest.raises(NotImplementedError, match="does not support LoRA"):
            moe_experts.VllmFunctionalW8A8Experts.apply(instance, **_apply_arguments())

    def test_prequantized_input_rejected(self, monkeypatch):
        """An a1q/a2 scale pair raises the pre-quantized-input guard."""
        _forbid_implementations(monkeypatch)
        arguments = _apply_arguments(a1q_scale=torch.ones((2, 1), dtype=torch.float32))
        with pytest.raises(ValueError, match="quantized before"):
            moe_experts.VllmFunctionalW8A8Experts.apply(
                _experts_instance(), **arguments
            )

    def test_non_floating_hidden_states_rejected(self, monkeypatch):
        """An integer hidden_states tensor raises TypeError."""
        _forbid_implementations(monkeypatch)
        arguments = _apply_arguments()
        arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
        with pytest.raises(TypeError, match="floating-point hidden states"):
            moe_experts.VllmFunctionalW8A8Experts.apply(
                _experts_instance(), **arguments
            )


# ---------------------------------------------------------------------------
# FlagGemsW8A8Experts.apply guards (missing 390, 392, 396, 409).
# ---------------------------------------------------------------------------


class TestFlagGemsApplyGuards:
    """``FlagGemsW8A8Experts.apply`` validation branches."""

    def test_non_int8_config_rejected(self, monkeypatch):
        """A config without use_int8_w8a8 raises before dispatch."""
        _forbid_implementations(monkeypatch)
        instance = _experts_instance(_quant_config(use_int8_w8a8=False))
        with pytest.raises(ValueError, match="INT8 W8A8 quant config"):
            moe_experts.FlagGemsW8A8Experts.apply(instance, **_apply_arguments())

    @pytest.mark.parametrize(
        "config_override",
        [
            pytest.param({"per_act_token_quant": False}, id="static-activation"),
            pytest.param({"block_shape": [128, 128]}, id="blocked-weights"),
        ],
    )
    def test_non_per_token_config_rejected(self, monkeypatch, config_override):
        """Static or blocked quant configs are rejected."""
        _forbid_implementations(monkeypatch)
        instance = _experts_instance(_quant_config(**config_override))
        with pytest.raises(ValueError, match="per-token/channel-wise"):
            moe_experts.FlagGemsW8A8Experts.apply(instance, **_apply_arguments())

    def test_lora_context_rejected(self, monkeypatch):
        """A LoRA context raises NotImplementedError."""
        _forbid_implementations(monkeypatch)
        instance = _experts_instance(lora_context=object())
        with pytest.raises(NotImplementedError, match="does not support LoRA"):
            moe_experts.FlagGemsW8A8Experts.apply(instance, **_apply_arguments())

    def test_non_floating_hidden_states_rejected(self, monkeypatch):
        """An integer hidden_states tensor raises TypeError."""
        _forbid_implementations(monkeypatch)
        arguments = _apply_arguments()
        arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.int8)
        with pytest.raises(TypeError, match="floating-point hidden states"):
            moe_experts.FlagGemsW8A8Experts.apply(_experts_instance(), **arguments)


# ---------------------------------------------------------------------------
# _ascend_w8a8_grouped_experts dtype guard and empty-token short-circuits
# (missing lines 152, 153, 155, 162).
# ---------------------------------------------------------------------------


class TestAscendGroupedExpertsGuards:
    """``_ascend_w8a8_grouped_experts`` dtype and empty-input branches."""

    def _arguments(self):
        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
        return {
            "hidden_states": hidden_states,
            "w1": torch.ones((2, 8, 4), dtype=torch.int8),
            "w2": torch.ones((2, 4, 4), dtype=torch.int8),
            "topk_weights": torch.ones((2, 1), dtype=torch.float32),
            "topk_ids": torch.zeros((2, 1), dtype=torch.int64),
            "w1_scale": torch.ones((2, 8, 1), dtype=torch.float32),
            "w2_scale": torch.ones((2, 4, 1), dtype=torch.float32),
            "w1_bias": None,
            "w2_bias": None,
            "expert_map": None,
            "apply_router_weight_on_input": False,
        }

    def _fail_torch_npu(self, monkeypatch):
        """Install a torch_npu double whose kernels fail loudly."""
        monkeypatch.setitem(
            sys.modules,
            "torch_npu",
            types.SimpleNamespace(
                npu_dynamic_quant=lambda *a, **k: pytest.fail("kernel reached"),
                npu_grouped_matmul=lambda *a, **k: pytest.fail("kernel reached"),
                npu_swiglu=lambda *a, **k: pytest.fail("kernel reached"),
            ),
        )

    @pytest.mark.parametrize(
        ("dtype", "fragment"),
        [
            pytest.param(
                torch.bfloat16, "bfloat16 or float16", id="bfloat16-scale-dtype"
            ),
            pytest.param(
                torch.float16, "bfloat16 or float16", id="float16-scale-dtype"
            ),
        ],
    )
    def test_supported_dtypes_select_scale_dtype(self, monkeypatch, dtype, fragment):
        """bf16/fp16 inputs pass the dtype guard (never reach a kernel)."""
        self._fail_torch_npu(monkeypatch)
        arguments = self._arguments()
        arguments["hidden_states"] = torch.ones((0, 4), dtype=dtype)
        arguments["topk_ids"] = torch.zeros((0, 1), dtype=torch.int64)
        arguments["topk_weights"] = torch.ones((0, 1), dtype=torch.float32)
        result = moe_experts._ascend_w8a8_grouped_experts(**arguments, clamp_limit=None)
        # Zero-token short-circuit: empty (0, w2.shape[1]) output.
        assert result.shape == (0, 4)
        assert result.dtype == dtype

    def test_unsupported_dtype_raises_type_error(self, monkeypatch):
        """A float32 input is rejected by the dtype guard."""
        self._fail_torch_npu(monkeypatch)
        arguments = self._arguments()
        arguments["hidden_states"] = torch.ones((2, 4), dtype=torch.float32)
        with pytest.raises(TypeError, match="bfloat16 or float16"):
            moe_experts._ascend_w8a8_grouped_experts(**arguments, clamp_limit=None)

    def test_zero_tokens_short_circuits(self, monkeypatch):
        """An empty routing tensor returns an empty output before kernels."""
        self._fail_torch_npu(monkeypatch)
        arguments = self._arguments()
        arguments["topk_ids"] = torch.zeros((0, 1), dtype=torch.int64)
        arguments["topk_weights"] = torch.ones((0, 1), dtype=torch.float32)
        result = moe_experts._ascend_w8a8_grouped_experts(**arguments, clamp_limit=None)
        assert result.shape == (0, 4)

    def test_all_nonlocal_routes_return_zeros(self, monkeypatch):
        """Routing every token to a nonlocal expert returns zero rows."""
        self._fail_torch_npu(monkeypatch)
        arguments = self._arguments()
        arguments["expert_map"] = torch.tensor([-1, -1], dtype=torch.int64)
        result = moe_experts._ascend_w8a8_grouped_experts(**arguments, clamp_limit=None)
        assert torch.equal(result, torch.zeros((2, 4), dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# _native_w8a8_fused_experts expert_map skip (missing line 84).
# ---------------------------------------------------------------------------


class TestNativeExpertsSkipsNonlocalRoutes:
    """``_native_w8a8_fused_experts`` with a nonlocal expert_map."""

    def test_negative_local_ids_are_skipped(self):
        """Tokens routed to nonlocal experts produce zero output rows."""
        hidden_states = torch.tensor(
            [[1.0, 2.0, -1.0, 0.5], [3.0, -2.0, 1.0, -0.5]],
            dtype=torch.bfloat16,
        )
        # Two experts but expert 1 is not local (map -> -1).
        w1 = torch.ones((2, 8, 4), dtype=torch.int8)
        w2 = torch.ones((2, 4, 4), dtype=torch.int8)
        w1_scale = torch.ones((2, 8, 1), dtype=torch.float32)
        w2_scale = torch.ones((2, 4, 1), dtype=torch.float32)
        # Token 0 routes to global expert 0 (local), token 1 to global
        # expert 1 (nonlocal, local id -1): only token 0 gets output.
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)
        expert_map = torch.tensor([0, -1], dtype=torch.int64)

        actual = moe_experts._native_w8a8_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            None,
            None,
            expert_map,
            False,
        )

        # Independent oracle: token 1 must be untouched (zeros); token 0's
        # row is computed with the same scalar int8 emulation.
        assert torch.equal(actual[1], torch.zeros(4, dtype=torch.bfloat16))
        expected_row0 = _independent_row0(
            hidden_states[0].float(),
            w1[0],
            w2[0],
            w1_scale[0],
            w2_scale[0],
            float(topk_weights[0, 0]),
        )
        torch.testing.assert_close(
            actual[0].float(), expected_row0.float(), rtol=0.05, atol=0.05
        )


def _independent_row0(row, w1_e, w2_e, w1_scale_e, w2_scale_e, route_weight):
    """Scalar int8 emulation of the single routed token's expert pass."""

    def qdq(value):
        scale = value.abs().amax(dim=-1, keepdim=True) / 127.0
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (torch.round(value / safe).clamp(-127, 127) * safe).to(value.dtype)

    inputs = qdq(row.unsqueeze(0))
    gate_up_weight = w1_e.to(inputs.dtype) * w1_scale_e.squeeze(-1).to(
        inputs.dtype
    ).unsqueeze(-1)
    gate_up = inputs @ gate_up_weight.transpose(0, 1)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = qdq(torch.nn.functional.silu(gate) * up)
    down_weight = w2_e.to(activated.dtype) * w2_scale_e.squeeze(-1).to(
        activated.dtype
    ).unsqueeze(-1)
    out = activated @ down_weight.transpose(0, 1)
    return (out.float() * route_weight).squeeze(0).to(torch.bfloat16)
