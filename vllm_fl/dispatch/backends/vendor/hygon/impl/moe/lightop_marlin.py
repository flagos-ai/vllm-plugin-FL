# Copyright (c) 2026 BAAI. All rights reserved.

"""Optional Hygon LightOp Marlin backend for dynamic INT8 W8A8 MoE.

The default Hygon W8A8 backend remains FL Triton. Set
``VLLM_FL_HYGON_W8A8_MOE_BACKEND=lightop_marlin`` before model construction
to opt into this backend.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.utils import replace_parameter

from vllm_fl.ops.fused_moe.layer import CompressedTensorsW8A8Int8MoEMethodFL

logger = init_logger(__name__)

HYGON_W8A8_MOE_BACKEND_ENV = "VLLM_FL_HYGON_W8A8_MOE_BACKEND"
HYGON_W8A8_MOE_BACKEND_TRITON = "triton"
HYGON_W8A8_MOE_BACKEND_LIGHTOP = "lightop_marlin"
_HYGON_W8A8_MOE_BACKENDS = {
    HYGON_W8A8_MOE_BACKEND_TRITON,
    HYGON_W8A8_MOE_BACKEND_LIGHTOP,
}


def get_hygon_w8a8_moe_backend(value: str | None = None) -> str:
    """Resolve the Hygon W8A8 backend before model weights are processed."""
    backend = value
    if backend is None:
        backend = os.getenv(
            HYGON_W8A8_MOE_BACKEND_ENV,
            HYGON_W8A8_MOE_BACKEND_TRITON,
        )
    backend = backend.strip().lower()
    if backend == "lightop":
        backend = HYGON_W8A8_MOE_BACKEND_LIGHTOP
    if backend not in _HYGON_W8A8_MOE_BACKENDS:
        choices = ", ".join(sorted(_HYGON_W8A8_MOE_BACKENDS))
        raise ValueError(
            f"Invalid {HYGON_W8A8_MOE_BACKEND_ENV}={backend!r}; "
            f"expected one of: {choices}"
        )
    return backend


def pack_int8_weight_for_lightop_marlin(
    weight: torch.Tensor,
    k_tile: int = 64,
) -> torch.Tensor:
    """Pack an ``[E, N, K]`` INT8 weight using vllm_hcu's Marlin layout."""
    if weight.dtype != torch.int8:
        raise TypeError(f"LightOp Marlin requires INT8 weights, got {weight.dtype}")
    if weight.ndim != 3:
        raise ValueError(
            f"LightOp Marlin expects an [E, N, K] weight, got {tuple(weight.shape)}"
        )

    num_experts, size_n, size_k = weight.shape
    if size_k % k_tile != 0:
        raise ValueError(
            f"LightOp Marlin requires K divisible by {k_tile}, got K={size_k}"
        )

    return (
        weight.transpose(1, 2)
        .reshape(num_experts, size_k // k_tile, k_tile, size_n)
        .transpose(2, 3)
        .contiguous()
        .reshape(num_experts, size_k // k_tile, size_n * k_tile)
    )


def _load_lightop_int8_marlin() -> Callable:
    try:
        from lightop.moe import fused_experts_impl_int8_marlin
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "VLLM_FL_HYGON_W8A8_MOE_BACKEND=lightop_marlin requires "
            "lightop.moe.fused_experts_impl_int8_marlin"
        ) from exc
    from .lightop_moe_sum import adapt_lightop_moe_sum
    from .prefill_moe_gemm2 import adapt_prefill_moe_gemm2

    return adapt_prefill_moe_gemm2(adapt_lightop_moe_sum(fused_experts_impl_int8_marlin))


class HygonLightOpMarlinMoEMethodFL(CompressedTensorsW8A8Int8MoEMethodFL):
    """Use LightOp's monolithic Marlin experts behind vLLM's external router.

    This deliberately remains non-monolithic from ``MoERunner``'s perspective:
    DeepSeek-V4 routing (including sqrt-softplus and hash-table inputs) must be
    computed by the normal FL router before LightOp receives top-k ids/weights.
    """

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        # The backend is explicitly requested, so fail before allocating and
        # loading all expert weights when the vendor library is unavailable.
        _load_lightop_int8_marlin()
        super().__init__(
            weight_quant,
            input_quant,
            moe,
            layer_name,
            hygon_moe_backend=HYGON_W8A8_MOE_BACKEND_LIGHTOP,
        )
        parallel = moe.moe_parallel_config
        if parallel.dp_size > 1 or parallel.use_ep or parallel.use_all2all_kernels:
            raise NotImplementedError(
                "Hygon LightOp Marlin W8A8 MoE currently supports TP-only "
                "deployment without DP/EP all-to-all"
            )

    @property
    def supports_internal_mk(self) -> bool:
        # Prevent MoERunner from wrapping this method in the modular Triton
        # kernel. LightOp owns align, quantization, both GEMMs, activation and
        # reduction, while the vLLM/FL router remains external.
        return True

    @property
    def is_monolithic(self) -> bool:
        # Keep routing external so DeepSeek-V4 input_ids/hash routing is
        # preserved. ``apply`` below receives the selected top-k tensors.
        return False

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        _load_lightop_int8_marlin()
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        raw_w1_shape = tuple(layer.w13_weight.shape)
        raw_w2_shape = tuple(layer.w2_weight.shape)
        with torch.no_grad():
            w1 = pack_int8_weight_for_lightop_marlin(layer.w13_weight)
        replace_parameter(layer, "w13_weight", w1)

        # Replace W1 before packing W2 to reduce the transient device-memory
        # peak during post-load conversion of the large expert tensors.
        with torch.no_grad():
            w2 = pack_int8_weight_for_lightop_marlin(layer.w2_weight)
        replace_parameter(layer, "w2_weight", w2)

        logger.info_once(
            "Using Hygon LightOp Marlin INT8 W8A8 MoE; packed weights "
            "W1 %s -> %s, W2 %s -> %s",
            raw_w1_shape,
            tuple(w1.shape),
            raw_w2_shape,
            tuple(w2.shape),
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if self.moe_quant_config is None:
            raise RuntimeError("LightOp Marlin MoE weights were not initialized")

        fused_experts_impl_int8_marlin = _load_lightop_int8_marlin()
        return fused_experts_impl_int8_marlin(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not getattr(self.moe, "disable_inplace", False),
            activation=layer.activation.value,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            use_nn_moe=False,
            i_q=None,
            i_s=None,
            shared_output=None,
            routed_scaling_factor=1.0,
        )

    @property
    def supports_eplb(self) -> bool:
        return False


__all__ = [
    "HYGON_W8A8_MOE_BACKEND_ENV",
    "HYGON_W8A8_MOE_BACKEND_LIGHTOP",
    "HYGON_W8A8_MOE_BACKEND_TRITON",
    "HygonLightOpMarlinMoEMethodFL",
    "get_hygon_w8a8_moe_backend",
    "pack_int8_weight_for_lightop_marlin",
]
