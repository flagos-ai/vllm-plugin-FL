# Copyright (c) 2026 BAAI. All rights reserved.
"""Native-Torch MXFP4 experts used as the correctness fallback."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
)


def dequantize_mxfp4_torch(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed E2M1 values using only native Torch operations."""
    if weight.dtype != torch.uint8:
        raise TypeError(f"Expected packed MXFP4 uint8 weights, got {weight.dtype}")

    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(-2)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
        dtype=torch.float32,
    )
    values = magnitude[(codes & 0x07).long()]
    values = torch.where((codes & 0x08).bool(), -values, values)

    group_size = 32
    if values.size(-1) != scale.size(-1) * group_size:
        raise ValueError(
            "MXFP4 scale shape does not match the packed weight shape: "
            f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    values = values.unflatten(-1, (scale.size(-1), group_size))
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    values = values * scale.float().unsqueeze(-1)
    return values.flatten(-2).to(dtype)


class ReferenceMxfp4Experts(mk.FusedMoEExpertsModular):
    """Slow, native-Torch MXFP4 MoE implementation used as a fallback."""

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return weight_key == kMxfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        return (M, K), (1,), (M, K)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ) -> None:
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                f"Torch MXFP4 reference only supports SiLU, got {activation}"
            )
        if a1q_scale is not None or a2_scale is not None:
            raise NotImplementedError(
                "Torch MXFP4 reference does not support quantized activations"
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("MXFP4 reference requires both weight scale tensors")

        dequantized_experts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        result = torch.zeros_like(hidden_states)

        for route in range(topk_ids.size(1)):
            route_ids = topk_ids[:, route]
            for global_expert in torch.unique(route_ids).tolist():
                local_expert = int(global_expert)
                if expert_map is not None:
                    local_expert = int(expert_map[local_expert].item())
                    if local_expert < 0:
                        continue

                token_mask = route_ids == global_expert
                x = hidden_states[token_mask]
                route_weight = topk_weights[token_mask, route].unsqueeze(-1)
                if apply_router_weight_on_input:
                    x = x * route_weight

                if local_expert not in dequantized_experts:
                    dequantized_experts[local_expert] = (
                        dequantize_mxfp4_torch(
                            w1[local_expert],
                            self.w1_scale[local_expert],
                            hidden_states.dtype,
                        ),
                        dequantize_mxfp4_torch(
                            w2[local_expert],
                            self.w2_scale[local_expert],
                            hidden_states.dtype,
                        ),
                    )
                expert_w1, expert_w2 = dequantized_experts[local_expert]

                gate_up = F.linear(
                    x,
                    expert_w1,
                    None if self.w1_bias is None else self.w1_bias[local_expert],
                )
                gate, up = gate_up.chunk(2, dim=-1)
                clamp_limit = self.quant_config.gemm1_clamp_limit
                if clamp_limit is not None and clamp_limit > 0:
                    gate = gate.clamp(max=clamp_limit)
                    up = up.clamp(min=-clamp_limit, max=clamp_limit)
                activated = F.silu(gate) * up
                expert_output = F.linear(
                    activated,
                    expert_w2,
                    None if self.w2_bias is None else self.w2_bias[local_expert],
                )
                if not apply_router_weight_on_input:
                    expert_output = expert_output * route_weight
                result[token_mask] += expert_output

        output.copy_(result)


class ReferenceMxfp4MoEMethod(Mxfp4MoEMethod):
    """MXFP4 method that preserves native weights for the Torch fallback."""

    @classmethod
    def from_method(cls, method: Mxfp4MoEMethod) -> "ReferenceMxfp4MoEMethod":
        converted = cls.__new__(cls)
        converted.__dict__.update(method.__dict__)
        converted.experts_cls = ReferenceMxfp4Experts
        converted.mxfp4_backend = Mxfp4MoeBackend.EMULATION
        return converted

    def _setup_kernel(
        self,
        layer,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
    ) -> None:
        self.moe_quant_config = mxfp4_w4a16_moe_quant_config(
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            w1_bias=w13_bias,
            w2_bias=w2_bias,
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )
        self.moe_kernel = make_mxfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            mxfp4_backend=Mxfp4MoeBackend.EMULATION,
            experts_cls=ReferenceMxfp4Experts,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )


__all__ = [
    "ReferenceMxfp4Experts",
    "ReferenceMxfp4MoEMethod",
    "dequantize_mxfp4_torch",
]
