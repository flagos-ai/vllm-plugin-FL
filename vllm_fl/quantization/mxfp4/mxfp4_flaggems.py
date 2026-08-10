# Copyright (c) 2026 BAAI. All rights reserved.
"""Native-layout MXFP4 experts for the FlagGems backend."""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
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
    QuantKey,
    kMxfp4Static,
)

class FlagGemsMxfp4Experts(mk.FusedMoEExpertsModular):
    """FlagGems MXFP4 experts consuming checkpoint-native uint8 weights."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # FlagGems owns its temporary buffers. Reuse workspace13 for the final
        # output and keep workspace2 at its minimum allocation.
        return (M, K), (1,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if expert_map is not None:
            raise NotImplementedError("FlagGems MXFP4 does not support expert_map")
        if global_num_experts not in (-1, w1.size(0)):
            raise NotImplementedError(
                "FlagGems MXFP4 requires local and global expert counts to match"
            )
        if self.w1_bias is not None or self.w2_bias is not None:
            raise NotImplementedError("FlagGems MXFP4 does not support expert bias")
        if activation != MoEActivation.SILU:
            raise NotImplementedError(
                f"FlagGems MXFP4 only supports SiLU, got {activation}"
            )
        if a1q_scale is not None or a2_scale is not None:
            raise NotImplementedError(
                "FlagGems MXFP4 expects unquantized bf16/fp16 activations"
            )
        if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise TypeError(
                "FlagGems MXFP4 requires checkpoint-native uint8 weights, "
                f"got w1={w1.dtype}, w2={w2.dtype}"
            )
        assert self.w1_scale is not None and self.w2_scale is not None

        from flag_gems.fused.fused_marlin_moe import fused_moe_mxfp4

        result = fused_moe_mxfp4(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
            group_size=32,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=False,
        )
        output.copy_(result)


class FlagGemsMxfp4MoEMethod(Mxfp4MoEMethod):
    """MXFP4 method that deliberately skips Marlin weight conversion."""

    @classmethod
    def from_method(cls, method: Mxfp4MoEMethod) -> "FlagGemsMxfp4MoEMethod":
        converted = cls.__new__(cls)
        converted.__dict__.update(method.__dict__)
        converted.experts_cls = FlagGemsMxfp4Experts
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
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise TypeError("FlagGems MXFP4 weights must remain uint8 before setup")

        # Build the regular W4A16 quant descriptor from the original E8M0
        # scales. Crucially, do not call convert_weight_to_mxfp4_moe_kernel_format:
        # that function irreversibly repacks the weights into Marlin int32 layout.
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
            mxfp4_backend=Mxfp4MoeBackend.MARLIN,
            experts_cls=FlagGemsMxfp4Experts,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )


def flaggems_mxfp4_compatible(moe: FusedMoEConfig) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
        and moe.in_dtype in (torch.float16, torch.bfloat16)
        and moe.activation == MoEActivation.SILU
        and not moe.has_bias
        and not moe.is_lora_enabled
        and not moe.moe_parallel_config.use_ep
        and moe.num_local_experts == moe.num_experts
    )


__all__ = [
    "FlagGemsMxfp4Experts",
    "FlagGemsMxfp4MoEMethod",
    "flaggems_mxfp4_compatible",
]
