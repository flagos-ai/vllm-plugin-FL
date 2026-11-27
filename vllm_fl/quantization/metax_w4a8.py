# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors W4A8 parameter lifecycle for the MetaX INT8 backend."""

import torch

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import set_weight_attrs

from vllm_fl.dispatch import resolve_op


class MetaXCompressedTensorsW4A8Method(FusedMoEMethodBase):
    """Keep packed INT4 checkpoint weights and delegate numerics to FGV."""

    def __init__(self, weight_quant, input_quant, moe, layer_name=None):
        super().__init__(moe)
        if (
            weight_quant.num_bits != 4
            or weight_quant.type != "int"
            or not weight_quant.symmetric
            or weight_quant.strategy != "channel"
            or weight_quant.group_size is not None
        ):
            raise NotImplementedError(
                "MetaX W4A8 requires symmetric per-channel INT4 weights"
            )
        if (
            input_quant.num_bits != 8
            or input_quant.type != "int"
            or not input_quant.dynamic
            or not input_quant.symmetric
            or input_quant.strategy != "token"
        ):
            raise NotImplementedError(
                "MetaX W4A8 requires dynamic symmetric per-token INT8 activations"
            )
        if moe.has_bias or moe.is_lora_enabled:
            raise NotImplementedError(
                "MetaX W4A8 supports bias-free experts without LoRA"
            )
        self.weight_quant, self.input_quant = weight_quant, input_quant
        self._compute = resolve_op("w4a8_moe_experts")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        h, i, e = hidden_size, intermediate_size_per_partition, num_experts
        if h % 8 or i % 8:
            raise ValueError(
                "packed W4A8 H and local intermediate must be divisible by eight"
            )
        for name, shape in [
            ("w13_weight_packed", (e, 2 * i, h // 8)),
            ("w2_weight_packed", (e, h, i // 8)),
        ]:
            parameter = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int32), requires_grad=False
            )
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)
        for name, shape in [
            ("w13_weight_scale", (e, 2 * i, 1)),
            ("w2_weight_scale", (e, h, 1)),
        ]:
            parameter = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.float32), requires_grad=False
            )
            set_weight_attrs(
                parameter, {"quant_method": "channel", **extra_weight_attrs}
            )
            layer.register_parameter(name, parameter)
        layer._metax_w4a8_processed = False

    def process_weights_after_loading(self, layer):
        if layer._metax_w4a8_processed:
            return
        # MCTLASS requires a multiple of 32 in the down GEMM K dimension.
        # Load the exact TP checkpoint slices first, then zero-pad packed INT4
        # gate/up rows and down columns once. No requantization is performed.
        w13, w2 = layer.w13_weight_packed, layer.w2_weight_packed
        e, twice_i, packed_h = w13.shape
        i = twice_i // 2
        padded_i = ((i + 31) // 32) * 32
        if padded_i != i:
            padded_w13 = w13.new_zeros((e, 2 * padded_i, packed_h))
            padded_w13[:, :i].copy_(w13[:, :i])
            padded_w13[:, padded_i : padded_i + i].copy_(w13[:, i:])
            padded_w2 = w2.new_zeros((e, w2.shape[1], padded_i // 8))
            padded_w2[:, :, : i // 8].copy_(w2)
            scale = layer.w13_weight_scale
            padded_scale = scale.new_zeros((e, 2 * padded_i, 1))
            padded_scale[:, :i].copy_(scale[:, :i])
            padded_scale[:, padded_i : padded_i + i].copy_(scale[:, i:])
            for name, value in (
                ("w13_weight_packed", padded_w13),
                ("w2_weight_packed", padded_w2),
                ("w13_weight_scale", padded_scale),
            ):
                setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
        for name in ("w13_weight_scale", "w2_weight_scale"):
            value = getattr(layer, name)
            setattr(
                layer,
                name,
                torch.nn.Parameter(
                    value.transpose(1, 2).contiguous(), requires_grad=False
                ),
            )
        layer._metax_w4a8_processed = True
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(torch.int8, GroupShape.PER_TOKEN),
            _a2=FusedMoEQuantDesc(torch.int8, GroupShape.PER_TOKEN),
            _w1=FusedMoEQuantDesc("int4", GroupShape.PER_TOKEN, layer.w13_weight_scale),
            _w2=FusedMoEQuantDesc("int4", GroupShape.PER_TOKEN, layer.w2_weight_scale),
        )

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        # The native runner executes non-overlapping shared experts separately.
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "input-side router weighting is not supported by this W4A8 path"
            )
        if not layer._metax_w4a8_processed:
            raise RuntimeError("packed W4A8 weights require post-load preparation")
        return self._compute(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights,
            topk_ids,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            clamp_limit=layer.swiglu_limit,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )
