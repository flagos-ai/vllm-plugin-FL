# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware-specific experts adapters for dynamic per-token W8A8 MoE."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts as _vllm_fused_experts,
)

import vllm_fl.envs as fl_envs


def _flaggems_fused_experts_impl(**kwargs) -> torch.Tensor:
    """Resolve FlagGems lazily after the platform runtime is initialized."""
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        import flaggems_vllm as fused_ops
    else:
        import flag_gems as fused_ops

    return fused_ops.fused_experts_impl(**kwargs)


def _validate_w8a8_contract(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> None:
    """Validate the exact channel-wise FlagGems W8A8 tensor contract."""
    if activation not in (
        MoEActivation.SILU,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    ):
        raise NotImplementedError(
            "FlagGems W8A8 experts support only silu and swigluoai_uninterleave, "
            f"got {activation.value}"
        )
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("FlagGems W8A8 MoE requires contiguous [M, K] activations")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("FlagGems W8A8 MoE requires 3D expert weights")
    if w1.dtype != torch.int8 or w2.dtype != torch.int8:
        raise TypeError("FlagGems W8A8 MoE expert weights must be int8")
    if w1.shape[0] != w2.shape[0] or hidden_states.shape[1] != w1.shape[2]:
        raise ValueError("FlagGems W8A8 MoE weight shapes are incompatible")
    expected_w1_width = w2.shape[2] * (2 if activation.is_gated else 1)
    if w1.shape[1] != expected_w1_width:
        raise ValueError(
            "FlagGems W8A8 MoE gate/up and down-projection shapes are incompatible"
        )
    if w1.stride(-1) != 1 or w2.stride(-1) != 1:
        raise ValueError("FlagGems W8A8 MoE weights must be contiguous in K")
    if topk_weights.shape != topk_ids.shape or topk_ids.ndim != 2:
        raise ValueError("FlagGems W8A8 MoE requires matching [M, top_k] routing")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("FlagGems W8A8 MoE routing and activation rows must match")
    if topk_weights.stride(-1) != 1 or topk_ids.stride(-1) != 1:
        raise ValueError("FlagGems W8A8 MoE routing must be contiguous in top_k")

    expected_w1_scale = (w1.shape[0], w1.shape[1], 1)
    expected_w2_scale = (w2.shape[0], w2.shape[1], 1)
    if w1_scale is None or tuple(w1_scale.shape) != expected_w1_scale:
        raise ValueError(
            f"w1_scale must have shape {expected_w1_scale}, "
            f"got {None if w1_scale is None else tuple(w1_scale.shape)}"
        )
    if w2_scale is None or tuple(w2_scale.shape) != expected_w2_scale:
        raise ValueError(
            f"w2_scale must have shape {expected_w2_scale}, "
            f"got {None if w2_scale is None else tuple(w2_scale.shape)}"
        )
    if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise TypeError("FlagGems W8A8 MoE weight scales must be float32")
    if not w1_scale.is_contiguous() or not w2_scale.is_contiguous():
        raise ValueError("FlagGems W8A8 MoE weight scales must be contiguous")
    if w1_scale.device != w1.device or w2_scale.device != w2.device:
        raise ValueError("FlagGems W8A8 MoE weights and scales must share a device")

    expected_w1_bias = (w1.shape[0], w1.shape[1])
    expected_w2_bias = (w2.shape[0], w2.shape[1])
    if w1_bias is not None and tuple(w1_bias.shape) != expected_w1_bias:
        raise ValueError(f"w1_bias must have shape {expected_w1_bias}")
    if w2_bias is not None and tuple(w2_bias.shape) != expected_w2_bias:
        raise ValueError(f"w2_bias must have shape {expected_w2_bias}")


def _validate_dynamic_w8a8_inputs(experts, hidden_states, a1q_scale, a2_scale):
    """Shared input contract for the native and FlagGems functional adapters."""
    config = experts.quant_config
    if not config.use_int8_w8a8:
        raise ValueError("Functional W8A8 experts require an INT8 W8A8 quant config")
    if not config.per_act_token_quant or config.block_shape is not None:
        raise ValueError(
            "Functional W8A8 experts require dynamic per-token/channel-wise INT8"
        )
    if getattr(experts, "_lora_context", None) is not None:
        raise NotImplementedError("Functional W8A8 experts do not support LoRA")
    if a1q_scale is not None or a2_scale is not None:
        raise ValueError(
            "W8A8 activation was quantized before the functional experts adapter; "
            "expected an unquantized floating-point input"
        )
    if hidden_states.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            "W8A8 functional experts require floating-point hidden states, "
            f"got {hidden_states.dtype}"
        )


class VllmFunctionalW8A8Experts(TritonExperts):
    """Run native MoE with floating-point inputs and preserved activations.

    Ordinary SiLU uses vLLM's functional API. Configurable activations use
    TritonExperts.apply directly because the functional API drops their
    clamp, alpha, and beta parameters. Both paths quantize inputs internally.
    """

    def __init__(self, moe_config, quant_config):
        # The compressed-tensors INT8 builder does not carry these fields.
        # Copy rather than mutate the quant config shared with other layers.
        activation_params = {
            quant_name: getattr(moe_config, config_name)
            for quant_name, config_name in (
                ("gemm1_clamp_limit", "swiglu_limit"),
                ("gemm1_alpha", "swiglu_alpha"),
                ("gemm1_beta", "swiglu_beta"),
            )
            if getattr(moe_config, config_name, None) is not None
        }
        self._use_modular = bool(activation_params)
        if activation_params:
            quant_config = replace(quant_config, **activation_params)
        super().__init__(moe_config, quant_config)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

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
        _validate_dynamic_w8a8_inputs(self, hidden_states, a1q_scale, a2_scale)

        if activation is not MoEActivation.SILU or self._use_modular:
            return super().apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        del workspace13, workspace2, expert_tokens_meta

        quant_config = self.quant_config
        result = _vllm_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=quant_config,
        )
        output.copy_(result)


class FlagGemsW8A8Experts(TritonExperts):
    """Keep inputs floating-point and let FlagGems own the full W8A8 pipeline.

    The checkpoint contract is ``w1=[E,2I,K]``, ``w2=[E,K,I]`` with
    per-channel scales ``[E,2I,1]`` and ``[E,K,1]``. FlagGems produces the
    two dynamic activation scales internally as ``[M,1]`` and
    ``[M*top_k,1]``.
    """

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self._activation_kwargs = {}
        if moe_config.activation is MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            values = tuple(
                getattr(moe_config, name, None)
                for name in ("swiglu_alpha", "swiglu_beta", "swiglu_limit")
            )
            if any(v is None or not math.isfinite(float(v)) for v in values):
                raise ValueError(
                    "W8A8 split SwiGLU-OAI requires finite alpha, beta and limit"
                )
            if float(values[2]) <= 0:
                raise ValueError(
                    "W8A8 split SwiGLU-OAI requires a positive clamp limit"
                )
            if not quant_config.use_int8_w8a8 or not quant_config.per_act_token_quant:
                raise ValueError("W8A8 requires the FL dynamic-token INT8 adapter")
            # These fields belong to this selected experts instance. Leave both
            # the upstream quantization method and shared quant_config unchanged.
            self._activation_kwargs = {
                "gemm1_alpha": float(values[0]),
                "gemm1_beta": float(values[1]),
                "gemm1_clamp_limit": float(values[2]),
            }

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

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
        quant_config = self.quant_config
        _validate_dynamic_w8a8_inputs(self, hidden_states, a1q_scale, a2_scale)

        _validate_w8a8_contract(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            quant_config.w1_scale,
            quant_config.w2_scale,
            quant_config.w1_bias,
            quant_config.w2_bias,
        )

        del workspace13, workspace2, expert_tokens_meta
        operator = _flaggems_fused_experts_impl
        gate_params = {}
        if activation is MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            from vllm_fl.ops.fused_moe.fused_moe import fused_experts_impl

            gate_params = getattr(self, "_activation_kwargs", {})
            if any(
                gate_params.get(name) is None
                for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
            ):
                raise ValueError(
                    "W8A8 split SwiGLU-OAI requires explicit gate parameters"
                )
            operator = fused_experts_impl

        result = operator(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation.value,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            w1_bias=quant_config.w1_bias,
            w2_bias=quant_config.w2_bias,
            **gate_params,
        )
        output.copy_(result)


__all__ = [
    "FlagGemsW8A8Experts",
    "VllmFunctionalW8A8Experts",
]
