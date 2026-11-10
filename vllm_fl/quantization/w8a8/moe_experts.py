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
from vllm_fl.dispatch import CachedOp


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
    if activation is not MoEActivation.SILU:
        raise NotImplementedError(
            "FlagGems fused_experts_impl currently supports only silu, "
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
        if not quant_config.use_int8_w8a8:
            raise ValueError(
                "VllmFunctionalW8A8Experts requires an INT8 W8A8 quant config"
            )
        if not quant_config.per_act_token_quant or quant_config.block_shape is not None:
            raise ValueError(
                "VllmFunctionalW8A8Experts requires dynamic per-token/channel-wise INT8"
            )
        if getattr(self, "_lora_context", None) is not None:
            raise NotImplementedError(
                "The vLLM functional W8A8 MoE bridge does not support LoRA"
            )
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError(
                "W8A8 activation was quantized before the vLLM functional "
                "experts bridge; expected an unquantized floating-point input"
            )
        if hidden_states.dtype not in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }:
            raise TypeError(
                "W8A8 functional experts require floating-point hidden states, "
                f"got {hidden_states.dtype}"
            )

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
        del workspace13, workspace2, expert_tokens_meta

        quant_config = self.quant_config
        if not quant_config.use_int8_w8a8:
            raise ValueError("FlagGemsW8A8Experts requires an INT8 W8A8 quant config")
        if not quant_config.per_act_token_quant or quant_config.block_shape is not None:
            raise ValueError(
                "FlagGemsW8A8Experts requires dynamic per-token/channel-wise INT8"
            )
        if getattr(self, "_lora_context", None) is not None:
            raise NotImplementedError(
                "The FlagGems W8A8 MoE adapter does not support LoRA"
            )
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError(
                "W8A8 activation was quantized before the FlagGems experts "
                "adapter; expected an unquantized floating-point input"
            )
        if hidden_states.dtype not in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }:
            raise TypeError(
                "W8A8 functional experts require floating-point hidden states, "
                f"got {hidden_states.dtype}"
            )

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

        result = _flaggems_fused_experts_impl(
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
        )
        output.copy_(result)


_experts = CachedOp("w8a8_experts")


class AscendW8A8DynamicExperts(mk.FusedMoEExpertsModular):
    """Native NPU experts; quantization and routing reduction stay vendor-owned."""

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

    @staticmethod
    def validate_config(moe):
        """Validate the native execution and communication contract."""
        parallel = moe.moe_parallel_config
        if moe.moe_backend not in ("auto", "triton"):
            raise NotImplementedError("Unsupported explicit W8A8 MoE backend")
        if parallel.is_sequence_parallel and not (
            parallel.use_ep
            and parallel.dp_size > 1
            and parallel.all2all_backend == "allgather_reducescatter"
        ):
            raise NotImplementedError("W8A8 MoE token splitting requires DP EP AgRs")
        if (
            moe.is_lora_enabled
            or moe.has_bias
            or parallel.enable_eplb
            or parallel.pcp_size != 1
            or parallel.use_batched_activation_format
        ):
            raise NotImplementedError(
                "W8A8 experts do not support LoRA/bias/EPLB/PCP/batched routing"
            )
        if parallel.dp_size > 1 and (
            not parallel.use_ep or parallel.all2all_backend != "allgather_reducescatter"
        ):
            raise NotImplementedError("W8A8 distributed experts require AgRs EP")
        if moe.activation.value not in ("silu", "swiglustep"):
            raise NotImplementedError("Unsupported W8A8 expert activation")
        if moe.swiglu_alpha not in (None, 1.0) or moe.swiglu_beta not in (
            None,
            0.0,
        ):
            raise NotImplementedError("Unsupported W8A8 SwiGLU alpha/beta")
        if parallel.use_ep:
            from vllm.config import get_current_vllm_config
            from vllm.platforms import current_platform

            config = get_current_vllm_config()
            if (
                current_platform.dist_backend != "hccl"
                or config.parallel_config.expert_placement_strategy != "linear"
                or config.compilation_config.pass_config.enable_sp
                or moe.num_experts % parallel.ep_size
            ):
                raise NotImplementedError(
                    "W8A8 EP requires HCCL and equal linear experts without compiler SP"
                )
        if not moe.is_act_and_mul:
            raise NotImplementedError("W8A8 experts require gated act-and-mul weights")

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        try:
            cls.validate_config(moe_config)
        except NotImplementedError as error:
            return False, str(error)
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        from vllm.platforms import current_platform

        return (
            getattr(current_platform, "vendor_name", None) == "ascend"
            and current_platform.device_type == "npu"
        )

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt8DynamicTokenSym,
            kInt8StaticChannelSym,
        )

        return (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        )

    @staticmethod
    def _supports_activation(activation):
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUSTEP)

    @staticmethod
    def _supports_parallel_config(parallel):
        agrs = (
            parallel.use_ep
            and parallel.dp_size > 1
            and parallel.all2all_backend == "allgather_reducescatter"
        )
        return not (
            parallel.enable_eplb
            or parallel.pcp_size != 1
            or parallel.use_batched_activation_format
            or (parallel.is_sequence_parallel and not agrs)
            or (parallel.dp_size > 1 and not agrs)
        )

    @staticmethod
    def _supports_batch_invariance():
        return False

    @staticmethod
    def supports_lora():
        return False

    def finalize_weight_and_reduce_impl(self):
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Prepare/AgRs must not quantize inputs through a CUDA-only kernel.
        return True

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
        return (0,), (0,), (M, K)

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
    ):
        if a1q_scale is not None or a2_scale is not None:
            raise ValueError("Dynamic W8A8 experts require floating prepare inputs")
        if (
            apply_router_weight_on_input
            or self.moe_config.is_lora_enabled
            or self.w1_bias is not None
            or self.w2_bias is not None
        ):
            raise NotImplementedError(
                "W8A8 experts require output routing and no LoRA/bias"
            )
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("Dynamic W8A8 experts require BF16/FP16 input")
        first_expert = 0
        if expert_map is not None:
            from vllm.distributed import get_ep_group

            group = get_ep_group()
            if (
                expert_map.numel() != w1.shape[0] * group.world_size
                or global_num_experts != expert_map.numel()
            ):
                raise NotImplementedError("W8A8 EP requires evenly partitioned experts")
            first_expert = group.rank_in_group * w1.shape[0]
        elif global_num_experts not in (-1, w1.shape[0]):
            raise NotImplementedError("Sharded W8A8 experts require an expert map")
        output.copy_(
            _experts(
                hidden_states,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                first_expert=first_expert,
                global_num_experts=global_num_experts,
                activation=activation.value,
                limit=self.gemm1_clamp_limit,
            )
        )


__all__ = [
    "AscendW8A8DynamicExperts",
    "FlagGemsW8A8Experts",
    "VllmFunctionalW8A8Experts",
]
