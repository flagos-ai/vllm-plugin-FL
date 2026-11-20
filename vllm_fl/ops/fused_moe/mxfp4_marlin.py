# SPDX-License-Identifier: Apache-2.0
"""Opt-in MXFP4 W4A16 service adapter for FlagGems-vllm's Marlin kernels.

Keep vLLM's checkpoint loaders, router, prepare/finalize and native activation.
Consume the ORIGINAL loader layout, never vLLM's Marlin-repacked weights.
The split GEMM1/activation path deliberately retains SwiGLU clamp/SiTU semantics.
"""

from __future__ import annotations

import importlib
import logging
import os
from types import MethodType

import torch

from .mxfp4_marlin_align import align_mxfp4_tokens
from .mxfp4_marlin_reduce import masked_moe_sum, safe_moe_sum
from .raw_retirement import (
    assert_retired_aliases,
    compact_raw_intermediate,
    retire_raw_parameters,
)

logger = logging.getLogger(__name__)
GROUP_SIZE = 32


def validate_raw_weights(
    w1, w2, s1, s2, *, hidden_size, num_experts, intermediate_alignment=128
):
    """Check the byte and E8M0 layouts before replacing any service kernel."""
    if w1.ndim != 3 or w1.shape[1] % 2:
        raise ValueError("MXFP4 requires contiguous gate-then-up rows")
    intermediate = w1.shape[1] // 2
    if hidden_size % 128 or intermediate % intermediate_alignment:
        raise ValueError("MXFP4 Marlin requires H and local I divisible by 128")
    expected = (
        (num_experts, 2 * intermediate, hidden_size // 2),
        (num_experts, hidden_size, intermediate // 2),
        (num_experts, 2 * intermediate, hidden_size // GROUP_SIZE),
        (num_experts, hidden_size, intermediate // GROUP_SIZE),
    )
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    for index, (tensor, shape) in enumerate(zip((w1, w2, s1, s2), expected)):
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"raw MXFP4 tensor {index}: expected contiguous {shape}")
        allowed = (torch.uint8,) if index < 2 else (torch.uint8, e8m0)
        if tensor.dtype not in allowed:
            raise ValueError("weights must be packed uint8; scales must be E8M0 bytes")
        if tensor.device != w1.device:
            raise ValueError("MXFP4 weights and scales must share a device")
        if index >= 2 and bool((tensor.view(torch.uint8) == 255).any().item()):
            raise ValueError("E8M0 NaN scale byte 255 is unsupported")
    return intermediate


def deployment_rejection(moe, activation_key=None):
    """Return a specific unsupported condition; no implicit parallelism changes."""
    parallel = moe.moe_parallel_config
    if parallel.enable_eplb:
        return "EPLB weight swaps would invalidate the packed weight cache"
    if parallel.use_batched_activation_format:
        return "batched expert dispatch is unsupported"
    if not 0 < moe.num_local_experts <= moe.num_experts:
        return "invalid global and local expert counts"
    if moe.w13_num_shards != 2 or not moe.is_act_and_mul:
        return "requires gate/up and a gated activation"
    activation = getattr(moe.activation, "value", moe.activation)
    if activation not in ("silu", "situ"):
        return f"unsupported activation {activation}"
    if moe.has_bias or moe.is_lora_enabled:
        return "bias and LoRA are unsupported"
    if activation_key is not None or getattr(moe, "use_mxfp4_w4a4_dsv4", False):
        return "activation quantization must remain unchanged; only W4A16 is supported"
    if moe.in_dtype not in (torch.bfloat16, torch.float16):
        return "requires BF16/FP16 activations"
    return None


def localize_routing(
    topk_weights,
    topk_ids,
    expert_map,
    *,
    local_experts,
    global_experts,
    return_valid_mask=False,
):
    """Map standard-format EP routes without handing invalid IDs to Gems.

    Foreign/invalid routes select a valid dummy local expert and have weight
    zero. This deliberately trades extra compute for safe local partial sums;
    the original vLLM prepare/finalize still dispatches/combines across ranks.
    """
    if topk_ids.is_cuda:
        from .mxfp4_marlin_route import localize_routing as localize_gpu

        return localize_gpu(
            topk_weights,
            topk_ids,
            expert_map,
            local_experts=local_experts,
            global_experts=global_experts,
            return_valid_mask=return_valid_mask,
        )
    if expert_map is None:
        if global_experts not in (-1, local_experts):
            raise RuntimeError("EP requires the original global-to-local expert_map")
        local_ids = topk_ids
        valid = (topk_ids >= 0) & (topk_ids < local_experts)
    else:
        if expert_map.ndim != 1 or expert_map.numel() != global_experts:
            raise RuntimeError("invalid global-to-local expert_map shape")
        in_range = (topk_ids >= 0) & (topk_ids < global_experts)
        local_ids = expert_map[topk_ids.long().clamp(0, global_experts - 1)]
        valid = in_range & (local_ids >= 0) & (local_ids < local_experts)
    result = (
        torch.where(valid, topk_weights, 0).contiguous(),
        torch.where(valid, local_ids, 0).to(topk_ids.dtype).contiguous(),
    )
    return (*result, valid) if return_valid_mask else result


def execute_split_gemms(
    gems,
    hidden,
    packed,
    topk_weights,
    topk_ids,
    output,
    activation,
    *,
    num_experts,
    apply_router_weight_on_input,
    fold_scale,
    compute_type,
    route_mask=None,
    second_block_k=128,
    gemm2=None,
    gemm1=None,
):
    """Use #4483/#4943 GEMMs, with vLLM's exact activation callback between them.

    Dependency injection also permits a CPU numerical contract test. `activation`
    writes into its first argument and uses the original vLLM activation config.
    """
    w1, w2, s1, s2 = packed
    m, h = hidden.shape
    topk = topk_ids.shape[1]
    intermediate = w1.shape[2] // 2  # packed Gems layout is (E,H/8,2I)
    if m == 0:
        output.zero_()
        return
    policy = gems._select_w4a16_mxfp4_kernel_policy(
        hidden.device, m, num_experts, topk, True
    )
    sorted_ids, expert_ids, padded = align_mxfp4_tokens(
        gems, topk_ids, num_experts, policy
    )
    first = torch.empty(
        (m * topk, 2 * intermediate), device=hidden.device, dtype=hidden.dtype
    )
    activated = torch.empty(
        (m * topk, intermediate), device=hidden.device, dtype=hidden.dtype
    )
    second = torch.empty((m, topk, h), device=hidden.device, dtype=hidden.dtype)
    common = dict(
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=padded,
        block_m=policy.block_m,
        block_size_k=128,
        group_size=32,
        compute_type=compute_type,
        swap_ab=True,
        fold_scale=fold_scale,
    )
    (gemm1 or gems._invoke_w4a16_mxfp4_moe_gemm)(
        A=hidden,
        B=w1,
        C=first,
        B_scale=s1,
        topk_weights=topk_weights if apply_router_weight_on_input else None,
        mul_routed_weight=apply_router_weight_on_input,
        top_k=topk,
        **common,
    )
    # This is vLLM's native activation, including clamp/SITU parameters.
    activation(activated, first)
    (gemm2 or gemm1 or gems._invoke_w4a16_mxfp4_moe_gemm)(
        A=activated,
        B=w2,
        C=second,
        B_scale=s2,
        topk_weights=None if apply_router_weight_on_input else topk_weights,
        mul_routed_weight=not apply_router_weight_on_input,
        top_k=1,
        **dict(common, block_size_k=second_block_k),
    )
    if route_mask is not None:
        masked_moe_sum(gems, second, route_mask, output)
    else:
        safe_moe_sum(gems, second, output)


def _make_experts_class():
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

    class FlagGemsMxfp4Experts(mk.FusedMoEExpertsModular):
        @staticmethod
        def activation_format():
            return mk.FusedMoEActivationFormat.Standard

        @staticmethod
        def _supports_current_device():
            return (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
            )

        @staticmethod
        def _supports_no_act_and_mul():
            return False

        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key):
            return weight_key == kMxfp4Static and activation_key is None

        @staticmethod
        def _supports_activation(activation):
            return getattr(activation, "value", activation) in ("silu", "situ")

        @staticmethod
        def _supports_parallel_config(parallel):
            return (
                not parallel.enable_eplb and not parallel.use_batched_activation_format
            )

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
            # Output is already top-k weighted and reduced, as in native Marlin.
            return (0,), (0,), (M, K)

        def finalize_weight_and_reduce_impl(self):
            return TopKWeightAndReduceNoOP()

        def process_weights_after_loading(self, layer):
            if getattr(layer, "_fl_mxfp4_retirement", None) is not None:
                assert_retired_aliases(layer, self.packed)
                return
            self._layer = layer
            compact_i = int(os.getenv("VLLM_FL_MXFP4_COMPACT_I", "0"))
            if compact_i:
                compact_raw_intermediate(layer, compact_i)
            self.second_block_k = 64 if layer.w13_weight.shape[1] // 2 % 128 else 128
            self.gemm2 = None
            if self.second_block_k == 64:
                from .mxfp4_marlin_k64 import invoke_down_k64

                self.gemm2 = invoke_down_k64
            self.gems = importlib.import_module("flaggems_vllm.ops.fused_marlin_moe")
            from .mxfp4_marlin_invoke import _invoke_w4a16_mxfp4_moe_gemm

            self.gemm1 = _invoke_w4a16_mxfp4_moe_gemm
            validate_raw_weights(
                layer.w13_weight,
                layer.w2_weight,
                self.w1_scale,
                self.w2_scale,
                hidden_size=self.moe_config.hidden_dim,
                num_experts=self.moe_config.num_local_experts,
                intermediate_alignment=self.second_block_k,
            )
            dtype = self.moe_config.in_dtype
            self._packed_activation_dtype = dtype
            self._packed_device = layer.w13_weight.device
            self.fold_scale = (
                dtype == torch.bfloat16
                and self.gems._e8m0_fold_safe(self.w1_scale, dtype)
                and self.gems._e8m0_fold_safe(self.w2_scale, dtype)
            )
            if self.second_block_k == 128:
                self.packed = self.gems.w4a16_mxfp4_pack(
                    layer.w13_weight,
                    layer.w2_weight,
                    self.w1_scale,
                    self.w2_scale,
                    dtype,
                    cached=True,
                    block_size_k=128,
                    fold_scale=self.fold_scale,
                )
            else:
                scale_pack = (
                    self.gems._cached_pack_scale_e8m0_fold
                    if self.fold_scale
                    else self.gems._cached_pack_scale_e8m0
                )
                self.packed = (
                    self.gems._cached_pack_w(layer.w13_weight, 128, cached=True),
                    self.gems._cached_pack_w(layer.w2_weight, 64, cached=True),
                    scale_pack(self.w1_scale, dtype, cached=True),
                    scale_pack(self.w2_scale, dtype, cached=True),
                )
            if os.getenv("VLLM_FL_MXFP4_RETIRE_RAW", "1") == "1":
                retire_raw_parameters(layer, self.packed, self.gems)
            # Pack before CUDA graph capture; never call .max().item() in forward.
            logger.warning(
                "FL_MXFP4_MARLIN_READY E=%d H=%d I=%d clamp=%s activation=%s",
                layer.w13_weight.shape[0],
                self.moe_config.hidden_dim,
                layer.w13_weight.shape[1] // 2,
                self.quant_config.gemm1_clamp_limit,
                self.moe_config.activation,
            )

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
            if getattr(self._layer, "_fl_mxfp4_retirement", None) is not None:
                assert_retired_aliases(self._layer, self.packed)
            topk_weights, topk_ids, route_mask = localize_routing(
                topk_weights,
                topk_ids,
                expert_map,
                local_experts=w1.shape[0],
                global_experts=global_num_experts,
                return_valid_mask=True,
            )
            if a1q_scale is not None or a2_scale is not None:
                raise RuntimeError("FL MXFP4 Marlin refuses activation quantization")
            if (
                hidden_states.dtype != self._packed_activation_dtype
                or hidden_states.device != self._packed_device
            ):
                raise RuntimeError(
                    "FL MXFP4 Marlin activation dtype changed after packing"
                )
            if not hidden_states.is_contiguous():
                raise RuntimeError("FL MXFP4 Marlin requires contiguous activations")
            if not getattr(self, "_fl_hit_logged", False):
                logger.warning(
                    "FL_MXFP4_MARLIN_HIT M=%d H=%d topk=%d clamp=%s",
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    topk_ids.shape[1],
                    self.quant_config.gemm1_clamp_limit,
                )
                self._fl_hit_logged = True
            compute_type = (
                self.gems.tl.bfloat16
                if hidden_states.dtype == torch.bfloat16
                else self.gems.tl.float16
            )
            execute_split_gemms(
                self.gems,
                hidden_states,
                self.packed,
                topk_weights,
                topk_ids,
                output,
                lambda out, inp: self.activation(activation, out, inp),
                num_experts=w1.shape[0],
                apply_router_weight_on_input=apply_router_weight_on_input,
                fold_scale=self.fold_scale,
                compute_type=compute_type,
                route_mask=route_mask,
                second_block_k=self.second_block_k,
                gemm2=self.gemm2,
                gemm1=self.gemm1,
            )

    return FlagGemsMxfp4Experts


def maybe_install_mxfp4_marlin(runner):
    """Call once after the upstream MoE factory, before checkpoint loading.

    VLLM_FL_MXFP4_MARLIN=1 is strict: an applicable MXFP4 layer that cannot
    preserve the deployment fails explicitly. Other quantization methods remain
    untouched. No checkpoint precision, TP, DP, EP, or routing changes are made.
    """
    if os.getenv("VLLM_FL_MXFP4_MARLIN", "0") != "1":
        return False
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    method = runner.routed_experts.quant_method
    if type(method) is not Mxfp4MoEMethod:
        return (
            False  # GPT-OSS and checkpoint-specific subclasses are separate contracts.
        )
    if getattr(method, "_fl_mxfp4_marlin_installed", False):
        return True
    reason = deployment_rejection(method.moe, method.activation_key)
    if reason:
        raise RuntimeError(f"VLLM_FL_MXFP4_MARLIN rejected deployment: {reason}")
    if method.moe_kernel is not None:
        raise RuntimeError(
            "MXFP4 adapter must be installed before weights are processed"
        )
    experts_cls = _make_experts_class()
    if not experts_cls._supports_current_device():
        raise RuntimeError("FL MXFP4 Marlin currently requires Hopper SM90")
    from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_kernel,
    )
    from vllm.model_executor.utils import is_weights_pre_processed

    def quant_config(self, layer):
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
        )

    def process(self, layer):
        if getattr(layer, "_fl_mxfp4_retirement", None) is not None:
            assert_retired_aliases(layer, self.moe_kernel.fused_experts.packed)
            return
        if is_weights_pre_processed():
            raise RuntimeError("FL MXFP4 Marlin requires raw checkpoint weights")
        validate_raw_weights(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
        )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_mxfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            mxfp4_backend=self.mxfp4_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    # Keep the original loader and all its TP slicing and E8M0 encoding intact.
    method.experts_cls = experts_cls
    method.mxfp4_backend = Mxfp4MoeBackend.MARLIN
    method.get_fused_moe_quant_config = MethodType(quant_config, method)
    method.process_weights_after_loading = MethodType(process, method)
    method._fl_mxfp4_marlin_installed = True
    return True
