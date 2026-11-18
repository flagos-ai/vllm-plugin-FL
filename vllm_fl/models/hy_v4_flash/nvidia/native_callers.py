# SPDX-License-Identifier: Apache-2.0
"""Bridge the v0.32 direct CUDA calls in an isolated Hy4 Flash W8A8 engine."""

from functools import wraps
from importlib import import_module

import torch

from vllm import _custom_ops
from vllm.model_executor.layers.fused_moe import utils as moe_utils
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops import hy4_flash_kernels as routes
from vllm_fl.ops.fused_moe.activation import apply_moe_activation as flagos_activation
from vllm_fl.utils import use_flaggems_op

_silu_and_mul = CachedOp("silu_and_mul")
_rms_norm = CachedOp("rms_norm")
_installed = False


def install():
    """Install bridges before constructing the target model or capturing graphs."""
    global _installed
    if _installed:
        return

    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

    functional_moe = import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    original_gemm = _custom_ops.cutlass_scaled_mm
    original_concat = _custom_ops.concat_mla_q
    original_cache = _custom_ops.concat_and_cache_mla
    original_align = _custom_ops.moe_align_block_size
    original_sum = _custom_ops.moe_sum
    original_quant = moe_utils.per_token_quant_int8
    original_activation = TritonExperts.activation
    original_functional_activation = functional_moe.apply_moe_activation
    original_rms = RMSNorm.forward_cuda
    original_mrope = MRotaryEmbedding.forward_cuda

    @wraps(original_gemm)
    def scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
        if (
            a.is_cuda
            and a.ndim == b.ndim == 2
            and a.dtype == b.dtype == torch.int8
            and a.stride(1) == 1
            and b.stride(0) == 1
            and b.shape[0] % 16 == b.shape[1] % 16 == b.stride(1) % 16 == 0
            and scale_a.dtype == scale_b.dtype == torch.float32
            and scale_a.is_contiguous()
            and scale_b.is_contiguous()
            and scale_a.numel() in (1, a.shape[0])
            and scale_b.numel() in (1, b.shape[1])
            and out_dtype == torch.bfloat16
        ):
            return routes.int8_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
        return original_gemm(a, b, scale_a, scale_b, out_dtype, bias)

    @wraps(original_concat)
    def concat(ql_nope, q_pe, q_out):
        if (
            ql_nope.is_cuda
            and ql_nope.dtype == torch.bfloat16
            and ql_nope.shape[-1] == 512
        ):
            return routes.concat_mla_q(ql_nope, q_pe, q_out)
        return original_concat(ql_nope, q_pe, q_out)

    @wraps(original_cache)
    def cache(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
        if (
            kv_c.is_cuda
            and kv_c.dtype == kv_cache.dtype == torch.bfloat16
            and kv_c.shape[-1] == 512
            and kv_cache_dtype == "auto"
        ):
            return routes.concat_and_cache_mla(
                kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
            )
        return original_cache(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

    @wraps(original_align)
    def align(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map=None,
        scatter_idx=None,
    ):
        if (
            topk_ids.is_cuda
            and topk_ids.dtype == torch.int32
            and topk_ids.is_contiguous()
            and topk_ids.ndim == 2
            and num_experts == 512
            and topk_ids.shape[1] == 16
        ):
            return routes.moe_align(
                topk_ids,
                num_experts,
                block_size,
                sorted_token_ids,
                experts_ids,
                num_tokens_post_pad,
                expert_map,
                scatter_idx,
            )
        return original_align(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            expert_map,
            scatter_idx,
        )

    @wraps(original_sum)
    def sum_experts(input, output, topk_ids=None, expert_map=None):
        if (
            input.is_cuda
            and input.dtype == torch.bfloat16
            and input.ndim == 3
            and input.shape[1:] == (16, 1536)
        ):
            return routes.moe_sum(input, output, topk_ids, expert_map)
        return original_sum(input, output, topk_ids, expert_map)

    @wraps(original_quant)
    def quant(x):
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.is_contiguous()
            and x.shape[1] == 1536
        ):
            return routes.expert_quant_int8(x)
        return original_quant(x)

    @wraps(original_activation)
    def activate(self, activation, output, input, **kwargs):
        if (
            self.quant_config.use_int8_w8a8
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and input.is_cuda
            and input.dtype == torch.bfloat16
            and input.ndim == 2
        ):
            output.copy_(_silu_and_mul(None, input))
            return None
        return original_activation(self, activation, output, input, **kwargs)

    @wraps(original_functional_activation)
    def activate_functional(
        activation,
        output,
        input,
        *,
        activation_config=None,
        topk_ids=None,
        expert_map=None,
        valid_token_counts=None,
    ):
        if (
            activation == MoEActivation.SILU
            and (activation_config is None or activation_config.clamp_limit is None)
            and valid_token_counts is None
            and input.is_cuda
            and input.dtype == torch.bfloat16
            and input.ndim == 2
            and input.shape[-1] == 3072
            and output.shape == (input.shape[0], 1536)
        ):
            return flagos_activation(activation, output, input)
        return original_functional_activation(
            activation,
            output,
            input,
            activation_config=activation_config,
            topk_ids=topk_ids,
            expert_map=expert_map,
            valid_token_counts=valid_token_counts,
        )

    @wraps(original_rms)
    def rms(self, x, residual=None):
        if (
            residual is None
            and self.variance_size_override is None
            and self.pass_weight
            and x.is_cuda
            and x.dtype == torch.bfloat16
        ):
            return _rms_norm(self, x, None)
        return original_rms(self, x, residual)

    @wraps(original_mrope)
    def mrope(self, positions, query, key=None, offsets=None):
        if (
            positions.ndim == 2
            and positions.shape[0] == 3
            and query.is_cuda
            and query.dtype == torch.bfloat16
            and key is not None
            and key.dtype == query.dtype
            and query.ndim == key.ndim == 2
            and offsets is None
            and self.mrope_section
            and self.rotary_dim == 64
        ):
            return routes.mrope(
                query,
                key,
                self._match_cos_sin_cache_dtype(query),
                positions,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.mrope_interleaved,
                self.is_neox_style,
            )
        return original_mrope(self, positions, query, key, offsets)

    if use_flaggems_op("hy4_flash_int8_scaled_mm"):
        _custom_ops.cutlass_scaled_mm = scaled_mm
    if use_flaggems_op("hy4_flash_concat_mla_q"):
        _custom_ops.concat_mla_q = concat
    if use_flaggems_op("hy4_flash_concat_and_cache_mla"):
        _custom_ops.concat_and_cache_mla = cache
    if use_flaggems_op("hy4_flash_moe_align"):
        _custom_ops.moe_align_block_size = align
    if use_flaggems_op("hy4_flash_moe_sum"):
        _custom_ops.moe_sum = sum_experts
    if use_flaggems_op("hy4_flash_expert_quant_int8"):
        moe_utils.per_token_quant_int8 = quant
    if use_flaggems_op("silu_and_mul"):
        TritonExperts.activation = activate
        functional_moe.apply_moe_activation = activate_functional
    if use_flaggems_op("rms_norm"):
        RMSNorm.forward_cuda = rms
    if use_flaggems_op("hy4_flash_mrope"):
        MRotaryEmbedding.forward_cuda = mrope
    _installed = True
