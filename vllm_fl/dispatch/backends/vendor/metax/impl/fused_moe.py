# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX fused MoE operator implementations.
"""

from typing import Any, Optional
import os
import time

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

_SGL_ZERO_BIAS_CACHE: dict[tuple[int, int, torch.dtype, torch.device], torch.Tensor] = {}
_MOE_STAGE_TRACE_COUNTS: dict[tuple[Any, ...], int] = {}


def moe_align_block_size_maca(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm._custom_ops import moe_align_block_size

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def topk_softmax_maca(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from vllm._custom_ops import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_indices


def moe_sum_maca(inp, out):
    from vllm._custom_ops import moe_sum

    moe_sum(inp, out)


def _enable_sgl_zero_bias() -> bool:
    value = os.getenv("VLLM020_SGL_ZERO_BIAS", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_bf16_noscale_moe() -> bool:
    value = os.getenv("VLLM020_MOE_BF16_NOSCALE", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_even_k_fastpath() -> bool:
    value = os.getenv("VLLM020_MOE_EVEN_K_FASTPATH", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_stage2_full_n_fastpath() -> bool:
    value = os.getenv("VLLM020_MOE_STAGE2_FULL_N_FASTPATH", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_all_experts_local_fastpath() -> bool:
    value = os.getenv("VLLM020_MOE_ALL_EXPERTS_LOCAL_FASTPATH", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _enable_stage_trace() -> bool:
    value = os.getenv("VLLM020_MOE_STAGE_TRACE", "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _naive_decode_max_m() -> int:
    try:
        return int(os.getenv("VLLM020_MOE_NAIVE_DECODE_MAX_M", "0"))
    except ValueError:
        return 0


def _stage_trace_limit() -> int:
    try:
        return int(os.getenv("VLLM020_MOE_STAGE_TRACE_MAX", "256"))
    except ValueError:
        return 256


def _stage_name(topk_weights: Optional[torch.Tensor], mul_routed_weight: bool,
                top_k: int) -> str:
    if topk_weights is None and not mul_routed_weight and top_k != 1:
        return "stage1_w1"
    if topk_weights is not None and top_k == 1:
        return "stage2_w2"
    return "unknown"


def _trace_moe_stage(
    *,
    stage: str,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    EM: int,
    top_k: int,
    mul_routed_weight: bool,
    config: dict[str, Any],
    block_size_k: int,
    even_ks: bool,
    sorted_token_ids: Optional[torch.Tensor],
    naive_decode: bool,
) -> None:
    if not _enable_stage_trace():
        return

    key = (
        stage,
        int(A.size(0)),
        int(B.size(1)),
        int(B.size(2)),
        int(EM),
        int(top_k),
        bool(mul_routed_weight),
        int(config.get("BLOCK_SIZE_M", -1)),
        int(config.get("BLOCK_SIZE_N", -1)),
        int(block_size_k),
        int(config.get("GROUP_SIZE_M", -1)),
        int(config.get("SPLIT_K", 1)),
        int(config.get("num_warps", -1)),
        int(config.get("num_stages", -1)),
        bool(even_ks),
        bool(sorted_token_ids is None),
        bool(naive_decode),
    )
    count = _MOE_STAGE_TRACE_COUNTS.get(key, 0) + 1
    _MOE_STAGE_TRACE_COUNTS[key] = count
    if count != 1:
        return
    if len(_MOE_STAGE_TRACE_COUNTS) > _stage_trace_limit():
        return

    logger.info(
        "[VLLM020_MOE_STAGE_TRACE] ts=%.6f pid=%d stage=%s M=%d N=%d K=%d "
        "C_shape=%s EM=%d top_k=%d mul_routed=%s sorted=%s cfg=%s "
        "BLOCK_SIZE_K=%d even_ks=%s naive_decode=%s",
        time.time(),
        os.getpid(),
        stage,
        A.size(0),
        B.size(1),
        B.size(2),
        tuple(C.shape),
        EM,
        top_k,
        mul_routed_weight,
        sorted_token_ids is not None,
        {
            "BLOCK_SIZE_M": config.get("BLOCK_SIZE_M"),
            "BLOCK_SIZE_N": config.get("BLOCK_SIZE_N"),
            "GROUP_SIZE_M": config.get("GROUP_SIZE_M"),
            "SPLIT_K": config.get("SPLIT_K", 1),
            "num_warps": config.get("num_warps"),
            "num_stages": config.get("num_stages"),
        },
        block_size_k,
        even_ks,
        naive_decode,
    )


def _get_sgl_zero_bias(B: torch.Tensor) -> torch.Tensor:
    key = (B.size(0), B.size(1), B.dtype, B.device)
    cached = _SGL_ZERO_BIAS_CACHE.get(key)
    if cached is None:
        cached = torch.zeros((B.size(0), B.size(1)), dtype=B.dtype, device=B.device)
        _SGL_ZERO_BIAS_CACHE[key] = cached
    return cached


def invoke_fused_moe_triton_kernel_maca(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: Optional[torch.Tensor],
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
    _topk_ids: Optional[torch.Tensor] = None,
):
    """
    MetaX implementation of invoke_fused_moe_triton_kernel using mcoplib's
    precompiled Triton kernel.  This kernel is compiled ahead-of-time for the
    MACA backend and works correctly under CUDA graph capture/replay.
    """
    from mcoplib.triton_fused_moe import fused_moe_triton_kernel

    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if (
        os.getenv("VLLM020_SGL_MOE", "0").lower() in ("1", "true", "yes", "on")
        and _topk_ids is not None
    ):
        try:
            from mcoplib.triton_fused_moe import sgl_invoke_fused_moe_kernel

            sgl_keys = {
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
                "GROUP_SIZE_M",
                "num_warps",
                "num_stages",
            }
            sgl_config = {key: value for key, value in config.items() if key in sgl_keys}
            sgl_bias = (
                _get_sgl_zero_bias(B)
                if B_bias is None and _enable_sgl_zero_bias()
                else B_bias
            )
            sgl_invoke_fused_moe_kernel(
                A,
                B,
                sgl_bias,
                C,
                A_scale,
                B_scale,
                None,
                topk_weights,
                _topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                top_k,
                sgl_config,
                compute_type,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_int4_w4a16,
                per_channel_quant,
                block_shape=block_shape,
                filter_expert=True,
            )
            return
        except Exception as err:
            logger.warning_once(
                "mcoplib sgl_invoke_fused_moe_kernel failed, fallback to "
                "fused_moe_triton_kernel: %s",
                err,
            )

    stage = _stage_name(topk_weights, mul_routed_weight, top_k)
    request_tokens = C.size(0) if C.ndim >= 2 else A.size(0)
    naive_decode = False
    naive_decode_max_m = _naive_decode_max_m()
    if (
        naive_decode_max_m > 0
        and _topk_ids is not None
        and request_tokens <= naive_decode_max_m
        and not use_fp8_w8a8
        and not use_int8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
    ):
        sorted_token_ids = None
        expert_ids = _topk_ids.reshape(-1)
        config = config.copy()
        config["BLOCK_SIZE_M"] = 1
        config["GROUP_SIZE_M"] = 1
        config["SPLIT_K"] = 1
        naive_decode = True

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            EM = min(
                sorted_token_ids.size(0),
                A.size(0) * top_k * config["BLOCK_SIZE_M"],
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )

    HAS_BIAS = B_bias is not None
    config = config.copy()
    if HAS_BIAS and config.get("SPLIT_K", 1) != 1:
        config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))
    even_ks = (
        _enable_even_k_fastpath()
        and B.size(2) % (BLOCK_SIZE_K * config.get("SPLIT_K", 1)) == 0
    )
    stage2_full_n_fastpath = (
        _enable_stage2_full_n_fastpath()
        and stage == "stage2_w2"
        and A.dtype == torch.bfloat16
        and B.dtype == torch.bfloat16
        and C.dtype == torch.bfloat16
        and A_scale is None
        and B_scale is None
        and B_bias is None
        and not use_fp8_w8a8
        and not use_int8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and config.get("SPLIT_K", 1) == 1
        and top_k == 1
        and even_ks
        and B.size(1) % config["BLOCK_SIZE_N"] == 0
    )
    if stage2_full_n_fastpath:
        logger.info_once(
            "VLLM020_MOE_STAGE2_FULL_N_FASTPATH enabled: N=%d BLOCK_SIZE_N=%d",
            B.size(1),
            config["BLOCK_SIZE_N"],
        )
    _trace_moe_stage(
        stage=stage,
        A=A,
        B=B,
        C=C,
        EM=EM,
        top_k=top_k,
        mul_routed_weight=mul_routed_weight,
        config=config,
        block_size_k=BLOCK_SIZE_K,
        even_ks=even_ks,
        sorted_token_ids=sorted_token_ids,
        naive_decode=naive_decode,
    )

    if (
        _enable_bf16_noscale_moe()
        and A.dtype == torch.bfloat16
        and B.dtype == torch.bfloat16
        and C.dtype == torch.bfloat16
        and A_scale is None
        and B_scale is None
        and B_bias is None
        and not use_fp8_w8a8
        and not use_int8_w8a8
        and not use_int8_w8a16
        and not use_int4_w4a16
        and config.get("SPLIT_K", 1) == 1
    ):
        try:
            from mcoplib.triton_fused_moe import fused_moe_triton_kernel_bf16_noscale

            noscale_config = config.copy()
            noscale_config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
            noscale_config.pop("SPLIT_K", None)
            noscale_grid = lambda META: (
                triton.cdiv(EM, META["BLOCK_SIZE_M"])
                * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
            )
            fused_moe_triton_kernel_bf16_noscale(
                noscale_grid,
                A,
                B,
                C,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                B.size(1),
                B.size(2),
                EM,
                num_tokens,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                C.stride(1),
                C.stride(2),
                naive_block_assignment=(sorted_token_ids is None),
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                FAST_F32_TO_BF16=True,
                **noscale_config,
            )
            return
        except Exception as err:
            logger.warning_once(
                "bf16 noscale fused MoE kernel failed, fallback to "
                "fused_moe_triton_kernel: %s",
                err,
            )

    fused_moe_triton_kernel(
        grid,
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        naive_block_assignment=(sorted_token_ids is None),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        E=B.size(0),
        FAST_F32_TO_BF16=True,
        EVEN_KS=even_ks,
        STAGE2_FULL_N_FASTPATH=stage2_full_n_fastpath,
        ALL_EXPERTS_LOCAL=_enable_all_experts_local_fastpath(),
        **config,
    )
