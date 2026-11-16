# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagGems-vllm CUDA operators for the registered NVIDIA Qwen4Exp models."""

from functools import cache

import torch
from vllm.models.qwen4_exp.common.ops import ple
from vllm.models.qwen4_exp.nvidia.ops import hc
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


@cache
def _fl_hyperconnection():
    try:
        from flaggems_vllm.ops.qwen4 import hyperconnection
    except ModuleNotFoundError as error:
        if error.name not in {"flaggems_vllm", "flaggems_vllm.ops.qwen4"}:
            raise
        return None

    return hyperconnection


@cache
def _fl_ple_fusion():
    try:
        from flaggems_vllm.ops.qwen4.ple_fusion import (
            can_use_ple_gate_norm_triton,
            ple_gate_norm,
        )
    except ModuleNotFoundError as error:
        if error.name not in {
            "flaggems_vllm",
            "flaggems_vllm.ops.qwen4",
            "flaggems_vllm.ops.qwen4.ple_fusion",
        }:
            raise
        return None
    return can_use_ple_gate_norm_triton, ple_gate_norm


def _grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int
) -> torch.Tensor:
    fl = _fl_hyperconnection()
    if fl is not None and x.shape[-1] == weight.numel() and fl.can_use_hc_triton(x, weight):
        return fl.qwen4_grouped_gemma_rmsnorm(x, weight, groups, eps)
    return hc.grouped_gemma_rmsnorm(x, weight, eps, groups)


def _hc_gate_mix(x: torch.Tensor, logits: torch.Tensor, groups: int) -> torch.Tensor:
    fl = _fl_hyperconnection()
    if fl is not None and fl.can_use_hc_triton(x, logits):
        return fl.qwen4_hc_gate_reduce(logits, x, groups)
    return hc.hc_gate_mix(x, logits, groups)


def _hc_combine(
    residual: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor | None,
    groups: int,
) -> torch.Tensor:
    fl = _fl_hyperconnection()
    if (
        fl is not None
        and injection is not None
        and fl.can_use_hc_inject_triton(injection, block, residual)
    ):
        return fl.qwen4_hc_inject_combine(injection, block, residual, groups)
    return hc.hc_combine(residual, block, injection, groups)


def _hc_combine_norm(
    residual: torch.Tensor,
    block: torch.Tensor,
    injection: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fl = _fl_hyperconnection()
    supports_combine_norm = getattr(fl, "can_use_hc_combine_norm_triton", None)
    if (
        supports_combine_norm is not None
        and injection is not None
        and supports_combine_norm(injection, block, residual, weight)
    ):
        return fl.qwen4_hc_combine_norm(residual, block, injection, weight, eps, groups)
    return hc.hc_combine_norm(residual, block, injection, weight, eps, groups)


def _ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key: torch.Tensor,
    norm_query: torch.Tensor,
    norm_conv: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fusion = _fl_ple_fusion()
    if fusion is not None and current_platform.vendor_name == "thead":
        # The merged projection produces row-strided K and V views. The
        # library kernel accepts dense rows.
        key, value, hidden = key.contiguous(), value.contiguous(), hidden.contiguous()
    if fusion is not None and key.is_contiguous() and hidden.is_contiguous():
        can_use_ple_gate_norm_triton, ple_gate_norm = fusion
        groups = hidden.shape[-1] // value.shape[-1]
        grouped_key = key.unflatten(-1, (groups, value.shape[-1]))
        query = hidden.unflatten(-1, (groups, value.shape[-1]))
        if can_use_ple_gate_norm_triton(
            grouped_key, query, value, norm_key, norm_query, norm_conv, groups
        ):
            gated, normalized = ple_gate_norm(
                grouped_key, query, value, norm_key, norm_query, norm_conv, groups, eps
            )
            return gated.flatten(-2), normalized.flatten(-2)
    return ple.ple_gate(key, value, hidden, norm_key, norm_query, norm_conv, eps)


for op_name, implementation, fake in (
    ("fl_qwen4_grouped_norm", _grouped_gemma_rmsnorm, hc._same_shape_fake),
    ("fl_qwen4_hc_gate_mix", _hc_gate_mix, hc._hc_gate_mix_fake),
    ("fl_qwen4_hc_combine", _hc_combine, hc._hc_combine_fake),
    ("fl_qwen4_hc_combine_norm", _hc_combine_norm, hc._hc_combine_norm_fake),
    ("fl_qwen4_ple_gate", _ple_gate, ple._ple_gate_fake),
):
    direct_register_custom_op(
        op_name=op_name,
        op_func=implementation,
        mutates_args=[],
        fake_impl=fake,
    )

grouped_gemma_rmsnorm = torch.ops.vllm.fl_qwen4_grouped_norm
hc_gate_mix = torch.ops.vllm.fl_qwen4_hc_gate_mix
hc_combine = torch.ops.vllm.fl_qwen4_hc_combine
hc_combine_norm = torch.ops.vllm.fl_qwen4_hc_combine_norm
ple_gate = torch.ops.vllm.fl_qwen4_ple_gate
