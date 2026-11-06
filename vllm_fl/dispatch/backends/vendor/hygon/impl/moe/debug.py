# Copyright (c) 2026 BAAI. All rights reserved.

# VLLM_FL_MOE_DEBUG_ONLY_FILE
# This entire module is temporary MoE instrumentation. Remove this file and
# every matching MoE debug-only integration block before production.

"""Opt-in runtime diagnostics for the Hygon MoE path.

Enable with ``VLLM_FL_HYGON_MOE_DEBUG=1``. By default, each worker logs one
runtime sample per distinct input/weight shape. Set
``VLLM_FL_HYGON_MOE_DEBUG_SAMPLES`` to a larger value when several routing
samples per shape are needed.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_SETUP_KEYS: set[tuple[Any, ...]] = set()
_RUNTIME_COUNTS: dict[tuple[Any, ...], int] = {}


def moe_debug_enabled() -> bool:
    return os.getenv("VLLM_FL_HYGON_MOE_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_sample_limit() -> int:
    try:
        return max(1, int(os.getenv("VLLM_FL_HYGON_MOE_DEBUG_SAMPLES", "1")))
    except ValueError:
        return 1


def _rank_fields(moe_config) -> tuple[int, int, int, int, int]:
    parallel = moe_config.moe_parallel_config
    global_rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "-1")))
    return (
        global_rank,
        parallel.tp_rank,
        parallel.tp_size,
        parallel.ep_rank,
        parallel.ep_size,
    )


def _tensor_desc(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    return f"shape={tuple(tensor.shape)},dtype={tensor.dtype}"


def _mib(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024**2)


def _quant_mode(quant_config) -> str:
    if quant_config.use_int8_w8a8:
        suffix = "-dynamic-token" if quant_config.per_act_token_quant else ""
        return f"INT8-W8A8{suffix}"
    if quant_config.use_int8_w8a16:
        return "INT8-W8A16"
    if quant_config.use_fp8_w8a8:
        return "FP8-W8A8"
    if quant_config.use_fp8_w8a16:
        return "FP8-W8A16"
    return f"activation={quant_config.quant_dtype},weight={quant_config.weight_quant_dtype}"


def log_moe_setup(method, layer) -> None:
    """Log the quant method, selected experts class and resident weights once."""
    if not moe_debug_enabled():
        return

    quant_config = method.moe_quant_config
    rank, tp_rank, tp_size, ep_rank, ep_size = _rank_fields(method.moe)
    key = (
        rank,
        tp_rank,
        tuple(layer.w13_weight.shape),
        tuple(layer.w2_weight.shape),
        _quant_mode(quant_config),
        method.experts_cls.__name__,
    )
    if key in _SETUP_KEYS:
        return
    _SETUP_KEYS.add(key)

    logger.info(
        "[Hygon MoE Debug][setup] rank=%d device=%s tp=%d/%d ep=%d/%d "
        "method=%s experts_cls=%s backend=%s quant=%s "
        "w1=%s,%.2fMiB w2=%s,%.2fMiB w1_scale=%s w2_scale=%s",
        rank,
        torch.cuda.current_device(),
        tp_rank,
        tp_size,
        ep_rank,
        ep_size,
        method.__class__.__name__,
        method.experts_cls.__name__,
        getattr(method, "int8_backend", None),
        _quant_mode(quant_config),
        tuple(layer.w13_weight.shape),
        _mib(layer.w13_weight),
        tuple(layer.w2_weight.shape),
        _mib(layer.w2_weight),
        _tensor_desc(layer.w13_weight_scale),
        _tensor_desc(layer.w2_weight_scale),
    )


def log_moe_runtime(
    experts,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    gemm1_config: dict[str, Any],
    gemm2_config: dict[str, Any],
) -> None:
    """Log one representative runtime sample for each local input shape.

    Active-expert calculation copies the small top-k tensor to the host and
    therefore synchronizes the device. It is intentionally opt-in and skipped
    during graph capture.
    """
    if not moe_debug_enabled():
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except (RuntimeError, AttributeError):
        pass

    rank, tp_rank, tp_size, ep_rank, ep_size = _rank_fields(experts.moe_config)
    key = (
        rank,
        tp_rank,
        hidden_states.shape[0],
        hidden_states.shape[1],
        tuple(w1.shape),
        tuple(w2.shape),
        topk_ids.shape[1],
        _quant_mode(experts.quant_config),
    )
    sample = _RUNTIME_COUNTS.get(key, 0)
    if sample >= _runtime_sample_limit():
        return
    _RUNTIME_COUNTS[key] = sample + 1

    ids = [
        int(expert_id)
        for expert_id in topk_ids.detach().to("cpu").reshape(-1).tolist()
        if 0 <= int(expert_id) < global_num_experts
    ]
    global_counts = Counter(ids)
    active_global = len(global_counts)

    if expert_map is None:
        local_ids = [expert_id for expert_id in ids if expert_id < w1.shape[0]]
    else:
        mapping = expert_map.detach().to("cpu").reshape(-1).tolist()
        local_ids = [
            int(mapping[expert_id])
            for expert_id in ids
            if expert_id < len(mapping) and int(mapping[expert_id]) >= 0
        ]
    local_counts = Counter(local_ids)
    active_local = len(local_counts)
    max_routes = max(local_counts.values(), default=0)

    per_expert_weight_bytes = (
        w1[0].numel() * w1.element_size()
        + w2[0].numel() * w2.element_size()
    )
    active_weight_mib = active_local * per_expert_weight_bytes / (1024**2)
    intermediate_size = w1.shape[1] // 2

    logger.info(
        "[Hygon MoE Debug][runtime] rank=%d device=%s tp=%d/%d ep=%d/%d "
        "sample=%d input=%s topk=%s routes=%d active_global=%d/%d "
        "active_local=%d/%d max_routes_per_local_expert=%d quant=%s "
        "a1_scale=%s gemm1=A[%d,%d]*W[%d,%d,%d]->[%d,%d] "
        "gemm2=A[%d,%d]*W[%d,%d,%d]->[%d,%d] "
        "active_weight=%.2fMiB gemm1_config=%s gemm2_config=%s",
        rank,
        torch.cuda.current_device(),
        tp_rank,
        tp_size,
        ep_rank,
        ep_size,
        sample,
        _tensor_desc(hidden_states),
        _tensor_desc(topk_weights),
        len(ids),
        active_global,
        global_num_experts,
        active_local,
        w1.shape[0],
        max_routes,
        _quant_mode(experts.quant_config),
        _tensor_desc(a1q_scale),
        len(ids),
        hidden_states.shape[1],
        w1.shape[0],
        w1.shape[1],
        w1.shape[2],
        len(ids),
        w1.shape[1],
        len(ids),
        intermediate_size,
        w2.shape[0],
        w2.shape[1],
        w2.shape[2],
        len(ids),
        w2.shape[1],
        active_weight_mib,
        gemm1_config,
        gemm2_config,
    )


__all__ = ["log_moe_runtime", "log_moe_setup", "moe_debug_enabled"]
