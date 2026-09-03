# Copyright (c) 2026 BAAI. All rights reserved.
"""Eager-mode wiring for the AscendC ``matmul_allreduce_add_rmsnorm`` MC2 op.

The op fuses ``x @ weight.T -> TP all-reduce (as ReduceScatter+AllGather
pipeline) -> + residual -> RMSNorm`` into a single launch, replacing the
``RowParallelLinear`` internal all-reduce followed by a separate
``npu_add_rms_norm_bias`` call. It targets the non-overlapped HCCL
all-reduce that shows up in eager decode (~18.5% of step time on
Qwen3.6-35B-A3B TP4).

Enabled via ``VLLM_FL_ENABLE_MM_AR_RMSNORM=1`` (default off). Falls back to
the unfused path when disabled or when TP == 1.

Reference usage: the npugraph_ex inductor pass
``allreduce_rmsnorm_fusion_pass.py`` (same call convention: is_trans_b=True,
is_gather_add_out=False).
"""

import os

import torch
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.utils.torch_utils import direct_register_custom_op

_ENABLED = None
_TP_CTX = None  # (hccl_group_name, tp_size, tp_rank)

# Only fuse when the step has enough tokens: the RS+AG pipeline inside the
# op pays off on large M (chunked prefill, high-concurrency), while at small
# M (e.g. single-token decode) the extra add_out all-gather cancels the win.
# Same threshold as the npugraph_ex pass (ALLREDUCE_NORM_FUSE_THRESHOLD).
MM_AR_RMSNORM_MIN_TOKENS = int(
    os.environ.get("VLLM_FL_MM_AR_RMSNORM_MIN_TOKENS", "512"))


def mm_ar_rmsnorm_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = (os.environ.get("VLLM_FL_ENABLE_MM_AR_RMSNORM", "0") == "1")
        # Logging is skipped under torch.compile tracing: Dynamo does not
        # support logger calls in non-export cases and would error out.
        tracing = torch.compiler.is_compiling()
        if _ENABLED and get_tensor_model_parallel_world_size() <= 1:
            if not tracing:
                logger.info("mm_allreduce_rmsnorm fusion needs TP > 1, disabled")
            _ENABLED = False
        if _ENABLED and not tracing:
            logger.info(
                "mm_allreduce_rmsnorm eager fusion enabled "
                "(VLLM_FL_ENABLE_MM_AR_RMSNORM=1)")
    return _ENABLED


def _tp_ctx():
    global _TP_CTX
    if _TP_CTX is None:
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        rank = torch.distributed.get_rank(group=device_group)
        _TP_CTX = (
            backend.get_hccl_comm_name(rank),
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )
    return _TP_CTX


def fused_mm_allreduce_add_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``x @ weight.T`` + TP all-reduce + residual add + RMSNorm fused.

    Args:
        x: [M, K] activation (pre-projection), bf16/fp16.
        weight: [N, K] projection weight (nn.Linear layout, is_trans_b=True).
        residual: [M, N] residual to add after the all-reduce.
        gamma: [N] RMSNorm weight. For GemmaRMSNorm semantics pass
            ``1.0 + weight`` (same convention as AscendCGemmaRMSNorm).
        epsilon: RMSNorm epsilon.

    Returns:
        (normed_output [M, N], new_residual [M, N]).
    """
    group_name, tp_size, tp_rank = _tp_ctx()
    logger.info_once(
        "matmul_allreduce_add_rmsnorm fused call: M=%d K=%d N=%d",
        x.shape[0], x.shape[1], weight.shape[0])
    return torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
        x,
        weight,
        residual,
        gamma,
        group_name,
        tp_size,
        tp_rank,
        epsilon,
        True,   # is_trans_b: nn.Linear weight layout [N, K]
        # is_gather_add_out: without this the returned new-residual is only
        # the reduce-scattered partial chunk (rest is garbage); the op then
        # runs an extra all-gather so add_out is complete on every rank.
        True,
    )


def fused_mm_allreduce_add_rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(residual), torch.empty_like(residual)


# Registered as an opaque custom op so Dynamo can trace call sites inside
# compiled (piecewise) regions: the eager body touches ProcessGroup/HCCL
# internals (torch.device, get_hccl_comm_name) and logging, none of which
# are traceable.
if not hasattr(torch.ops.vllm, "fused_mm_allreduce_add_rmsnorm"):
    direct_register_custom_op(
        op_name="fused_mm_allreduce_add_rmsnorm",
        op_func=fused_mm_allreduce_add_rmsnorm,
        fake_impl=fused_mm_allreduce_add_rmsnorm_fake,
        mutates_args=[],
        dispatch_key="PrivateUse1",
    )
