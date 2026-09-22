# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU fix for ``invoke_fused_moe_triton_kernel`` / ``fused_moe_kernel``.

GCU (Libra / L600) hardware limits:
  - Grid (SP 数量): <= 48 (1-D only, no 2-D grid)
  - num_warps: <= 4
  - 禁止 int64 工作项索引，统一使用 int32

The upstream launcher uses ``grid = (num_pid_m * num_pid_n,)`` which exceeds
the GCU grid limit for large token counts (e.g. ``EM=114176`` blocks).

This patch converts the launch to GCU Fixed-Grid + strided-loop pattern:
  - grid = (min(total_pid, GCU_NUM_GRID),)   — 1-D only
  - kernel 内跨步循环 ``for pid in range(pid_start, total_pid, NUM_SPC)``
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Any

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GCU (L600 / Libra) hardware constants
# ---------------------------------------------------------------------------
# S60 板卡:  GCU_NUM_GRID = 24, GCU_MAX_DSM_MEMORY = int(1.5 * 1024 * 1024)
# Libra 板卡: GCU_NUM_GRID = 48, GCU_MAX_DSM_MEMORY = 917504 // 2
GCU_NUM_GRID = 48
GCU_MAX_DSM_MEMORY = 917504 // 2  # 448 KB
GCU_MAX_WARPS = 4

_patched = False


# ---------------------------------------------------------------------------
# GCU-compatible Triton kernel (Fixed-Grid + strided loop)
# ---------------------------------------------------------------------------


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
    naive_block_assignment: tl.constexpr,
    pid_m,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    if naive_block_assignment:
        # Naive layout: only row 0 (token slot pid_m) is valid.  Store a full
        # (BLOCK_SIZE_M, BLOCK_SIZE_N) zero tile whose M extent is clamped to
        # pid_m+1 so boundary_check keeps only block-row 0 (-> C row pid_m).
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr, shape=(pid_m + 1, N), strides=(stride_cm, stride_cn),
            offsets=(pid_m, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0),
        )
        tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
    else:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gcu(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    """GCU Fixed-Grid version of ``fused_moe_kernel``.

    Identical semantics to the upstream kernel, but uses a 1-D Fixed Grid
    plus an internal strided loop over the flattened ``pid`` space instead of
    one CTA per (m_block, n_block) tile.
    """
    pid_start = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_pid = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Hoist loop-invariant scalar load out of the strided loop.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)

    for pid in range(pid_start, total_pid, NUM_SPC):
        # -----------------------------------------------------------
        # Map program id `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # GCU Triton does not support `continue`, so guard the whole body
        # with an `if` instead of an early skip.
        if pid_m * BLOCK_SIZE_M < num_tokens_post_padded:
            if not naive_block_assignment:
                offs_token_id = pid_m * BLOCK_SIZE_M + offs
                offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
            else:
                offs_token = tl.where(
                    offs == 0,
                    pid_m,  # first element = pid_m
                    num_valid_tokens,  # remaining elements = constant
                )
            # Cast to int64 to prevent overflow in stride*offset products
            # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
            offs_token = offs_token.to(tl.int64)

            token_mask = offs_token < num_valid_tokens

            off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
            if off_experts == -1:
                # -----------------------------------------------------------
                # Write back zeros to the output when the expert is not
                # in the current expert parallel rank.
                _write_zeros_to_output(
                    c_ptr,
                    stride_cm,
                    stride_cn,
                    pid_n,
                    N,
                    offs_token,
                    token_mask,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    compute_type,
                    naive_block_assignment,
                    pid_m,
                )
            else:
                offs_bn = (
                    pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
                ) % N
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                # A tile via block pointer (naive layout): only row 0 of the
                # tile is a valid token (token slot pid_m -> A row
                # pid_m // top_k); the remaining rows are masked out at store
                # time, so loading extra rows is harmless.
                # Row bound = every row the gather could reach:
                # cdiv(num_valid_tokens, top_k).
                # The sorted-token-id layout keeps the masked row gather.
                if naive_block_assignment:
                    num_token_rows = (num_valid_tokens + top_k - 1) // top_k
                    a_block_ptr = tl.make_block_ptr(
                        base=a_ptr, shape=(num_token_rows, K),
                        strides=(stride_am, stride_ak),
                        offsets=(pid_m // top_k, 0),
                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0),
                    )
                else:
                    a_ptrs = a_ptr + (
                        offs_token[:, None] // top_k * stride_am
                        + offs_k[None, :] * stride_ak
                    )

                # B tile via block pointer: a regular block inside expert
                # ``off_experts``.  B is stored [E, N, K] (K contiguous);
                # view the expert slice as (K, N), same as the reference
                # block-scaled MM.
                b_block_ptr = tl.make_block_ptr(
                    base=b_ptr + off_experts * stride_be,
                    shape=(K, N), strides=(stride_bk, stride_bn),
                    offsets=(0, pid_n * BLOCK_SIZE_N),
                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(0, 1),
                )

                if use_int8_w8a16:
                    b_scale_ptrs = (
                        b_scale_ptr
                        + off_experts * stride_bse
                        + offs_bn[None, :] * stride_bsn
                    )
                    b_scale = tl.load(b_scale_ptrs)

                if use_fp8_w8a8 or use_int8_w8a8:
                    # block-wise
                    if group_k > 0 and group_n > 0:
                        a_scale_ptrs = (
                            a_scale_ptr + (offs_token // top_k) * stride_asm
                        )
                        offs_bsn = offs_bn // group_n
                        b_scale_ptrs = (
                            b_scale_ptr
                            + off_experts * stride_bse
                            + offs_bsn * stride_bsn
                        )
                    # channel-wise
                    elif per_channel_quant:
                        b_scale_ptrs = (
                            b_scale_ptr
                            + off_experts * stride_bse
                            + offs_bn[None, :] * stride_bsn
                        )
                        b_scale = tl.load(b_scale_ptrs)
                        # Load per-token scale for activations
                        a_scale_ptrs = (
                            a_scale_ptr + (offs_token // top_k) * stride_asm
                        )
                        a_scale = tl.load(
                            a_scale_ptrs, mask=token_mask, other=0.0
                        )[:, None]
                    # tensor-wise
                    else:
                        a_scale = tl.load(a_scale_ptr)
                        b_scale = tl.load(b_scale_ptr + off_experts)
                if HAS_BIAS:
                    # bias shape: [num_experts, N]
                    bias_ptrs = (
                        b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
                    )
                    bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
                # -----------------------------------------------------------
                # Iterate to compute a block of the C matrix.
                accumulator = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )
                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    # Load the next block of A and B.  Block pointers handle
                    # the K/N bounds via boundary_check (zero padding); the
                    # sorted layout keeps the explicit K mask on its gather.
                    if naive_block_assignment:
                        a = tl.load(
                            a_block_ptr,
                            boundary_check=(0, 1),
                            padding_option="zero",
                        )
                    else:
                        a = tl.load(
                            a_ptrs,
                            mask=token_mask[:, None]
                            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0,
                        )
                    b = tl.load(
                        b_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    if use_int8_w8a16:
                        accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
                    elif use_fp8_w8a8 or use_int8_w8a8:
                        if group_k > 0 and group_n > 0:
                            k_start = k * BLOCK_SIZE_K
                            offs_ks = k_start // group_k
                            a_scale = tl.load(
                                a_scale_ptrs + offs_ks * stride_ask,
                                mask=token_mask,
                                other=0.0,
                            )
                            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                            accumulator += (
                                tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                            )
                        else:
                            if use_fp8_w8a8:
                                # acc used to enable fp8_fast_accum
                                accumulator = tl.dot(a, b, acc=accumulator)
                            else:
                                accumulator += tl.dot(a, b)
                    else:
                        accumulator += tl.dot(a, b)
                    # Advance the ptrs to the next K block.
                    if naive_block_assignment:
                        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
                    else:
                        a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

                # Dequantization for supported quantization schemes.
                if use_int8_w8a16:
                    accumulator = accumulator * b_scale
                elif (use_fp8_w8a8 or use_int8_w8a8) and not (
                    group_k > 0 and group_n > 0
                ):
                    accumulator = accumulator * a_scale * b_scale

                # Bias addition (after dequantization).
                if HAS_BIAS:
                    accumulator += bias[None, :]

                # Router (MoE) weight multiplication.
                if MUL_ROUTED_WEIGHT:
                    moe_weight = tl.load(
                        topk_weights_ptr + offs_token,
                        mask=token_mask,
                        other=0,
                    )
                    accumulator *= moe_weight[:, None]

                # Final precision conversion.
                accumulator = accumulator.to(compute_type)

                # -----------------------------------------------------------
                # Write back the block of the output.  Naive layout: only
                # row 0 of the tile holds a valid token result (slot pid_m).
                # Store the full tile through a block pointer whose M extent
                # is clamped to pid_m+1, so boundary_check keeps only
                # block-row 0 (-> C row pid_m) and masks the padding rows.
                # This avoids a tl.sum/tl.reshape row extraction that the
                # GCU compiler cannot lower.  Sorted layout: keep the masked
                # scatter store.
                if naive_block_assignment:
                    c_block_ptr = tl.make_block_ptr(
                        base=c_ptr, shape=(pid_m + 1, N),
                        strides=(stride_cm, stride_cn),
                        offsets=(pid_m, pid_n * BLOCK_SIZE_N),
                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0),
                    )
                    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
                else:
                    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c_ptrs = (
                        c_ptr
                        + stride_cm * offs_token[:, None]
                        + stride_cn * offs_cn[None, :]
                    )
                    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# GCU-compatible launcher (replaces invoke_fused_moe_triton_kernel)
# ---------------------------------------------------------------------------


def invoke_fused_moe_triton_kernel_gcu(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
):
    """GCU-compatible version of ``invoke_fused_moe_triton_kernel``.

    Identical semantics to the upstream function, but launches ``fused_moe_kernel_gcu``
    with a GCU Fixed-Grid (1-D, <= GCU_NUM_GRID) + strided-loop pattern instead of a
    potentially oversized 1-D grid.
    """
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]

    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    # GCU hard constraint: num_warps <= 4.
    num_warps = min(int(config.pop("num_warps", 4)), GCU_MAX_WARPS)
    num_stages = int(config.pop("num_stages", 3))
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    # --- GCU Fixed-Grid + strided loop ---
    num_pid_m = triton.cdiv(EM, config["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(B.size(1), config["BLOCK_SIZE_N"])
    total_pid = num_pid_m * num_pid_n

    # 1-D Fixed Grid (GCU only supports 1-D grid).
    grid = (min(total_pid, GCU_NUM_GRID),)
    assert grid[0] <= GCU_NUM_GRID, (
        f"Grid {grid[0]} exceeds GCU_NUM_GRID {GCU_NUM_GRID}"
    )

    fused_moe_kernel_gcu[grid](
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
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SPC=GCU_NUM_GRID,
        num_warps=num_warps,
        num_stages=num_stages,
        **config,
    )


# ---------------------------------------------------------------------------
# GCU MoE launch-config override (block-wise fp8/int8)
# ---------------------------------------------------------------------------
#
# Empirically (bench_fused_moe_gcu.py, Qwen3.5-35B-A3B: E=256, top_k=8,
# w13 N=1024/K=2048, w2 N=2048/K=512), BLOCK_SIZE_M=16 + BLOCK_SIZE_N=128 is
# optimal on GCU for *every* M from decode (1) to prefill (4096):
#   * With 256 experts the per-expert token count is tiny, so a 16-row tile is
#     well filled while 64/128-row tiles compute mostly padding.  At M=1024,
#     BM=16 is ~1.4x faster than BM=64 and ~2.8x faster than BM=128.
#   * BN=128 beats BN=64 by ~1.7x (narrow tiles lose GCU DMA efficiency).
#   * num_stages 3 ~= 4; GROUP_SIZE_M=1 (too few M-blocks per expert to group).
#
# BLOCK_SIZE_M is coupled to ``moe_align_block_size``'s block_size (both read
# the same config dict), so the override MUST happen at config-selection time,
# not inside the launcher.  We wrap ``try_get_optimal_moe_config``.
def _gcu_tune_moe_config(config: Any, dtype: Any, block_shape: Any) -> Any:
    if not isinstance(config, dict):
        return config
    is_block = (
        block_shape is not None
        and len(block_shape) == 2
        and block_shape[0] > 0
        and block_shape[1] > 0
        and dtype in ("fp8_w8a8", "int8_w8a8")
    )
    if not is_block:
        return config
    config = dict(config)
    config["BLOCK_SIZE_M"] = 16
    config["BLOCK_SIZE_N"] = min(int(block_shape[0]), 128)
    config["BLOCK_SIZE_K"] = min(int(block_shape[1]), 128)
    config["GROUP_SIZE_M"] = 1
    config["num_warps"] = min(
        int(config.get("num_warps", GCU_MAX_WARPS)), GCU_MAX_WARPS
    )
    config["num_stages"] = 3
    return config


def _make_gcu_moe_config_wrapper(orig):
    """Wrap ``try_get_optimal_moe_config`` to apply the GCU tile override."""

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        cfg = orig(*args, **kwargs)
        # signature: (w1_shape, w2_shape, top_k, dtype, M, block_shape=None)
        dtype = kwargs.get("dtype")
        if dtype is None and len(args) > 3:
            dtype = args[3]
        block_shape = kwargs.get("block_shape")
        if block_shape is None and len(args) > 5:
            block_shape = args[5]
        return _gcu_tune_moe_config(cfg, dtype, block_shape)

    wrapper._gcu_patched = True
    return wrapper


def apply_fused_moe_config_gcu_patch() -> None:
    """Force GCU-optimal MoE tile config for the block-wise fp8/int8 path.

    ``try_get_optimal_moe_config`` is imported by-reference into the vllm_fl
    fused_moe module, so both the definition site and the importer binding are
    wrapped (each call site resolves its own module global at call time).
    """
    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    _MODULES = [
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm_fl.ops.fused_moe.fused_moe",
    ]
    patched_any = False
    for module_name in _MODULES:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        orig = getattr(mod, "try_get_optimal_moe_config", None)
        if orig is None or getattr(orig, "_gcu_patched", False):
            continue
        mod.try_get_optimal_moe_config = _make_gcu_moe_config_wrapper(orig)
        patched_any = True

    if patched_any:
        logger.info(
            "Patched try_get_optimal_moe_config for GCU "
            "(block-wise fp8/int8 -> BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, "
            "GROUP_SIZE_M=1, num_warps<=%d, num_stages=3)",
            GCU_MAX_WARPS,
        )


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_fused_moe_triton_kernel_gcu_patch() -> None:
    """Patch ``invoke_fused_moe_triton_kernel`` in fused_moe and all importers
    for GCU grid limits.

    ``invoke_fused_moe_triton_kernel`` is defined and called inside
    ``vllm.model_executor.layers.fused_moe.fused_moe`` (both from
    ``dispatch_fused_moe_kernel`` and the ``FusedMoE`` apply path).  Patching
    the module attribute is sufficient because both call sites resolve the
    module-global name at call time.
    """
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    # All modules known to import/use invoke_fused_moe_triton_kernel.
    _IMPORTERS: list[str] = [
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.modular_kernel",
    ]

    try:
        import sys

        all_patched = True
        for module_name in _IMPORTERS:
            try:
                mod = __import__(
                    module_name, fromlist=["invoke_fused_moe_triton_kernel"]
                )
            except ImportError:
                continue
            if hasattr(mod, "invoke_fused_moe_triton_kernel"):
                mod.invoke_fused_moe_triton_kernel = (
                    invoke_fused_moe_triton_kernel_gcu
                )
            elif module_name in sys.modules:
                # Module exists but is only partially initialized (circular
                # import): the target name is not defined yet.  Do NOT mark
                # as patched so a later apply_gcu_patches() retry can rebind.
                all_patched = False

        _patched = all_patched
        if all_patched:
            logger.info(
                "Patched invoke_fused_moe_triton_kernel for GCU "
                "(grid <= %d, 1-D only, NUM_SPC=%d, num_warps <= %d)",
                GCU_NUM_GRID,
                GCU_NUM_GRID,
                GCU_MAX_WARPS,
            )
        else:
            logger.debug(
                "invoke_fused_moe_triton_kernel patch deferred (will retry)"
            )
    except Exception as exc:
        # Circular import — will be retried by subsequent
        # apply_gcu_patches() calls from other lifecycle hooks.
        logger.debug(
            "invoke_fused_moe_triton_kernel patch deferred (will retry): %s",
            exc,
        )
