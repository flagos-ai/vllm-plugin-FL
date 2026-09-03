# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU-optimized Triton kernel for W8A8 block-scaled FP8 matmul.

Replaces ``flag_gems.w8a8_block_fp8_matmul`` on GCU with a ``tl.make_block_ptr``
based kernel that lets the GCU compiler lower A/B tile loads to DMA.

VPD profile (Qwen3.5-397B-A17B-FP8, TP=8):
  _w8a8_triton_block_scaled_mm = 21.64% of GPU time (122.66s, 19200 calls)

Launch config (auto-tuned on GCU, see _select_gcu_config):
  BLOCK_SIZE_N=128 BLOCK_SIZE_K=128 GROUP_SIZE_M=64 num_warps=4 num_stages=3
  BLOCK_SIZE_M adapts to M: min(128, max(16, next_pow2(M)))  — decode win.
"""

from __future__ import annotations

import importlib
import logging

import torch
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

_patched = False

# The torch.library.Library object that owns our PrivateUse1 impl overrides.
# It MUST stay alive for the lifetime of the process: a custom-op impl's
# lifetime is tied to its Library object, so a function-local Library would be
# garbage-collected on return and silently un-register the override (the op
# would fall back to the upstream impl).  Kept at module scope on purpose.
_impl_lib: "torch.library.Library | None" = None

# ---------------------------------------------------------------------------
# GCU hardware constants (must match fused_moe.py)
# ---------------------------------------------------------------------------
GCU_NUM_GRID = 48
GCU_MAX_WARPS = 4


# ---------------------------------------------------------------------------
# Optimized Triton kernel (make_block_ptr for DMA-friendly tile loads)
# ---------------------------------------------------------------------------

@triton.jit
def _w8a8_triton_block_scaled_mm_gcu(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # GCU Fixed-Grid strided loop
    NUM_SPC: tl.constexpr,
):
    """Triton-accelerated W8A8 block-scaled matmul with make_block_ptr.

    Uses 1-D Fixed Grid + strided loop (GCU constraint: grid <= 48)
    and ``tl.make_block_ptr`` for A/B/C tile loads/stores so the GCU
    compiler can lower them to DMA.
    """
    pid_start = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_pid = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for pid in range(pid_start, total_pid, NUM_SPC):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

        # Matmul operand tiles via block pointers (lower to DMA on GCU).
        a_block_ptr = tl.make_block_ptr(
            base=A, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0),
        )
        # B is stored [N, K] (K contiguous), i.e. transposed: view as (K, N).
        b_block_ptr = tl.make_block_ptr(
            base=B, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(0, 1),
        )

        # Per-group scale indices (1-D, tiny — keep as masked gathers,
        # no DMA benefit).
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        As_ptrs = As + offs_am * stride_As_m
        Bs_ptrs = Bs + (offs_bn // group_n) * stride_Bs_n
        am_mask = offs_am < M
        bn_mask = offs_bn < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_block_ptr, boundary_check=(0, 1),
                        padding_option="zero")
            b = tl.load(b_block_ptr, boundary_check=(0, 1),
                        padding_option="zero")

            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_s = tl.load(As_ptrs + offs_ks * stride_As_k,
                          mask=am_mask, other=0.0)
            b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k,
                          mask=bn_mask, other=0.0)

            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        if C.dtype.element_ty == tl.bfloat16:
            c = accumulator.to(tl.bfloat16)
        elif C.dtype.element_ty == tl.float16:
            c = accumulator.to(tl.float16)
        else:
            c = accumulator.to(tl.float32)

        c_block_ptr = tl.make_block_ptr(
            base=C, shape=(M, N), strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0),
        )
        tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Launch-config selection (auto-tuned on GCU S60/L600, see sweep_w8a8_config.py)
# ---------------------------------------------------------------------------
#
# Empirical findings on GCU (N=4096, K=7168, block=[128,128]):
#   * BLOCK_SIZE_N=128 is always optimal — narrower N tiles (64) are ~2x
#     slower because the GCU DMA loses efficiency on short contiguous runs,
#     even though they expose more (idle) SPs.  So we keep BN=128.
#   * BLOCK_SIZE_K=128 (== group_k) so each K-iter maps to exactly one scale
#     group; larger K would straddle groups (incorrect), smaller is slower.
#   * BLOCK_SIZE_M should track M: for decode (small M) a 128-row tile wastes
#     ~8x compute on padding rows and inflates the fp32 accumulator.  Shrinking
#     BM to the next power of two >= M removes that waste (M=1: 0.291 -> 0.227
#     ms, ~22% faster; M=64: 0.295 -> 0.260 ms).  For prefill (M>=128) BM=128.
#   * num_stages=3 pipelines the K-loop best within the DSM budget.
def _select_gcu_config(M: int, group_k: int) -> dict:
    # BM = smallest power-of-two in [16, 128] that covers M.
    BLOCK_SIZE_M = min(128, max(16, triton.next_power_of_2(M)))
    BLOCK_SIZE_N = 128
    # K tile must not exceed the quant group (else a tile spans 2 groups).
    BLOCK_SIZE_K = min(128, group_k) if group_k > 0 else 128
    return {
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": BLOCK_SIZE_N,
        "BLOCK_SIZE_K": BLOCK_SIZE_K,
        "GROUP_SIZE_M": 64,
        "num_warps": GCU_MAX_WARPS,
        "num_stages": 3,
    }


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------

def _w8a8_block_scaled_mm_gcu(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the GCU-optimized W8A8 block-scaled FP8 matmul kernel.

    Args:
        A: [M, K] fp8 activation tensor.
        B: [N, K] fp8 weight tensor (transposed layout, K contiguous).
        As: [M, K//group_k] activation scales.
        Bs: [N//group_n, K//group_k] weight scales.
        block_size: [group_n, group_k] quantization block shape.
        output_dtype: output dtype (bf16/fp16/fp32).

    Returns:
        C: [M, N] output tensor.
    """
    M, K = A.shape
    N, _ = B.shape
    group_n, group_k = block_size

    C = torch.empty((M, N), dtype=output_dtype, device=A.device)

    cfg = _select_gcu_config(M, group_k)
    BLOCK_SIZE_M = cfg["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = cfg["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = cfg["BLOCK_SIZE_K"]
    GROUP_SIZE_M = cfg["GROUP_SIZE_M"]
    num_warps = cfg["num_warps"]
    num_stages = cfg["num_stages"]

    # GCU Fixed-Grid: 1-D, <= GCU_NUM_GRID.
    total_pid = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    grid = (min(total_pid, GCU_NUM_GRID),)

    _w8a8_triton_block_scaled_mm_gcu[grid](
        A, B, C, As, Bs,
        M, N, K,
        group_n, group_k,
        A.stride(0), A.stride(1),
        B.stride(1), B.stride(0),
        C.stride(0), C.stride(1),
        As.stride(0), As.stride(1),
        Bs.stride(1), Bs.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SPC=GCU_NUM_GRID,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

# Custom-op names whose PrivateUse1 impl we override with the GCU kernel.
#
#   * ``vllm::w8a8_triton_block_scaled_mm_func`` — the op that ACTUALLY runs
#     on GCU.  ``TritonFp8BlockScaledMMKernel`` is selected because its
#     ``is_supported`` only needs ``is_cuda_alike()`` (True on GCU), and it
#     dispatches to this op.  This is the one that shows up in the VPD profile
#     as ``_w8a8_triton_block_scaled_mm``.
#   * ``vllm::flaggems_fp8_block_gemm`` — kept for configs that route through
#     the FlagGems op.  NOTE: ``FlagGemsFp8BlockScaledMMLinearKernel.is_supported``
#     requires BOTH its base (FlashInfer) and fallback (DeepGemm) sub-kernels
#     to be supported, which is false on GCU, so this op is normally NOT called.
_TARGET_OPS = (
    "vllm::w8a8_triton_block_scaled_mm_func",
    "vllm::flaggems_fp8_block_gemm",
)


def _ensure_op_registered(op_name: str) -> None:
    """Import the module that defines ``op_name`` so its schema exists.

    ``torch.library.impl`` can only override an op whose schema is already
    defined.  ``w8a8_triton_block_scaled_mm_func`` is registered at import
    time of ``...scaled_mm.triton``; force that import here.
    """
    if op_name == "vllm::w8a8_triton_block_scaled_mm_func":
        importlib.import_module(
            "vllm.model_executor.kernels.linear.scaled_mm.triton"
        )


def apply_w8a8_block_scaled_mm_gcu_patch() -> None:
    """Route the GCU W8A8 block-scaled FP8 matmul to the GCU Triton kernel.

    Replaces the PrivateUse1 implementation of the block-scaled-mm custom
    op(s) with ``_w8a8_block_scaled_mm_gcu`` (a ``tl.make_block_ptr`` kernel
    the GCU compiler lowers to DMA, with an M-adaptive launch config).  The
    custom-op schema is unchanged, so ``torch.compile`` tracing still works.

    The critical target is ``vllm::w8a8_triton_block_scaled_mm_func`` — the op
    actually invoked on GCU via ``TritonFp8BlockScaledMMKernel``.  Overriding
    only the FlagGems op leaves the profiled ``_w8a8_triton_block_scaled_mm``
    kernel running, because the FlagGems kernel is filtered out at selection.
    """
    global _patched, _impl_lib
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    # Hoist the Library to module scope so the override is not GC'd (see
    # _impl_lib comment above).
    if _impl_lib is None:
        _impl_lib = torch.library.Library("vllm", "IMPL")
    patched_ops: list[str] = []
    for op_name in _TARGET_OPS:
        try:
            _ensure_op_registered(op_name)
            _impl_lib.impl(
                op_name,
                _w8a8_block_scaled_mm_gcu,
                "PrivateUse1",
                allow_override=True,
            )
            patched_ops.append(op_name)
        except Exception as exc:
            logger.warning("Failed to patch %s for GCU: %s", op_name, exc)

    if patched_ops:
        _patched = True
        logger.info(
            "Patched W8A8 block-scaled mm for GCU -> _w8a8_triton_block_scaled_mm_gcu "
            "(ops=%s; make_block_ptr, grid <= %d, NUM_SPC=%d, num_warps <= %d)",
            patched_ops,
            GCU_NUM_GRID,
            GCU_NUM_GRID,
            GCU_MAX_WARPS,
        )
