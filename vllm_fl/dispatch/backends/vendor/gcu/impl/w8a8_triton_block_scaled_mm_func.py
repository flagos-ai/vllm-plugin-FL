"""
PyTorch native implementation of w8a8 block-scaled matrix multiplication.

This is a drop-in replacement for `w8a8_triton_block_scaled_mm` that uses
only PyTorch native operations (no Triton dependency). It dequantizes the
fp8 inputs using their per-block/per-token-group scales, then performs a
standard float32 matmul and casts to the desired output dtype.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def w8a8_torch_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Perform matrix multiplication with block-wise quantization.

    This function takes two fp8 input tensors ``A`` and ``B`` with their
    corresponding per-block/per-token-group scales ``As`` and ``Bs``,
    dequantizes them to float32, and returns the matmul result in the
    specified ``output_dtype``.

    Args:
        A: Input activation tensor (fp8), shape ``(..., K)``.
        B: Weight tensor (fp8), shape ``(N, K)``.
        As: Per-token-group quantization scale for ``A`` (float32),
            shape ``(..., ceil(K / block_k))``.
        Bs: Per-block quantization scale for ``B`` (float32),
            shape ``(ceil(N / block_n), ceil(K / block_k))``.
        block_size: 2-element list ``[block_n, block_k]`` specifying the
            quantization block size.
        output_dtype: Desired output dtype (default: ``torch.float16``).

    Returns:
        Result tensor with shape ``(..., N)`` and dtype ``output_dtype``.
    """
    assert len(block_size) == 2, "block_size must be [block_n, block_k]"
    block_n, block_k = block_size[0], block_size[1]

    # --- Validate shapes ---
    K = A.shape[-1]
    assert A.shape[-1] == B.shape[-1], (
        f"A and B must share the same K dimension, got {A.shape[-1]} vs {B.shape[-1]}"
    )
    assert A.shape[:-1] == As.shape[:-1], (
        f"Leading dims of A {A.shape[:-1]} must match As {As.shape[:-1]}"
    )
    assert A.is_contiguous(), "A must be contiguous"
    assert B.ndim == 2 and Bs.ndim == 2, "B and Bs must be 2D"

    N = B.shape[0]

    # --- Expand A scales and dequantize A ---
    # As has shape (..., ceil(K / block_k)).  Repeat each scale block_k times
    # along the last dimension, then slice to exact K.
    As_expanded = torch.repeat_interleave(As, block_k, dim=-1)
    if As_expanded.shape[-1] > K:
        As_expanded = As_expanded[..., :K]

    A_deq = A.float() * As_expanded  # (..., K) in fp32

    # --- Expand B scales and dequantize B ---
    # Bs has shape (ceil(N / block_n), ceil(K / block_k)).
    # Repeat block_n along dim 0, block_k along dim 1, then slice to exact N, K.
    Bs_expanded_n = torch.repeat_interleave(Bs, block_n, dim=0)
    if Bs_expanded_n.shape[0] > N:
        Bs_expanded_n = Bs_expanded_n[:N, :]

    Bs_expanded = torch.repeat_interleave(Bs_expanded_n, block_k, dim=1)
    if Bs_expanded.shape[1] > K:
        Bs_expanded = Bs_expanded[:, :K]

    B_deq = B.float() * Bs_expanded  # (N, K) in fp32

    # --- Matmul ---
    # A_deq: (..., K),  B_deq^T: (K, N)  →  C: (..., N)
    C = torch.matmul(A_deq, B_deq.T).to(output_dtype)

    return C

def _w8a8_torch_block_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return w8a8_torch_block_scaled_mm(
        qx, weight, x_scale, weight_scale, block_size, output_dtype
    )


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_w8a8_triton_block_scaled_mm_patch() -> None:
    """Patch ``_w8a8_triton_block_scaled_mm_func`` to use the PyTorch native
    implementation instead of the Triton-based one.

    The actual call path is::

        TritonFp8BlockScaledMMKernel.apply_block_scaled_mm()
          -> torch.ops.vllm.w8a8_triton_block_scaled_mm_func()  (custom op)
            -> _w8a8_triton_block_scaled_mm_func()               (module fn)

    We cannot override the custom-op impl via ``vllm_lib.impl()`` because
    ``direct_register_custom_op`` has already registered a handler for the
    ``PrivateUse1`` dispatch key.  Instead we monkey-patch the instance
    method ``apply_block_scaled_mm`` on
    :class:`~vllm.model_executor.kernels.linear.scaled_mm.triton.TritonFp8BlockScaledMMKernel`
    so that it calls our torch-native function directly, bypassing the
    custom-op altogether.
    """
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return

    try:
        from vllm.model_executor.kernels.linear.scaled_mm.triton import (
            TritonFp8BlockScaledMMKernel,
        )

        # Save original for potential fallback / inspection.
        _original_apply_block_scaled_mm = (
            TritonFp8BlockScaledMMKernel.apply_block_scaled_mm
        )

        def _patched_apply_block_scaled_mm(
            self,
            A: torch.Tensor,
            B: torch.Tensor,
            As: torch.Tensor,
            Bs: torch.Tensor,
        ) -> torch.Tensor:
            return _w8a8_torch_block_scaled_mm_func(
                A, B, As, Bs,
                list(self.weight_group_shape),
                self.config.out_dtype,
            )

        TritonFp8BlockScaledMMKernel.apply_block_scaled_mm = (
            _patched_apply_block_scaled_mm
        )

        _patched = True
        logger.info(
            "Patched TritonFp8BlockScaledMMKernel.apply_block_scaled_mm "
            "to use PyTorch native implementation for GCU"
        )
    except Exception as exc:
        logger.warning(
            "Failed to patch _w8a8_triton_block_scaled_mm_func for GCU: %s",
            exc,
        )


