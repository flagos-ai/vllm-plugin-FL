# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Replace _scatter_num_accepted_kernel and _fill_num_accepted_kernel with
pure-torch on FlagTree.

Both kernels have early-return guards (`if req_state_idx < 0: return`) that
FlagTree's TritonXPULegalize rejects: kernels with block-terminator-less
predecessors abort compilation with
"operation with block successors must terminate its parent block".

The torch replacements reproduce the same semantics with index_put on the
req-state index buffer, which works correctly on P800.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_mamba_hybrid_scatter_torch"


def _scatter_num_accepted_torch(
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_accepted: torch.Tensor,
) -> None:
    """Pure-torch replacement for _scatter_num_accepted_kernel.

    For each batch position i:
      req_state_idx = idx_mapping[i]
      if req_state_idx >= 0:
          num_accepted[req_state_idx] = max(num_sampled[i], 1)
    """
    mask = idx_mapping >= 0
    if not mask.any():
        return
    valid_state_idx = idx_mapping[mask]
    valid_sampled = num_sampled[mask]
    values = torch.clamp(valid_sampled, min=1)
    num_accepted.index_put_((valid_state_idx,), values)


def _fill_num_accepted_torch(
    idx_mapping: torch.Tensor,
    num_accepted: torch.Tensor,
    num_sampled_scalar: int,
) -> None:
    """Pure-torch replacement for _fill_num_accepted_kernel.

    For each batch position i:
      req_state_idx = idx_mapping[i]
      if req_state_idx >= 0:
          num_accepted[req_state_idx] = num_sampled_scalar
    """
    mask = idx_mapping >= 0
    if not mask.any():
        return
    valid_state_idx = idx_mapping[mask]
    fill_val = torch.full(
        (valid_state_idx.shape[0],),
        num_sampled_scalar,
        dtype=num_accepted.dtype,
        device=num_accepted.device,
    )
    num_accepted.index_put_((valid_state_idx,), fill_val)


def apply_mamba_hybrid_compat() -> None:
    """Patch _scatter_num_accepted_kernel and _fill_num_accepted_kernel."""
    try:
        import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh
    except ImportError:
        return

    if getattr(mh, _PATCH_FLAG, False):
        return  # idempotent

    # Wrap the launchers that call these kernels: scatter_num_accepted and
    # fill_num_accepted are not module-level functions; the kernels are called
    # inline inside MambaHybridGPUContext methods. We replace the kernel
    # objects directly so the existing call sites pick up the torch path.

    orig_scatter = getattr(mh, "_scatter_num_accepted_kernel", None)
    orig_fill = getattr(mh, "_fill_num_accepted_kernel", None)

    if orig_scatter is not None:
        # Build a callable that mimics the Triton kernel's grid-launch interface
        # so existing call-sites (`_scatter_num_accepted_kernel[(num_reqs,)](...)`)
        # continue to work without modification.
        class _ScatterKernelShim:
            """Triton kernel shim: `kernel[(grid,)](args...)` → torch."""

            def __getitem__(self, _grid):
                return self._call

            def _call(
                self,
                idx_mapping_ptr,
                num_sampled_ptr,
                num_accepted_ptr,
            ):
                _scatter_num_accepted_torch(
                    idx_mapping_ptr, num_sampled_ptr, num_accepted_ptr
                )

        setattr(mh, "_scatter_num_accepted_kernel", _ScatterKernelShim())

    if orig_fill is not None:
        class _FillKernelShim:
            def __getitem__(self, _grid):
                return self._call

            def _call(
                self,
                idx_mapping_ptr,
                num_accepted_ptr,
                num_sampled_scalar,
            ):
                _fill_num_accepted_torch(
                    idx_mapping_ptr, num_accepted_ptr, int(num_sampled_scalar)
                )

        setattr(mh, "_fill_num_accepted_kernel", _FillKernelShim())

    setattr(mh, _PATCH_FLAG, True)
    logger.info(
        "kunlunxin mamba_hybrid_compat: replaced _scatter_num_accepted_kernel "
        "and _fill_num_accepted_kernel with pure-torch (FlagTree cannot lower "
        "early-return guards in these kernels)"
    )
