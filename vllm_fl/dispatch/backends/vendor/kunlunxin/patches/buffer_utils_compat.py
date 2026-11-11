# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

# --------------------------------------------------------------------------------
# Hotfix: FlagTree's XPU Triton backend cannot lower the pointer-to-pointer
# dereference in ``_apply_write_kernel`` when ``MULTI_GROUP=True``.
#
# The MoE path (Qwen3.6-35B-A3B and similar) uses ``FusedStagedWriter.apply``
# which launches the kernel with ``MULTI_GROUP=True`` — each write resolves its
# target tensor via ``_load_ptr(output_ptr + group_id, tl.int32)``, a
# pointer-load-then-cast that lowers through ``triton_xpu.gm2lm``.  FlagTree
# silently miscompiles this indirection, corrupting KV-cache writes and
# producing garbled multilingual output on image requests.
#
# The dense path (Qwen3.6-27B) uses ``StagedWriteTensor.apply_write`` with
# ``MULTI_GROUP=False`` — a direct tensor pointer, no indirection — and works
# correctly.
#
# Replace both ``apply_write`` and ``FusedStagedWriter.apply`` with pure-torch
# loops that write directly to each tensor's ``.gpu`` attribute, avoiding the
# Triton kernel entirely.  The fused writer simply iterates over its tensors and
# calls each one's ``apply_write`` individually — less efficient (N launches
# become N small torch ops) but correct.
# --------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_buffer_utils_torch"


def _make_apply_write_torch():
    """Build a pure-torch replacement for StagedWriteTensor.apply_write."""

    def apply_write_torch(self):
        n = len(self._staged_write_indices)
        if n == 0:
            return

        stride = self.gpu.stride(0) if self.gpu.dim() > 0 else 1
        flat = self.gpu.view(-1)

        for i in range(n):
            row_idx = self._staged_write_indices[i]
            start_idx = self._staged_write_starts[i]
            cu_start = self._staged_write_cu_lens[i - 1] if i > 0 else 0
            cu_end = self._staged_write_cu_lens[i]
            content = self._staged_write_contents[cu_start:cu_end]
            if not content:
                continue

            import torch

            content_tensor = torch.tensor(
                content, dtype=self.dtype, device=self.device
            )
            offset = row_idx * stride + start_idx
            flat[offset : offset + len(content)] = content_tensor

        self.clear_staged_writes()

    return apply_write_torch


def _make_fused_apply_torch():
    """Build a pure-torch replacement for FusedStagedWriter.apply.

    Each tensor's staged writes are applied individually via its own
    ``apply_write`` (now the torch version), avoiding the MULTI_GROUP=True
    pointer-indirection kernel entirely.
    """

    def fused_apply_torch(self, tensors, output_ptrs, output_strides):
        for t in tensors:
            if len(t._staged_write_indices) > 0:
                t.apply_write()

    return fused_apply_torch


def apply_buffer_utils_compat() -> None:
    """Replace Triton staged-write kernels with pure-torch loops."""
    try:
        from vllm.v1.worker.gpu import buffer_utils
    except ImportError:
        return

    current = getattr(buffer_utils, _PATCH_FLAG, False)
    if current:
        return

    import types

    # Patch StagedWriteTensor.apply_write
    apply_write_torch = _make_apply_write_torch()
    StagedWriteTensor = buffer_utils.StagedWriteTensor
    StagedWriteTensor.apply_write = apply_write_torch

    # Patch FusedStagedWriter.apply
    fused_apply_torch = _make_fused_apply_torch()
    FusedStagedWriter = buffer_utils.FusedStagedWriter
    FusedStagedWriter.apply = fused_apply_torch

    setattr(buffer_utils, _PATCH_FLAG, True)
    logger.info(
        "kunlunxin buffer_utils_compat: replaced StagedWriteTensor.apply_write "
        "and FusedStagedWriter.apply with pure-torch loops; FlagTree's XPU "
        "backend cannot lower the _load_ptr pointer-to-pointer dereference in "
        "_apply_write_kernel when MULTI_GROUP=True (MoE path)"
    )
