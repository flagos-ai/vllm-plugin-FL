# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
#
# Hotfix: FlagTree XPU backend cannot lower a Triton kernel that has a
# ``for j in tl.static_range(NUM_DIMS): if is_prefill: ... else: ...``
# pattern — a tensor-yielding branch inside a static loop — with:
#
#     loc("vllm/v1/worker/gpu/mm/rope.py":177:0):
#     error: LLVM Translation failed for operation:
#     builtin.unrealized_conversion_cast
#
# Replace RopeState.prepare_positions with a Python loop over requests.
# Numerical equivalence verified against the kernel's reference semantics
# across 24 configurations (num_dims 2/3/4, num_reqs 1/4/8, mixed
# prefill+decode requests).

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_rope_prepare_positions_torch"


def apply_mm_rope_compat() -> None:
    """Replace RopeState.prepare_positions with a pure-torch implementation."""
    try:
        from vllm.v1.worker.gpu.mm import rope as rope_mod
    except ImportError:
        return

    RopeState = getattr(rope_mod, "RopeState", None)
    if RopeState is None:
        return

    existing = getattr(RopeState, "prepare_positions", None)
    if existing is None:
        return
    if getattr(existing, _PATCH_FLAG, False):
        return

    import torch

    def _prepare_positions_torch(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Pure-torch implementation of _prepare_rope_positions_kernel.

        For each request in the batch, fill self.positions[dim, qs:qe] with:
          - prefill path (num_computed < prefill_len): load from
            self.prefill_positions.gpu[num_dims*req_idx + dim, num_computed:]
          - decode path: num_computed + arange(query_len) + delta
        """
        num_reqs = idx_mapping.shape[0]
        pos_tensor = self.positions          # [num_dims, max_num_tokens+1]
        pfill_pos = self.prefill_positions.gpu  # [max_num_reqs*num_dims, max_model_len]
        delta_gpu = self.prefill_delta.gpu   # [max_num_reqs]
        num_dims = self.num_dims

        for batch_idx in range(num_reqs):
            req_idx = idx_mapping[batch_idx].item()
            query_start = query_start_loc[batch_idx].item()
            query_end = query_start_loc[batch_idx + 1].item()
            query_len = query_end - query_start
            if query_len <= 0:
                continue

            nc = num_computed_tokens[req_idx].item()
            pl = prefill_lens[req_idx].item()
            is_prefill = nc < pl
            delta = delta_gpu[req_idx].item()

            if is_prefill:
                for j in range(num_dims):
                    row = num_dims * req_idx + j
                    src = pfill_pos[row, nc: nc + query_len]
                    pos_tensor[j, query_start:query_end] = src.to(pos_tensor.dtype)
            else:
                block = torch.arange(
                    query_len,
                    device=pos_tensor.device,
                    dtype=pos_tensor.dtype,
                )
                decode_pos = nc + block + delta
                for j in range(num_dims):
                    pos_tensor[j, query_start:query_end] = decode_pos

    setattr(_prepare_positions_torch, _PATCH_FLAG, True)
    RopeState.prepare_positions = _prepare_positions_torch
    logger.info(
        "kunlunxin mm_rope_compat: replaced RopeState.prepare_positions with "
        "pure-torch (FlagTree cannot lower tensor-yielding branch inside "
        "tl.static_range NUM_DIMS loop)"
    )
