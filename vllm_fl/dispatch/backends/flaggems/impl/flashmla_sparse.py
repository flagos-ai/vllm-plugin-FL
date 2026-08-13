# Copyright (c) 2026 BAAI. All rights reserved.

import torch


def flash_mla_sparse_fwd_flaggems(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the FlagGems Sparse MLA prefill kernel."""
    from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd

    return flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        topk_length=topk_length,
    )[0]
