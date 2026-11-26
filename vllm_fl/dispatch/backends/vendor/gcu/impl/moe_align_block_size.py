# Copyright (c) 2026 BAAI. All rights reserved.

"""vLLM 0.32 ``scatter_idx`` compatibility for the GCU ``_moe_C`` build.

vLLM 0.32 adds an 8th ``scatter_idx`` argument to
``_moe_C::moe_align_block_size``; the Enflame ``_moe_C`` build still exposes
the 7-argument schema, so the original call aborts with

    RuntimeError: _moe_C::moe_align_block_size() expected at most 7 argument(s)
    but received 8 argument(s)

Drop the optional argument when it is unused (the common GCU path).  When a
caller really needs ``scatter_idx`` we fail loudly instead of silently
returning an unfilled tensor.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_PATCH_STATE = {"applied": False}


def apply_moe_align_block_size_gcu_patch() -> None:
    if _PATCH_STATE["applied"]:
        return
    try:
        import vllm._custom_ops as custom_ops

        def moe_align_block_size_compat(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            expert_map=None,
            scatter_idx=None,
        ):
            if scatter_idx is not None:
                raise NotImplementedError(
                    "GCU _moe_C::moe_align_block_size does not support scatter_idx"
                )
            # Call the 7-argument op directly: vLLM's Python wrapper always
            # appends ``scatter_idx`` positionally, even when it is None.
            return torch.ops._moe_C.moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_token_ids,
                experts_ids,
                num_tokens_post_pad,
                expert_map,
            )

        custom_ops.moe_align_block_size = moe_align_block_size_compat
        _PATCH_STATE["applied"] = True
        logger.info(
            "Patched vLLM moe_align_block_size for the GCU 7-argument _moe_C schema"
        )
    except Exception as exc:  # noqa: BLE001 - never block startup
        logger.warning("Failed to patch moe_align_block_size for GCU: %s", exc)
