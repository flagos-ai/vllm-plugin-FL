# Copyright (c) 2026 BAAI. All rights reserved.

"""Patches for missing ``_C``/``_moe_C`` native operator entry points."""

import functools
import logging


logger = logging.getLogger(__name__)


def patch_topk_softplus_sqrt() -> None:
    """Replace unavailable ``_moe_C`` sqrt-softplus routing on Hygon."""
    import vllm._custom_ops as custom_ops

    original = custom_ops.topk_hash_softplus_sqrt
    if getattr(original, "_vllm_fl_hygon", False):
        return

    from ..impl.native_ops.topk_softplus_sqrt import topk_softplus_sqrt_hygon

    @functools.wraps(original)
    def _topk_hash_softplus_sqrt_hygon(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
    ) -> None:
        topk_softplus_sqrt_hygon(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
        )

    _topk_hash_softplus_sqrt_hygon._vllm_fl_hygon = True
    custom_ops.topk_hash_softplus_sqrt = _topk_hash_softplus_sqrt_hygon
    logger.info("Patched topk_softplus_sqrt dispatch for Hygon")


def patch_moe_sum() -> None:
    """Replace unavailable ``_moe_C::moe_sum`` on Hygon."""
    import vllm._custom_ops as custom_ops

    original = custom_ops.moe_sum
    if getattr(original, "_vllm_fl_hygon", False):
        return

    from ..impl.native_ops.moe_sum import moe_sum_hygon_out

    @functools.wraps(original)
    def _moe_sum_hygon(input, output) -> None:
        moe_sum_hygon_out(input, output)

    _moe_sum_hygon._vllm_fl_hygon = True
    custom_ops.moe_sum = _moe_sum_hygon
    logger.info("Patched moe_sum dispatch for Hygon")


_patch_topk_softplus_sqrt = patch_topk_softplus_sqrt
_patch_moe_sum = patch_moe_sum

__all__ = [
    "patch_moe_sum",
    "patch_topk_softplus_sqrt",
]
