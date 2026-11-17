# SPDX-License-Identifier: Apache-2.0
"""Flash routing with explicit mask capabilities and no execution-time retry.

An absent mask can use the pinned native ABI8. Every supplied mask, including
all-false masks and all-padding batches, is dispatched to the padded FlagGems
implementation. ABI8 is never used with a supplied mask.
"""

import torch

from vllm.logger import init_logger

from vllm_fl.dispatch import CachedOp

_grouped_topk = CachedOp("grouped_topk")
_native_none = torch.ops._moe_C.grouped_topk.default
native_grouped_topk = _native_none
if len(_native_none._schema.arguments) != 8:
    raise RuntimeError(
        "Flash FlagOS requires the pinned vLLM0.32 ABI8; mixed MoE extensions are unsupported"
    )
logger = init_logger(__name__)
_native_reported = False


def grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
    is_padding=None,
):
    global _native_reported
    if is_padding is None and scores.shape[0] > 0:
        if not _native_reported:
            logger.info(
                "Hy4 Flash grouped_topk: native vLLM0.32 ABI8, absent mask only"
            )
            _native_reported = True
        return _native_none(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )
    return _grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
        is_padding=is_padding,
    )
