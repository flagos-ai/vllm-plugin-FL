# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve empty-state normalization in upstream Triton attention.

Some compiler pipelines retain the initial denominator seed when the running
maximum starts at -inf. Explicitly exclude that seed instead of relying on
its multiplication by exp(-inf). Finite attention sinks retain their seed.
"""

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def softmax_step(S, M, L):
    next_max = tl.maximum(M, tl.max(S, axis=1))
    next_max = tl.where(next_max > float("-inf"), next_max, 0.0)
    P = tl.exp(S - next_max[:, None])
    alpha = tl.exp(M - next_max)
    tile_sum = tl.sum(P, axis=1)
    next_sum = tl.where(M == float("-inf"), tile_sum, L * alpha + tile_sum)  # noqa: SIM300
    return next_max, next_sum, P, alpha


def patch_triton_attention_softmax():
    """Install before the upstream unified kernel is first compiled."""
    from vllm.v1.attention.ops import triton_unified_attention as upstream

    if upstream.softmax_step is not softmax_step:
        upstream.softmax_step = softmax_step
        logger.info("Triton attention: explicit empty-state softmax normalization")
