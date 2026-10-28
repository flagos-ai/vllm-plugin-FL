# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview adapters to the public FlagGems-vllm iHC operators."""

import torch

from vllm import envs
from vllm.platforms import current_platform


def triton_ihc_supported(x):
    return (
        current_platform.is_cuda()
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and not envs.VLLM_BATCH_INVARIANT
    )


def triton_ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    from flaggems_vllm.ops.ihc.ihc_pre import ihc_pre

    return ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps)


def triton_ihc_post(x, residual, post):
    from flaggems_vllm.ops.ihc.ihc_post import ihc_post

    return ihc_post(x, residual, post)
