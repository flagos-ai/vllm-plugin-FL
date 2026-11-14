# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Use vLLM's PyTorch top-k/top-p filtering on Kunlunxin."""

import logging

logger = logging.getLogger(__name__)


def _xpu_sampler_not_supported() -> bool:
    # The fused XPU sampler bypasses the PyTorch filter.
    return False


def apply_topk_topp_compat() -> None:
    from vllm.v1.sample.ops import topk_topp_sampler
    from vllm.v1.worker.gpu.sample import sampler as v2_sampler

    # Keep the dispatcher object that Runner V2 already imported. Its module
    # flag also controls registration of the incompatible kernel's warmups.
    topk_topp_sampler.HAS_TRITON = False
    topk_topp_sampler.xpu_sampler_supported = _xpu_sampler_not_supported
    v2_sampler.xpu_sampler_supported = _xpu_sampler_not_supported
    logger.info("Enabled vLLM's PyTorch top-k/top-p filtering on Kunlunxin")
