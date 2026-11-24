# Copyright (c) 2026 BAAI. All rights reserved.

from functools import wraps

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _load_fused_topk():
    from flaggems_vllm import topk_sample_from_logits

    return topk_sample_from_logits


def enable_fused_topk_sampling(sampler):
    """Install the fused path on one sampler with vLLM's native dispatch."""
    original = sampler.forward
    if original != getattr(sampler, "forward_native", None):
        logger.info("CosyVoice3 will keep the selected vLLM sampler backend")
        return False
    try:
        fused = _load_fused_topk()
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        logger.warning("CosyVoice3 fused Top-K could not load, using native sampling: %s", exc)
        return False
    if not callable(fused):
        logger.warning("CosyVoice3 fused Top-K API is unavailable, using native sampling")
        return False

    @wraps(original)
    def forward(logits, generators, k, p):
        if (
            p is None
            and k is not None
            and not generators
            and not getattr(sampler, "use_fp64_gumbel", True)
            and sampler.logprobs_mode not in ("processed_logits", "processed_logprobs")
            and logits.ndim == 2
            and logits.dtype == torch.float32
            and logits.device.type == "musa"
            and logits.is_contiguous()
            and 1 <= logits.shape[1] <= 8192
            and k.shape == (logits.shape[0],)
            and k.dtype in (torch.int32, torch.int64)
            and k.device == logits.device
            and k.is_contiguous()
        ):
            # Draw from the same Torch RNG as vLLM's FP32 random_sample.
            noise = torch.empty_like(logits).exponential_()
            return fused(logits, k, noise), None
        return original(logits, generators, k, p)

    sampler.forward = forward
    logger.info("Enabled FlagGems-vllm fused Top-K for the CosyVoice3 sampler")
    return True
