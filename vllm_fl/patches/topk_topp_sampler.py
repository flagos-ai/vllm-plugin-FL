# Copyright (c) 2026 BAAI. All rights reserved.
"""Route vLLM Top-K / Top-P sampling through FlagGems fused sort-free Triton kernels.

Dual-path fused sampling architecture:
  * Nucleus-only (``p is not None and k is None``): routed through FlagOS dispatch op
    ``top_p_sampling`` -> FlagGems 2-Pass Candidate-First Gumbel Tournament kernel
    (``top_p_sampling_from_logits``, 2 streaming passes over logits, ~0.82 ms at [64, 130k]).
  * Top-K / Joint Top-K + Top-P / Pure Gumbel-Max (``k is not None or p is None``): routed
    through FlagGems 8-Pivot IEEE-754 Radix + Philox Gumbel-Max kernel
    (``topk_topp_random_sample``).
  * In-place logit masking (when per-request generators or processed logprobs are requested):
    routed through FlagGems multi-pivot ``apply_top_k_top_p``.
"""

from __future__ import annotations

import functools
import logging
from types import ModuleType

import torch

from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

logger = logging.getLogger(__name__)


def _flagos_top_p_available() -> bool:
    try:
        import flag_gems

        return hasattr(flag_gems, "top_p_sampling_from_logits")
    except ImportError:
        return False


def _torch_available() -> bool:
    return True


def _bind(fn, avail):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = avail
    return wrapper


def top_p_sampling_flagos(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    from flag_gems.fused.top_p_sampling import top_p_sampling_from_logits

    return top_p_sampling_from_logits(logits, p)


def top_p_sampling_reference(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    from vllm.v1.sample.ops.topk_topp_sampler import (
        apply_top_k_top_p_pytorch,
        random_sample,
    )

    logits = apply_top_k_top_p_pytorch(logits, None, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, {}, False)


def register_top_p_sampling(registry) -> None:
    registry.register_many(
        [
            OpImpl(
                op_name="top_p_sampling",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=_bind(top_p_sampling_flagos, _flagos_top_p_available),
                vendor=None,
                priority=BackendPriority.DEFAULT,
            ),
            OpImpl(
                op_name="top_p_sampling",
                impl_id="reference.torch",
                kind=BackendImplKind.REFERENCE,
                fn=_bind(top_p_sampling_reference, _torch_available),
                vendor=None,
                priority=BackendPriority.REFERENCE,
            ),
        ]
    )


_top_p_sampling = CachedOp("top_p_sampling")


def patch_vllm_topk_topp_sampler(sampler_module: ModuleType | None = None) -> bool:
    """Patch ``vllm.v1.sample.ops.topk_topp_sampler`` with FlagGems dual-path fused kernels."""
    try:
        from flag_gems.fused.topk_topp_sampler import (
            apply_top_k_top_p as gems_apply_top_k_top_p,
            topk_topp_random_sample as gems_topk_topp_random_sample,
        )
    except ImportError as exc:
        logger.debug("FlagGems topk_topp_sampler unavailable: %s", exc)
        return False

    if sampler_module is None:
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_module

    if getattr(sampler_module, "_vllm_fl_topk_topp_patched", False):
        return False

    _top_p_sampling.pre_resolve()
    orig_apply = sampler_module.apply_top_k_top_p
    orig_forward_native = sampler_module.TopKTopPSampler.forward_native

    def apply_top_k_top_p_flagos(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        if p is None and k is None:
            return logits
        if logits.is_cuda:
            return gems_apply_top_k_top_p(logits, k, p)
        return orig_apply(logits, k, p)

    def forward_native_flagos(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            logits.is_cuda
            and self.logprobs_mode not in ("processed_logits", "processed_logprobs")
            and not generators
            and not self.use_fp64_gumbel
        ):
            if p is not None and k is None:
                return _top_p_sampling(logits, p), None
            return gems_topk_topp_random_sample(logits, k, p), None

        logits = apply_top_k_top_p_flagos(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return (
            sampler_module.random_sample(probs, generators, self.use_fp64_gumbel),
            logits_to_return,
        )

    sampler_module.apply_top_k_top_p = apply_top_k_top_p_flagos
    sampler_module.TopKTopPSampler.forward_native = forward_native_flagos
    sampler_module._vllm_fl_topk_topp_patched = True
    sampler_module._vllm_fl_orig_apply_top_k_top_p = orig_apply
    sampler_module._vllm_fl_orig_forward_native = orig_forward_native
    logger.info("Bridged vLLM TopKTopPSampler -> FlagGems dual-path fused sampler")
    return True


__all__ = ["patch_vllm_topk_topp_sampler", "register_top_p_sampling"]
