# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU compatibility patch for vLLM's top-k/top-p sampler."""

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def _apply_top_k_top_p_on_cpu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    native_sampler: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Apply vLLM's existing filtering implementation to a CPU copy.

    Enflame's GCU sort/cumsum path can trigger a cold Triton autotune/compile
    for every previously unseen vocabulary shape.  Keep the filtering
    semantics in vLLM and only move this temporary working set to the CPU;
    the returned logits are copied back so softmax and random sampling remain
    on the accelerator.
    """
    device = logits.device
    dtype = logits.dtype
    cpu_logits = logits.detach().to(device="cpu", dtype=torch.float32)
    cpu_k = None if k is None else k.detach().to(device="cpu")
    cpu_p = None if p is None else p.detach().to(device="cpu")
    filtered = native_sampler(
        cpu_logits,
        cpu_k,
        cpu_p,
        allow_cpu_sync=True,
    )
    return filtered.to(device=device, dtype=dtype)


def apply_sampler_cpu_detour() -> bool:
    """Route GCU top-p filtering through CPU to avoid broken GCU sort."""
    from vllm.v1.sample.ops import topk_topp_sampler

    current = topk_topp_sampler.apply_top_k_top_p
    if getattr(current, "_vllm_fl_gcu_cpu_detour", False):
        return False

    native_sampler = topk_topp_sampler.apply_top_k_top_p_pytorch

    def apply_top_k_top_p_gcu(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        if logits.device.type != "gcu" or p is None:
            return current(logits, k, p)
        return _apply_top_k_top_p_on_cpu(logits, k, p, native_sampler)

    # Keep the marker on the wrapper, rather than module-global state, so a
    # reload or a spawned worker can safely re-run this capability check.
    apply_top_k_top_p_gcu._vllm_fl_gcu_cpu_detour = True
    topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p_gcu
    logger.info("Enabled GCU CPU detour for top-k/top-p sampling")
    return True


__all__ = ["apply_sampler_cpu_detour"]
