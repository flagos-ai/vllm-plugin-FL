# SPDX-License-Identifier: Apache-2.0
"""Atomic K3-only binding for the pinned native and V2 sampler imports."""

import inspect

from vllm_fl.kimi_k3_config import is_kimi_k3_profile


def prepare():
    """Validate both native imports before changing either binding."""
    if not is_kimi_k3_profile():
        return
    import vllm.v1.sample.ops.topk_topp_sampler as native
    import vllm.v1.worker.gpu.sample.sampler as runner

    from vllm_fl.ops.kimi_k3.sampling import apply_top_k_top_p, baseline_filter

    targets = (native, runner)
    current = [module.apply_top_k_top_p for module in targets]
    if all(fn is apply_top_k_top_p for fn in current):
        return targets
    if (
        not all(fn is baseline_filter for fn in current)
        or baseline_filter.__module__ != "vllm.v1.sample.ops.topk_topp_sampler"
        or tuple(inspect.signature(baseline_filter).parameters) != ("logits", "k", "p")
    ):
        raise RuntimeError("K3 sampling requires the pinned native/V2 filter contract")
    return targets


def install():
    targets = prepare()
    if targets is None:
        return
    from vllm_fl.ops.kimi_k3.sampling import apply_top_k_top_p

    current = [module.apply_top_k_top_p for module in targets]
    if all(fn is apply_top_k_top_p for fn in current):
        return
    try:
        for module in targets:
            module.apply_top_k_top_p = apply_top_k_top_p
    except BaseException:
        for module, previous in zip(targets, current):
            module.apply_top_k_top_p = previous
        raise
