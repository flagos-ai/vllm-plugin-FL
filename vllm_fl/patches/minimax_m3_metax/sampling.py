# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""One model-scoped hook over the already-installed MetaX sampling fallback."""

import inspect

from vllm_fl.minimax_m3_config import is_enabled


def install():
    if not is_enabled():
        return
    import vllm.v1.sample.ops.topk_topp_sampler as native

    from vllm_fl.ops.minimax_m3.sampling import apply_top_k_top_p

    current = native.apply_top_k_top_p
    if current is apply_top_k_top_p:
        return
    if (
        not callable(current)
        or tuple(inspect.signature(current).parameters) != ("logits", "k", "p")
        or current.__module__
        != "vllm_fl.dispatch.backends.vendor.metax.patches.topk_topp_sampler"
    ):
        raise RuntimeError("MiniMax sampling requires the pinned MetaX fallback")
    # Validate everything before the single assignment; installation is atomic.
    native.apply_top_k_top_p = apply_top_k_top_p
