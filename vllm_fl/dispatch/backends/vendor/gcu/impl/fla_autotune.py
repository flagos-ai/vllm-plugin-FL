# Copyright (c) 2026 BAAI. All rights reserved.

"""Bound FLA's cold-start compilation on GCU without changing its kernels."""

import logging

import torch
from triton.runtime.autotuner import Autotuner

logger = logging.getLogger(__name__)
_patched = False


def apply_fla_autotune_gcu_patch() -> None:
    global _patched
    gcu = getattr(torch, "gcu", None)
    if _patched or gcu is None or not gcu.is_available():
        return

    from vllm.third_party.flash_linear_attention.ops import (
        chunk_scaled_dot_kkt,
        solve_tril,
        wy_fast,
    )

    from .chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu
    from .chunk_o_gcu import chunk_fwd_kernel_o

    kernels = (
        chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd_kernel,
        solve_tril.solve_tril_16x16_kernel,
        solve_tril.merge_16x16_to_32x32_inverse_kernel,
        solve_tril.merge_16x16_to_64x64_inverse_kernel,
        wy_fast.recompute_w_u_fwd_kernel,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu,
        chunk_fwd_kernel_o,
    )
    for kernel in kernels:
        tuner = kernel
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        # Upstream sweeps NVIDIA warp/pipeline combinations for every tile.
        # GDN prefill warmup executes that sweep before server readiness. Keep
        # tile selection and the original kernels, but use GCU's bounded
        # four-warp, two-stage launch schedule to avoid minutes of redundant
        # compilation per worker on an empty cache.
        configs = [
            config
            for config in tuner.configs
            if config.num_warps == 4 and config.num_stages == 2
        ]
        if not configs:
            raise RuntimeError("FLA kernel has no GCU launch configuration")
        tuner.configs = configs
    _patched = True
    logger.info("Bound GCU FLA autotuning to four warps and two stages")
