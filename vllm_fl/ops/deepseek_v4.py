# SPDX-License-Identifier: Apache-2.0
"""OpManager frontends for DeepSeek-V4 model-specific compute."""

from __future__ import annotations

from vllm_fl.dispatch import CachedOp

_OPS = {
    name: CachedOp(f"deepseek_v4_{name}")
    for name in (
        "inv_rope_quant_fp8",
        "int8_scaled_mm",
        "mhc_pre",
        "mhc_fused_post_pre",
        "mhc_post",
        "hc_head",
    )
}


def inv_rope_quant_fp8(*args, **kwargs):
    return _OPS["inv_rope_quant_fp8"](*args, **kwargs)


def int8_scaled_mm(*args, **kwargs):
    return _OPS["int8_scaled_mm"](*args, **kwargs)


def mhc_pre(*args, **kwargs):
    return _OPS["mhc_pre"](*args, **kwargs)


def mhc_fused_post_pre(*args, **kwargs):
    return _OPS["mhc_fused_post_pre"](*args, **kwargs)


def mhc_post(*args, **kwargs):
    return _OPS["mhc_post"](*args, **kwargs)


def hc_head(*args, **kwargs):
    return _OPS["hc_head"](*args, **kwargs)
