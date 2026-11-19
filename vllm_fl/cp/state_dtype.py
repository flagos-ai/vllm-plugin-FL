# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 storage for the Qwen3.5/3.6 GDN recurrent state cache.

Only the step-to-step SSM cache dtype changes. Prefill (FlashInfer, FLA and
the CP GDN-SP path) still upcasts the state to FP32 and runs FP32 math, and
the decode recurrence loads the state as FP32 before writing it back. The CP
operator fusions (fused_norm, fused_moe, fused_gdn) never touch this cache.

vLLM 0.20.2's --mamba-ssm-cache-dtype only accepts auto/float16/float32, so
the effective dtype is set after the Qwen3.5 config hook. P and D must agree:
the FlagCX handshake rejects mismatched GDN block lengths.
"""

import functools
import os

import torch

ENV = "VLLM_FL_GDN_STATE_BF16"


def enabled() -> bool:
    value = os.environ.get(ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def apply_gdn_state_bf16() -> None:
    from vllm.logger import init_logger
    from vllm.model_executor.models.config import (
        Qwen3_5ForConditionalGenerationConfig as config_class)

    original = config_class.verify_and_update_config
    if getattr(original, "_fl_state_bf16", False):
        return

    @functools.wraps(original)
    def verify_and_update_config(vllm_config):
        requested = vllm_config.cache_config.mamba_ssm_cache_dtype
        original(vllm_config)
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError(f"{ENV}=1 requires a BF16 model dtype")
        if vllm_config.speculative_config is not None:
            raise ValueError(f"{ENV}=1 is not validated with speculative decoding")
        if requested not in ("auto", "bfloat16"):
            raise ValueError(f"{ENV}=1 conflicts with --mamba-ssm-cache-dtype {requested}")
        vllm_config.cache_config.mamba_ssm_cache_dtype = "bfloat16"
        init_logger("vllm." + __name__).info(
            "GDN state cache: bfloat16 storage, FP32 recurrence math")

    verify_and_update_config._fl_state_bf16 = True
    config_class.verify_and_update_config = staticmethod(verify_and_update_config)
