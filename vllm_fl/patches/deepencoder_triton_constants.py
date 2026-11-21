# SPDX-License-Identifier: Apache-2.0
"""Repair a vLLM 0.32 DeepEncoder constant for the pinned Triton 3.6."""


def patch_deepencoder_triton_constants(model_config) -> bool:
    # Keep unrelated model imports and kernel globals unchanged. Unlimited-OCR
    # uses the shared DeepSeek OCR vision encoder even with its native runner.
    architectures = getattr(model_config.hf_config, "architectures", ()) or ()
    if "UnlimitedOCRForCausalLM" not in architectures:
        return False

    from vllm.model_executor.models import deepencoder
    from vllm.triton_utils import tl

    # Triton 3.6 rejects a plain Python global inside a JIT function. Explicit
    # constexpr preserves the exact value and permits compile-time folding.
    if isinstance(deepencoder.LOG2E, (float, int)):
        deepencoder.LOG2E = tl.constexpr(deepencoder.LOG2E)
        return True
    return False
