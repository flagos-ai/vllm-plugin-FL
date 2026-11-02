# SPDX-License-Identifier: Apache-2.0
"""Native MLA/indexer entry points and their actual execution capabilities."""

from vllm_fl.dispatch.capabilities import has_device_kernel

_CACHE_OPS = {
    "concat_and_cache_mla",
    "concat_mla_q",
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
}
_TOPK_OPS = {"top_k_per_row_prefill", "top_k_per_row_decode"}


def get_operator(name):
    if name in _CACHE_OPS | _TOPK_OPS:
        from vllm import _custom_ops

        return getattr(_custom_ops, name, None)
    if name == "per_token_group_quant_fp8":
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        return getattr(fp8_utils, name, None)
    if name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"):
        from vllm.utils import deep_gemm

        return getattr(deep_gemm, name, None)
    if name == "flash_mla_sparse_fwd":
        from vllm.v1.attention.ops import flashmla

        return getattr(flashmla, name, None)
    if name == "mla_prefill_backend":
        from vllm.v1.attention.backends.mla.prefill.selector import (
            get_mla_prefill_backend,
        )

        return get_mla_prefill_backend
    raise KeyError(name)


def is_available(name, implementation):
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or not callable(implementation):
        return False
    if name in _CACHE_OPS:
        return has_device_kernel("_C_cache_ops::" + name, "cuda")
    if name in _TOPK_OPS:
        return has_device_kernel("_C::" + name, "cuda")
    if name == "per_token_group_quant_fp8":
        return has_device_kernel("_C::per_token_group_fp8_quant", "cuda")
    if name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"):
        from vllm.utils.deep_gemm import has_deep_gemm

        return has_deep_gemm()
    if name == "flash_mla_sparse_fwd":
        from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

        return is_flashmla_sparse_supported()[0]
    return True  # Configuration-specific prefill availability is checked before resolution.


def register_mla_operators(registry):
    from functools import wraps

    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

    names = sorted(_CACHE_OPS | _TOPK_OPS) + [
        "per_token_group_quant_fp8",
        "fp8_fp4_mqa_logits",
        "fp8_fp4_paged_mqa_logits",
        "flash_mla_sparse_fwd",
        "mla_prefill_backend",
    ]
    implementations = []
    for name in names:
        # Capture before any model-specific compatibility installation.
        implementation = get_operator(name)
        if not callable(implementation):
            continue

        def bind(op_name, fn):
            @wraps(fn)
            def call(*args, **kwargs):
                return fn(*args, **kwargs)

            call._is_available = lambda: is_available(op_name, fn)
            return call

        implementations.append(
            OpImpl(
                name,
                "vendor.cuda",
                BackendImplKind.VENDOR,
                bind(name, implementation),
                vendor="cuda",
                priority=BackendPriority.VENDOR,
            )
        )
    registry.register_many(implementations)
