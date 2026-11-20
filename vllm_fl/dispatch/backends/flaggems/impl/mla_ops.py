# SPDX-License-Identifier: Apache-2.0
"""Public FlagGems MLA/indexer providers; no numerical kernels live here."""

from functools import wraps

import torch

from .query_quant import flaggems_per_token_group_quant_fp8
from vllm_fl.dispatch.capabilities import get_flaggems_vllm_operator
from vllm_fl.utils import use_flaggems_op

MLA_OPERATORS = (
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "flash_mla_sparse_fwd",
    "concat_and_cache_mla",
    "concat_mla_q",
    "flash_attn_varlen_func",
    "per_token_group_quant_fp8",
    "mla_prefill_backend",
)


def _flaggems_query_quantizer(implementation):
    @wraps(implementation)
    def quantize(x, group_size, *, use_ue8m0):
        if torch.compiler.is_compiling():
            return flaggems_per_token_group_quant_fp8(x, group_size, use_ue8m0)
        return implementation(x, group_size, scale_ue8m0=use_ue8m0)

    return quantize


def _metax_concat_and_cache_mla(implementation):
    @wraps(implementation)
    def cache(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
        if kv_cache_dtype == "bfloat16":
            if kv_cache.dtype != torch.bfloat16 or kv_c.dtype != torch.bfloat16:
                raise ValueError("BF16 MLA cache requires BF16 input and cache")
            # The public FlagGems-vllm API uses "auto" for matching
            # input and cache dtypes without quantization.
            kv_cache_dtype = "auto"
        return implementation(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)

    return cache


def get_operator(name):
    from vllm.platforms import current_platform

    is_metax = getattr(current_platform, "vendor_name", None) == "metax"
    if name == "per_token_group_quant_fp8":
        if is_metax:
            fn = get_flaggems_vllm_operator(name)
            if not callable(fn):
                return None

            @wraps(fn)
            def quantize(x, group_size, *, use_ue8m0):
                return fn(
                    x,
                    group_size,
                    dtype=torch.float8_e4m3fn,
                    scale_ue8m0=use_ue8m0,
                )

            return quantize
        import flag_gems

        fn = getattr(flag_gems, name, None)
        return _flaggems_query_quantizer(fn) if callable(fn) else None
    if name == "mla_prefill_backend":
        from vllm_fl.attention.mla_prefill import FlagGemsMLAPrefillBackend

        return lambda config: FlagGemsMLAPrefillBackend
    if name == "fp8_fp4_paged_mqa_logits" and not is_metax:
        graph_fn = get_flaggems_vllm_operator("fp8_fp4_paged_mqa_logits_graph")
        if callable(graph_fn):
            return graph_fn
    implementation = get_flaggems_vllm_operator(name)
    if is_metax and name == "concat_and_cache_mla" and callable(implementation):
        return _metax_concat_and_cache_mla(implementation)
    return implementation


def is_available(name):
    from vllm.platforms import current_platform

    if not use_flaggems_op(name):
        return False
    if not current_platform.is_cuda():
        if getattr(current_platform, "vendor_name", None) != "metax":
            return False
        sparse = get_flaggems_vllm_operator("flash_mla_sparse_fwd")
        supports_c550 = getattr(sparse, "supports_metax_c550", None)
        if not (callable(supports_c550) and supports_c550()):
            return False
    if name == "mla_prefill_backend":
        return is_available("flash_attn_varlen_func")
    return callable(get_operator(name))


def register_mla_operators(registry, *, allow=None):
    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

    if allow is None:
        allow = use_flaggems_op

    def bind(name):
        def call(*args, **kwargs):
            implementation = get_operator(name)
            if not callable(implementation):
                raise RuntimeError(f"Selected FlagGems operator {name} is unavailable")
            return implementation(*args, **kwargs)

        call.__name__ = name
        call._is_available = lambda: is_available(name)
        call._supports_cuda_graph = lambda: bool(
            getattr(get_operator(name), "supports_cuda_graph", False)
        )
        return call

    registry.register_many(
        [
            OpImpl(
                name,
                "default.flagos",
                BackendImplKind.DEFAULT,
                bind(name),
                priority=BackendPriority.DEFAULT,
            )
            for name in MLA_OPERATORS
            if allow(name)
        ]
    )
