# SPDX-License-Identifier: Apache-2.0
"""Public FlagGems MLA/indexer providers; no numerical kernels live here."""

from functools import wraps

import torch

from .query_quant import (
    flaggems_per_token_group_quant_fp8,
    flaggems_per_token_group_quant_int8,
)
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
    "per_token_group_quant_int8",
    "indexer_k_quant_and_cache_int8",
    "int8_mqa_logits",
    "int8_paged_mqa_logits",
)


def _flaggems_query_quantizer(implementation):
    @wraps(implementation)
    def quantize(x, group_size, *, use_ue8m0):
        if torch.compiler.is_compiling():
            return flaggems_per_token_group_quant_fp8(x, group_size, use_ue8m0)
        return implementation(
            x, group_size, dtype=torch.float8_e4m3fn, scale_ue8m0=use_ue8m0
        )

    return quantize


def get_operator(name):
    if name == "per_token_group_quant_int8":
        implementation = get_flaggems_vllm_operator(name)
        if not callable(implementation):
            return None

        @wraps(implementation)
        def quantize(x, group_size, *, use_ue8m0):
            if torch.compiler.is_compiling():
                return flaggems_per_token_group_quant_int8(x, group_size, use_ue8m0)
            return implementation(x, group_size, use_ue8m0=use_ue8m0)

        return quantize
    if name == "per_token_group_quant_fp8":
        import flag_gems

        fn = getattr(flag_gems, name, None)
        return _flaggems_query_quantizer(fn) if callable(fn) else None
    if name == "mla_prefill_backend":
        from vllm_fl.attention.mla_prefill import FlagGemsMLAPrefillBackend

        return lambda config: FlagGemsMLAPrefillBackend
    if name == "fp8_fp4_paged_mqa_logits":
        graph_fn = get_flaggems_vllm_operator("fp8_fp4_paged_mqa_logits_graph")
        if callable(graph_fn):
            return graph_fn
    return get_flaggems_vllm_operator(name)


def is_available(name):
    from vllm.platforms import current_platform

    if not current_platform.is_cuda() or not use_flaggems_op(name):
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
