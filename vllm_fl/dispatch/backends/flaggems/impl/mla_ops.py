# SPDX-License-Identifier: Apache-2.0
"""Public FlagGems MLA/indexer providers; no numerical kernels live here."""

from functools import wraps

import logging

import torch

from .query_quant import _fp8_query_dtype, flaggems_per_token_group_quant_fp8
from vllm_fl.dispatch.capabilities import get_flaggems_vllm_operator
from vllm_fl.utils import use_flaggems_op

logger = logging.getLogger(__name__)

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
        return implementation(
            x, group_size, dtype=_fp8_query_dtype(), scale_ue8m0=use_ue8m0
        )

    return quantize


def _portable_concat_mla_q(
    ql_nope: torch.Tensor, q_pe: torch.Tensor, q_out: torch.Tensor
) -> None:
    """Eager fallback for ``concat_mla_q(ql_nope, q_pe, q_out)``.

    vLLM's flashmla_sparse calls this when q arrives as a (ql_nope, q_pe)
    tuple: it concatenates the nope and rope halves along the last dim into a
    preallocated ``q_out`` (vLLM reuses ``q_concat_buffer``). Some
    flaggems_vllm releases -- the FlagOS release images ship 0.1.0 -- do not
    provide the op at all, so fall back to a copy, which is semantically
    identical and only loses the kernel fusion.
    """
    nope = ql_nope.shape[-1]
    if ql_nope.shape[:-1] != q_out.shape[:-1] or q_pe.shape[:-1] != q_out.shape[:-1]:
        raise ValueError(
            "concat_mla_q shape mismatch: "
            f"ql_nope={tuple(ql_nope.shape)} q_pe={tuple(q_pe.shape)} "
            f"q_out={tuple(q_out.shape)}"
        )
    if q_out.shape[-1] != nope + q_pe.shape[-1]:
        raise ValueError(
            "concat_mla_q last-dim mismatch: "
            f"q_out={q_out.shape[-1]} vs {nope}+{q_pe.shape[-1]}"
        )
    q_out[..., :nope].copy_(ql_nope)
    q_out[..., nope:].copy_(q_pe)


def get_operator(name):
    if name == "concat_mla_q":
        # Newer flaggems_vllm provide the real op; prefer it when available.
        fn = get_flaggems_vllm_operator(name)
        if callable(fn):
            return fn
        logger.warning_once(
            "flag_gems_vllm has no `concat_mla_q`; using portable eager fallback"
        )
        return _portable_concat_mla_q
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

    # These providers are portable Triton kernels and do not need NVIDIA.
    # CUDA-alike vendors expose the same torch.cuda API, and Iluvatar (CoreX)
    # reports device_type == "cuda" as well -- note that PlatformFL.is_cuda()
    # also requires vendor_name == "nvidia", so it must not be the gate here.
    cuda_ok = current_platform.is_cuda_alike() or (
        getattr(current_platform, "device_type", None) == "cuda"
    )
    if not cuda_ok or not use_flaggems_op(name):
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
