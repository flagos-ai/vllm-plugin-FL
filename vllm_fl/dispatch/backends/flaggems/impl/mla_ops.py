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
    """`concat_mla_q(ql_nope, q_pe, q_out)` 的 eager 兜底实现。

    vLLM 的 flashmla_sparse 在 q 是 (ql_nope, q_pe) 元组时调用它，把 nope 段与
    rope 段沿最后一维拼进预分配的 q_out（vLLM 会复用 q_concat_buffer）。
    部分 flaggems_vllm 版本（如 FlagOS release 镜像自带的 0.1.0）没有这个 op，
    这里用 copy_ 拼接，语义等价，只是少了内核融合。
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
        # 新版 flaggems_vllm 直接提供；旧版没有，退回可移植实现。
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

    # These providers are portable Triton kernels and do not need NVIDIA;
    # CUDA-alike vendors (e.g. MetaX) expose the same torch.cuda API.
    # Iluvatar (CoreX) exposes a CUDA-compatible device_type but is
    # reported as non-cuda-alike by vllm_fl.platform; the portable
    # Triton providers below do not require NVIDIA.
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
