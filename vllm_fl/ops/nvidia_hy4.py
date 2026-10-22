# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep HY V4's upstream layers and runner while supplying EMPTY kernels."""

import torch

from vllm.platforms import current_platform

from vllm_fl.nvidia_compat import has_native_vllm_kernels
from vllm_fl.ops.nvidia_dsv41_indexer import _validate_fp8_indexer_cache
from vllm_fl.ops.nvidia_sparse_indexer import register_nvidia_sparse_indexer

_libs: list[torch.library.Library] = []


def _is_hy4(config):
    model = config.model_config
    return model is not None and "HYV4ForCausalLM" in model.architectures


def quantize_indexer_query(
    x,
    group_size,
    eps=1e-10,
    dtype=None,
    column_major_scales=False,
    tma_aligned_scales=False,
    out_q=None,
    use_ue8m0=None,
):
    from flag_gems import per_token_group_quant_fp8

    if (
        x.ndim != 2
        or x.shape[-1] != 128
        or group_size != 128
        or not x.is_cuda
        or column_major_scales
        or tma_aligned_scales
        or use_ue8m0 is not True
        or dtype not in (None, torch.float8_e4m3fn)
    ):
        raise NotImplementedError(
            "HY V4 query quantization requires row-major FP8/E8M0 groups of 128"
        )
    quantized, scales = per_token_group_quant_fp8(
        x.contiguous(), group_size, eps=eps, dtype=dtype, scale_ue8m0=True
    )
    if out_q is not None:
        if (
            out_q.shape != quantized.shape
            or out_q.dtype != quantized.dtype
            or out_q.device != quantized.device
        ):
            raise ValueError("Query quantization output buffer must match FP8 result")
        out_q.copy_(quantized)
        quantized = out_q
    return quantized, scales


def indexer_k_quant_and_cache(
    k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
):
    from flag_gems import indexer_k_quant_and_cache as gems_insert

    _validate_fp8_indexer_cache(kv_cache)
    if (
        k.ndim != 2
        or k.shape[-1] != 128
        or quant_block_size != 128
        or kv_cache_dtype != "ue8m0"
        or slot_mapping.ndim != 1
        or slot_mapping.numel() > k.shape[0]
        or not slot_mapping.is_contiguous()
        or slot_mapping.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "HY V4 indexer cache requires 128-dim keys and contiguous integer slots"
        )
    if any(not t.is_cuda or t.device != k.device for t in (k, kv_cache, slot_mapping)):
        raise ValueError("HY V4 indexer cache inputs must share a CUDA device")
    if slot_mapping.numel():
        gems_insert(
            k.contiguous(), kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
        )


def concat_mla_q(ql_nope, q_pe, output):
    if (
        ql_nope.ndim != 3
        or ql_nope.shape[-1] != 512
        or q_pe.shape != (*ql_nope.shape[:2], 64)
        or output.ndim != 3
        or not output.is_contiguous()
        or output.shape[0] != ql_nope.shape[0]
        or output.shape[1] < ql_nope.shape[1]
        or output.shape[2] != 576
        or any(t.dtype != torch.bfloat16 for t in (ql_nope, q_pe, output))
        or any(
            not t.is_cuda or t.device != ql_nope.device for t in (ql_nope, q_pe, output)
        )
    ):
        raise ValueError(
            "HY V4 MLA query requires BF16 latent 512 and RoPE 64 on one CUDA device"
        )
    heads = ql_nope.shape[1]
    if output.shape[1] > heads:
        output.zero_()
    output[:, :heads, :512].copy_(ql_nope)
    output[:, :heads, 512:].copy_(q_pe)


def install_hy4_empty_adapters(config) -> bool:
    """Bind only HY's EMPTY path; preserve native libraries and user OOT classes."""
    if (
        not current_platform.is_cuda()
        or has_native_vllm_kernels()
        or not _is_hy4(config)
    ):
        return False
    model = config.model_config
    if model.dtype != torch.bfloat16 or not model.hf_text_config.learnable_sink:
        raise NotImplementedError(
            "HY V4 EMPTY currently supports BF16 activations with learnable sinks"
        )
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 9:
        raise NotImplementedError(
            "HY V4 EMPTY sparse MLA currently requires Hopper SM90"
        )
    if config.speculative_config is not None:
        raise NotImplementedError(
            "HY V4 EMPTY speculative decoding is not yet validated"
        )
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError(
            "HY V4 EMPTY currently requires BF16 KV cache; the 656-byte FP8 cache writer is not installed"
        )
    parallel = config.parallel_config
    if (
        parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        raise NotImplementedError(
            "HY V4 EMPTY currently supports TP without context parallelism"
        )
    attention = config.attention_config
    if (
        attention.indexer_kv_dtype not in ("auto", "fp8")
        or attention.indexer_sparse_logits
    ):
        raise NotImplementedError("HY V4 EMPTY requires FP8 dense indexer logits")
    if getattr(attention.backend, "name", attention.backend) not in (
        None,
        "FLASHMLA_SPARSE",
    ):
        raise NotImplementedError(
            "HY V4 EMPTY requires the sink-capable sparse MLA backend"
        )
    if not register_nvidia_sparse_indexer():
        return False

    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.models.hy_v4.nvidia import (
        attention as hy_attention,
        flashmla_sparse as hy_mla,
    )

    from vllm_fl.dispatch.backends.flaggems.impl.dsv41_mla import flash_mla_sparse_fwd
    from vllm_fl.ops import nvidia_dsv41_indexer, nvidia_hy4_mla
    from vllm_fl.ops.nvidia_dsv41_ops import register_nvidia_dsv41_ops

    register_nvidia_dsv41_ops()
    if not _libs:
        definitions = (
            (
                "_C_cache_ops",
                "indexer_k_quant_and_cache",
                "(Tensor k, Tensor! kv_cache, Tensor slot_mapping, int quant_block_size, str kv_cache_dtype) -> ()",
                indexer_k_quant_and_cache,
            ),
            (
                "_C_cache_ops",
                "concat_mla_q",
                "(Tensor ql_nope, Tensor q_pe, Tensor! q) -> ()",
                concat_mla_q,
            ),
        )
        for namespace, name, schema, implementation in definitions:
            lib = torch.library.Library(namespace, "FRAGMENT")
            if not hasattr(getattr(torch.ops, namespace), name):
                lib.define(name + schema)
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                f"{namespace}::{name}", "CUDA"
            ):
                lib.impl(name, implementation, "CUDA")
            _libs.append(lib)
    sparse_attn_indexer.fp8_fp4_mqa_logits = nvidia_dsv41_indexer.fp8_mqa_logits
    sparse_attn_indexer.fp8_fp4_paged_mqa_logits = (
        nvidia_dsv41_indexer.fp8_paged_mqa_logits
    )
    hy_attention.per_token_group_quant_fp8 = quantize_indexer_query
    hy_mla.HYV4FlashMLASparseBackend = nvidia_hy4_mla.NvidiaHYV4FlashMLASparseBackend
    hy_mla.flash_mla_sparse_fwd = flash_mla_sparse_fwd
    return True
