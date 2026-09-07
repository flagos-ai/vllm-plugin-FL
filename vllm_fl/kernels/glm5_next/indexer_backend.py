# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the GLM5-Next sparse indexer operator chain.

NVIDIA keeps the reference vLLM/DeepGEMM/Triton path.  Other accelerators use
matching FlagGems kernels when present and fall back, per operator, to the
backend-neutral PyTorch implementations in this module.
"""

from __future__ import annotations

import importlib
from functools import cached_property
from typing import Callable

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

from . import portable
from .provider import get_glm5_provider, use_nvidia_reference

logger = init_logger(__name__)


def _graph_safe_flaggems_paged_mqa_logits(
    loaded,
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Launch FlagGems paged MQA without a device-to-host sync.

    FlagGems 5.3.3 derives its launch grid with
    ``context_lens.max().item()``.  Decode context lengths live in a static
    graph input, so that host read is both unnecessary and illegal during
    CUDA graph capture.  Use the configured model length as the static launch
    bound; the kernel already exits tiles beyond each row's actual context.
    """
    del schedule_metadata
    q_values, _q_scale = q
    if q_values.dim() == 3:
        q_values = q_values.unsqueeze(1)

    batch, next_n, num_heads, head_dim = q_values.shape
    total_rows = batch * next_n
    block_size = kv_cache.shape[1]
    cache_head_dim = kv_cache.shape[3] - 4
    if cache_head_dim != head_dim:
        raise ValueError(
            f"Paged MQA head mismatch: query={head_dim}, cache={cache_head_dim}"
        )

    if context_lens.dim() == 2:
        context_lens_flat = (
            context_lens.reshape(-1)[:total_rows].contiguous().to(torch.int32)
        )
    else:
        context_lens_flat = (
            context_lens.repeat_interleave(next_n).contiguous().to(torch.int32)
        )

    num_physical_blocks = kv_cache.shape[0]
    flat_size = num_physical_blocks * block_size
    block_stride = block_size * (head_dim + 4)
    cache_flat = kv_cache.reshape(num_physical_blocks, block_stride)
    cache_values = cache_flat[:, : block_size * head_dim]
    cache_values = cache_values.reshape(flat_size, head_dim).contiguous()
    scale_bytes = cache_flat[:, block_size * head_dim :]
    cache_scales = (
        scale_bytes.reshape(num_physical_blocks, block_size, 4)
        .contiguous()
        .reshape(flat_size, 4)
        .view(torch.float32)
        .reshape(flat_size)
        .contiguous()
    )

    if block_tables.dim() == 2:
        block_tables_expanded = (
            block_tables.unsqueeze(1)
            .expand(batch, next_n, -1)
            .reshape(total_rows, -1)
            .contiguous()
            .to(torch.int32)
        )
    else:
        block_tables_expanded = block_tables.contiguous().to(torch.int32)

    query = q_values.reshape(total_rows, num_heads, head_dim).contiguous()
    query_bytes = query.view(torch.uint8).reshape(
        total_rows, num_heads * head_dim
    )
    logits = torch.full(
        (total_rows, max_model_len),
        float("-inf") if clean_logits else 0.0,
        device=q_values.device,
        dtype=torch.float32,
    )

    block_kv, num_blocks = loaded._select_block_kv(max_model_len, block_size)
    max_blocks_per_sequence = block_tables_expanded.shape[1]
    grid = (loaded.triton.cdiv(max_model_len, block_kv), total_rows)
    loaded._mqa_logits_kernel[grid](
        query_bytes,
        cache_values,
        cache_scales,
        weights,
        block_tables_expanded,
        logits,
        context_lens_flat,
        total_rows=total_rows,
        max_ctx=max_model_len,
        num_heads=num_heads,
        head_dim=head_dim,
        max_model_len=max_model_len,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_sequence,
        num_phys_blocks=num_physical_blocks,
        stride_q_row=num_heads * head_dim,
        stride_kv_flat=head_dim,
        stride_bt_row=max_blocks_per_sequence,
        stride_out_row=max_model_len,
        stride_w_row=num_heads,
        BLOCK_KV=block_kv,
        BLOCK_D=128,
        NUM_BLOCKS=num_blocks,
    )
    return logits


def _load_flaggems_op(module: str, name: str) -> Callable | None:
    try:
        loaded = importlib.import_module(f"flag_gems.fused.{module}")
        # FlagGems enables its TLE top-k implementation from the Triton version
        # alone.  Some Triton 3.6 packages expose the TLE namespace without the
        # ``cumsum`` primitive required by that implementation.  In that case
        # select FlagGems' own non-TLE Triton kernel instead of failing at the
        # first JIT launch (or falling all the way back to PyTorch).  Triton's
        # dependency scanner still resolves the dead TLE branch while hashing
        # the shared JIT helper, so provide the standard tl.cumsum symbol too.
        if module in {"top_k_per_row_prefill", "top_k_per_row_decode"}:
            tle = getattr(loaded, "tle", None)
            compat_cumsum = getattr(tle, "_vllm_fl_cumsum_compat", False)
            if getattr(loaded, "HAS_TLE", False) and (
                compat_cumsum or not hasattr(tle, "cumsum")
            ):
                loaded.HAS_TLE = False
                if not hasattr(tle, "cumsum"):
                    setattr(tle, "cumsum", loaded.tl.cumsum)
                    setattr(tle, "_vllm_fl_cumsum_compat", True)
                logger.warning_once(
                    "FlagGems %s TLE path requires tle.cumsum; using its "
                    "non-TLE Triton kernel for GLM5-Next",
                    name,
                )
        function = getattr(loaded, name)
        if module == "fp8_fp4_paged_mqa_logits":
            required = ("_mqa_logits_kernel", "_select_block_kv", "triton")
            if all(hasattr(loaded, attr) for attr in required):
                logger.info_once(
                    "Using graph-safe FlagGems paged-MQA wrapper without "
                    "context_lens.max().item()"
                )

                def graph_safe_paged_mqa(*args, **kwargs):
                    return _graph_safe_flaggems_paged_mqa_logits(
                        loaded, *args, **kwargs
                    )

                return graph_safe_paged_mqa
        return function
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("FlagGems GLM5-Next op %s is unavailable: %s", name, exc)
        return None


def _load_flaggems_ops_op(module: str, name: str) -> Callable | None:
    try:
        loaded = importlib.import_module(f"flag_gems.ops.{module}")
        return getattr(loaded, name)
    except (ImportError, AttributeError, OSError) as exc:
        logger.debug("FlagGems GLM5-Next op %s is unavailable: %s", name, exc)
        return None


def _torch_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % group_size:
        raise ValueError("Last dimension must be divisible by group_size")
    default_dtype, fp8_max = portable.get_fp8_dtype_and_max()
    fp8_dtype = dtype or default_dtype
    fp8_max = float(torch.finfo(fp8_dtype).max) if dtype is not None else fp8_max
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(eps) / fp8_max
    if use_ue8m0:
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    quantized = (grouped / scales.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    quantized = quantized.reshape_as(x).to(fp8_dtype)
    if column_major_scales:
        scales = scales.transpose(-1, -2)
    return quantized, scales.float()


def _torch_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    num_tokens, head_dim = k.shape
    if head_dim % quant_block_size:
        raise ValueError("head_dim must be divisible by quant_block_size")
    blocks = k.float().view(num_tokens, -1, quant_block_size)
    fp8_dtype, fp8_max = portable.get_fp8_dtype_and_max()
    absmax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    scales = absmax / fp8_max
    if scale_fmt is not None:
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    quantized = (blocks / scales.unsqueeze(-1)).clamp(
        -fp8_max, fp8_max
    )
    try:
        quantized = quantized.reshape(num_tokens, head_dim).to(fp8_dtype)
    except RuntimeError as exc:
        raise RuntimeError(
            "The portable indexer cache writer requires float8_e4m3fn support"
        ) from exc
    portable.write_fp8_cache(
        kv_cache,
        quantized,
        scales.float(),
        slot_mapping,
        head_dim,
    )


def _torch_cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
) -> None:
    head_dim = k_fp8.shape[-1]
    values, scales = portable._cache_views(k_cache, head_dim)
    page_size = k_cache.shape[1]
    boundaries = cu_seqlen.detach().to("cpu", torch.int64).tolist()
    cursor = 0
    for request, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        length = end - start
        num_pages = (length + page_size - 1) // page_size
        physical = block_table[request, :num_pages].to(torch.int64)
        gathered_values = values.index_select(0, physical).reshape(-1, head_dim)[:length]
        gathered_scales = scales.index_select(0, physical).reshape(
            -1, scales.shape[-1]
        )[:length]
        k_fp8[cursor : cursor + length].view(torch.uint8).copy_(gathered_values)
        k_fp8_scale[cursor : cursor + length].view(torch.float32).copy_(
            gathered_scales
        )
        cursor += length


def _dequantize_grouped(
    values: torch.Tensor, scales: torch.Tensor | None
) -> torch.Tensor:
    output = values.float()
    if scales is None:
        return output
    scales = scales.float()
    num_groups = scales.shape[-1]
    if num_groups == 1:
        return output * scales
    if output.shape[-1] % num_groups:
        raise ValueError("Quantized width must be divisible by the scale groups")
    group_size = output.shape[-1] // num_groups
    grouped = output.reshape(*output.shape[:-1], num_groups, group_size)
    return (grouped * scales.unsqueeze(-1)).reshape_as(output)


def _torch_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    del clean_logits
    q_values, q_scale = q
    k_values, k_scale = kv
    q_float = _dequantize_grouped(q_values, q_scale)
    k_float = _dequantize_grouped(k_values, k_scale)
    score = torch.einsum("mhd,nd->hmn", q_float, k_float)
    logits = (score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)).sum(0)
    columns = torch.arange(k_values.shape[0], device=q_values.device).unsqueeze(0)
    valid = (columns >= cu_seqlen_ks.reshape(-1, 1)) & (
        columns < cu_seqlen_ke.reshape(-1, 1)
    )
    return logits.masked_fill(~valid, float("-inf"))


def _dequantize_cache(kv_cache: torch.Tensor, head_dim: int) -> torch.Tensor:
    values, scales = portable._cache_views(kv_cache, head_dim)
    fp8_dtype, _ = portable.get_fp8_dtype_and_max()
    return _dequantize_grouped(values.view(fp8_dtype), scales)


def _torch_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata,
    max_model_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    del schedule_metadata, clean_logits
    q_values, q_scale = q
    if q_values.ndim == 3:
        q_values = q_values.unsqueeze(1)
    batch, next_n, heads, head_dim = q_values.shape
    cache = _dequantize_cache(kv_cache.squeeze(-2), head_dim)
    page_size = cache.shape[1]
    logits = torch.full(
        (batch * next_n, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_values.device,
    )
    q_float = _dequantize_grouped(q_values, q_scale)

    for request in range(batch):
        request_lens = context_lens[request]
        if request_lens.ndim == 0:
            limits = request_lens.expand(next_n)
        else:
            limits = request_lens.reshape(-1)[:next_n]
        max_len = int(limits.max().item())
        num_pages = (max_len + page_size - 1) // page_size
        physical = block_tables[request, :num_pages].to(torch.int64)
        keys = cache.index_select(0, physical).reshape(-1, head_dim)[:max_len]
        scores = torch.einsum("thd,nd->htn", q_float[request], keys)
        scores = scores.relu() * weights[
            request * next_n : (request + 1) * next_n
        ].float().transpose(0, 1).unsqueeze(-1)
        request_logits = scores.sum(dim=0)
        columns = torch.arange(max_len, device=q_values.device).unsqueeze(0)
        request_logits.masked_fill_(columns >= limits.reshape(-1, 1), float("-inf"))
        logits[
            request * next_n : (request + 1) * next_n, :max_len
        ] = request_logits
    return logits


def _torch_topk(
    logits: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    top_k: int,
    relative_to_start: bool,
) -> torch.Tensor:
    columns = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    starts = starts.reshape(-1, 1).to(columns.dtype)
    ends = ends.reshape(-1, 1).to(columns.dtype)
    masked = logits.masked_fill((columns < starts) | (columns >= ends), float("-inf"))
    actual_k = min(top_k, logits.shape[1])
    values, indices = torch.topk(masked, k=actual_k, dim=-1)
    indices = indices.to(torch.int32)
    if relative_to_start:
        indices = indices - starts.to(torch.int32)
    indices = torch.where(values == float("-inf"), -1, indices)
    if actual_k == top_k:
        return indices
    out = torch.full(
        (logits.shape[0], top_k), -1, dtype=torch.int32, device=logits.device
    )
    out[:, :actual_k] = indices
    return out


def _torch_pack_seq(
    tensor: torch.Tensor, lengths: torch.Tensor, pad_value=-float("inf")
) -> torch.Tensor:
    lengths_cpu = lengths.detach().to("cpu", torch.int64).tolist()
    max_length = max(lengths_cpu, default=0)
    out = torch.full(
        (len(lengths_cpu), max_length, *tensor.shape[1:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    cursor = 0
    for request, length in enumerate(lengths_cpu):
        out[request, :length].copy_(tensor[cursor : cursor + length])
        cursor += length
    return out


def _torch_unpack_seq(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths_cpu = lengths.detach().to("cpu", torch.int64).tolist()
    pieces = [tensor[request, :length] for request, length in enumerate(lengths_cpu)]
    if not pieces:
        return tensor.new_empty((0, *tensor.shape[2:]))
    return torch.cat(pieces, dim=0)


class Glm5NextIndexerBackend:
    """Per-op provider selection with a stable NVIDIA reference path."""

    def __init__(self) -> None:
        self.is_nvidia = use_nvidia_reference()
        # kpool has no FlagGems equivalent yet.  When H100 explicitly forces
        # the FlagGems A/B path, retain the plugin's reference Triton kpool
        # rather than replacing it with the intentionally slow PyTorch
        # bring-up fallback.  Real non-NVIDIA platforms still use portable.
        self.use_nvidia_kpool = current_platform.is_cuda()
        self._flag_ops: dict[str, Callable | None] = {}
        self._provider_logged: set[str] = set()
        if self.is_nvidia:
            self.name = "nvidia-reference"
        else:
            probe = _load_flaggems_op(
                "indexer_k_quant_and_cache", "indexer_k_quant_and_cache"
            )
            self._flag_ops["indexer_k_quant_and_cache"] = probe
            self.name = "flaggems" if probe is not None else "torch-portable"
        logger.info_once(
            "GLM5-Next provider override=%s, indexer provider=%s",
            get_glm5_provider(),
            self.name,
        )

    def _flag(self, module: str, name: str) -> Callable | None:
        if name not in self._flag_ops:
            self._flag_ops[name] = _load_flaggems_op(module, name)
        fn = self._flag_ops[name]
        if name not in self._provider_logged:
            logger.info(
                "GLM5-Next op %s provider: %s",
                name,
                "FlagGems" if fn is not None else "PyTorch correctness fallback",
            )
            self._provider_logged.add(name)
        return fn

    def _call_flag(
        self,
        name: str,
        fn: Callable | None,
        fallback: Callable,
        *args,
        **kwargs,
    ):
        if fn is not None:
            try:
                return fn(*args, **kwargs)
            except (NotImplementedError, RuntimeError) as exc:
                logger.warning(
                    "FlagGems op %s rejected this GLM5-Next workload; "
                    "using the PyTorch correctness fallback: %s",
                    name,
                    exc,
                )
        return fallback(*args, **kwargs)

    def per_token_group_quant_fp8(self, *args, **kwargs):
        if self.is_nvidia:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8,
            )

            return per_token_group_quant_fp8(*args, **kwargs)
        name = "per_token_group_quant_fp8"
        if name not in self._flag_ops:
            self._flag_ops[name] = _load_flaggems_ops_op(name, name)
        fn = self._flag_ops[name]
        if name not in self._provider_logged:
            logger.info(
                "GLM5-Next op %s provider: %s",
                name,
                "FlagGems" if fn is not None else "PyTorch correctness fallback",
            )
            self._provider_logged.add(name)
        if fn is not None:
            flag_kwargs = dict(kwargs)
            flag_kwargs["scale_ue8m0"] = flag_kwargs.pop("use_ue8m0", False)
            try:
                return fn(*args, **flag_kwargs)
            except (NotImplementedError, RuntimeError) as exc:
                logger.warning(
                    "FlagGems op %s rejected this GLM5-Next workload; "
                    "using the PyTorch correctness fallback: %s",
                    name,
                    exc,
                )
        return _torch_per_token_group_quant_fp8(*args, **kwargs)

    def fwht128_quant_fp8(self, q: torch.Tensor):
        if self.use_nvidia_kpool:
            from .kpool_compress import fwht128_quant_fp8

            return fwht128_quant_fp8(q)
        return portable.fwht128_quant_fp8(q)

    @cached_property
    def has_nvidia_deep_gemm(self) -> bool:
        if not self.is_nvidia:
            return False
        from vllm.utils.deep_gemm import has_deep_gemm

        return has_deep_gemm()

    def indexer_k_quant_and_cache(self, *args, **kwargs) -> None:
        if self.is_nvidia:
            from vllm import _custom_ops as ops

            return ops.indexer_k_quant_and_cache(*args, **kwargs)
        fn = self._flag("indexer_k_quant_and_cache", "indexer_k_quant_and_cache")
        return self._call_flag(
            "indexer_k_quant_and_cache",
            fn,
            _torch_indexer_k_quant_and_cache,
            *args,
            **kwargs,
        )

    def cp_gather_indexer_k_quant_cache(self, *args, **kwargs) -> None:
        if self.is_nvidia:
            from vllm import _custom_ops as ops

            return ops.cp_gather_indexer_k_quant_cache(*args, **kwargs)
        fn = self._flag(
            "cp_gather_indexer_k_quant_cache", "cp_gather_indexer_k_quant_cache"
        )
        return self._call_flag(
            "cp_gather_indexer_k_quant_cache",
            fn,
            _torch_cp_gather_indexer_k_quant_cache,
            *args,
            **kwargs,
        )

    def mqa_logits(self, *args, **kwargs) -> torch.Tensor:
        if self.is_nvidia:
            from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

            return fp8_fp4_mqa_logits(*args, **kwargs)
        fn = self._flag("fp8_fp4_mqa_logits", "fp8_fp4_mqa_logits")
        return self._call_flag(
            "fp8_fp4_mqa_logits", fn, _torch_mqa_logits, *args, **kwargs
        )

    def paged_mqa_logits(self, *args, **kwargs) -> torch.Tensor:
        if self.is_nvidia:
            from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

            return fp8_fp4_paged_mqa_logits(*args, **kwargs)
        fn = self._flag("fp8_fp4_paged_mqa_logits", "fp8_fp4_paged_mqa_logits")
        # A failed GPU launch invalidates CUDA graph capture, so attempting a
        # PyTorch fallback from an exception only obscures the first error.
        # Fall back when the FlagGems operator is absent; otherwise surface a
        # launch error directly.
        if fn is not None:
            return fn(*args, **kwargs)
        return _torch_paged_mqa_logits(*args, **kwargs)

    def topk_prefill(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        top_k: int,
    ) -> None:
        if self.is_nvidia:
            torch.ops._C.top_k_per_row_prefill(
                logits,
                row_starts,
                row_ends,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
            )
            return
        fn = self._flag("top_k_per_row_prefill", "top_k_per_row_prefill")
        if fn is not None:
            try:
                fn(
                    logits,
                    row_starts,
                    row_ends,
                    indices,
                    num_rows,
                    stride0,
                    stride1,
                    top_k,
                )
                return
            except (NotImplementedError, RuntimeError) as exc:
                logger.warning(
                    "FlagGems top_k_per_row_prefill rejected this workload; "
                    "using the PyTorch correctness fallback: %s",
                    exc,
                )
        indices.copy_(_torch_topk(logits, row_starts, row_ends, top_k, True))

    def topk_decode(
        self,
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        top_k: int,
    ) -> None:
        if self.is_nvidia:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
            )
            return
        fn = self._flag("top_k_per_row_decode", "top_k_per_row_decode")
        if fn is not None:
            try:
                fn(
                    logits,
                    next_n,
                    seq_lens,
                    indices,
                    num_rows,
                    stride0,
                    stride1,
                    top_k,
                )
                return
            except (NotImplementedError, RuntimeError) as exc:
                logger.warning(
                    "FlagGems top_k_per_row_decode rejected this workload; "
                    "using the PyTorch correctness fallback: %s",
                    exc,
                )
        if seq_lens.ndim == 2:
            ends = seq_lens.reshape(-1)[:num_rows]
        else:
            ends = seq_lens.repeat_interleave(next_n)[:num_rows]
        starts = torch.zeros_like(ends)
        indices.copy_(_torch_topk(logits, starts, ends, top_k, False))

    def pack_seq(self, tensor, lengths, pad_value=-float("inf")):
        if self.is_nvidia:
            from vllm.v1.attention.ops.common import pack_seq_triton

            return pack_seq_triton(tensor, lengths, pad_value=pad_value)
        fn = self._flag("pack_seq", "pack_seq_triton")
        return self._call_flag(
            "pack_seq_triton",
            fn,
            _torch_pack_seq,
            tensor,
            lengths,
            pad_value=pad_value,
        )

    def unpack_seq(self, tensor, lengths):
        if self.is_nvidia:
            from vllm.v1.attention.ops.common import unpack_seq_triton

            return unpack_seq_triton(tensor, lengths)
        fn = self._flag("unpack_seq", "unpack_seq_triton")
        return self._call_flag(
            "unpack_seq_triton", fn, _torch_unpack_seq, tensor, lengths
        )

    def kpool_compress_and_write_cache(self, *args, **kwargs):
        if self.use_nvidia_kpool:
            from .kpool_compress import kpool_compress_and_write_cache

            return kpool_compress_and_write_cache(*args, **kwargs)
        return portable.kpool_compress_and_write_cache(*args, **kwargs)

    def kpool_decode_update_and_maybe_write_cache_batched(self, *args, **kwargs):
        if self.use_nvidia_kpool:
            from .kpool_compress import (
                kpool_decode_update_and_maybe_write_cache_batched,
            )

            return kpool_decode_update_and_maybe_write_cache_batched(*args, **kwargs)
        return portable.kpool_decode_update_and_maybe_write_cache_batched(
            *args, **kwargs
        )

    def kpool_seed_tail_cache(self, *args, **kwargs):
        if self.use_nvidia_kpool:
            from .kpool_compress import kpool_seed_tail_cache

            return kpool_seed_tail_cache(*args, **kwargs)
        return portable.kpool_seed_tail_cache(*args, **kwargs)

    def expand_pools_to_tokens(self, *args, **kwargs):
        if self.use_nvidia_kpool:
            from .kpool_compress import expand_pools_to_tokens

            return expand_pools_to_tokens(*args, **kwargs)
        return portable.expand_pools_to_tokens(*args, **kwargs)

    def append_tail_to_topk(self, *args, **kwargs):
        if self.use_nvidia_kpool:
            from .kpool_compress import append_tail_to_topk

            return append_tail_to_topk(*args, **kwargs)
        return portable.append_tail_to_topk(*args, **kwargs)

    def expand_pools_and_append_tail(self, *args, **kwargs):
        if self.use_nvidia_kpool:
            from .kpool_compress import expand_pools_and_append_tail

            return expand_pools_and_append_tail(*args, **kwargs)
        return portable.expand_pools_and_append_tail(*args, **kwargs)


INDEXER_BACKEND = Glm5NextIndexerBackend()

__all__ = ["Glm5NextIndexerBackend", "INDEXER_BACKEND"]
