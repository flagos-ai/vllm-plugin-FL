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
from .provider import use_nvidia_reference

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
    # Keep the original page storage. Values and scales are separate regions
    # within each page; flattening them would copy the entire physical pool.
    if (
        head_dim != 128
        or block_size not in (32, 64)
        or num_heads not in (16, 32, 64)
        or q_values.dtype != torch.float8_e4m3fn
        or _q_scale is not None
        or kv_cache.dtype != torch.uint8
        or kv_cache.stride(-1) != 1
        or kv_cache.stride(1) != head_dim + 4
        or kv_cache.stride(0) < block_size * (head_dim + 4)
    ):
        raise ValueError("Unsupported GLM5 paged MQA FP8 page/query layout")
    from .paged_mqa import _paged_mqa_logits_kernel

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
    query_bytes = query.view(torch.uint8).reshape(total_rows, num_heads * head_dim)
    logits = torch.full(
        (total_rows, max_model_len),
        float("-inf") if clean_logits else 0.0,
        device=q_values.device,
        dtype=torch.float32,
    )

    block_kv, num_blocks = loaded._select_block_kv(max_model_len, block_size)
    max_blocks_per_sequence = block_tables_expanded.shape[1]
    grid = (loaded.triton.cdiv(max_model_len, block_kv), total_rows)
    _paged_mqa_logits_kernel[grid](
        query_bytes,
        kv_cache,
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
        stride_kv_page=kv_cache.stride(0),
        stride_bt_row=max_blocks_per_sequence,
        stride_out_row=max_model_len,
        stride_w_row=num_heads,
        BLOCK_KV=block_kv,
        BLOCK_D=128,
        NUM_BLOCKS=num_blocks,
    )
    return logits


_TLE_PATCHES = {}


def _bind_tle_compat(loaded):
    from vllm_fl.activation import PendingPatch, bind_patches

    key = loaded.__name__
    if key in _TLE_PATCHES:
        bind_patches(_TLE_PATCHES[key])
        return
    tle = getattr(loaded, "tle", None)
    if not getattr(loaded, "HAS_TLE", False) or tle is None:
        return
    if hasattr(tle, "cumsum") and not getattr(tle, "_vllm_fl_cumsum_compat", False):
        return
    patches = []

    def stage(owner, attr, value):
        existed = hasattr(owner, attr)
        original = getattr(owner, attr, None)
        patches.append(
            PendingPatch(
                target=f"{owner.__name__}.{attr}",
                owner=owner,
                attr=attr,
                replacement=value,
                pristine=original,
                get_current=lambda: getattr(owner, attr, None),
                undo=lambda: (
                    setattr(owner, attr, original) if existed else delattr(owner, attr)
                ),
                fingerprint="glm5.flaggems.tle-cumsum",
                phase="operator binding",
            )
        )

    stage(loaded, "HAS_TLE", False)
    if not hasattr(tle, "cumsum"):
        stage(tle, "cumsum", loaded.tl.cumsum)
        stage(tle, "_vllm_fl_cumsum_compat", True)
    bind_patches(patches)
    _TLE_PATCHES[key] = patches
    logger.warning_once("GLM5 FlagGems top-k: bound tracked non-TLE compatibility")


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
            _bind_tle_compat(loaded)
        function = getattr(loaded, name)
        if module == "fp8_fp4_paged_mqa_logits" and current_platform.is_cuda():
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
    quantized = (blocks / scales.unsqueeze(-1)).clamp(-fp8_max, fp8_max)
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
        gathered_values = values.index_select(0, physical).reshape(-1, head_dim)[
            :length
        ]
        gathered_scales = scales.index_select(0, physical).reshape(
            -1, scales.shape[-1]
        )[:length]
        k_fp8[cursor : cursor + length].view(torch.uint8).copy_(gathered_values)
        k_fp8_scale[cursor : cursor + length].view(torch.float32).copy_(gathered_scales)
        cursor += length


def _dequantize_grouped(
    values: torch.Tensor, scales: torch.Tensor | None
) -> torch.Tensor:
    output = values.float()
    if scales is None:
        return output
    scales = scales.float()
    # MQA prefill supplies one scale per key as [N], whereas cache views
    # retain the trailing group dimension [pages, tokens, groups]. Normalize
    # only the exact per-vector shape; do not broadcast a token axis as groups.
    if scales.shape == output.shape[:-1]:
        scales = scales.unsqueeze(-1)
    if scales.ndim != output.ndim or scales.shape[:-1] != output.shape[:-1]:
        raise ValueError(
            f"Scale shape {tuple(scales.shape)} must match quantized vectors "
            f"{tuple(output.shape[:-1])} with an optional trailing group axis"
        )
    num_groups = scales.shape[-1]
    if num_groups == 1:
        return output * scales
    if num_groups == 0 or output.shape[-1] % num_groups:
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
        logits[request * next_n : (request + 1) * next_n, :max_len] = request_logits
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
    """GLM implementations selected by the published common dispatch policy."""

    def __init__(self) -> None:
        from vllm_fl.dispatch.manager import OpManager

        self.is_nvidia = use_nvidia_reference()
        self.use_nvidia_kpool = current_platform.is_cuda()
        self._flag_ops = {}
        self._bindings = {}
        self._manager = OpManager()
        self.name = "common-policy"

    def _flag(self, module, name):
        from vllm_fl.utils import use_flaggems_op

        if not use_flaggems_op(name):
            return None
        if name not in self._flag_ops:
            loader = (
                _load_flaggems_ops_op
                if module == "per_token_group_quant_fp8"
                else _load_flaggems_op
            )
            self._flag_ops[name] = loader(module, name)
        return self._flag_ops[name]

    def _call_flag(
        self,
        name,
        fn,
        fallback,
        *args,
        native=None,
        native_available=None,
        graph=False,
        _preflight=False,
        **kwargs,
    ):
        from vllm_fl.dispatch.binding import OperatorBinding
        from vllm_fl.dispatch.types import BackendImplKind, OpImpl
        from vllm_fl.utils import use_flaggems_op

        if name not in self._bindings:
            implementations = []
            if fn is not None:

                def flag(*a, **k):
                    if name == "per_token_group_quant_fp8":
                        k["scale_ue8m0"] = k.pop("use_ue8m0", False)
                    return fn(*a, **k)

                flag._is_available = lambda: use_flaggems_op(name)
                implementations.append(
                    OpImpl(name, "glm5.flaggems", BackendImplKind.DEFAULT, flag)
                )
            if native is not None:
                native._is_available = native_available or (lambda: self.is_nvidia)
                implementations.append(
                    OpImpl(
                        name, "glm5.cuda", BackendImplKind.VENDOR, native, vendor="cuda"
                    )
                )
            if fallback is not None:
                implementations.append(
                    OpImpl(name, "glm5.torch", BackendImplKind.REFERENCE, fallback)
                )
            self._manager.registry.register_many(implementations)
            self._manager.bump_policy_epoch()
            self._bindings[name] = OperatorBinding(
                self._manager,
                name,
                graph_capabilities={
                    "glm5.flaggems": graph,
                    "glm5.cuda": None,
                    "glm5.torch": False,
                },
            )
        if _preflight:
            return self._bindings[name].describe()
        return self._bindings[name](*args, **kwargs)

    def _run(self, name, module, fallback, native_module, native_name, *args, **kwargs):
        if name in self._bindings:
            if kwargs.pop("_preflight", False):
                return self._bindings[name].describe()
            return self._bindings[name](*args, **kwargs)
        fn = self._flag(module, name)

        def native(*a, **k):
            module = (
                torch.ops._C
                if native_module == "torch.ops._C"
                else importlib.import_module(native_module)
            )
            return getattr(module, native_name)(*a, **k)

        return self._call_flag(
            name,
            fn,
            fallback,
            *args,
            native=native,
            graph=(name == "fp8_fp4_paged_mqa_logits" and current_platform.is_cuda()),
            **kwargs,
        )

    def preflight(self):
        descriptions = []
        for method in (
            self.per_token_group_quant_fp8,
            self.indexer_k_quant_and_cache,
            self.cp_gather_indexer_k_quant_cache,
            self.mqa_logits,
            self.paged_mqa_logits,
        ):
            descriptions.append(method(_preflight=True))
        descriptions.append(self.topk_prefill(*([None] * 8), _preflight=True))
        descriptions.append(self.topk_decode(*([None] * 8), _preflight=True))
        descriptions.append(self.pack_seq(None, None, _preflight=True))
        descriptions.append(self.unpack_seq(None, None, _preflight=True))
        for method in (
            self.fwht128_quant_fp8,
            self.kpool_compress_and_write_cache,
            self.kpool_decode_update_and_maybe_write_cache_batched,
            self.kpool_seed_tail_cache,
            self.expand_pools_to_tokens,
            self.append_tail_to_topk,
            self.expand_pools_and_append_tail,
        ):
            descriptions.append(method(_preflight=True))
        logger.warning("GLM5-Next resolved indexer bindings: %s", descriptions)
        return descriptions

    @cached_property
    def has_nvidia_deep_gemm(self) -> bool:
        if not self.is_nvidia:
            return False
        from vllm.utils.deep_gemm import has_deep_gemm

        return has_deep_gemm()

    def per_token_group_quant_fp8(self, *args, **kwargs):
        return self._run(
            "per_token_group_quant_fp8",
            "per_token_group_quant_fp8",
            _torch_per_token_group_quant_fp8,
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "per_token_group_quant_fp8",
            *args,
            **kwargs,
        )

    def indexer_k_quant_and_cache(self, *args, **kwargs):
        return self._run(
            "indexer_k_quant_and_cache",
            "indexer_k_quant_and_cache",
            _torch_indexer_k_quant_and_cache,
            "vllm._custom_ops",
            "indexer_k_quant_and_cache",
            *args,
            **kwargs,
        )

    def cp_gather_indexer_k_quant_cache(self, *args, **kwargs):
        return self._run(
            "cp_gather_indexer_k_quant_cache",
            "cp_gather_indexer_k_quant_cache",
            _torch_cp_gather_indexer_k_quant_cache,
            "vllm._custom_ops",
            "cp_gather_indexer_k_quant_cache",
            *args,
            **kwargs,
        )

    def mqa_logits(self, *args, **kwargs):
        return self._run(
            "fp8_fp4_mqa_logits",
            "fp8_fp4_mqa_logits",
            _torch_mqa_logits,
            "vllm.utils.deep_gemm",
            "fp8_fp4_mqa_logits",
            *args,
            **kwargs,
        )

    def paged_mqa_logits(self, *args, **kwargs):
        return self._run(
            "fp8_fp4_paged_mqa_logits",
            "fp8_fp4_paged_mqa_logits",
            _torch_paged_mqa_logits,
            "vllm.utils.deep_gemm",
            "fp8_fp4_paged_mqa_logits",
            *args,
            **kwargs,
        )

    def topk_prefill(
        self,
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
        *,
        _preflight=False,
    ):
        def fallback(logits, starts, ends, output, rows, s0, s1, k):
            output.copy_(_torch_topk(logits, starts, ends, k, True))

        return self._run(
            "top_k_per_row_prefill",
            "top_k_per_row_prefill",
            fallback,
            "torch.ops._C",
            "top_k_per_row_prefill",
            logits,
            row_starts,
            row_ends,
            indices,
            num_rows,
            stride0,
            stride1,
            top_k,
            _preflight=_preflight,
        )

    def topk_decode(
        self,
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        stride0,
        stride1,
        top_k,
        *,
        _preflight=False,
    ):
        def fallback(logits, next_n, lengths, output, rows, s0, s1, k):
            ends = (
                lengths.reshape(-1)
                if lengths.ndim == 2
                else lengths.repeat_interleave(next_n)
            )[:rows]
            output.copy_(_torch_topk(logits, torch.zeros_like(ends), ends, k, False))

        return self._run(
            "top_k_per_row_decode",
            "top_k_per_row_decode",
            fallback,
            "torch.ops._C",
            "top_k_per_row_decode",
            logits,
            next_n,
            seq_lens,
            indices,
            num_rows,
            stride0,
            stride1,
            top_k,
            _preflight=_preflight,
        )

    def pack_seq(self, tensor, lengths, pad_value=-float("inf"), *, _preflight=False):
        return self._run(
            "pack_seq_triton",
            "pack_seq",
            _torch_pack_seq,
            "vllm.v1.attention.ops.common",
            "pack_seq_triton",
            tensor,
            lengths,
            pad_value=pad_value,
            _preflight=_preflight,
        )

    def unpack_seq(self, tensor, lengths, *, _preflight=False):
        return self._run(
            "unpack_seq_triton",
            "unpack_seq",
            _torch_unpack_seq,
            "vllm.v1.attention.ops.common",
            "unpack_seq_triton",
            tensor,
            lengths,
            _preflight=_preflight,
        )

    def _kpool(self, name, *args, **kwargs):
        if name in self._bindings:
            if kwargs.pop("_preflight", False):
                return self._bindings[name].describe()
            return self._bindings[name](*args, **kwargs)

        def native(*a, **k):
            from . import kpool_compress

            return getattr(kpool_compress, name)(*a, **k)

        return self._call_flag(
            name,
            None,
            getattr(portable, name),
            *args,
            native=native,
            native_available=lambda: self.use_nvidia_kpool,
            **kwargs,
        )

    def fwht128_quant_fp8(self, *args, **kwargs):
        return self._kpool("fwht128_quant_fp8", *args, **kwargs)

    def kpool_compress_and_write_cache(self, *args, **kwargs):
        return self._kpool("kpool_compress_and_write_cache", *args, **kwargs)

    def kpool_decode_update_and_maybe_write_cache_batched(self, *args, **kwargs):
        return self._kpool(
            "kpool_decode_update_and_maybe_write_cache_batched", *args, **kwargs
        )

    def kpool_seed_tail_cache(self, *args, **kwargs):
        return self._kpool("kpool_seed_tail_cache", *args, **kwargs)

    def expand_pools_to_tokens(self, *args, **kwargs):
        return self._kpool("expand_pools_to_tokens", *args, **kwargs)

    def append_tail_to_topk(self, *args, **kwargs):
        return self._kpool("append_tail_to_topk", *args, **kwargs)

    def expand_pools_and_append_tail(self, *args, **kwargs):
        return self._kpool("expand_pools_and_append_tail", *args, **kwargs)


INDEXER_BACKEND = Glm5NextIndexerBackend()
