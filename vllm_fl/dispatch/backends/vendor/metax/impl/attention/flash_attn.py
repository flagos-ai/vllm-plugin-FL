# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

from dataclasses import dataclass
import functools
import math
from typing import Any, ClassVar

import numpy as np
import torch

from flag_gems.utils.tile_solver import get_device_topology
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

# --------------------------------------------------------------
# Note: use Maca's merge_attn_states to get cuda kernel invoked
# --------------------------------------------------------------
from .ops.merge_attn_states import merge_attn_states
from .utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from .utils.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
        flash_attn_with_kvcache,  # used for prefill decode split
    )
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    _batch_invariant_MODE as _bi_mode,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    reshape_query_for_spec_decode,  # used for prefill decode split with mtp
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec

# --------------------------------------------------------------
# Note: used for prefill decode split with mtp on maca
# --------------------------------------------------------------
from .mla.common import QueryLenSupport

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MacaFlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        raise NotImplementedError(
            "FP8 dtype is not supported for FlashAttention on Maca."
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # /------------------------  Metax Modification -------------------------\
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    decode_query_start_loc: torch.Tensor
    decode_max_seq_len: int
    decode_seq_lens: torch.Tensor
    decode_block_table: torch.Tensor

    num_prefills: int
    num_prefill_tokens: int
    prefill_query_start_loc: torch.Tensor
    prefill_max_seq_len: int
    prefill_seq_lens: torch.Tensor
    prefill_block_table: torch.Tensor
    # \------------------------- Metax Modification -------------------------/

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True
    # MetaX prefill/decode execution-regime metadata & cross-layer cached plan:
    # - `prefill_is_initial`: True when every prefill sequence starts at pos 0 (`q_len == k_len`),
    #   allowing linear `varlen_fwd` directly on in-place rotated K/V (`prefill_mode == 3`).
    # - `use_kvcache_for_prefill`: True for single-request continuation chunked prefill, routing
    #   either to fused ContigKV + `varlen_fwd` (`prefill_num_splits == -1`, `prefill_mode == 5`)
    #   or paged Split-KV `flash_attn_with_kvcache` (`prefill_mode == 1`).
    # - `prefill_split_transition`: True for 2-request prefill batches mixing 1 continuation chunk
    #   (`prefill_cont_*`) + 1 fresh initial prefill (`prefill_init_tokens`), decoupling them into
    #   ContigKV/Split-KV + linear `varlen_fwd` (`prefill_mode == 6` / `2`).
    # - `_fl_exec_plan`: Per-step immutable tuple of pre-sliced views & kernel launch parameters
    #   built on layer 0 and reused across all 42 layers to eliminate eager Python overhead.
    # - `_fl_kv_cached_ptr`: `q_ret.data_ptr()` sentinel marking KV cache already updated by RoPE+KV.
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_is_initial: bool = False
    use_kvcache_for_prefill: bool = False
    prefill_split_transition: bool = False
    prefill_cont_tokens: int = 0
    prefill_init_tokens: int = 0
    prefill_num_splits: int = 0
    decode_num_splits: int = 0
    positions: torch.Tensor | None = None
    prefill_cont_block_table: torch.Tensor | None = None
    prefill_cont_seq_lens: torch.Tensor | None = None
    prefill_cont_max_seq_len: int = 0
    prefill_cont_cu_q: torch.Tensor | None = None
    prefill_cont_cu_k: torch.Tensor | None = None
    _fl_exec_plan: dict[tuple[int, int, int], tuple] | None = None
    _fl_kv_cached_ptr: int = -1


# Persistent per-(device, dtype, heads, head_dim) workspaces and static `cu_seqlens` table.
_ROPE_KV_KERNEL: Any = None
_Q_ROT_WORKSPACE: dict[tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}
_KV_CONTIG_WORKSPACE: dict[tuple[torch.device, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_CU_SINGLE_LEN_TABLE: dict[torch.device, torch.Tensor] = {}
_ACTIVE_ATTN_GEOM: tuple[int, int, int, int] = (16, 2, 128, 0)
_MAX_ATTN_WORKSPACE_KEYS: int = 6


def _get_q_rot_workspace(
    device: torch.device,
    dtype: torch.dtype,
    num_heads: int,
    head_dim: int,
    min_tokens: int,
) -> torch.Tensor:
    """Return a persistent rotated-Q workspace `[>= max(16384, min_tokens), num_heads, head_dim]` with bounded LRU eviction."""
    q_key = (device, dtype, num_heads, head_dim)
    q_ws = _Q_ROT_WORKSPACE.get(q_key)
    need = max(16384, int(min_tokens))
    if q_ws is None or q_ws.shape[0] < need:
        if q_key not in _Q_ROT_WORKSPACE and len(_Q_ROT_WORKSPACE) >= _MAX_ATTN_WORKSPACE_KEYS:
            _Q_ROT_WORKSPACE.pop(next(iter(_Q_ROT_WORKSPACE)))
        q_ws = torch.empty((need, num_heads, head_dim), dtype=dtype, device=device)
        _Q_ROT_WORKSPACE[q_key] = q_ws
    return q_ws


def _get_kv_contig_workspace(
    device: torch.device,
    dtype: torch.dtype,
    num_kv_heads: int,
    head_dim: int,
    min_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return persistent contiguous `(k_contig, v_contig)` workspaces for fused ContigKV prefill (`prefill_mode == 5/6`)."""
    kv_key = (device, dtype, num_kv_heads, head_dim)
    kv_ws = _KV_CONTIG_WORKSPACE.get(kv_key)
    need = max(16384, int(min_tokens))
    if kv_ws is None or kv_ws[0].shape[0] < need:
        if kv_key not in _KV_CONTIG_WORKSPACE and len(_KV_CONTIG_WORKSPACE) >= _MAX_ATTN_WORKSPACE_KEYS:
            _KV_CONTIG_WORKSPACE.pop(next(iter(_KV_CONTIG_WORKSPACE)))
        kv_ws = (
            torch.empty((need, num_kv_heads, head_dim), dtype=dtype, device=device),
            torch.empty((need, num_kv_heads, head_dim), dtype=dtype, device=device),
        )
        _KV_CONTIG_WORKSPACE[kv_key] = kv_ws
    return kv_ws


@dataclass(frozen=True)
class AttnStreamingProfile:
    """Calibrated or analytical streaming parameters for `AttnCostModel`."""

    paged_penalty: float = 1.10
    gather_page_cost: float = 0.015
    combine_base: float = 0.35
    combine_split_cost: float = 0.18


_ATTN_STREAMING_PROFILES: dict[int, AttnStreamingProfile] = {}
_CALIBRATED_ATTN_KEYS: set[tuple[int, torch.dtype, int, int, int]] = set()


def _get_rope_kv_kernel() -> Any:
    """Lazily import and cache FlagGems' `_fused_qk_rope_and_cache_kv_kernel` Triton kernel."""
    global _ROPE_KV_KERNEL
    if _ROPE_KV_KERNEL is None:
        from flag_gems.runtime.backend._metax.fused import _fused_qk_rope_and_cache_kv_kernel

        _ROPE_KV_KERNEL = _fused_qk_rope_and_cache_kv_kernel
    return _ROPE_KV_KERNEL


def _select_rope_block_h(num_heads: int, prefer_eight: bool) -> int:
    """Choose the largest power-of-two `BLOCK_H` divisor of `num_heads` (preferring 8 for small batches, 4 for large prefills)."""
    if prefer_eight and num_heads % 8 == 0:
        return 8
    if num_heads % 4 == 0:
        return 4
    if num_heads % 8 == 0:
        return 8
    return 2 if num_heads % 2 == 0 else 1


def calibrate_attn_streaming_profile(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    num_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 128,
) -> AttnStreamingProfile:
    """Run a `< 3 ms` CUDA-event micro-probe measuring paged-vs-linear streaming ratio."""
    dev_idx = device.index or 0
    cal_key = (dev_idx, dtype, num_heads, num_kv_heads, head_dim)
    if cal_key in _CALIBRATED_ATTN_KEYS and dev_idx in _ATTN_STREAMING_PROFILES:
        return _ATTN_STREAMING_PROFILES[dev_idx]
    default_prof = _ATTN_STREAMING_PROFILES.get(dev_idx, AttnStreamingProfile())
    if not is_flash_attn_varlen_func_available() or num_heads <= 0 or num_kv_heads <= 0:
        return default_prof
    try:
        topo = get_device_topology()
        # Pick single-wave query length (`<= num_sms` CTAs at `BM=128`) and `k=2048` (`~140 us/call`)
        # so 2 warmup + 3 timed event pairs across 3 kernels finish in `< 2.5 ms` total.
        q_blocks = max(1, min(6, topo.num_sms // max(1, num_heads)))
        q_c = q_blocks * 128
        k_c = 2048
        bs = 16
        n_blocks = k_c // bs
        q_ws = _Q_ROT_WORKSPACE.get((device, dtype, num_heads, head_dim))
        q_buf = q_ws[:q_c].zero_() if (q_ws is not None and q_ws.shape[0] >= q_c) else torch.zeros((q_c, num_heads, head_dim), device=device, dtype=dtype)
        kv_ws = _KV_CONTIG_WORKSPACE.get((device, dtype, num_kv_heads, head_dim))
        if kv_ws is not None and kv_ws[0].shape[0] >= k_c:
            k_lin, v_lin = kv_ws[0][:k_c].zero_(), kv_ws[1][:k_c].zero_()
        else:
            k_lin = torch.zeros((k_c, num_kv_heads, head_dim), device=device, dtype=dtype)
            v_lin = torch.zeros((k_c, num_kv_heads, head_dim), device=device, dtype=dtype)
        k_pg = k_lin.view(n_blocks, bs, num_kv_heads, head_dim)
        v_pg = v_lin.view(n_blocks, bs, num_kv_heads, head_dim)
        cu_tbl = _get_cu_single_len_table(device)
        cu_q, cu_k = cu_tbl[q_c], cu_tbl[k_c]
        bt = torch.arange(n_blocks, dtype=torch.int32, device=device).view(1, n_blocks)
        sl = cu_k[1:2]
        out_buf = torch.empty_like(q_buf)
        q_dec = q_buf[:1].unsqueeze(0)
        out_dec = out_buf[:1].unsqueeze(0)
        scale = 1.0 / math.sqrt(head_dim)
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(9)]
        for _ in range(2):
            flash_attn_varlen_func(q=q_buf, k=k_lin, v=v_lin, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=q_c, max_seqlen_k=k_c, softmax_scale=scale, causal=True, out=out_buf)
            flash_attn_with_kvcache(q=q_buf.unsqueeze(0), k_cache=k_pg, v_cache=v_pg, block_table=bt, cache_seqlens=sl, softmax_scale=scale, causal=True, num_splits=1, out=out_buf.unsqueeze(0))
            flash_attn_with_kvcache(q=q_dec, k_cache=k_pg, v_cache=v_pg, block_table=bt, cache_seqlens=sl, softmax_scale=scale, causal=True, num_splits=4, out=out_dec)
        for r in range(3):
            ev_s[r].record()
            flash_attn_varlen_func(q=q_buf, k=k_lin, v=v_lin, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=q_c, max_seqlen_k=k_c, softmax_scale=scale, causal=True, out=out_buf)
            ev_e[r].record()
            ev_s[3 + r].record()
            flash_attn_with_kvcache(q=q_buf.unsqueeze(0), k_cache=k_pg, v_cache=v_pg, block_table=bt, cache_seqlens=sl, softmax_scale=scale, causal=True, num_splits=1, out=out_buf.unsqueeze(0))
            ev_e[3 + r].record()
            ev_s[6 + r].record()
            flash_attn_with_kvcache(q=q_dec, k_cache=k_pg, v_cache=v_pg, block_table=bt, cache_seqlens=sl, softmax_scale=scale, causal=True, num_splits=4, out=out_dec)
            ev_e[6 + r].record()
        torch.cuda.synchronize(device)
        t_lin = sorted(ev_s[r].elapsed_time(ev_e[r]) * 1000.0 for r in range(3))[1]
        t_pg1 = sorted(ev_s[3 + r].elapsed_time(ev_e[3 + r]) * 1000.0 for r in range(3))[1]
        t_dec4 = sorted(ev_s[6 + r].elapsed_time(ev_e[6 + r]) * 1000.0 for r in range(3))[1]
        paged_pen = min(1.35, max(1.02, t_pg1 / max(t_lin, 1.0)))
        tile_us = max(0.5, t_lin / max(1.0, (k_c - 0.5 * q_c) / 64.0))
        comb_split = min(0.28, max(0.12, (t_dec4 / (4.0 * tile_us)) * 0.16))
        prof = AttnStreamingProfile(paged_penalty=paged_pen, gather_page_cost=0.015, combine_base=0.35, combine_split_cost=comb_split)
        _ATTN_STREAMING_PROFILES[dev_idx] = prof
        _CALIBRATED_ATTN_KEYS.add(cal_key)
        _solve_chunked_prefill_cached.cache_clear()
        _solve_decode_splits_cached.cache_clear()
        solve_chunked_prefill_splits.cache_clear()
        solve_decode_splits.cache_clear()
        solve_cudagraph_dual_splits.cache_clear()
        return prof
    except Exception:
        return default_prof


def _get_cu_single_len_table(device: torch.device, min_len: int = 131072) -> torch.Tensor:
    """Return a persistent `[>= min_len + 1, 2]` int32 GPU table where row `L` equals `[0, L]`.

    Indexing `tbl[L]` yields a ready-made `cu_seqlens` tensor `[0, L]` for single-sequence
    prefill / continuation chunks with zero CPU->GPU H2D copy per step, growing dynamically on demand.
    """
    tbl = _CU_SINGLE_LEN_TABLE.get(device)
    if tbl is None or min_len >= tbl.shape[0]:
        cap = max(131072, int(min_len)) + 1
        if tbl is not None:
            cap = max(cap, tbl.shape[0] * 2)
        tbl = torch.zeros((cap, 2), dtype=torch.int32, device=device)
        tbl[:, 1] = torch.arange(cap, dtype=torch.int32, device=device)
        _CU_SINGLE_LEN_TABLE[device] = tbl
    return tbl


def prewarm_metax_rope_kv_workspaces(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    num_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    block_size: int = 16,
) -> None:
    """Pre-allocate persistent Q/ContigKV workspaces and compile all reachable RoPE+KV specializations.

    Why prewarm ahead of CUDAGraph capture & serving:
    1. Allocates fixed `16384`-row `_Q_ROT_WORKSPACE` (`[16384, 16, 128]`) and `_KV_CONTIG_WORKSPACE`
       (`2 x [16384, 2, 128]`) buffers up front so FULL decode CUDAGraph capture binds stable
       device pointers and long `16K` chunked prefills never trigger mid-step reallocations.
    2. Compiles all 8 `(SPLIT_TOKEN, GATHER_CONTIG_KV, WRITE_K_INPLACE, BLOCK_H)` Triton kernel
       specializations (`2` `WRITE_K_INPLACE` states x `4` launch geometries:
       - `sp_tok=True, g_kv=False, bh=8`: decode `M <= 64`, `2 * M` CTAs for CU occupancy;
       - `sp_tok=False, g_kv=False, bh=8`: medium batches `65 <= M <= 128`;
       - `sp_tok=False, g_kv=False, bh=4`: large prefill `M > 128` without ContigKV gather;
       - `sp_tok=False, g_kv=True, bh=8`: continuation prefill `prefill_mode == 5/6` fusing
         paged KV prefix gather + new token RoPE/KV write in one launch).
    """
    _get_cu_single_len_table(device)
    if head_dim not in (64, 128) or num_heads < num_kv_heads or num_kv_heads < 1 or (num_kv_heads & (num_kv_heads - 1)) != 0 or num_kv_heads > 16:
        calibrate_attn_streaming_profile(device, dtype, num_heads, num_kv_heads, head_dim)
        return
    q_ws = _get_q_rot_workspace(device, dtype, num_heads, head_dim, 16384)
    kv_ws = _get_kv_contig_workspace(device, dtype, num_kv_heads, head_dim, 16384)
    calibrate_attn_streaming_profile(device, dtype, num_heads, num_kv_heads, head_dim)
    kernel = _get_rope_kv_kernel()
    m_w = 512
    q_dim, kv_dim = num_heads * head_dim, num_kv_heads * head_dim
    stride_cb = block_size * kv_dim
    can_split = num_heads >= 2 and (num_heads & (num_heads - 1)) == 0
    can_gather = block_size >= 16 and stride_cb % 1024 == 0
    bh_hi = _select_rope_block_h(num_heads, prefer_eight=True)
    bh_lo = _select_rope_block_h(num_heads, prefer_eight=False)
    nw = 2 if num_kv_heads <= 8 else 4
    qkv_w = torch.zeros((m_w, q_dim + 2 * kv_dim), device=device, dtype=dtype)
    q_in = qkv_w[:, :q_dim].view(m_w, num_heads, head_dim)
    k_in = qkv_w[:, q_dim : q_dim + kv_dim].view(m_w, num_kv_heads, head_dim)
    v_in = qkv_w[:, q_dim + kv_dim :].view(m_w, num_kv_heads, head_dim)
    pos_w = torch.zeros((m_w,), device=device, dtype=torch.int64)
    slot_w = torch.arange(m_w, device=device, dtype=torch.int64)
    cos_sin_w = torch.zeros((64, head_dim), device=device, dtype=dtype)
    n_blk_w = max(4, (m_w + block_size - 1) // block_size)
    k_cache_w = torch.zeros((n_blk_w, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache_w = torch.zeros((n_blk_w, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    bt_w = torch.zeros((1, 4), device=device, dtype=torch.int32)
    specs = [
        (96, 0, 0, bh_hi, False, False, False), (96, 0, 0, bh_hi, False, False, True),
        (256, 0, 0, bh_lo, False, False, False), (256, 0, 0, bh_lo, False, False, True),
    ]
    if can_split:
        specs.extend([(16, 0, 0, bh_hi, True, False, False), (16, 0, 0, bh_hi, True, False, True)])
    if can_gather:
        specs.extend([(512, 2, 2 * block_size, bh_hi, False, True, False), (512, 2, 2 * block_size, bh_hi, False, True, True)])
    for n_tok, n_pg, pref_l, bh, sp_tok, g_kv, w_ip in specs:
        grid = (2 * n_tok,) if sp_tok else (n_tok + n_pg,)
        kernel[grid](
            q_in[:n_tok], k_in[:n_tok], v_in[:n_tok], q_ws[:n_tok], pos_w[:n_tok], cos_sin_w, slot_w[:n_tok],
            k_cache_w, v_cache_w, q_in.stride(0), k_in.stride(0), v_in.stride(0), stride_cb, kv_dim,
            kv_ws[0][: pref_l + n_tok] if g_kv else q_in, kv_ws[1][: pref_l + n_tok] if g_kv else q_in,
            bt_w if g_kv else slot_w, n_tok, 0, n_tok if g_kv else 0, pref_l,
            BLOCK_SIZE=block_size, BLOCK_H=bh, WRITE_K_INPLACE=w_ip, GATHER_CONTIG_KV=g_kv, SPLIT_TOKEN=sp_tok,
            NUM_Q_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim, num_warps=nw,
        )


def _launch_fused_qk_rope_and_cache_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_actual_tokens: int,
    write_k_inplace: bool,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    head_size: int | None = None,
) -> torch.Tensor:
    """Launch `_fused_qk_rope_and_cache_kv_kernel` directly without ContigKV gather (used by DCP/cascade semi-fused path)."""
    if num_actual_tokens <= 0:
        return query
    h_q = int(num_heads or query.shape[1])
    h_kv = int(num_kv_heads or key.shape[1])
    d_h = int(head_size or query.shape[2])
    q_ws = _get_q_rot_workspace(query.device, query.dtype, h_q, d_h, num_actual_tokens)
    q_out = q_ws[:num_actual_tokens]
    # Split each token across 2 CTAs when `M <= 64` to saturate C500's 104 CUs.
    split_tok = num_actual_tokens <= 64 and h_q >= 2 and (h_q & (h_q - 1)) == 0
    block_h = _select_rope_block_h(h_q, prefer_eight=(split_tok or num_actual_tokens <= 128))
    num_warps = 2 if h_kv <= 8 else 4
    grid = (2 * num_actual_tokens,) if split_tok else (num_actual_tokens,)
    _get_rope_kv_kernel()[grid](
        query, key, value, q_out, positions, cos_sin_cache, slot_mapping, key_cache, value_cache,
        query.stride(0), key.stride(0), value.stride(0), key_cache.stride(0), key_cache.stride(1),
        query, query, slot_mapping, num_actual_tokens, 0, 0, 0,
        BLOCK_SIZE=key_cache.shape[1], BLOCK_H=block_h, WRITE_K_INPLACE=write_k_inplace,
        GATHER_CONTIG_KV=False, SPLIT_TOKEN=split_tok,
        NUM_Q_HEADS=h_q, NUM_KV_HEADS=h_kv, HEAD_DIM=d_h, num_warps=num_warps,
    )
    return q_out


_ORIG_CG_WRAPPER_CALL = None
_LAST_DECODE_NUM_SPLITS: int = 0


def _install_adaptive_cudagraph_regime_hook() -> None:
    """Dual-Regime Execution Engine for MetaX C500 (`_install_adaptive_cudagraph_regime_hook`):

    1. **Zero-Sync Eager Mode for Large Prefill / Mixed Batches**:
       Avoids MACA `mcGraphLaunch` hidden stream-synchronization barriers across
       piecewise subgraphs per step, preserving async kernel launch overlap.
    2. **Analytical Dual-Split FULL CUDAGraph Replay for Uniform Decode**:
       Whenever `solve_cudagraph_dual_splits(B, H_q, H_kv, D)` identifies an L2-resident
       mid-context split `s_l2 < s_wave` (where `s_wave`'s partial workspace would exceed
       55% of hardware L2 capacity), captures a companion `s_l2` CUDAGraph sharing
       `self.graph_pool` and dynamically dispatches between `s_l2` and `s_wave` at replay.
    """
    global _ORIG_CG_WRAPPER_CALL
    if _ORIG_CG_WRAPPER_CALL is not None:
        return
    try:
        from contextlib import ExitStack
        from unittest.mock import patch
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm_fl.compilation.graph import GraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm.model_executor.offloader.base import get_offloader
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import current_stream, weak_ref_tensors
    except Exception:
        return

    _ORIG_CG_WRAPPER_CALL = CUDAGraphWrapper.__call__

    def _make_dual_cg_call(orig_call):
        def _patched_call(self, *args, **kwargs):
            if self.runtime_mode == CUDAGraphMode.FULL and is_forward_context_available():
                fc = get_forward_context()
                if fc.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    bd = fc.batch_descriptor
                    if bd is not None and bd.uniform and bd.num_tokens > 0:
                        h_q, h_kv, d_h, dev_idx = _ACTIVE_ATTN_GEOM
                        s_l2, s_wave = solve_cudagraph_dual_splits(bd.num_tokens, h_q, h_kv, d_h, dev_idx)
                        if 0 < s_l2 < s_wave:
                            entries = getattr(self, "concrete_graph_entries", None)
                            if entries is None:
                                entries = self.concrete_cudagraph_entries
                            entry = entries.get(bd)
                            base_g = getattr(entry, "graph", None) or getattr(entry, "cudagraph", None) if entry is not None else None
                            if base_g is not None:
                                # Replay fast path: route mid-context steps (`_LAST_DECODE_NUM_SPLITS <= s_l2`)
                                # to the L2-resident `cg_l2` graph and long-context steps to the primary `s_wave` graph.
                                if 0 < _LAST_DECODE_NUM_SPLITS <= getattr(entry, "_fl_splits_l2", s_l2):
                                    cg_l2 = getattr(entry, "_fl_cudagraph_l2", None)
                                    if cg_l2 is not None:
                                        try:
                                            get_offloader().sync_prev_onload()
                                        except Exception:
                                            pass
                                        cg_l2.replay()
                                        return entry._fl_output_l2
                                return orig_call(self, *args, **kwargs)
                            # First capture for this batch descriptor: capture primary `s_wave` graph first,
                            # then capture the companion `s_l2` graph sharing `self.graph_pool`.
                            out = orig_call(self, *args, **kwargs)
                            entry = entries.get(bd)
                            base_g = getattr(entry, "graph", None) or getattr(entry, "cudagraph", None) if entry is not None else None
                            if entry is not None and base_g is not None and getattr(entry, "_fl_cudagraph_l2", None) is None:
                                attn_md = fc.attn_metadata
                                mds = list(attn_md.values()) if isinstance(attn_md, dict) else ([attn_md] if attn_md is not None else [])
                                orig_splits = [getattr(m, "decode_num_splits", s_wave) for m in mds]
                                for m in mds:
                                    if hasattr(m, "decode_num_splits"):
                                        m.decode_num_splits = s_l2
                                try:
                                    _ = self.runnable(*args, **kwargs)
                                    cg_l2 = torch.cuda.CUDAGraph()
                                    with ExitStack() as stack:
                                        if self.cudagraph_options.gc_disable:
                                            stack.enter_context(patch("gc.collect", lambda *a, **kw: None))
                                            stack.enter_context(patch("torch.accelerator.empty_cache", lambda *a, **kw: None))
                                        set_graph_pool_id(self.graph_pool if self.graph_pool is not None else current_platform.graph_pool_handle())
                                        try:
                                            get_offloader().sync_prev_onload()
                                        except Exception:
                                            pass
                                        with torch.cuda.graph(cg_l2, pool=self.graph_pool, stream=current_stream()):
                                            out_l2 = self.runnable(*args, **kwargs)
                                            try:
                                                get_offloader().join_after_forward()
                                            except Exception:
                                                pass
                                            if self.cudagraph_options.weak_ref_output:
                                                out_l2 = weak_ref_tensors(out_l2)
                                    entry._fl_cudagraph_l2 = cg_l2
                                    entry._fl_output_l2 = weak_ref_tensors(out_l2)
                                    entry._fl_splits_l2 = s_l2
                                    return out_l2 if (0 < _LAST_DECODE_NUM_SPLITS <= s_l2) else out
                                finally:
                                    for m, ospl in zip(mds, orig_splits):
                                        if hasattr(m, "decode_num_splits"):
                                            m.decode_num_splits = ospl
                            return out
            return orig_call(self, *args, **kwargs)

        return _patched_call

    CUDAGraphWrapper.__call__ = _make_dual_cg_call(CUDAGraphWrapper.__call__)
    GraphWrapper.__call__ = _make_dual_cg_call(GraphWrapper.__call__)


class AttnCostModel:
    """Analytical `2-CTA/CU` Super-Wave + `L2`-Residency Attention Cost Model."""

    FA_VARLEN_BM: ClassVar[int] = 128
    FA_BLOCK_N: ClassVar[int] = 64
    L2_PARTIAL_BUDGET_FRAC: ClassVar[float] = 0.40
    L2_DUAL_GRAPH_TRIGGER_FRAC: ClassVar[float] = 0.55

    @staticmethod
    def _max_decode_splits_for_batch(batch_kv: int, w_super: int, epoch_ctas: int, max_hw_splits: int) -> int:
        """Bound Split-KV search space to 1 (or 2 when single-epoch fill < 94%) hardware super-wave epochs."""
        one_ep_s = max(1, epoch_ctas // max(1, batch_kv))
        one_ep_fill = (one_ep_s * batch_kv) / epoch_ctas
        max_epochs = 2 if (batch_kv > (w_super // 2) and one_ep_fill < 0.94) else 1
        return min(max_hw_splits, max(1, (max_epochs * epoch_ctas) // max(1, batch_kv)))


@functools.lru_cache(maxsize=4096)
def solve_cudagraph_dual_splits(
    num_decodes: int,
    num_heads_q: int = 16,
    num_heads_kv: int = 2,
    head_size: int = 128,
    device_idx: int = 0,
) -> tuple[int, int]:
    """Return `(s_l2, s_wave)` for padded FULL CUDAGraph decode buckets."""
    if num_decodes <= 0 or num_heads_kv <= 0:
        return (0, 0)
    topo = get_device_topology()
    num_cus, l2_bytes = topo.num_sms, topo.l2_cache_bytes
    # `w_super = 2 * num_cus` (2 CTAs/CU co-residency, 208 CTAs on C500); `epoch_ctas = 4 * w_super`.
    w_super = 2 * num_cus
    epoch_ctas = 4 * w_super
    max_hw_splits = max(16, num_cus // 2)
    prof = _ATTN_STREAMING_PROFILES.get(device_idx, AttnStreamingProfile())
    batch_kv = num_decodes * num_heads_kv
    ref_blocks = 256
    max_s = AttnCostModel._max_decode_splits_for_batch(batch_kv, w_super, epoch_ctas, max_hw_splits)
    best_s, best_cost = 1, float("inf")
    for s in range(1, max_s + 1):
        tot_ctas = batch_kv * s
        eff_waves = 0.75 * (4.0 * math.ceil(tot_ctas / epoch_ctas)) + 0.25 * math.ceil(tot_ctas / w_super)
        cost = eff_waves * math.ceil(ref_blocks / s) + prof.combine_split_cost * s
        if cost < best_cost:
            best_cost, best_s = cost, s
    s_wave = best_s
    # FP32 partial output + LSE workspace per split: `B * H_q * (D + 1) * 4` bytes.
    # Trigger companion `s_l2` CUDAGraph when `s_wave` partial workspace exceeds 55% of L2.
    per_split_bytes = num_decodes * num_heads_q * (head_size + 1) * 4
    if s_wave * per_split_bytes <= int(AttnCostModel.L2_DUAL_GRAPH_TRIGGER_FRAC * l2_bytes):
        return (s_wave, s_wave)
    l2_budget = int(AttnCostModel.L2_PARTIAL_BUDGET_FRAC * l2_bytes)
    s_l2 = max(1, min(s_wave, l2_budget // max(1, per_split_bytes)))
    return (s_l2, s_wave)


@functools.lru_cache(maxsize=4096)
def _solve_decode_splits_cached(
    num_decodes: int,
    num_blocks: int,
    num_heads_kv: int,
    num_heads_q: int,
    head_size: int,
    device_idx: int,
) -> int:
    """Minimize `eff_waves * blocks_per_split + combine_cost + l2_spill_penalty` over `s in [1, max_s]`."""
    topo = get_device_topology()
    num_cus, l2_bytes = topo.num_sms, topo.l2_cache_bytes
    w_super = 2 * num_cus
    epoch_ctas = 4 * w_super
    max_hw_splits = max(16, num_cus // 2)
    batch_kv = num_decodes * num_heads_kv
    if batch_kv >= 2 * epoch_ctas:
        return 1
    prof = _ATTN_STREAMING_PROFILES.get(device_idx, AttnStreamingProfile())
    max_s = min(AttnCostModel._max_decode_splits_for_batch(batch_kv, w_super, epoch_ctas, max_hw_splits), num_blocks // 2)
    per_split_bytes = num_decodes * num_heads_q * (head_size + 1) * 4
    l2_budget = int(AttnCostModel.L2_PARTIAL_BUDGET_FRAC * l2_bytes)
    kv_working_bytes = batch_kv * (num_blocks * AttnCostModel.FA_BLOCK_N) * head_size * 2
    # Penalize partial-accumulator L2 overflow only in the mid-context band where KV traffic does not yet dominate combine spills.
    l2_mid_band = kv_working_bytes < (9 * w_super * l2_bytes) // 64
    best_s, best_cost = 1, float("inf")
    for s in range(1, max_s + 1):
        tot_ctas = batch_kv * s
        eff_waves = 0.75 * (4.0 * math.ceil(tot_ctas / epoch_ctas)) + 0.25 * math.ceil(tot_ctas / w_super)
        bps = math.ceil(num_blocks / s)
        comb = 0.0 if s == 1 else (prof.combine_base + prof.combine_split_cost * s)
        p_bytes = per_split_bytes * s
        l2_pen = ((0.35 * bps + 6.0) * ((p_bytes - l2_budget) / l2_bytes)) if (l2_mid_band and p_bytes > l2_budget) else 0.0
        cost = eff_waves * bps + comb + l2_pen
        if cost < best_cost:
            best_cost, best_s = cost, s
    return 0 if best_s <= 1 else best_s


@functools.lru_cache(maxsize=4096)
def solve_decode_splits(
    num_decodes: int,
    max_seq_len: int,
    num_heads_kv: int,
    num_heads_q: int = 16,
    head_size: int = 128,
    device_idx: int = 0,
) -> int:
    """Closed-form Decode Split-KV solver separating CUDAGraph capture (`<= 64`) from live steps."""
    if num_decodes <= 0 or num_heads_kv <= 0:
        return 0
    if max_seq_len <= AttnCostModel.FA_BLOCK_N:
        return solve_cudagraph_dual_splits(num_decodes, num_heads_q, num_heads_kv, head_size, device_idx)[1]
    num_blocks = (max_seq_len + AttnCostModel.FA_BLOCK_N - 1) // AttnCostModel.FA_BLOCK_N
    if num_blocks < 16:
        return 0
    return _solve_decode_splits_cached(num_decodes, num_blocks, num_heads_kv, num_heads_q, head_size, device_idx)


@functools.lru_cache(maxsize=4096)
def _solve_chunked_prefill_cached(
    q_len: int,
    k_len: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_size: int,
    block_size: int,
    device_idx: int,
    allow_contig: bool = True,
) -> int:
    """Compare fused ContigKV + linear `varlen_fwd` (`-1`) against paged Split-KV `fwd_kvcache` (`s >= 1`)."""
    topo = get_device_topology()
    num_cus, l2_bytes = topo.num_sms, topo.l2_cache_bytes
    w_super = 2 * num_cus
    max_hw_splits = max(16, num_cus // 2)
    prof = _ATTN_STREAMING_PROFILES.get(device_idx, AttnStreamingProfile())
    # Causal trapezoid average KV length attended by a chunk of length `q_len` at total length `k_len`.
    eff_k_tiles = max(1.0, (k_len - 0.5 * q_len) / AttnCostModel.FA_BLOCK_N)

    # Candidate A: fused ContigKV gather + linear `varlen_fwd` (`kBlockM=128`), eligible when KV workspace <= 4x L2.
    kv_ws_bytes = k_len * 2 * num_heads_kv * head_size * 2
    contig_page_ok = (
        allow_contig
        and head_size in (64, 128)
        and block_size >= 16
        and (block_size * num_heads_kv * head_size) % 1024 == 0
    )
    if contig_page_ok and kv_ws_bytes <= 4 * l2_bytes:
        ctas_lin = math.ceil(q_len / AttnCostModel.FA_VARLEN_BM) * num_heads_q
        raw_waves = math.ceil(ctas_lin / num_cus)
        waves_lin = float(raw_waves) if ctas_lin <= 2 * num_cus else (ctas_lin / (0.94 * num_cus))
        prefix_pages = math.ceil((k_len - q_len) / max(1, block_size))
        cost_contig = waves_lin * eff_k_tiles + prof.gather_page_cost * math.ceil(prefix_pages / num_cus)
    else:
        cost_contig = float("inf")

    # Candidate B: paged Split-KV `flash_attn_with_kvcache`, sweeping `s in [1, max_s]` with combine BW overhead.
    gqa_ratio = max(1, num_heads_q // max(1, num_heads_kv))
    fa_kv_bm = max(16, 128 // gqa_ratio)
    packed_rows = min(q_len, fa_kv_bm) * gqa_ratio
    base_ctas = math.ceil(q_len / fa_kv_bm) * num_heads_kv
    wave_unit = w_super if packed_rows <= 64 else num_cus
    max_s = min(max_hw_splits, max(1, int(math.ceil(eff_k_tiles / 2.0))))
    partial_per_split_bytes = q_len * num_heads_q * (head_size + 1) * 4
    kv_tile_bytes = AttnCostModel.FA_BLOCK_N * 2 * num_heads_kv * head_size * 2
    combine_bw_tiles = (2.0 * partial_per_split_bytes) / max(1, (num_cus * kv_tile_bytes) // 4)

    best_s, best_paged = 1, float("inf")
    for s in range(1, max_s + 1):
        waves_p = math.ceil((base_ctas * s) / wave_unit)
        comb = 0.0 if s == 1 else (prof.combine_base + (prof.combine_split_cost + combine_bw_tiles) * s)
        c = waves_p * prof.paged_penalty * math.ceil(eff_k_tiles / s) + comb
        if c < best_paged:
            best_paged, best_s = c, s
    return -1 if cost_contig < best_paged else best_s


@functools.lru_cache(maxsize=4096)
def solve_chunked_prefill_splits(
    q_len: int,
    k_len: int,
    num_heads_q: int = 16,
    num_heads_kv: int = 2,
    head_size: int = 128,
    block_size: int = 16,
    device_idx: int = 0,
    allow_contig: bool = True,
) -> int:
    """Closed-form Continuation Chunked-Prefill solver (`-1` = ContigKV + `varlen_fwd`)."""
    if q_len <= 0 or k_len <= q_len or num_heads_kv <= 0:
        return 1
    q_b = ((q_len + 15) >> 4) << 4
    k_b = max(q_b + 16, ((k_len + 63) >> 6) << 6)
    return _solve_chunked_prefill_cached(
        q_b, k_b, num_heads_q, num_heads_kv, head_size, block_size, device_idx, allow_contig
    )


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    from vllm.model_executor.layers.attention.attention import Attention

    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # /------------------------  Metax Modification -------------------------\
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    # Defines the level of query length support for this backend.
    # - SINGLE_ONLY: Only single-token queries (no spec decode support)
    # - UNIFORM: Supports uniform multi-token queries (spec decode with uniform lengths)
    # - VARLEN: Supports variable-length queries (spec decode with mixed lengths)
    # If set to UNIFORM or VARLEN, this will increase `reorder_batch_threshold` when
    # speculative decoding is enabled.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    # The threshold for reordering the batch into decode and prefill requests.
    # If > 1, the batch will be reordered such that requests with
    # query length <= threshold are classified as decode requests.
    # Use `query_len_support` (above) to set this automatically
    # when speculative decoding is enabled.
    reorder_batch_threshold: int = 128  # process small prefills with decode pathway
    # \------------------------- Metax Modification -------------------------/

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self._dev_idx = self.device.index or 0
        global _ACTIVE_ATTN_GEOM
        _ACTIVE_ATTN_GEOM = (
            self.num_heads_q,
            self.num_heads_kv,
            self.headdim,
            self._dev_idx,
        )

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        # /------------------------  Metax Modification -------------------------\
        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode
        )

        # Validate consistency between query_len_support and reorder_batch_threshold
        if self.query_len_support == QueryLenSupport.SINGLE_ONLY:
            assert self.reorder_batch_threshold == 1, (
                f"reorder_batch_threshold must be 1 when query_len_support is "
                f"SINGLE_ONLY, got {self.reorder_batch_threshold}"
            )
        # \------------------------- Metax Modification -------------------------/

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None
        max_meta_elems = 2 * (vllm_config.scheduler_config.max_num_seqs + 1)
        self._cu_prefix_kv_lens = torch.zeros(
            max_meta_elems,
            dtype=torch.int32,
            device=self.device,
        )
        self._cu_prefix_kv_lens_cpu = torch.zeros(
            max_meta_elems,
            dtype=torch.int32,
            pin_memory=True,
        )
        self._cu_prefix_kv_lens_np = self._cu_prefix_kv_lens_cpu.numpy()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # /------------------------  Metax Modification -------------------------\
        # Partition the reordered V1 batch (`[decodes..., prefills...]`) purely on CPU numpy
        # views (`query_start_loc_cpu`, `seq_lens_cpu`) to avoid GPU->CPU syncs and tensor slices.
        qsl_np = common_attn_metadata.query_start_loc_cpu.numpy()
        sl_cpu_tensor = common_attn_metadata._seq_lens_cpu
        if sl_cpu_tensor is None:
            sl_cpu_tensor = common_attn_metadata.seq_lens_cpu
        sl_np = sl_cpu_tensor.numpy()

        decode_threshold = self.reorder_batch_threshold
        require_uniform = self.query_len_support != QueryLenSupport.VARLEN
        if max_query_len <= decode_threshold and (
            not require_uniform or decode_threshold <= 1
        ):
            num_decodes = num_reqs
            num_prefills = 0
            num_decode_tokens = num_actual_tokens
            num_prefill_tokens = 0
        else:
            q0_len = int(qsl_np[1] - qsl_np[0])
            if q0_len > decode_threshold:
                num_decodes = 0
                num_prefills = num_reqs
                num_decode_tokens = 0
                num_prefill_tokens = num_actual_tokens
            else:
                query_lens_np = qsl_np[1 : num_reqs + 1] - qsl_np[:num_reqs]
                if require_uniform:
                    is_prefill_np = query_lens_np != q0_len
                    has_prefill = bool((is_prefill_np & (query_lens_np != 0)).any())
                else:
                    is_prefill_np = query_lens_np > decode_threshold
                    has_prefill = bool(is_prefill_np.any())
                if not has_prefill:
                    num_decodes = num_reqs
                    num_prefills = 0
                    num_decode_tokens = num_actual_tokens
                    num_prefill_tokens = 0
                else:
                    first_prefill = int(np.argmax(is_prefill_np))
                    num_decodes = first_prefill
                    num_prefills = num_reqs - num_decodes
                    num_decode_tokens = int(qsl_np[first_prefill])
                    num_prefill_tokens = num_actual_tokens - num_decode_tokens
        # \------------------------- Metax Modification -------------------------/

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if self.use_full_cuda_graph and num_actual_tokens <= self.max_cudagraph_size:
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        # /------------------------  Metax Modification -------------------------\
        # Build decode slice metadata and publish `_LAST_DECODE_NUM_SPLITS` so the adaptive
        # FULL CUDAGraph wrapper selects the L2-resident `s_l2` vs wave-saturated `s_wave` graph.
        use_cascade = common_prefix_len > 0
        if num_decodes > 0:
            decode_max_seq_len = int(sl_np[:num_decodes].max())
            decode_query_start_loc = query_start_loc[: num_decodes + 1]
            decode_seq_lens = seq_lens[:num_decodes]
            decode_block_table_tensor = block_table_tensor[:num_decodes]
            if not _bi_mode and num_decode_tokens == num_decodes:
                decode_num_splits = solve_decode_splits(
                    num_decodes,
                    decode_max_seq_len,
                    self.num_heads_kv,
                    self.num_heads_q,
                    self.headdim,
                    self._dev_idx,
                )
            else:
                decode_num_splits = 1 if _bi_mode else 0
            global _LAST_DECODE_NUM_SPLITS
            _LAST_DECODE_NUM_SPLITS = decode_num_splits
        else:
            decode_max_seq_len = 0
            decode_query_start_loc = None
            decode_seq_lens = None
            decode_block_table_tensor = None
            decode_num_splits = 0

        use_kvcache_for_prefill = False
        prefill_split_transition = False
        prefill_cont_tokens = 0
        prefill_init_tokens = 0
        prefill_num_splits = 0
        prefill_cont_block_table = None
        prefill_cont_seq_lens = None
        prefill_cont_max_seq_len = 0
        prefill_cont_cu_q = None
        prefill_cont_cu_k = None

        if num_prefills > 0:
            # Classify prefill requests into initial vs continuation chunked regimes:
            # `prefill_is_initial` holds iff `num_prefill_tokens == sum(prefill_seq_lens)`
            # (zero cached prefix tokens), enabling direct linear `varlen_fwd` on K/V (`mode 3`).
            prefill_sl_np = sl_np[num_decodes:num_reqs]
            if num_prefills == 1:
                s0 = int(prefill_sl_np[0])
                prefill_max_seq_len = s0
                prefill_is_initial = num_prefill_tokens == s0
            elif num_prefills == 2:
                s0 = int(prefill_sl_np[0])
                s1 = int(prefill_sl_np[1])
                prefill_max_seq_len = s0 if s0 >= s1 else s1
                prefill_is_initial = num_prefill_tokens == (s0 + s1)
            else:
                prefill_max_seq_len = int(prefill_sl_np.max())
                prefill_is_initial = num_prefill_tokens == int(prefill_sl_np.sum())
            prefill_seq_lens = seq_lens[num_decodes:num_reqs]
            prefill_block_table_tensor = block_table_tensor[num_decodes:num_reqs]
            if prefill_is_initial:
                if num_decodes == 0:
                    prefill_query_start_loc = query_start_loc[: num_reqs + 1]
                elif num_prefills == 1:
                    # Reuse persistent `[0, L]` GPU row to avoid an H2D copy on mixed steps.
                    prefill_query_start_loc = _get_cu_single_len_table(
                        self.device, num_prefill_tokens
                    )[num_prefill_tokens]
                else:
                    n_p1 = num_prefills + 1
                    np.subtract(
                        qsl_np[num_decodes : num_reqs + 1],
                        num_decode_tokens,
                        out=self._cu_prefix_kv_lens_np[:n_p1],
                    )
                    cu_q = self._cu_prefix_kv_lens[:n_p1]
                    cu_q.copy_(self._cu_prefix_kv_lens_cpu[:n_p1], non_blocking=True)
                    prefill_query_start_loc = cu_q
                prefill_cu_seqlens_k = prefill_query_start_loc
            elif (
                num_prefills == 1
                and self.dcp_world_size <= 1
                and not use_cascade
                and not _bi_mode
            ):
                # Single continuation chunked prefill (`q_len < k_len`): `solve_chunked_prefill_splits`
                # returns `-1` when fused ContigKV gather + linear `varlen_fwd` (`mode 5`) beats
                # paged Split-KV `flash_attn_with_kvcache` (`mode 1`), both with zero H2D copies.
                use_kvcache_for_prefill = True
                prefill_num_splits = solve_chunked_prefill_splits(
                    num_prefill_tokens,
                    prefill_max_seq_len,
                    self.num_heads_q,
                    self.num_heads_kv,
                    self.headdim,
                    self.block_size,
                    self._dev_idx,
                )
                if prefill_num_splits == -1:
                    cu_tbl = _get_cu_single_len_table(
                        self.device, max(num_prefill_tokens, prefill_max_seq_len)
                    )
                    prefill_query_start_loc = cu_tbl[num_prefill_tokens]
                    prefill_cu_seqlens_k = cu_tbl[prefill_max_seq_len]
                else:
                    prefill_query_start_loc = None
                    prefill_cu_seqlens_k = None
            else:
                # Split-transition 2-prefill batch: when chunked scheduler packs the tail chunk
                # of an ongoing long prefill (`q0 < k0`, `k0 >= 2048`) together with the first
                # chunk of a brand-new request (`q1 == k1`), split them (`mode 6` or `mode 2`)
                # so `q0` uses ContigKV/Split-KV while `q1` runs linear `varlen_fwd` on uncached K/V.
                if (
                    num_prefills == 2
                    and self.dcp_world_size <= 1
                    and not use_cascade
                    and not _bi_mode
                ):
                    q0 = int(qsl_np[num_decodes + 1]) - num_decode_tokens
                    q1 = num_prefill_tokens - q0
                    k0 = int(prefill_sl_np[0])
                    k1 = int(prefill_sl_np[1])
                    if (
                        q0 > 0
                        and q1 > 0
                        and q0 < k0
                        and q1 == k1
                        and k0 >= 2048
                    ):
                        prefill_split_transition = True
                        prefill_cont_tokens = q0
                        prefill_init_tokens = q1
                        prefill_cont_max_seq_len = k0
                        prefill_num_splits = solve_chunked_prefill_splits(
                            q0,
                            k0,
                            self.num_heads_q,
                            self.num_heads_kv,
                            self.headdim,
                            self.block_size,
                            self._dev_idx,
                        )
                        prefill_cont_block_table = prefill_block_table_tensor[:1]
                        prefill_cont_seq_lens = prefill_seq_lens[:1]
                        cu_tbl = _get_cu_single_len_table(self.device, max(q0, q1, k0))
                        cu_q1 = cu_tbl[q1]
                        if prefill_num_splits == -1:
                            prefill_cont_cu_q = cu_tbl[q0]
                            prefill_cont_cu_k = cu_tbl[k0]
                        prefill_query_start_loc = cu_q1
                        prefill_cu_seqlens_k = cu_q1

                if not prefill_split_transition:
                    # General multi-request / paged continuation fallback (`prefill_mode == 4`):
                    # coalesce `cu_seqlens_q` and `cu_seqlens_k` into one pinned->GPU async copy (`gpu_both`).
                    n_p1 = num_prefills + 1
                    if num_decodes == 0:
                        prefill_query_start_loc = query_start_loc[:n_p1]
                        self._cu_prefix_kv_lens_np[0] = 0
                        np.cumsum(
                            prefill_sl_np,
                            dtype=np.int32,
                            out=self._cu_prefix_kv_lens_np[1:n_p1],
                        )
                        cu_k = self._cu_prefix_kv_lens[:n_p1]
                        cu_k.copy_(
                            self._cu_prefix_kv_lens_cpu[:n_p1], non_blocking=True
                        )
                        prefill_cu_seqlens_k = cu_k
                    else:
                        np.subtract(
                            qsl_np[num_decodes : num_reqs + 1],
                            num_decode_tokens,
                            out=self._cu_prefix_kv_lens_np[:n_p1],
                        )
                        self._cu_prefix_kv_lens_np[n_p1] = 0
                        np.cumsum(
                            prefill_sl_np,
                            dtype=np.int32,
                            out=self._cu_prefix_kv_lens_np[n_p1 + 1 : 2 * n_p1],
                        )
                        gpu_both = self._cu_prefix_kv_lens[: 2 * n_p1]
                        gpu_both.copy_(
                            self._cu_prefix_kv_lens_cpu[: 2 * n_p1], non_blocking=True
                        )
                        prefill_query_start_loc = gpu_both[:n_p1]
                        prefill_cu_seqlens_k = gpu_both[n_p1:]
        else:
            prefill_max_seq_len = 0
            prefill_query_start_loc = None
            prefill_seq_lens = None
            prefill_block_table_tensor = None
            prefill_cu_seqlens_k = None
            prefill_is_initial = False
        # \------------------------- Metax Modification -------------------------/

        if _bi_mode:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = MacaFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens

            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            # /------------------------  Metax Modification -------------------------\
            # For handling prefill decode split
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            decode_query_start_loc=decode_query_start_loc,
            decode_max_seq_len=decode_max_seq_len,
            decode_seq_lens=decode_seq_lens,
            decode_block_table=decode_block_table_tensor,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill_seq_lens=prefill_seq_lens,
            prefill_block_table=prefill_block_table_tensor,
            prefill_cu_seqlens_k=prefill_cu_seqlens_k,
            prefill_is_initial=prefill_is_initial,
            use_kvcache_for_prefill=use_kvcache_for_prefill,
            prefill_split_transition=prefill_split_transition,
            prefill_cont_tokens=prefill_cont_tokens,
            prefill_init_tokens=prefill_init_tokens,
            prefill_num_splits=prefill_num_splits,
            decode_num_splits=decode_num_splits,
            positions=common_attn_metadata.positions,
            prefill_cont_block_table=prefill_cont_block_table,
            prefill_cont_seq_lens=prefill_cont_seq_lens,
            prefill_cont_max_seq_len=prefill_cont_max_seq_len,
            prefill_cont_cu_q=prefill_cont_cu_q,
            prefill_cont_cu_k=prefill_cont_cu_k,
            # \------------------------- Metax Modification -------------------------/
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = _bi_mode

        if get_kv_quant_mode(self.kv_cache_dtype) != KVQuantMode.NONE and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self._is_encoder = attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER)
        self._is_fp8 = self.kv_cache_dtype.startswith("fp8")
        # Gate the MetaX (`head_size in (64, 128)`, power-of-2 `num_kv_heads <= 16`, BF16/FP16) fused
        # QK-RoPE + KV cache (+ optional ContigKV page gather) Triton kernel; `_fl_layer_ok` is
        # inspected by `RopeKvFusionPass` at compile time to emit `fl_rope_and_kv_cache_update`.
        self._can_fuse_rope_kv = (
            self.kv_sharing_target_layer_name is None
            and not self._is_encoder
            and not self._is_fp8
            and self.kv_cache_dtype in ("auto", "bfloat16", "float16")
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and self.head_size in (64, 128)
            and self.num_heads >= self.num_kv_heads
            and self.num_kv_heads >= 1
            and (self.num_kv_heads & (self.num_kv_heads - 1)) == 0
            and self.num_kv_heads <= 16
        )
        self._fl_layer_ok = bool(self._can_fuse_rope_kv)

    def _paged_prefill_splits(
        self, raw_splits: int, q_len: int, k_len: int, block_size: int, dev_idx: int
    ) -> int:
        """Resolve a valid `num_splits >= 1` for paged `fwd_kvcache` when ContigKV (`raw_splits == -1`) is bypassed."""
        if raw_splits >= 0:
            return raw_splits
        return solve_chunked_prefill_splits(
            q_len, k_len, self.num_heads, self.num_kv_heads, self.head_size, block_size, dev_idx, allow_contig=False
        )

    def supports_quant_query_input(self) -> bool:
        return False

    @staticmethod
    def _unbind_kv_cache(layer: Any, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """Cache `kv_cache.unbind(0)` views and strides on `layer` to avoid 42x view ops per step."""
        if getattr(layer, "_fl_kv_src", None) is kv_cache:
            key_cache, value_cache = layer._fl_kv_pair
            stride_cb, stride_cs, block_size = layer._fl_kv_strides
        else:
            key_cache, value_cache = kv_cache.unbind(0)
            stride_cb, stride_cs, block_size = key_cache.stride(0), key_cache.stride(1), key_cache.shape[1]
            layer._fl_kv_src = kv_cache
            layer._fl_kv_pair = (key_cache, value_cache)
            layer._fl_kv_strides = (stride_cb, stride_cs, block_size)
        return key_cache, value_cache, stride_cb, stride_cs, block_size

    def _build_or_get_exec_plan(
        self,
        attn_metadata: FlashAttentionMetadata,
        query: torch.Tensor,
        num_actual_tokens: int,
        stride_cb: int,
        stride_cs: int,
        block_size: int,
    ) -> tuple:
        """Build once per `(num_heads, num_kv_heads, head_size)` and memoize on `attn_metadata._fl_exec_plan`."""
        h_q, h_kv, d_h = self.num_heads, self.num_kv_heads, self.head_size
        geom_key = (h_q, h_kv, d_h)
        plans = attn_metadata._fl_exec_plan
        if plans is not None and (plan := plans.get(geom_key)) is not None:
            return plan
        q_ws = _get_q_rot_workspace(query.device, query.dtype, h_q, d_h, max(int(query.shape[0]), num_actual_tokens))
        q_out = q_ws[:num_actual_tokens]
        write_k_inplace = bool(attn_metadata.prefill_is_initial or attn_metadata.prefill_split_transition)
        num_decode_tokens, num_decodes, num_prefills = attn_metadata.num_decode_tokens, attn_metadata.num_decodes, attn_metadata.num_prefills
        mid_tok = num_decode_tokens + attn_metadata.prefill_cont_tokens
        cu_prefix_kv_lens = attn_metadata.prefill_cu_seqlens_k
        prefill_cont_bt, prefill_cont_sl = attn_metadata.prefill_cont_block_table, attn_metadata.prefill_cont_seq_lens
        gather_contig_kv, cont_end_tok, prefix_len, n_prefix_pages = False, 0, 0, 0
        k_contig, v_contig, gather_bt = query, query, attn_metadata.slot_mapping
        expected_cs = h_kv * d_h
        expected_cb = block_size * expected_cs
        contig_layout_ok = (
            block_size >= 16
            and stride_cs == expected_cs
            and stride_cb == expected_cb
            and expected_cb % 1024 == 0
        )

        # Classify prefill execution mode:
        #   0 = no prefills (pure decode)
        #   1 = single continuation chunk via paged Split-KV `fwd_kvcache`
        #   2 = split-transition 2-prefill (req 0 paged Split-KV + req 1 initial `varlen_fwd`)
        #   3 = initial prefill (`q_len == k_len`) via linear `varlen_fwd` on in-place rotated K/V
        #   4 = general multi-request / paged continuation via paged `varlen_fwd`
        #   5 = single continuation chunk via fused ContigKV gather + linear `varlen_fwd`
        #   6 = split-transition 2-prefill (req 0 fused ContigKV + req 1 initial `varlen_fwd`)
        if num_prefills <= 0:
            prefill_mode, q_prefill_view, q_init_view = 0, None, None
        elif attn_metadata.use_kvcache_for_prefill:
            if (
                attn_metadata.prefill_num_splits == -1
                and contig_layout_ok
                and attn_metadata.prefill_query_start_loc is not None
                and cu_prefix_kv_lens is not None
            ):
                prefill_mode, q_prefill_view, q_init_view, gather_contig_kv = 5, q_out[num_decode_tokens:num_actual_tokens], None, True
                cont_k_len, cont_end_tok = attn_metadata.prefill_max_seq_len, num_actual_tokens
                prefix_len = cont_k_len - attn_metadata.num_prefill_tokens
                n_prefix_pages, gather_bt = (prefix_len + block_size - 1) // block_size, attn_metadata.prefill_block_table
                kv_ws = _get_kv_contig_workspace(query.device, query.dtype, h_kv, d_h, cont_k_len)
                k_contig, v_contig = kv_ws[0][:cont_k_len], kv_ws[1][:cont_k_len]
            else:
                prefill_mode, q_prefill_view, q_init_view = 1, q_out[num_decode_tokens:num_actual_tokens].unsqueeze(0), None
        elif attn_metadata.prefill_split_transition:
            q_init_view = q_out[mid_tok:num_actual_tokens]
            if prefill_cont_bt is None and attn_metadata.prefill_block_table is not None:
                prefill_cont_bt = attn_metadata.prefill_cont_block_table = attn_metadata.prefill_block_table[:1]
            if prefill_cont_sl is None and attn_metadata.prefill_seq_lens is not None:
                prefill_cont_sl = attn_metadata.prefill_cont_seq_lens = attn_metadata.prefill_seq_lens[:1]
            if (
                attn_metadata.prefill_num_splits == -1
                and contig_layout_ok
                and attn_metadata.prefill_cont_cu_q is not None
                and attn_metadata.prefill_cont_cu_k is not None
            ):
                prefill_mode, q_prefill_view, gather_contig_kv = 6, q_out[num_decode_tokens:mid_tok], True
                cont_k_len, cont_end_tok = attn_metadata.prefill_cont_max_seq_len, mid_tok
                prefix_len = cont_k_len - attn_metadata.prefill_cont_tokens
                n_prefix_pages, gather_bt = (prefix_len + block_size - 1) // block_size, prefill_cont_bt
                kv_ws = _get_kv_contig_workspace(query.device, query.dtype, h_kv, d_h, cont_k_len)
                k_contig, v_contig = kv_ws[0][:cont_k_len], kv_ws[1][:cont_k_len]
            else:
                prefill_mode, q_prefill_view = 2, q_out[num_decode_tokens:mid_tok].unsqueeze(0)
        else:
            if cu_prefix_kv_lens is None:
                cu_prefix_kv_lens = attn_metadata.prefill_cu_seqlens_k = torch.nn.functional.pad(
                    attn_metadata.prefill_seq_lens.cumsum(dim=0, dtype=torch.int32), (1, 0)
                )
            prefill_mode, q_prefill_view, q_init_view = (3 if attn_metadata.prefill_is_initial else 4), q_out[num_decode_tokens:num_actual_tokens], None

        if num_decodes <= 0:
            decode_mode, q_decode_view = 0, None
        elif num_decode_tokens == num_decodes:
            decode_mode, q_decode_view = 1, q_out[:num_decode_tokens].unsqueeze(1)
        else:
            decode_mode, q_decode_view = 2, reshape_query_for_spec_decode(q_out[:num_decode_tokens], num_decodes)

        can_split_tok = h_q >= 2 and (h_q & (h_q - 1)) == 0
        split_tok = num_actual_tokens <= 64 and not gather_contig_kv and can_split_tok
        rope_warps = 2 if h_kv <= 8 else 4
        block_h = _select_rope_block_h(h_q, prefer_eight=(split_tok or gather_contig_kv or num_actual_tokens <= 128))
        if split_tok:
            rope_grid = (2 * num_actual_tokens,)
        elif gather_contig_kv or num_actual_tokens <= 128:
            rope_grid = (num_actual_tokens + n_prefix_pages,)
        else:
            rope_grid = (num_actual_tokens,)

        plan = (
            q_out, block_h, rope_warps, split_tok, rope_grid, write_k_inplace, prefill_mode, decode_mode,
            num_decode_tokens, num_decodes, mid_tok, q_prefill_view, q_init_view, q_decode_view,
            cu_prefix_kv_lens, prefill_cont_bt, prefill_cont_sl, gather_contig_kv, cont_end_tok,
            prefix_len, n_prefix_pages, k_contig, v_contig, gather_bt,
        )
        if plans is None:
            attn_metadata._fl_exec_plan = {geom_key: plan}
        else:
            plans[geom_key] = plan
        return plan

    def do_rope_and_kv_cache_update(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        attn_metadata: FlashAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """Execute fused QK-RoPE + paged KV scatter (+ optional ContigKV gather) before `forward()`.

        Invoked by `torch.ops.vllm.fl_rope_and_kv_cache_update` injected by `RopeKvFusionPass`.
        Writes rotated Q into `_Q_ROT_WORKSPACE` (`q_ret`, sharing `data_ptr()` with `plan[0]`) and
        returns `q_ret` as the `query` operand to `vllm::unified_attention_with_output`, allowing
        `forward()` to detect `query.data_ptr() == plan[0].data_ptr()` and skip redundant KV caching.
        """
        q_len = int(query.shape[0])
        n_act = int(attn_metadata.num_actual_tokens) if attn_metadata is not None else 0
        q_ws = _get_q_rot_workspace(
            query.device, query.dtype, self.num_heads, self.head_size, max(q_len, n_act)
        )
        q_ret = q_ws[:q_len]
        if attn_metadata is None or slot_mapping is None or kv_cache.numel() == 0:
            return q_ret.zero_()
        if cos_sin_cache.dtype != query.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=query.dtype)
        if not cos_sin_cache.is_contiguous():
            cos_sin_cache = cos_sin_cache.contiguous()

        key_cache, value_cache, stride_cb, stride_cs, block_size = self._unbind_kv_cache(layer, kv_cache)
        # Primary fast path: build/reuse `_fl_exec_plan` and launch `_fused_qk_rope_and_cache_kv_kernel`.
        if not attn_metadata.use_cascade and self.dcp_world_size <= 1 and not self._is_fp8 and self._can_fuse_rope_kv and is_neox:
            plan = self._build_or_get_exec_plan(attn_metadata, query, n_act, stride_cb, stride_cs, block_size)
            (
                q_out, block_h, rope_warps, split_tok, rope_grid, write_k_inplace, _, _,
                num_decode_tokens, _, _, _, _, _, _, _, _, gather_contig_kv, cont_end_tok,
                prefix_len, _, k_contig, v_contig, gather_bt,
            ) = plan
            if n_act > 0:
                _get_rope_kv_kernel()[rope_grid](
                    query, key, value, q_out, positions, cos_sin_cache, slot_mapping,
                    key_cache, value_cache, query.stride(0), key.stride(0), value.stride(0),
                    stride_cb, stride_cs, k_contig, v_contig, gather_bt,
                    n_act, num_decode_tokens, cont_end_tok, prefix_len,
                    BLOCK_SIZE=block_size, BLOCK_H=block_h, WRITE_K_INPLACE=write_k_inplace,
                    GATHER_CONTIG_KV=gather_contig_kv, SPLIT_TOKEN=split_tok,
                    NUM_Q_HEADS=self.num_heads, NUM_KV_HEADS=self.num_kv_heads,
                    HEAD_DIM=self.head_size, num_warps=rope_warps,
                )
            return q_ret

        # Semi-fused path for DCP / cascade attention: run fused QK-RoPE + KV scatter without `_fl_exec_plan`,
        # recording `_fl_kv_cached_ptr` so `forward()` does not re-run `reshape_and_cache_flash`.
        if self._can_fuse_rope_kv and is_neox:
            _launch_fused_qk_rope_and_cache_kv(
                query=query, key=key, value=value, positions=positions, cos_sin_cache=cos_sin_cache,
                slot_mapping=slot_mapping, key_cache=key_cache, value_cache=value_cache, num_actual_tokens=n_act,
                write_k_inplace=bool(attn_metadata.prefill_is_initial or attn_metadata.prefill_split_transition or self.dcp_world_size > 1),
                num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, head_size=self.head_size,
            )
            attn_metadata._fl_kv_cached_ptr = q_ret.data_ptr()
            return q_ret

        # Generic fallback for non-NeoX RoPE or non-`16:2x128` attention geometries.
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

        q_rot, k_rot = RotaryEmbedding.forward_static(
            positions, query[:n_act], key[:n_act], self.head_size, cos_sin_cache.shape[-1], cos_sin_cache, bool(is_neox)
        )
        q_ret[:n_act].copy_(q_rot)
        if k_rot is not None:
            key[:n_act].copy_(k_rot)
        reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        attn_metadata._fl_kv_cached_ptr = q_ret.data_ptr()
        return q_ret

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self._is_encoder:
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        key_cache, value_cache, stride_cb, stride_cs, block_size = self._unbind_kv_cache(layer, kv_cache)

        # Fast path: `RopeKvFusionPass` (`vllm::fl_rope_and_kv_cache_update`) already ran
        # `do_rope_and_kv_cache_update`, handing off `_Q_ROT_WORKSPACE` (`query.data_ptr() == plan[0].data_ptr()`)
        # with KV cache updated and (for `prefill_mode == 5/6`) linear `k_contig`/`v_contig` populated.
        plans = attn_metadata._fl_exec_plan
        plan = (
            plans.get((self.num_heads, self.num_kv_heads, self.head_size))
            if plans is not None
            else None
        )
        if plan is not None and query.data_ptr() == plan[0].data_ptr():
            (
                q_out, block_h, rope_warps, split_tok, rope_grid, write_k_inplace, prefill_mode, decode_mode,
                num_decode_tokens, num_decodes, mid_tok, q_prefill_view, q_init_view, q_decode_view,
                cu_prefix_kv_lens, prefill_cont_bt, prefill_cont_sl, gather_contig_kv, cont_end_tok,
                prefix_len, n_prefix_pages, k_contig, v_contig, gather_bt,
            ) = plan

            # Note: `prefill_mode == 0` (pure decode, `num_prefills == 0`) skips all prefill branches below.
            if prefill_mode == 5:
                # Single continuation chunked prefill via fused ContigKV gather + linear `varlen_fwd` (`kBlockM=128`).
                flash_attn_varlen_func(
                    q=q_prefill_view,
                    k=k_contig,
                    v=v_contig,
                    block_table=None,
                    cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                    cu_seqlens_k=cu_prefix_kv_lens,
                    max_seqlen_q=attn_metadata.num_prefill_tokens,
                    max_seqlen_k=attn_metadata.prefill_max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[num_decode_tokens:num_actual_tokens],
                )
            elif prefill_mode == 6:
                # Split-transition 2-prefill step: req 0 runs fused ContigKV `varlen_fwd`,
                # and req 1 (fresh initial prefill) runs linear `varlen_fwd` on in-place rotated `key`/`value`.
                flash_attn_varlen_func(
                    q=q_prefill_view,
                    k=k_contig,
                    v=v_contig,
                    block_table=None,
                    cu_seqlens_q=attn_metadata.prefill_cont_cu_q,
                    cu_seqlens_k=attn_metadata.prefill_cont_cu_k,
                    max_seqlen_q=attn_metadata.prefill_cont_tokens,
                    max_seqlen_k=attn_metadata.prefill_cont_max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[num_decode_tokens:mid_tok],
                )
                q1 = attn_metadata.prefill_init_tokens
                cu_q1 = attn_metadata.prefill_query_start_loc
                flash_attn_varlen_func(
                    q=q_init_view,
                    k=key[mid_tok:num_actual_tokens],
                    v=value[mid_tok:num_actual_tokens],
                    block_table=None,
                    cu_seqlens_q=cu_q1,
                    cu_seqlens_k=cu_q1,
                    max_seqlen_q=q1,
                    max_seqlen_k=q1,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[mid_tok:num_actual_tokens],
                )
            elif prefill_mode == 1:
                # Single continuation chunked prefill via paged Split-KV `flash_attn_with_kvcache`
                # (used on tail chunks where `104-CU` Split-KV beats `kBlockM=128` ContigKV).
                p_splits = self._paged_prefill_splits(
                    attn_metadata.prefill_num_splits,
                    attn_metadata.num_prefill_tokens,
                    attn_metadata.prefill_max_seq_len,
                    block_size,
                    query.device.index or 0,
                )
                flash_attn_with_kvcache(
                    q=q_prefill_view,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=attn_metadata.prefill_block_table,
                    cache_seqlens=attn_metadata.prefill_seq_lens,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    num_splits=p_splits,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[num_decode_tokens:num_actual_tokens].unsqueeze(0),
                )
            elif prefill_mode == 2:
                # Split-transition 2-prefill step: req 0 via paged Split-KV + req 1 via initial `varlen_fwd`.
                p_splits = self._paged_prefill_splits(
                    attn_metadata.prefill_num_splits,
                    attn_metadata.prefill_cont_tokens,
                    attn_metadata.prefill_cont_max_seq_len,
                    block_size,
                    query.device.index or 0,
                )
                flash_attn_with_kvcache(
                    q=q_prefill_view,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=prefill_cont_bt,
                    cache_seqlens=prefill_cont_sl,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    num_splits=p_splits,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[num_decode_tokens:mid_tok].unsqueeze(0),
                )
                q1 = attn_metadata.prefill_init_tokens
                cu_q1 = attn_metadata.prefill_query_start_loc
                flash_attn_varlen_func(
                    q=q_init_view,
                    k=key[mid_tok:num_actual_tokens],
                    v=value[mid_tok:num_actual_tokens],
                    block_table=None,
                    cu_seqlens_q=cu_q1,
                    cu_seqlens_k=cu_q1,
                    max_seqlen_q=q1,
                    max_seqlen_k=q1,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[mid_tok:num_actual_tokens],
                )
            elif prefill_mode == 3:
                # Initial prefill (`q_len == k_len`): stream linear in-place rotated `key` & `value` directly.
                flash_attn_varlen_func(
                    q=q_prefill_view,
                    k=key[num_decode_tokens:num_actual_tokens],
                    v=value[num_decode_tokens:num_actual_tokens],
                    block_table=None,
                    cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                    cu_seqlens_k=cu_prefix_kv_lens,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.prefill_max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[num_decode_tokens:num_actual_tokens],
                )
            elif prefill_mode == 4:
                # Multi-request / general continuation prefill reading paged `key_cache`/`value_cache` via `block_table`.
                flash_attn_varlen_func(
                    q=q_prefill_view,
                    k=key_cache,
                    v=value_cache,
                    block_table=attn_metadata.prefill_block_table,
                    cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                    cu_seqlens_k=cu_prefix_kv_lens,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.prefill_max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=output[num_decode_tokens:num_actual_tokens],
                )

            # Decode half (`decode_mode == 1`: 1-tok/req `[B, 1, H_q, D]`; `decode_mode == 2`: uniform multi-tok).
            if decode_mode > 0:
                decode_output = (
                    output[:num_decode_tokens].unsqueeze(1)
                    if decode_mode == 1
                    else reshape_query_for_spec_decode(
                        output[:num_decode_tokens], num_decodes
                    )
                )
                flash_attn_with_kvcache(
                    q=q_decode_view,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=attn_metadata.decode_block_table,
                    cache_seqlens=attn_metadata.decode_seq_lens,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    num_splits=attn_metadata.decode_num_splits,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks,
                    out=decode_output,
                )
            return output

        # Fallback path (`--enforce-eager`, cross-attention, DCP, cascade, or non-standard head sizes);
        # skip `reshape_and_cache_flash` when `do_rope_and_kv_cache_update` already populated `kv_cache` (`_fl_kv_cached_ptr`).
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and query.data_ptr() != attn_metadata._fl_kv_cached_ptr
        ):
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self._is_fp8:
            # queries are quantized in the attention layer
            dtype = MacaFlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            if self.dcp_world_size > 1:
                cu_seqlens_q = attn_metadata.query_start_loc
                descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                # ┌------------------------  Metax Modification -------------------------┐
                # For handling prefill decode split
                num_decode_tokens = attn_metadata.num_decode_tokens
                if attn_metadata.num_prefills > 0:
                    if attn_metadata.use_kvcache_for_prefill:
                        prefill_q_4d = query[
                            num_decode_tokens:num_actual_tokens
                        ].unsqueeze(0)
                        prefill_out_4d = output[
                            num_decode_tokens:num_actual_tokens
                        ].unsqueeze(0)
                        p_splits = self._paged_prefill_splits(
                            attn_metadata.prefill_num_splits,
                            attn_metadata.num_prefill_tokens,
                            attn_metadata.prefill_max_seq_len,
                            block_size,
                            query.device.index or 0,
                        )
                        flash_attn_with_kvcache(
                            q=prefill_q_4d,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            block_table=attn_metadata.prefill_block_table,
                            cache_seqlens=attn_metadata.prefill_seq_lens,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=self.sliding_window,
                            alibi_slopes=self.alibi_slopes,
                            num_splits=p_splits,
                            softcap=self.logits_soft_cap,
                            s_aux=self.sinks,
                            out=prefill_out_4d,
                        )
                    elif (
                        attn_metadata.prefill_split_transition
                        and key is not None
                        and not self.kv_cache_dtype.startswith("fp8")
                    ):
                        q0 = attn_metadata.prefill_cont_tokens
                        q1 = attn_metadata.prefill_init_tokens
                        mid_tok = num_decode_tokens + q0
                        p_splits = self._paged_prefill_splits(
                            attn_metadata.prefill_num_splits,
                            q0,
                            attn_metadata.prefill_cont_max_seq_len,
                            block_size,
                            query.device.index or 0,
                        )
                        flash_attn_with_kvcache(
                            q=query[num_decode_tokens:mid_tok].unsqueeze(0),
                            k_cache=key_cache,
                            v_cache=value_cache,
                            block_table=attn_metadata.prefill_block_table[:1],
                            cache_seqlens=attn_metadata.prefill_seq_lens[:1],
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=self.sliding_window,
                            alibi_slopes=self.alibi_slopes,
                            num_splits=p_splits,
                            softcap=self.logits_soft_cap,
                            s_aux=self.sinks,
                            out=output[num_decode_tokens:mid_tok].unsqueeze(0),
                        )
                        q_init = query[mid_tok:num_actual_tokens]
                        k_init = key[mid_tok:num_actual_tokens]
                        v_init = value[mid_tok:num_actual_tokens]
                        cu_q1 = attn_metadata.prefill_query_start_loc
                        flash_attn_varlen_func(
                            q=q_init,
                            k=k_init,
                            v=v_init,
                            block_table=None,
                            cu_seqlens_q=cu_q1,
                            cu_seqlens_k=cu_q1,
                            max_seqlen_q=q1,
                            max_seqlen_k=q1,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=self.sliding_window,
                            alibi_slopes=self.alibi_slopes,
                            softcap=self.logits_soft_cap,
                            s_aux=self.sinks,
                            out=output[mid_tok:num_actual_tokens],
                        )
                    else:
                        cu_prefix_kv_lens = attn_metadata.prefill_cu_seqlens_k
                        if cu_prefix_kv_lens is None:
                            cu_prefix_kv_lens = torch.nn.functional.pad(
                                attn_metadata.prefill_seq_lens.cumsum(
                                    dim=0, dtype=torch.int32
                                ),
                                (1, 0),
                            )
                            attn_metadata.prefill_cu_seqlens_k = cu_prefix_kv_lens
                        if (
                            attn_metadata.prefill_is_initial
                            and key is not None
                            and not self.kv_cache_dtype.startswith("fp8")
                        ):
                            prefill_k = key[num_decode_tokens:num_actual_tokens]
                            prefill_v = value[num_decode_tokens:num_actual_tokens]
                            prefill_block_table = None
                        else:
                            prefill_k = key_cache
                            prefill_v = value_cache
                            prefill_block_table = attn_metadata.prefill_block_table
                        flash_attn_varlen_func(
                            q=query[num_decode_tokens:num_actual_tokens],
                            k=prefill_k,
                            v=prefill_v,
                            block_table=prefill_block_table,
                            cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                            cu_seqlens_k=cu_prefix_kv_lens,
                            max_seqlen_q=attn_metadata.max_query_len,
                            max_seqlen_k=attn_metadata.prefill_max_seq_len,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=self.sliding_window,
                            alibi_slopes=self.alibi_slopes,
                            softcap=self.logits_soft_cap,
                            s_aux=self.sinks,
                            out=output[num_decode_tokens:num_actual_tokens],
                        )
                if attn_metadata.num_decodes > 0:
                    # Use flash_attn_with_kvcache for normal decoding.
                    decode_query = reshape_query_for_spec_decode(
                        query[:num_decode_tokens], attn_metadata.num_decodes
                    )
                    decode_output = reshape_query_for_spec_decode(
                        output[:num_decode_tokens], attn_metadata.num_decodes
                    )
                    flash_attn_with_kvcache(
                        q=decode_query,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        block_table=attn_metadata.decode_block_table,
                        cache_seqlens=attn_metadata.decode_seq_lens,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                        num_splits=attn_metadata.decode_num_splits,
                        softcap=self.logits_soft_cap,
                        s_aux=self.sinks,
                        out=decode_output,
                    )
                return output
            # └------------------------- Metax Modification -------------------------┘

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        # /------------------------  Metax Modification -------------------------\
        assert attn_metadata.dcp_context_kv_lens is not None
        cu_seqlens_k = torch.tensor(
            [0] + attn_metadata.dcp_context_kv_lens.tolist(),
            device=attn_metadata.dcp_context_kv_lens.device,
            dtype=torch.int32,
        ).cumsum(dim=0, dtype=torch.int32)

        context_attn_out, context_lse, _ = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_attn_probs=True,
            # scheduler_metadata=attn_metadata.scheduler_metadata,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=q_descale,
            # k_descale=k_descale,
            # v_descale=v_descale,
            s_aux=self.sinks,
        )
        # \------------------------- Metax Modification -------------------------/
        # FA returns LSE in shape [ H, B ] but cp_lse_ag_out_rs wants [ B, H ]
        context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        # /------------------------  Metax Modification -------------------------\
        query_attn_out, query_lse, _ = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            return_attn_probs=True,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=q_descale,
            # k_descale=k_descale,
            # v_descale=v_descale,
            s_aux=self.sinks,
        )
        # \------------------------- Metax Modification -------------------------/

        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        # Call flash attention directly on Q, K, V tensors
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            # fa_version=self.vllm_flash_attn_version,
            # q_descale=layer._q_scale.expand(descale_shape),
            # k_descale=layer._k_scale.expand(descale_shape),
            # v_descale=layer._v_scale.expand(descale_shape),
            s_aux=self.sinks,
            # num_splits=1 if self.batch_invariant_enabled else 0,
            out=output[:num_actual_tokens],
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # /------------------------  Metax Modification -------------------------\
    cu_prefix_kv_lens = torch.tensor(
        [0] + prefix_kv_lens.tolist(), device=prefix_kv_lens.device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    # \------------------------  Metax Modification -------------------------/

    # Process shared prefix.
    # /------------------------  Metax Modification -------------------------\
    prefix_output, prefix_lse, _ = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        cu_seqlens_k=cu_prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        s_aux=s_aux,
    )
    # \------------------------- Metax Modification -------------------------/

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])
    # /------------------------  Metax Modification -------------------------\
    cu_suffix_kv_lens = torch.tensor(
        [0] + suffix_kv_lens.tolist(), device=suffix_kv_lens.device, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    # \------------------------  Metax Modification -------------------------/

    # Process suffix per query.
    suffix_output, suffix_lse, _ = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_attn_probs=True,
        s_aux=s_aux,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
