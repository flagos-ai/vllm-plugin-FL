# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon FlashMLA adapters for the DeepSeek-V4 main sparse attention.

The FlashMLA wheel shipped for Hygon (``flash_mla`` 1.2.0+das) exposes sparse
prefill/decode kernels that read exactly the paged ``fp8_ds_mla`` layout the
FL plugin already writes:

- per 64-token SWA block: ``[0, bs*576)`` token data (448B fp8-e4m3 NoPE +
  128B bf16 RoPE) followed by ``[bs*576, +bs*8)`` UE8M0 scales
  (scale = 2^(byte-127), one byte per 64 dims);
- the compressed (topk) cache uses the same per-token layout.

So swapping the compute leaf requires no cache-format change.  Measured on
BW1000 (see ``bench_flash_mla_vs_ragged.py``): decode 2~15x and prefill
10~17x faster than vLLM's Triton ragged kernels at the fixed business shapes.

This module keeps the tensor-level adapters pure; layer-metadata plumbing
lives in ``patches.models.deepseek_v4._patch_sparse_attn_flash_mla``.
``NotSupported`` marks configuration misses that should permanently fall back
to the upstream Triton path.
"""

from __future__ import annotations

import functools
import logging
import weakref
from types import SimpleNamespace
from typing import Any

import torch
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

# DeepSeek-V4 sparse-MLA dims handled by the FlashMLA DV4 kernel.
_DV4_HEAD_DIM = 512
_DV4_NOPE_DIM = 448
_DV4_ROPE_DIM = 64
# FlashMLA sparse kernels require the per-query index width to be a multiple
# of 128; entries beyond ``topk_length`` are -1 sentinels the kernel skips.
# Note: no base-pointer alignment is required - verified empirically that a
# cache at data_ptr() % 576 != 0 still decodes correctly, matching the
# Hygon-native deployment on plain torch allocations.
_TOPK_ALIGN = 128


class NotSupported(RuntimeError):
    """Configuration the FlashMLA path cannot serve; caller falls back."""


@functools.lru_cache(maxsize=1)
def _resolve_flash_mla() -> Any | None:
    """Resolve the Hygon FlashMLA interface once; ``None`` disables the path."""
    try:
        from flash_mla.flash_mla_interface import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("FlashMLA unavailable on Hygon path (%s); using Triton ragged attention", exc)
        return None
    return SimpleNamespace(
        decode=flash_mla_with_kvcache,
        prefill=flash_mla_sparse_fwd,
        new_metadata=get_mla_metadata,
    )


class _SchedMetaCache:
    """FlashMLASchedMeta per (shape, step) configuration.

    flash_mla 的复用契约: shapes 与 topk_length/extra_topk_length 的**值**
    都不变时才可复用 sched_meta。decode 中 lens 随 context 每步增长, 因此
    复用范围是"同一步调用"：
      - 同 step 内同型层 lens 不变 → 命中, 每型只跑一次 in-kernel planner
        (与 vllm_hcu "每 step 每层型一次"的开销对齐)
      - 每 step lens 变化 → 首次调用 miss 重建, 正确性保持
    (此前按 shape 缓存导致跨 lens 复用, lens 越过规划阈值后偏差 >100%,
    生产表现为输出渐进退化; 每调用新建则 planner 每层每步都跑, +6% TPOT)

    步标识用调用方传入的 per-step 宿主对象身份(weakref 保活校验):
    ``DeepseekSparseSWAMetadataBuilder.build()`` 每个 scheduler step 构造新的
    metadata 实例, 同一步内所有层共享同一实例, 因此对象身份即步身份。
    相比旧的"lens 求和值做键"(每层两次 ``.sum().item()`` 设备同步)：
      - 同步消除: 不再阻塞 host(capture 期间该同步非法, 曾导致 flash_mla
        在图模式下永久回退到 Triton ragged)
      - 语义更严: 旧键存在"不同 lens 分布、相同求和"的碰撞风险, 身份键
        跨步不可能假命中
      - 捕获友好: 捕获用的也是当步新 metadata → 必然 miss → fresh meta
        (未初始化) → planner 连同 flash kernel 一起进图, replay 期重新规划
    """

    def __init__(self, capacity: int = 64) -> None:
        self._cache: dict[tuple, tuple[Any, Any]] = {}
        self._capacity = capacity

    def get(self, key: tuple, step_token: Any, make) -> Any:
        if step_token is None:
            # 无步标识(调用方未传) → 无法证明同 lens, 每次新建保正确性。
            return make()
        entry = self._cache.get(key)
        if entry is not None:
            step_ref, meta = entry
            # 同一 step 对象且仍存活 → 同 lens, 复用; weakref 已死或对象不同
            # (新 step) → miss 重建。
            if step_ref() is step_token:
                return meta
        meta = make()
        if len(self._cache) >= self._capacity:
            self._cache.clear()
        self._cache[key] = (weakref.ref(step_token), meta)
        return meta


_SCHED_META_CACHE = _SchedMetaCache()


def _pad_dense_indices(
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    align: int = _TOPK_ALIGN,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Right-pad the per-query index width to ``align`` with -1 sentinels.

    A no-op when the width already satisfies the alignment (the DV4 decode
    widths 128/512 and the prefill width 640 all do).
    """
    width = indices.shape[-1]
    if width % align == 0:
        return indices, lengths
    pad = align - width % align
    padded = torch.full(
        (*indices.shape[:-1], width + pad),
        -1,
        dtype=indices.dtype,
        device=indices.device,
    )
    padded[..., :width] = indices
    return padded, lengths


def _check_dv4_dims(head_dim: int, nope_head_dim: int, rope_head_dim: int) -> None:
    if (
        head_dim != _DV4_HEAD_DIM
        or nope_head_dim != _DV4_NOPE_DIM
        or rope_head_dim != _DV4_ROPE_DIM
    ):
        raise NotSupported(
            f"FlashMLA DV4 path expects head_dim=512/nope=448/rope=64, got "
            f"{head_dim}/{nope_head_dim}/{rope_head_dim}"
        )


def _check_paged_uint8(cache: torch.Tensor, name: str) -> None:
    if cache.dtype != torch.uint8:
        raise NotSupported(f"FlashMLA sparse path needs a uint8 {name}, got {cache.dtype}")


def _sink_slice(attn_sink: torch.Tensor | None, num_heads: int) -> torch.Tensor | None:
    if attn_sink is None:
        return None
    sink = attn_sink.reshape(-1)
    if sink.numel() < num_heads:
        raise NotSupported("attn_sink is shorter than the local head count")
    return sink[:num_heads].contiguous()


# ---------------------------------------------------------------------------
# Decode: flash_mla_with_kvcache(sparse fp8) with indices + extra topk cache
# ---------------------------------------------------------------------------


def sparse_attn_decode_flash_mla(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    swa_only: bool,
    topk_indices: torch.Tensor | None,
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    output: torch.Tensor,
    step_token: Any = None,
) -> None:
    """Sparse-MLA decode on the FlashMLA kernel; writes ``output`` in place.

    Accepts the dense index form of vLLM's ``rocm_sparse_attn_decode`` ABI
    (``swa_indices``/``topk_indices`` + lens); the ragged form is not handled
    here because producing dense indices upstream is cheaper than packing.

    ``step_token`` 携带 per-step 宿主对象(如 ``swa_metadata``)作为 sched_meta
    缓存的步标识, 见 ``_SchedMetaCache``。

    CUDA graph 兼容性(对齐 vllm_hcu 原生行为, 其 trace 中 planner 与 flash
    kernel 1:1 零间隙同图重放): 捕获时传入**未初始化**的 sched_meta, 捕获
    序列会同时包含 in-kernel planner 与 flash kernel —— 每次 replay planner
    从静态 buffer 读当前 lens 重新规划, schedule 不会陈旧。本函数全链路无
    host 同步(`.item()` 已随缓存键改造移除), 可安全在捕获中调用。
    """
    fm = _resolve_flash_mla()
    if fm is None:
        raise NotSupported("flash_mla is not importable")
    _check_dv4_dims(head_dim, nope_head_dim, rope_head_dim)
    if q.dim() != 3:
        raise NotSupported(f"decode expects q [B, H, {head_dim}], got {tuple(q.shape)}")
    batch, num_heads, _ = q.shape
    _check_paged_uint8(swa_k_cache, "SWA K cache")

    main_indices = swa_indices.reshape(batch, -1).to(torch.int32)
    main_lens = swa_lens.reshape(-1).to(torch.int32)
    main_indices, main_lens = _pad_dense_indices(main_indices, main_lens)

    extra_view = None
    extra_indices = None
    extra_lens = None
    extra_block = 0
    if not swa_only:
        if kv_cache is None or topk_indices is None or topk_lens is None:
            raise NotSupported("non-SWA decode needs the compressed cache and dense topk indices")
        _check_paged_uint8(kv_cache, "compressed K cache")
        extra_view = kv_cache.unsqueeze(-2)
        extra_indices = topk_indices.reshape(batch, -1).to(torch.int32)
        extra_lens = topk_lens.reshape(-1).to(torch.int32)
        extra_indices, extra_lens = _pad_dense_indices(extra_indices, extra_lens)
        extra_block = kv_cache.shape[1]

    # 缓存键 = shapes + 步标识(对象身份); lens 值本身不再读回 host。
    # 正确性: 同 step 同型层 lens 不变 → 命中; 新 step 新对象 → 必 miss。
    # 见 _SchedMetaCache 类注释。
    key = (
        batch,
        main_indices.shape[-1],
        swa_k_cache.shape[1],
        extra_indices.shape[-1] if extra_indices is not None else 0,
        extra_block,
    )
    meta = _SCHED_META_CACHE.get(key, step_token, lambda: fm.new_metadata()[0])

    out, _ = fm.decode(
        q=q.unsqueeze(1),
        k_cache=swa_k_cache.unsqueeze(-2),
        block_table=None,
        head_dim_v=head_dim,
        tile_scheduler_metadata=meta,
        cache_seqlens=None,
        is_fp8_kvcache=True,
        indices=main_indices.view(batch, 1, -1),
        topk_length=main_lens,
        softmax_scale=scale,
        attn_sink=_sink_slice(attn_sink, num_heads),
        extra_k_cache=extra_view,
        extra_indices_in_kvcache=(
            extra_indices.view(batch, 1, -1) if extra_indices is not None else None
        ),
        extra_topk_length=extra_lens,
    )
    output.copy_(out.squeeze(1).to(output.dtype))


# ---------------------------------------------------------------------------
# Prefill: flash_mla_sparse_fwd on the bf16 gathered KV workspace
# ---------------------------------------------------------------------------


def sparse_attn_prefill_flash_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    attn_sink: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    """Sparse-MLA prefill on the FlashMLA kernel; writes ``output`` in place.

    ``kv`` is the bf16 gather workspace vLLM's prefill already materializes
    (``[skv, 1, head_dim]``); ``indices``/``topk_length`` are the dense
    combined topk+SWA lists.
    """
    fm = _resolve_flash_mla()
    if fm is None:
        raise NotSupported("flash_mla is not importable")
    _check_dv4_dims(head_dim, nope_head_dim, rope_head_dim)
    if q.dim() != 3:
        raise NotSupported(f"prefill expects q [T, H, {head_dim}], got {tuple(q.shape)}")
    if kv.dim() != 3 or kv.shape[1] != 1:
        raise NotSupported(f"prefill expects kv [skv, 1, {head_dim}], got {tuple(kv.shape)}")
    num_tokens, num_heads, _ = q.shape

    indices = indices.reshape(num_tokens, -1).to(torch.int32)
    if topk_length is not None:
        topk_length = topk_length.reshape(-1).to(torch.int32)
    indices, topk_length = _pad_dense_indices(indices, topk_length)

    out, _, _ = fm.prefill(
        q=q,
        kv=kv,
        indices=indices.view(num_tokens, 1, -1),
        sm_scale=scale,
        d_v=head_dim,
        attn_sink=_sink_slice(attn_sink, num_heads),
        topk_length=topk_length,
    )
    if out.dim() == 4:
        out = out.squeeze(1)
    output.copy_(out.to(output.dtype))


# ---------------------------------------------------------------------------
# Dense topk mapping for C4A decode (ratio 4)
# ---------------------------------------------------------------------------


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV slots and count valid entries.

    Dense counterpart of vLLM's ``compute_global_topk_ragged_indices_and_indptr``
    with the same inputs; keeps a single fused Triton kernel (block-table
    lookup + valid counting + padding masking) so the FlashMLA decode path
    gets dense indices without an extra ragged->dense scatter.
    Kernel adapted from the Hygon vLLM deepseek_v4_ops implementation.
    """
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """One program per token: block-table lookup + valid counting + masking."""
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    if not is_valid_token:
        for i in range(0, topk, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            mask = offset < topk
            tl.store(
                global_topk_indices_ptr
                + token_idx * global_topk_indices_stride
                + offset,
                -1,
                mask=mask,
            )
        tl.store(topk_lens_ptr + token_idx, 0)
        return

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        block_indices = local_idx // block_size
        is_valid = (local_idx >= 0) & (block_indices < block_table_stride)
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & is_valid,
        )
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, count)
