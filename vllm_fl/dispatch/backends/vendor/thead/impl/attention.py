# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Thead (T-Head / PPU) FlashAttention backend.
#
# This module provides a custom attention backend for PPU accelerators that
# uses the flash_attn_3 wheel (torch.ops.flash_attn_3.fwd) directly.
#
# At module load time we:
#   1. Import flash_attn_3._C to register FA3 custom ops.
#   2. Provide a custom flash_attn_varlen_func that calls the wheel's fwd
#      with the wheel's 35-argument schema, including vLLM 0.28 mask arguments.
#   3. Inject the needed functions into the flash_attn module namespace so that
#      the inherited FlashAttentionImpl.forward() can resolve them.
#   4. Use masked Triton cache writes that preserve padding slots during graphs.
#   5. Handle PPU-specific requirements:
#      - When cu_seqlens_k is None (paged attention), max_seqlen_k must be 1.
#      - FA3 kernel uses max_seqlen_k to select tile size (Aone#75639039).

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 1 — load the flash_attn_3 wheel
# ---------------------------------------------------------------------------
import flash_attn_3._C  # noqa: F401 — registers torch.ops.flash_attn_3
import torch

# ---------------------------------------------------------------------------
# Step 2 — provide a custom flash_attn_varlen_func for PPU
# ---------------------------------------------------------------------------
# The vLLM wrapper accepts context parallel and FA4 mask arguments; the PPU
# flash_attn_3 wheel exposes the following 35-argument FA3 schema:
#   ... window_size_right, attention_chunk, softcap, is_rotary_interleaved,
#       scheduler_metadata, num_splits, pack_gqa, sm_margin, s_aux
#
# So we provide our own varlen wrapper that calls the wheel directly.


def _thead_flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector (the PPU wheel always uses FA3)
    fa_version: int = 3,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    dynamic_causal=None,
    mask_mod=None,
    aux_tensors=None,
):
    """Custom flash_attn_varlen_func for PPU using the flash_attn_3 wheel.

    Accepts the same signature as vLLM's flash_attn_varlen_func (including
    the extra cp_* args), but calls torch.ops.flash_attn_3.fwd with the
    correct 35-argument signature.
    """
    del fa_version
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("PPU FA3 does not support context parallel attention")
    if dynamic_causal is not None or mask_mod is not None or aux_tensors is not None:
        raise NotImplementedError("PPU FA3 does not support FA4 dynamic masks")
    del dropout_p, deterministic, return_attn_probs  # unused in FA3

    assert alibi_slopes is None, "Alibi is not supported in FA3"

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    # PPU Note Aone#75639039:
    # PPU FA3 uses max_seqlen_k to choose tile size.
    # In paged attention cu_seqlens_k is None, force max_seqlen_k = 1.
    if cu_seqlens_k is None:
        max_seqlen_k = 1

    out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
        q,
        k,
        v,
        None,
        None,  # k_new, v_new
        q_v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        None,
        seqused_k,  # seqused_q, seqused_k
        max_seqlen_q,
        max_seqlen_k,
        block_table,
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,
        None,
        None,  # rotary_cos, rotary_sin, seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        real_window_size[0],
        real_window_size[1],
        0,  # attention_chunk
        softcap,
        True,  # is_rotary_interleaved
        scheduler_metadata,
        num_splits,
        None,  # pack_gqa
        0,  # sm_margin
        s_aux,
    )

    return (out, softmax_lse) if return_softmax_lse else out


# ---------------------------------------------------------------------------
# Step 2b — inject into flash_attn module namespace
# ---------------------------------------------------------------------------
import vllm.v1.attention.backends.flash_attn as _flash_attn_mod

_flash_attn_mod.flash_attn_varlen_func = _thead_flash_attn_varlen_func


def _thead_get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens,
    qkv_dtype,
    headdim_v=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    cache_leftpad=None,
    page_size=None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),
    softcap=False,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
):
    """Bridge vLLM 0.28 metadata to the installed PPU FA3 operator schema."""
    return torch.ops.flash_attn_3.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim if headdim_v is None else headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,
        cu_seqlens_k_new,
        None,
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0],
        window_size[1],
        0,
        bool(softcap),
        num_splits,
        pack_gqa,
        sm_margin,
    )


_flash_attn_mod.get_scheduler_metadata = _thead_get_scheduler_metadata
# The upstream CC selector chooses FA2 on CC8.0. The PPU wheel implements
# FA3 on that device, including graph-safe AOT metadata scheduling.
_flash_attn_mod.get_flash_attn_version = lambda **kwargs: 3

# ---------------------------------------------------------------------------
# Step 2c — masked, stride-aware cache writes preserve graph padding slots.
# ---------------------------------------------------------------------------
from .cache import reshape_and_cache_flash_thead

_flash_attn_mod.reshape_and_cache_flash = reshape_and_cache_flash_thead

# ---------------------------------------------------------------------------
# Step 3 — custom backend & impl
# ---------------------------------------------------------------------------

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)


class TheadFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS


class TheadFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention implementation for PPU that uses FA3 (flash_attn_3 wheel).

    The only difference from FlashAttentionImpl:
    - vllm_flash_attn_version is forced to 3 (FA3) regardless of CC.
    """

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
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        # Override FA version to 3 — our custom flash_attn_varlen_func
        # handles the wheel call correctly.
        self.vllm_flash_attn_version = 3


class TheadFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend for PPU that delegates to TheadFlashAttentionImpl."""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TheadFlashAttentionImpl]:
        return TheadFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return TheadFlashAttentionMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # PPU CC = 8.0
        return capability >= DeviceCapability(8, 0) and capability < DeviceCapability(
            9, 0
        )

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink:
            return "sink not supported on PPU (CC < 9.0)"
        if use_mla:
            return "MLA not supported in thead flash attention backend"
        if use_sparse:
            return "sparse attention not supported in thead flash attention backend"
        if use_mm_prefix:
            return "multimodal prefix masks require FA4, unavailable on PPU FA3"
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return "quantized KV cache not supported by the PPU cache writer"
        return None
