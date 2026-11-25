# SPDX-License-Identifier: Apache-2.0
"""PPU FA3 ABI shared by ordinary attention and MLA prefill."""

from __future__ import annotations

import flash_attn_3._C  # noqa: F401 -- register the PPU operators
import torch


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
    # Version selector (ignored — we always use FA3)
    fa_version: int = 3,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
):
    """Custom flash_attn_varlen_func for PPU using the flash_attn_3 wheel.

    Accepts the same signature as vLLM's flash_attn_varlen_func (including
    the extra cp_* args), but calls torch.ops.flash_attn_3.fwd with the
    correct 35-argument signature.
    """
    del fa_version, cp_world_size, cp_rank, cp_tot_seqused_k  # unused
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
