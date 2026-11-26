"""Regression for empty-state softmax normalization, including sink semantics."""

import pytest
import torch

from vllm.v1.attention.ops import triton_unified_attention as upstream

from vllm_fl.attention.hybrid_selfcheck import _make_case
from vllm_fl.patches.triton_attention import patch_triton_attention_softmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize(
    "mode,with_sinks",
    [("mixed", True), ("decode_2d", False), ("decode_3d", True)],
)
def test_normalization(mode, with_sinks):
    patch_triton_attention_softmax()
    qlens = [5, 1, 1] if mode == "mixed" else [1, 1, 1]
    lengths = [qlens[0], 37, 9]
    case, mq, mk = _make_case(qlens, lengths, 16, "cuda:0")
    # Include a -inf sink (no mass), positive and negative finite sinks.
    sinks = (
        torch.tensor([float("-inf"), -2.0, 0.0, 2.0] * 4, device="cuda")
        if with_sinks
        else None
    )
    extra = {}
    if mode == "decode_3d":
        extra = dict(
            seq_threshold_3D=64,
            num_par_softmax_segments=4,
            softmax_segm_output=torch.empty((3, 16, 4, 128), device="cuda"),
            softmax_segm_max=torch.empty((3, 16, 4), device="cuda"),
            softmax_segm_expsum=torch.empty((3, 16, 4), device="cuda"),
        )
    upstream.unified_attention(
        **case,
        max_seqlen_q=mq,
        max_seqlen_k=mk,
        causal=True,
        window_size=(-1, -1),
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
        **extra,
    )
    actual = case["out"].cpu().float()
    q, k, v = (case[name].cpu().float() for name in ("q", "k", "v"))
    bt = case["block_table"].cpu().long()
    offset = 0
    for seq, (nq, nk) in enumerate(zip(qlens, lengths)):
        ids = bt[seq, : (nk + 15) // 16]
        keys = k[ids].flatten(0, 1)[:nk].repeat_interleave(8, dim=1).transpose(0, 1)
        values = v[ids].flatten(0, 1)[:nk].repeat_interleave(8, dim=1).transpose(0, 1)
        scores = (
            q[offset : offset + nq].transpose(0, 1) @ keys.transpose(1, 2)
        ) * 128**-0.5
        causal_mask = torch.arange(nk)[None, :] > (nk - nq + torch.arange(nq))[:, None]
        scores.masked_fill_(causal_mask[None, :, :], float("-inf"))
        if sinks is not None:
            scores = torch.cat(
                (scores, sinks.cpu()[:, None, None].expand(-1, nq, 1)), dim=-1
            )
            values = torch.cat((values, torch.zeros((16, 1, 128))), dim=1)
        expected = (scores.softmax(-1) @ values).transpose(0, 1)
        torch.testing.assert_close(
            actual[offset : offset + nq], expected, atol=3e-2, rtol=3e-2
        )
        offset += nq
