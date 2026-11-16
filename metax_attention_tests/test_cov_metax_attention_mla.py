"""Targeted MetaX native Attention / MLA coverage tests on real C550 (cuda:0)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _require_metax(device):
    _ = device
    import vllm.utils.nccl  # noqa: F401

    from vllm_fl.platform import PlatformFL

    if PlatformFL.vendor_name != "metax":
        pytest.skip("MetaX-only native Attention/MLA coverage tests")


def _ref_causal_sdpa(q, k, v, scale, window_left=-1, softcap=0.0):
    _, lq, hq, d = q.shape
    _, lk, hk, _ = k.shape
    rep = hq // hk
    k_rep = k.repeat_interleave(rep, dim=2)
    v_rep = v.repeat_interleave(rep, dim=2)
    q_t = q[0].transpose(0, 1).float()
    k_t = k_rep[0].transpose(0, 1).float()
    v_t = v_rep[0].transpose(0, 1).float()
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    if softcap and softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)
    prefix = lk - lq
    q_pos = torch.arange(lq, device=q.device)[:, None] + prefix
    k_pos = torch.arange(lk, device=q.device)[None, :]
    mask = k_pos <= q_pos
    if window_left >= 0:
        mask = mask & (k_pos >= (q_pos - window_left))
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_t).transpose(0, 1).to(q.dtype)


class TestTritonUnifiedAttention:
    def test_pure_helpers(self):
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops import (
            triton_unified_attention as tua,
        )

        assert tua._is_gemma3_attention(128, 1024) is True
        assert tua._is_gemma3_attention(256, 1024) is True
        assert tua._is_gemma3_attention(64, 1024) is False
        assert tua._get_tile_size(128, 1024, 2, is_prefill=False) == 32
        assert tua._get_tile_size(64, 4096, 2, is_prefill=True) == 32
        assert tua._get_tile_size(64, 4096, 2, is_prefill=False) == 16
        assert tua._get_tile_size(64, 4096, 1, is_prefill=False) == 32

    def test_unified_attention_2d_prefill_decode_window_softcap(self, device):
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        torch.manual_seed(42)
        dtype = torch.float16
        num_q_heads, num_kv_heads, head_size = 8, 2, 64
        block_size = 16
        seq_lens = [19, 11]
        num_blocks = 8

        k_cache = (
            torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                device=device,
                dtype=dtype,
            )
            * 0.25
        )
        v_cache = (
            torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                device=device,
                dtype=dtype,
            )
            * 0.25
        )
        block_table = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)
        seqused_k = torch.tensor(seq_lens, device=device, dtype=torch.int32)

        total_q = sum(seq_lens)
        q = (
            torch.randn(total_q, num_q_heads, head_size, device=device, dtype=dtype)
            * 0.25
        )
        out = torch.empty_like(q)
        cu_seqlens_q = torch.tensor([0, 19, 30], device=device, dtype=torch.int32)
        scale = float(head_size**-0.5)

        unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max(seq_lens),
            seqused_k=seqused_k,
            max_seqlen_k=max(seq_lens),
            softmax_scale=scale,
            causal=True,
            window_size=(8, 0),
            block_table=block_table,
            softcap=30.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

        for i, sl in enumerate(seq_lens):
            s0, s1 = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            k_seq = torch.cat([k_cache[b] for b in block_table[i].tolist()], dim=0)[
                :sl
            ].unsqueeze(0)
            v_seq = torch.cat([v_cache[b] for b in block_table[i].tolist()], dim=0)[
                :sl
            ].unsqueeze(0)
            ref = _ref_causal_sdpa(
                q[s0:s1].unsqueeze(0), k_seq, v_seq, scale, window_left=8, softcap=30.0
            )
            torch.testing.assert_close(out[s0:s1], ref, rtol=2e-2, atol=2e-2)

    def test_unified_attention_3d_segmented_decode(self, device):
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        torch.manual_seed(7)
        dtype = torch.float16
        num_q_heads, num_kv_heads, head_size = 4, 2, 64
        block_size = 16
        kv_len = 48
        k_cache = (
            torch.randn(
                4, block_size, num_kv_heads, head_size, device=device, dtype=dtype
            )
            * 0.25
        )
        v_cache = (
            torch.randn(
                4, block_size, num_kv_heads, head_size, device=device, dtype=dtype
            )
            * 0.25
        )
        block_table = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
        q = torch.randn(1, num_q_heads, head_size, device=device, dtype=dtype) * 0.25
        out = torch.empty_like(q)
        cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        scale = float(head_size**-0.5)

        num_segs = 2
        segm_out = torch.empty(
            (1, num_q_heads, num_segs, head_size), device=device, dtype=torch.float32
        )
        segm_max = torch.empty(
            (1, num_q_heads, num_segs), device=device, dtype=torch.float32
        )
        segm_expsum = torch.empty(
            (1, num_q_heads, num_segs), device=device, dtype=torch.float32
        )

        unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=8,
            num_par_softmax_segments=num_segs,
            softmax_segm_output=segm_out,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
        )

        k_seq = torch.cat([k_cache[b] for b in [0, 1, 2]], dim=0)[:kv_len].unsqueeze(0)
        v_seq = torch.cat([v_cache[b] for b in [0, 1, 2]], dim=0)[:kv_len].unsqueeze(0)
        ref = _ref_causal_sdpa(q.unsqueeze(0), k_seq, v_seq, scale)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


class TestTritonDecodeAttention:
    @pytest.mark.parametrize(
        "num_q_heads,num_kv_heads,logit_cap", [(4, 4, 0.0), (8, 2, 20.0)]
    )
    def test_decode_attention_fwd_normal_and_grouped(
        self, device, num_q_heads, num_kv_heads, logit_cap
    ):
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.triton_decode_attention import (
            decode_attention_fwd,
        )

        torch.manual_seed(123)
        dtype = torch.float16
        batch = 2
        head_dim = 64
        seq_lens = [20, 12]
        max_len = max(seq_lens)
        total_slots = sum(seq_lens)
        num_kv_splits = 4
        sm_scale = float(head_dim**-0.5)

        q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=dtype) * 0.25
        k_buf = (
            torch.randn(total_slots, num_kv_heads, head_dim, device=device, dtype=dtype)
            * 0.25
        )
        v_buf = (
            torch.randn(total_slots, num_kv_heads, head_dim, device=device, dtype=dtype)
            * 0.25
        )
        out = torch.empty_like(q)
        lse = torch.empty(batch, num_q_heads, device=device, dtype=torch.float32)
        attn_logits = torch.empty(
            batch,
            num_q_heads,
            num_kv_splits,
            head_dim + 1,
            device=device,
            dtype=torch.float32,
        )
        req_to_token = torch.zeros(batch, max_len, device=device, dtype=torch.int32)
        req_to_token[0, :20] = torch.arange(0, 20, device=device, dtype=torch.int32)
        req_to_token[1, :12] = torch.arange(20, 32, device=device, dtype=torch.int32)
        b_seq_len = torch.tensor(seq_lens, device=device, dtype=torch.int32)

        decode_attention_fwd(
            q=q,
            k_buffer=k_buf,
            v_buffer=v_buf,
            o=out,
            lse=lse,
            req_to_token=req_to_token,
            b_seq_len=b_seq_len,
            attn_logits=attn_logits,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            page_size=1,
            logit_cap=logit_cap,
        )

        for i, sl in enumerate(seq_lens):
            idxs = req_to_token[i, :sl].long()
            ref = _ref_causal_sdpa(
                q[i : i + 1].unsqueeze(0),
                k_buf[idxs].unsqueeze(0),
                v_buf[idxs].unsqueeze(0),
                sm_scale,
                softcap=logit_cap,
            ).squeeze(0)
            torch.testing.assert_close(out[i], ref, rtol=2e-2, atol=2e-2)


class TestMacaFlashAndMLABackends:
    def test_maca_flash_backend_metadata_and_layout(self, device, monkeypatch):
        _ = device
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention import (
            flash_attn as fa,
        )

        assert fa.MacaFlashAttentionBackend.get_name() == "FLASH_ATTN"
        assert fa.MacaFlashAttentionBackend.get_impl_cls() is fa.FlashAttentionImpl
        assert (
            fa.MacaFlashAttentionBackend.get_builder_cls()
            is fa.FlashAttentionMetadataBuilder
        )
        assert (
            fa.MacaFlashAttentionBackend.supports_attn_type(fa.AttentionType.DECODER)
            is True
        )
        assert fa.MacaFlashAttentionBackend.supports_head_size(128) is True
        assert fa.MacaFlashAttentionBackend.supports_head_size(130) is False
        assert fa.MacaFlashAttentionBackend.supports_kv_cache_dtype(None) is True
        assert fa.MacaFlashAttentionBackend.supports_kv_cache_dtype("auto") is True
        assert (
            fa.MacaFlashAttentionBackend.supports_compute_capability(
                fa.DeviceCapability(8, 0)
            )
            is True
        )
        assert (
            fa.MacaFlashAttentionBackend.supports_combination(
                128,
                torch.float16,
                "auto",
                16,
                False,
                True,
                False,
                fa.DeviceCapability(8, 0),
            )
            is not None
        )
        assert (
            fa.MacaFlashAttentionBackend.supports_combination(
                128,
                torch.float16,
                "auto",
                16,
                False,
                False,
                False,
                fa.DeviceCapability(8, 0),
            )
            is None
        )

        assert fa.MacaFlashAttentionBackend.get_kv_cache_shape(4, 16, 2, 128) == (
            2,
            4,
            16,
            2,
            128,
        )
        with pytest.raises(ValueError, match="multiple of 16"):
            fa.MacaFlashAttentionBackend.get_kv_cache_shape(4, 12, 2, 128)
        with pytest.raises(NotImplementedError):
            fa.MacaFlashAttentionBackend.get_fp8_dtype_for_flashattn("fp8")

        monkeypatch.setattr(fa, "get_kv_cache_layout", lambda: "NHD")
        assert fa.MacaFlashAttentionBackend.get_kv_cache_stride_order(False) == (
            0,
            1,
            2,
            3,
            4,
        )
        assert fa.MacaFlashAttentionBackend.get_kv_cache_stride_order(True) == (
            2,
            0,
            1,
            3,
            4,
            5,
        )
        monkeypatch.setattr(fa, "get_kv_cache_layout", lambda: "HND")
        assert fa.MacaFlashAttentionBackend.get_kv_cache_stride_order(False) == (
            0,
            1,
            3,
            2,
            4,
        )
        assert fa.MacaFlashAttentionBackend.get_kv_cache_stride_order(True) == (
            2,
            4,
            0,
            1,
            3,
            5,
        )

        fake_cfg = NS(
            model_config=NS(is_hybrid=True),
            cache_config=NS(mamba_ssm_cache_dtype="float32", mamba_cache_dtype="auto"),
        )
        monkeypatch.setattr(fa, "get_current_vllm_config", lambda: fake_cfg)
        assert fa.MacaFlashAttentionBackend.get_supported_kernel_block_sizes() == [
            16,
            32,
            64,
        ]

        assert (
            fa.use_cascade_attention(
                common_prefix_len=512,
                query_lens=np.array([8] * 16),
                num_query_heads=32,
                num_kv_heads=4,
                use_alibi=False,
                use_sliding_window=False,
                use_local_attention=False,
                num_sms=64,
                dcp_world_size=1,
            )
            is True
        )
        assert (
            fa.use_cascade_attention(
                common_prefix_len=64,
                query_lens=np.array([8] * 16),
                num_query_heads=32,
                num_kv_heads=4,
                use_alibi=False,
                use_sliding_window=False,
                use_local_attention=False,
                num_sms=64,
                dcp_world_size=1,
            )
            is False
        )

    def test_mla_common_backend_and_reorg_kvcache(self, device, monkeypatch):
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla import (
            common as mc,
        )

        assert mc.MLACommonBackend.get_name() == "TRITON_MLA"
        assert mc.MLACommonBackend.get_builder_cls() is mc.MLACommonMetadataBuilder
        assert mc.MLACommonBackend.get_kv_cache_shape(4, 16, 1, 576) == (4, 16, 576)
        assert mc.MLACommonBackend.get_kv_cache_stride_order(False) == (0, 1, 2)
        assert mc.MLACommonBackend.get_kv_cache_stride_order(True) == (1, 0, 2, 3)
        assert mc.MLACommonBackend.get_supported_head_sizes() == [576]
        assert mc.MLACommonBackend.is_mla() is True

        kv_c = torch.arange(10 * 4, device=device, dtype=torch.float16).reshape(10, 4)
        k_pe = torch.arange(10 * 2, device=device, dtype=torch.float16).reshape(10, 2)
        out_kv, out_pe = mc.reorg_kvcache(
            allgatered_kv_c_normed=kv_c,
            allgatered_k_pe=k_pe,
            padded_local_chunk_seq_lens_lst=[3, 2],
            local_context_lens_allranks=[[2, 1], [2, 0]],
            sum_seq_len=5,
            max_seq_len=3,
            chunk_size=3,
            chunk_idx=0,
            toks=5,
        )
        assert out_kv.shape == (5, 4)
        assert out_pe.shape == (5, 2)
        _fake_vllm_cfg = lambda: NS(
            attention_config=NS(
                use_cudnn_prefill=False,
                use_trtllm_ragged_deepseek_prefill=False,
                disable_flashinfer_prefill=True,
            )
        )
        monkeypatch.setattr(
            mc, "get_current_vllm_config", _fake_vllm_cfg, raising=False
        )
        monkeypatch.setattr(
            "vllm.config.get_current_vllm_config", _fake_vllm_cfg, raising=False
        )
        monkeypatch.setattr(
            "vllm.config.vllm.get_current_vllm_config", _fake_vllm_cfg, raising=False
        )
        assert mc.use_flashinfer_prefill() is False
        assert mc.use_cudnn_prefill() is False
        assert mc.use_trtllm_ragged_deepseek_prefill() is False
        expected_idx = torch.tensor([0, 1, 5, 3, 4], device=device)
        torch.testing.assert_close(out_kv, kv_c[expected_idx])
        torch.testing.assert_close(out_pe, k_pe[expected_idx])
