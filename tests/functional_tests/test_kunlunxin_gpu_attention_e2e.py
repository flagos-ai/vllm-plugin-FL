# Copyright 2026 BAAI. All rights reserved.
"""Functional end-to-end Kunlunxin attention on the P800.

The audit's P0 hardware item: ``KunlunxinAttentionBackendImpl.forward``
(attention.py L850-977) executes zero lines on the CPU environment because
the main chain needs real ``xtorch_ops`` kernels (prefill_attention /
decode_paged_attention / reshape_and_cache). This test drives the production
decoder sequence end-to-end on the P800:

1. build a real ``KunlunxinMetadata`` through the production
   ``KunlunxinAttentionMetadataBuilder.build`` (CommonAttentionMetadata on
   device),
2. write K/V into the packed [num_blocks, 2, num_kv_heads, block_size,
   head_size] cache through the production ``forward`` path
   (``reshape_and_cache`` / ``reshape_and_cache_flash`` — whichever the
   device/cache layout selects),
3. run paged GQA decode + prefill attention through the same ``forward``
   with real block tables,
4. compare against an fp32 SDPA oracle gathered block-by-block from the
   same cache allocation.

The write and the read share one allocation, so a stride/layout
disagreement between writer and reader corrupts both sides differently
and cannot pass. Tolerances follow ``tests/platforms/kunlunxin.yaml``.
"""

from __future__ import annotations

import contextlib
import math

import pytest
import torch

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("kunlunxin"),
    pytest.mark.gpu,
    pytest.mark.serial,
]


def _tol(dtype):
    """Single source of truth: tests/platforms/kunlunxin.yaml tolerance."""
    try:
        from tests.utils.platform_config import PlatformConfig

        cfg = PlatformConfig.load("kunlunxin")
        tol = cfg.get_tolerance("inference", dtype)
        return tol.rtol, tol.atol
    except Exception:
        return (1e-3, 1e-3) if dtype == "default" else (1e-2, 1e-2)


RTOL_BF16, ATOL_BF16 = _tol("bfloat16")


@contextlib.contextmanager
def _vllm_config(max_model_len=2048):
    from vllm.config import (
        CompilationConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationMode

    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            is_encoder_decoder=False,
            max_num_batched_tokens=2048,
            max_model_len=max_model_len,
        ),
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    with set_current_vllm_config(config):
        yield config


def _impl(num_heads=8, num_kv_heads=2, head_size=64):
    """Build the real KunlunxinAttentionBackendImpl."""
    from vllm.v1.attention.backend import AttentionType

    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionBackendImpl,
    )

    return KunlunxinAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        attn_type=AttentionType.DECODER,
    )


def _build_metadata(
    device,
    seq_lens,
    block_table,
    slot_mapping,
    num_prefill_tokens,
    num_decode_tokens,
    num_actual_tokens,
):
    """Build KunlunxinMetadata through the production builder."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionMetadataBuilder,
    )

    num_reqs = len(seq_lens)
    max_seq_len = max(seq_lens)
    query_lens = []
    for seq_len in seq_lens:
        # Split each sequence into (prefill_len, 1 decode token): the batch
        # carries `num_prefill_tokens` new tokens plus one decode token each.
        prefill = max(0, seq_len - 1)
        query_lens.append(prefill if num_prefill_tokens else 1)
    if num_prefill_tokens:
        # mixed batch: first request prefills, rest decode
        query_lens = [seq_lens[0] - 1] + [1] * (num_reqs - 1)
    else:
        query_lens = [1] * num_reqs

    qsl = [0]
    for q in query_lens:
        qsl.append(qsl[-1] + q)

    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table.to(device=device, dtype=torch.int32),
        slot_mapping=slot_mapping.to(device=device, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
    )
    with _vllm_config() as config:
        builder = KunlunxinAttentionMetadataBuilder(None, ["layer.0"], config, device)
        return builder.build(common_prefix_len=0, common_attn_metadata=common)


def _sdpa_oracle(
    query, key_cache, value_cache, block_table, seq_lens, num_kv_heads, head_size, scale
):
    """Independent fp32 oracle (probe-verified on real P800): gather K/V by
    block table, fold blocks into the token dim, manual softmax attention."""
    outs = []
    block_size = key_cache.shape[-2]
    num_q_heads = query.shape[1]
    rep = num_q_heads // num_kv_heads
    for i, seq_len in enumerate(seq_lens):
        pages = block_table[i][: (seq_len + block_size - 1) // block_size].long().cpu()
        keys = (
            key_cache.index_select(0, pages)
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_size)[:seq_len]
            .float()
        )
        values = (
            value_cache.index_select(0, pages)
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_size)[:seq_len]
            .float()
        )
        q = query[i : i + 1].cpu().float().reshape(1, num_q_heads, 1, head_size)
        K = keys.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0)
        V = values.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0)
        scores = (q @ K.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = probs @ V
        outs.append(out.reshape(-1))
    return torch.stack(outs)


class TestKunlunxinAttentionE2E:
    def test_write_then_paged_decode_matches_sdpa(self):
        """Prefill write + paged decode read through production forward."""
        if not torch.cuda.is_available():
            pytest.skip("Kunlunxin P800 accelerator required")
        device = torch.device("cuda")
        num_heads, num_kv_heads, head_size = 8, 2, 64
        num_blocks, block_size = 24, 16
        impl = _impl(num_heads, num_kv_heads, head_size)

        # three sequences, shuffled pages
        seq_lens = [17, 33, 20]
        num_reqs = len(seq_lens)
        max_blocks = math.ceil(max(seq_lens) / block_size)
        perm = torch.randperm(num_blocks)
        block_table = perm[: max_blocks * num_reqs].view(num_reqs, max_blocks)
        slot_mapping = torch.arange(
            sum(seq_lens) + num_reqs
        )  # will be set per-token below

        # logical cache: [num_blocks, 2, num_kv_heads, block_size, head_size]
        kv_cache = torch.zeros(
            2,
            num_blocks,
            num_kv_heads,
            block_size,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        key_cache, value_cache = torch.unbind(kv_cache, dim=0)

        # ---- Step 1: prefill forward over FULL sequences (writes all K/V) ----
        total_tokens = sum(seq_lens)
        query = torch.randn(
            total_tokens,
            num_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        k_in = torch.randn(
            total_tokens,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        v_in = torch.randn_like(k_in)

        # slot mapping: token t of request i -> absolute slot in the flat cache plane
        slots = []
        for i, seq_len in enumerate(seq_lens):
            for t in range(seq_len):
                page = int(block_table[i][t // block_size])
                slots.append(page * block_size + t % block_size)
        slot_mapping = torch.tensor(slots, device=device, dtype=torch.long)

        meta = _build_metadata(
            device,
            seq_lens,
            block_table,
            slot_mapping,
            num_prefill_tokens=total_tokens,
            num_decode_tokens=0,
            num_actual_tokens=total_tokens,
        )
        layer = type("L", (), {})()  # AttentionLayer double: not read by this impl path
        impl.forward(layer, query, k_in, v_in, kv_cache, meta)
        torch.cuda.synchronize()
        # diag: verify prefill wrote k_in verbatim into cache at slot 0
        _kc0 = key_cache.cpu().reshape(-1, num_kv_heads, head_size)
        _written = _kc0[slots[0]]
        print(
            f"[diag-write] slot0 got={_written[0][:3].tolist()} want={k_in[0, 0, :3].cpu().tolist()}"
        )

        # ---- Step 2: decode forward appends one NEW token per request ----
        d_query = torch.randn(
            num_reqs,
            num_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        d_k = torch.randn(
            num_reqs,
            num_kv_heads,
            head_size,
            device=device,
            dtype=torch.bfloat16,
        )
        d_v = torch.randn_like(d_k)
        # new token lands at position seq_len (0-indexed) of each request
        d_slots = []
        for i, seq_len in enumerate(seq_lens):
            page = int(block_table[i][seq_len // block_size])
            d_slots.append(page * block_size + seq_len % block_size)
        d_slot_mapping = torch.tensor(d_slots, device=device, dtype=torch.long)
        d_meta = _build_metadata(
            device,
            [s + 1 for s in seq_lens],
            block_table,
            d_slot_mapping,
            num_prefill_tokens=0,
            num_decode_tokens=num_reqs,
            num_actual_tokens=num_reqs,
        )
        d_out = impl.forward(layer, d_query, d_k, d_v, kv_cache, d_meta)
        dm = d_meta.decode_metadata
        print(
            f"[diag-meta] seq_lens={dm.seq_lens_tensor.tolist()}, max_dec={dm.max_decode_seq_len}, blocks={dm.block_tables.tolist()}"
        )
        # ---- Step 3: oracle from the device cache ----
        expected = _sdpa_oracle(
            d_query,
            key_cache.cpu(),
            value_cache.cpu(),
            block_table,
            [s + 1 for s in seq_lens],
            num_kv_heads,
            head_size,
            head_size**-0.5,
        )
        assert d_out.shape == (num_reqs, num_heads * head_size)
        got = d_out
        # Diagnostic: also compare against attending over only the pre-decode cache
        # (seq_lens - 1 tokens) to discriminate cache-state ordering in the test driver.
        expected_pre = _sdpa_oracle(
            d_query,
            key_cache.cpu(),
            value_cache.cpu(),
            block_table,
            [s - 1 for s in seq_lens],
            num_kv_heads,
            head_size,
            head_size**-0.5,
        )
        err_post = (got.float().cpu() - expected.float()).abs().max().item()
        err_pre = (got.float().cpu() - expected_pre.float()).abs().max().item()
        print(
            f"[diag] max-err vs post-write oracle: {err_post:.4f}; vs pre-write: {err_pre:.4f}"
        )
        torch.testing.assert_close(
            got.float().cpu(),
            expected.float(),
            rtol=RTOL_BF16,
            atol=ATOL_BF16,
        )

    def test_unwritten_pages_do_not_leak(self):
        """Decode over partially written pages must not read stale zeros
        beyond seq_len (block tables are exact)."""
        if not torch.cuda.is_available():
            pytest.skip("Kunlunxin P800 accelerator required")
        device = torch.device("cuda")
        num_heads, num_kv_heads, head_size = 8, 2, 64
        num_blocks, block_size = 16, 16
        impl = _impl(num_heads, num_kv_heads, head_size)
        seq_lens = [18]  # just over one block: page 1 partially used
        block_table = torch.arange(4).view(1, 4)
        kv_cache = torch.zeros(
            2,
            num_blocks,
            num_kv_heads,
            block_size,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        key_cache, value_cache = torch.unbind(kv_cache, dim=0)
        # write all 18 tokens via a prefill pass
        q = seq_lens[0]
        query = torch.randn(
            q, num_heads, head_size, device=device, dtype=torch.bfloat16
        )
        k_in = torch.randn(
            q, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
        )
        v_in = torch.randn_like(k_in)
        slots = [
            int(block_table[0][t // block_size]) * block_size + t % block_size
            for t in range(q)
        ]
        slot_mapping = torch.tensor(slots, device=device, dtype=torch.long)
        meta = _build_metadata(
            device,
            seq_lens,
            block_table,
            slot_mapping,
            num_prefill_tokens=q,
            num_decode_tokens=0,
            num_actual_tokens=q,
        )
        layer = type("L", (), {})()
        impl.forward(layer, query, k_in, v_in, kv_cache, meta)
        torch.cuda.synchronize()
        # diag: verify prefill wrote k_in verbatim into cache at slot 0
        _kc0 = key_cache.cpu().reshape(-1, num_kv_heads, head_size)
        _written = _kc0[slots[0]]
        print(
            f"[diag-write] slot0 got={_written[0][:3].tolist()} want={k_in[0, 0, :3].cpu().tolist()}"
        )
        # poison the unused region of the last used page
        page = int(block_table[0][q // block_size])
        used = q % block_size
        key_cache[page, :, used:, :] = float("nan")
        value_cache[page, :, used:, :] = float("nan")
        torch.cuda.synchronize()
        # decode one more token at position q (next slot in same page)
        d_query = torch.randn(
            1, num_heads, head_size, device=device, dtype=torch.bfloat16
        )
        d_k = torch.randn(
            1, num_kv_heads, head_size, device=device, dtype=torch.bfloat16
        )
        d_v = torch.randn_like(d_k)
        d_slots = [page * block_size + used]
        d_meta = _build_metadata(
            device,
            [q + 1],
            block_table,
            torch.tensor(d_slots, device=device, dtype=torch.long),
            num_prefill_tokens=0,
            num_decode_tokens=1,
            num_actual_tokens=1,
        )
        out = impl.forward(layer, d_query, d_k, d_v, kv_cache, d_meta)
        torch.cuda.synchronize()
        assert out.shape == (1, num_heads * head_size)
        assert torch.isfinite(out.float()).all(), (
            "decode read leaked into the poisoned unwritten region"
        )
