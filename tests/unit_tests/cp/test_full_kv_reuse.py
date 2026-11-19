"""Check prefix boundaries and exact full-head restoration/native cache writes."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.cp.layout import PagedKVPlan, ZigZagPlan


@pytest.mark.parametrize(
    "starts,lengths,expected",
    [
        ([0, 32], [32], True),
        ([0, 32], [64], False),
        ([0, 16, 32], [16, 16], False),
        ([0, 16, 32], [32, 16], False),
        ([0, 31], [31], True),
    ],
)
def test_complete_single_request_detection(starts, lengths, expected):
    plans = ZigZagPlan.build_all(starts, lengths, 4)
    assert all(p.single_full_prefill == expected for p in plans)
    if starts == [0, 31]:
        assert not plans[0].uniform_rows


def test_virtual_pages_are_bounded_cached_and_page_aligned():
    p = PagedKVPlan.contiguous_prefill(64, 16)
    assert p is PagedKVPlan.contiguous_prefill(64, 16)
    assert p.block_table == ((0, 1, 2, 3),)
    for n, block in ((63, 16), (0, 16), (64, 0)):
        with pytest.raises(ValueError):
            PagedKVPlan.contiguous_prefill(n, block)


@pytest.mark.parametrize(
    "lengths,prefix",
    [
        ((16, 48), 0),
        ((16, 16, 32), 0),
        ((8, 24), 0),
        ((16, 48), 16),
        ((16, 48), 1),
    ],
)
def test_complete_batch_requires_every_request_to_have_no_history(lengths, prefix):
    starts = tuple(accumulate(lengths, initial=0))
    kv_lengths = (lengths[0] + prefix, *lengths[1:])
    plans = ZigZagPlan.build_all(starts, kv_lengths, 4)
    assert all(
        p.full_prefill_lengths == (lengths if prefix == 0 else ()) for p in plans
    )
    assert not any(p.single_full_prefill for p in plans)


def test_virtual_pages_keep_request_offsets_and_padding_separate():
    lengths = (16, 48, 32)
    pages = PagedKVPlan.contiguous_prefills(lengths, 16)
    assert pages is PagedKVPlan.contiguous_prefills(lengths, 16)
    assert pages.block_ids == tuple(range(6))
    assert pages.block_table == ((0, 0, 0), (1, 2, 3), (4, 5, 0))
    # Same total token count with different boundaries must not share a table.
    other = PagedKVPlan.contiguous_prefills((48, 16, 32), 16)
    assert pages is not other
    assert other.block_table == ((0, 1, 2), (3, 0, 0), (4, 5, 0))
    rows = torch.arange(sum(lengths)).view(-1, 16)
    starts = tuple(accumulate(lengths, initial=0))
    for plan in ZigZagPlan.build_all(starts, lengths, 4):
        for i, req in enumerate(plan.request_ids):
            visible = rows[list(pages.block_table[req])].flatten()[: plan.seq_lens[i]]
            assert visible.tolist() == list(
                range(starts[req], starts[req] + plan.seq_lens[i])
            )
            local = plan.rows[plan.query_start_loc[i] : plan.query_start_loc[i + 1]]
            # Bottom-right causal alignment ends at this segment's last query.
            assert visible[-1].item() == local[-1]


@pytest.mark.parametrize(
    "lengths,block",
    [
        ((), 16),
        ((0, 16), 16),
        ((-16, 32), 16),
        ((16, 32), 0),
        ((8, 24), 16),
        ((31, 33), 16),
    ],
)
def test_virtual_batch_rejects_incomplete_request_pages(lengths, block):
    # Total alignment alone is insufficient: every request needs whole pages.
    with pytest.raises(ValueError, match="complete KV pages"):
        PagedKVPlan.contiguous_prefills(lengths, block)


@pytest.mark.gpu
@pytest.mark.parametrize("lengths", [(65535,), (65535, 1), (1, 17, 7), (16, 48)])
def test_partial_virtual_pages_copy_only_and_separate_requests(lengths):
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    block, heads, dim = 16, 4, 128
    pages = PagedKVPlan.contiguous_prefills(lengths, block, allow_partial=True)
    torch.manual_seed(36)
    k = torch.randn(sum(lengths), heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    pk, pv = pages.prefill_kv(k, v, lengths, block)
    # Copy every request into an independently built native paged reference.
    rk, rv = torch.zeros_like(pk), torch.zeros_like(pv)
    first = 0
    for req, length in enumerate(lengths):
        ids = list(pages.block_table[req][: (length + block - 1) // block])
        for dst, src in ((rk, k), (rv, v)):
            padded = torch.zeros(
                (len(ids) * block, heads, dim), device="cuda", dtype=k.dtype
            )
            padded[:length] = src[first : first + length]
            dst[ids] = padded.view(-1, block, heads, dim)
        first += length
    assert torch.equal(pk, rk) and torch.equal(pv, rv)
    # Attention must ignore all padding, including the one-token request after
    # an incomplete last page. Use one query per request to bound test cost.
    q = torch.randn(len(lengths), heads, dim, device="cuda", dtype=k.dtype)
    arguments = dict(
        q=q,
        cu_seqlens_q=torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int32),
        max_seqlen_q=1,
        seqused_k=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_seqlen_k=max(lengths),
        block_table=pages.table("cuda"),
        causal=True,
        fa_version=3,
        softmax_scale=dim**-0.5,
    )
    actual = flash_attn_varlen_func(k=pk, v=pv, **arguments)
    expected = flash_attn_varlen_func(k=rk, v=rv, **arguments)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "lengths,prefix,dtype,version,reuse",
    [
        ((64,), 0, torch.bfloat16, 3, True),
        ((16, 48), 0, torch.bfloat16, 3, True),
        ((16, 16, 32), 0, torch.bfloat16, 3, True),
        ((16, 48), 16, torch.bfloat16, 3, False),
        ((8, 24), 0, torch.bfloat16, 3, True),
        ((31, 33), 0, torch.bfloat16, 3, True),
        ((63, 1), 0, torch.bfloat16, 3, True),
        ((16, 48), 0, torch.float32, 3, False),
        ((16, 48), 0, torch.bfloat16, 2, False),
        ((2,), 0, torch.bfloat16, 3, True),
        ((2,), 16, torch.bfloat16, 3, False),
        ((3, 2), 16, torch.bfloat16, 3, False),
        ((7,), 0, torch.bfloat16, 3, True),
    ],
)
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("gated", [False, True])
def test_attention_checks_each_request_before_reusing_kv(
    monkeypatch, lengths, prefix, dtype, version, reuse, rank, gated
):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    from vllm_fl.cp import prstyle

    block, heads, dim = 16, 4, 8
    starts = tuple(accumulate(lengths, initial=0))
    kv_lengths = (lengths[0] + prefix, *lengths[1:])
    plans = ZigZagPlan.build_all(starts, kv_lengths, 4)
    plan = plans[rank]
    k = torch.randn(starts[-1], heads, dim, dtype=dtype)
    v = torch.randn_like(k)
    packed = torch.zeros(4 * plan.padded_rows, 2, heads, dim, dtype=dtype)
    for owner_rank, owner in enumerate(plans):
        begin = owner_rank * plan.padded_rows
        packed[begin : begin + len(owner.rows), 0] = owner.select(k)
        packed[begin : begin + len(owner.rows), 1] = owner.select(v)
    calls, writes, fa = [], [], []

    def gather(value, dim=0):
        calls.append(dim)
        return packed if dim == 0 else value.repeat(1, 1, 1, heads, 1)

    def attention(**kwargs):
        fa.append(kwargs)
        kwargs["out"].copy_(kwargs["q"])

    counts = tuple((n + block - 1) // block for n in kv_lengths)
    width = max(counts)
    # Deliberately noncontiguous native block IDs.
    native_table = torch.tensor(
        [[1 + 2 * (r * width + c) for c in range(width)] for r in range(len(lengths))],
        dtype=torch.int32,
    )
    cache = torch.zeros(2, 2 * len(lengths) * width + 1, block, 1, dim, dtype=dtype)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=starts[-1],
        max_query_len=max(lengths),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        max_seq_len=max(kv_lengths),
        seq_lens=torch.tensor(kv_lengths, dtype=torch.int32),
        block_table=native_table,
        slot_mapping=torch.arange(starts[-1]),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    layer = SimpleNamespace(
        total_num_kv_heads=heads,
        num_kv_heads=1,
        head_dim=dim,
        pr_group=SimpleNamespace(world_size=4, rank_in_group=rank, all_gather=gather),
        attn=SimpleNamespace(
            kv_cache=cache, impl=SimpleNamespace(vllm_flash_attn_version=version)
        ),
        scaling=dim**-0.5,
        _full_o_weight=torch.eye(heads * dim, dtype=dtype),
    )
    monkeypatch.setattr(
        prstyle, "unified_kv_cache_update", lambda *args: writes.append(args)
    )
    monkeypatch.setattr(prstyle, "flash_attn_varlen_func", attention)
    monkeypatch.setattr(
        prstyle, "get_forward_context", lambda: SimpleNamespace(additional_kwargs={})
    )
    q = torch.randn(len(plan.rows), heads, dim, dtype=dtype)
    gate = torch.randn(len(plan.rows), heads * dim, dtype=dtype) if gated else None
    result = prstyle.QwenPRStyleAttention._pr_attention.__wrapped__(
        layer, q, plan.select(k), plan.select(v), gate, plan, plans, metadata, "test"
    )
    expected = q.flatten(1)
    if gated:
        expected = expected * torch.sigmoid(gate)
    assert torch.equal(result, expected)
    assert result.shape == (len(plan.rows), heads * dim)
    # Empty query owners still gather K/V and write their native cache heads.
    assert calls == ([0] if reuse else [0, 3])
    assert len(writes) == 1
    assert len(fa) == bool(plan.rows)
    assert torch.equal(writes[0][0], k[:, rank : rank + 1])
    assert torch.equal(writes[0][1], v[:, rank : rank + 1])
    if reuse and plan.rows:
        pages = PagedKVPlan.contiguous_prefills(lengths, block, allow_partial=True)
        assert torch.equal(
            fa[0]["block_table"], pages.table("cpu")[list(plan.request_ids)]
        )
        for req, length in enumerate(lengths):
            first = starts[req]
            pages_for_req = list(
                pages.block_table[req][: (length + block - 1) // block]
            )
            for name, original in (("k", k), ("v", v)):
                paged = fa[0][name][pages_for_req].flatten(0, 1)
                assert torch.equal(paged[:length], original[first : first + length])
                assert torch.count_nonzero(paged[length:]) == 0


@pytest.mark.parametrize("heads,local_heads", [(2, 1), (4, 1), (8, 2)])
@pytest.mark.parametrize("lengths", [(64,), (16, 48), (16, 16, 32)])
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_heads_match_scattered_native_cache(heads, local_heads, lengths):
    from vllm import _custom_ops as ops

    n, dim, block = 64, 256, 16
    plans = ZigZagPlan.build_all(tuple(accumulate(lengths, initial=0)), lengths, 4)
    torch.manual_seed(51)
    source = torch.randn((n, 3, heads, dim), device="cuda", dtype=torch.bfloat16)
    k, v = source[:, 0], source[:, 2]
    packed = torch.cat([torch.stack((p.select(k), p.select(v)), dim=1) for p in plans])
    physical_blocks = torch.tensor([5, 2, 7, 1], device="cuda")
    slots = (
        physical_blocks[:, None] * block + torch.arange(block, device="cuda")
    ).flatten()
    cached_heads = []
    for rank, plan in enumerate(plans):

        def gather(value, dim=0, *, rank=rank):
            assert torch.equal(value, packed.chunk(4)[rank])
            return packed

        full_k, full_v = plan.gather_native_kv(
            plan.select(k), plan.select(v), 0, heads, gather
        )
        assert torch.equal(full_k, k) and torch.equal(full_v, v)
        first = rank * local_heads if heads >= 4 else rank // (4 // heads)
        nk, nv = (
            full_k[:, first : first + local_heads],
            full_v[:, first : first + local_heads],
        )
        cache = torch.zeros(
            (2, 8, block, local_heads, dim), device="cuda", dtype=k.dtype
        )
        scale = torch.ones((), device="cuda")
        ops.reshape_and_cache_flash(
            nk, nv, cache[0], cache[1], slots, "auto", scale, scale
        )
        compact = cache.index_select(1, physical_blocks)
        assert torch.equal(compact[0].flatten(0, 1), k[:, first : first + local_heads])
        assert torch.equal(compact[1].flatten(0, 1), v[:, first : first + local_heads])
        cached_heads.append(compact)
    if heads < 4:
        cached_heads = cached_heads[:: 4 // heads]
    original_full_cache = torch.cat(cached_heads, dim=3)
    assert torch.equal(original_full_cache[0], full_k.view(-1, block, heads, dim))
    assert torch.equal(original_full_cache[1], full_v.view(-1, block, heads, dim))
