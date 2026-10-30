# Copyright (c) 2026 BAAI. All rights reserved.
"""Independent FP64 oracle for the paged Ascend C4 indexer."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


def implementation():
    from vllm_fl.dispatch.backends.vendor.ascend.impl import quant_lightning_indexer

    return quant_lightning_indexer


def oracle(args):
    q, key, w, bt, req, pos, lengths, topk, qs, ks = args
    q, w, bt, req, pos, lengths = [x.cpu() for x in (q, w, bt, req, pos, lengths)]
    qs = qs.cpu() if qs is not None else None
    t, h, d = q.shape
    cap = bt.shape[1] * key.shape[1]
    result = torch.full((t, cap), -torch.inf, dtype=torch.float64)
    key_pages, scale_pages = {}, {}
    for row in range(t):
        r = int(req[row])
        if not 0 <= r < min(bt.shape[0], lengths.numel()) or pos[row] < 0:
            continue
        limit = min((int(pos[row]) + 1) // 4, int(lengths[r]), cap)
        qq = q[row].double()
        if qs is not None:
            qq = qq * qs[row].reshape(h, 1).double()
        for j in range(max(limit, 0)):
            page = int(bt[r, j // key.shape[1]])
            if not 0 <= page < key.shape[0] or (ks is not None and page >= ks.shape[0]):
                continue
            if page not in key_pages:
                key_pages[page] = key[page].cpu().double()
                if ks is not None:
                    scale_pages[page] = ks[page].cpu().double()
            kk = key_pages[page][j % key.shape[1]]
            if ks is not None:
                kk = kk * scale_pages[page][j % key.shape[1]]
            result[row, j] = (((qq * kk).sum(-1)).relu() * w[row].double()).sum()
    return result


def invoke(mod, args):
    q, k, w, bt, r, p, le, top, qs, ks = args
    return mod.deepseek_v4_indexer(
        q, k, w, bt, r, p, le, top, query_scale=qs, key_scale_cache=ks
    )


def check(args, *, exact_ids=False):
    mod = implementation()
    expected = oracle(args)
    q, k, w, bt, r, p, le, top, qs, ks = args
    score = mod._indexer_scores(q, k, w, bt, r, p, le, qs, ks).cpu().double()
    assert torch.equal(torch.isfinite(score), torch.isfinite(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(score[finite], expected[finite], atol=2e-3, rtol=4e-6)
    max_abs = (
        float((score[finite] - expected[finite]).abs().max()) if finite.any() else 0
    )
    out = invoke(mod, args).cpu()
    assert out.dtype == torch.int32 and out.shape == (q.shape[0], top)
    for row in range(q.shape[0]):
        ids = out[row][out[row] >= 0].long()
        count = min(top, int(finite[row].sum()))
        assert ids.numel() == count
        assert ids.unique().numel() == count
        if count:
            assert bool((ids < expected.shape[1]).all())
            assert bool(finite[row, ids].all())
            wanted = expected[row].topk(count).values
            actual = expected[row, ids]
            # This compares ranks and tie membership, without imposing tie IDs.
            torch.testing.assert_close(
                actual.sort(descending=True).values,
                wanted,
                atol=2 * max_abs + 1e-8,
                rtol=0,
            )
            if exact_ids:
                assert torch.equal(ids, expected[row].topk(count).indices)
        assert bool((out[row, count:] == -1).all())
    max_abs = (
        float((score[finite] - expected[finite]).abs().max()) if finite.any() else 0
    )
    print(f"case shape={tuple(q.shape)} topk={top} max_abs={max_abs}", flush=True)
    return out


def data(t=4, h=3, d=33, pages=6, block=8, cols=3, top=7, floating=False):
    gen = torch.Generator().manual_seed(345)
    dtype = torch.float32 if floating else torch.int8
    q = torch.randint(-7, 8, (t, h, d), generator=gen).to(dtype).npu()
    k = torch.randint(-7, 8, (pages, block, d), generator=gen).to(dtype).npu()
    if floating:
        q = torch.randn((t, h, d), generator=gen).npu()
        k = torch.randn((pages, block, d), generator=gen).npu()
    w = torch.rand((t, h), generator=gen).npu()
    bt = (
        torch.tensor([[0, 2, 4], [5, 3, 1]], dtype=torch.int64)[:, :cols]
        .contiguous()
        .npu()
    )
    r = torch.tensor([1, 0, 1, 0], dtype=torch.int64)[:t].npu()
    p = torch.tensor([95, 63, 3, -1], dtype=torch.int64)[:t].npu()
    le = torch.tensor([24, 17], dtype=torch.int64).npu()
    qs = None if floating else torch.rand(t, h, 1, generator=gen).npu()
    ks = None if floating else torch.rand(pages, block, 1, generator=gen).npu()
    return [q, k, w, bt, r, p, le, top, qs, ks]


def test_quantized_general_and_scale_invariant():
    a = data()
    initial = check(a)
    a[8].mul_(2)
    a[9].mul_(0.5)
    assert torch.equal(check(a), initial)


def test_float_and_mixed():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        a = data(floating=True)
        a[0] = a[0].to(dtype)
        a[1] = a[1].to(dtype)
        check(a)
    a = data()
    a[0] = a[0].float()
    check(a)
    a = data()
    a[1] = a[1].float()
    check(a)


def test_invalid_and_padding():
    a = data(top=39)
    a[3].copy_(torch.tensor([[0, -1, 2**40], [5, 2**40, -1]], device="npu"))
    a[4].copy_(torch.tensor([0, 1, -1, 2**40], device="npu"))
    a[5].copy_(torch.tensor([2**63 - 1, 95, 100, 100], device="npu"))
    check(a)
    a[4].fill_(-1)
    check(a)
    a = data()
    a[5].copy_(torch.tensor([0, 2, 3, 4], device="npu"))
    check(a)


def test_unique_scores_and_ties():
    a = data(t=1, h=1, d=1, pages=6, block=8, top=9)
    a[0].fill_(1)
    a[1].copy_(torch.arange(48, dtype=torch.int8).npu().reshape(6, 8, 1))
    a[2].fill_(1)
    a[4].zero_()
    a[5].fill_(95)
    a[6].fill_(24)
    a[8].fill_(1)
    a[9].fill_(1)
    check(a, exact_ids=True)
    a[1].zero_()
    check(a)


def test_empty_and_missing_scales():
    a = data()
    mod = implementation()
    for idx in (8, 9):
        b = list(a)
        b[idx] = None
        with pytest.raises(ValueError, match="scales"):
            invoke(mod, b)
    for idx in (0, 2, 4, 5, 8):
        a[idx] = a[idx][:0]
    assert invoke(mod, a).shape == (0, 7)
    a = data()
    a[3] = a[3][:, :0]
    check(a)


def test_model_shape_and_graph_replay():
    a = data(t=4, h=64, d=128, pages=40, block=32, top=512)
    a[3] = (torch.arange(64, device="npu").reshape(2, 32) % 40).long()
    a[5].fill_(4095)
    a[6].fill_(1024)
    check(a)
    mod = implementation()
    # Warm all kernels and allocations before capture.
    invoke(mod, a)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = invoke(mod, a)
    old = out.cpu()
    a[3].copy_((torch.arange(64).reshape(2, 32).flip(1) % 40).long().npu())
    a[4].copy_(torch.tensor([0, 1, 0, 1], device="npu"))
    a[5].copy_(torch.tensor([319, 255, 127, 383], device="npu"))
    a[6].copy_(torch.tensor([72, 88], device="npu"))
    a[0].neg_()
    a[1].neg_()
    a[8].mul_(0.75)
    a[9].mul_(0.5)
    graph.replay()
    torch.npu.synchronize()
    expected = check(a)
    assert torch.equal(out.cpu(), expected)
    assert not torch.equal(old, expected)
    for idx in (0, 1):
        previous = out.cpu()
        a[idx].neg_()
        graph.replay()
        torch.npu.synchronize()
        expected = check(a)
        assert torch.equal(out.cpu(), expected)
        assert not torch.equal(previous, expected)


def test_wide_high_pages():
    a = data(t=1, h=64, d=128, block=32, top=512)
    # A bounded 4.5GB parent reproduces both key and scale offsets >2**31.
    pages = 3000
    required = max(
        1409024 + (pages - 1) * 1496384 + 4096, 2 * (747520 + (pages - 1) * 748192 + 32)
    )
    if torch.npu.mem_get_info()[0] < required + 1024**3:
        pytest.skip("Wide-address gate requires 4.5GB plus 1GB free NPU memory")
    key_storage = torch.empty(required, dtype=torch.int8, device="npu")
    scale = key_storage.view(torch.float16)
    a[1] = key_storage.as_strided((pages, 32, 128), (1496384, 128, 1), 1409024)
    a[9] = scale.as_strided((pages, 32, 1), (748192, 1, 1), 747520)
    # Initialize just selected page views; never copy the large backing store.
    for i, page in enumerate([2997, 2998, 2999]):
        a[1][page].copy_(
            (torch.arange(4096, device="npu").reshape(32, 128) % 17 - 8 + i).to(
                torch.int8
            )
        )
        a[9][page].fill_(0.1 + i * 0.01)
    a[3] = torch.tensor([[2999, 2997, 2998]], device="npu")
    a[4].zero_()
    a[5].fill_(383)
    a[6] = torch.tensor([96], device="npu")
    assert (2999 * a[1].stride(0)) > 2**31 and (2999 * a[9].stride(0)) > 2**31
    check(a)


def test_topk_transport_exact_bits():
    mod = implementation()
    import triton

    # Every logical ID must survive subnormal FP32 bit transport, including
    # IDs beyond 256. Negative scores and ties exercise both comparisons.
    for cap, top in [(1024, 1024), (1301, 512)]:
        cpu = (
            torch.randperm(cap, generator=torch.Generator().manual_seed(91)).float()
            - cap / 2
        )
        cpu[::37] = 0
        score = cpu[None].npu()
        p = triton.next_power_of_2(top)
        ns = triton.cdiv(cap, p)
        v = torch.empty((1, ns, p), device="npu")
        i = torch.empty((1, ns, p), dtype=torch.int32, device="npu")
        mod._sort_segments[(ns,)](score, v, i, cap, p, ns, 0)
        while ns > 1:
            os = triton.cdiv(ns, 2)
            nv = torch.empty((1, os, p), device="npu")
            ni = torch.empty((1, os, p), dtype=torch.int32, device="npu")
            mod._merge[(os,)](v, i, nv, ni, p, ns, os, 0)
            v, i, ns = nv, ni, os
        ids = i.cpu().long()[0, 0, :top]
        expected = torch.argsort(cpu, descending=True, stable=True)[:top]
        assert torch.equal(ids, expected)


def test_strided_metadata_and_scales():
    a = data()
    for idx in (3, 4, 5, 6, 8):
        x = a[idx]
        expanded = torch.empty((*x.shape, 2), dtype=x.dtype, device=x.device)
        expanded[..., 0].copy_(x)
        a[idx] = expanded[..., 0]
    check(a)
    b = list(a)
    b[3] = b[3].float()
    with pytest.raises(ValueError, match="metadata"):
        invoke(implementation(), b)


def test_negative_weights_and_scales():
    a = data()
    a[2].sub_(0.5)
    a[8][::2].neg_()
    a[9][::2].neg_()
    check(a)


@pytest.mark.parametrize("dim", [33, 128])
def test_page_tile_fast_metadata_masks(dim):
    a = data(t=4, h=64, d=dim, pages=6, block=32, top=128)
    a[3].copy_(torch.tensor([[0, -1, 2**40], [5, 2**40, -1]], device="npu"))
    a[4].copy_(torch.tensor([0, 1, -1, 2**40], device="npu"))
    a[5].copy_(torch.tensor([2**63 - 1, 383, 383, 383], device="npu"))
    a[6].fill_(96)
    out = check(a)
    assert bool((out[0, :32] >= 0).all())  # Physical page zero is valid.
    assert bool((out[0, :32] < 32).all())
    a[4].zero_()
    a[5].fill_(2)
    check(a)  # No causal candidates: the whole fast tile is skipped.
    for limit in (31, 33, 63):
        a[3].copy_(torch.tensor([[0, 2, 4], [5, 3, 1]], device="npu"))
        a[5].fill_(limit * 4 - 1)
        check(a)  # A valid first row must retain the partial tile vector mask.
    a[5].fill_(383)
    a[3].fill_(-1)
    check(a)  # Every physical page is invalid.


@pytest.mark.parametrize("p", [32, 64, 512])
def test_sorted_merge_tuple_order(p):
    mod = implementation()
    for ns in (1, 3, 5):
        generator = torch.Generator().manual_seed(781 + p + ns)
        values = torch.randint(-17, 18, (2, ns, p), generator=generator).float() / 4
        flat = torch.arange(2 * ns * p).reshape(2, ns, p)
        # These IDs include subnormal and NaN FP32 bit patterns as well as
        # integers that cannot be represented numerically by FP32.
        ids = torch.where(
            flat % 3 == 0,
            2**24 + 2 * flat + 1,
            torch.where(flat % 3 == 1, 2**31 - 2 - 2 * flat, flat),
        ).to(torch.int32)
        lists = []
        for row in range(2):
            row_lists = []
            for seg in range(ns):
                pairs = sorted(
                    zip(values[row, seg].tolist(), ids[row, seg].tolist()),
                    key=lambda pair: (-pair[0], pair[1]),
                )
                row_lists.append(pairs)
                values[row, seg] = torch.tensor([pair[0] for pair in pairs])
                ids[row, seg] = torch.tensor([pair[1] for pair in pairs])
            lists.append(row_lists)
        os = (ns + 1) // 2
        expected = [
            [
                sorted(
                    sum(row[2 * seg : 2 * seg + 2], []),
                    key=lambda pair: (-pair[0], pair[1]),
                )[:p]
                for seg in range(os)
            ]
            for row in lists
        ]
        want_values = torch.tensor(
            [[[pair[0] for pair in seg] for seg in row] for row in expected]
        )
        want_ids = torch.tensor(
            [[[pair[1] for pair in seg] for seg in row] for row in expected],
            dtype=torch.int32,
        )
        out_values = torch.empty((2, os, p), device="npu")
        out_ids = torch.empty((2, os, p), dtype=torch.int32, device="npu")
        mod._launch_chunks(
            mod._merge, 2 * os, values.npu(), ids.npu(), out_values, out_ids, p, ns, os
        )
        assert torch.equal(out_values.cpu(), want_values)
        assert torch.equal(out_ids.cpu(), want_ids)
