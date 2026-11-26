"""INT8 split-K decode kernel against a CPU FP32 reference on dequantized values."""

import pytest
import torch
import triton
import triton.language as tl

from vllm_fl.attention import int8_decode_config as configs, kv_int8 as k8
from vllm_fl.attention.kv_int8 import _int8_to_bf16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SCALE = 128**-0.5


@triton.jit
def _convert_int8_values(X, Y):
    i = tl.arange(0, 256)
    tl.store(Y + i, _int8_to_bf16(tl.load(X + i)))


def build(lengths, qlens=None, seed=0):
    """Random quantized pool with shuffled block tables; returns (case, cache)."""
    g = torch.Generator().manual_seed(seed)
    b = len(lengths)
    npages = [(n + 15) // 16 for n in lengths]
    total = max(sum(npages), 1)
    pool = total + 3
    k = (torch.randn(pool * 16, 2, 128, generator=g) * 0.7).to(torch.bfloat16).cuda()
    v = (torch.randn(pool * 16, 2, 128, generator=g) * 0.7).to(torch.bfloat16).cuda()
    raw = torch.empty(pool * k8.PAGE_BYTES, dtype=torch.int8, device="cuda")
    cache = torch.as_strided(
        raw, (pool, 2, 16, 2, 128), (k8.PAGE_BYTES, 4096, 256, 128, 1)
    )
    k8.kv_int8_write(
        k, v, cache, torch.arange(pool * 16, device="cuda", dtype=torch.int64)
    )
    perm = torch.randperm(pool, generator=g)  # shuffled physical pages
    width = max(max(npages), 1) + 2
    bt = torch.zeros((b, width), dtype=torch.int32)
    at = 0
    for i, n in enumerate(npages):
        bt[i, :n] = perm[at : at + n].to(torch.int32)
        at += n
    qlens = qlens or [1] * b
    cu = torch.zeros(b + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(qlens).cumsum(0)
    q = (torch.randn(int(cu[-1]), 16, 128, generator=g)).to(torch.bfloat16)
    case = dict(
        q=q.cuda(),
        bt=bt.cuda(),
        lens=torch.tensor(lengths, dtype=torch.int32).cuda(),
        cu=cu.cuda(),
        out=torch.full_like(q, float("nan")).cuda(),
    )
    return case, cache


def reference(case, cache):
    """FP32 attention on the dequantized cache, per request and KV head."""
    scales = k8.scale_view(cache).cpu().float()  # (P, 2 kv, 2 head, 16)
    vals = cache.cpu().float()  # (P, 2 kv, 16, 2 head, 128)
    kd = vals[:, 0] * scales[:, 0].permute(0, 2, 1)[..., None]  # (P, 16, 2, 128)
    vd = vals[:, 1] * scales[:, 1].permute(0, 2, 1)[..., None]
    q, bt = case["q"].cpu().float(), case["bt"].cpu().long()
    lens, cu = case["lens"].cpu().tolist(), case["cu"].cpu().tolist()
    out = torch.full_like(q, float("nan"))
    for i, n in enumerate(lens):
        if n == 0 or cu[i + 1] - cu[i] != 1:
            continue
        pages = bt[i, : (n + 15) // 16]
        kk = kd[pages].reshape(-1, 2, 128)[:n]
        vv = vd[pages].reshape(-1, 2, 128)[:n]
        for h in range(2):
            qh = q[cu[i], h * 8 : (h + 1) * 8]
            p = torch.softmax(qh @ kk[:, h].T * SCALE, -1)
            out[cu[i], h * 8 : (h + 1) * 8] = p @ vv[:, h]
    return out


def run(case, cache, **kw):
    ws = k8.DecodeWorkspace("cuda:0")
    k8.kv_int8_decode(
        case["q"],
        cache,
        k8.half_base(cache),
        case["out"],
        case["cu"],
        case["lens"],
        case["bt"],
        SCALE,
        ws,
        **kw,
    )
    torch.cuda.synchronize()
    return ws


def check(case, cache, expected, variant):
    got = case["out"].cpu().float()
    live = ~torch.isnan(expected)
    tol = 3e-2 if variant == 1 else 1.5e-2
    assert torch.isfinite(got[live]).all()
    torch.testing.assert_close(got[live], expected[live], atol=tol, rtol=tol)


def test_int8_to_bf16_exact_all_values():
    values = torch.arange(-128, 128, dtype=torch.int16, device="cuda").to(torch.int8)
    got = torch.empty((256,), dtype=torch.bfloat16, device="cuda")
    _convert_int8_values[(1,)](values, got, num_warps=1)
    torch.cuda.synchronize()
    assert torch.equal(got, values.to(torch.bfloat16))


def test_high_dynamic_range_query_preserves_bf16_semantics():
    source, cache = configs._high_dynamic_query_case("cuda:0")
    case = {
        "q": source["q"],
        "bt": source["block_table"],
        "lens": source["seqused_k"],
        "cu": source["cu_seqlens_q"],
        "out": source["out"],
    }
    expected = reference(case, cache)
    assert torch.allclose(expected, torch.ones_like(expected), atol=1e-3, rtol=1e-3)
    case["out"].fill_(float("nan"))
    run(case, cache, segments=1, tile=32, num_warps=1, num_stages=1, variant=8)
    check(case, cache, expected, 8)


def test_fallback_variant_uses_reconstructed_bf16_operands():
    # Quantized-space cancellation differs from dotting reconstructed BF16 K.
    case, cache = build([2], seed=3)
    cache.zero_()
    scales = k8.scale_view(cache)
    scales.fill_(0.01)
    case["bt"].zero_()
    case["q"].zero_()
    case["q"][:, :, 0] = 65536
    case["q"][:, :, 1] = -66560
    cache[0, 0, 0, :, 0] = 65
    cache[0, 0, 0, :, 1] = 64
    cache[0, 1, 0].fill_(1)
    cache[0, 1, 1].fill_(-1)
    kd = (
        (cache[0, 0, :2].cpu().float() * scales[0, 0, :, :2].T.cpu().float()[..., None])
        .bfloat16()
        .float()
    )
    vd = (
        (cache[0, 1, :2].cpu().float() * scales[0, 1, :, :2].T.cpu().float()[..., None])
        .bfloat16()
        .float()
    )
    q = case["q"].cpu().float()[0]
    expected = (
        torch.stack(
            [
                torch.softmax(q[h * 8 : (h + 1) * 8] @ kd[:, h].T * SCALE, -1)
                @ vd[:, h]
                for h in range(2)
            ]
        )
        .flatten(0, 1)
        .unsqueeze(0)
    )
    assert (expected < 0).all()
    for splits in (1, 4):
        run(case, cache, segments=splits, tile=32, num_warps=2, num_stages=1, variant=3)
        torch.testing.assert_close(
            case["out"].cpu().float(), expected, atol=1e-5, rtol=1e-4
        )


def test_selected_variant_handles_page_tail_and_long_context_splits():
    for length in (17, 16384):
        case, cache = build([length], seed=length)
        expected = reference(case, cache)
        for splits in (1, 8, 64):
            case["out"].fill_(float("nan"))
            run(
                case,
                cache,
                segments=splits,
                tile=32,
                num_warps=1,
                num_stages=1,
                variant=8,
            )
            check(case, cache, expected, 8)


def test_selected_and_fallback_variants_handle_mixed_lengths():
    case, cache = build([1, 15, 16, 17, 4095, 4096], seed=12)
    expected = reference(case, cache)
    for variant in (3, 8):
        for splits in (1, 16):
            case["out"].fill_(float("nan"))
            run(
                case,
                cache,
                segments=splits,
                tile=32,
                num_warps=1,
                num_stages=1,
                variant=variant,
            )
            check(case, cache, expected, variant)


def test_empty_rows_untouched():
    variant = 8
    lengths = [0, 33, 0, 1, 0, 17]
    qlens = [1, 1, 1, 1, 3, 1]  # row 4 is not a decode row: skipped
    lengths_k = list(lengths)
    case, cache = build(lengths_k, qlens=qlens, seed=5)
    expected = reference(case, cache)
    for splits in (1, 8):
        case["out"].fill_(float("nan"))
        run(
            case,
            cache,
            segments=splits,
            tile=32,
            num_warps=1,
            num_stages=1,
            variant=variant,
        )
        check(case, cache, expected, variant)
        got = case["out"].cpu()
        cu = case["cu"].cpu().tolist()
        for i in (0, 2, 4):  # zero length / non-decode rows are not written
            assert torch.isnan(got[cu[i] : cu[i + 1]]).all()


def test_cuda_graph_replay_in_place():
    variant = 8
    rows = 8
    case, cache = build([4096] * rows, seed=11)  # fixes pool, table width, buffers
    pool = cache.shape[0]
    ws = k8.DecodeWorkspace("cuda:0")
    half = k8.half_base(cache)

    def launch():
        k8.kv_int8_decode(
            case["q"],
            cache,
            half,
            case["out"],
            case["cu"],
            case["lens"],
            case["bt"],
            SCALE,
            ws,
            segments=8,
            tile=32,
            num_warps=1,
            num_stages=1,
            variant=variant,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    g = torch.Generator().manual_seed(7)
    steps = (
        [1, 15, 16, 17, 300, 1025, 77, 4096],
        [4096, 1, 2049, 31, 32, 33, 0, 640],
        [0, 0, 5, 4000, 4001, 4095, 16, 1],
    )
    for step, lengths in enumerate(steps):
        # new KV values, block tables, lengths and queries, all in place
        k = (torch.randn(pool * 16, 2, 128, generator=g)).to(torch.bfloat16).cuda()
        v = (torch.randn(pool * 16, 2, 128, generator=g)).to(torch.bfloat16).cuda()
        k8.kv_int8_write(
            k, v, cache, torch.arange(pool * 16, device="cuda", dtype=torch.int64)
        )
        perm = torch.randperm(pool, generator=g)[: rows * 256].reshape(rows, 256)
        case["bt"][:, :256].copy_(perm.to(torch.int32))
        case["lens"].copy_(torch.tensor(lengths, dtype=torch.int32))
        case["q"].copy_(torch.randn(rows, 16, 128, generator=g).to(torch.bfloat16))
        case["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(case, cache, reference(case, cache), variant)
