"""Focused geometry and metadata checks for padded INT8 KV pages."""

import pytest
import torch

from vllm_fl.attention import int8_sparse, kv_int8


def _cache(pages, page_bytes, *, device="cpu", offset_pages=0):
    raw = torch.empty(
        ((pages + offset_pages) * page_bytes,), dtype=torch.int8, device=device
    )
    cache = torch.as_strided(
        raw,
        (pages, 2, kv_int8.PAGE_TOKENS, kv_int8.NUM_KV_HEADS, kv_int8.HEAD_DIM),
        (page_bytes, kv_int8.DATA_BYTES // 2, 256, 128, 1),
        storage_offset=offset_pages * page_bytes,
    )
    return raw, cache


def test_supported_page_strides_and_sparse_metadata_offsets():
    _, ordinary = _cache(2, kv_int8.PAGE_BYTES, offset_pages=1)
    assert kv_int8.is_int8_layout(ordinary)
    assert kv_int8.page_stride_bytes(ordinary) == kv_int8.PAGE_BYTES
    assert kv_int8.page_stride_bytes(ordinary[:, 0]) == kv_int8.PAGE_BYTES
    assert tuple(kv_int8.scale_view(ordinary).shape) == (2, 2, 2, 16)
    with pytest.raises(ValueError, match="9344-byte"):
        int8_sparse.metadata_views(ordinary)

    _, padded = _cache(3, kv_int8.SPARSE_PAGE_BYTES, offset_pages=1)
    padded.zero_()
    assert kv_int8.is_int8_layout(padded)
    assert kv_int8.page_stride_bytes(padded) == kv_int8.SPARSE_PAGE_BYTES
    assert kv_int8.page_stride_bytes(padded[:, 0]) == kv_int8.SPARSE_PAGE_BYTES
    scales = kv_int8.scale_view(padded)
    kmin, kmax = int8_sparse.metadata_views(padded)
    assert tuple(scales.shape) == (3, 2, 2, 16)
    assert tuple(kmin.shape) == tuple(kmax.shape) == (3, 2, 128)
    assert kmin.stride() == kmax.stride() == (kv_int8.SPARSE_PAGE_BYTES // 2, 128, 1)
    assert kmin.data_ptr() == padded.data_ptr() + kv_int8.PAGE_BYTES
    assert (
        kmax.data_ptr()
        == padded.data_ptr() + kv_int8.PAGE_BYTES + kv_int8.SPARSE_METADATA_BYTES // 2
    )

    kmin.fill_(17)
    kmax.fill_(-19)
    assert torch.all(kmin == 17) and torch.all(kmax == -19)
    assert torch.all(padded[:, :, :, :, :] == 0)


def _dequantized_k_reference(cache):
    key = cache[:, 0].cpu().float()
    scales = kv_int8.scale_view(cache)[:, 0].cpu().float().permute(0, 2, 1)[..., None]
    return key * scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("page_bytes", [kv_int8.PAGE_BYTES, kv_int8.SPARSE_PAGE_BYTES])
def test_write_decode_dequant_and_sparse_metadata_graph_replay(page_bytes):
    device = torch.device("cuda")
    pages = 3
    _, cache = _cache(pages, page_bytes, device=device, offset_pages=1)
    cache.zero_()
    if page_bytes == kv_int8.SPARSE_PAGE_BYTES:
        kmin, kmax = int8_sparse.metadata_views(cache)
        kmin.fill_(17)
        kmax.fill_(-19)

    generator = torch.Generator(device=device).manual_seed(1211)
    key = torch.randn(
        (pages * 16, 2, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        (pages * 16, 2, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    slots = torch.arange(pages * 16, device=device, dtype=torch.int64)
    kv_int8.kv_int8_write(key, value, cache, slots)
    torch.cuda.synchronize()
    if page_bytes == kv_int8.SPARSE_PAGE_BYTES:
        assert torch.all(kmin == 17) and torch.all(kmax == -19)

    # Dequantization exercises the page-stride argument and the fixed scale base.
    work = torch.tensor([[0, 0]], device=device, dtype=torch.int32)
    scratch = torch.empty((1, 2, 16, 2, 128), device=device, dtype=torch.bfloat16)
    kv_int8.kv_int8_dequant_pages(
        cache[:, 0],
        kv_int8.half_base(cache),
        torch.zeros((1, 1), device=device, dtype=torch.int32),
        work,
        scratch,
    )
    torch.cuda.synchronize()
    expected_k = _dequantized_k_reference(cache)[0].to(torch.bfloat16).float()
    torch.testing.assert_close(
        scratch[0, 0].cpu().float(), expected_k, atol=1e-3, rtol=1e-3
    )

    # Decode reads one selected physical page using the existing INT8 kernel.
    q = torch.randn(
        (1, 16, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    out = torch.empty_like(q)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    lengths = torch.tensor([16], device=device, dtype=torch.int32)
    table = torch.tensor([[1]], device=device, dtype=torch.int32)
    workspace = kv_int8.DecodeWorkspace(device)
    kv_int8.kv_int8_decode(
        q,
        cache,
        kv_int8.half_base(cache),
        out,
        cu,
        lengths,
        table,
        128**-0.5,
        workspace,
        segments=1,
        tile=32,
        num_warps=1,
        num_stages=1,
        variant=8,
    )
    torch.cuda.synchronize()
    q_cpu = q[0].cpu().float().reshape(2, 8, 128)
    scales_cpu = kv_int8.scale_view(cache).cpu().float()
    k_cpu = (
        cache[1, 0].cpu().float() * scales_cpu[1, 0].permute(1, 0)[..., None]
    ).permute(1, 0, 2)
    v_cpu = (
        cache[1, 1].cpu().float() * scales_cpu[1, 1].permute(1, 0)[..., None]
    ).permute(1, 0, 2)
    ref = torch.empty((2, 8, 128), dtype=torch.float32)
    for head in range(2):
        probabilities = torch.softmax(q_cpu[head] @ k_cpu[head].T * (128**-0.5), dim=-1)
        ref[head] = probabilities @ v_cpu[head]
    torch.testing.assert_close(
        out[0].cpu().float().reshape(2, 8, 128), ref, atol=3e-2, rtol=3e-2
    )

    if page_bytes != kv_int8.SPARSE_PAGE_BYTES:
        return

    row = torch.tensor([1, 2], device=device, dtype=torch.int32)
    int8_sparse.refresh_full_pages(cache, row, 1)
    torch.cuda.synchronize()
    min_view, max_view = int8_sparse.metadata_views(cache)
    stored_k = _dequantized_k_reference(cache)[1].to(torch.bfloat16).float()
    expected = stored_k.amin(dim=0)
    expected_max = stored_k.amax(dim=0)
    torch.testing.assert_close(min_view[1].cpu().float(), expected)
    torch.testing.assert_close(max_view[1].cpu().float(), expected_max)

    # The cross-layer updater deduplicates aliases, keeps its inputs alive, and
    # replays against modified stored values without rebuilding its pointer table.
    _, second = _cache(pages, page_bytes, device=device, offset_pages=1)
    second.zero_()
    kv_int8.kv_int8_write(key, value, second, slots)
    refresher = int8_sparse.MultiLayerMetadataRefresher([cache, cache, second])
    assert refresher.num_blocks == pages
    assert refresher.num_layers == 2
    jobs = torch.tensor([2], device=device, dtype=torch.int32)
    refresher.refresh(jobs, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refresher.refresh(jobs, 1)

    cache[2, 0].fill_(31)
    kv_int8.scale_view(cache)[2, 0].fill_(0.5)
    second[2, 0].fill_(-27)
    kv_int8.scale_view(second)[2, 0].fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    cache_min, cache_max = int8_sparse.metadata_views(cache)
    second_min, second_max = int8_sparse.metadata_views(second)
    torch.testing.assert_close(cache_min[2].cpu().float(), torch.full((2, 128), 15.5))
    torch.testing.assert_close(cache_max[2].cpu().float(), torch.full((2, 128), 15.5))
    torch.testing.assert_close(second_min[2].cpu().float(), torch.full((2, 128), -6.75))
    torch.testing.assert_close(second_max[2].cpu().float(), torch.full((2, 128), -6.75))
