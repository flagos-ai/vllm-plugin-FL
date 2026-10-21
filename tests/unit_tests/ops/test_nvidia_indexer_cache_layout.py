# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise the upstream allocator's padded and interleaved indexer pages."""

import pytest
import torch

from tests.utils.device_utils import get_backend, requires_cuda_fp8

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA sparse indexer cache layouts"
)


@pytest.fixture
def gems():
    import flag_gems

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    _, blacklist = get_flag_gems_whitelist_blacklist()
    with flag_gems.use_gems(exclude=blacklist):
        yield


def _allocated_cache(kind, layers, compress_ratio=1):
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
    from vllm.v1.worker.utils import allocate_kv_cache

    cfg = VllmConfig(
        cache_config=CacheConfig(
            block_size=64, cache_dtype="fp8_ds_mla" if kind == "ds" else "bfloat16"
        )
    )
    cfg.cache_config.kv_cache_layout = "BLHNC"
    with set_current_vllm_config(cfg):
        caches = []
        for layer in range(layers):
            args = (132, torch.uint8, f"layout.{layer}", cfg.cache_config)
            cache = (
                DeepseekV4IndexerCache(*args, compress_ratio=compress_ratio)
                if kind == "ds"
                else DeepseekV32IndexerCache(*args)
            )
            caches.append(cache)
        spec = caches[0].get_kv_cache_spec(cfg)
        names = [cache.prefix for cache in caches]
        plan = get_kv_cache_config_from_groups(
            cfg, [KVCacheGroupSpec(names, spec)], 4 * layers * spec.page_size_bytes
        )
        allocated = allocate_kv_cache(
            plan,
            torch.device("cuda"),
            cfg.cache_config.get_resolved_kv_cache_layout(),
            [64],
        )
        cache = caches[-1]
        cache.bind_kv_cache(allocated[cache.prefix])
    view = cache.kv_cache
    raw = view.as_strided((view.untyped_storage().nbytes(),), (1,), storage_offset=0)
    raw.fill_(165)
    untouched = torch.ones(raw.numel(), dtype=torch.bool)
    page_bytes = view.shape[1] * 132
    for page in range(view.shape[0]):
        start = view.storage_offset() + page * view.stride(0)
        untouched[start : start + page_bytes] = False
    return view, raw, untouched


def _fp8(shape, seed):
    return (
        torch.randn(shape, generator=torch.Generator().manual_seed(seed))
        .clamp(-3, 3)
        .to(torch.float8_e4m3fn)
        .cuda()
    )


def _write_pages(cache, keys, scales):
    pages, block = cache.shape[:2]
    flat = cache.view(pages, -1)
    flat[:, : block * 128].copy_(keys.view(torch.uint8).reshape(pages, -1))
    flat[:, block * 128 :].view(torch.float32).copy_(scales.view(pages, block))


def _selected(keys, scales, table, lengths, block):
    selected_k, selected_s = [], []
    for row, length in enumerate(lengths.cpu().flatten().tolist()):
        ids = torch.cat(
            [
                torch.arange(int(p) * block, (int(p) + 1) * block)
                for p in table[row].cpu()
            ]
        )[:length]
        selected_k.append(keys.float().cpu()[ids])
        selected_s.append(scales.cpu()[ids])
    return selected_k, selected_s


@pytest.mark.gpu
@requires_cuda_fp8
@pytest.mark.parametrize(
    "kind,layers,ratio", [("ds", 1, 1), ("ds", 2, 2), ("hy", 2, 1)]
)
def test_upstream_allocated_pages_gather_decode_and_graph(gems, kind, layers, ratio):
    from vllm_fl.ops import nvidia_dsv41_indexer as ops
    from vllm_fl.ops.nvidia_dsv41_ops import register_nvidia_dsv41_ops

    assert register_nvidia_dsv41_ops()
    cache, raw, untouched = _allocated_cache(kind, layers, ratio)
    assert not cache.is_contiguous()
    if layers > 1:
        assert cache.storage_offset() > 0
    pages, block = cache.shape[:2]
    keys = _fp8((pages * block, 128), 71)
    scales = torch.linspace(0.01, 0.05, pages * block, device="cuda")
    _write_pages(cache, keys, scales)
    q = _fp8((2, 1, 32, 128), 72)
    weights = torch.linspace(-0.3, 0.7, 64, device="cuda").reshape(2, 32)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([[block + 7], [block + 3]], dtype=torch.int32, device="cuda")
    total = 2 * block + 10
    cumulative = torch.tensor([0, block + 7, total], dtype=torch.int32, device="cuda")
    dst_storage = torch.full((total, 136), 90, dtype=torch.uint8, device="cuda")
    dst = dst_storage[:, :128].view(torch.float8_e4m3fn)
    dst_scale = torch.empty((total, 4), dtype=torch.uint8, device="cuda")

    def run():
        torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
            cache, dst, dst_scale, table, cumulative
        )
        return ops.fp8_paged_mqa_logits(
            (q, None),
            cache.unsqueeze(2),
            weights,
            lengths,
            table,
            None,
            2 * block,
            clean_logits=True,
        )

    def check(actual):
        selected_k, selected_s = _selected(keys, scales, table, lengths, block)
        torch.testing.assert_close(
            dst.float().cpu(), torch.cat(selected_k), rtol=0, atol=0
        )
        torch.testing.assert_close(
            dst_scale.view(torch.float32).cpu().flatten(),
            torch.cat(selected_s),
            rtol=0,
            atol=0,
        )
        assert torch.all(dst_storage[:, 128:] == 90)
        for row, (k, scale) in enumerate(zip(selected_k, selected_s)):
            dot = torch.einsum("hd,nd->hn", q[row, 0].float().cpu(), k)
            expected = (
                torch.relu(dot * scale[None, :]) * weights[row].cpu()[:, None]
            ).sum(0)
            torch.testing.assert_close(
                actual[row, : len(k)].cpu(), expected, rtol=3e-4, atol=3e-4
            )
            assert torch.isneginf(actual[row, len(k) :]).all()
        assert torch.all(raw.cpu()[untouched] == 165)

    snapshot = raw.clone()
    check(run())
    torch.testing.assert_close(raw, snapshot, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    keys = _fp8(keys.shape, 73)
    scales.mul_(2)
    _write_pages(cache, keys, scales)
    table.copy_(torch.tensor([[1, 3], [0, 2]], dtype=torch.int32, device="cuda"))
    lengths.copy_(
        torch.tensor([[block - 3], [block + 13]], dtype=torch.int32, device="cuda")
    )
    cumulative[1] = block - 3
    q.copy_(_fp8(q.shape, 74))
    snapshot = raw.clone()
    graph.replay()
    check(actual)
    torch.testing.assert_close(raw, snapshot, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("bad_layout", ["inner_stride", "overlap", "unaligned"])
def test_indexer_rejects_invalid_page_storage(bad_layout):
    from vllm_fl.ops.nvidia_dsv41_indexer import _validate_fp8_indexer_cache

    raw = torch.empty(40000, dtype=torch.uint8, device="cuda")
    stride = {
        "inner_stride": (17000, 264, 2),
        "overlap": (4, 132, 1),
        "unaligned": (8449, 132, 1),
    }[bad_layout]
    cache = raw.as_strided((2, 64, 132), stride)
    with pytest.raises(ValueError, match="page strides"):
        _validate_fp8_indexer_cache(cache)
