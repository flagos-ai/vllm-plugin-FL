# Copyright (c) 2026 BAAI. All rights reserved.

"""Device regressions for THead's graph-safe cache scatter.

Run on a CUDA-compatible accelerator, including the THead PPU. These tests load
the cache module independently of the PPU FlashAttention wheel so the cache
contract can also be exercised by standard CUDA CI.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def cache_writer():
    pytest.importorskip("vllm.triton_utils")
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/dispatch/backends/vendor/thead/impl/cache.py"
    )
    spec = importlib.util.spec_from_file_location("_thead_cache_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.reshape_and_cache_flash_thead


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/PPU required")
_SHAPE = (3, 4, 3, 33)  # Non-power-of-two head size exercises element masking.


def _cache(layout, dtype, fill):
    blocks, block_size, heads, dim = _SHAPE
    if layout == "HND":
        return torch.full(
            (blocks, heads, block_size, dim), fill, device="cuda", dtype=dtype
        ).permute(0, 2, 1, 3)
    return torch.full(_SHAPE, fill, device="cuda", dtype=dtype)


def _packed_cache(layout, dtype):
    blocks, block_size, heads, dim = _SHAPE
    if layout == "NHD":
        # Physical NHD storage, exposed in vLLM's logical [B, H, N, 2D] shape.
        packed = torch.full(
            (blocks, block_size, heads, 2 * dim), -31, device="cuda", dtype=dtype
        ).transpose(1, 2)
    else:
        packed = torch.full(
            (blocks, heads, block_size, 2 * dim), -31, device="cuda", dtype=dtype
        )
    # This is the target vLLM 0.28 do_kv_cache_update split, including V's offset.
    key_cache, value_cache = packed.transpose(1, 2).split(dim, dim=-1)
    key_cache.fill_(-13)
    value_cache.fill_(-17)
    return packed, key_cache, value_cache


def _inputs(dtype, tokens=6):
    shape = (tokens, _SHAPE[2], _SHAPE[3])
    key = (
        (torch.arange(tokens * shape[1] * shape[2], device="cuda") % 29)
        .reshape(shape)
        .to(dtype)
    )
    return key, key + 41


def _write(cache_writer, key, value, key_cache, value_cache, slots, cache_dtype="auto"):
    scale = torch.ones((), device=key.device)
    cache_writer(key, value, key_cache, value_cache, slots, cache_dtype, scale, scale)


def _expected(cache, source, slots):
    result = cache.detach().cpu().clone()
    source = source.detach().cpu()
    block_size = cache.shape[1]
    for token, slot in enumerate(slots.detach().cpu().tolist()):
        if 0 <= slot < cache.shape[0] * block_size:
            result[slot // block_size, slot % block_size] = source[token]
    return result


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,cache_dtype", [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("slots_list", [[0, -1, 5, -7, 11], [-1, 5, -1]])
def test_padding_preserves_slot_zero_and_unwritten_cache(
    cache_writer, dtype, cache_dtype, layout, slots_list
):
    key, value = _inputs(dtype)
    key_cache = _cache(layout, dtype, -13)
    # Deliberately use an independent value cache stride order.
    value_cache = _cache("HND" if layout == "NHD" else "NHD", dtype, -17)
    slots = torch.tensor(slots_list, device="cuda", dtype=torch.int64)
    expected_key = _expected(key_cache, key, slots)
    expected_value = _expected(value_cache, value, slots)

    _write(cache_writer, key, value, key_cache, value_cache, slots, cache_dtype)

    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("slots_list", [[0, -1, 5, -7, 11], [-1, 5, -1], [-1, -2, -7]])
def test_packed_kv_cache_split_views_preserve_both_halves(
    cache_writer, dtype, layout, slots_list
):
    key, value = _inputs(dtype)
    packed, key_cache, value_cache = _packed_cache(layout, dtype)
    assert (
        key_cache.untyped_storage().data_ptr()
        == value_cache.untyped_storage().data_ptr()
    )
    assert value_cache.storage_offset() - key_cache.storage_offset() == _SHAPE[-1]
    assert key_cache.stride() == value_cache.stride()
    if all(slot < 0 for slot in slots_list):
        key.fill_(float("nan"))
        value.fill_(float("nan"))
    slots = torch.tensor(slots_list, device="cuda", dtype=torch.int64)
    expected_packed = packed.cpu().clone()
    expected_key, expected_value = expected_packed.transpose(1, 2).split(
        _SHAPE[-1], dim=-1
    )
    expected_key.copy_(_expected(key_cache, key, slots))
    expected_value.copy_(_expected(value_cache, value, slots))

    _write(cache_writer, key, value, key_cache, value_cache, slots)

    # The complete shared allocation checks K/V isolation and every unwritten slot.
    torch.testing.assert_close(packed.cpu(), expected_packed, rtol=0, atol=0)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_all_padding_and_empty_mapping_leave_cache_untouched(
    cache_writer, dtype, layout
):
    key, value = _inputs(dtype)
    key.fill_(float("nan"))
    value.fill_(float("nan"))
    key_cache = _cache(layout, dtype, 7)
    value_cache = _cache(layout, dtype, 9)
    initial_key = key_cache.cpu().clone()
    initial_value = value_cache.cpu().clone()
    for slots_list in [[-1, -2, -1, -9], []]:
        slots = torch.tensor(slots_list, device="cuda", dtype=torch.int32)
        _write(cache_writer, key, value, key_cache, value_cache, slots)
        torch.testing.assert_close(key_cache.cpu(), initial_key, rtol=0, atol=0)
        torch.testing.assert_close(value_cache.cpu(), initial_value, rtol=0, atol=0)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_noncontiguous_sources_slots_and_cache_storage(cache_writer, dtype):
    key, value = _inputs(dtype)
    key_storage = torch.full((12, 6, 66), -3, device="cuda", dtype=dtype)
    value_storage = torch.full((3, 12, 66), -5, device="cuda", dtype=dtype)
    key_view = key_storage[::2, ::2, ::2]
    value_view = value_storage.permute(1, 0, 2)[::2, :, ::2]
    key_view.copy_(key)
    value_view.copy_(value)
    key_storage_cache = torch.full((6, 8, 6, 66), -13, device="cuda", dtype=dtype)
    value_storage_cache = torch.full((3, 3, 4, 66), -17, device="cuda", dtype=dtype)
    key_cache = key_storage_cache[::2, ::2, ::2, ::2]
    value_cache = value_storage_cache.permute(0, 2, 1, 3)[..., ::2]
    slot_storage = torch.tensor(
        [0, 99, -1, 99, 5, 99, 11, 99], device="cuda", dtype=torch.int32
    )
    slots = slot_storage[::2]
    expected_key_storage = key_storage_cache.cpu().clone()
    expected_value_storage = value_storage_cache.cpu().clone()
    expected_key_view = expected_key_storage[::2, ::2, ::2, ::2]
    expected_value_view = expected_value_storage.permute(0, 2, 1, 3)[..., ::2]
    expected_key_view.copy_(_expected(key_cache, key_view, slots))
    expected_value_view.copy_(_expected(value_cache, value_view, slots))

    _write(cache_writer, key_view, value_view, key_cache, value_cache, slots)

    # Compare backing storage too: intervening elements must remain untouched.
    torch.testing.assert_close(
        key_storage_cache.cpu(), expected_key_storage, rtol=0, atol=0
    )
    torch.testing.assert_close(
        value_storage_cache.cpu(), expected_value_storage, rtol=0, atol=0
    )


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multiple_element_tiles_and_out_of_range_slots(cache_writer, dtype):
    # 645 elements per token require three independent 256-element programs.
    key = torch.arange(4 * 5 * 129, device="cuda").reshape(4, 5, 129).to(dtype)
    value = key + 4
    key_cache = torch.full((2, 4, 5, 129), -13, device="cuda", dtype=dtype)
    value_cache = torch.full((2, 5, 4, 129), -17, device="cuda", dtype=dtype).permute(
        0, 2, 1, 3
    )
    slots = torch.tensor([0, -1, 7, 8], device="cuda", dtype=torch.int64)
    expected_key = _expected(key_cache, key, slots)
    expected_value = _expected(value_cache, value, slots)

    _write(cache_writer, key, value, key_cache, value_cache, slots)

    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("cache_storage", ["separate", "packed"])
def test_cuda_graph_replay_uses_new_slots_and_preserves_unwritten_cache(
    cache_writer, dtype, layout, cache_storage
):
    key, value = _inputs(dtype)
    if cache_storage == "packed":
        _, key_cache, value_cache = _packed_cache(layout, dtype)
    else:
        key_cache = _cache(layout, dtype, -13)
        value_cache = _cache(layout, dtype, -17)
    initial_key = key_cache.clone()
    initial_value = value_cache.clone()
    slots = torch.tensor([-1, 5, -1], device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda")
    current_stream = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream):
        cache_writer(key, value, key_cache, value_cache, slots, "auto", scale, scale)
    current_stream.wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache_writer(key, value, key_cache, value_cache, slots, "auto", scale, scale)

    key_cache.copy_(initial_key)
    value_cache.copy_(initial_value)
    slots.copy_(torch.tensor([0, -1, 11], device="cuda"))
    key.add_(20)
    value.add_(30)
    expected_key = _expected(key_cache, key, slots)
    expected_value = _expected(value_cache, value, slots)
    graph.replay()
    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)

    slots.fill_(-1)
    key.fill_(float("nan"))
    value.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)

    slots.copy_(torch.tensor([5, -1, -2], device="cuda"))
    key.fill_(42)
    value.fill_(53)
    expected_key = _expected(key_cache, key, slots)
    expected_value = _expected(value_cache, value, slots)
    graph.replay()
    torch.testing.assert_close(key_cache.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache.cpu(), expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2", "int8"])
def test_quantized_cache_formats_are_rejected(cache_writer, cache_dtype):
    key = torch.zeros((1, 1, 4), dtype=torch.float16)
    cache = torch.zeros((1, 4, 1, 4), dtype=torch.float16)
    slots = torch.zeros(1, dtype=torch.int64)
    scale = torch.ones(())
    with pytest.raises(NotImplementedError, match="do not support cache dtype"):
        cache_writer(key, key, cache, cache, slots, cache_dtype, scale, scale)


def test_quantized_tensor_storage_is_rejected_even_with_auto(cache_writer):
    key = torch.zeros((1, 1, 4), dtype=torch.float16)
    cache = torch.zeros((1, 4, 1, 4), dtype=torch.uint8)
    slots = torch.zeros(1, dtype=torch.int64)
    scale = torch.ones(())
    with pytest.raises(NotImplementedError, match="require FP16 or BF16"):
        cache_writer(key, key, cache, cache, slots, "auto", scale, scale)
