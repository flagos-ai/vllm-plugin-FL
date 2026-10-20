# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical coverage for the EMPTY wheel's actual CUDA cache registrations."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA empty cache bridge test"
)


@pytest.fixture
def empty_cache_ops():
    from vllm_fl.nvidia_compat import has_native_vllm_kernels
    from vllm_fl.ops import nvidia_cache_ops

    if has_native_vllm_kernels():
        pytest.skip("Native wheel supplies its own cache operations")
    assert nvidia_cache_ops.register_nvidia_cache_ops()
    return torch.ops._C_cache_ops


def _mha_cache(layout, *, dtype):
    if layout == "LBHNC":
        cache = torch.zeros(3, 2, 16, 128, device="cuda", dtype=dtype)
    else:
        cache = torch.zeros(3, 16, 2, 128, device="cuda", dtype=dtype).transpose(1, 2)
    return cache.transpose(1, 2).split(64, dim=-1)


def _fp8_capability():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("The upstream Triton FP8 E4M3 writer requires SM89+")


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["LBHNC", "LBNHC"])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_e4m3"])
def test_registered_mha_cache_writes_packed_layouts(
    empty_cache_ops, layout, cache_dtype
):
    if cache_dtype != "auto":
        _fp8_capability()
    storage_dtype = torch.bfloat16 if cache_dtype == "auto" else torch.uint8
    keys, values = _mha_cache(layout, dtype=storage_dtype)
    # Sliced input rows exercise non-contiguous row strides, while preserving
    # contiguous head dimensions as required by the native CUDA contract.
    generator = torch.Generator(device="cuda").manual_seed(41)
    source = torch.randn(4, 2, 3, 64, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    key, value = source[:, 0, :2], source[:, 1, :2]
    slots = torch.tensor([0, 17, -1, 35], device="cuda", dtype=torch.int64)
    k_scale = torch.tensor(0.5, device="cuda")
    v_scale = torch.tensor(0.25, device="cuda")
    empty_cache_ops.reshape_and_cache_flash(
        key, value, keys, values, slots, cache_dtype, k_scale, v_scale
    )
    expected_k = torch.zeros(3, 16, 2, 64)
    expected_v = torch.zeros_like(expected_k)
    for token, slot in enumerate([0, 17, -1, 35]):
        if slot < 0:
            continue
        key_ref, value_ref = key[token].cpu().float(), value[token].cpu().float()
        if cache_dtype != "auto":
            key_ref = (key_ref / 0.5).to(torch.float8_e4m3fn).float()
            value_ref = (value_ref / 0.25).to(torch.float8_e4m3fn).float()
        expected_k[slot // 16, slot % 16] = key_ref
        expected_v[slot // 16, slot % 16] = value_ref
    if cache_dtype != "auto":
        keys, values = keys.view(torch.float8_e4m3fn), values.view(torch.float8_e4m3fn)
    torch.testing.assert_close(keys.cpu().float(), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(values.cpu().float(), expected_v, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("padded_entries", [False, True])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8_e4m3", "fp8_e5m2"])
def test_registered_mla_cache_preserves_strides_and_quantization(
    empty_cache_ops, padded_entries, cache_dtype
):
    if cache_dtype != "auto":
        _fp8_capability()
    storage_dtype = torch.bfloat16 if cache_dtype == "auto" else torch.uint8
    width = 32 if padded_entries else 24
    storage = torch.zeros(3, 16, width, device="cuda", dtype=storage_dtype)
    cache = storage[..., :24]
    fp8_dtype = torch.float8_e5m2 if cache_dtype == "fp8_e5m2" else torch.float8_e4m3fn
    # Exercise both typed-FP8 and raw-byte cache storage without copies.
    if cache_dtype != "auto" and padded_entries:
        cache = cache.view(fp8_dtype)
    source = torch.arange(4 * 32, device="cuda").reshape(4, 32).to(torch.bfloat16) / 32
    kv_c, k_pe = source[:, :16], source[:, 20:28]
    slots = torch.tensor([0, 17, -1, 35], device="cuda", dtype=torch.int64)
    scale = torch.tensor(0.5, device="cuda")
    empty_cache_ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, cache_dtype, scale)
    expected = torch.zeros(3, 16, 24)
    for token, slot in enumerate([0, 17, -1, 35]):
        if slot < 0:
            continue
        entry = torch.cat((kv_c[token], k_pe[token])).cpu().float()
        if cache_dtype != "auto":
            entry = (entry / 0.5).to(fp8_dtype).float()
        expected[slot // 16, slot % 16] = entry
    actual = cache if cache_dtype == "auto" else cache.view(fp8_dtype)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=0, atol=0)
    if padded_entries:
        assert torch.count_nonzero(storage[..., 24:]).item() == 0


@pytest.mark.gpu
def test_cache_bridge_runs_inside_cuda_graph(empty_cache_ops):
    keys, values = _mha_cache("LBHNC", dtype=torch.bfloat16)
    key = torch.ones(1, 2, 64, device="cuda", dtype=torch.bfloat16)
    value = key * 2
    mla_cache = torch.zeros(1, 16, 24, device="cuda", dtype=torch.bfloat16)
    kv_c = torch.ones(1, 16, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.ones(1, 8, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(1, device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda")

    def update():
        empty_cache_ops.reshape_and_cache_flash(
            key, value, keys, values, slots, "auto", scale, scale
        )
        empty_cache_ops.concat_and_cache_mla(
            kv_c, k_pe, mla_cache, slots, "auto", scale
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        update()  # compile kernels outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()
    key.fill_(3)
    kv_c.fill_(4)
    graph.replay()
    torch.testing.assert_close(keys[0, 0], key[0])
    torch.testing.assert_close(mla_cache[0, 0, :16], kv_c[0])


def test_cache_bridge_registration_is_idempotent(empty_cache_ops):
    from vllm_fl.ops import nvidia_cache_ops

    previous = nvidia_cache_ops._lib
    assert nvidia_cache_ops.register_nvidia_cache_ops()
    assert nvidia_cache_ops._lib is previous
    for name in ("reshape_and_cache_flash", "concat_and_cache_mla"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C_cache_ops::{name}", "CUDA"
        )


def test_cache_bridge_does_not_override_native_wheel(monkeypatch):
    from vllm_fl.ops import nvidia_cache_ops

    previous = nvidia_cache_ops._lib
    monkeypatch.setattr(nvidia_cache_ops, "has_native_vllm_kernels", lambda: True)
    assert not nvidia_cache_ops.register_nvidia_cache_ops()
    assert nvidia_cache_ops._lib is previous


@pytest.mark.gpu
def test_cache_bridge_rejects_unsupported_quantization(empty_cache_ops):
    keys, values = _mha_cache("LBHNC", dtype=torch.uint8)
    key = torch.zeros(1, 2, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(1, device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda")
    with pytest.raises(NotImplementedError, match="FP8 E4M3"):
        empty_cache_ops.reshape_and_cache_flash(
            key, key, keys, values, slots, "fp8_e5m2", scale, scale
        )
    with pytest.raises(NotImplementedError, match="fp8_ds_mla"):
        empty_cache_ops.concat_and_cache_mla(
            key.reshape(1, -1)[:, :16],
            key.reshape(1, -1)[:, :8],
            torch.zeros(1, 16, 24, device="cuda", dtype=torch.uint8),
            slots,
            "fp8_ds_mla",
            scale,
        )
