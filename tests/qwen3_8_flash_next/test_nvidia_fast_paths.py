"""Capability and correctness gates for optional NVIDIA QSA fast paths."""

from __future__ import annotations

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.gpu import nvidia_fast_paths as fast


class _NonNvidiaPlatform:
    @staticmethod
    def is_cuda() -> bool:
        return False

    @staticmethod
    def is_rocm() -> bool:
        return False


def test_non_nvidia_platform_never_probes_private_cuda_ops(monkeypatch):
    monkeypatch.setattr(fast, "current_platform", _NonNvidiaPlatform())

    def fail(*_args, **_kwargs):
        raise AssertionError("private CUDA dispatch must not be probed")

    monkeypatch.setattr(
        torch._C, "_dispatch_has_kernel_for_dispatch_key", fail
    )
    assert fast.is_nvidia_platform() is False
    assert fast.has_native_topk() is False
    assert fast.has_native_cache_update() is False


@pytest.mark.gpu
def test_nvidia_fused_cache_update_mutates_backing_and_graph_replays():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not fast.is_nvidia_platform():
        pytest.skip("NVIDIA-only capability test")
    assert fast.has_native_cache_update(), (
        "NVIDIA image exposes the QSA fast path but has no CUDA cache-update kernel"
    )

    torch.manual_seed(20260819)
    rows, blocks, page, heads, dim = 64, 32, 16, 1, 256
    backing = torch.zeros(
        2, blocks, page, heads, dim, dtype=torch.bfloat16, device="cuda"
    )
    key_cache, value_cache = backing.unbind(0)
    key = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slots = torch.randperm(blocks * page, device="cuda")[:rows].to(torch.int64)
    scale = torch.ones((), dtype=torch.float32, device="cuda")

    def update() -> None:
        fast.native_cache_update(
            key,
            value,
            key_cache,
            value_cache,
            slots,
            "auto",
            scale,
            scale,
        )

    update()
    flat_key = key_cache.view(-1, heads, dim)
    flat_value = value_cache.view(-1, heads, dim)
    torch.testing.assert_close(flat_key[slots], key, rtol=0, atol=0)
    torch.testing.assert_close(flat_value[slots], value, rtol=0, atol=0)

    for _ in range(5):
        update()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        update()
    key.add_(1)
    value.sub_(1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(flat_key[slots], key, rtol=0, atol=0)
    torch.testing.assert_close(flat_value[slots], value, rtol=0, atol=0)
