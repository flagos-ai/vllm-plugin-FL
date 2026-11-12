# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU provider/ABI contracts for MoE alignment, not hardware kernel numerics."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_SOURCE = (
    Path(__file__).parents[3] / "vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py"
)


@pytest.fixture
def moe_adapter(monkeypatch):

    # Load only the adapter. All dependency stubs are restored after each test.
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    utils = ModuleType("vllm.utils")
    utils.__path__ = []
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton = SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b)
    math_utils = ModuleType("vllm.utils.math_utils")
    math_utils.round_up = lambda n, b: ((n + b - 1) // b) * b
    for name, module in (
        ("vllm", vllm),
        ("vllm.utils", utils),
        ("vllm.triton_utils", triton_utils),
        ("vllm.utils.math_utils", math_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_gcu_moe_align_contract", _SOURCE)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


class _TopkIds:
    def __init__(self, device, count=12):
        self.device = SimpleNamespace(type=device)
        self.values = torch.arange(count, device="cpu", dtype=torch.int32)

    def numel(self):
        return self.values.numel()


def _install_providers(monkeypatch, official, fused):
    for name, provider in (("flag_gems", official), ("flaggems_vllm", fused)):
        module = ModuleType(name)
        # Selection must follow input device metadata, not a vendor string.
        module.vendor_name = "enflame"
        if provider is not None:
            module.moe_align_block_size_triton = provider
        monkeypatch.setitem(sys.modules, name, module)


def _cpu_allocations(adapter, topk_ids, cuda_alias_available=False):
    buffers = []

    def empty(shape, *, dtype, device):
        assert device is topk_ids.device
        assert dtype == torch.int32
        value = torch.empty(shape, dtype=dtype, device="cpu")
        buffers.append(value)
        return value

    # Keep CUDA alias metadata available to catch accidental CUDA-based routing.
    adapter.torch = SimpleNamespace(
        empty=empty,
        int32=torch.int32,
        cuda=SimpleNamespace(is_available=lambda: cuda_alias_available),
    )
    return buffers


def _fill_outputs(*args):
    assert len(args) == 6
    topk_ids, num_experts, block_size, sorted_ids, expert_ids, post_pad = args
    assert block_size > 0 and topk_ids.numel() >= 0
    sorted_ids.copy_(torch.arange(sorted_ids.numel(), dtype=torch.int32, device="cpu"))
    expert_ids.copy_(
        torch.arange(expert_ids.numel(), dtype=torch.int32, device="cpu") % num_experts
    )
    post_pad.fill_(sorted_ids.numel())


@pytest.mark.parametrize("device", ["gcu", "cuda", "npu", "xpu", "cpu"])
@pytest.mark.parametrize("cuda_alias_available", [False, True])
def test_provider_follows_actual_device_and_preserves_abi(
    monkeypatch, moe_adapter, device, cuda_alias_available
):
    topk_ids = _TopkIds(device)
    buffers = _cpu_allocations(moe_adapter, topk_ids, cuda_alias_available)
    calls = []

    def official(*args):
        calls.append(("flag_gems", args))
        _fill_outputs(*args)

    def fused(*args):
        calls.append(("flaggems_vllm", args))
        _fill_outputs(*args)

    _install_providers(monkeypatch, official, fused)
    actual = moe_adapter.moe_align_block_size_flaggems(topk_ids, 4, 4)
    assert len(calls) == 1
    provider, args = calls[0]
    assert provider == ("flag_gems" if device == "gcu" else "flaggems_vllm")
    assert len(args) == 6
    assert args[0] is topk_ids and args[1:3] == (4, 4)
    assert [tuple(value.shape) for value in buffers] == [(24,), (6,), (1,)]
    for returned, allocated, passed in zip(actual, buffers, args[3:]):
        assert returned is allocated is passed
        assert returned.dtype == torch.int32
    torch.testing.assert_close(
        actual[0], torch.arange(24, dtype=torch.int32, device="cpu")
    )
    assert actual[2].item() == 24


@pytest.mark.parametrize(
    "count,num_experts,block_size,pad_sorted_ids,capacity,blocks",
    [
        (12, 4, 5, False, 28, 6),
        (12, 4, 5, True, 30, 6),
        (2, 8, 4, False, 8, 2),
        (2, 8, 4, True, 8, 2),
        (0, 8, 4, True, 0, 0),
    ],
)
@pytest.mark.parametrize("ignore_invalid_experts", [False, True])
def test_padding_and_expert_mapping_contract_is_preserved(
    monkeypatch,
    moe_adapter,
    count,
    num_experts,
    block_size,
    pad_sorted_ids,
    capacity,
    blocks,
    ignore_invalid_experts,
):
    topk_ids = _TopkIds("gcu", count)
    buffers = _cpu_allocations(moe_adapter, topk_ids)
    expert_map = torch.arange(num_experts, dtype=torch.int32, device="cpu") * 10 - 1
    calls = []

    def official(*args):
        calls.append(args)
        _fill_outputs(*args)

    def reject_fused(*args):
        raise AssertionError("GCU must not switch to the fused provider")

    _install_providers(monkeypatch, official, reject_fused)
    actual = moe_adapter.moe_align_block_size_flaggems(
        topk_ids,
        block_size,
        num_experts,
        expert_map=expert_map,
        pad_sorted_ids=pad_sorted_ids,
        ignore_invalid_experts=ignore_invalid_experts,
    )
    assert len(calls) == 1 and len(calls[0]) == 6
    assert calls[0][0] is topk_ids
    assert calls[0][1:3] == (num_experts, block_size)
    assert [tuple(value.shape) for value in buffers] == [(capacity,), (blocks,), (1,)]
    assert actual[0] is buffers[0] and actual[2] is buffers[2]
    assert actual[1] is not buffers[1]
    expected_experts = torch.tensor(
        [(index % num_experts) * 10 - 1 for index in range(blocks)], dtype=torch.int32
    )
    torch.testing.assert_close(actual[1], expected_experts, rtol=0, atol=0)
    torch.testing.assert_close(
        buffers[1],
        torch.arange(blocks, dtype=torch.int32, device="cpu") % num_experts,
        rtol=0,
        atol=0,
    )
    assert actual[2].item() == capacity


@pytest.mark.parametrize("device", ["gcu", "cuda"])
@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
def test_provider_errors_propagate_without_retry_or_fallback(
    monkeypatch, moe_adapter, device, error_type
):
    topk_ids = _TopkIds(device)
    _cpu_allocations(moe_adapter, topk_ids)
    error = error_type("alignment provider failed")
    calls = []

    def failing(*args):
        calls.append(args)
        raise error

    def reject_fallback(*args):
        raise AssertionError("must not try another provider")

    official, fused = (
        (failing, reject_fallback) if device == "gcu" else (reject_fallback, failing)
    )
    _install_providers(monkeypatch, official, fused)
    with pytest.raises(error_type) as caught:
        moe_adapter.moe_align_block_size_flaggems(topk_ids, 4, 4)
    assert caught.value is error
    assert len(calls) == 1 and len(calls[0]) == 6


def test_missing_official_export_does_not_fall_back(monkeypatch, moe_adapter):
    topk_ids = _TopkIds("gcu")
    buffers = _cpu_allocations(moe_adapter, topk_ids)

    def reject_fused(*args):
        raise AssertionError("missing official export must fail")

    _install_providers(monkeypatch, None, reject_fused)
    with pytest.raises(ImportError, match="moe_align_block_size_triton"):
        moe_adapter.moe_align_block_size_flaggems(topk_ids, 4, 4)
    assert buffers == []
