# Copyright (c) 2026 BAAI. All rights reserved.
"""Check physical KV bytes independently of the segment construction."""

import gc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/txda_kv_zeroer.py"


@pytest.fixture
def adapter(monkeypatch):
    interface = ModuleType("vllm.v1.kv_cache_interface")

    class AttentionSpec:
        def __init__(self, block_size=2):
            self.block_size = block_size

    interface.AttentionSpec = AttentionSpec
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    logging = ModuleType("vllm.logger")
    logger = Mock()
    logging.init_logger = Mock(return_value=logger)
    monkeypatch.setitem(sys.modules, logging.__name__, logging)
    spec = importlib.util.spec_from_file_location("txda_zeroer_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module, spec=AttentionSpec, logger=logger, logging=logging
    )


def group(adapter, names, *, block_size=2, group_id=0, **kwargs):
    return SimpleNamespace(
        kv_cache_spec=adapter.spec(block_size),
        kv_cache_group_id=group_id,
        layer_names=names,
        kernel_block_stride=kwargs.pop("kernel_block_stride", 0),
        **kwargs,
    )


def make_zeroer(
    adapter, tensors, *, groups=None, blocks=4, kernel_sizes=None, **kwargs
):
    names = [str(i) for i in range(len(tensors))]
    return adapter.module.TxdaKVBlockZeroer(
        torch.device("cpu"),
        groups if groups is not None else [group(adapter, names)],
        [2] if kernel_sizes is None else kernel_sizes,
        {
            name: SimpleNamespace(kv_cache=tensor)
            for name, tensor in zip(names, tensors)
        },
        blocks,
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["legacy", "LBHNC", "LBNHC", "LHBNC", "BHLNC", "padding", "virtual"]
)
def test_selected_physical_blocks_zero_and_other_bytes_survive(adapter, dtype, layout):
    # A nonzero storage offset and suffix surround every layout. Expectations
    # address the original physical allocation, not the zeroer's segments.
    payload = {"legacy": 128, "padding": 144, "virtual": 160}.get(layout, 128)
    raw = (torch.arange(payload + 16, dtype=torch.float32) + 1).to(dtype)
    expected = raw.clone()
    body, oracle = raw[8:-8], expected[8:-8]
    if layout == "legacy":
        physical = body.view(2, 4, 2, 2, 4)
        tensors = [physical.permute(1, 0, 3, 2, 4)]
        target = oracle.view(2, 4, 2, 2, 4)
        for block in (1, 3):
            target[:, block].zero_()
    elif layout in ("LBHNC", "LBNHC"):
        physical = body.view(2, 4, 2, 2, 4)
        tensors = [physical[layer] for layer in range(2)]
        if layout == "LBNHC":
            tensors = [tensor.permute(0, 2, 1, 3) for tensor in tensors]
        target = oracle.view(2, 4, 2, 2, 4)
        for block in (1, 3):
            target[:, block].zero_()
    elif layout == "LHBNC":
        physical = body.view(2, 2, 4, 2, 4)
        tensors = [physical[layer].permute(1, 0, 2, 3) for layer in range(2)]
        target = oracle.view(2, 2, 4, 2, 4)
        for block in (1, 3):
            target[:, :, block].zero_()
    elif layout in ("BHLNC", "padding"):
        block_stride = 32 if layout == "BHLNC" else 36
        physical = body.as_strided((4, 2, 2, 2, 4), (block_stride, 16, 8, 4, 1))
        tensors = [physical[:, :, layer] for layer in range(2)]
        target = oracle.view(4, block_stride)
        for block in (1, 3):
            target[block, :32].zero_()
    else:
        tensors = [body.as_strided((8, 2, 2, 4), (20, 8, 4, 1))]
        target = oracle.view(8, 20)
        for kernel_block in (2, 3, 6, 7):
            target[kernel_block, :16].zero_()
    zeroer = make_zeroer(
        adapter,
        tensors,
        groups=[group(adapter, [str(i) for i in range(len(tensors))], block_size=4)]
        if layout == "virtual"
        else None,
    )
    zeroer.zero_block_ids([3, 1, 3])
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_overlaid_address_keeps_widest_span_in_bytes(adapter, reverse, mixed_dtype):
    raw = torch.arange(80, dtype=torch.float32) + 1
    expected = raw.clone()
    body = raw[8:-8]
    narrow = body.as_strided((4, 1, 1, 4), (16, 4, 4, 1))
    if mixed_dtype:
        wide = body.view(torch.int16).as_strided((4, 1, 4, 4), (32, 16, 4, 1))
    else:
        wide = body.as_strided((4, 1, 2, 4), (16, 8, 4, 1))
    tensors = [wide, narrow] if reverse else [narrow, wide]
    zeroer = make_zeroer(adapter, tensors)
    assert len(zeroer._segments) == 1
    zeroer.zero_block_ids([2])
    expected[8:-8].view(4, 16)[2, :8].zero_()
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


def test_bound_kernel_view_is_mapped_back_to_manager_blocks(adapter):
    raw = torch.arange(144, dtype=torch.float32) + 1
    expected = raw.clone()
    body = raw[8:-8]
    bound = body.as_strided((14, 1, 2, 4), (8, 8, 4, 1))
    manager = body.as_strided((4, 1, 4, 4), (32, 16, 4, 1))
    mapping = Mock(return_value=manager)
    groups = [
        group(
            adapter,
            ["0"],
            block_size=4,
            kernel_block_stride=4,
            map_kv_cache_to_manager_blocks=mapping,
        )
    ]
    zeroer = make_zeroer(adapter, [bound], groups=groups)
    mapping.assert_called_once_with(bound, 4)
    zeroer.zero_block_ids([1])
    expected[8:-8].view(4, 32)[1, :16].zero_()
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


def test_non_attention_runner_only_wrong_device_and_missing_groups_are_skipped(adapter):
    raw = torch.ones(4, 1, 2, 4)
    missing = group(adapter, ["missing"])
    missing.kv_cache_spec = object()
    groups = [
        missing,
        group(adapter, ["missing"], group_id=1),
        group(adapter, ["0", "1", "2"]),
    ]
    zeroer = make_zeroer(
        adapter,
        [raw, [raw], torch.empty(4, 1, 2, 4, device="meta")],
        groups=groups,
        runner_only_attn_layers={"0"},
    )
    assert not zeroer._segments
    zeroer.zero_block_ids([0])
    assert bool(torch.all(raw == 1))


@pytest.mark.parametrize("bad_id", [-1, 4, 1.5, True, torch.tensor(1)])
def test_all_ids_are_checked_before_any_block_is_modified(adapter, bad_id):
    raw = torch.ones(4, 1, 2, 4)
    zeroer = make_zeroer(adapter, [raw])
    with pytest.raises((IndexError, TypeError)):
        zeroer.zero_block_ids([1, bad_id])
    assert bool(torch.all(raw == 1))


def test_warmup_clears_only_zero_and_empty_ids_do_nothing(adapter):
    raw = torch.ones(4, 1, 2, 4)
    zeroer = make_zeroer(adapter, [raw])
    zeroer.zero_block_ids([])
    zeroer.warmup(0)
    assert bool(torch.all(raw == 1))
    zeroer.warmup(4)
    assert bool(torch.all(raw[0] == 0))
    assert bool(torch.all(raw[1:] == 1))
    adapter.logging.init_logger.assert_called_once_with("vllm.fl.txda.kv_zeroer")
    adapter.logger.info.assert_called_once_with(
        "TXDA KV zeroer initialized: device=%s segments=%d num_blocks=%d",
        torch.device("cpu"),
        1,
        4,
    )


def test_segments_retain_storage_after_source_references_are_discarded(adapter):
    zeroer = make_zeroer(adapter, [torch.ones(4, 1, 2, 4)])
    gc.collect()
    zeroer.zero_block_ids([2])
    assert bool(torch.all(zeroer._segments[0][2] == 0))
    assert bool(torch.all(zeroer._segments[0][1] == 1))


@pytest.mark.parametrize(
    "kind", ["count", "kernel_size", "stride", "page", "address", "alias_stride"]
)
def test_invalid_geometry_fails_before_zeroing(adapter, kind):
    raw = torch.ones(128, dtype=torch.bfloat16)
    tensors = [raw.as_strided((4, 1, 2, 4), (16, 8, 4, 1))]
    kwargs = {}
    if kind == "count":
        kwargs["blocks"] = 3
    elif kind == "kernel_size":
        kwargs["kernel_sizes"] = [3]
    elif kind == "stride":
        tensors = [raw.as_strided((4, 1, 1, 4), (9, 4, 4, 1))]
    elif kind == "page":
        tensors = [raw.as_strided((4, 1, 1, 3), (8, 3, 3, 1))]
    elif kind == "address":
        tensors = [raw[1:].as_strided((4, 1, 2, 4), (16, 8, 4, 1))]
    else:
        tensors.append(raw.as_strided((4, 1, 2, 4), (20, 8, 4, 1)))
    with pytest.raises(ValueError):
        make_zeroer(adapter, tensors, **kwargs)
    assert bool(torch.all(raw == 1))


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu", "txda"])
def test_install_changes_only_txda_v2_binding_and_preserves_real_runner(
    adapter, monkeypatch, device
):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type=device)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    original, runner_class = object(), type("GPUModelRunner", (), {})
    runner = SimpleNamespace(KVBlockZeroer=original, GPUModelRunner=runner_class)
    assert adapter.module.patch_txda_v2_kv_zeroer(runner) is (device == "txda")
    expected = adapter.module.TxdaKVBlockZeroer if device == "txda" else original
    assert runner.KVBlockZeroer is expected
    assert runner.GPUModelRunner is runner_class
    if device == "txda":
        assert adapter.module.patch_txda_v2_kv_zeroer(runner)
        assert runner.KVBlockZeroer is expected
