# Copyright (c) 2026 BAAI. All rights reserved.
"""Use real Tensor indexing to check GDN row copies, guards and isolation."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker/txda_gdn_state.py"


@pytest.fixture
def adapter(monkeypatch):
    logger = Mock()
    logging = ModuleType("vllm.logger")
    logging.init_logger = Mock(return_value=logger)
    monkeypatch.setitem(sys.modules, logging.__name__, logging)
    spec = importlib.util.spec_from_file_location("txda_gdn_state_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(device_type="txda")
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    upstream = ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")

    class QwenGatedDeltaNetAttention:
        def _forward_core(self, *args, **kwargs):
            return self.operation(*args, **kwargs)

    upstream.QwenGatedDeltaNetAttention = QwenGatedDeltaNetAttention
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    return SimpleNamespace(
        module=module,
        logger=logger,
        platforms=platforms,
        cls=QwenGatedDeltaNetAttention,
    )


def padded_cache(dtype=torch.float32, padded=True):
    rows, row_size, offset = 6, 12, 11
    stride = row_size + (7 if padded else 0)
    backing = torch.arange(offset + rows * stride + 13, dtype=torch.float32).to(dtype)
    cache = backing.as_strided((rows, 2, 2, 3), (stride, 6, 3, 1), offset)
    return backing, cache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("rows", [[4], [4, 1, 3], [-1, 0, -3], []])
def test_exact_public_gather_scatter_preserves_all_unselected_storage(
    adapter, dtype, index_dtype, padded, rows
):
    raw, cache = padded_cache(dtype, padded)
    expected_raw = raw.clone()
    expected_cache = expected_raw.as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    )
    index = torch.tensor(rows, dtype=index_dtype)
    values = (
        torch.arange(len(rows) * 12, dtype=torch.float32)
        .reshape(len(rows), 2, 2, 3)
        .to(dtype)
        - 80
    )
    expected_gather = expected_cache[index]
    expected_cache[index] = values
    report = Mock()
    mode = adapter.module.TxdaStateCacheIndexMode(cache, report)
    with mode:
        gathered = cache[index]
        returned = cache.index_put_((index,), values, accumulate=False)
    torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)
    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
    assert returned is cache
    assert mode.gather_hits == mode.scatter_hits == 1
    if rows:
        assert report.call_args_list == [
            (("gather", len(rows)),),
            (("scatter", len(rows)),),
        ]
        gathered.fill_(-300)
        torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
    else:
        report.assert_not_called()


@pytest.mark.parametrize("count", [1, 2])
def test_strided_source_values_use_native_write_and_keyword_gather_still_works(
    adapter, count
):
    raw, cache = padded_cache()
    index = torch.tensor([4, 1][:count], dtype=torch.int64)
    values = torch.arange(count * 24, dtype=torch.float32).reshape(count, 2, 2, 6)[
        ..., ::2
    ]
    expected = raw.clone()
    expected.as_strided(cache.shape, cache.stride(), cache.storage_offset())[index] = (
        values
    )
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        if count == 1:
            cache[index] = values
            result = cache
        else:
            result = torch.ops.aten.index_put_.default(
                self=cache, indices=[index], values=values, accumulate=False
            )
        gathered = torch.ops.aten.index.Tensor(self=cache, indices=[index])
    assert result is cache
    assert mode.gather_hits == 1 and mode.scatter_hits == 0
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    torch.testing.assert_close(gathered, values, rtol=0, atol=0)


def test_gradient_tracking_cache_preserves_native_gather_autograd(adapter):
    base = torch.arange(72.0, requires_grad=True)
    cache = base.reshape(6, 2, 2, 3)
    index = torch.tensor([4, 1])
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        gathered = cache[index]
    assert mode.gather_hits == 0
    gathered.sum().backward()
    expected = torch.zeros_like(cache)
    expected[index] = 1
    torch.testing.assert_close(base.grad, expected.reshape(-1), rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 2])
def test_gradient_tracking_values_preserve_native_scatter_autograd(adapter, count):
    _, cache = padded_cache()
    values = torch.full((count, 2, 2, 3), -90.0, requires_grad=True)
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        cache[torch.tensor([4, 1][:count])] = values
    assert mode.scatter_hits == 0
    cache.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [-7, 6, 100])
def test_all_bounds_are_validated_before_any_scatter_mutation(adapter, invalid):
    raw, cache = padded_cache()
    before = raw.clone()
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with pytest.raises(IndexError, match="outside the cache"), mode:
        cache[torch.tensor([1, invalid])] = torch.zeros(2, 2, 2, 3)
    assert mode.scatter_hits == 0
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_other_tensors_alias_objects_and_other_operators_are_not_intercepted(adapter):
    _, cache = padded_cache()
    alias = cache.as_strided(cache.shape, cache.stride(), cache.storage_offset())
    other = cache.clone()
    index = torch.tensor([1, 3])
    callback = Mock()
    mode = adapter.module.TxdaStateCacheIndexMode(cache, callback)
    with mode:
        aliased = alias[index]
        unrelated = other[index]
        sliced = cache[1:3]
        added = cache + 4
    assert mode.gather_hits == mode.scatter_hits == 0
    callback.assert_not_called()
    torch.testing.assert_close(aliased, cache[index], rtol=0, atol=0)
    torch.testing.assert_close(unrelated, other[index], rtol=0, atol=0)
    torch.testing.assert_close(sliced, cache[1:3], rtol=0, atol=0)
    torch.testing.assert_close(added, cache + 4, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [[1, 1], [-1, 5]])
def test_duplicate_normalized_ids_retain_native_gather_and_scatter(adapter, rows):
    _, cache = padded_cache()
    expected = cache.clone()
    index = torch.tensor(rows)
    values = torch.full((2, 2, 2, 3), -90.0)
    expected_gather = expected[index]
    expected[index] = values
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        gathered = cache[index]
        cache[index] = values
    assert mode.gather_hits == mode.scatter_hits == 0
    torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "pattern",
    ["boolean", "scalar", "multidim", "multi_axis", "accumulate", "broadcast"],
)
def test_unknown_index_patterns_and_broadcast_values_use_native_dispatch(
    adapter, pattern
):
    _, cache = padded_cache()
    expected = cache.clone()
    index = torch.tensor([1, 3])
    if pattern == "boolean":
        index = torch.tensor([True, False, False, True, False, False])
    elif pattern == "scalar":
        index = torch.tensor(2)
    elif pattern == "multidim":
        index = torch.tensor([[1, 3], [2, 4]])
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        if pattern == "multi_axis":
            actual = cache[index, index % 2]
            reference = expected[index, index % 2]
        elif pattern == "accumulate":
            values = torch.full((2, 2, 2, 3), 10.0)
            actual = cache.index_put_((index,), values, accumulate=True)
            reference = expected.index_put_((index,), values, accumulate=True)
        elif pattern == "broadcast":
            cache[index] = 10.0
            expected[index] = 10.0
            actual, reference = cache, expected
        else:
            actual, reference = cache[index], expected[index]
    assert mode.gather_hits == mode.scatter_hits == 0
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "layout",
    ["strided_inner", "transposed", "overlapping_pages", "wrong_rank", "wrong_dtype"],
)
def test_unproven_cache_layouts_retain_native_indexing(adapter, layout):
    _, cache = padded_cache()
    if layout == "strided_inner":
        cache = torch.arange(288.0).reshape(6, 2, 2, 12)[..., ::4]
    elif layout == "transposed":
        cache = cache.transpose(-1, -2)
    elif layout == "overlapping_pages":
        cache = torch.arange(60.0).as_strided((6, 2, 2, 3), (8, 6, 3, 1))
    elif layout == "wrong_rank":
        cache = torch.arange(72.0).reshape(6, 2, 6)
    else:
        cache = cache.to(torch.float16)
    index = torch.tensor([4, 1])
    expected = cache[index]
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        actual = cache[index]
    assert mode.gather_hits == mode.scatter_hits == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 2])
def test_overlapping_values_keep_native_overlap_errors_or_order(adapter, count):
    def run(use_mode):
        raw, cache = padded_cache()
        values = cache[1 : 1 + count]
        index = torch.tensor([2, 3][:count])
        mode = adapter.module.TxdaStateCacheIndexMode(cache)
        error = None
        try:
            if use_mode:
                with mode:
                    cache[index] = values
            else:
                cache[index] = values
        except RuntimeError as exc:
            error = str(exc)
        return raw, error, mode

    expected, native_error, _ = run(False)
    actual, mode_error, mode = run(True)
    assert mode.scatter_hits == 0
    assert mode_error == native_error
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("values_shape", [(), (3,), (2, 3)])
def test_single_row_setter_does_not_intercept_general_broadcasts(adapter, values_shape):
    raw, cache = padded_cache()
    expected = raw.clone()
    expected_cache = expected.as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    )
    index = torch.tensor([4], dtype=torch.int32)
    values = torch.full(values_shape, -80.0)
    expected_cache[index] = values
    mode = adapter.module.TxdaStateCacheIndexMode(cache)
    with mode:
        cache[index] = values
    assert mode.scatter_hits == 0
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu", "musa"])
def test_install_does_not_modify_other_platforms(adapter, device):
    original = adapter.cls._forward_core
    adapter.platforms.current_platform.device_type = device
    assert not adapter.module.patch_txda_gdn_state()
    assert adapter.cls._forward_core is original


@pytest.mark.parametrize(
    "cache", [None, [], [None], [None, object()], [None, torch.zeros(2, 2, 2, 3)]]
)
def test_non_txda_or_non_tensor_cache_calls_original_unchanged(adapter, cache):
    original = adapter.cls._forward_core
    assert adapter.module.patch_txda_gdn_state()
    instance = adapter.cls()
    instance.kv_cache = cache
    instance.operation = Mock(return_value=object())
    marker = object()
    result = instance._forward_core(marker, key="kept")
    assert result is instance.operation.return_value
    instance.operation.assert_called_once_with(marker, key="kept")
    assert adapter.cls._forward_core.__wrapped__ is original
    adapter.logger.info.assert_not_called()


def test_thin_wrapper_keeps_upstream_core_and_reports_only_first_actual_copy(
    adapter, monkeypatch
):
    original = adapter.cls._forward_core
    assert adapter.module.patch_txda_gdn_state()
    wrapped = adapter.cls._forward_core
    assert adapter.module.patch_txda_gdn_state()
    assert adapter.cls._forward_core is wrapped
    assert wrapped.__wrapped__ is original
    # Exercise the production wrapper with real CPU Tensors, changing only
    # the test-local device guard. No fake Tensor or global Torch patch is used.
    monkeypatch.setattr(
        adapter.module,
        "_is_txda_state_cache",
        lambda value: type(value) is torch.Tensor,
    )
    _, cache = padded_cache()
    other = cache.clone()
    index = torch.tensor([4, 1], dtype=torch.int32)
    values = torch.full((2, 2, 2, 3), -60.0)
    instance = adapter.cls()
    instance.kv_cache = [object(), cache]
    calls = []

    def public_core(argument, *, marker):
        calls.append((argument, marker))
        initial = cache[index]
        cache[index] = values
        return initial, other[index], marker

    instance.operation = public_core
    before, other_expected = cache[index], other[index]
    marker = object()
    result = instance._forward_core("argument", marker=marker)
    torch.testing.assert_close(result[0], before, rtol=0, atol=0)
    torch.testing.assert_close(result[1], other_expected, rtol=0, atol=0)
    assert result[2] is marker
    instance._forward_core("again", marker=marker)
    assert calls == [("argument", marker), ("again", marker)]
    adapter.logger.info.assert_called_once_with(
        "TXDA GDN state indexing active: operation=%s device=%s "
        "cache_shape=%s cache_strides=%s rows=%d",
        "gather",
        cache.device,
        tuple(cache.shape),
        cache.stride(),
        2,
    )


def test_wrapper_propagates_upstream_errors_and_leaves_no_mode_installed(
    adapter, monkeypatch
):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    assert adapter.module.patch_txda_gdn_state()
    monkeypatch.setattr(adapter.module, "_is_txda_state_cache", lambda _: True)
    instance = adapter.cls()
    _, cache = padded_cache()
    instance.kv_cache = [object(), cache]
    instance.operation = Mock(side_effect=RuntimeError("original core failed"))
    before_mode = _get_current_dispatch_mode()
    with pytest.raises(RuntimeError, match="original core failed"):
        instance._forward_core("unchanged")
    assert _get_current_dispatch_mode() is before_mode
    adapter.logger.info.assert_not_called()
