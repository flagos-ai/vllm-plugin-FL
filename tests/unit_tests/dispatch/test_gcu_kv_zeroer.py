# Copyright (c) 2026 BAAI. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific unit tests only run on Enflame", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import kv_zeroer


class _CaptureKernel:
    def __init__(self):
        self.grid = None
        self.arguments = None
        self.keywords = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*arguments, **keywords):
            self.arguments = arguments
            self.keywords = keywords

        return launch


@pytest.mark.parametrize("program_limit", [65535, 4])
def test_launcher_bounds_grid_and_preserves_segment_metadata(
    monkeypatch, program_limit
):
    kernel = _CaptureKernel()
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", kernel)
    monkeypatch.setattr(kv_zeroer, "GCU_MAX_GRID_X", program_limit)
    copied = []
    index_tensor = object()

    def copy_ids(ids, **kwargs):
        copied.append((ids, kwargs))
        return index_tensor

    monkeypatch.setattr(kv_zeroer, "async_tensor_h2d", copy_ids)
    segments = (object(), object(), object())
    zeroer = SimpleNamespace(device="gcu", _meta=(*segments, 3, 4, 784))
    ids = list(range(100))
    kv_zeroer.zero_block_ids_gcu(zeroer, ids)

    assert copied == [(ids, {"device": "gcu", "dtype": torch.int64})]
    assert kernel.grid == (program_limit,)
    assert kernel.arguments == (*segments, index_tensor, 100 * 784 * 3)
    assert kernel.keywords == {
        "BLOCK_SIZE": 4,
        "N_SEGS": 784,
        "MAX_CHUNKS": 3,
    }


def test_empty_blocks_and_missing_metadata_do_not_copy_or_launch(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Empty zeroing work must not copy or launch")

    monkeypatch.setattr(kv_zeroer, "async_tensor_h2d", unexpected)
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", unexpected)
    kv_zeroer.zero_block_ids_gcu(SimpleNamespace(_meta=None), [0])
    kv_zeroer.zero_block_ids_gcu(SimpleNamespace(_meta=object()), [])


def test_patch_is_gcu_only_and_warmup_uses_new_launcher(monkeypatch):
    import vllm.platforms
    from vllm.v1.worker import utils as worker_utils

    official_kernel = worker_utils._zero_kv_blocks_kernel
    original_method = worker_utils.KVBlockZeroer.zero_block_ids
    monkeypatch.setattr(worker_utils.KVBlockZeroer, "zero_block_ids", original_method)
    monkeypatch.setattr(kv_zeroer, "_patched", False)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="nvidia")
    )
    kv_zeroer.apply_kv_zeroer_gcu_patch()
    assert worker_utils.KVBlockZeroer.zero_block_ids is original_method
    assert not kv_zeroer._patched

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="enflame")
    )
    kv_zeroer.apply_kv_zeroer_gcu_patch()
    kv_zeroer.apply_kv_zeroer_gcu_patch()
    assert worker_utils.KVBlockZeroer.zero_block_ids is kv_zeroer.zero_block_ids_gcu
    assert worker_utils._zero_kv_blocks_kernel is official_kernel

    kernel = _CaptureKernel()
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", kernel)
    copied = []
    monkeypatch.setattr(
        kv_zeroer,
        "async_tensor_h2d",
        lambda ids, **kwargs: copied.append(ids) or object(),
    )
    zeroer = object.__new__(worker_utils.KVBlockZeroer)
    zeroer.device = "gcu"
    zeroer._meta = (object(), object(), object(), 3, 4, 784)
    zeroer.warmup(4)
    assert copied == [[0]]
    assert kernel.grid == (784 * 3,)
    zeroer.warmup(0)
    assert copied == [[0]]


@pytest.mark.skipif(_test_platform != "enflame", reason="Requires an actual GCU")
def test_native_zeroing_784_segments_preserves_unselected_blocks_and_padding(
    monkeypatch,
):
    # Force multiple iterations per program, including skipped chunks of short
    # pages. This covers both the grid.y=784 failure and bounded-grid striding.
    monkeypatch.setattr(kv_zeroer, "GCU_MAX_GRID_X", 4)
    n_segs, n_blocks = 784, 4
    row_size = n_blocks * 16
    original = torch.arange(1, n_segs * row_size + 1, dtype=torch.int32).reshape(
        n_segs, row_size
    )
    storage = original.to("gcu")
    strides = [16 if segment % 2 == 0 else 12 for segment in range(n_segs)]
    pages = [(3, 7, 11)[segment % 3] for segment in range(n_segs)]
    addresses = [
        storage.data_ptr() + segment * row_size * 4 for segment in range(n_segs)
    ]
    device = storage.device
    meta = (
        torch.tensor(addresses, dtype=torch.uint64, device=device),
        torch.tensor(strides, dtype=torch.int64, device=device),
        torch.tensor(pages, dtype=torch.int64, device=device),
        3,
        4,
        n_segs,
    )
    zeroer = SimpleNamespace(device=device, _meta=meta)
    block_ids = [2, 0]
    kv_zeroer.zero_block_ids_gcu(zeroer, block_ids)

    expected = original.clone()
    for segment in range(n_segs):
        for block_id in block_ids:
            start = block_id * strides[segment]
            expected[segment, start : start + pages[segment]] = 0
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)
