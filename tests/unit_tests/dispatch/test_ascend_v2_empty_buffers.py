# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep V2 empty-buffer creation safe under the complete Ascend policy."""

from contextlib import contextmanager

import pytest
import torch

pytest.importorskip("torch_npu")
flag_gems = pytest.importorskip("flag_gems")

from vllm_fl.utils import get_flag_gems_whitelist_blacklist

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]

_NATIVE_EMPTY_OPS = {"zero_", "full_like", "arange", "arange_start"}
_ENABLED_FACTORIES = {"zeros", "zeros_like"}


@pytest.fixture(autouse=True)
def _require_ascend():
    if not torch.npu.is_available() or flag_gems.vendor_name != "ascend":
        pytest.skip("Requires Ascend NPU and Ascend FlagGems")


@contextmanager
def _selected_policy():
    _, blacklist = get_flag_gems_whitelist_blacklist()
    # Own a temporary registration library rather than replacing the global
    # worker's FlagGems enable() registration. Duplicate-registration handling
    # remains the library's normal behavior in this full-policy regression.
    with flag_gems.use_gems(exclude=blacklist):
        registered = set(flag_gems.all_registered_ops())
        assert not (_NATIVE_EMPTY_OPS & registered), (
            "Empty-buffer operations must retain their native NPU routes: "
            f"{sorted(_NATIVE_EMPTY_OPS & registered)}"
        )
        assert _ENABLED_FACTORIES.issubset(registered)
        yield
        torch.npu.synchronize()


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_native_zero_and_full_like_preserve_empty_v2_buffers(dtype):
    buffer = torch.empty((0, 33), device="npu", dtype=dtype)
    with _selected_policy():
        result = buffer.zero_()
        filled = torch.full_like(buffer, 1)
        torch.npu.synchronize()
        assert result is buffer
    for output in (result, filled):
        assert output.shape == (0, 33)
        assert output.numel() == 0
        assert output.dtype == dtype
        assert output.device == buffer.device


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_native_arange_preserves_empty_v2_ranges(dtype):
    with _selected_policy():
        end_only = torch.arange(0, device="npu", dtype=dtype)
        start_end = torch.arange(3, 3, device="npu", dtype=dtype)
        torch.npu.synchronize()
    for output in (end_only, start_end):
        assert output.shape == (0,)
        assert output.numel() == 0
        assert output.dtype == dtype
        assert output.device.type == "npu"


@pytest.mark.parametrize("factory", ["zeros", "zeros_like"])
@pytest.mark.parametrize("shape", [(0, 33), (2048,)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_flaggems_zeros_factories_remain_enabled(factory, shape, dtype):
    expected = torch.zeros(shape, dtype=dtype)
    template = torch.empty(shape, device="npu", dtype=dtype)
    with _selected_policy():
        if factory == "zeros":
            result = torch.zeros(shape, device="npu", dtype=dtype)
        else:
            result = torch.zeros_like(template)
        torch.npu.synchronize()
    assert result.shape == shape
    assert result.dtype == dtype
    assert result.device == template.device
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
