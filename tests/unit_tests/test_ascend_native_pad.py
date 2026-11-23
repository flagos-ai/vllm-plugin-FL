# SPDX-License-Identifier: Apache-2.0
"""Native pad regression for the actual Qwen3.6-27B TP2 r2 vector fault.

Worker plogs identify _pad_jit_function_1, hash 17118823940470532360.
The original failure was in the complete model; standalone pad need not
reproduce its allocator/lifetime conditions. These cases verify the narrow
native fallback's mathematics, source guards and output alias contract.
"""

import importlib
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _pad_case(length, dtype, layout):
    heads, dim = 24, 128
    prefix = 0 if layout == "contiguous" else 32
    width = 8192 if layout == "packed_offset" else heads * dim
    size = width * length
    storage = torch.full((prefix + size + 32,), -777.0, dtype=dtype)
    values = (torch.arange(size).remainder(97) - 48).to(dtype)
    storage[prefix : prefix + size] = values
    if layout == "packed_offset":
        rows = storage[prefix : prefix + size].view(1, length, width)
        x = (
            rows[:, :, 512 : 512 + heads * dim]
            .view(1, length, heads, dim)
            .transpose(1, 2)
        )
    else:
        x = storage[prefix : prefix + size].view(1, heads, length, dim)
    padded = ((length + 63) // 64) * 64
    expected = torch.zeros(1, heads, padded, dim, dtype=dtype)
    expected[:, :, :length] = x
    return storage, x, (0, 0, 0, padded - length), expected


def test_ascend_excludes_both_registered_entries_for_the_faulting_pad_kernel():
    policy = yaml.safe_load(
        (ROOT / "vllm_fl/dispatch/config/ascend.yaml").read_text(encoding="utf-8")
    )
    assert {"pad", "constant_pad_nd"} <= set(policy["flagos_blacklist"])


@pytest.mark.parametrize("length", [1, 2, 15, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "offset", "packed_offset"])
def test_native_pad_preserves_data_storage_and_output_contract(length, dtype, layout):
    storage, x, padding, expected = _pad_case(length, dtype, layout)
    original = storage.clone()
    if length == 2 and layout == "contiguous":
        assert x.shape == (1, 24, 2, 128)
        assert x.stride() == (6144, 256, 128, 1)
    for actual in (
        F.pad(x, padding),
        torch.ops.aten.constant_pad_nd.default(x, padding, 0),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.is_contiguous()
        assert actual.data_ptr() != x.data_ptr()
        assert (
            actual.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(storage, original, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("length", [1, 2, 15, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "offset", "packed_offset"])
def test_ascend_policy_uses_native_pad_with_full_gems_active(
    length, dtype, layout, monkeypatch
):
    pytest.importorskip("torch_npu")
    flag_gems = pytest.importorskip("flag_gems")
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    pad_module = importlib.import_module("flag_gems.ops.pad")

    def faulting_gems_pad_must_not_run(*args, **kwargs):
        raise AssertionError(
            "Ascend entered the pad implementation identified by r2 plog"
        )

    monkeypatch.setattr(
        pad_module.PadFunction, "__call__", faulting_gems_pad_must_not_run
    )
    storage, x, padding, expected = _pad_case(length, dtype, layout)
    npu_storage = storage.npu()
    npu_input = npu_storage.as_strided(x.shape, x.stride(), x.storage_offset())
    # This regression exercises the default full policy even when the CI
    # launcher supplies a minimal operator whitelist to other unit tests.
    for name in (
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(name, raising=False)
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert whitelist is None
    assert {"pad", "constant_pad_nd"} <= set(blacklist)
    with (
        torch.inference_mode(),
        flag_gems.use_gems(include=whitelist, exclude=blacklist),
    ):
        outputs = (
            F.pad(npu_input, padding),
            torch.ops.aten.constant_pad_nd.default(npu_input, padding, 0),
        )
    torch.npu.synchronize()
    for actual in outputs:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert actual.is_contiguous()
        assert (
            actual.untyped_storage().data_ptr()
            != npu_storage.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(npu_storage.cpu(), storage, rtol=0, atol=0)
