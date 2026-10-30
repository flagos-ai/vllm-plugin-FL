# SPDX-License-Identifier: Apache-2.0
"""CPU policy/fallback contracts; numerical GPU evidence is recorded separately."""

from types import SimpleNamespace as NS

import pytest
import torch
from _qwen38_cpu import ROOT, declarations

VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"
moe = declarations(VENDOR / "moe_launch.py", "qwen38_hotspot_moe")
linear = declarations(VENDOR / "linear_policy.py", "qwen38_hotspot_linear")
qsa = declarations(
    VENDOR / "__init__.py",
    "qwen38_hotspot_split",
    names={"qsa_sparse_split_count"},
)
workspace = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/ops/qsa.py",
    "qwen38_hotspot_workspace",
    names={"qsa_prepare_split_workspace"},
)


def _moe_args():
    return dict(
        num_tokens=64,
        w1_shape=(512, 160, 2560),
        w2_shape=(512, 2560, 80),
        topk=10,
        dtype=torch.bfloat16,
        vendor="metax",
        device_name="MetaX C550",
    )


@pytest.mark.parametrize(
    "rows,eligible",
    [
        (1, False),
        (7, False),
        (8, True),
        (64, True),
        (1024, True),
        (1025, False),
        (2048, False),
    ],
)
def test_moe_scope_preserves_assignment_and_base(rows, eligible):
    base = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "sentinel": 123}
    args = _moe_args()
    args["num_tokens"] = rows
    g1, g2, selected = moe.select_stage_configs(base, **args)
    assert selected is eligible
    assert base == {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "sentinel": 123}
    if eligible:
        # Both GEMMs must consume the same BM16 expert assignment.
        assert g1["BLOCK_SIZE_M"] == g2["BLOCK_SIZE_M"] == 16
        assert (g1["BLOCK_SIZE_N"], g1["BLOCK_SIZE_K"]) == (32, 64)
        assert (g2["BLOCK_SIZE_N"], g2["BLOCK_SIZE_K"]) == (256, 32)
        assert g1["sentinel"] == g2["sentinel"] == 123
    else:
        assert g1 == g2 == base
    g1["sentinel"] = -1
    assert base["sentinel"] == g2["sentinel"] == 123


@pytest.mark.parametrize(
    "override",
    [
        {"vendor": "cuda"},
        {"device_name": "other"},
        {"dtype": torch.float16},
        {"topk": 8},
        {"quantized": True},
        {"has_lora": True},
        {"has_bias": True},
        {"activation": "relu"},
        {"w1_shape": (512, 128, 2560)},
    ],
)
def test_moe_unsupported_contract_retains_complete_config(override):
    base = {"BLOCK_SIZE_M": 32, "num_stages": 5, "SPLIT_K": 2}
    args = _moe_args()
    args.update(override)
    g1, g2, selected = moe.select_stage_configs(base, **args)
    assert not selected
    assert g1 == g2 == base


def _tensor(shape, *, device="cuda:0", dtype=torch.bfloat16, contiguous=True):
    return NS(
        shape=shape,
        ndim=len(shape),
        device=torch.device(device),
        dtype=dtype,
        is_contiguous=lambda: contiguous,
    )


@pytest.mark.parametrize(
    "rows,eligible",
    [
        (0, False),
        (1, True),
        (64, True),
        (65, False),
        (1023, False),
        (1024, True),
        (2048, True),
        (2049, False),
    ],
)
def test_gr_row_boundaries(rows, eligible):
    assert (
        linear.supports_native_grdown(_tensor((rows, 10240)), _tensor((320, 10240)))
        is eligible
    )


@pytest.mark.parametrize(
    "override",
    [
        {"device": "cpu"},
        {"dtype": torch.float16},
        {"contiguous": False},
    ],
)
def test_gr_unsupported_input_requires_caller_fallback(override):
    x = _tensor((64, 10240), **override)
    weight = _tensor((320, 10240))
    assert not linear.supports_native_grdown(x, weight)
    with pytest.raises(ValueError, match="caller"):
        linear.native_grdown(x, weight)


@pytest.mark.parametrize(
    "rows,splits",
    [
        (0, 1),
        (1, 16),
        (32, 16),
        (33, 8),
        (64, 8),
        (65, 1),
    ],
)
def test_qsa_decode_split_boundaries(rows, splits):
    query = _tensor((rows, 3, 256))
    cache = _tensor((1, 64, 1, 256))
    assert qsa.qsa_sparse_split_count(query, cache, 2051) == splits
    assert qsa.qsa_sparse_split_count(query, cache, 1024) == 1


def test_qsa_workspace_is_owned_by_layer_and_graph_bucket(monkeypatch):
    query = torch.empty(1, 3, 256, dtype=torch.bfloat16)
    cache = torch.empty(1, 64, 1, 256, dtype=torch.bfloat16)
    first, second = {}, {}
    prepare = workspace.qsa_prepare_split_workspace
    _, a = prepare(query, cache, 2051, first, num_splits=16)
    _, reused = prepare(query, cache, 2051, first, num_splits=16)
    _, other = prepare(query, cache, 2051, second, num_splits=16)
    _, bucket = prepare(query, cache, 2051, first, num_splits=8)
    assert a is reused
    assert a[0].data_ptr() != other[0].data_ptr() != bucket[0].data_ptr()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    # Cache hit is reused without allocation; capture-time miss returns None.
    fake_query = _tensor((1, 3, 256))
    assert prepare(fake_query, cache, 2051, first, num_splits=16)[1] is a
    assert prepare(fake_query, cache, 2051, {}, num_splits=16) == (16, None)
