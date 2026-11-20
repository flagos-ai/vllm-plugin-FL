from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_MLA_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl"
    / "dispatch"
    / "backends"
    / "vendor"
    / "hygon"
    / "impl"
    / "mla.py"
)
_MLA_SPEC = importlib.util.spec_from_file_location(
    "_vllm_fl_hygon_mla_test_module", _MLA_PATH
)
assert _MLA_SPEC is not None and _MLA_SPEC.loader is not None
_MLA_MODULE = importlib.util.module_from_spec(_MLA_SPEC)
_MLA_SPEC.loader.exec_module(_MLA_MODULE)

_VendorInt8Backend = _MLA_MODULE._VendorInt8Backend
_dequantize_int8_wo_a = _MLA_MODULE._dequantize_int8_wo_a
_int8_wo_a_group_gemm = _MLA_MODULE._int8_wo_a_group_gemm


class _FakeWoA(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)


@pytest.mark.parametrize("triton_layout", [False, True])
def test_dequantize_int8_wo_a_supports_checkpoint_and_triton_layouts(
    triton_layout: bool,
):
    groups, rank, hidden = 2, 3, 4
    out_features = groups * rank
    weight_nk = torch.arange(
        -12, -12 + out_features * hidden, dtype=torch.int8
    ).reshape(out_features, hidden)
    weight = weight_nk.t() if triton_layout else weight_nk
    scales = torch.arange(1, out_features + 1, dtype=torch.float32).reshape(-1, 1)
    wo_a = _FakeWoA(weight, scales)

    actual = _dequantize_int8_wo_a(wo_a, groups, rank, hidden)
    expected = (weight_nk.float() * scales).to(torch.bfloat16).reshape(
        groups, rank, hidden
    )

    assert actual.shape == (groups, rank, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dequantize_int8_wo_a_supports_per_tensor_scale():
    groups, rank, hidden = 2, 2, 3
    weight_nk = torch.arange(12, dtype=torch.int8).reshape(4, 3)
    wo_a = _FakeWoA(weight_nk.t(), torch.tensor([0.25]))

    actual = _dequantize_int8_wo_a(wo_a, groups, rank, hidden)
    expected = (weight_nk.float() * 0.25).to(torch.bfloat16).reshape(
        groups, rank, hidden
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dequantize_int8_wo_a_rejects_unknown_layout():
    wo_a = _FakeWoA(torch.empty((5, 7), dtype=torch.int8), torch.ones(5, 1))

    with pytest.raises(ValueError, match="Unexpected INT8 wo_a weight shape"):
        _dequantize_int8_wo_a(wo_a, 2, 3, 4)


def test_int8_wo_a_group_gemm_slices_weights_and_scales_per_group():
    groups, rank, hidden, tokens = 2, 3, 4, 2
    out_features = groups * rank
    weight_nk = torch.arange(
        -12, -12 + out_features * hidden, dtype=torch.int8
    ).reshape(out_features, hidden)
    scales = torch.arange(1, out_features + 1, dtype=torch.float32).reshape(-1, 1)
    wo_a = _FakeWoA(weight_nk, scales)
    o_ref = torch.arange(
        tokens * groups * hidden, dtype=torch.bfloat16
    ).reshape(tokens, groups, hidden)
    gemm_calls: list[tuple[int, int, int, str]] = []

    def _quantize(input: torch.Tensor):
        return input.to(torch.int8), torch.ones((input.shape[0], 1))

    def _gemm(input_q, weight, input_scale, weight_scale, m, n, k, trans, dtype):
        gemm_calls.append((m, n, k, trans))
        output = input_q.float() @ weight.float().t()
        output *= input_scale
        output *= weight_scale.t()
        return None, output.to(dtype)

    backend = _VendorInt8Backend("fake", _quantize, _gemm)
    actual = _int8_wo_a_group_gemm(o_ref, wo_a, groups, rank, backend)

    expected_groups = []
    for group_idx in range(groups):
        channel_slice = slice(group_idx * rank, (group_idx + 1) * rank)
        expected_groups.append(
            (
                o_ref[:, group_idx, :].float()
                @ weight_nk[channel_slice].float().t()
            )
            * scales[channel_slice].t()
        )
    expected = torch.stack(expected_groups, dim=1).to(torch.bfloat16)

    assert gemm_calls == [(tokens, rank, hidden, "NT")] * groups
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
