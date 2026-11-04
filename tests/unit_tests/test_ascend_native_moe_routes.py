# SPDX-License-Identifier: Apache-2.0
"""CPU numerical checks for the native Ascend MoE route boundary."""

import ast
from pathlib import Path

import pytest
import torch


def _prepare_routes(*args):
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_native_routes"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_prepare_native_routes"](*args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_route_preparation_preserves_weighted_valid_experts(dtype):
    x = torch.tensor([[1, 2, 3], [float("nan")] * 3, [3, 1, 2], [2, 3, 1]], dtype=dtype)
    ids = torch.tensor([[1, -1], [-1, -1], [99, 2], [0, 3]], dtype=torch.int64)
    weights = torch.tensor([[0.25, float("nan")], [7, 9], [11, 0.6], [0.2, 0.8]])
    originals = (x.clone(), ids.clone(), weights.clone())
    native_x, native_ids, native_weights = _prepare_routes(x, ids, weights, 4)
    assert native_ids.dtype == torch.int32
    assert ((native_ids >= 0) & (native_ids < 4)).all()
    assert torch.isfinite(native_x).all()
    assert torch.isfinite(native_weights).all()
    assert torch.count_nonzero(native_x[1]) == 0
    torch.testing.assert_close(
        native_weights, torch.tensor([[0.25, 0], [0, 0], [0, 0.6], [0.2, 0.8]])
    )

    # A distinct affine expert function makes wrong expert IDs observable.
    def expert(row, index):
        return row.float() * (index + 1) + index

    expected = torch.zeros(4, 3)
    actual = torch.zeros_like(expected)
    for row in range(4):
        for slot in range(2):
            index = int(ids[row, slot])
            if 0 <= index < 4:
                expected[row] += expert(x[row], index) * weights[row, slot]
            actual[row] += (
                expert(native_x[row], int(native_ids[row, slot]))
                * native_weights[row, slot]
            )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_input, original in zip((x, ids, weights), originals):
        torch.testing.assert_close(
            actual_input, original, rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_route_preparation_keeps_positive_routes_unchanged(dtype):
    x = torch.arange(12).view(4, 3).to(dtype)
    ids = torch.tensor([[1, 3], [0, 2], [2, 2], [3, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.8]] * 4)
    prepared = _prepare_routes(x, ids, weights, 4)
    for actual, expected in zip(prepared, (x, ids, weights)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
