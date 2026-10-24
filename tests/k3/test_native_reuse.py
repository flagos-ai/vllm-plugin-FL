# SPDX-License-Identifier: Apache-2.0
"""Local KDA bindings must not change native callers or hide an incompatible pin."""

import pytest

triton = pytest.importorskip("triton")
pytest.importorskip("vllm")
from triton import language as tl  # noqa: E402

from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.launch import (
    local_kernel,  # noqa: E402
)


@triton.jit
def precision_kernel(X, DEFAULT_DOT_PRECISION: tl.constexpr = None):
    pass


@triton.jit
def old_interface_kernel(X):
    pass


def test_local_binding_preserves_native_configuration():
    native = triton.heuristics({})(
        triton.autotune(configs=[triton.Config({}, num_warps=8)], key=[])(
            precision_kernel
        )
    )
    original_tuner = native.fn
    original_config = original_tuner.configs[0]
    candidate = local_kernel(
        native,
        heuristics={},
        configs=[triton.Config({"DEFAULT_DOT_PRECISION": "ieee"}, num_warps=4)],
        key=[],
    )
    assert native.fn is original_tuner
    assert original_tuner.configs[0] is original_config
    assert original_config.kwargs == {}
    assert original_config.num_warps == 8
    assert candidate.fn is not original_tuner
    assert candidate.fn.fn is precision_kernel
    assert candidate.fn.configs[0].kwargs == {"DEFAULT_DOT_PRECISION": "ieee"}
    assert candidate.fn.configs[0].num_warps == 4


def test_missing_native_precision_interface_fails_explicitly():
    with pytest.raises(TypeError, match="pinned vLLM KDA precision interface"):
        local_kernel(old_interface_kernel, heuristics={}, configs=[], key=[])


def test_unknown_wrapper_is_not_silently_unwrapped():
    class ForeignWrapper:
        fn = precision_kernel

    with pytest.raises(TypeError, match="Unsupported wrapper"):
        local_kernel(ForeignWrapper(), heuristics={}, configs=[], key=[])


@pytest.mark.parametrize("rows", [17, 16384, 17001])
@pytest.mark.parametrize("data_parallel", [False, True])
def test_vision_mlp_reuses_native_forward_with_same_chunk_boundaries(
    rows, data_parallel
):
    import torch

    from vllm_fl.models.kimi_k3_vision import K3VisionMLP

    class Projection(torch.nn.Module):
        output_size = 8

        def __init__(self, factor):
            super().__init__()
            self.factor = factor
            self.rows = []

        def forward(self, x):
            self.rows.append(x.shape[0])
            return x * self.factor, None

    mlp = K3VisionMLP.__new__(K3VisionMLP)
    torch.nn.Module.__init__(mlp)
    mlp.fc0 = Projection(2.0)
    mlp.fc1 = Projection(0.25)
    mlp.activation = torch.nn.ReLU()
    mlp.use_data_parallel = data_parallel
    x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) - rows
    out = mlp.forward(x)
    torch.testing.assert_close(out, torch.relu(x * 2.0) * 0.25, rtol=0, atol=0)
    expected = (
        [min(8192, rows - start) for start in range(0, rows, 8192)]
        if data_parallel and rows >= 16384
        else [rows]
    )
    assert mlp.fc0.rows == mlp.fc1.rows == expected
