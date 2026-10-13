# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU index_fill_ regression coverage under the normal FlagGems policy."""

from math import prod

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "shape,dim,indices,dtype,value,transpose,tensor_value",
    [
        # Synthetic vocabulary width, not a captured CI tensor shape. A contiguous
        # dim-0 fill produces grid.y=ceil(151936/64)=2374 in the affected FlagGems
        # kernel, exceeding the GCU limit of 255 even with a small row count.
        pytest.param(
            (4, 151936),
            0,
            [0, 2],
            torch.int32,
            0,
            False,
            False,
            id="large-vocabulary-rows",
        ),
        pytest.param(
            (4, 151936),
            0,
            [2, 0, 2],
            torch.int32,
            0,
            False,
            False,
            id="duplicate-row-indices",
        ),
        pytest.param(
            (4, 151936),
            0,
            [],
            torch.int32,
            0,
            False,
            False,
            id="empty-row-indices",
        ),
        pytest.param(
            (4, 151936),
            0,
            [0, 2],
            torch.int32,
            0,
            True,
            False,
            id="noncontiguous-rows",
        ),
        pytest.param(
            (4, 151936),
            -2,
            [1, 3],
            torch.int32,
            0,
            False,
            False,
            id="negative-dimension",
        ),
        pytest.param(
            (4, 151936),
            0,
            [0, 2],
            torch.int32,
            0,
            False,
            True,
            id="tensor-scalar-value",
        ),
        pytest.param(
            (4, 17),
            1,
            [0, 8, 16],
            torch.int32,
            -3,
            False,
            False,
            id="column-fill",
        ),
        pytest.param(
            (4, 151936),
            0,
            [1, 3],
            torch.int64,
            -3,
            False,
            False,
            id="int64-value",
        ),
        pytest.param(
            (4, 151936),
            0,
            [1, 3],
            torch.float32,
            0.5,
            False,
            False,
            id="float32-value",
        ),
        pytest.param(
            (4, 151936),
            0,
            [1, 3],
            torch.float16,
            0.5,
            False,
            False,
            id="float16-value",
        ),
        pytest.param(
            (4, 151936),
            0,
            [1, 3],
            torch.bfloat16,
            0.5,
            False,
            False,
            id="bfloat16-value",
        ),
        pytest.param(
            (4, 151936),
            0,
            [1, 3],
            torch.bool,
            False,
            False,
            False,
            id="boolean-value",
        ),
        pytest.param(
            (1, 1, 1, 1, 1, 1, 4, 1, 151936),
            6,
            [0, 2],
            torch.int32,
            0,
            False,
            False,
            id="high-rank-contiguous",
        ),
        pytest.param(
            (4, 151936),
            0,
            [0, 2] * 128,
            torch.int32,
            0,
            False,
            False,
            id="duplicate-indices-across-launch-tiles",
        ),
    ],
)
def test_index_fill_gcu_with_platform_policy(
    device, shape, dim, indices, dtype, value, transpose, tensor_value
):
    if device.type != "gcu":
        pytest.skip("Requires the Enflame GCU backend")

    # A missing FlagGems dependency on GCU must fail this test, not skip it.
    import flag_gems

    from vllm_fl.dispatch.backends.vendor.gcu.impl.index_fill import (
        apply_index_fill_gcu_patch,
    )
    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    apply_index_fill_gcu_patch()

    base_shape = shape[::-1] if transpose else shape
    source = (torch.arange(prod(shape)) % 7).to(dtype).reshape(base_shape)
    if transpose:
        source = source.t()
    expected = source.clone()
    cpu_indices = torch.tensor(indices, dtype=torch.int64)
    cpu_value = torch.tensor(value, dtype=dtype) if tensor_value else value
    expected.index_fill_(dim, cpu_indices, cpu_value)

    actual = source.to(device)
    device_indices = cpu_indices.to(device)
    device_value = cpu_value.to(device) if tensor_value else value
    assert actual.is_contiguous() is (not transpose)
    original_pointer = actual.data_ptr()

    # Match worker registration using the actual platform policy. In this
    # FlagGems API, use_gems(exclude=...) delegates to enable(unused=...).
    # Keep the actual platform policy without test-only exclusions: the
    # unadapted launcher must reproduce the large-vocabulary launch failure.
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    with flag_gems.use_gems(include=whitelist, exclude=blacklist):
        returned = actual.index_fill_(dim, device_indices, device_value)
        torch.gcu.synchronize()

    assert returned is actual
    assert actual.data_ptr() == original_pointer
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
