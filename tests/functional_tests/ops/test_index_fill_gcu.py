# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU index_fill_ regression coverage under the normal FlagGems policy."""

import json
from math import prod

import pytest
import torch

pytestmark = pytest.mark.gpu


def _mismatch_facts(actual, expected):
    """Summarize CPU snapshots without dumping whole tensors into CI logs."""
    assert actual.device.type == expected.device.type == "cpu"
    assert actual.shape == expected.shape and actual.ndim == 2
    different = actual != expected
    coordinates = different.nonzero(as_tuple=False)

    def sample(coordinate):
        row, column = coordinate.tolist()
        return {
            "coordinate": [row, column],
            "actual": actual[row, column].item(),
            "expected": expected[row, column].item(),
        }

    return {
        "mismatch_count": len(coordinates),
        "per_row_counts": different.sum(dim=1).tolist()[:16],
        "rows_truncated": actual.shape[0] > 16,
        "first_mismatches": [sample(c) for c in coordinates[:16]],
        "last_bad": sample(coordinates[-1]) if len(coordinates) else None,
    }


def _record_int64_roundtrips(actual, first_cpu, expected, record_property):
    """Probe only after the original D2H snapshot, retaining any probe errors."""
    copies = {"first_int64": first_cpu}
    errors = {}
    probes = {
        "raw_int32_words": lambda: actual.view(torch.int32).cpu().view(torch.int64),
        "second_int64": actual.cpu,
    }
    for name, probe in probes.items():
        try:
            copies[name] = probe()
        except Exception as error:
            errors[name] = {"type": type(error).__name__, "message": str(error)[:256]}

    facts = {
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "elements": expected.numel(),
        "versus_expected": {
            name: _mismatch_facts(copy, expected) for name, copy in copies.items()
        },
        "versus_first_mismatch_counts": {
            name: (copy != first_cpu).sum().item()
            for name, copy in copies.items()
            if name != "first_int64"
        },
        "probe_errors": errors,
    }
    encoded = json.dumps(facts, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(encoded) <= 8192, "Int64 diagnostic exceeded its byte budget"
    record_property("gcu_int64_roundtrips", facts)
    return copies, errors


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
        pytest.param(
            (4, 151808),
            0,
            [1, 3],
            torch.int64,
            -3,
            False,
            False,
            id="int64-aligned-fill-count",
        ),
    ],
)
def test_index_fill_gcu_with_platform_policy(
    device, shape, dim, indices, dtype, value, transpose, tensor_value, record_property
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
    first_cpu = actual.cpu()
    try:
        torch.testing.assert_close(first_cpu, expected, rtol=0, atol=0)
    except AssertionError:
        if dtype == torch.int64:
            try:
                _record_int64_roundtrips(actual, first_cpu, expected, record_property)
            except Exception as error:
                # Diagnostic failures must not replace the original exact assertion.
                record_property(
                    "gcu_int64_diagnostic_error",
                    {"type": type(error).__name__, "message": str(error)[:256]},
                )
        raise


def test_int64_gcu_transfer_without_index_fill(device, record_property):
    """Keep an independent transfer baseline for the original int64 pattern."""
    if device.type != "gcu":
        pytest.skip("Requires the Enflame GCU backend")

    source = (torch.arange(4 * 151936) % 7).reshape(4, 151936)
    actual = source.to(device)
    first_cpu = actual.cpu()
    copies, errors = _record_int64_roundtrips(
        actual, first_cpu, source, record_property
    )
    for copy in copies.values():
        torch.testing.assert_close(copy, source, rtol=0, atol=0)
    assert not errors, f"Int64 transfer diagnostic failed: {errors}"
