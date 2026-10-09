# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU numeric contracts for the complete current TheadBackend implementation.

The fixture deliberately isolates parent-package initialization: normal
vllm_fl imports initialize FlagGems, whose accelerator autotuner cannot start in
this no-device environment. It executes the original complete Backend,
TheadBackend and reference activation modules, and replaces no numeric code.
This suite does not test plugin startup, dispatch registration or device kernels.
Source revision and byte bindings belong to the external run manifest so future
implementation fixes do not require changing permanent test-source hash literals.
"""

import importlib
import importlib.machinery
import math
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def backend():
    original = {
        k: v
        for k, v in sys.modules.items()
        if k == "vllm_fl" or k.startswith("vllm_fl.")
    }
    assert not original, "Run this explicit isolated suite in its own pytest process"
    try:
        # Namespace shells specify only package paths; all tested modules/classes
        # execute from their actual source files, without AST or method extraction.
        for name in [
            "vllm_fl",
            "vllm_fl.dispatch",
            "vllm_fl.dispatch.backends",
            "vllm_fl.dispatch.backends.vendor",
            "vllm_fl.dispatch.backends.vendor.thead",
            "vllm_fl.dispatch.backends.reference",
            "vllm_fl.dispatch.backends.reference.impl",
        ]:
            package = ModuleType(name)
            package.__path__ = [str(ROOT / name.replace(".", "/"))]
            package.__package__ = name
            package.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            sys.modules[name] = package
        base = importlib.import_module("vllm_fl.dispatch.backends.base")
        module = importlib.import_module("vllm_fl.dispatch.backends.vendor.thead.thead")
        activation = importlib.import_module(
            "vllm_fl.dispatch.backends.reference.impl.activation"
        )
        assert (
            Path(module.__file__).resolve()
            == ROOT / "vllm_fl/dispatch/backends/vendor/thead/thead.py"
        )
        assert module.TheadBackend.__bases__ == (base.Backend,)
        for name in ["rms_norm", "rotary_embedding", "silu_and_mul"]:
            assert (
                Path(getattr(module.TheadBackend, name).__code__.co_filename).resolve()
                == Path(module.__file__).resolve()
            )
        assert (
            Path(activation.__file__).resolve()
            == ROOT / "vllm_fl/dispatch/backends/reference/impl/activation.py"
        )
        yield module.TheadBackend()
    finally:
        for name in list(sys.modules):
            if name == "vllm_fl" or name.startswith("vllm_fl."):
                sys.modules.pop(name, None)
        sys.modules.update(original)


def _tolerance(dtype):
    return {
        torch.float32: (3e-6, 3e-6),
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _rms_oracle(x, residual, weight, epsilon, variance_size, weighted):
    # Scalar arithmetic provides an independent mathematical oracle. Precision
    # boundaries are part of the operator's mixed-dtype public contract.
    values = x.reshape(-1, x.shape[-1]).double().tolist()
    additions = (
        None
        if residual is None
        else residual.reshape(-1, x.shape[-1]).double().tolist()
    )
    result, sums = [], []
    for row_number, row in enumerate(values):
        row = [_float32(value) for value in row]
        if additions is not None:
            row = [
                _float32(value + _float32(additions[row_number][i]))
                for i, value in enumerate(row)
            ]
        count = len(row) if variance_size is None else variance_size
        inverse = 1.0 / math.sqrt(
            sum(value * value for value in row[:count]) / count + epsilon
        )
        normalized = [value * inverse for value in row]
        if weighted:
            rounded = torch.tensor(normalized, dtype=weight.dtype).double().tolist()
            normalized = [value * float(weight[i]) for i, value in enumerate(rounded)]
            normalized = torch.tensor(normalized, dtype=weight.dtype).double().tolist()
        result.append(normalized)
        sums.append(row)
    output = torch.tensor(result, dtype=x.dtype).reshape(x.shape)
    updated = (
        None if residual is None else torch.tensor(sums, dtype=x.dtype).reshape(x.shape)
    )
    return output, updated


@pytest.mark.parametrize(
    "x_dtype,weight_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("residual_mode", ["none", "same", "fp32"])
@pytest.mark.parametrize(
    "pass_weight,pass_weight_add",
    [(True, True), (False, True), (True, False), (False, False)],
)
@pytest.mark.parametrize("variance_size", [None, 3])
def test_actual_backend_rms_mixed_dtype_residual_flags_variance(
    backend,
    x_dtype,
    weight_dtype,
    residual_mode,
    pass_weight,
    pass_weight_add,
    variance_size,
):
    x = (
        torch.linspace(-3.25, 4.75, 96, dtype=torch.float32)
        .reshape(2, 3, 16)[..., ::2]
        .to(x_dtype)
    )
    residual = (
        None
        if residual_mode == "none"
        else torch.linspace(0.17, -0.63, x.numel())
        .reshape(x.shape)
        .to(torch.float32 if residual_mode == "fp32" else x_dtype)
    )
    weight = torch.tensor(
        [0.3, -0.7, 1.2, 0.6, -1.4, 0.9, 1.7, -0.2], dtype=weight_dtype
    )
    obj = SimpleNamespace(
        weight=weight,
        variance_epsilon=0.07,
        variance_size_override=variance_size,
        pass_weight=pass_weight,
        pass_weight_add=pass_weight_add,
    )
    saved_x = x.clone()
    saved_residual = None if residual is None else residual.clone()
    expected, expected_residual = _rms_oracle(
        x,
        residual,
        weight,
        obj.variance_epsilon,
        variance_size,
        pass_weight if residual is None else pass_weight_add,
    )
    actual = backend.rms_norm(obj, x, residual)
    if residual is None:
        assert isinstance(actual, torch.Tensor)
        output = actual
    else:
        assert isinstance(actual, tuple) and len(actual) == 2
        output, updated = actual
        assert updated.dtype == x_dtype and updated.device.type == "cpu"
        torch.testing.assert_close(updated, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(residual, saved_residual, rtol=0, atol=0)
    assert (
        output.dtype == x_dtype
        and output.shape == x.shape
        and output.device.type == "cpu"
    )
    effective_dtype = (
        weight_dtype
        if (pass_weight if residual is None else pass_weight_add)
        and x_dtype == torch.float32
        else x_dtype
    )
    # A weighted fp32 output retains the weight dtype's quantization boundary.
    rtol, atol = _tolerance(effective_dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)


def _rotary_oracle(tensor, cos, sin, positions, interleaved):
    result = tensor.clone()
    half = tensor.shape[-1] // 2
    # Independent per-pair 2x2 rotations, rather than a vectorized copied formula.
    import itertools

    for index in itertools.product(*(range(size) for size in tensor.shape[:-1])):
        position = int(
            positions[index[0]]
            if tensor.ndim == 3
            else positions[index[2]]
            if positions.ndim == 1
            else positions[index[0], index[2]]
        )
        for pair in range(half):
            left, right = (
                (2 * pair, 2 * pair + 1) if interleaved else (pair, pair + half)
            )
            cosine = float(cos[position, pair].to(tensor.dtype))
            sine = float(sin[position, pair].to(tensor.dtype))
            a = float(tensor[index + (left,)])
            b = float(tensor[index + (right,)])
            result[index + (left,)] = a * cosine - b * sine
            result[index + (right,)] = b * cosine + a * sine
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["3d", "4d_shared", "4d_batched"])
@pytest.mark.parametrize("interleaved", [False, True], ids=["neox", "gptj"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("partial_tail", [False, True])
def test_actual_backend_rotary_layout_style_inplace_caller_view_tail(
    backend, dtype, layout, interleaved, inplace, partial_tail
):
    rotary_size = 8
    width = 12 if partial_tail else rotary_size
    qshape = (4, 3, width) if layout == "3d" else (2, 3, 4, width)
    kshape = (4, 2, width) if layout == "3d" else (2, 2, 4, width)
    query = (
        torch.linspace(-1.75, 2.25, math.prod(qshape) * 2)
        .reshape(qshape[:-1] + (width * 2,))[..., ::2]
        .to(dtype)
    )
    key = (
        torch.linspace(1.25, -2.5, math.prod(kshape) * 2)
        .reshape(kshape[:-1] + (width * 2,))[..., ::2]
        .to(dtype)
    )
    before_q, before_k = query.clone(), key.clone()
    angles = [
        [
            (position + 1) * (frequency + 1) * 0.17
            for frequency in range(rotary_size // 2)
        ]
        for position in range(7)
    ]
    cos = torch.tensor(
        [[math.cos(angle) for angle in row] for row in angles], dtype=torch.float64
    )
    sin = torch.tensor(
        [[math.sin(angle) for angle in row] for row in angles], dtype=torch.float64
    )
    positions = torch.tensor(
        [[0, 4, 1, 3], [6, 2, 5, 0]] if layout == "4d_batched" else [0, 4, 1, 3],
        dtype=torch.int64,
    )
    # The FL caller passes the rotary prefix view. The direct backend method must
    # rotate that view and leave the backing tensor's unpassed tail unchanged.
    qview, kview = query[..., :rotary_size], key[..., :rotary_size]
    expected_q = _rotary_oracle(qview, cos, sin, positions, interleaved)
    expected_k = _rotary_oracle(kview, cos, sin, positions, interleaved)
    actual_q, actual_k = backend.rotary_embedding(
        None,
        qview,
        kview,
        cos,
        sin,
        positions,
        rotary_interleaved=interleaved,
        inplace=inplace,
    )
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)
    assert (
        actual_q.dtype == actual_k.dtype == dtype
        and actual_q.device.type == actual_k.device.type == "cpu"
    )
    if inplace:
        assert actual_q is qview and actual_k is kview
    else:
        assert (
            actual_q.data_ptr() != qview.data_ptr()
            and actual_k.data_ptr() != kview.data_ptr()
        )
        torch.testing.assert_close(query, before_q, rtol=0, atol=0)
        torch.testing.assert_close(key, before_k, rtol=0, atol=0)
    if partial_tail:
        torch.testing.assert_close(
            query[..., rotary_size:], before_q[..., rotary_size:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            key[..., rotary_size:], before_k[..., rotary_size:], rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 16), (2, 3, 16)])
def test_actual_backend_silu_numeric_oracle(backend, dtype, shape):
    x = (
        torch.linspace(-4.25, 3.75, math.prod(shape) * 2)
        .reshape(shape[:-1] + (shape[-1] * 2,))[..., ::2]
        .to(dtype)
    )
    original = x.clone()
    half = shape[-1] // 2
    expected_rows = []
    for row in x.reshape(-1, shape[-1]).double().tolist():
        activated = [value / (1.0 + math.exp(-value)) for value in row[:half]]
        rounded = torch.tensor(activated, dtype=dtype).double().tolist()
        expected_rows.append(
            [value * row[half + index] for index, value in enumerate(rounded)]
        )
    expected = torch.tensor(expected_rows, dtype=dtype).reshape(shape[:-1] + (half,))
    actual = backend.silu_and_mul(None, x)
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert (
        actual.dtype == dtype
        and actual.shape == expected.shape
        and actual.device.type == "cpu"
    )
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_actual_backend_rms_defaults_apply_weight(backend, dtype, with_residual):
    x = torch.tensor(
        [[0.31, -1.27, 2.43, -0.57], [1.11, 0.43, -0.89, 1.91]], dtype=dtype
    )
    residual = (
        torch.tensor(
            [[0.17, -0.31, 0.11, 0.23], [-0.09, 0.37, -0.13, 0.29]], dtype=torch.float32
        )
        if with_residual
        else None
    )
    weight = torch.tensor([0.3, -0.7, 1.2, 0.6], dtype=torch.float32)
    # Both flag attributes and variance_size_override are genuinely absent.
    obj = SimpleNamespace(weight=weight, variance_epsilon=0.07)
    expected, expected_residual = _rms_oracle(
        x, residual, weight, obj.variance_epsilon, None, True
    )
    actual = backend.rms_norm(obj, x, residual)
    output = actual[0] if with_residual else actual
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    if with_residual:
        torch.testing.assert_close(actual[1], expected_residual, rtol=0, atol=0)
