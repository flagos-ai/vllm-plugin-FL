# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical regressions for the Ascend FlagGems operator policy.

Exercise the complete platform policy so interactions between enabled operators
are covered. References are computed on CPU before enabling FlagGems; transfers
and assertions happen after its registration is removed.
"""

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
flag_gems = pytest.importorskip("flag_gems")

from vllm_fl.utils import get_flag_gems_whitelist_blacklist

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


@pytest.fixture(autouse=True)
def _require_ascend():
    if not torch.npu.is_available() or flag_gems.vendor_name != "ascend":
        pytest.skip("Requires real Ascend NPU and Ascend FlagGems")


@contextmanager
def _strict_policy():
    from flag_gems.runtime import error as registration_errors

    def fail_registration(error):
        raise RuntimeError(f"FlagGems operator registration failed: {error}") from error

    _, blacklist = get_flag_gems_whitelist_blacklist()
    # The pinned registrar records names before lib.impl and otherwise logs
    # registration failures. Reject them so native kernels cannot pass these
    # tests while an intended FlagGems override was silently skipped.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registration_errors, "register_error", fail_registration)
        with flag_gems.use_gems(exclude=blacklist):
            yield


@contextmanager
def _enable(op, *, native=False):
    with _strict_policy():
        registered = flag_gems.all_registered_ops()
        if native:
            assert op not in registered, f"{op} must use the native implementation"
        else:
            assert op in registered, f"{op} was not registered"
        yield
        torch.npu.synchronize()


def _inputs(dtype, layout):
    values = torch.linspace(-1.75, 2.25, 17 * 66).reshape(17, 66).to(dtype)
    base = values.to("npu")
    if layout == "strided":
        return values[:, ::2], base[:, ::2]
    return values[:, :33].contiguous(), base[:, :33].contiguous()


def _preserve_layout(cpu, layout):
    if layout == "contiguous":
        cpu = cpu.contiguous()
        return cpu, cpu.to("npu")
    storage = torch.zeros((cpu.shape[0], cpu.shape[1] * 2), dtype=cpu.dtype)
    storage[:, ::2] = cpu
    npu_storage = storage.to("npu")
    return storage[:, ::2], npu_storage[:, ::2]


def _assert(actual, expected, dtype, *, reduction=False):
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(
        actual.cpu(),
        expected.to(dtype),
        rtol=tolerance,
        atol=tolerance if reduction else tolerance / 4,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "op", ["add", "sub", "mul", "true_divide", "rsqrt", "reciprocal", "silu"]
)
def test_flaggems_arithmetic_on_npu(op):
    for dtype in (torch.float32, torch.bfloat16):
        for layout in ("contiguous", "strided"):
            cpu, npu = _inputs(dtype, layout)
            # Broadcast over a non-power-of-two width as in packed model heads.
            rhs = torch.linspace(0.5, 1.5, cpu.shape[-1]).to(dtype)
            npu_rhs = rhs.to("npu")
            if op in ("rsqrt", "reciprocal"):
                cpu, npu = _preserve_layout(cpu.abs() + 0.25, layout)
                assert npu.is_contiguous() == (layout == "contiguous")
            unary = {
                "rsqrt": torch.rsqrt,
                "reciprocal": torch.reciprocal,
                "silu": F.silu,
            }
            binary = {
                "add": lambda x, y: torch.add(x, y, alpha=0.5),
                "sub": lambda x, y: torch.sub(x, y, alpha=0.5),
                "mul": torch.mul,
                "true_divide": torch.true_divide,
            }
            if op in unary:
                expected = unary[op](cpu.float())
                with _enable(op):
                    actual = unary[op](npu)
            else:
                expected = binary[op](cpu.float(), rhs.float())
                with _enable(op):
                    actual = binary[op](npu, npu_rhs)
            _assert(actual, expected, dtype)


def test_flaggems_acos_domain_and_strides_on_npu():
    for dtype in (torch.float32, torch.bfloat16):
        for layout in ("contiguous", "strided"):
            cpu, _ = _inputs(dtype, layout)
            cpu = (cpu / 3).to(dtype)
            cpu[0, :10] = torch.tensor(
                [
                    -1,
                    -0.999999,
                    0,
                    0.999999,
                    1,
                    -1.01,
                    1.01,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                ],
                dtype=dtype,
            )
            cpu, npu = _preserve_layout(cpu, layout)
            assert npu.is_contiguous() == (layout == "contiguous")
            expected = torch.acos(cpu.float()).to(dtype)
            with _enable("acos"):
                actual = torch.acos(npu)
            # The pinned FlagGems testing.assert_close uses atol=1e-4 for
            # float32 transcendental kernels, including test_acos.
            torch.testing.assert_close(
                actual.cpu(),
                expected,
                atol=1e-4 if dtype == torch.float32 else 2e-2,
                rtol=1.3e-6 if dtype == torch.float32 else 2e-2,
                equal_nan=True,
            )


@pytest.mark.parametrize("op", ["index_select", "repeat"])
def test_flaggems_indexing_and_layout_on_npu(op):
    for dtype in (torch.float32, torch.bfloat16, torch.int64):
        for layout in ("contiguous", "strided"):
            cpu, npu = _inputs(dtype, layout)
            for dim in (0, 1):
                index = torch.tensor([cpu.shape[dim] - 1, 0, 3, 3], dtype=torch.int64)
                npu_index = index.to("npu")
                functions = {
                    "index_select": lambda x, i, dim=dim: torch.index_select(x, dim, i),
                    "repeat": lambda x, i: x.repeat(2, 3),
                }
                expected = functions[op](cpu, index)
                with _enable(op):
                    actual = functions[op](npu, npu_index)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    if op == "index_select":
        cpu, npu = _inputs(torch.float32, "strided")
        index = torch.empty((0,), dtype=torch.int64)
        npu_index = index.to("npu")
        expected = getattr(torch, op)(cpu, 1, index)
        with _enable(op):
            actual = getattr(torch, op)(npu, 1, npu_index)
        torch.testing.assert_close(actual.cpu(), expected)


def test_flaggems_reduction_max_on_npu():
    for dtype in (torch.float32, torch.bfloat16):
        for layout in ("contiguous", "strided"):
            cpu, npu = _inputs(dtype, layout)
            expected = torch.max(cpu.float())
            with _enable("max"):
                actual = torch.max(npu)
            _assert(actual, expected, dtype, reduction=True)


def test_default_ascend_empty_gather_uses_native():
    # The pinned FlagGems kernel launches with coreDim=0 for an empty index.
    cpu, npu = _inputs(torch.float32, "strided")
    index = torch.empty((17, 0), dtype=torch.int64)
    npu_index = index.to("npu")
    expected = torch.gather(cpu, 1, index)
    with _enable("gather", native=True):
        actual = torch.gather(npu, 1, npu_index)
    torch.testing.assert_close(actual.cpu(), expected)


def test_default_ascend_integer_sum_uses_native():
    # Native sum promotes int32 to int64; the pinned FlagGems sum_dim does not.
    for shape in ((3, 0), (3, 7)):
        cpu = torch.arange(shape[0] * shape[1]).reshape(shape).to(torch.int32)
        npu = cpu.to("npu")
        expected = torch.sum(cpu, dim=1)
        assert expected.dtype == torch.int64
        with _enable("sum_dim", native=True):
            actual = torch.sum(npu, dim=1)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "op", ["eq_scalar", "ge_scalar", "floor_divide", "bitwise_not"]
)
def test_default_ascend_integer_and_boolean_on_npu(op):
    for dtype in (torch.int32, torch.int64):
        cpu_base = (torch.arange(17 * 66).reshape(17, 66) % 71 - 35).to(dtype)
        cpu = cpu_base[:, ::2]
        npu = cpu_base.to("npu")[:, ::2]
        functions = {
            "eq_scalar": lambda x: torch.eq(x, -3),
            "ge_scalar": lambda x: torch.ge(x, 0),
            "floor_divide": lambda x: torch.floor_divide(x, 3),
            "bitwise_not": torch.bitwise_not,
        }
        expected = functions[op](cpu)
        with _enable(op, native=op == "bitwise_not"):
            actual = functions[op](npu)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    if op == "bitwise_not":
        cpu = torch.tensor([[True, False], [False, True]])
        npu = cpu.to("npu")
        expected = torch.bitwise_not(cpu)
        with _enable(op, native=True):
            actual = torch.bitwise_not(npu)
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("length", [1, 2, 17, 33])
@pytest.mark.parametrize("op", ["bitwise_not", "bitwise_not_"])
def test_default_ascend_boolean_not_preserves_strided_values(length, op):
    # The pinned kernel reads [True, False] from this view as [True, True].
    cpu_storage = torch.arange(2 * length + 1) % 3 == 1
    for strided in (False, True):
        storage = cpu_storage.to("npu")
        expected_storage = cpu_storage.clone()
        cpu = cpu_storage[1::2]
        npu = storage[1::2]
        if not strided:
            cpu, npu = cpu.contiguous().clone(), npu.contiguous().clone()
        expected = torch.bitwise_not(cpu)
        if op == "bitwise_not_" and strided:
            expected_storage[1::2] = expected
        expected_state = torch.arange(length * 3).reshape(length, 3).float()
        state = expected_state.to("npu")
        expected_state[expected] = 0
        with _enable(op, native=True):
            actual = (
                npu.bitwise_not_() if op == "bitwise_not_" else torch.bitwise_not(npu)
            )
            # Exercise its consumer as in GDN's initial state reset.
            indices = torch.nonzero(actual)
            state[actual] = 0
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(indices.cpu(), torch.nonzero(expected))
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
        torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)


def test_flaggems_creation_zeros_on_npu():
    for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool):
        for shape in ((17, 33), (0, 33)):
            expected = torch.zeros(shape, dtype=dtype)
            with _enable("zeros"):
                actual = torch.zeros(shape, device="npu", dtype=dtype)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_flaggems_mm_on_npu():
    generator = torch.Generator().manual_seed(17)
    for dtype in (torch.float32, torch.bfloat16):
        for transposed in (False, True):
            a = torch.randn(
                (65, 17) if transposed else (17, 65), generator=generator
            ).to(dtype)
            b = (torch.randn(65, 33, generator=generator) / 8).to(dtype)
            npu_a, npu_b = a.to("npu"), b.to("npu")
            if transposed:
                a, npu_a = a.T, npu_a.T
            expected = torch.mm(a.float(), b.float())
            with _enable("mm"):
                actual = torch.mm(npu_a, npu_b)
            _assert(actual, expected, dtype, reduction=True)


@pytest.mark.parametrize("op", ["conv1d", "conv2d"])
def test_default_ascend_convolution_on_npu(op, monkeypatch):
    # torch_npu enables HF32 convolution by default. Validate the FP32 CPU
    # reference in full precision without relaxing its original tolerance.
    monkeypatch.setattr(torch.npu.conv, "allow_hf32", False)
    generator = torch.Generator().manual_seed(19)
    for dtype in (torch.float32, torch.bfloat16):
        for groups in (1, 2):
            shape = (2, 2, 17) if op == "conv1d" else (2, 2, 9, 11)
            kernel_shape = (
                (4, 2 // groups, 3) if op == "conv1d" else (4, 2 // groups, 3, 3)
            )
            cpu = torch.randn(shape, generator=generator).to(dtype)
            weight = (torch.randn(kernel_shape, generator=generator) / 8).to(dtype)
            bias = torch.randn((4,), generator=generator).to(dtype)
            npu, npu_weight, npu_bias = cpu.to("npu"), weight.to("npu"), bias.to("npu")
            expected = getattr(F, op)(
                cpu.float(), weight.float(), bias.float(), padding=1, groups=groups
            )
            with _enable(op, native=op == "conv1d"):
                actual = getattr(F, op)(
                    npu, npu_weight, npu_bias, padding=1, groups=groups
                )
            _assert(actual, expected, dtype, reduction=True)


@pytest.mark.parametrize("op", ["conv1d", "conv2d"])
def test_default_ascend_gdn_depthwise_convolution_on_npu(op):
    # Cover the depthwise shape observed at the model-process failure site.
    # Standalone kernel replays do not reproduce that process-level fault.
    generator = torch.Generator().manual_seed(101)
    channels, tokens, width = 5120, 15, 4
    cpu = (torch.randn(1, channels, tokens, generator=generator) / 4).to(torch.bfloat16)
    weight = (torch.randn(channels, 1, width, generator=generator) / 4).to(
        torch.bfloat16
    )
    expected = F.conv1d(cpu.float(), weight.float(), padding=3, groups=channels)
    if op == "conv2d":
        cpu, weight = cpu.unsqueeze(-1), weight.unsqueeze(-1)
        expected = expected.unsqueeze(-1)
    npu, npu_weight = cpu.to("npu"), weight.to("npu")
    with _enable(op, native=op == "conv1d"):
        actual = getattr(F, op)(
            npu,
            npu_weight,
            padding=3 if op == "conv1d" else (3, 0),
            groups=channels,
        )
    _assert(actual, expected, torch.bfloat16, reduction=True)
    torch.testing.assert_close(npu.cpu(), cpu, rtol=0, atol=0)
    torch.testing.assert_close(npu_weight.cpu(), weight, rtol=0, atol=0)


@pytest.mark.parametrize("has_initial", [False, True])
def test_default_ascend_gdn_prefill_output_and_state_on_npu(has_initial):
    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import (
        causal_conv1d_fn,
    )

    generator = torch.Generator().manual_seed(103)
    channels, tokens, width = 5120, 15, 4
    cpu = (torch.randn(tokens, channels, generator=generator) / 4).to(torch.bfloat16)
    weight = (torch.randn(channels, width, generator=generator) / 4).to(torch.bfloat16)
    storage = (torch.randn(3, width - 1, channels, generator=generator) / 4).to(
        torch.bfloat16
    )
    expected_storage = storage.clone()
    sequence = cpu.T.float()
    if has_initial:
        sequence = torch.cat((storage[1].T.float(), sequence), dim=-1)
    conv_reference = F.conv1d(
        sequence.unsqueeze(0),
        weight.float().unsqueeze(1),
        padding=0 if has_initial else width - 1,
        groups=channels,
    )[..., :tokens].to(cpu.dtype)
    # The production convolution rounds to BF16 before SiLU, so use the
    # same staging rather than silently comparing a fused FP32 reference.
    expected = F.silu(conv_reference.float()).to(cpu.dtype).squeeze(0).T
    expected_storage[1] = sequence[:, -(width - 1) :].T
    npu, npu_weight, npu_storage = (
        cpu.to("npu"),
        weight.to("npu"),
        storage.to("npu"),
    )
    query_start = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    cache_indices = torch.tensor([1], dtype=torch.int32, device="npu")
    initial_mask = torch.tensor([has_initial], device="npu")
    with _enable("conv1d", native=True):
        assert "conv2d" in flag_gems.all_registered_ops()
        actual = causal_conv1d_fn(
            npu.T,
            npu_weight,
            None,
            npu_storage.transpose(1, 2),
            query_start,
            cache_indices=cache_indices,
            has_initial_state=initial_mask,
        ).T
    assert actual.stride(-1) == 1
    _assert(actual, expected, torch.bfloat16, reduction=True)
    torch.testing.assert_close(npu_storage.cpu(), expected_storage, rtol=0, atol=0)
    torch.testing.assert_close(npu.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["masked_scatter", "slice_scatter"])
def test_flaggems_scatter_on_npu(op):
    for dtype in (torch.float32, torch.bfloat16):
        cpu, npu = _inputs(dtype, "strided")
        mask = torch.arange(cpu.numel()).reshape(cpu.shape) % 3 == 0
        npu_mask = mask.to("npu")
        source = torch.linspace(1, 2, cpu.numel()).to(dtype)
        npu_source = source.to("npu")
        slice_source = source[: 17 * 16].reshape(17, 16)
        npu_slice_source = slice_source.to("npu")
        if op == "masked_scatter":
            expected = cpu.masked_scatter(mask, source)
            with _enable(op):
                actual = npu.masked_scatter(npu_mask, npu_source)
        else:
            expected = torch.slice_scatter(
                cpu, slice_source, dim=1, start=1, end=33, step=2
            )
            with _enable(op):
                actual = torch.slice_scatter(
                    npu, npu_slice_source, dim=1, start=1, end=33, step=2
                )
        _assert(actual, expected, dtype)
        torch.testing.assert_close(npu.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["isclose", "allclose"])
def test_flaggems_close_nan_semantics_on_npu(op):
    cpu = torch.tensor([0.0, 1.0, -2.0, float("nan"), float("inf"), -float("inf")])
    other = torch.tensor(
        [1e-5, 1.0001, -2.0, float("nan"), float("inf"), -float("inf")]
    )
    npu, npu_other = cpu.to("npu"), other.to("npu")
    for equal_nan in (False, True):
        expected = getattr(torch, op)(
            cpu, other, atol=2e-4, rtol=1e-4, equal_nan=equal_nan
        )
        with _enable(op):
            actual = getattr(torch, op)(
                npu, npu_other, atol=2e-4, rtol=1e-4, equal_nan=equal_nan
            )
        if op == "allclose":
            assert actual == expected
        else:
            torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("op", ["randn", "uniform_"])
def test_default_ascend_native_random_distribution_on_npu(op):
    # Broad confidence intervals avoid seed-dependent CI flakes while catching
    # constant/unwritten output or an incorrect distribution.
    npu = torch.empty((257, 257), device="npu", dtype=torch.float32)
    with _enable(op, native=True):
        actual = (
            npu.uniform_(-2, 4)
            if op == "uniform_"
            else torch.randn(npu.shape, device="npu")
        )
    values = actual.cpu()
    assert torch.isfinite(values).all()
    if op == "randn":
        assert abs(values.mean().item()) < 0.025
        assert 0.97 < values.std().item() < 1.03
    else:
        assert (values >= -2).all() and (values < 4).all()
        normalized = (values + 2) / 6
        assert 0.49 < normalized.mean().item() < 0.51
        assert 0.28 < normalized.std().item() < 0.30


def test_default_ascend_strided_uniform_preserves_storage():
    # The pinned FlagGems kernel writes raw contiguous offsets for this view.
    cpu = torch.full((17, 66), -7.0)
    storage = cpu.to("npu")
    view = storage[:, ::2]
    assert not view.is_contiguous()
    with _enable("uniform_", native=True):
        actual = view.uniform_(-2, 4)
    assert actual is view
    result = storage.cpu()
    torch.testing.assert_close(result[:, 1::2], cpu[:, 1::2], rtol=0, atol=0)
    assert (result[:, ::2] >= -2).all() and (result[:, ::2] < 4).all()


def test_default_ascend_empty_randn_uses_native():
    # The pinned FlagGems kernel launches with coreDim=0 for an empty shape.
    with _enable("randn", native=True):
        # Native randn decomposes into normal_, which also needs exclusion.
        assert "normal_" not in flag_gems.all_registered_ops()
        actual = torch.randn((0, 33), dtype=torch.float32, device="npu")
    torch.testing.assert_close(actual.cpu(), torch.empty((0, 33)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_silu_and_mul_flaggems_fallback_on_npu(dtype, layout):
    from vllm_fl.dispatch.backends.flaggems.impl.activation import (
        silu_and_mul_flaggems,
    )

    generator = torch.Generator().manual_seed(71)
    storage = torch.randn(17, 2048, generator=generator).to(dtype)
    cpu, npu = _preserve_layout(storage, layout)
    left, right = cpu.float().chunk(2, dim=-1)
    expected = F.silu(left) * right
    _, blacklist = get_flag_gems_whitelist_blacklist()
    assert "silu_and_mul" not in blacklist
    # This direct FL fallback is not an ATen registration. Exercise its actual
    # work tensors under the same policy as the model worker.
    with _strict_policy():
        actual = silu_and_mul_flaggems(None, npu)
        torch.npu.synchronize()
    _assert(actual, expected, dtype)
