#!/usr/bin/env python3
"""Validate NVIDIA FlagGems blacklist entries against native PyTorch.

This is a diagnostic tool, not a unit test. It reads the blacklist from the
selected NVIDIA YAML file, enables one FlagGems implementation at a time in an
isolated subprocess, and compares it with the native PyTorch result.

A PASS means only that the samples in this script work in the tested runtime.
It is a candidate for model-level revalidation, not proof that the blacklist
entry is safe to remove for every shape, dtype, stride, or overload.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "vllm_fl" / "dispatch" / "config" / "nvidia.yaml"
RESULT_PREFIX = "VLLM_FL_BLACKLIST_RESULT="


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[bool], Callable[[], Any]]
    rtol: float = 1e-2
    atol: float = 1e-2
    registration: str = "aten"


@dataclass
class CaseResult:
    name: str
    status: str
    duration_s: float
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None


def _torch():
    import torch

    return torch


def _randn(shape: Sequence[int], dtype):
    torch = _torch()
    return torch.randn(tuple(shape), device="cuda", dtype=dtype)


def _positive(shape: Sequence[int], dtype):
    return _randn(shape, dtype).abs().add_(0.25)


def _case(
    name: str,
    factory: Callable[[], Callable[[], Any]],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Case:
    return Case(name, lambda _use_flaggems: factory(), rtol=rtol, atol=atol)


def _unary_cases(
    function: Callable[[Any], Any],
    *,
    positive: bool = False,
) -> list[Case]:
    torch = _torch()

    def build(shape, dtype):
        def factory():
            x = (_positive if positive else _randn)(shape, dtype)
            return lambda: function(x)

        return factory

    return [
        _case("fp32_small", build((17, 31), torch.float32)),
        _case("bf16_model_like", build((4, 128, 256), torch.bfloat16)),
    ]


def _binary_cases(function: Callable[[Any, Any], Any]) -> list[Case]:
    torch = _torch()

    def build(shape, dtype):
        def factory():
            x = _randn(shape, dtype)
            y = _randn(shape, dtype)
            return lambda: function(x, y)

        return factory

    return [
        _case("fp32_small", build((17, 31), torch.float32)),
        _case("bf16_model_like", build((4, 128, 256), torch.bfloat16)),
    ]


def _comparison_cases(function: Callable[[Any, Any], Any]) -> list[Case]:
    return _binary_cases(function)


def _scalar_cases(function: Callable[[Any], Any]) -> list[Case]:
    torch = _torch()

    def build(shape, dtype):
        def factory():
            x = _randn(shape, dtype)
            return lambda: function(x)

        return factory

    return [
        _case("fp32_small", build((17, 31), torch.float32)),
        _case("bf16_model_like", build((4, 128, 256), torch.bfloat16)),
    ]


def _inplace_binary_cases(function: Callable[[Any, Any], Any]) -> list[Case]:
    torch = _torch()

    def build(shape, dtype):
        def factory():
            x = _randn(shape, dtype)
            y = _positive(shape, dtype)

            def invoke():
                function(x, y)
                return x

            return invoke

        return factory

    return [
        _case("fp32_small", build((17, 31), torch.float32)),
        _case("bf16_model_like", build((4, 128, 256), torch.bfloat16)),
    ]


def _index_put_cases(kind: str) -> list[Case]:
    torch = _torch()

    def build(dtype, accumulate):
        def factory():
            x = _randn((32, 64), dtype)
            indices = [torch.tensor([0, 3, 7, 11, 15, 19, 23, 31], device="cuda")]
            values = _randn((8, 64), dtype)

            def invoke():
                if kind == "index_put":
                    return torch.index_put(x, indices, values, accumulate)
                if kind == "index_put_":
                    torch.index_put_(x, indices, values, accumulate)
                    return x
                torch._index_put_impl_(
                    x,
                    indices,
                    values,
                    accumulate=accumulate,
                    unsafe=False,
                )
                return x

            return invoke

        return factory

    return [
        _case("fp32_replace", build(torch.float32, False)),
        _case("bf16_accumulate", build(torch.bfloat16, True), rtol=2e-2, atol=2e-2),
    ]


def _nonzero_cases() -> list[Case]:
    torch = _torch()

    def float_factory():
        x = torch.arange(-128, 128, device="cuda", dtype=torch.float32).reshape(16, 16)
        x[::3, ::5] = 0
        return lambda: torch.nonzero(x)

    def bool_factory():
        x = torch.arange(0, 4096, device="cuda").reshape(64, 64) % 7 == 0
        return lambda: torch.nonzero(x)

    return [_case("fp32_2d", float_factory), _case("bool_2d", bool_factory)]


def _copy_cases() -> list[Case]:
    torch = _torch()

    def same_dtype():
        source = _randn((4, 128, 256), torch.bfloat16)
        destination = torch.zeros_like(source)

        def invoke():
            destination.copy_(source)
            return destination

        return invoke

    def broadcast():
        source = torch.arange(31, device="cuda", dtype=torch.float32)
        destination = torch.zeros((17, 31), device="cuda")

        def invoke():
            destination.copy_(source)
            return destination

        return invoke

    return [_case("bf16_same_dtype", same_dtype), _case("fp32_broadcast", broadcast)]


def _to_copy_cases() -> list[Case]:
    torch = _torch()

    def build(source_dtype, destination_dtype):
        def factory():
            x = _randn((4, 128, 256), source_dtype)
            return lambda: torch.ops.aten._to_copy.default(x, dtype=destination_dtype)

        return factory

    return [
        _case("fp32_to_bf16", build(torch.float32, torch.bfloat16)),
        _case("bf16_to_fp32", build(torch.bfloat16, torch.float32)),
    ]


def _index_cases() -> list[Case]:
    torch = _torch()

    def integer_indices():
        x = _randn((32, 64), torch.bfloat16)
        row = torch.tensor([0, 3, 7, 11, 31], device="cuda")
        return lambda: torch.ops.aten.index.Tensor(x, [row])

    def boolean_mask():
        x = _randn((32, 64), torch.float32)
        mask = torch.arange(32, device="cuda") % 3 == 0
        return lambda: torch.ops.aten.index.Tensor(x, [mask])

    return [_case("bf16_integer", integer_indices), _case("fp32_boolean", boolean_mask)]


def _fill_cases(tensor_value: bool) -> list[Case]:
    torch = _torch()

    def contiguous():
        x = _randn((4, 128, 256), torch.bfloat16)
        value = torch.tensor(2, device="cuda", dtype=x.dtype) if tensor_value else 2

        def invoke():
            x.fill_(value)
            return x

        return invoke

    def sliced_view():
        base = _randn((4, 128), torch.float32)
        x = base[:, 64:]
        value = torch.tensor(-1, device="cuda", dtype=x.dtype) if tensor_value else -1

        def invoke():
            x.fill_(value)
            return base

        return invoke

    return [
        _case("bf16_contiguous", contiguous),
        _case("fp32_sliced_view", sliced_view),
    ]


def _masked_fill_cases() -> list[Case]:
    torch = _torch()

    def scalar_value():
        x = _randn((4, 128, 256), torch.bfloat16)
        mask = torch.arange(x.numel(), device="cuda").reshape(x.shape) % 7 == 0

        def invoke():
            x.masked_fill_(mask, float("-inf"))
            return x

        return invoke

    def tensor_value():
        x = _randn((17, 31), torch.float32)
        mask = x > 0
        value = torch.tensor(-2.5, device="cuda")

        def invoke():
            x.masked_fill_(mask, value)
            return x

        return invoke

    return [_case("bf16_scalar", scalar_value), _case("fp32_tensor", tensor_value)]


def _gelu_cases() -> list[Case]:
    torch = _torch()
    return _unary_cases(lambda x: torch.nn.functional.gelu(x, approximate="none"))


def _clamp_cases(inplace: bool) -> list[Case]:
    torch = _torch()

    def build(shape, dtype):
        def factory():
            x = _randn(shape, dtype)

            def invoke():
                if inplace:
                    x.clamp_(min=-0.75, max=0.5)
                    return x
                return torch.clamp(x, min=-0.75, max=0.5)

            return invoke

        return factory

    return [
        _case("fp32_small", build((17, 31), torch.float32)),
        _case("bf16_model_like", build((4, 128, 256), torch.bfloat16)),
    ]


def _pow_scalar_cases() -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            exponent = _randn((4, 128, 256), dtype).clamp_(-2, 2)
            return lambda: torch.pow(2.0, exponent)

        return factory

    return [
        _case("fp32_exponent", build(torch.float32)),
        _case("bf16_exponent", build(torch.bfloat16), rtol=2e-2, atol=2e-2),
    ]


def _pow_tensor_scalar_cases() -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            x = _positive((4, 128, 256), dtype)
            return lambda: torch.pow(x, 1.5)

        return factory

    return [
        _case("fp32_base", build(torch.float32)),
        _case("bf16_base", build(torch.bfloat16), rtol=2e-2, atol=2e-2),
    ]


def _floor_divide_cases() -> list[Case]:
    torch = _torch()

    def integer():
        x = torch.arange(-256, 256, device="cuda", dtype=torch.int64).reshape(16, 32)
        y = torch.arange(1, 33, device="cuda", dtype=torch.int64).repeat(16, 1)
        return lambda: torch.floor_divide(x, y)

    def floating():
        x = _randn((4, 128, 256), torch.bfloat16) * 8
        y = _positive((4, 128, 256), torch.bfloat16)
        return lambda: torch.floor_divide(x, y)

    return [_case("int64", integer), _case("bf16", floating)]


def _where_cases(out_variant: bool) -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            x = _randn((4, 128, 256), dtype)
            y = _randn((4, 128, 256), dtype)
            condition = x > 0
            if not out_variant:
                return lambda: torch.where(condition, x, y)
            out = torch.empty_like(x)
            return lambda: torch.where(condition, x, y, out=out)

        return factory

    return [
        _case("fp32", build(torch.float32)),
        _case("bf16", build(torch.bfloat16)),
    ]


def _bitwise_unary_cases() -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            x = torch.arange(0, 4096, device="cuda", dtype=dtype).reshape(64, 64)
            return lambda: torch.bitwise_not(x)

        return factory

    return [_case("int32", build(torch.int32)), _case("int64", build(torch.int64))]


def _bitwise_binary_cases(function) -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            x = torch.arange(0, 4096, device="cuda", dtype=dtype).reshape(64, 64)
            y = torch.arange(4096, 8192, device="cuda", dtype=dtype).reshape(64, 64)
            return lambda: function(x, y)

        return factory

    return [_case("int32", build(torch.int32)), _case("int64", build(torch.int64))]


def _sum_cases() -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            x = _randn((4, 128, 256), dtype)
            return lambda: torch.sum(x)

        return factory

    return [
        _case("fp32_all", build(torch.float32), rtol=2e-3, atol=2e-3),
        _case("bf16_all", build(torch.bfloat16), rtol=5e-2, atol=5e-2),
    ]


def _repeat_interleave_cases(tensor_repeats: bool) -> list[Case]:
    torch = _torch()

    def contiguous():
        x = _randn((20, 320, 15), torch.bfloat16)
        repeats = (
            torch.arange(0, 320, device="cuda", dtype=torch.int64) % 4
            if tensor_repeats
            else 2
        )
        return lambda: torch.repeat_interleave(x, repeats, dim=1)

    def noncontiguous():
        x = _randn((32, 64, 16), torch.float32)[::2]
        repeats = (
            torch.arange(0, x.shape[0], device="cuda", dtype=torch.int64) % 3
            if tensor_repeats
            else 2
        )
        return lambda: torch.repeat_interleave(x, repeats, dim=0)

    return [
        _case("bf16_contiguous", contiguous),
        _case("fp32_noncontiguous", noncontiguous),
    ]


def _silu_and_mul_cases() -> list[Case]:
    torch = _torch()

    def build(dtype):
        def builder(use_flaggems: bool):
            x = _randn((4, 128, 512), dtype)
            gate, up = x.chunk(2, dim=-1)
            if use_flaggems:
                from flag_gems.modules.activation import gems_silu_and_mul

                return lambda: gems_silu_and_mul(gate, up)
            return lambda: torch.nn.functional.silu(gate) * up

        return builder

    return [
        Case("fp32", build(torch.float32), registration="custom"),
        Case(
            "bf16_model_like",
            build(torch.bfloat16),
            rtol=2e-2,
            atol=2e-2,
            registration="custom",
        ),
    ]


def _pad_cases(aten_pad: bool) -> list[Case]:
    torch = _torch()

    def constant():
        x = _randn((4, 64, 128), torch.bfloat16)
        if aten_pad:
            return lambda: torch.ops.aten.pad.default(x, [3, 5, 2, 4], "constant", 0.25)
        return lambda: torch.ops.aten.constant_pad_nd.default(x, [3, 5, 2, 4], 0.25)

    def noncontiguous():
        x = _randn((2, 32, 64, 64), torch.float32)[:, :, ::2, ::2]
        if aten_pad:
            return lambda: torch.ops.aten.pad.default(x, [1, 2, 3, 4], "constant", -1.0)
        return lambda: torch.ops.aten.constant_pad_nd.default(x, [1, 2, 3, 4], -1.0)

    return [
        _case("bf16_constant", constant),
        _case("fp32_noncontiguous", noncontiguous),
    ]


def _bmm_out_cases() -> list[Case]:
    torch = _torch()

    def build(dtype, m, n, k):
        def factory():
            left = _randn((4, m, k), dtype)
            right = _randn((4, k, n), dtype)
            out = torch.empty((4, m, n), device="cuda", dtype=dtype)
            return lambda: torch.ops.aten.bmm.out(left, right, out=out)

        return factory

    return [
        _case("fp32_small", build(torch.float32, 15, 31, 64), rtol=2e-3, atol=2e-3),
        _case(
            "bf16_model_like", build(torch.bfloat16, 32, 64, 128), rtol=5e-2, atol=5e-2
        ),
    ]


def _topk_cases() -> list[Case]:
    torch = _torch()

    def build(dtype, largest):
        def factory():
            x = _randn((32, 256), dtype)
            return lambda: torch.topk(x, 8, dim=-1, largest=largest, sorted=True)

        return factory

    return [
        _case("fp32_largest", build(torch.float32, True)),
        _case("fp32_smallest", build(torch.float32, False)),
        _case("bf16_largest", build(torch.bfloat16, True)),
        _case("bf16_smallest", build(torch.bfloat16, False)),
    ]


def _full_cases() -> list[Case]:
    torch = _torch()

    def build(dtype, value):
        return lambda: torch.full((4, 128, 256), value, device="cuda", dtype=dtype)

    return [
        _case("fp32", lambda: build(torch.float32, 1.25)),
        _case("bf16", lambda: build(torch.bfloat16, -2.5)),
    ]


def _gather_cases() -> list[Case]:
    torch = _torch()

    def build(dtype):
        def factory():
            x = _randn((32, 256), dtype)
            index = torch.arange(0, 128, device="cuda").repeat(32, 1) * 2
            return lambda: torch.gather(x, 1, index)

        return factory

    return [_case("fp32", build(torch.float32)), _case("bf16", build(torch.bfloat16))]


def _argmax_cases() -> list[Case]:
    torch = _torch()

    def build(dtype, dim):
        def factory():
            x = _randn((32, 256), dtype)
            return lambda: torch.argmax(x, dim=dim)

        return factory

    return [
        _case("fp32_all", build(torch.float32, None)),
        _case("bf16_last_dim", build(torch.bfloat16, -1)),
    ]


def _fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids):
    torch = _torch()
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros_like(hidden_states)
    for token in range(num_tokens):
        for slot in range(topk):
            expert = topk_ids[token, slot].item()
            projected = (
                hidden_states[token].float() @ w1[expert].transpose(0, 1).float()
            )
            gate, up = projected.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            down = activated @ w2[expert].transpose(0, 1).float()
            output[token] += (topk_weights[token, slot].float() * down).to(output.dtype)
    assert output.shape == (num_tokens, hidden_size)
    return output


def _fused_moe_cases() -> list[Case]:
    torch = _torch()

    def build(config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config

        def builder(use_flaggems: bool):
            hidden_states = _randn((num_tokens, hidden_size), dtype)
            w1 = _randn((num_experts, intermediate_size * 2, hidden_size), dtype)
            w1.mul_(hidden_size**-0.5)
            w2 = _randn((num_experts, hidden_size, intermediate_size), dtype)
            w2.mul_(intermediate_size**-0.5)
            gating = _randn((num_tokens, num_experts), torch.float32)
            topk_weights, topk_ids = torch.topk(
                torch.softmax(gating, dim=-1), topk, dim=-1
            )
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(dtype)
            if use_flaggems:
                import flag_gems

                return lambda: flag_gems.fused_experts_impl(
                    hidden_states, w1, w2, topk_weights, topk_ids
                )
            return lambda: _fused_moe_reference(
                hidden_states, w1, w2, topk_weights, topk_ids
            )

        return builder

    return [
        Case(
            "bf16_small",
            build((4, 8, 128, 256, 2), torch.bfloat16),
            rtol=1e-1,
            atol=5e-2,
            registration="custom",
        )
    ]


def build_cases() -> dict[str, list[Case]]:
    torch = _torch()
    cases = {
        "index_put_": _index_put_cases("index_put_"),
        "index_put": _index_put_cases("index_put"),
        "_index_put_impl_": _index_put_cases("_index_put_impl_"),
        "nonzero": _nonzero_cases(),
        "copy_": _copy_cases(),
        "to_copy": _to_copy_cases(),
        "index": _index_cases(),
        "fill_scalar_": _fill_cases(False),
        "fill_tensor_": _fill_cases(True),
        "sub": _binary_cases(torch.sub),
        "rsub_scalar": _scalar_cases(lambda x: torch.rsub(x, 1.25)),
        "masked_fill_": _masked_fill_cases(),
        "le": _comparison_cases(torch.le),
        "le_scalar": _scalar_cases(lambda x: torch.le(x, 0.25)),
        "gelu": _gelu_cases(),
        "clamp": _clamp_cases(False),
        "pow_scalar": _pow_scalar_cases(),
        "pow_tensor_scalar": _pow_tensor_scalar_cases(),
        "cos": _unary_cases(torch.cos),
        "sin": _unary_cases(torch.sin),
        "floor_divide": _floor_divide_cases(),
        "lt": _comparison_cases(torch.lt),
        "lt_scalar": _scalar_cases(lambda x: torch.lt(x, 0.25)),
        "gt_scalar": _scalar_cases(lambda x: torch.gt(x, 0.25)),
        "rsqrt": _unary_cases(torch.rsqrt, positive=True),
        "sigmoid": _unary_cases(torch.sigmoid),
        "where_self": _where_cases(False),
        "where_self_out": _where_cases(True),
        "true_divide_": _inplace_binary_cases(torch.Tensor.div_),
        "bitwise_not": _bitwise_unary_cases(),
        "clamp_": _clamp_cases(True),
        "sqrt": _unary_cases(torch.sqrt, positive=True),
        "sum": _sum_cases(),
        "bitwise_or_tensor": _bitwise_binary_cases(torch.bitwise_or),
        "mul_": _inplace_binary_cases(torch.Tensor.mul_),
        "ge_scalar": _scalar_cases(lambda x: torch.ge(x, 0.25)),
        "eq_scalar": _scalar_cases(lambda x: torch.eq(x, 0.25)),
        "repeat_interleave_self_tensor": _repeat_interleave_cases(True),
        "bitwise_and_tensor": _bitwise_binary_cases(torch.bitwise_and),
        "reciprocal": _unary_cases(torch.reciprocal, positive=True),
        "mul": _binary_cases(torch.mul),
        "true_divide": _binary_cases(
            lambda x, y: torch.true_divide(x, y.abs().add_(0.25))
        ),
        "repeat_interleave_self_int": _repeat_interleave_cases(False),
        "neg": _unary_cases(torch.neg),
        "add": _binary_cases(torch.add),
        "silu_and_mul": _silu_and_mul_cases(),
        "constant_pad_nd": _pad_cases(False),
        "pad": _pad_cases(True),
        "bmm_out": _bmm_out_cases(),
        "topk": _topk_cases(),
        "full": _full_cases(),
        "gather": _gather_cases(),
        "argmax": _argmax_cases(),
        "fused_moe": _fused_moe_cases(),
    }
    return cases


def _assert_close(actual: Any, expected: Any, *, rtol: float, atol: float) -> None:
    torch = _torch()
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol if expected.is_floating_point() else 0,
            atol=atol if expected.is_floating_point() else 0,
            equal_nan=True,
            check_dtype=True,
            check_device=True,
        )
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError(
                f"container mismatch: actual={type(actual).__name__}, "
                f"expected={type(expected).__name__}"
            )
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_close(actual_item, expected_item, rtol=rtol, atol=atol)
        return
    if actual != expected:
        raise AssertionError(f"actual={actual!r}, expected={expected!r}")


def _exception_result(name: str, status: str, started: float, error: Exception):
    return CaseResult(
        name=name,
        status=status,
        duration_s=round(time.monotonic() - started, 4),
        error_type=type(error).__name__,
        error=str(error),
        traceback="".join(traceback.format_exception(error))[-12000:],
    )


def run_child(op_name: str, config_path: Path) -> int:
    import flag_gems
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    configured = load_config(config_path)
    configured_ops = configured["flagos_blacklist"] + configured["oot_blacklist"]
    if op_name not in configured_ops:
        raise ValueError(f"{op_name!r} is not blacklisted by {config_path}")

    cases = build_cases().get(op_name, [])
    schemas = [entry[0] for entry in flag_gems.FULL_CONFIG_BY_FUNC.get(op_name, [])]
    results: list[CaseResult] = []

    if not cases:
        payload = {
            "op": op_name,
            "status": "UNTESTED",
            "schemas": schemas,
            "cases": [],
            "reason": "no case is defined",
        }
        print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
        return 0

    for case in cases:
        started = time.monotonic()
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        try:
            native_call = case.build(False)
            expected = native_call()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001 - diagnostic harness
            results.append(
                _exception_result(case.name, "HARNESS_ERROR", started, error)
            )
            continue

        torch.manual_seed(0)
        try:
            flaggems_call = case.build(True)
            context = (
                flag_gems.use_gems(include=[op_name])
                if case.registration == "aten"
                else nullcontext()
            )
            with context:
                actual = flaggems_call()
                torch.cuda.synchronize()
            _assert_close(actual, expected, rtol=case.rtol, atol=case.atol)
        except AssertionError as error:
            results.append(_exception_result(case.name, "MISMATCH", started, error))
        except Exception as error:  # noqa: BLE001 - failures are the output
            results.append(
                _exception_result(case.name, "RUNTIME_ERROR", started, error)
            )
        else:
            results.append(
                CaseResult(
                    name=case.name,
                    status="PASS",
                    duration_s=round(time.monotonic() - started, 4),
                )
            )

    statuses = {result.status for result in results}
    if statuses == {"PASS"}:
        status = "PASS"
    elif "HARNESS_ERROR" in statuses:
        status = "INCOMPLETE"
    elif "PASS" in statuses:
        status = "PARTIAL"
    else:
        status = "FAIL"

    payload = {
        "op": op_name,
        "status": status,
        "schemas": schemas,
        "cases": [asdict(result) for result in results],
    }
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False))
    return 0


def load_config(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return {
        "flagos_blacklist": list(config.get("flagos_blacklist", [])),
        "oot_blacklist": list(config.get("oot_blacklist", [])),
    }


def _extract_child_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    return None


def _environment() -> dict[str, Any]:
    import flag_gems
    import torch
    import triton

    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "flaggems": flag_gems.__version__,
        "flaggems_vendor": flag_gems.vendor_name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
    }


def run_parent(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    configured = load_config(config_path)
    selected = configured["flagos_blacklist"]
    if not args.skip_oot:
        selected += configured["oot_blacklist"]
    if args.ops:
        requested = [name.strip() for name in args.ops.split(",") if name.strip()]
        unknown = sorted(set(requested) - set(selected))
        if unknown:
            raise ValueError(f"not selected from the current blacklist: {unknown}")
        selected = [name for name in selected if name in requested]

    available_cases = build_cases()
    missing_cases = sorted(set(selected) - set(available_cases))
    report: dict[str, Any] = {
        "config": str(config_path),
        "environment": _environment(),
        "limitations": (
            "PASS means the scripted samples work. Re-run the affected real model "
            "before removing any blacklist entry."
        ),
        "selected_ops": selected,
        "coverage": {
            "selected": len(selected),
            "covered": len(selected) - len(missing_cases),
            "missing": missing_cases,
        },
        "results": [],
    }

    print(
        f"Testing {len(selected)} blacklist entries on {report['environment']['gpu']}"
    )
    for position, op_name in enumerate(selected, start=1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--op",
            op_name,
            "--config",
            str(config_path),
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as error:
            result = {
                "op": op_name,
                "status": "TIMEOUT",
                "schemas": [],
                "cases": [],
                "duration_s": round(time.monotonic() - started, 4),
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
            }
        else:
            result = _extract_child_result(completed.stdout)
            if result is None:
                result = {
                    "op": op_name,
                    "status": "CRASH",
                    "schemas": [],
                    "cases": [],
                }
            result["exit_code"] = completed.returncode
            result["duration_s"] = round(time.monotonic() - started, 4)
            if completed.stderr:
                result["stderr"] = completed.stderr[-12000:]
            if result["status"] == "CRASH":
                result["stdout"] = completed.stdout[-12000:]

        report["results"].append(result)
        print(
            f"[{position:02d}/{len(selected):02d}] "
            f"{op_name:<32} {result['status']:<10} "
            f"{result['duration_s']:>7.2f}s",
            flush=True,
        )

    summary: dict[str, int] = {}
    for result in report["results"]:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    report["summary"] = summary
    report["total_cases"] = sum(
        len(result.get("cases", [])) for result in report["results"]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary}")
    print(f"Report: {args.report}")
    if missing_cases:
        return 2
    return 1 if set(summary) != {"PASS"} else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("nvidia-flaggems-blacklist-report.json"),
    )
    parser.add_argument("--ops", help="Comma-separated subset of blacklist entries")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per operator")
    parser.add_argument("--skip-oot", action="store_true", help="Skip oot_blacklist")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--op", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        if not args.op:
            raise ValueError("--child requires --op")
        return run_child(args.op, args.config.resolve())
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
