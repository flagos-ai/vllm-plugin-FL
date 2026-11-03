# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for opt-in MoE destination buffers; GPU arithmetic is separate."""

import __future__

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
VLLM = Path(importlib.util.find_spec("vllm").origin).parent


def declaration(path, name, namespace):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    ns = dict(namespace)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    return ns[name]


def config():
    parallel = SimpleNamespace(
        tp_size=8,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        sp_size=1,
        use_all2all_kernels=False,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        hidden_dim=2560,
        intermediate_size_per_partition=80,
        num_experts=512,
        num_local_experts=512,
        experts_per_token=10,
        activation=SimpleNamespace(value="silu"),
        has_bias=False,
        is_lora_enabled=False,
    )


def supported():
    return declaration(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/mctlass_moe.py",
        "direct_outputs_supported",
        {"torch": torch},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_dim", 128),
        ("intermediate_size_per_partition", 160),
        ("in_dtype", torch.float16),
        ("num_experts", 256),
        ("num_local_experts", 64),
        ("experts_per_token", 8),
        ("has_bias", True),
        ("is_lora_enabled", True),
        ("tp_size", 4),
        ("pcp_size", 2),
        ("dp_size", 2),
        ("ep_size", 8),
        ("sp_size", 2),
        ("use_all2all_kernels", True),
    ],
)
def test_direct_buffers_reject_unvalidated_config(field, value):
    cfg = config()
    quant = SimpleNamespace(quant_dtype=None, gemm1_clamp_limit=None)
    assert supported()(cfg, quant, "MetaX C550")
    setattr(
        cfg.moe_parallel_config if hasattr(cfg.moe_parallel_config, field) else cfg,
        field,
        value,
    )
    assert not supported()(cfg, quant, "MetaX C550")


@pytest.mark.parametrize(
    "device,quant,clamp",
    [
        ("other", None, None),
        ("C550", torch.int8, None),
        ("C550", None, 7.0),
    ],
)
def test_direct_buffers_reject_device_quant_clamp(device, quant, clamp):
    assert not supported()(
        config(), SimpleNamespace(quant_dtype=quant, gemm1_clamp_limit=clamp), device
    )


@pytest.mark.parametrize(
    "enabled,kind",
    [
        (True, "valid"),
        (False, "valid"),
        (True, "shape"),
        (True, "dtype"),
        (True, "strided"),
        (True, "none"),
    ],
)
def test_native_modular_runner_passes_only_valid_opted_in_destination(enabled, kind):
    fn = declaration(
        VLLM / "model_executor/layers/fused_moe/modular_kernel.py",
        "_fused_experts",
        {"torch": torch, "current_platform": SimpleNamespace(is_rocm=lambda: False)},
    )
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    workspace = torch.full_like(x, -5)
    alias = torch.full_like(x, 99)
    if kind == "shape":
        alias = torch.empty(2, 4)
    elif kind == "dtype":
        alias = torch.empty_like(x, dtype=torch.float64)
    elif kind == "strided":
        alias = torch.empty(3, 8)[:, ::2]
    elif kind == "none":
        alias = None
    received = []

    def apply(**kw):
        received.append(kw["output"])
        kw["output"].copy_(kw["hidden_states"] * 3 + 1)

    expert = SimpleNamespace(
        supports_output_buffer_alias=enabled,
        moe_problem_size=lambda *args: (2, 3, 6, 4, 2),
        a2_scale=None,
        apply=apply,
    )
    runner = SimpleNamespace(
        fused_experts=expert,
        _allocate_buffers=lambda *args: (torch.empty(1), torch.empty(1), workspace),
    )
    got = fn(
        runner,
        torch.float32,
        x,
        None,
        torch.empty(2, 6, 4),
        torch.empty(2, 4, 3),
        torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int32),
        None,
        2,
        2,
        None,
        False,
        None,
        output_alias=alias,
    )
    target = alias if enabled and kind == "valid" else workspace
    assert got is target and received == [target]
    torch.testing.assert_close(got, x * 3 + 1, atol=0, rtol=0)


def test_native_default_capability_is_disabled():
    fn = declaration(
        VLLM / "model_executor/layers/fused_moe/modular_kernel.py",
        "supports_output_buffer_alias",
        {},
    )
    assert fn(None) is False


@pytest.mark.parametrize("strided", [False, True])
def test_silu_out_passes_original_views_and_supplied_destination(monkeypatch, strided):
    name = "flaggems_vllm.ops.silu_and_mul"
    module = ModuleType(name)
    calls = []

    def out(a, b, destination):
        calls.append((a, b, destination))
        destination.copy_((torch.nn.functional.silu(a.float()) * b.float()).to(a.dtype))
        return destination

    module.silu_and_mul_out = out
    monkeypatch.setitem(sys.modules, name, module)
    fn = declaration(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/mctlass_moe.py",
        "silu_activation_out",
        {},
    )
    x = torch.randn(7, 320 if strided else 160, dtype=torch.bfloat16)
    if strided:
        x = x[:, ::2]
    y = torch.empty(7, 80, dtype=x.dtype)
    assert fn(x, y) is y and calls[0][2] is y
    assert calls[0][0].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    torch.testing.assert_close(
        y,
        (torch.nn.functional.silu(x[:, :80].float()) * x[:, 80:].float()).to(x.dtype),
        atol=0,
        rtol=0,
    )
