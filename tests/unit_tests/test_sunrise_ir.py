# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Real vLLM IR registration and numerical contracts for the PTPU empty build."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.ir")

from vllm.config.compilation import CompilationMode
from vllm.ir import ops
from vllm.ir.op import enable_torch_wrap

from vllm_fl.platform import PlatformFL


@pytest.fixture
def native_ir_priority(monkeypatch):
    monkeypatch.setattr(PlatformFL, "device_type", "ptpu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(backend="eager", mode=CompilationMode.NONE)
    )
    priority = PlatformFL.get_default_ir_op_priority(config)
    PlatformFL.import_ir_kernels()
    return priority


def test_empty_ptpu_priority_contains_only_registered_native_providers(
    native_ir_priority,
):
    from vllm.ir.op import IrOp

    for name, providers in asdict(native_ir_priority).items():
        assert providers == ["native"]
        assert set(providers) <= IrOp.registry[name].impls.keys()
        # Exercise the same real registration/validation gate used by WorkerBase.
        with IrOp.registry[name].set_priority(providers):
            assert IrOp.registry[name].get_priority() == ["native"]


def test_native_ptpu_ir_executes_real_layernorm_and_activation(native_ir_priority):
    x = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    residual = torch.full_like(x, 0.5)
    weight = torch.linspace(0.5, 1.5, 8)
    epsilon = 1e-5
    with native_ir_priority.set_priority(), enable_torch_wrap(False):
        result = ops.rms_norm(x, weight, epsilon)
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight
        torch.testing.assert_close(result, expected)
        normalized, added = ops.fused_add_rms_norm(x, residual, weight, epsilon)
        expected_added = x + residual
        expected_norm = (
            expected_added
            * torch.rsqrt(expected_added.square().mean(-1, keepdim=True) + epsilon)
            * weight
        )
        torch.testing.assert_close(added, expected_added)
        torch.testing.assert_close(normalized, expected_norm)
        output = ops.gelu_and_mul_sparse(x, 0.5)
        gate = x[..., :4]
        sparse = torch.relu(
            gate
            - gate.mean(-1, keepdim=True)
            - 0.5 * gate.std(-1, keepdim=True, unbiased=False)
        )
        torch.testing.assert_close(
            output, torch.nn.functional.gelu(sparse) * x[..., 4:]
        )
