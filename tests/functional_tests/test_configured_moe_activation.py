# SPDX-License-Identifier: Apache-2.0
"""Configured MoE activations must retain native numerical semantics."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts


@pytest.mark.parametrize(
    "activation,limit,alpha",
    [
        (MoEActivation.SILU, 10.0, None),
        (MoEActivation.SWIGLUOAI, 7.0, 1.702),
        (MoEActivation.SWIGLUOAI_UNINTERLEAVE, 7.0, 1.702),
    ],
)
def test_configured_activation_matches_native(activation, limit, alpha):
    # An isolated source copy permits qualification without editing a running
    # model's frozen plugin checkout.
    source = os.environ.get("FL_MOE_QUAL_SOURCE")
    if source:
        spec = importlib.util.spec_from_file_location("fl_moe_qual", source)
        impl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(impl)
        Experts = impl.TritonExpertsFL
    else:
        from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL as Experts
    torch.manual_seed(147)
    m, h, i, e, k = 16, 128, 128, 4, 2
    cfg = SimpleNamespace(
        hidden_dim=h,
        in_dtype=torch.bfloat16,
        swiglu_limit=limit,
        swiglu_alpha=alpha,
        swiglu_beta=1.0 if alpha else None,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        activation=activation,
        max_num_tokens=m,
    )
    native = TritonExperts(cfg, FUSED_MOE_UNQUANTIZED_CONFIG)
    fl = Experts(cfg, FUSED_MOE_UNQUANTIZED_CONFIG)
    x = (torch.randn(m, h, device="cuda") * 8).bfloat16()
    w1 = (torch.randn(e, 2 * i, h, device="cuda") * 0.25).bfloat16()
    w2 = (torch.randn(e, h, i, device="cuda") * 0.05).bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(m)])
    weights = torch.softmax(torch.randn(m, k, device="cuda"), -1)

    def run(expert):
        shapes = expert.workspace_shapes(m, 2 * i, h, k, e, e, None, activation)
        a, b, _ = shapes
        out = torch.empty_like(x)
        expert.apply(
            out,
            x,
            w1,
            w2,
            weights,
            ids,
            activation,
            e,
            None,
            None,
            None,
            torch.empty(a, device="cuda", dtype=x.dtype),
            torch.empty(b, device="cuda", dtype=x.dtype),
            None,
            False,
        )
        return out

    expected = run(native)
    actual = run(fl)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)
    # The interleaved native SWIGLUOAI kernel fixes alpha/clamp internally;
    # only SILU and UNINTERLEAVE consume the configurable clamp parameter.
    if activation != MoEActivation.SWIGLUOAI:
        cfg.swiglu_limit = 10000.0
        unclamped = run(TritonExperts(cfg, FUSED_MOE_UNQUANTIZED_CONFIG))
        assert (expected - unclamped).abs().max().item() > 0.1
