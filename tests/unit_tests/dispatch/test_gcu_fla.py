# Copyright (c) 2026 BAAI. All rights reserved.

import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific unit tests only run on Enflame", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import (
    chunk_delta_h,
    fused_recurrent_packed_decode,
)


class _CaptureKernel:
    def __init__(self):
        self.arguments = None

    def __getitem__(self, grid):
        def launch(**arguments):
            self.arguments = arguments

        return launch


@pytest.mark.parametrize("use_exp2", [False, True])
def test_chunk_wrapper_preserves_kda_exp2_option(monkeypatch, use_exp2):
    kernel = _CaptureKernel()
    monkeypatch.setattr(
        chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gcu", kernel
    )
    k = torch.zeros(1, 4, 2, 8)
    u = torch.zeros(1, 4, 2, 8)

    chunk_delta_h.chunk_gated_delta_rule_fwd_h_gcu(k, k, u, use_exp2=use_exp2)

    assert kernel.arguments["USE_EXP2"] is use_exp2
    assert kernel.arguments["NH_TOTAL"] == 2


def test_gcu_packed_beta_does_not_round_to_input_dtype():
    # Evaluate the actual kernel expression on CPU; a BF16 round trip changes
    # this recurrent coefficient despite converting back to FP32 afterwards.
    source = Path(inspect.getfile(fused_recurrent_packed_decode)).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "fused_recurrent_gated_delta_rule_packed_decode_kernel_gcu"
    )
    assignment = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "beta_val"
            for target in node.targets
        )
    )
    expression = ast.Expression(body=assignment.value)
    coefficient = torch.tensor([0.13, 0.73, -1.11], dtype=torch.float32)
    beta = eval(
        compile(expression, "GCU packed beta", "eval"),
        {
            "tl": SimpleNamespace(sigmoid=torch.sigmoid, float32=torch.float32),
            "b_val": coefficient,
            "b": SimpleNamespace(dtype=SimpleNamespace(element_ty=torch.bfloat16)),
        },
    )

    expected = torch.sigmoid(coefficient)
    assert beta.dtype == torch.float32
    torch.testing.assert_close(beta, expected, rtol=0, atol=0)
    assert not torch.equal(beta, expected.to(torch.bfloat16).float())


def test_gcu_launchers_patch_consumers_and_preserve_official_kernels(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.third_party.flash_linear_attention import ops
    from vllm.third_party.flash_linear_attention.ops import (
        chunk,
        chunk_delta_h as upstream_chunk,
        fused_recurrent,
        kda,
    )

    chunk_kernel = upstream_chunk.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    packed_kernel = (
        fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
    )
    chunk_targets = [upstream_chunk, chunk, kda]
    packed_targets = [fused_recurrent, ops, qwen_gdn_linear_attn]
    # Register the original bindings for restoration after the patch runs.
    for target in chunk_targets:
        name = "chunk_gated_delta_rule_fwd_h"
        monkeypatch.setattr(target, name, getattr(target, name))
    for target in packed_targets:
        name = "fused_recurrent_gated_delta_rule_packed_decode"
        monkeypatch.setattr(target, name, getattr(target, name))
    monkeypatch.setattr(
        torch, "gcu", SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(chunk_delta_h, "_patched", False)
    monkeypatch.setattr(fused_recurrent_packed_decode, "_patched", False)

    chunk_delta_h.apply_chunk_delta_h_gcu_patch()
    fused_recurrent_packed_decode.apply_fused_recurrent_packed_decode_gcu_patch()

    for target in chunk_targets:
        assert target.chunk_gated_delta_rule_fwd_h is (
            chunk_delta_h.chunk_gated_delta_rule_fwd_h_gcu
        )
    for target in packed_targets:
        assert target.fused_recurrent_gated_delta_rule_packed_decode is (
            fused_recurrent_packed_decode.fused_recurrent_gated_delta_rule_packed_decode_gcu
        )
    assert upstream_chunk.chunk_gated_delta_rule_fwd_kernel_h_blockdim64 is chunk_kernel
    assert (
        fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
        is packed_kernel
    )
