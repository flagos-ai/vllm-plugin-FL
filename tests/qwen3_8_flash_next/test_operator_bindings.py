"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

import flaggems_vllm
import torch

from vllm_fl.models.qwen3_8_flash_next.gpu.ops import (
    hyperconnection,
    ple_fusion,
    qsa,
)


def test_library_numerical_delegates():
    for module, names in (
        (hyperconnection, ("qwen4_hc_combine_norm", "qwen4_hc_gate_reduce")),
        (
            qsa,
            (
                "qsa_mqa_paged",
                "qsa_compress_groups_with_ratio",
                "qsa_compress_norm_mrope_store_groups",
            ),
        ),
    ):
        for name in names:
            assert getattr(module, name) is getattr(flaggems_vllm, name)


def test_hc_fake_output_shapes():
    residual = torch.empty((3, 4 * 64), device="meta", dtype=torch.bfloat16)
    block = torch.empty((3, 64), device="meta", dtype=torch.bfloat16)
    injection = torch.empty((3, 4), device="meta", dtype=torch.bfloat16)
    weight = torch.empty((4 * 64,), device="meta", dtype=torch.bfloat16)
    combined, normalized = torch.ops.vllm.qwen4_hc_combine_norm(
        residual, block, injection, weight, 1e-6, 4
    )
    assert combined.shape == normalized.shape == residual.shape
    assert combined.dtype == normalized.dtype == residual.dtype


def test_ple_fake_output_shapes():
    key = torch.empty((3, 4, 64), device="meta", dtype=torch.bfloat16)
    value = torch.empty((3, 64), device="meta", dtype=torch.bfloat16)
    weight = torch.empty((4 * 64,), device="meta", dtype=torch.bfloat16)
    gated, normalized = ple_fusion.ple_gate_norm(
        key, key, value, weight, weight, weight, 4, 1e-6
    )
    assert gated.shape == normalized.shape == key.shape
    assert gated.dtype == normalized.dtype == key.dtype
