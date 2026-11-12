# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HyperConnection stream math, packing and fused kernel boundaries.

The average-pooling variant is pure Torch and is checked against a loop
reference. The gated variant's mix/combine math is checked against an explicit
implementation of the documented kernels: per-branch Gemma RMSNorm, a low-rank
SiLU gate, a sigmoid-weighted mean and the ``2*sigmoid(logit/HC)`` injection.
The weight-packing contracts run on CPU; the fused-kernel tests need CUDA.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.common.hyperconnection import (
    GatedResidualSimple,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
    pack_gated_hc_projection_weights,
)


def _config(**overrides):
    values = dict(
        hc_count=3,
        hidden_size=4,
        params_dtype=torch.float32,
        hc_lowrank=2,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    values.update(overrides)
    return HyperConnectionConfig(**values)


class TestConfigAndNorm:
    def test_defaults_match_the_reference_config(self):
        config = HyperConnectionConfig()
        assert (config.hc_count, config.hidden_size) == (4, 64)
        assert config.mtp_hc is False
        assert config.hc_per_branch_norm is False

    def test_grouped_norm_validates_divisibility_and_weight(self):
        norm = GroupedGemmaRMSNorm(8, eps=1e-5, group_size=4, dtype=torch.float32)
        assert norm.variance_epsilon == 1e-5
        assert torch.count_nonzero(norm.weight) == 0
        assert norm.weight.shape == (8,)
        with pytest.raises(ValueError, match="divisible by group_size"):
            GroupedGemmaRMSNorm(10, eps=1e-5, group_size=4, dtype=torch.float32)


class TestAverageVariant:
    """HyperConnectionBase averages the HC streams and broadcasts back."""

    def test_mix_and_combine_match_loop_reference(self):
        hc_count, hidden = 3, 4
        module = HyperConnectionBase(_config(hc_count=hc_count, hidden_size=hidden))
        assert module.hyper_hidden_size == hc_count * hidden
        hyper_input = torch.arange(2 * hc_count * hidden, dtype=torch.float32)
        hyper_input = hyper_input.reshape(2, hc_count * hidden)
        block_output = torch.full((2, hidden), 0.5)

        mixed, residual = module.mix(hyper_input)
        assert mixed.shape == (2, hidden)
        for row in range(2):
            for channel in range(hidden):
                expected = sum(
                    float(hyper_input[row, stream * hidden + channel])
                    for stream in range(hc_count)
                ) / hc_count
                assert abs(float(mixed[row, channel]) - expected) < 1e-6
        assert residual is hyper_input

        combined = module.combine(block_output, residual)
        assert combined.shape == hyper_input.shape
        for row in range(2):
            for stream in range(hc_count):
                for channel in range(hidden):
                    expected = (
                        float(hyper_input[row, stream * hidden + channel])
                        + float(block_output[row, channel])
                    )
                    assert (
                        abs(
                            float(combined[row, stream * hidden + channel])
                            - expected
                        )
                        < 1e-6
                    )

    def test_shape_assertions_guard_the_layout(self):
        module = HyperConnectionBase(_config())
        with pytest.raises(AssertionError):
            module.mix(torch.zeros(2, module.hyper_hidden_size - 1))
        with pytest.raises(AssertionError):
            module.combine(
                torch.zeros(2, module.hidden_size),
                torch.zeros(2, module.hyper_hidden_size - 1),
            )


class TestGatedVariantConstruction:
    """Packing preserves checkpoint names while sharing one allocation."""

    def test_combined_module_shares_one_packed_allocation(self):
        module = GatedResidualSimple(_config())
        packed = module._packed_down_inject_weight
        assert packed is not None
        assert packed.shape == (2 + 3, 3 * 4)
        assert (
            packed.untyped_storage().data_ptr()
            == module.input_mix_weight_down.weight.untyped_storage().data_ptr()
            == module.block_inject_weight.weight.untyped_storage().data_ptr()
        )
        assert module.pack_down_inject_weights() is True

    def test_mix_only_module_has_no_injection_projection(self):
        module = GatedResidualSimple(_config(), use_combine=False)
        assert hasattr(module, "input_mix_weight_down")
        assert not hasattr(module, "block_inject_weight")
        assert module._packed_down_inject_weight is None
        assert module.pack_down_inject_weights() is False
        with pytest.raises(RuntimeError, match="combine was disabled"):
            module.combine(torch.zeros(2, 4), (torch.zeros(2, 12), None, None))

    def test_combine_only_module_has_no_mix_projection(self):
        module = GatedResidualSimple(_config(), use_mix=False)
        assert not hasattr(module, "input_mix_weight_down")
        assert hasattr(module, "block_inject_weight")
        assert module.pack_down_inject_weights() is False
        with pytest.raises(RuntimeError, match="mix was disabled"):
            module.mix(torch.zeros(2, 12))

    def test_fully_disabled_module_raises_for_both_halves(self):
        module = GatedResidualSimple(_config(), use_mix=False, use_combine=False)
        assert module.pack_down_inject_weights() is False
        with pytest.raises(RuntimeError, match="mix was disabled"):
            module.mix(torch.zeros(2, 12))
        with pytest.raises(RuntimeError, match="combine was disabled"):
            module.combine(torch.zeros(2, 4), (torch.zeros(2, 12), None, None))

    def test_pending_combine_requires_both_tensors(self):
        module = GatedResidualSimple(_config())
        with pytest.raises(ValueError, match="provided together"):
            module.combine_and_mix(
                torch.zeros(2, 12),
                prev_block_output=None,
                prev_injection=torch.zeros(2, 3),
            )

    def test_pack_walks_the_module_tree(self):
        root = nn.Module()
        root.add_module("packed", GatedResidualSimple(_config()))
        root.add_module("mix_only", GatedResidualSimple(_config(), use_combine=False))
        assert pack_gated_hc_projection_weights(root) == 1


def _branch_norm_reference(x, weight, eps, hc_count):
    grouped = x.reshape(*x.shape[:-1], hc_count, -1)
    rrms = torch.rsqrt(grouped.float().pow(2).mean(-1, keepdim=True) + eps)
    normalized = grouped.float() * rrms
    normalized = normalized * (1.0 + weight.reshape(hc_count, -1))
    return normalized.reshape_as(x)


def _mix_reference(module, hyper_input):
    """Independent implementation of the documented gated-residual math."""
    config = module.config
    hc, hidden = config.hc_count, config.hidden_size
    normed = _branch_norm_reference(
        hyper_input, module.hc_norm.weight, config.rms_norm_eps, hc
    )
    # The fused norm stores its result in the activation dtype.
    normed = normed.to(hyper_input.dtype)
    down_and_inject = F.linear(normed, module._packed_down_inject_weight)
    down, injection = down_and_inject.split((config.hc_lowrank, hc), dim=-1)
    gate = F.silu(down / hc)
    gate_logits = F.linear(gate, module.input_mix_weight_up.weight)
    weights = torch.sigmoid(gate_logits.reshape(-1, hc, hidden)).float()
    mixed = (weights * normed.float().reshape(-1, hc, hidden)).sum(dim=1) / hc
    return mixed.to(hyper_input.dtype), normed, injection


def _combine_reference(hyper_input, block_output, injection_logits, hc_count):
    hc_dim = block_output.shape[-1]
    grouped = hyper_input.reshape(*hyper_input.shape[:-1], hc_count, hc_dim)
    injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    combined = grouped.float() + block_output.float().unsqueeze(-2) * injection.unsqueeze(-1)
    return combined.reshape_as(hyper_input).to(hyper_input.dtype)


@pytest.mark.gpu
class TestGatedVariantKernels:
    # The fused HC kernels operate on the production BF16 activation dtype.
    DTYPE = torch.bfloat16
    RTOL = 2e-2
    ATOL = 2e-2

    def test_mix_and_combine_match_documented_reference(self, device):
        config = _config(
            hc_count=2,
            hidden_size=4,
            hc_lowrank=2,
            params_dtype=self.DTYPE,
        )
        module = GatedResidualSimple(config).to(device)
        with torch.no_grad():
            module.hc_norm.weight.copy_(torch.linspace(0.0, 0.5, 8, device=device))
        hyper_input = torch.randn(2, 2 * 4, device=device, dtype=self.DTYPE)
        block_output = torch.randn(2, 4, device=device, dtype=self.DTYPE)

        mixed, residuals = module.mix(hyper_input)
        expected_mixed, expected_normed, expected_injection = _mix_reference(
            module, hyper_input
        )
        torch.testing.assert_close(
            mixed, expected_mixed, rtol=self.RTOL, atol=self.ATOL
        )
        torch.testing.assert_close(
            residuals[1], expected_normed, rtol=self.RTOL, atol=self.ATOL
        )
        torch.testing.assert_close(
            residuals[2], expected_injection, rtol=self.RTOL, atol=self.ATOL
        )

        combined = module.combine(block_output, residuals)
        expected_combined = _combine_reference(
            hyper_input, block_output, expected_injection, config.hc_count
        )
        torch.testing.assert_close(
            combined, expected_combined, rtol=self.RTOL, atol=self.ATOL
        )

    def test_grouped_norm_matches_gemma_reference(self, device):
        hidden, group_size = 8, 4
        norm = GroupedGemmaRMSNorm(
            hidden, eps=1e-6, group_size=group_size, dtype=self.DTYPE
        ).to(device)
        x = torch.arange(8, dtype=self.DTYPE, device=device).reshape(1, 8)
        expected = _branch_norm_reference(
            x, torch.zeros(8, device=device), 1e-6, hidden // group_size
        ).to(self.DTYPE)
        torch.testing.assert_close(
            norm(x), expected, rtol=self.RTOL, atol=self.ATOL
        )

        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.0, 1.0, hidden, device=device))
        expected = _branch_norm_reference(
            x, norm.weight, 1e-6, hidden // group_size
        ).to(self.DTYPE)
        torch.testing.assert_close(
            norm(x), expected, rtol=self.RTOL, atol=self.ATOL
        )
